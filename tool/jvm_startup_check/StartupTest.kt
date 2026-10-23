import com.murdock.murdock.EchoBrain
import org.tensorflow.lite.Interpreter
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files

fun main() {
    val dir = Files.createTempDirectory("brain").toFile()
    val weights = File(dir, "brain/head.weights")
    val model = ByteBuffer.allocateDirect(1024).put(ByteArray(1024) { it.toByte() })
    // Launch 1: fresh install, learn a bit, save (writes weights + meta with the model id).
    val first = EchoBrain(model, weights, File(dir, "samples"))
    // Novelty gate: the same view over and over is only kept NEAR_LIMIT times.
    val kept = (0 until 10).count { first.addSample(FloatArray(16 * 128 * 2), EchoBrain.poseFeatures(30f, 0f), floorMap(), 35f) }
    println("same view x10: kept $kept, recent novelty ${first.recentNovelty}")
    check(kept == EchoBrain.NEAR_LIMIT && first.sampleCount == kept) { "novelty gate" }
    first.reset()
    // Ten different views (the floor at different heights) are all kept.
    repeat(10) { i ->
        check(first.addSample(FloatArray(16 * 128 * 2), EchoBrain.poseFeatures(30f, 0f), floorMap(2 + i * 2), tiltDeg = if (i < 7) 35f else 62f)) { "new view $i dropped" }
    }
    val queued = first.pendingSteps
    // Both heads train: depth and shape labels (fitted to the depth) go in every step.
    val loss = first.train(2)
    println("trained: loss $loss, shape loss ${first.lastShapeLoss}, train inputs ${Interpreter.lastTrainInputs}")
    check(loss.isFinite() && first.lastShapeLoss.isFinite()) { "training failed" }
    check(Interpreter.lastTrainInputs == setOf("echo", "pose", "depth", "shapes", "lr")) { "train inputs" }
    first.save()
    println("launch 1 OK; queued $queued steps; meta = " + File(weights.path + ".meta").readText())
    // Launch 2: the case that crashed on the phone (saved meta present at startup), and the
    // training queue must come back with it.
    val second = EchoBrain(model, weights, File(dir, "samples"))
    val p = second.infer(FloatArray(16 * 128 * 2), EchoBrain.poseFeatures(35f, 0f))
    println("launch 2 OK; infer ok = " + (p.depth.all { it.isFinite() } && p.shapes.size == 24) +
        "; queue restored = ${second.pendingSteps == queued} (${second.pendingSteps})" +
        "; samples reloaded = ${second.sampleCount}; steps = ${second.trainedSteps}")
    check(second.pendingSteps == queued) { "training queue lost across restart" }
    check(second.trainedSteps == 2) { "learning lost across restart" }
    println("angle coverage restored: at 35° = ${second.coverageAt(35f)}, at 62° = ${second.coverageAt(62f)}, at 0° = ${second.coverageAt(0f)}")
    check(second.coverageAt(35f) == 7 && second.coverageAt(62f) == 3 && second.coverageAt(0f) == 0) { "coverage lost" }
    check(second.sampleCount == 10) { "samples (with pose) not reloaded" }
    check(second.train(1).isFinite()) { "reloaded samples don't train" }

    // Launch 3: an update from the model before the shapes head. Its weights file holds only
    // the depth head (w0..w3) and its meta names that model; the learning must carry over.
    val legacyDir = Files.createTempDirectory("legacy").toFile()
    val legacyWeights = File(legacyDir, "brain/head.weights").apply { parentFile.mkdirs() }
    DataOutputStream(legacyWeights.outputStream()).use { out ->
        val sizes = intArrayOf(514 * 1024, 1024, 1024 * 1024, 1024)
        out.writeInt(sizes.size)
        for (n in sizes) { out.writeInt(n * 4); out.write(ByteArray(n * 4)) }
    }
    File(legacyWeights.path + ".meta").writeText("1234 56 2608876991 40.0")
    // A sample file from that version (same .mdp layout: echo + pose + depth).
    val samples = File(legacyDir, "samples").apply { mkdirs() }
    val rec = ByteBuffer.allocate((16 * 128 * 2 + 2 + 32 * 32) * 4 * 3).order(ByteOrder.LITTLE_ENDIAN)
    repeat(3) { rec.asFloatBuffer().also { fb -> fb.position(it * (16 * 128 * 2 + 2 + 32 * 32) + 16 * 128 * 2 + 2); fb.put(floorMap()) } }
    File(samples, "session_1.mdp").writeBytes(rec.array())
    val third = EchoBrain(model, legacyWeights, samples)
    println("launch 3 (pre-shapes learning) OK; steps = ${third.trainedSteps}, samples = ${third.totalSamples}, " +
        "queue = ${third.pendingSteps}, reloaded = ${third.sampleCount}")
    check(third.trainedSteps == 1234 && third.totalSamples == 56 && third.pendingSteps == 40) { "pre-shapes learning discarded" }
    check(third.sampleCount == 3) { "pre-shapes samples not reloaded" }
    third.save()
    check(File(legacyWeights.path + ".meta").readText().split(" ")[2] != "2608876991") { "meta not moved to the new model" }

    // Launch 4: a weights file from an unrelated model is discarded, never crashes startup.
    File(legacyWeights.path + ".meta").writeText("9 9 42 0")
    val fourth = EchoBrain(model, legacyWeights, samples)
    check(fourth.trainedSteps == 0 && !legacyWeights.exists()) { "foreign weights not discarded" }
    println("launch 4 (foreign model) OK: learning reset, samples kept = ${fourth.sampleCount}")
}

/** A floor rising towards the bottom of the view, so the shape fitter has something to find. */
fun floorMap(horizon: Int = 12) =
    FloatArray(32 * 32) { i -> val y = i / 32; if (y < horizon) 0.1f else 0.1f + 0.9f * (y - horizon) / (31f - horizon) }
