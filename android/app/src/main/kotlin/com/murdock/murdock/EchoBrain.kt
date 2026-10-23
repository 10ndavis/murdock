package com.murdock.murdock

import android.util.Log
import com.murdock.murdock.SonarConfig.DEPTH_FLOATS
import com.murdock.murdock.SonarConfig.STACK_FLOATS
import org.tensorflow.lite.Interpreter
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * The echo -> depth + shapes student (assets/models/echo_student.tflite, built by
 * ml/export_tflite.py).
 *
 * Signatures: infer (depth + scene shapes), train (both heads, trunk frozen), get_weights /
 * set_weights for persistence. Shape labels are fitted to the camera depth (ShapeFit).
 * Every call runs on the caller's thread; MainActivity funnels them through one executor
 * because the interpreter is not thread-safe and infer/train share its variables.
 */
class EchoBrain(
    private val modelBytes: ByteBuffer,
    private val weightsFile: File,
    private val samplesDir: File,
) {
    /**
     * One camera-labelled training example: echoes, phone pose (head input), camera depth,
     * and the scene shapes fitted to that depth. Shapes are derived (ShapeFit), not stored,
     * so samples recorded before the shapes head existed get labels too.
     */
    private class Sample(val echo: FloatArray, val pose: FloatArray, val depth: FloatArray) {
        // Fitted on first use (by the trainer), so reloading 2000 samples at startup stays fast.
        val shapes by lazy(LazyThreadSafetyMode.NONE) { ShapeFit.fit(depth) }
    }

    /** The echo AI's output: nearness map and scene shapes (ml/shapes.py layout). */
    class Prediction(val depth: FloatArray, val shapes: FloatArray)

    private val replay = ArrayList<Sample>()

    /** How many samples were learned at each phone tilt (TILT_BIN_DEG bins from TILT_FROM_DEG). */
    val tiltCoverage = IntArray(TILT_BINS)
    private val coverageFile get() = File(weightsFile.path + ".coverage")
    private var sampleOut: FileOutputStream? = null

    // Lifetime counters, persisted next to the weights so progress survives restarts.
    var trainedSteps = 0; private set
    var totalSamples = 0; private set
    var lastLoss = Float.NaN; private set
    var lastShapeLoss = Float.NaN; private set
    val sampleCount get() = replay.size

    @Volatile private var stepBudget = 0f

    // Created in the init block at the END of the class, never in a property initializer:
    // open() reads modelId (a lazy delegate) and the counters, and Kotlin runs property
    // initializers in textual order -- touching a later-declared delegate from an earlier
    // initializer crashed startup with "getValue() on a null object reference".
    private lateinit var interpreter: Interpreter


    var recordToDisk = true

    private val metaFile get() = File(weightsFile.path + ".meta")

    /** Identifies the shipped model, so learning saved on top of an older model is dropped. */
    private val modelId: Long by lazy {
        java.util.zip.CRC32().apply { update(modelBytes.duplicate().apply { rewind() }) }.value
    }

    private fun open(): Interpreter {
        modelBytes.rewind()
        val it = Interpreter(modelBytes, Interpreter.Options().setNumThreads(2))
        // meta: "<steps> <samples> <modelId> [<queued steps>]". A head trained on another model's trunk features
        // would produce garbage, so after an app update with a new model we start fresh.
        val meta = metaFile.takeIf { f -> f.exists() }?.readText()?.trim()?.split(" ")
        // Models that only added a head keep the same trunk: their learning carries over.
        val savedId = meta?.getOrNull(2)?.toLongOrNull()
        val sameModel = savedId != null && (savedId == modelId || savedId in SAME_TRUNK_MODELS)
        if (weightsFile.exists() && sameModel) {
            try {
                restore(it)
                if (!outputIsFinite(it)) {
                    // Saved learning is corrupt (e.g. training blew up in an older version):
                    // start again from the shipped weights instead of drawing garbage forever.
                    Log.w(TAG, "saved weights give invalid output; discarding them")
                    it.close()
                    weightsFile.delete(); metaFile.delete()
                    modelBytes.rewind()
                    return Interpreter(modelBytes, Interpreter.Options().setNumThreads(2))
                }
                trainedSteps = meta!![0].toInt(); totalSamples = meta[1].toInt()
                coverageFile.takeIf { f -> f.exists() }?.readText()?.trim()?.split(" ")
                    ?.mapNotNull { v -> v.toIntOrNull() }?.forEachIndexed { i, v -> if (i < TILT_BINS) tiltCoverage[i] = v }
                // The training queue survives restarts too (older files don't have it).
                stepBudget = meta.getOrNull(3)?.toFloatOrNull() ?: 0f
            } catch (e: Exception) { Log.w(TAG, "restore failed: $e") }
        } else if (weightsFile.exists()) {
            Log.i(TAG, "model changed; discarding on-device learning from the previous model")
            weightsFile.delete(); metaFile.delete()
        }
        return it
    }

    private fun outputIsFinite(target: Interpreter): Boolean {
        val p = run(target, direct(STACK_FLOATS), direct(POSE_DIM))
        return p.depth.all { it.isFinite() } && p.shapes.all { it.isFinite() }
    }

    private fun run(target: Interpreter, echo: ByteBuffer, pose: ByteBuffer): Prediction {
        val depth = direct(DEPTH_FLOATS)
        val shapes = direct(ShapeFit.SHAPE_DIM)
        target.runSignature(mapOf("echo" to echo, "pose" to pose), mapOf("depth" to depth, "shapes" to shapes), "infer")
        return Prediction(depth.toFloats(DEPTH_FLOATS), shapes.toFloats(ShapeFit.SHAPE_DIM))
    }

    // ── Training safety ─────────────────────────────────────────────────────────────────
    // The exported train step clips gradients, but if a step or a prediction ever comes out
    // invalid anyway, roll back to the last good weights and halve the learning rate.
    var lr = DEFAULT_LR; private set
    var rollbacks = 0; private set
    private var good: Map<String, ByteBuffer>? = null

    /** Copy of the current head weights, or null if any value is invalid. */
    private fun currentWeights(target: Interpreter = interpreter): Map<String, ByteBuffer>? {
        val outputs = weightNames(target).associateWith {
            direct(target.getOutputTensorFromSignature(it, "get_weights").numBytes() / 4)
        }
        target.runSignature(mapOf("dummy" to direct(1)), outputs, "get_weights")
        for (buf in outputs.values) {
            buf.rewind()
            val fb = buf.asFloatBuffer()
            for (i in 0 until fb.remaining()) if (!fb.get(i).isFinite()) return null
        }
        return outputs
    }

    private fun snapshot() {
        currentWeights()?.let { good = it }
    }

    private fun rollback(reason: String) {
        rollbacks++
        lr = maxOf(lr * 0.5f, MIN_LR)
        Log.w(TAG, "rollback #$rollbacks ($reason); lr -> $lr")
        val g = good
        if (g != null) {
            g.values.forEach { it.rewind() }
            interpreter.runSignature(g.mapKeys { (k, _) -> k }, mapOf("ok" to direct(1)), "set_weights")
        } else {
            interpreter.close()
            weightsFile.delete(); metaFile.delete()
            modelBytes.rewind()
            interpreter = Interpreter(modelBytes, Interpreter.Options().setNumThreads(2))
            snapshot()
        }
    }

    /** "ok", or how often training had to be rolled back (shown in System Check). */
    val health get() = if (rollbacks == 0) "ok" else "recovered $rollbacks× (learning rate ${"%.3f".format(lr)})"

    fun infer(stack: FloatArray, pose: FloatArray): Prediction {
        repeat(2) {
            val p = run(interpreter, direct(stack), direct(pose))
            if (p.depth.all { it.isFinite() } && p.shapes.all { it.isFinite() }) return p
            rollback("invalid prediction")
        }
        return Prediction(FloatArray(DEPTH_FLOATS), FloatArray(ShapeFit.SHAPE_DIM))
    }

    /**
     * Store an (echo, pose, camera depth) example for replay training and the PC dataset
     * (.mdp, see ml/dataset.py). [tiltDeg] feeds the per-angle coverage shown in the app.
     */
    fun addSample(stack: FloatArray, pose: FloatArray, depth: FloatArray, tiltDeg: Float) {
        if (replay.size >= MAX_REPLAY) replay.removeAt(Random.nextInt(replay.size))
        replay.add(Sample(stack, pose, depth))
        totalSamples++
        tiltCoverage[tiltBin(tiltDeg)]++
        stepBudget = minOf(stepBudget + STEPS_PER_SAMPLE, MAX_BUDGET)
        if (recordToDisk) {
            val out = sampleOut ?: FileOutputStream(
                File(samplesDir.apply { mkdirs() }, "session_${System.currentTimeMillis()}.mdp")
            ).also { sampleOut = it }
            val bb = ByteBuffer.allocate(RECORD_FLOATS * 4).order(ByteOrder.LITTLE_ENDIAN)
            bb.asFloatBuffer().put(stack).put(pose).put(depth)
            out.write(bb.array())
        }
    }

    /** Samples learned near [tiltDeg]. */
    fun coverageAt(tiltDeg: Float) = tiltCoverage[tiltBin(tiltDeg)]

    /**
     * Continuous learning: each new camera-labelled sample earns STEPS_PER_SAMPLE training
     * steps, spent here in short bursts. So the AI learns as fast as new data arrives, but
     * doesn't keep grinding old data with nothing new (that only memorises it).
     * Returns the steps taken.
     */
    fun trainFor(maxMillis: Long): Int {
        if (replay.size < BATCH || stepBudget < 1f) return 0
        val end = System.nanoTime() + maxMillis * 1_000_000
        var n = 0
        while (stepBudget >= 1f && System.nanoTime() < end) {
            train(1)
            stepBudget -= 1f
            n++
        }
        return n
    }

    /** Steps still owed to recently added samples. */
    val pendingSteps get() = stepBudget.toInt()

    /** Reload the most recent samples from disk so learning continues across restarts. */
    private fun loadRecentSamples() {
        val files = samplesDir.listFiles { f -> f.name.endsWith(".mdp") }?.sortedByDescending { it.lastModified() } ?: return
        val recordBytes = RECORD_FLOATS * 4
        val bb = ByteBuffer.allocate(recordBytes).order(ByteOrder.LITTLE_ENDIAN)
        for (f in files) {
            if (replay.size >= MAX_REPLAY) break
            try {
                java.io.DataInputStream(f.inputStream().buffered(1 shl 16)).use { inp ->
                    while (replay.size < MAX_REPLAY) {
                        bb.clear()
                        inp.readFully(bb.array())
                        val fb = bb.asFloatBuffer()
                        val e = FloatArray(STACK_FLOATS).also { fb.get(it) }
                        val p = FloatArray(POSE_DIM).also { fb.get(it) }
                        val d = FloatArray(DEPTH_FLOATS).also { fb.get(it) }
                        replay.add(Sample(e, p, d))
                    }
                }
            } catch (_: java.io.EOFException) {
                // end of this file (or a partial last record); move to the next one
            } catch (e: Exception) {
                Log.w(TAG, "skip ${f.name}: $e")
            }
        }
        Log.i(TAG, "reloaded ${replay.size} samples for replay")
    }

    /** A few SGD steps on random replay batches. Returns the mean loss (NaN if no data yet). */
    // Reused across steps (allocating direct buffers 50+ times a second churns native memory).
    private val trainEcho = direct(BATCH * STACK_FLOATS)
    private val trainPose = direct(BATCH * POSE_DIM)
    private val trainDepth = direct(BATCH * DEPTH_FLOATS)
    private val trainShapes = direct(BATCH * ShapeFit.SHAPE_DIM)
    private val trainLoss = direct(1)
    private val trainShapeLoss = direct(1)
    private val trainLr = direct(1)

    fun train(steps: Int): Float {
        if (replay.size < BATCH) return Float.NaN
        var total = 0f
        var shapeTotal = 0f
        var done = 0
        repeat(steps) {
            trainEcho.clear(); trainPose.clear(); trainDepth.clear(); trainShapes.clear()
            val ef = trainEcho.asFloatBuffer(); val pf = trainPose.asFloatBuffer(); val df = trainDepth.asFloatBuffer()
            val sf = trainShapes.asFloatBuffer()
            repeat(BATCH) {
                val sample = replay[Random.nextInt(replay.size)]
                ef.put(sample.echo); pf.put(sample.pose); df.put(sample.depth); sf.put(sample.shapes)
            }
            trainLoss.clear(); trainShapeLoss.clear()
            trainLr.clear(); trainLr.asFloatBuffer().put(lr)
            interpreter.runSignature(
                mapOf("echo" to trainEcho, "pose" to trainPose, "depth" to trainDepth, "shapes" to trainShapes, "lr" to trainLr),
                mapOf("loss" to trainLoss, "shape_loss" to trainShapeLoss), "train")
            val l = trainLoss.toFloats(1)[0]
            val sl = trainShapeLoss.toFloats(1)[0]
            if (!l.isFinite() || l > MAX_SANE_LOSS || !sl.isFinite() || sl > MAX_SANE_LOSS) {
                rollback("loss $l / shapes $sl")
                return Float.NaN
            }
            total += l
            shapeTotal += sl
            done++
            trainedSteps++
            if (trainedSteps % SNAPSHOT_EVERY == 0) snapshot()
        }
        lastLoss = if (done > 0) total / done else Float.NaN
        lastShapeLoss = if (done > 0) shapeTotal / done else Float.NaN
        return lastLoss
    }

    fun save() {
        val outputs = currentWeights() ?: run {
            Log.w(TAG, "not saving: weights are invalid")
            return
        }
        weightsFile.parentFile?.mkdirs()
        val tmp = File(weightsFile.path + ".tmp")
        DataOutputStream(tmp.outputStream().buffered()).use { out ->
            out.writeInt(outputs.size)
            for ((_, buf) in outputs) {
                buf.rewind()
                out.writeInt(buf.remaining())
                val bytes = ByteArray(buf.remaining()); buf.get(bytes); out.write(bytes)
            }
        }
        tmp.renameTo(weightsFile) // atomic: a crash mid-save can't leave a corrupt file
        metaFile.writeText("$trainedSteps $totalSamples $modelId $stepBudget")
        coverageFile.writeText(tiltCoverage.joinToString(" "))
        sampleOut?.flush()
    }

    private fun restore(target: Interpreter) {
        val names = target.getSignatureInputs("set_weights").sortedBy { it.removePrefix("w").toInt() }
        // Start from the shipped weights: a file saved before the shapes head existed holds
        // only the depth head (w0..w3), and the shapes head keeps its pretrained weights.
        val inputs = LinkedHashMap<String, Any>(currentWeights(target) ?: error("shipped weights invalid"))
        DataInputStream(weightsFile.inputStream().buffered()).use { inp ->
            val count = inp.readInt()
            require(count in 1..names.size) { "weights file does not match model" }
            for (name in names.take(count)) {
                val bytes = ByteArray(inp.readInt()); inp.readFully(bytes)
                require(bytes.size == target.getInputTensorFromSignature(name, "set_weights").numBytes())
                inputs[name] = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes)
                    .also { it.rewind() }
            }
        }
        inputs.values.forEach { (it as ByteBuffer).rewind() }
        target.runSignature(inputs, mapOf("ok" to direct(1)), "set_weights")
    }

    /** Forget on-device learning, including the recorded samples it would otherwise reload. */
    fun reset() {
        weightsFile.delete()
        metaFile.delete()
        interpreter.close()
        interpreter = open()
        replay.clear()
        stepBudget = 0f
        lr = DEFAULT_LR
        rollbacks = 0
        snapshot()
        sampleOut?.close(); sampleOut = null
        samplesDir.listFiles { f -> f.name.endsWith(".mdp") }?.forEach { it.delete() }
        tiltCoverage.fill(0)
        coverageFile.delete()
        trainedSteps = 0
        totalSamples = 0
        lastLoss = Float.NaN
        lastShapeLoss = Float.NaN
    }

    fun close() {
        sampleOut?.close()
        interpreter.close()
    }

    private fun weightNames(target: Interpreter = interpreter) =
        target.getSignatureOutputs("get_weights").sortedBy { it.removePrefix("w").toInt() }

    // Last in the class on purpose: Kotlin runs property initializers and init blocks in
    // textual order, and these use fields declared above (good, trainEcho, ...).
    init {
        interpreter = open()
        val freshHead = !weightsFile.exists()
        snapshot()
        loadRecentSamples()
        // New model (or first run) but samples on disk: queue learning them again.
        if (freshHead && stepBudget < 1f) stepBudget = minOf(replay.size * STEPS_PER_SAMPLE, MAX_BUDGET)
    }

    companion object {
        private const val TAG = "EchoBrain"
        /** CRC32s of earlier shipped models with this model's trunk and depth head layout. */
        private val SAME_TRUNK_MODELS = setOf(2608876991L)   // pose input, before the shapes head
        const val BATCH = 8           // fixed by the exported train signature
        const val POSE_DIM = 2        // (tilt / 90°, roll / 90°); ml/murdock_model.py pose_features
        private const val RECORD_FLOATS = STACK_FLOATS + POSE_DIM + DEPTH_FLOATS  // one .mdp record
        const val TILT_FROM_DEG = -30f
        const val TILT_BIN_DEG = 10f
        const val TILT_BINS = 12      // -30° .. 90°

        fun tiltBin(tiltDeg: Float) = ((tiltDeg - TILT_FROM_DEG) / TILT_BIN_DEG).toInt().coerceIn(0, TILT_BINS - 1)

        /** Phone angle -> head input; mirrors ml/murdock_model.py pose_features. */
        fun poseFeatures(tiltDeg: Float, rollDeg: Float) =
            floatArrayOf((tiltDeg / 90f).coerceIn(-1.5f, 1.5f), (rollDeg / 90f).coerceIn(-1.5f, 1.5f))
        // Simulated on BatVision with the clipped trainer: 0.1 -> 23% better, 0.5 -> 27% and
        // still stable; 0.1 leaves headroom for noisier phone data.
        private const val DEFAULT_LR = 0.1f
        private const val MIN_LR = 0.005f
        private const val MAX_SANE_LOSS = 5f      // L1 + edge loss on [0,1] maps stays well below 2
        private const val SNAPSHOT_EVERY = 500
        private const val MAX_REPLAY = 2000           // ~40 MB in memory
        // Simulated on BatVision (ml/simulate_teach.py, 12k steps): 600 samples give 23% lower
        // error, 1500 give 25%; so ~20 steps per new sample captures most of the gain quickly.
        const val STEPS_PER_SAMPLE = 20f
        private const val MAX_BUDGET = 20000f

        fun direct(floats: Int): ByteBuffer =
            ByteBuffer.allocateDirect(floats * 4).order(ByteOrder.nativeOrder())

        fun direct(values: FloatArray): ByteBuffer =
            direct(values.size).also { it.asFloatBuffer().put(values) }

        fun ByteBuffer.toFloats(n: Int): FloatArray {
            rewind()
            return FloatArray(n).also { asFloatBuffer().get(it) }
        }
    }
}
