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
 * The echo -> depth student (assets/models/echo_student.tflite, built by ml/export_tflite.py).
 *
 * Signatures: infer, train (head only), get_weights / set_weights for persistence.
 * Every call runs on the caller's thread; MainActivity funnels them through one executor
 * because the interpreter is not thread-safe and infer/train share its variables.
 */
class EchoBrain(
    private val modelBytes: ByteBuffer,
    private val weightsFile: File,
    private val samplesDir: File,
) {
    private val replay = ArrayList<Pair<FloatArray, FloatArray>>()
    private var sampleOut: FileOutputStream? = null

    // Lifetime counters, persisted next to the weights so progress survives restarts.
    var trainedSteps = 0; private set
    var totalSamples = 0; private set
    var lastLoss = Float.NaN; private set
    val sampleCount get() = replay.size

    @Volatile private var stepBudget = 0f

    // Declared after the counters: open() restores them, and Kotlin runs initializers in order.
    private var interpreter = open()


    var recordToDisk = true

    private val metaFile get() = File(weightsFile.path + ".meta")

    /** Identifies the shipped model, so learning saved on top of an older model is dropped. */
    private val modelId: Long by lazy {
        java.util.zip.CRC32().apply { update(modelBytes.duplicate().apply { rewind() }) }.value
    }

    private fun open(): Interpreter {
        modelBytes.rewind()
        val it = Interpreter(modelBytes, Interpreter.Options().setNumThreads(2))
        // meta: "<steps> <samples> <modelId>". A head trained on another model's trunk features
        // would produce garbage, so after an app update with a new model we start fresh.
        val meta = metaFile.takeIf { f -> f.exists() }?.readText()?.trim()?.split(" ")
        val sameModel = meta != null && meta.size >= 3 && meta[2].toLongOrNull() == modelId
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
            } catch (e: Exception) { Log.w(TAG, "restore failed: $e") }
        } else if (weightsFile.exists()) {
            Log.i(TAG, "model changed; discarding on-device learning from the previous model")
            weightsFile.delete(); metaFile.delete()
        }
        return it
    }

    private fun outputIsFinite(target: Interpreter): Boolean {
        val out = direct(DEPTH_FLOATS)
        target.runSignature(mapOf("echo" to direct(STACK_FLOATS)), mapOf("depth" to out), "infer")
        return out.toFloats(DEPTH_FLOATS).all { it.isFinite() }
    }

    // ── Training safety ─────────────────────────────────────────────────────────────────
    // The exported train step clips gradients, but if a step or a prediction ever comes out
    // invalid anyway, roll back to the last good weights and halve the learning rate.
    var lr = DEFAULT_LR; private set
    var rollbacks = 0; private set
    private var good: Map<String, ByteBuffer>? = null

    /** Copy of the current head weights, or null if any value is invalid. */
    private fun currentWeights(): Map<String, ByteBuffer>? {
        val outputs = weightNames().associateWith {
            direct(interpreter.getOutputTensorFromSignature(it, "get_weights").numBytes() / 4)
        }
        interpreter.runSignature(mapOf("dummy" to direct(1)), outputs, "get_weights")
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

    fun infer(stack: FloatArray): FloatArray {
        repeat(2) {
            val out = direct(DEPTH_FLOATS)
            interpreter.runSignature(mapOf("echo" to direct(stack)), mapOf("depth" to out), "infer")
            val d = out.toFloats(DEPTH_FLOATS)
            if (d.all { it.isFinite() }) return d
            rollback("invalid prediction")
        }
        return FloatArray(DEPTH_FLOATS)
    }

    /** Store a (echo, camera depth) pair for replay training and the PC dataset. */
    fun addSample(stack: FloatArray, depth: FloatArray) {
        if (replay.size >= MAX_REPLAY) replay.removeAt(Random.nextInt(replay.size))
        replay.add(stack to depth)
        totalSamples++
        stepBudget = minOf(stepBudget + STEPS_PER_SAMPLE, MAX_BUDGET)
        if (recordToDisk) {
            val out = sampleOut ?: FileOutputStream(
                File(samplesDir.apply { mkdirs() }, "session_${System.currentTimeMillis()}.mdk")
            ).also { sampleOut = it }
            val bb = ByteBuffer.allocate((STACK_FLOATS + DEPTH_FLOATS) * 4).order(ByteOrder.LITTLE_ENDIAN)
            bb.asFloatBuffer().put(stack).put(depth)
            out.write(bb.array())
        }
    }

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
        val files = samplesDir.listFiles { f -> f.name.endsWith(".mdk") }?.sortedByDescending { it.lastModified() } ?: return
        val recordBytes = (STACK_FLOATS + DEPTH_FLOATS) * 4
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
                        val d = FloatArray(DEPTH_FLOATS).also { fb.get(it) }
                        replay.add(e to d)
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
    private val trainDepth = direct(BATCH * DEPTH_FLOATS)
    private val trainLoss = direct(1)
    private val trainLr = direct(1)

    fun train(steps: Int): Float {
        if (replay.size < BATCH) return Float.NaN
        var total = 0f
        var done = 0
        repeat(steps) {
            trainEcho.clear(); trainDepth.clear()
            val ef = trainEcho.asFloatBuffer(); val df = trainDepth.asFloatBuffer()
            repeat(BATCH) {
                val (e, d) = replay[Random.nextInt(replay.size)]
                ef.put(e); df.put(d)
            }
            trainLoss.clear()
            trainLr.clear(); trainLr.asFloatBuffer().put(lr)
            interpreter.runSignature(
                mapOf("echo" to trainEcho, "depth" to trainDepth, "lr" to trainLr), mapOf("loss" to trainLoss), "train")
            val l = trainLoss.toFloats(1)[0]
            if (!l.isFinite() || l > MAX_SANE_LOSS) {
                rollback("loss $l")
                return Float.NaN
            }
            total += l
            done++
            trainedSteps++
            if (trainedSteps % SNAPSHOT_EVERY == 0) snapshot()
        }
        lastLoss = if (done > 0) total / done else Float.NaN
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
        metaFile.writeText("$trainedSteps $totalSamples $modelId")
        sampleOut?.flush()
    }

    private fun restore(target: Interpreter) {
        val names = target.getSignatureInputs("set_weights").sortedBy { it.removePrefix("w").toInt() }
        val inputs = LinkedHashMap<String, Any>()
        DataInputStream(weightsFile.inputStream().buffered()).use { inp ->
            require(inp.readInt() == names.size) { "weights file does not match model" }
            for (name in names) {
                val bytes = ByteArray(inp.readInt()); inp.readFully(bytes)
                require(bytes.size == target.getInputTensorFromSignature(name, "set_weights").numBytes())
                inputs[name] = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes)
                    .also { it.rewind() }
            }
        }
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
        samplesDir.listFiles { f -> f.name.endsWith(".mdk") }?.forEach { it.delete() }
        trainedSteps = 0
        totalSamples = 0
        lastLoss = Float.NaN
    }

    fun close() {
        sampleOut?.close()
        interpreter.close()
    }

    private fun weightNames() =
        interpreter.getSignatureOutputs("get_weights").sortedBy { it.removePrefix("w").toInt() }

    // Last in the class on purpose: Kotlin runs property initializers and init blocks in
    // textual order, and these use fields declared above (good, trainEcho, ...).
    init {
        snapshot()
        loadRecentSamples()
    }

    companion object {
        private const val TAG = "EchoBrain"
        const val BATCH = 8           // fixed by the exported train signature
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
