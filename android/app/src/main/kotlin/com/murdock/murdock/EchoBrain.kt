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

    // Declared after the counters: open() restores them, and Kotlin runs initializers in order.
    private var interpreter = open()
    var recordToDisk = true

    private val metaFile get() = File(weightsFile.path + ".meta")

    private fun open(): Interpreter {
        modelBytes.rewind()
        val it = Interpreter(modelBytes, Interpreter.Options().setNumThreads(2))
        if (weightsFile.exists()) {
            try {
                restore(it)
                metaFile.takeIf { f -> f.exists() }?.readText()?.trim()?.split(" ")?.let { (steps, samples) ->
                    trainedSteps = steps.toInt(); totalSamples = samples.toInt()
                }
            } catch (e: Exception) { Log.w(TAG, "restore failed: $e") }
        }
        return it
    }

    fun infer(stack: FloatArray): FloatArray {
        val out = direct(DEPTH_FLOATS)
        interpreter.runSignature(mapOf("echo" to direct(stack)), mapOf("depth" to out), "infer")
        return out.toFloats(DEPTH_FLOATS)
    }

    /** Store a (echo, camera depth) pair for replay training and the PC dataset. */
    fun addSample(stack: FloatArray, depth: FloatArray) {
        if (replay.size >= MAX_REPLAY) replay.removeAt(Random.nextInt(replay.size))
        replay.add(stack to depth)
        totalSamples++
        if (recordToDisk) {
            val out = sampleOut ?: FileOutputStream(
                File(samplesDir.apply { mkdirs() }, "session_${System.currentTimeMillis()}.mdk")
            ).also { sampleOut = it }
            val bb = ByteBuffer.allocate((STACK_FLOATS + DEPTH_FLOATS) * 4).order(ByteOrder.LITTLE_ENDIAN)
            bb.asFloatBuffer().put(stack).put(depth)
            out.write(bb.array())
        }
    }

    /** A few SGD steps on random replay batches. Returns the mean loss (NaN if no data yet). */
    fun train(steps: Int): Float {
        if (replay.size < BATCH) return Float.NaN
        val echo = direct(BATCH * STACK_FLOATS)
        val depth = direct(BATCH * DEPTH_FLOATS)
        val loss = direct(1)
        var total = 0f
        repeat(steps) {
            echo.clear(); depth.clear()
            val ef = echo.asFloatBuffer(); val df = depth.asFloatBuffer()
            repeat(BATCH) {
                val (e, d) = replay[Random.nextInt(replay.size)]
                ef.put(e); df.put(d)
            }
            loss.clear()
            interpreter.runSignature(mapOf("echo" to echo, "depth" to depth), mapOf("loss" to loss), "train")
            total += loss.toFloats(1)[0]
            trainedSteps++
        }
        lastLoss = total / steps
        return lastLoss
    }

    fun save() {
        val outputs = weightNames().associateWith {
            direct(interpreter.getOutputTensorFromSignature(it, "get_weights").numBytes() / 4)
        }
        interpreter.runSignature(mapOf("dummy" to direct(1)), outputs, "get_weights")
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
        metaFile.writeText("$trainedSteps $totalSamples")
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

    /** Forget on-device training: back to the weights shipped in the asset. */
    fun reset() {
        weightsFile.delete()
        metaFile.delete()
        interpreter.close()
        interpreter = open()
        replay.clear()
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

    companion object {
        private const val TAG = "EchoBrain"
        const val BATCH = 8           // fixed by the exported train signature
        private const val MAX_REPLAY = 3000

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
