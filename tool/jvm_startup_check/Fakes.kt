package org.tensorflow.lite

import java.nio.ByteBuffer

/** Stand-in for LiteRT so EchoBrain's own startup logic can run on a desktop JVM. */
interface Tensor { fun numBytes(): Int }

class Interpreter(model: ByteBuffer, options: Options) : AutoCloseable {
    class Options {
        fun setNumThreads(n: Int) = this
        fun addDelegate(d: Any) = this
    }
    // w0..w3 depth head, w4..w7 shapes head (ml/export_tflite.py).
    private val head = mapOf("w0" to 514 * 1024, "w1" to 1024, "w2" to 1024 * 1024, "w3" to 1024,
        "w4" to 514 * 256, "w5" to 256, "w6" to 256 * 24, "w7" to 24)
    companion object {
        /** Inputs of the last "train" call, for the test to check what EchoBrain sends. */
        var lastTrainInputs: Set<String> = emptySet()
    }
    private fun t(n: Int) = object : Tensor { override fun numBytes() = n * 4 }
    fun getSignatureInputs(key: String): Array<String> = if (key == "set_weights") head.keys.toTypedArray() else arrayOf("echo")
    fun getSignatureOutputs(key: String): Array<String> = if (key == "get_weights") head.keys.toTypedArray() else arrayOf("depth")
    fun getInputTensorFromSignature(name: String, key: String): Tensor = t(head[name] ?: 0)
    fun getOutputTensorFromSignature(name: String, key: String): Tensor = t(head[name] ?: 0)
    fun runSignature(inputs: Map<String, Any>, outputs: Map<String, Any>, key: String) {
        if (key == "set_weights") check(inputs.keys == head.keys) { "set_weights needs all of ${head.keys}, got ${inputs.keys}" }
        if (key == "train") lastTrainInputs = inputs.keys
        for (o in outputs.values) {
            val b = o as ByteBuffer
            b.clear()
            val fb = b.asFloatBuffer()
            while (fb.hasRemaining()) fb.put(if (key == "infer") 0.5f else 0.01f)
        }
    }
    override fun close() {}
}
