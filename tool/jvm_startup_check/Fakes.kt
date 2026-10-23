package org.tensorflow.lite

import java.nio.ByteBuffer

/** Stand-in for LiteRT so EchoBrain's own startup logic can run on a desktop JVM. */
interface Tensor { fun numBytes(): Int }

class Interpreter(model: ByteBuffer, options: Options) : AutoCloseable {
    class Options {
        fun setNumThreads(n: Int) = this
        fun addDelegate(d: Any) = this
    }
    private val head = mapOf("w0" to 512 * 1024, "w1" to 1024, "w2" to 1024 * 1024, "w3" to 1024)
    private fun t(n: Int) = object : Tensor { override fun numBytes() = n * 4 }
    fun getSignatureInputs(key: String): Array<String> = if (key == "set_weights") head.keys.toTypedArray() else arrayOf("echo")
    fun getSignatureOutputs(key: String): Array<String> = if (key == "get_weights") head.keys.toTypedArray() else arrayOf("depth")
    fun getInputTensorFromSignature(name: String, key: String): Tensor = t(head[name] ?: 0)
    fun getOutputTensorFromSignature(name: String, key: String): Tensor = t(head[name] ?: 0)
    fun runSignature(inputs: Map<String, Any>, outputs: Map<String, Any>, key: String) {
        for (o in outputs.values) {
            val b = o as ByteBuffer
            b.clear()
            val fb = b.asFloatBuffer()
            while (fb.hasRemaining()) fb.put(if (key == "infer") 0.5f else 0.01f)
        }
    }
    override fun close() {}
}
