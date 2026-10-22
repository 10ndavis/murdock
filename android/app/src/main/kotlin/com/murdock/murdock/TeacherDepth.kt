package com.murdock.murdock

import com.murdock.murdock.EchoBrain.Companion.direct
import com.murdock.murdock.EchoBrain.Companion.toFloats
import com.murdock.murdock.SonarConfig.DEPTH_SIZE
import org.tensorflow.lite.Interpreter
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Camera "teacher": MiDaS v2.1 small (monocular relative depth) turns a camera frame
 * into the depth label the echo student learns from. Only used while the lights are on.
 */
class TeacherDepth(file: File) {
    private val interpreter = Interpreter(file, Interpreter.Options().setNumThreads(4))

    /** rgb: SIZE*SIZE*3 bytes, row-major, upright. Returns DEPTH_SIZE^2 nearness in [0, 1]. */
    fun run(rgb: ByteArray): FloatArray {
        require(rgb.size == SIZE * SIZE * 3)
        val input = direct(SIZE * SIZE * 3)
        val f = input.asFloatBuffer()
        for (i in rgb.indices) {
            val c = i % 3
            f.put(((rgb[i].toInt() and 0xff) - MEAN[c]) / STD[c])
        }
        val out = direct(SIZE * SIZE)
        interpreter.run(input, out)
        val inv = out.toFloats(SIZE * SIZE)

        // Box-downsample to the student's resolution, then robust min/max normalise
        // (same as ml/features.py normalize_nearness).
        val k = SIZE / DEPTH_SIZE
        val small = FloatArray(DEPTH_SIZE * DEPTH_SIZE)
        for (y in 0 until SIZE) for (x in 0 until SIZE) {
            small[(y / k) * DEPTH_SIZE + x / k] += inv[y * SIZE + x] / (k * k)
        }
        val sorted = small.sortedArray()
        val lo = sorted[(sorted.size * 0.02).toInt()]
        val hi = sorted[(sorted.size * 0.98).toInt()]
        val span = maxOf(hi - lo, 1e-6f)
        for (i in small.indices) small[i] = ((small[i] - lo) / span).coerceIn(0f, 1f)
        return small
    }

    fun close() = interpreter.close()

    companion object {
        const val SIZE = 256
        const val URL_MIDAS = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model_opt.tflite"
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)

        /** Downloads MiDaS (~66 MB) to [dest]; reports progress in [0, 1]. */
        fun download(dest: File, progress: (Float) -> Unit) {
            val tmp = File(dest.path + ".part")
            var url = URL(URL_MIDAS)
            var conn: HttpURLConnection
            while (true) { // follow GitHub's cross-host release redirect
                conn = url.openConnection() as HttpURLConnection
                conn.instanceFollowRedirects = false
                val code = conn.responseCode
                if (code in 300..399) { url = URL(url, conn.getHeaderField("Location")); conn.disconnect(); continue }
                if (code != 200) throw IllegalStateException("HTTP $code")
                break
            }
            val total = conn.contentLengthLong
            conn.inputStream.use { input ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(1 shl 16)
                    var done = 0L
                    var lastReport = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        if (total > 0 && done - lastReport > 1_000_000) {
                            lastReport = done; progress(done.toFloat() / total)
                        }
                    }
                }
            }
            if (!tmp.renameTo(dest)) throw IllegalStateException("could not save model")
            progress(1f)
        }
    }
}
