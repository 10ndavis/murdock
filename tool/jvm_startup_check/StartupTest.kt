import com.murdock.murdock.EchoBrain
import java.io.File
import java.nio.ByteBuffer
import java.nio.file.Files

fun main() {
    val dir = Files.createTempDirectory("brain").toFile()
    val weights = File(dir, "brain/head.weights")
    val model = ByteBuffer.allocateDirect(1024).put(ByteArray(1024) { it.toByte() })
    // Launch 1: fresh install, learn a bit, save (writes weights + meta with the model id).
    val first = EchoBrain(model, weights, File(dir, "samples"))
    first.save()
    println("launch 1 OK; meta = " + File(weights.path + ".meta").readText())
    // Launch 2: the case that crashed on the phone (saved meta present at startup).
    val second = EchoBrain(model, weights, File(dir, "samples"))
    println("launch 2 OK; infer ok = " + second.infer(FloatArray(16 * 128 * 2)).all { it.isFinite() })
}
