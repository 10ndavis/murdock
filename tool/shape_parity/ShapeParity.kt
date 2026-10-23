import com.murdock.murdock.ShapeFit
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** ShapeParity <maps.f32> <shapes.f32>: runs ShapeFit.kt over little-endian 32x32 maps. */
fun main(args: Array<String>) {
    val n = ShapeFit.N * ShapeFit.N
    val inp = ByteBuffer.wrap(File(args[0]).readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val count = inp.remaining() / n
    val out = ByteBuffer.allocate(count * ShapeFit.SHAPE_DIM * 4).order(ByteOrder.LITTLE_ENDIAN)
    val map = FloatArray(n)
    repeat(count) {
        inp.get(map)
        for (v in ShapeFit.fit(map)) out.putFloat(v)
    }
    File(args[1]).writeBytes(out.array())
}
