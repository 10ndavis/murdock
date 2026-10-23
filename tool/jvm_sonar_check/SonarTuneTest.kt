import android.content.ContextWrapper
import com.murdock.murdock.SonarConfig
import com.murdock.murdock.SonarEngine
import java.io.File
import java.nio.file.Files
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Drives SonarEngine's real per-ping DSP (speaker cancellation, clarity, AUTO-TUNE) with
 * simulated audio: the speaker's own chirp plus a wall 4.5 ft away, in noise, with speaker
 * rattle that grows with loudness. No audio hardware is touched.
 */
class FilesContext(private val dir: File) : ContextWrapper(null) {
    override fun getFilesDir() = dir
}

private val N = SonarConfig.RANGE_SAMPLES + SonarConfig.CHIRP_LEN + 64
private const val DIRECT = 32
private const val WALL_LAG = 384      // 4.5 ft round trip at 48 kHz
private const val LATENCY_PINGS = 3   // audio queued in the speaker before a change is heard

class Room(val wall: Boolean, val noise: Float, val rattle: Float, seed: Int = 1) {
    private val rnd = Random(seed)
    private val chirp = SonarConfig.chirpIQ[0]

    /** One ping's mic audio (both channels) at the loudness the speaker is playing. */
    fun ping(loud: Float): Array<FloatArray> = Array(SonarConfig.CHANNELS) { c ->
        val x = FloatArray(N)
        for (n in 0 until SonarConfig.CHIRP_LEN) {
            // Direct path, plus a short ringing tail through the phone's body.
            x[DIRECT + n] += 0.5f * loud * chirp[n]
            x[DIRECT + 40 + n] += 0.15f * loud * chirp[n] * (if (c == 0) 1f else -1f)
            if (wall) x[DIRECT + WALL_LAG + n] += 0.004f * loud * chirp[n]
        }
        val sigma = noise + rattle * loud * loud * loud
        for (i in x.indices) x[i] += (rnd.nextFloat() * 2 - 1) * sigma * 1.73f
        x
    }
}

class Rig(private val engine: SonarEngine) {
    private val emit = SonarEngine::class.java.getDeclaredMethod("emitPing", Array<FloatArray>::class.java, Int::class.java)
        .apply { isAccessible = true }
    private val peak = SonarEngine::class.java.getDeclaredField("micPeak").apply { isAccessible = true }
    private val heard = ArrayDeque<Float>()
    var lastClean: FloatArray? = null

    fun run(room: Room, pings: Int) = repeat(pings) {
        heard.addLast(engine.loudness)
        val loud = if (heard.size > LATENCY_PINGS) heard.removeFirst() else heard.first()
        val buf = room.ping(loud)
        var p = 0f
        for (ch in buf) for (v in ch) p = max(p, abs(v))
        peak.setFloat(engine, max(peak.getFloat(engine), p))
        emit.invoke(engine, buf, DIRECT)
    }
}

/** Mean leftover speaker sound in the first ~3 ft after cancellation (display profile, 0..1). */
fun residual(clean: FloatArray): Float {
    var s = 0f
    for (b in SonarConfig.ECHO_ZONE_BINS until 20) s += clean[b * SonarConfig.CHANNELS]
    return s / (20 - SonarConfig.ECHO_ZONE_BINS)
}

fun check(ok: Boolean, what: String) {
    println((if (ok) "PASS " else "FAIL ") + what)
    if (!ok) throw IllegalStateException(what)
}

fun newEngine(
    dir: File, events: MutableList<Map<String, Any>>,
    raw: (FloatArray) -> Unit = {}, clean: (FloatArray) -> Unit = {},
): SonarEngine =
    SonarEngine(FilesContext(dir), { r, c, _, _ -> raw(r); clean(c) }, { println("error: $it") }, {}).also {
        it.onTune = { e -> events += e }
    }

fun main() {
    val dir = Files.createTempDirectory("sonar").toFile()

    // 1. AUTO-TUNE in front of a wall: quiet levels are noisy, the loudest rattles.
    val events = ArrayList<Map<String, Any>>()
    var lastClean = FloatArray(0)
    var lastRaw = FloatArray(0)
    val engine = newEngine(dir, events, raw = { lastRaw = it.copyOf() }) { lastClean = it.copyOf() }
    val rig = Rig(engine)
    val wallRoom = Room(wall = true, noise = 0.0006f, rattle = 0.0016f)
    engine.calibrate()
    rig.run(wallRoom, 60)
    check(engine.calibrationState == "done", "initial calibration at default loudness")
    engine.autoTune()
    rig.run(wallRoom, 400)
    val done = events.last()
    @Suppress("UNCHECKED_CAST") val clarity = done["clarity"] as List<Double>
    println("  clarity by loudness ${done["levels"]}: ${clarity.map { "%.1f".format(it) }}")
    check(done["state"] == "done", "tune finishes (state=${done["state"]}, ${events.size} progress events)")
    val best = (done["best"] as Double).toFloat()
    println("  picked loudness $best at %.1f dB".format(done["bestDb"] as Double))
    check(best > 0.3f && best < 1.0f, "picks a middle loudness when quiet is noisy and loud rattles")
    check(engine.loudness == best, "engine keeps the picked loudness")
    check(!engine.busy, "not busy after the tune")
    check((done["progress"] as Double) == 1.0, "progress reaches 100%")
    check(File(dir, "speaker_template.bin").length() == (SonarConfig.CHANNELS * SonarEngine.CANCEL_LEN * 8 + 8).toLong(),
        "speaker filter saved with its loudness")

    // 2. Changing loudness by hand keeps the speaker filter working (template scales).
    val quiet = Room(wall = true, noise = 0.0001f, rattle = 0f)
    rig.run(quiet, 20)
    val atTuned = residual(lastClean)
    engine.loudness = 0.3f
    rig.run(quiet, 20)
    val atQuiet = residual(lastClean)
    engine.loudness = 1.0f
    rig.run(quiet, 20)
    val atLoud = residual(lastClean)
    val unfiltered = residual(lastRaw)
    // Same, but pretending the filter was learned at 100% (no scaling): the old behavior.
    val calLoud = SonarEngine::class.java.getDeclaredField("calLoudness").apply { isAccessible = true }
    val learnedAt = calLoud.getFloat(engine)
    calLoud.setFloat(engine, 1.0f)
    rig.run(quiet, 5)
    val unscaled = residual(lastClean)
    calLoud.setFloat(engine, learnedAt)
    println("  speaker residual: tuned %.3f, 30%% %.3f, 100%% %.3f; unscaled filter %.3f, no filter %.3f"
        .format(atTuned, atQuiet, atLoud, unscaled, unfiltered))
    check(atLoud < unscaled - 0.03f && atLoud < unfiltered - 0.1f, "scaling the filter beats not scaling it")
    check(atLoud < atTuned + 0.08f && atQuiet < atTuned + 0.08f, "speaker stays filtered at other loudness levels")

    // 3. Reload: the template remembers the loudness it was learned at.
    val events2 = ArrayList<Map<String, Any>>()
    var clean2 = FloatArray(0)
    val reloaded = newEngine(dir, events2, clean = { clean2 = it.copyOf() })
    SonarEngine::class.java.getDeclaredMethod("loadTemplate").apply { isAccessible = true }.invoke(reloaded)
    reloaded.loudness = 1.0f
    Rig(reloaded).run(quiet, 20)
    println("  after reload at 100%%: residual %.3f".format(residual(clean2)))
    check(residual(clean2) < atTuned + 0.08f, "reloaded filter scales to the current loudness")

    // 4. No wall, only noise: tune fails and puts everything back.
    events.clear()
    engine.loudness = 0.6f
    engine.autoTune()
    rig.run(Room(wall = false, noise = 0.02f, rattle = 0f, seed = 7), 400)
    println("  no wall: ${events.last()["state"]} · ${events.last()["message"]}")
    check(events.last()["state"] == "failed", "no echo -> tune fails")
    check(engine.loudness == 0.6f, "failed tune restores the loudness")
    check(engine.calibrationState != "calibrating" && !engine.busy, "failed tune leaves the filter usable")

    // 5. Cancel half-way.
    events.clear()
    engine.autoTune()
    rig.run(wallRoom, 50)
    check(engine.busy, "busy while tuning")
    engine.cancelTune()
    rig.run(wallRoom, 1)
    check(events.last()["state"] == "cancelled" && engine.loudness == 0.6f && !engine.busy, "cancel restores the loudness")

    // 6. Mic overloaded at every level: refuses, with advice.
    events.clear()
    engine.autoTune()
    val loudRoom = object {
        fun ping() = Array(SonarConfig.CHANNELS) { FloatArray(N) { 1f } }
    }
    val emit = SonarEngine::class.java.getDeclaredMethod("emitPing", Array<FloatArray>::class.java, Int::class.java)
        .apply { isAccessible = true }
    val peak = SonarEngine::class.java.getDeclaredField("micPeak").apply { isAccessible = true }
    repeat(400) { peak.setFloat(engine, 1f); emit.invoke(engine, loudRoom.ping(), DIRECT) }
    println("  clipping: ${events.last()["state"]} · ${events.last()["message"]}")
    check(events.last()["state"] == "failed" && (events.last()["message"] as String).contains("overloaded"), "clipping is caught")

    println("all sonar checks passed")
}
