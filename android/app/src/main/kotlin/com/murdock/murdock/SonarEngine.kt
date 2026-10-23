package com.murdock.murdock

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.util.Log
import com.murdock.murdock.SonarConfig.CHANNELS
import com.murdock.murdock.SonarConfig.CHIRP_LEN
import com.murdock.murdock.SonarConfig.LOG_FLOOR
import com.murdock.murdock.SonarConfig.PING_PERIOD
import com.murdock.murdock.SonarConfig.POOL
import com.murdock.murdock.SonarConfig.PROFILE_FLOATS
import com.murdock.murdock.SonarConfig.RANGE_BINS
import com.murdock.murdock.SonarConfig.RANGE_SAMPLES
import com.murdock.murdock.SonarConfig.SAMPLE_RATE
import com.murdock.murdock.SonarConfig.STACK
import com.murdock.murdock.SonarConfig.STACK_FLOATS
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Active sonar using the phone's own speaker and microphones.
 *
 * The speaker loops a 5 ms 17.5-21.5 kHz chirp every PING_PERIOD samples while the mics
 * record continuously. Each ping is matched-filtered (complex correlation with the chirp),
 * aligned to the direct speaker->mic peak, and reduced to a RANGE_BINS echo profile per
 * mic. This mirrors ml/features.py so models trained on a PC see the same features.
 */
class SonarEngine(
    private val context: Context,
    /** profile: raw echoes (model input); clean: speaker sound cancelled (for display). */
    private val onPing: (profile: FloatArray, clean: FloatArray, stack: FloatArray, directLevel: Float) -> Unit,
    private val onError: (String) -> Unit,
    private val onStats: (Map<String, Any>) -> Unit,
) {
    @Volatile private var running = false
    private var playThread: Thread? = null
    private var recordThread: Thread? = null

    @Volatile var stereo = false; private set
    @Volatile var sourceName = ""; private set

    // Live diagnostics, reported once a second through onStats.
    @Volatile private var speakerState = "starting"
    @Volatile private var micState = "starting"
    @Volatile private var micRms = 0f
    @Volatile private var pings = 0
    @Volatile private var calibrations = 0
    @Volatile private var lastDirect = 0f

    // Rolling history of profiles, oldest first when assembled into the model input.
    private val history = Array(STACK) { FloatArray(PROFILE_FLOATS) }
    private var historyHead = 0

    fun start() {
        if (running) return
        running = true
        speakerState = "starting"; micState = "starting"
        playThread = Thread({ guarded("speaker", ::playLoop) }, "sonar-play")
            .apply { priority = Thread.MAX_PRIORITY; start() }
        recordThread = Thread({ guarded("microphone", ::recordLoop) }, "sonar-record")
            .apply { priority = Thread.MAX_PRIORITY; start() }
    }

    fun stop() {
        running = false
        playThread?.join(500)
        recordThread?.join(500)
        playThread = null
        recordThread = null
    }

    /** An exception escaping a raw Thread kills the app, so report it and stop instead. */
    private fun guarded(what: String, loop: () -> Unit) {
        try {
            loop()
        } catch (t: Throwable) {
            Log.e(TAG, "$what failed", t)
            if (what == "speaker") speakerState = "failed: ${t.message}" else micState = "failed: ${t.message}"
            running = false
            onError("$what failed: ${t.message ?: t.javaClass.simpleName}")
            reportStats()
        }
    }

    private fun reportStats() {
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val vol = am.getStreamVolume(AudioManager.STREAM_MUSIC)
        val maxVol = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        onStats(mapOf(
            "running" to running,
            "speaker" to speakerState,
            "mic" to micState,
            "source" to sourceName,
            "stereo" to stereo,
            "micDb" to (20 * log10(max(micRms, 1e-7f))).toDouble(),
            "pingsPerSec" to pings,
            "calibrations" to calibrations,
            "direct" to lastDirect.toDouble(),
            "volume" to if (maxVol > 0) vol.toDouble() / maxVol else 0.0,
            "calibration" to calibrationState,
        ))
        pings = 0
    }

    private fun playLoop() {
        // 16-bit PCM: the one output format every Android device supports.
        val ping = ShortArray(PING_PERIOD)
        val chirp = SonarConfig.chirpIQ[0]
        for (n in 0 until CHIRP_LEN) ping[n] = (chirp[n] * SonarConfig.CHIRP_AMPLITUDE * Short.MAX_VALUE).toInt().toShort()
        val track = openTrack()
        try {
            track.play()
            speakerState = "playing"
            while (running) {
                val n = track.write(ping, 0, ping.size)
                if (n < 0) throw IllegalStateException("AudioTrack.write error $n")
            }
        } finally {
            track.stop()
            track.release()
        }
    }

    private fun openTrack(): AudioTrack {
        val minBuf = AudioTrack.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        var lastError: Exception? = null
        for (lowLatency in booleanArrayOf(true, false)) {
            try {
                val b = AudioTrack.Builder()
                    .setAudioAttributes(AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build())
                    .setAudioFormat(AudioFormat.Builder()
                        .setSampleRate(SAMPLE_RATE)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build())
                    .setBufferSizeInBytes(max(minBuf, PING_PERIOD * 2 * 2))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                if (lowLatency) b.setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
                val t = b.build()
                if (t.state == AudioTrack.STATE_INITIALIZED) return t
                t.release()
            } catch (e: Exception) {
                lastError = e
            }
        }
        throw IllegalStateException("speaker unavailable: ${lastError?.message}")
    }

    /**
     * Tries mic configurations in order of preference. A config only counts if it actually
     * delivers non-silent audio: some phones accept a config, then return zeros or errors.
     */
    @SuppressLint("MissingPermission") // RECORD_AUDIO is requested by the Dart side before start().
    private fun openRecorder(): AudioRecord {
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val unprocessed = am.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
        // UNPROCESSED skips noise suppression/AGC, which otherwise smear or erase echoes.
        val sources = buildList {
            if (unprocessed) add(MediaRecorder.AudioSource.UNPROCESSED to "unprocessed")
            add(MediaRecorder.AudioSource.VOICE_RECOGNITION to "voice_recognition")
            add(MediaRecorder.AudioSource.CAMCORDER to "camcorder")
            add(MediaRecorder.AudioSource.MIC to "mic")
        }
        val tried = ArrayList<String>()
        for ((source, name) in sources) {
            for (mask in intArrayOf(AudioFormat.CHANNEL_IN_STEREO, AudioFormat.CHANNEL_IN_MONO)) {
                if (!running) throw IllegalStateException("stopped")
                val label = "$name/${if (mask == AudioFormat.CHANNEL_IN_STEREO) "stereo" else "mono"}"
                micState = "trying $label"
                var rec: AudioRecord? = null
                try {
                    val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, mask, AudioFormat.ENCODING_PCM_16BIT)
                    if (minBuf <= 0) { tried += "$label: unsupported"; continue }
                    rec = AudioRecord.Builder()
                        .setAudioSource(source)
                        .setAudioFormat(AudioFormat.Builder()
                            .setSampleRate(SAMPLE_RATE)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setChannelMask(mask)
                            .build())
                        .setBufferSizeInBytes(max(minBuf, PING_PERIOD * 2 * 2 * 8))
                        .build()
                    if (rec.state != AudioRecord.STATE_INITIALIZED) { tried += "$label: init failed"; rec.release(); continue }
                    rec.startRecording()
                    // Probe ~0.3 s: must read without errors and not be digital silence.
                    val probe = ShortArray(SAMPLE_RATE / 10 * rec.channelCount)
                    var energy = 0.0
                    var ok = true
                    repeat(3) {
                        val n = rec.read(probe, 0, probe.size)
                        if (n <= 0) ok = false else for (i in 0 until n) energy += probe[i].toDouble() * probe[i]
                    }
                    if (!ok || energy == 0.0) {
                        tried += "$label: ${if (!ok) "read error" else "silent"}"
                        rec.stop(); rec.release(); continue
                    }
                    stereo = rec.channelCount == 2
                    sourceName = label
                    micState = "listening"
                    return rec
                } catch (e: Exception) {
                    tried += "$label: ${e.message}"
                    rec?.release()
                }
            }
        }
        throw IllegalStateException("no working microphone (${tried.joinToString("; ")})")
    }

    private fun recordLoop() {
        if (!hasTemplate) loadTemplate()
        val rec = openRecorder()
        val inChannels = if (stereo) 2 else 1
        val chunk = ShortArray(PING_PERIOD * inChannels)
        // De-interleaved rolling buffer with absolute sample indexing.
        val capacity = PING_PERIOD * 8
        val buf = Array(CHANNELS) { FloatArray(capacity) }
        var bufStart = 0L      // absolute index of buf[*][0]
        var bufLen = 0
        var nextDirect = -1L   // absolute index of the next expected direct-path peak
        var weakPings = 0
        var avgDirect = 0f
        val need = SEARCH + RANGE_SAMPLES + CHIRP_LEN
        val warmup = SAMPLE_RATE / 4  // let the speaker/AGC settle before calibrating
        var seen = 0L
        var lastStats = System.currentTimeMillis()
        val scale = 1f / 32768f

        try {
            while (running) {
                val got = rec.read(chunk, 0, chunk.size)
                if (got < 0) throw IllegalStateException("AudioRecord.read error $got")
                if (got == 0) continue
                val frames = got / inChannels
                if (bufLen + frames > capacity) {
                    val drop = bufLen + frames - capacity
                    for (c in 0 until CHANNELS) System.arraycopy(buf[c], drop, buf[c], 0, bufLen - drop)
                    bufStart += drop; bufLen -= drop
                }
                var energy = 0f
                for (f in 0 until frames) {
                    val l = chunk[f * inChannels] * scale
                    buf[0][bufLen + f] = l
                    buf[1][bufLen + f] = if (inChannels == 2) chunk[f * inChannels + 1] * scale else l
                    energy += l * l
                }
                micRms = sqrt(energy / frames)
                bufLen += frames
                seen += frames

                val now = System.currentTimeMillis()
                if (now - lastStats >= 1000) { lastStats = now; reportStats() }
                if (seen < warmup) continue

                if (nextDirect < 0) {
                    // Calibrate: strongest chirp correlation within one full period.
                    if (bufLen < PING_PERIOD + CHIRP_LEN) continue
                    val from = bufLen - PING_PERIOD - CHIRP_LEN
                    var best = 0f; var bestAt = 0
                    for (lag in 0 until PING_PERIOD) {
                        val e = env(buf[0], from + lag)
                        if (e > best) { best = e; bestAt = lag }
                    }
                    nextDirect = bufStart + from + bestAt + PING_PERIOD
                    avgDirect = best
                    weakPings = 0
                    calibrations++
                    // First lock ever on this phone: learn the speaker sound automatically.
                    if (!hasTemplate && calibratingLeft == 0) calibrate()
                    continue
                }

                while (nextDirect - SEARCH >= bufStart && nextDirect - bufStart + need <= bufLen) {
                    val base = (nextDirect - bufStart).toInt()
                    // Track small playback/record drift by re-locking on the direct peak.
                    var best = 0f; var shift = 0
                    for (s in -SEARCH..SEARCH) {
                        val e = env(buf[0], base + s)
                        if (e > best) { best = e; shift = s }
                    }
                    val direct = base + shift
                    lastDirect = best
                    pings++
                    emitPing(buf, direct)
                    avgDirect = 0.95f * avgDirect + 0.05f * best
                    weakPings = if (best < 0.25f * avgDirect) weakPings + 1 else 0
                    nextDirect += shift + PING_PERIOD
                    if (weakPings > 10) { nextDirect = -1; break }
                }
                if (nextDirect >= 0 && nextDirect - SEARCH < bufStart) nextDirect = -1 // fell behind
            }
        } finally {
            rec.stop()
            rec.release()
            micState = "stopped"
        }
    }

    // ── Speaker (direct-path) cancellation ──────────────────────────────────────────────
    // The chirp also reaches the mic straight from the speaker, through air and the phone's
    // body. It is far louder than any echo and its ringing tail buries echoes from the first
    // few feet. It is identical every ping (speaker and mic never move relative to each
    // other), so we average its complex matched-filter response once ("calibration") and
    // subtract it from every ping in the first CANCEL_LEN lags. Echoes are left untouched.
    private val templateI = Array(CHANNELS) { FloatArray(CANCEL_LEN) }
    private val templateQ = Array(CHANNELS) { FloatArray(CANCEL_LEN) }
    @Volatile private var hasTemplate = false
    @Volatile private var calibratingLeft = 0
    @Volatile var calibrationState = "none"; private set
    private val templateFile get() = java.io.File(context.filesDir, "speaker_template.bin")

    /** Learn the speaker->mic sound over the next CALIBRATION_PINGS pings. */
    fun calibrate() {
        for (c in 0 until CHANNELS) { templateI[c].fill(0f); templateQ[c].fill(0f) }
        calibrationState = "calibrating"
        calibratingLeft = CALIBRATION_PINGS
    }

    private fun loadTemplate() {
        val f = templateFile
        if (!f.exists()) return
        try {
            java.io.DataInputStream(f.inputStream().buffered()).use { inp ->
                for (c in 0 until CHANNELS) for (j in 0 until CANCEL_LEN) {
                    templateI[c][j] = inp.readFloat(); templateQ[c][j] = inp.readFloat()
                }
            }
            hasTemplate = true
            calibrationState = "saved"
        } catch (e: Exception) {
            Log.w(TAG, "bad speaker template: $e")
        }
    }

    private fun saveTemplate() {
        java.io.DataOutputStream(templateFile.outputStream().buffered()).use { out ->
            for (c in 0 until CHANNELS) for (j in 0 until CANCEL_LEN) {
                out.writeFloat(templateI[c][j]); out.writeFloat(templateQ[c][j])
            }
        }
    }

    private val corrI = Array(CHANNELS) { FloatArray(RANGE_SAMPLES) }
    private val corrQ = Array(CHANNELS) { FloatArray(RANGE_SAMPLES) }

    private fun emitPing(buf: Array<FloatArray>, direct: Int) {
        val ci = SonarConfig.chirpIQ[0]
        val cq = SonarConfig.chirpIQ[1]
        // Complex matched filter for every lag in range, per mic.
        for (c in 0 until CHANNELS) {
            val x = buf[c]
            for (j in 0 until RANGE_SAMPLES) {
                var i = 0f; var q = 0f
                val at = direct + j
                for (n in 0 until CHIRP_LEN) {
                    val v = x[at + n]
                    i += v * ci[n]; q += v * cq[n]
                }
                corrI[c][j] = i; corrQ[c][j] = q
            }
        }

        if (calibratingLeft > 0) {
            for (c in 0 until CHANNELS) for (j in 0 until CANCEL_LEN) {
                templateI[c][j] += corrI[c][j] / CALIBRATION_PINGS
                templateQ[c][j] += corrQ[c][j] / CALIBRATION_PINGS
            }
            if (--calibratingLeft == 0) {
                hasTemplate = true
                calibrationState = "done"
                try { saveTemplate() } catch (e: Exception) { Log.w(TAG, "save template: $e") }
            }
        }

        val profile = history[historyHead]
        val clean = FloatArray(PROFILE_FLOATS)
        val cancel = hasTemplate && calibratingLeft == 0
        var directLevel = 0f
        for (c in 0 until CHANNELS) {
            val ref = max(hypot(corrI[c][0], corrQ[c][0]), 1e-9f)
            if (c == 0) directLevel = ref
            for (b in 0 until RANGE_BINS) {
                var raw = 0f
                var cleaned = 0f
                for (k in 0 until POOL) {
                    val j = b * POOL + k
                    val i = corrI[c][j]; val q = corrQ[c][j]
                    raw = max(raw, hypot(i, q))
                    cleaned = max(cleaned,
                        if (cancel && j < CANCEL_LEN) hypot(i - templateI[c][j], q - templateQ[c][j]) else hypot(i, q))
                }
                profile[b * CHANNELS + c] = logScale(raw / ref)
                clean[b * CHANNELS + c] = logScale(cleaned / ref)
            }
        }
        historyHead = (historyHead + 1) % STACK
        val stack = FloatArray(STACK_FLOATS)
        for (t in 0 until STACK) {
            System.arraycopy(history[(historyHead + t) % STACK], 0, stack, t * PROFILE_FLOATS, PROFILE_FLOATS)
        }
        onPing(profile.copyOf(), clean, stack, directLevel)
    }

    private fun logScale(v: Float) = min(1f, max(0f, (log10(v + LOG_FLOOR) + 4f) / 4f))

    private fun hypot(a: Float, b: Float) = sqrt(a * a + b * b)

    /** Matched-filter envelope at one lag: |<x[at..at+CHIRP_LEN), chirp_I + j chirp_Q>|. */
    private fun env(x: FloatArray, at: Int): Float {
        val ci = SonarConfig.chirpIQ[0]
        val cq = SonarConfig.chirpIQ[1]
        var i = 0f; var q = 0f
        for (n in 0 until CHIRP_LEN) {
            val v = x[at + n]
            i += v * ci[n]; q += v * cq[n]
        }
        return sqrt(i * i + q * q)
    }

    companion object {
        private const val TAG = "SonarEngine"
        private const val SEARCH = 32
        /** Lags (from the direct peak) covered by speaker cancellation: 252 ≈ 0.9 m / 3 ft. */
        const val CANCEL_LEN = 252
        private const val CALIBRATION_PINGS = 48   // ~2 s
    }
}
