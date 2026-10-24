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
import com.murdock.murdock.SonarConfig.ECHO_ZONE_BINS
import com.murdock.murdock.SonarConfig.LOG_FLOOR
import com.murdock.murdock.SonarConfig.MAX_LOUDNESS
import com.murdock.murdock.SonarConfig.MIN_LOUDNESS
import com.murdock.murdock.SonarConfig.PING_PERIOD
import com.murdock.murdock.SonarConfig.POOL
import com.murdock.murdock.SonarConfig.PROFILE_FLOATS
import com.murdock.murdock.SonarConfig.RANGE_BINS
import com.murdock.murdock.SonarConfig.RANGE_SAMPLES
import com.murdock.murdock.SonarConfig.SAMPLE_RATE
import com.murdock.murdock.SonarConfig.STACK
import com.murdock.murdock.SonarConfig.STACK_FLOATS
import kotlin.math.abs
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
 *
 * Adjustable without changing the model input's shape: the chirp's loudness (slider,
 * AUTO-TUNE), which speaker plays it on two-speaker phones, and which two microphones are
 * listened to when Android offers more than two (AUTO-TUNE measures each and keeps the
 * clearest). Band, length and timing stay fixed because the echo AI was trained on those.
 */
class SonarEngine(
    private val context: Context,
    /** profile/clean: display profiles with/without the speaker's sound; stack: model input. */
    private val onPing: (profile: FloatArray, clean: FloatArray, stack: FloatArray, directLevel: Float) -> Unit,
    private val onError: (String) -> Unit,
    private val onStats: (Map<String, Any>) -> Unit,
) {
    @Volatile private var running = false
    private var playThread: Thread? = null
    private var recordThread: Thread? = null

    @Volatile var stereo = false; private set

    /** Chirp loudness, as a fraction of full scale. Applied live; cancellation follows it. */
    @Volatile var loudness = SonarConfig.CHIRP_AMPLITUDE
        set(v) { field = v.coerceIn(MIN_LOUDNESS, MAX_LOUDNESS) }

    /**
     * Which speaker plays the chirp: EMIT_BOTH, or one stereo channel alone (EMIT_LEFT /
     * EMIT_RIGHT; two-speaker phones send each to its own speaker). Only takes effect when
     * the output opened in stereo ([stereoOut]).
     */
    @Volatile var emitter = EMIT_BOTH; private set
    /** The speaker output is stereo, so each speaker can be driven on its own. */
    @Volatile var stereoOut = false; private set
    /** Mic channels Android delivers, and the two the echo AI listens to. */
    @Volatile var micChannels = 1; private set
    @Volatile var micA = 0; private set
    @Volatile var micB = 1; private set
    /** AUTO-TUNE's finding about the speakers: untested, separate, or shared (one sound). */
    @Volatile var speakerLayout = "untested"

    /** Picks the speaker and mic pair. The speaker filter is per setup: saved, or relearned. */
    fun setSetup(emit: Int, a: Int, b: Int) {
        if (emit == emitter && a == micA && b == micB) return
        emitter = emit; micA = a; micB = b
        if (tune != null) return   // AUTO-TUNE switches setups itself and relearns at the end
        hasTemplate = false
        loadTemplate()
        if (!hasTemplate && running) calibrate()
    }

    /** True while the speaker filter is being learned or AUTO-TUNE runs: echoes aren't clean. */
    val busy get() = calibratingLeft > 0 || tune != null
    @Volatile var sourceName = ""; private set

    // Live diagnostics, reported once a second through onStats.
    @Volatile private var speakerState = "starting"
    @Volatile private var micState = "starting"
    @Volatile private var micRms = 0f
    @Volatile private var pings = 0
    @Volatile private var calibrations = 0
    @Volatile private var lastDirect = 0f
    @Volatile private var directAvg = 0f          // smoothed speaker->mic level (for "covered?")

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
        // A tune can't finish without pings: put the old settings back.
        tuneRequest = false
        tune?.let { it.saved.restore(); tune = null; it.report("cancelled") }
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
            "loudness" to loudness.toDouble(),
            // How far the strongest echo stands out of the noise, in dB (NaN until heard).
            "clarityDb" to clarityDb(clarityProfile),
            "tuning" to (tune != null),
            "emitter" to emitter,
            "stereoOut" to stereoOut,
            "micChannels" to micChannels,
            "micA" to micA,
            "micB" to micB,
            "speakers" to speakerLayout,
            // Speaker->mic loudness vs. calibration, in dB. Strongly negative = something
            // (usually a finger) is covering the speaker or mic. NaN until calibrated.
            "directDb" to if (calDirect > 0f && directAvg > 0f) (20 * log10(directAvg / (calDirect * cancelGain()))).toDouble() else Double.NaN,
        ))
        pings = 0
    }

    private fun playLoop() {
        // 16-bit PCM: the one output format every Android device supports.
        val chirp = SonarConfig.chirpIQ[0]
        var builtFor = -1f
        var builtEmit = -1
        val track = openTrack()
        val outCh = track.channelCount
        stereoOut = outCh == 2
        if (!stereoOut && emitter != EMIT_BOTH) setSetup(EMIT_BOTH, micA, micB)
        // Interleaved when stereo: left and right can each be silenced to use one speaker.
        val ping = ShortArray(PING_PERIOD * outCh)
        try {
            track.play()
            speakerState = "playing"
            while (running) {
                val l = loudness
                val e = if (stereoOut) emitter else EMIT_BOTH
                if (l != builtFor || e != builtEmit) {
                    for (n in 0 until CHIRP_LEN) {
                        val v = (chirp[n] * l * Short.MAX_VALUE).toInt().toShort()
                        if (outCh == 1) ping[n] = v else {
                            ping[2 * n] = if (e == EMIT_RIGHT) 0 else v
                            ping[2 * n + 1] = if (e == EMIT_LEFT) 0 else v
                        }
                    }
                    builtFor = l; builtEmit = e
                }
                val n = track.write(ping, 0, ping.size)
                if (n < 0) throw IllegalStateException("AudioTrack.write error $n")
            }
        } finally {
            track.stop()
            track.release()
        }
    }

    /** Stereo first (each speaker addressable), else mono. */
    private fun openTrack(): AudioTrack {
        var lastError: Exception? = null
        for (mask in intArrayOf(AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.CHANNEL_OUT_MONO))
        for (lowLatency in booleanArrayOf(true, false)) {
            try {
                val minBuf = AudioTrack.getMinBufferSize(SAMPLE_RATE, mask, AudioFormat.ENCODING_PCM_16BIT)
                if (minBuf <= 0) continue
                val ch = if (mask == AudioFormat.CHANNEL_OUT_STEREO) 2 else 1
                val b = AudioTrack.Builder()
                    .setAudioAttributes(AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build())
                    .setAudioFormat(AudioFormat.Builder()
                        .setSampleRate(SAMPLE_RATE)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(mask)
                        .build())
                    .setBufferSizeInBytes(max(minBuf, PING_PERIOD * 2 * 2 * ch))
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
        // More than two mics: only reachable with a channel *index* mask (and only if the
        // phone allows it). Then plain stereo and mono.
        val layouts = listOf(4 to "4ch", 3 to "3ch", 2 to "stereo", 1 to "mono")
        for ((source, name) in sources) {
            for ((count, layoutName) in layouts) {
                if (!running) throw IllegalStateException("stopped")
                val label = "$name/$layoutName"
                micState = "trying $label"
                var rec: AudioRecord? = null
                try {
                    val format = AudioFormat.Builder().setSampleRate(SAMPLE_RATE).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    when (count) {
                        1 -> format.setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        2 -> format.setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
                        else -> format.setChannelIndexMask((1 shl count) - 1)
                    }
                    val built = format.build()
                    val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                        if (count == 1) AudioFormat.CHANNEL_IN_MONO else AudioFormat.CHANNEL_IN_STEREO, AudioFormat.ENCODING_PCM_16BIT)
                    if (minBuf <= 0) { tried += "$label: unsupported"; continue }
                    rec = AudioRecord.Builder()
                        .setAudioSource(source)
                        .setAudioFormat(built)
                        .setBufferSizeInBytes(max(minBuf, PING_PERIOD * 2 * count * 8))
                        .build()
                    if (rec.state != AudioRecord.STATE_INITIALIZED) { tried += "$label: init failed"; rec.release(); continue }
                    if (rec.channelCount != count) { tried += "$label: got ${rec.channelCount} channels"; rec.release(); continue }
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
                    // Extra channels must be real mics, not copies of one another.
                    if (count >= 3 && !distinctChannels(probe, count)) {
                        tried += "$label: duplicate channels"
                        rec.stop(); rec.release(); continue
                    }
                    micChannels = count
                    stereo = count >= 2
                    if (micA >= count || micB >= count) setSetup(emitter, 0, min(1, count - 1))
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

    private fun distinctChannels(x: ShortArray, ch: Int): Boolean {
        val frames = x.size / ch
        for (a in 0 until ch) for (b in a + 1 until ch) {
            var diff = 0L
            for (f in 0 until frames) diff += abs(x[f * ch + a] - x[f * ch + b])
            if (diff == 0L) return false
        }
        for (a in 0 until ch) if ((0 until frames).all { x[it * ch + a].toInt() == 0 }) return false
        return true
    }

    private fun recordLoop() {
        if (!hasTemplate) loadTemplate()
        val rec = openRecorder()
        val inChannels = rec.channelCount
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
                // The two mics the echo AI listens to (AUTO-TUNE may pick another pair).
                val a = micA.coerceIn(0, inChannels - 1)
                val b = micB.coerceIn(0, inChannels - 1)
                for (f in 0 until frames) {
                    val l = chunk[f * inChannels + a] * scale
                    val r = chunk[f * inChannels + b] * scale
                    buf[0][bufLen + f] = l
                    buf[1][bufLen + f] = r
                    energy += l * l
                    micPeak = max(micPeak, max(abs(l), abs(r)))
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
    // One speaker filter per speaker/mic setup (the default setup keeps the original name).
    private val templateFile get() = java.io.File(context.filesDir,
        if (emitter == EMIT_BOTH && micA == 0 && micB == 1) "speaker_template.bin"
        else "speaker_template_e${emitter}_m$micA$micB.bin")
    @Volatile private var calDirect = 0f          // speaker->mic level while calibrating
    @Volatile private var calLoudness = SonarConfig.CHIRP_AMPLITUDE  // loudness the template was learned at
    private var calDirectSum = 0f

    /** Learn the speaker->mic sound over the next CALIBRATION_PINGS pings. */
    fun calibrate() {
        for (c in 0 until CHANNELS) { templateI[c].fill(0f); templateQ[c].fill(0f) }
        calDirectSum = 0f
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
            return
        }
        // Appended in later versions; older files simply don't have them (and were all
        // recorded at the default loudness).
        try {
            java.io.DataInputStream(f.inputStream().buffered()).use { inp ->
                inp.skipBytes(CHANNELS * CANCEL_LEN * 8)
                calDirect = inp.readFloat()
                calLoudness = inp.readFloat()
            }
        } catch (e: Exception) { /* keep defaults */ }
        if (!(calLoudness > 0f)) calLoudness = SonarConfig.CHIRP_AMPLITUDE
    }

    private fun saveTemplate() {
        java.io.DataOutputStream(templateFile.outputStream().buffered()).use { out ->
            for (c in 0 until CHANNELS) for (j in 0 until CANCEL_LEN) {
                out.writeFloat(templateI[c][j]); out.writeFloat(templateQ[c][j])
            }
            out.writeFloat(calDirect)
            out.writeFloat(calLoudness)
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
            calDirectSum += hypot(corrI[0][0], corrQ[0][0]) / CALIBRATION_PINGS
            if (--calibratingLeft == 0) {
                calDirect = calDirectSum
                calLoudness = loudness
                hasTemplate = true
                calibrationState = "done"
                // AUTO-TUNE saves only the template it keeps.
                if (tune == null) persistTemplate()
            }
        }

        val raw = FloatArray(PROFILE_FLOATS)       // display: speaker sound included
        val clean = FloatArray(PROFILE_FLOATS)     // display: speaker sound cancelled
        val pooled = FloatArray(PROFILE_FLOATS)    // model: cancelled magnitudes, pooled
        val cancel = hasTemplate && calibratingLeft == 0
        // The speaker's sound scales with loudness; so does the template learned for it.
        val gain = cancelGain()
        var directLevel = 0f
        val refs = FloatArray(CHANNELS)
        for (c in 0 until CHANNELS) {
            val ref = max(hypot(corrI[c][0], corrQ[c][0]), 1e-9f)
            refs[c] = ref
            if (c == 0) {
                directLevel = ref
                directAvg = if (directAvg == 0f) ref else 0.9f * directAvg + 0.1f * ref
            }
            for (b in 0 until RANGE_BINS) {
                var r = 0f
                var cleaned = 0f
                for (k in 0 until POOL) {
                    val j = b * POOL + k
                    val i = corrI[c][j]; val q = corrQ[c][j]
                    r = max(r, hypot(i, q))
                    cleaned = max(cleaned,
                        if (cancel && j < CANCEL_LEN) hypot(i - gain * templateI[c][j], q - gain * templateQ[c][j]) else hypot(i, q))
                }
                raw[b * CHANNELS + c] = logScale(r / ref)
                clean[b * CHANNELS + c] = logScale(cleaned / ref)
                pooled[b * CHANNELS + c] = cleaned
            }
        }

        // Model input (mirrors ml/features.py echo_features): speaker zone masked, both mics
        // scaled by the strongest echo beyond it, so phone and BatVision data share one scale.
        var echoRef = 1e-9f
        for (i in ECHO_ZONE_BINS * CHANNELS until PROFILE_FLOATS) echoRef = max(echoRef, pooled[i])
        val features = history[historyHead]
        for (i in 0 until PROFILE_FLOATS) {
            features[i] = if (i < ECHO_ZONE_BINS * CHANNELS) 0f else logScale(pooled[i] / echoRef)
        }
        historyHead = (historyHead + 1) % STACK
        val stack = FloatArray(STACK_FLOATS)
        for (t in 0 until STACK) {
            System.arraycopy(history[(historyHead + t) % STACK], 0, stack, t * PROFILE_FLOATS, PROFILE_FLOATS)
        }
        for (b in 0 until RANGE_BINS) {
            val v = 0.5f * (pooled[b * CHANNELS] + pooled[b * CHANNELS + 1])
            clarityProfile[b] = if (clarityProfile[b] == 0f) v else 0.9f * clarityProfile[b] + 0.1f * v
        }
        stepTune(pooled, refs)
        onPing(raw, clean, stack, directLevel)
    }

    private fun cancelGain() = if (calLoudness > 0f) loudness / calLoudness else 1f

    private fun persistTemplate() {
        try { saveTemplate() } catch (e: Exception) { Log.w(TAG, "save template: $e") }
    }

    // ── Echo clarity + AUTO-TUNE ────────────────────────────────────────────────────────
    // Clarity: how far the strongest echo beyond the speaker zone stands out above the
    // typical (median) range bin of a time-averaged profile. Averaging over pings keeps
    // steady echoes and flattens random noise, so pure noise scores only a few dB and a
    // wall 4-6 ft away scores well above that.
    private val clarityProfile = FloatArray(RANGE_BINS)
    private var micPeak = 0f   // loudest mic sample since the tuner last reset it (clipping)

    private fun clarityDb(profile: FloatArray): Double {
        val far = profile.copyOfRange(CANCEL_BINS, RANGE_BINS)
        val peak = far.max()
        far.sort()
        val median = far[far.size / 2]
        if (!(median > 0f) || !(peak > 0f)) return Double.NaN
        return 20 * log10((peak / median).toDouble())
    }

    /** Receives AUTO-TUNE progress and results (on the recorder thread). */
    @Volatile var onTune: ((Map<String, Any>) -> Unit)? = null
    @Volatile private var tune: Tune? = null
    @Volatile private var tuneRequest = false
    @Volatile private var tuneCancel = false

    /**
     * AUTO-TUNE, with the phone aimed at a wall 4-6 ft away, in up to three stages:
     *  1. each loudness in [TUNE_LEVELS] (both speakers): keeps the quietest within 1 dB of
     *     the clearest echo;
     *  2. two-speaker phones: each speaker alone, which also shows whether Android really
     *     drives them separately ([speakerLayout]); one is kept if it is clearly (1 dB) better;
     *  3. more than two mics: every other mic pair, kept if clearly better.
     * Then it relearns the speaker filter for the chosen setup. Starts on the next ping.
     */
    fun autoTune() { tuneRequest = true }
    fun cancelTune() { tuneCancel = true }

    /** Everything a tune changes, so cancelling or failing puts it all back. */
    private inner class Snapshot {
        private val loud = loudness
        private val emit = emitter
        private val a = micA
        private val b = micB
        private val ti = Array(CHANNELS) { templateI[it].copyOf() }
        private val tq = Array(CHANNELS) { templateQ[it].copyOf() }
        private val direct = calDirect
        private val calLoud = calLoudness
        private val had = hasTemplate
        private val state = calibrationState

        fun restore() {
            calibratingLeft = 0
            for (c in 0 until CHANNELS) { ti[c].copyInto(templateI[c]); tq[c].copyInto(templateQ[c]) }
            calDirect = direct; calLoudness = calLoud; hasTemplate = had; calibrationState = state
            loudness = loud
            emitter = emit; micA = a; micB = b
        }
    }

    /** One setup AUTO-TUNE measures. */
    private class Trial(val loud: Float, val emit: Int, val a: Int, val b: Int, val label: String) {
        var clarity = Double.NaN
        var clipped = false
        val direct = FloatArray(CHANNELS)   // speaker->mic level per mic, averaged
    }

    private inner class Tune {
        val saved = Snapshot()
        val trials = ArrayList<Trial>()
        val queue = ArrayDeque<Trial>()
        var stage = STAGE_LOUDNESS
        lateinit var current: Trial
        val sum = FloatArray(RANGE_BINS)
        var phase = SETTLE
        var count = 0
        var best: Trial? = null
        var layout = speakerLayout
        var done = 0              // pings so far, for the progress bar
        val total = (TUNE_LEVELS.size + (if (stereoOut) 2 else 0) + micChannels * (micChannels - 1) / 2 - 1) *
            (TUNE_SETTLE_PINGS + TUNE_MEASURE_PINGS) + TUNE_SETTLE_PINGS + CALIBRATION_PINGS

        fun start(t: Trial) {
            current = t
            trials += t
            loudness = t.loud; emitter = t.emit; micA = t.a; micB = t.b
            phase = SETTLE; count = 0
        }

        fun report(state: String, message: String = "") {
            val b = best
            onTune?.invoke(mapOf(
                "type" to "tune",
                "state" to state,
                "phase" to phase,
                "progress" to (done.toDouble() / total).coerceIn(0.0, 1.0),
                "levels" to trials.map { it.loud.toDouble() },
                "labels" to trials.map { it.label },
                "clarity" to trials.map { it.clarity },
                "clipped" to trials.map { it.clipped },
                "testing" to loudness.toDouble(),
                "testingLabel" to if (phase == SETTLE || phase == MEASURE) current.label else "",
                "best" to (b?.loud?.toDouble() ?: Double.NaN),
                "bestDb" to (b?.clarity ?: Double.NaN),
                "bestLabel" to (b?.let { setupLabel(it) } ?: ""),
                "speakers" to layout,
                "micChannels" to micChannels,
                "message" to message,
            ))
        }
    }

    private fun pct(v: Float) = "${(v * 100).toInt()}%"

    private fun setupLabel(t: Trial) = buildList {
        add(pct(t.loud))
        add(when (t.emit) { EMIT_LEFT -> "SPEAKER 1"; EMIT_RIGHT -> "SPEAKER 2"; else -> "BOTH SPEAKERS" })
        if (micChannels > 2) add("MICS ${t.a + 1}+${t.b + 1}")
    }.joinToString(" · ")

    /** Advances AUTO-TUNE by one ping. Runs on the recorder thread. */
    private fun stepTune(pooled: FloatArray, refs: FloatArray) {
        if (tuneCancel) {
            tuneCancel = false
            tune?.let { it.saved.restore(); tune = null; it.report("cancelled") }
        }
        if (tuneRequest) {
            tuneRequest = false
            if (tune == null) {
                val pair = if (micChannels > 1) 1 else 0
                tune = Tune().also { t ->
                    for (l in TUNE_LEVELS) t.queue += Trial(l, EMIT_BOTH, 0, pair, pct(l))
                    t.start(t.queue.removeFirst())
                    t.report("running")
                }
            }
        }
        val t = tune ?: return
        t.done++
        when (t.phase) {
            SETTLE -> if (++t.count >= TUNE_SETTLE_PINGS) {
                // Audio already queued with the previous setup has played out by now.
                t.phase = MEASURE; t.count = 0; t.sum.fill(0f); micPeak = 0f
                t.report("running")
            }
            MEASURE -> {
                for (b in 0 until RANGE_BINS) t.sum[b] += 0.5f * (pooled[b * CHANNELS] + pooled[b * CHANNELS + 1])
                for (c in 0 until CHANNELS) t.current.direct[c] += refs[c] / TUNE_MEASURE_PINGS
                if (++t.count >= TUNE_MEASURE_PINGS) {
                    t.current.clarity = clarityDb(t.sum)
                    t.current.clipped = micPeak >= CLIP_LEVEL
                    val next = t.queue.removeFirstOrNull() ?: nextStage(t)
                    when {
                        tune == null -> return     // failed
                        next != null -> { t.start(next); t.report("running") }
                        else -> {
                            // Settle on the chosen setup, then relearn its speaker sound.
                            val b = t.best!!
                            loudness = b.loud; emitter = b.emit; micA = b.a; micB = b.b
                            t.phase = FINAL_SETTLE; t.count = 0
                            t.report("running")
                        }
                    }
                } else if (t.count % 8 == 0) t.report("running")
            }
            FINAL_SETTLE -> if (++t.count >= TUNE_SETTLE_PINGS) {
                // The wall is beyond the 3 ft speaker zone, so it isn't learned as part of it.
                calibrate(); t.phase = FINAL_CALIBRATE
                t.report("running")
            }
            FINAL_CALIBRATE -> if (calibratingLeft == 0) {
                persistTemplate()
                speakerLayout = t.layout
                tune = null
                t.done = t.total
                t.report("done")
            } else if (t.done % 8 == 0) t.report("running")
        }
    }

    private fun usable(t: Trial) = !t.clipped && !t.clarity.isNaN()

    /** Clearly (1 dB) better than [base], or null. */
    private fun clearlyBetter(base: Trial, others: List<Trial>) =
        others.filter { usable(it) && it.clarity >= base.clarity + 1.0 }.maxByOrNull { it.clarity }

    /** Closes the stage just measured and queues the next; null when there is nothing left. */
    private fun nextStage(t: Tune): Trial? {
        while (true) {
            when (t.stage) {
                STAGE_LOUDNESS -> {
                    val ok = t.trials.filter(::usable)
                    val top = ok.maxOfOrNull { it.clarity } ?: Double.NaN
                    if (ok.isEmpty() || top < TUNE_MIN_DB) {
                        t.saved.restore()
                        tune = null
                        t.report("failed", if (ok.isEmpty() && t.trials.any { it.clipped })
                            "The mic is overloaded at every loudness. Turn the media volume down a little and try again."
                        else "No clear echo. Aim the top edge at a flat wall 4-6 ft away, keep still, and try a quieter room.")
                        return null
                    }
                    // Quietest within 1 dB of the clearest: same picture, less distortion and battery.
                    val b = ok.filter { it.clarity >= top - 1.0 }.minBy { it.loud }
                    t.best = b
                    t.stage = STAGE_SPEAKERS
                    if (stereoOut) {
                        t.queue += Trial(b.loud, EMIT_LEFT, b.a, b.b, "SPK 1")
                        t.queue += Trial(b.loud, EMIT_RIGHT, b.a, b.b, "SPK 2")
                    }
                }
                STAGE_SPEAKERS -> {
                    val both = t.best!!
                    val one = t.trials.filter { it.emit != EMIT_BOTH }
                    if (one.size == 2) {
                        t.layout = speakerLayoutOf(one[0], one[1])
                        // Single speakers only count if Android really drives them separately.
                        if (t.layout == "separate") clearlyBetter(both, one)?.let { t.best = it }
                    }
                    t.stage = STAGE_MICS
                    val b = t.best!!
                    for (i in 0 until micChannels) for (j in i + 1 until micChannels) {
                        if (micChannels > 2 && !(i == b.a && j == b.b)) t.queue += Trial(b.loud, b.emit, i, j, "MIC ${i + 1}+${j + 1}")
                    }
                }
                STAGE_MICS -> {
                    val b = t.best!!
                    clearlyBetter(b, t.trials.filter { it.a != b.a || it.b != b.b })?.let { t.best = it }
                    t.stage = STAGE_DONE
                }
                else -> return null
            }
            t.queue.removeFirstOrNull()?.let { return it }
        }
    }

    /**
     * Did left-only and right-only come from two different speakers? Then the mics hear them
     * at clearly different levels (the speakers sit at opposite ends of the phone). If
     * Android mixes both channels into one speaker, the two sound the same.
     */
    private fun speakerLayoutOf(l: Trial, r: Trial): String {
        fun db(x: Float, y: Float) = 20 * log10(max(x, 1e-9f) / max(y, 1e-9f))
        val levelGap = abs(db(l.direct.max(), r.direct.max()))
        val micGap = if (micChannels > 1) abs(db(l.direct[0], l.direct[1]) - db(r.direct[0], r.direct[1])) else 0f
        return if (levelGap > 3f || micGap > 3f) "separate" else "shared"
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
        private const val CANCEL_BINS = (CANCEL_LEN + POOL - 1) / POOL   // speaker zone, in range bins

        /** Loudness levels AUTO-TUNE tries, quietest first. */
        val TUNE_LEVELS = floatArrayOf(0.3f, 0.45f, 0.6f, 0.8f, 1.0f)
        private const val TUNE_SETTLE_PINGS = 10     // ~0.4 s for queued audio to play out
        private const val TUNE_MEASURE_PINGS = 24    // ~1 s of echoes per level
        private const val TUNE_MIN_DB = 4.0          // below this, nothing was clearly heard
        private const val CLIP_LEVEL = 0.98f         // mic samples this loud are clipping
        // Which speaker plays the chirp.
        const val EMIT_BOTH = 0
        const val EMIT_LEFT = 1
        const val EMIT_RIGHT = 2

        // AUTO-TUNE stages.
        private const val STAGE_LOUDNESS = 0
        private const val STAGE_SPEAKERS = 1
        private const val STAGE_MICS = 2
        private const val STAGE_DONE = 3

        // AUTO-TUNE phases, also sent to the UI.
        private const val SETTLE = "settle"
        private const val MEASURE = "measure"
        private const val FINAL_SETTLE = "final_settle"
        private const val FINAL_CALIBRATE = "calibrate"
    }
}
