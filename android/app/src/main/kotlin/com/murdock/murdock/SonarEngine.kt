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
 * Only the chirp's loudness is adjustable (slider, AUTO-TUNE): the model input is scaled by
 * the strongest echo, so loudness changes the noise level but not the features' shape.
 * Band, length and timing stay fixed because the echo AI was trained on exactly those.
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
            // Speaker->mic loudness vs. calibration, in dB. Strongly negative = something
            // (usually a finger) is covering the speaker or mic. NaN until calibrated.
            "directDb" to if (calDirect > 0f && directAvg > 0f) (20 * log10(directAvg / (calDirect * cancelGain()))).toDouble() else Double.NaN,
        ))
        pings = 0
    }

    private fun playLoop() {
        // 16-bit PCM: the one output format every Android device supports.
        val ping = ShortArray(PING_PERIOD)
        val chirp = SonarConfig.chirpIQ[0]
        var builtFor = -1f
        val track = openTrack()
        try {
            track.play()
            speakerState = "playing"
            while (running) {
                val l = loudness
                if (l != builtFor) {
                    for (n in 0 until CHIRP_LEN) ping[n] = (chirp[n] * l * Short.MAX_VALUE).toInt().toShort()
                    builtFor = l
                }
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
                    val r = if (inChannels == 2) chunk[f * inChannels + 1] * scale else l
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
    private val templateFile get() = java.io.File(context.filesDir, "speaker_template.bin")
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
        for (c in 0 until CHANNELS) {
            val ref = max(hypot(corrI[c][0], corrQ[c][0]), 1e-9f)
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
        stepTune(pooled)
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
     * AUTO-TUNE: with the phone aimed at a wall 4-6 ft away, tries each loudness in
     * [TUNE_LEVELS], keeps the quietest one within 1 dB of the clearest echo, then relearns
     * the speaker filter at that loudness. Starts on the next ping; ~10 s.
     */
    fun autoTune() { tuneRequest = true }
    fun cancelTune() { tuneCancel = true }

    /** Everything a tune changes, so cancelling or failing puts it all back. */
    private inner class Snapshot {
        private val loud = loudness
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
        }
    }

    private inner class Tune {
        val saved = Snapshot()
        val clarity = DoubleArray(TUNE_LEVELS.size) { Double.NaN }
        val clipped = BooleanArray(TUNE_LEVELS.size)
        val sum = FloatArray(RANGE_BINS)
        var level = 0
        var phase = SETTLE
        var count = 0
        var best = -1
        var done = 0              // pings so far, for the progress bar

        fun report(state: String, message: String = "") {
            onTune?.invoke(mapOf(
                "type" to "tune",
                "state" to state,
                "phase" to phase,
                "progress" to (done.toDouble() / TUNE_TOTAL_PINGS).coerceIn(0.0, 1.0),
                "levels" to TUNE_LEVELS.map { it.toDouble() },
                "clarity" to clarity.toList(),
                "clipped" to clipped.toList(),
                "testing" to loudness.toDouble(),
                "best" to if (best >= 0) TUNE_LEVELS[best].toDouble() else Double.NaN,
                "bestDb" to if (best >= 0) clarity[best] else Double.NaN,
                "message" to message,
            ))
        }
    }

    /** Advances AUTO-TUNE by one ping. Runs on the recorder thread. */
    private fun stepTune(pooled: FloatArray) {
        if (tuneCancel) {
            tuneCancel = false
            tune?.let { it.saved.restore(); tune = null; it.report("cancelled") }
        }
        if (tuneRequest) {
            tuneRequest = false
            if (tune == null) tune = Tune().also { loudness = TUNE_LEVELS[0]; it.report("running") }
        }
        val t = tune ?: return
        t.done++
        when (t.phase) {
            SETTLE -> if (++t.count >= TUNE_SETTLE_PINGS) {
                // Audio already queued at the previous loudness has played out by now.
                t.phase = MEASURE; t.count = 0; t.sum.fill(0f); micPeak = 0f
                t.report("running")
            }
            MEASURE -> {
                for (b in 0 until RANGE_BINS) t.sum[b] += 0.5f * (pooled[b * CHANNELS] + pooled[b * CHANNELS + 1])
                if (++t.count >= TUNE_MEASURE_PINGS) {
                    t.clarity[t.level] = clarityDb(t.sum)
                    t.clipped[t.level] = micPeak >= CLIP_LEVEL
                    if (++t.level < TUNE_LEVELS.size) {
                        loudness = TUNE_LEVELS[t.level]
                        t.phase = SETTLE; t.count = 0
                        t.report("running")
                    } else pickBest(t)
                } else if (t.count % 8 == 0) t.report("running")
            }
            FINAL_SETTLE -> if (++t.count >= TUNE_SETTLE_PINGS) {
                // Relearn the speaker's own sound at the chosen loudness. The wall is beyond
                // the 3 ft speaker zone, so it isn't learned as part of it.
                calibrate(); t.phase = FINAL_CALIBRATE
                t.report("running")
            }
            FINAL_CALIBRATE -> if (calibratingLeft == 0) {
                persistTemplate()
                tune = null
                t.done = TUNE_TOTAL_PINGS
                t.report("done")
            } else if (t.done % 8 == 0) t.report("running")
        }
    }

    private fun pickBest(t: Tune) {
        val ok = TUNE_LEVELS.indices.filter { !t.clipped[it] && !t.clarity[it].isNaN() }
        val top = ok.maxOfOrNull { t.clarity[it] } ?: Double.NaN
        if (ok.isEmpty() || top < TUNE_MIN_DB) {
            t.saved.restore()
            tune = null
            t.report("failed", if (ok.isEmpty() && t.clipped.any { it })
                "The mic is overloaded at every loudness. Turn the media volume down a little and try again."
            else "No clear echo. Aim the top edge at a flat wall 4-6 ft away, keep still, and try a quieter room.")
            return
        }
        // Quietest level within 1 dB of the clearest: the same picture, with less distortion
        // and battery.
        t.best = ok.filter { t.clarity[it] >= top - 1.0 }.minBy { TUNE_LEVELS[it] }
        loudness = TUNE_LEVELS[t.best]
        t.phase = FINAL_SETTLE; t.count = 0
        t.report("running")
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
        private val TUNE_TOTAL_PINGS =
            TUNE_LEVELS.size * (TUNE_SETTLE_PINGS + TUNE_MEASURE_PINGS) + TUNE_SETTLE_PINGS + CALIBRATION_PINGS

        // AUTO-TUNE phases, also sent to the UI.
        private const val SETTLE = "settle"
        private const val MEASURE = "measure"
        private const val FINAL_SETTLE = "final_settle"
        private const val FINAL_CALIBRATE = "calibrate"
    }
}
