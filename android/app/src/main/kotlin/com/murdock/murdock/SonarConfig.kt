package com.murdock.murdock

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** Shared sonar/model constants. Keep in sync with ml/murdock_model.py and ml/features.py. */
object SonarConfig {
    const val SAMPLE_RATE = 48000
    const val PING_PERIOD = 2048        // samples between chirps (~23 pings/s)
    const val RANGE_SAMPLES = 1536      // samples kept after the direct path (~5.5 m one way)
    const val RANGE_BINS = 128
    const val POOL = RANGE_SAMPLES / RANGE_BINS
    const val STACK = 16
    const val CHANNELS = 2
    const val DEPTH_SIZE = 32

    const val CHIRP_F0 = 17500.0
    const val CHIRP_F1 = 21500.0
    const val CHIRP_LEN = 240           // 5 ms
    const val LOG_FLOOR = 1e-4f
    const val ECHO_ZONE_BINS = 4        // first ~7 in of the model input: masked (speaker zone)
    const val CHIRP_AMPLITUDE = 0.6f    // default loudness; the user/auto-tune can change it
    const val MIN_LOUDNESS = 0.2f
    const val MAX_LOUDNESS = 1.0f

    const val PROFILE_FLOATS = RANGE_BINS * CHANNELS
    const val STACK_FLOATS = STACK * PROFILE_FLOATS
    const val DEPTH_FLOATS = DEPTH_SIZE * DEPTH_SIZE

    /** Hann-windowed linear chirp, in-phase (index 0) and quadrature (index 1) references. */
    val chirpIQ: Array<FloatArray> by lazy {
        val i = FloatArray(CHIRP_LEN)
        val q = FloatArray(CHIRP_LEN)
        val dur = CHIRP_LEN.toDouble() / SAMPLE_RATE
        for (n in 0 until CHIRP_LEN) {
            val t = n.toDouble() / SAMPLE_RATE
            val phase = 2 * PI * (CHIRP_F0 * t + (CHIRP_F1 - CHIRP_F0) * t * t / (2 * dur))
            val w = 0.5 - 0.5 * cos(2 * PI * n / (CHIRP_LEN - 1))
            i[n] = (w * cos(phase)).toFloat()
            q[n] = (w * sin(phase)).toFloat()
        }
        arrayOf(i, q)
    }
}
