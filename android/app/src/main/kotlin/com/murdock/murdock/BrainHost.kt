package com.murdock.murdock

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException

/**
 * Process-wide owner of the echo AI and its continuous trainer.
 *
 * It lives as long as the app's process, not the screen, so the background
 * [LearningService] can keep training (and saving) after the user closes the app. The
 * activity and the service both use it; the brain is never closed by either.
 *
 * Everything that touches [brain] runs on [thread]: the interpreter is not thread-safe.
 */
object BrainHost {
    private const val TAG = "BrainHost"
    private const val TRAIN_BURST_MS = 30L     // live: short bursts, gaps for inference
    private const val FINISH_BURST_MS = 250L   // FINISH LEARNING: nothing else to share with
    private const val MODEL_ASSET = "flutter_assets/assets/models/echo_student.tflite"

    private val main = Handler(Looper.getMainLooper())
    val thread = Executors.newSingleThreadExecutor()

    @Volatile var brain: EchoBrain? = null; private set
    /** "loading", "ready" or "failed: ...". */
    @Volatile var state = "loading"; private set

    /** FINISH LEARNING: train flat out until the queue is empty, then save. */
    @Volatile var finishing = false
    @Volatile var stepsPerSec = 0.0; private set

    /** Error reporter (the activity, while it's alive). */
    @Volatile var onError: ((String) -> Unit)? = null
    /** Called on the main thread when FINISH LEARNING has emptied the queue and saved. */
    @Volatile var onFinished: (() -> Unit)? = null

    private var loadStarted = false
    private var trainerOn = false
    private var rateSteps = 0
    private var rateFrom = System.nanoTime()

    /** Loads the brain once per process (from the activity or the service, whichever is first). */
    @Synchronized
    fun ensureLoaded(context: Context) {
        if (loadStarted) return
        loadStarted = true
        val app = context.applicationContext
        submit("load model") {
            try {
                val bytes = app.assets.open(MODEL_ASSET).use { it.readBytes() }
                val buf = ByteBuffer.allocateDirect(bytes.size).put(bytes)
                brain = EchoBrain(buf, File(app.filesDir, "brain/head.weights"), samplesDir(app))
                state = "ready"
            } catch (t: Throwable) {
                state = "failed: ${t.message ?: t.javaClass.simpleName}"
                throw t
            }
        }
        startTrainer()
    }

    fun samplesDir(context: Context) = File(context.getExternalFilesDir(null), "samples/v2") // v2: echo_features input

    /** Runs [task] on the brain thread; failures are logged and reported, never crash the app. */
    fun submit(what: String, onFail: ((Throwable) -> Unit)? = null, task: () -> Unit): Boolean = try {
        thread.execute {
            try {
                task()
            } catch (t: Throwable) {
                Log.e(TAG, "$what failed", t)
                onError?.invoke("$what failed: ${t.message ?: t.javaClass.simpleName}")
                onFail?.invoke(t)
            }
        }
        true
    } catch (e: RejectedExecutionException) {
        false
    }

    // ── Continuous trainer ──────────────────────────────────────────────────────────────
    // Spends the brain's step budget in short bursts, leaving gaps so live echo inference
    // (same thread) stays smooth; while finishing, in long bursts with no gaps.
    @Synchronized
    private fun startTrainer() {
        if (trainerOn) return
        trainerOn = true
        tick()
    }

    private fun tick() {
        submit("training") {
            val b = brain
            val steps = b?.trainFor(if (finishing) FINISH_BURST_MS else TRAIN_BURST_MS) ?: 0
            rateSteps += steps
            val now = System.nanoTime()
            if (now - rateFrom > 1_000_000_000L) {
                stepsPerSec = rateSteps * 1e9 / (now - rateFrom)
                rateSteps = 0; rateFrom = now
            }
            if (finishing && b != null && b.pendingSteps == 0) {
                b.save()
                finishing = false
                main.post { onFinished?.invoke() }
            }
            // Busy -> come back soon (immediately when finishing); idle -> check in a while.
            main.postDelayed(::tick, if (steps > 0) (if (finishing) 0L else 5L) else 250L)
        }
    }
}
