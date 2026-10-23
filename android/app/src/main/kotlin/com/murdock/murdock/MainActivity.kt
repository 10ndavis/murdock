package com.murdock.murdock

import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Bridges the native sonar + models to Flutter.
 *   MethodChannel "murdock/sonar": requestPermission, start, stop, teach, setTraining, save,
 *                                  reset, teacherReady, downloadTeacher, status
 *   EventChannel  "murdock/frames": one map per ping {profile, depth?, direct, ...},
 *                                   download progress, and {type: "error"} reports
 *
 * The echo AI and its trainer live in [BrainHost] (process-wide, shared with the background
 * [LearningService]); this activity only borrows them. The camera teacher has its own
 * thread here. An exception escaping an executor task kills the whole app on Android, so
 * every task goes through [submit]/[BrainHost.submit], which report failures to Dart instead.
 */
class MainActivity : FlutterActivity() {
    private val main = Handler(Looper.getMainLooper())
    private val teacherThread = Executors.newSingleThreadExecutor() // owns TeacherDepth
    private val inferBusy = AtomicBoolean(false)

    private var sink: EventChannel.EventSink? = null
    private var sonar: SonarEngine? = null
    private var teacher: TeacherDepth? = null
    @Volatile private var latestStack: FloatArray? = null

    private val prefs by lazy { getSharedPreferences("murdock", MODE_PRIVATE) }
    private val samplesDir by lazy { BrainHost.samplesDir(this) }
    private val brain get() = BrainHost.brain
    private val teacherFile by lazy { File(filesDir, "models/midas_v21_small.tflite") }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        EventChannel(messenger, "murdock/frames").setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink) { sink = events }
            override fun onCancel(arguments: Any?) { sink = null }
        })
        MethodChannel(messenger, "murdock/sonar").setMethodCallHandler(::onCall)
        BrainHost.onError = ::reportError
        BrainHost.ensureLoaded(this)
    }

    /** Brain-thread task whose failure is also returned to Dart as a channel error. */
    private fun submitBrain(what: String, result: MethodChannel.Result? = null, task: () -> Unit) =
        BrainHost.submit(what, onFail = { t -> if (result != null) main.post { result.error(what, t.message, null) } }, task = task)

    private fun onCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "start" -> {
                if (sonar == null) {
                    sonar = SonarEngine(this, ::onPing, ::reportError) { stats ->
                        // "ready", or e.g. "recovered 2× (learning rate 0.025)" after a rollback.
                        val b = brain
                        val state = if (b != null && BrainHost.state == "ready" && b.health != "ok") b.health else BrainHost.state
                        emit(stats + ("type" to "stats") + ("brain" to state))
                    }
                }
                sonar!!.start()
                result.success(null)
            }
            "stop" -> { sonar?.stop(); result.success(null) }
            "calibrate" -> { sonar?.calibrate(); result.success(null) }
            "finishLearning" -> {
                // Stop listening (no new samples, no inference) and train flat out in a
                // foreground service, which keeps going if the app is closed.
                sonar?.stop()
                LearningService.start(this)
                result.success(null)
            }
            "endFinishLearning" -> {
                LearningService.stop(this)   // saves what it learned so far
                result.success(null)
            }
            "status" -> submitBrain("status", result) {
                val b = brain
                val s = sonar
                reply(result, mapOf(
                    "stereo" to (s?.stereo ?: false),
                    "source" to (s?.sourceName ?: ""),
                    "samples" to (b?.sampleCount ?: 0),
                    "steps" to (b?.trainedSteps ?: 0),
                    "totalSamples" to (b?.totalSamples ?: 0),
                    "loss" to (b?.lastLoss?.toDouble() ?: Double.NaN),
                    "teacherReady" to teacherFile.exists(),
                    "samplesDir" to samplesDir.path,
                    "pendingSteps" to (b?.pendingSteps ?: 0),
                    "stepsPerSec" to BrainHost.stepsPerSec,
                    "finishing" to BrainHost.finishing,
                    // Samples learned per tilt bin, and at the current tilt, for the GRIP strip.
                    "tiltCoverage" to (b?.tiltCoverage?.toList() ?: emptyList<Int>()),
                    "coverageHere" to (b?.coverageAt(BrainHost.tiltDeg) ?: 0),
                    "teacherBackend" to (teacher?.backend ?: ""),
                    "teacherMs" to (teacher?.lastMs ?: 0L).toInt(),
                ))
            }
            "requestPermission" -> requestPermission(call.arguments as String, result)
            // Kept for older Dart code: training is continuous now, driven by new samples.
            "setTraining" -> result.success(null)
            "teacherReady" -> result.success(teacherFile.exists())
            "getFlag" -> result.success(prefs.getBoolean(call.arguments as String, false))
            "getNumber" -> result.success(prefs.getFloat(call.arguments as String, Float.NaN).toDouble())
            "setNumber" -> {
                val (key, value) = call.arguments as List<*>
                prefs.edit().putFloat(key as String, (value as Number).toFloat()).apply()
                result.success(null)
            }
            "clearFlag" -> {
                prefs.edit().remove(call.arguments as String).apply()
                result.success(null)
            }
            "setFlag" -> {
                prefs.edit().putBoolean(call.arguments as String, true).apply()
                result.success(null)
            }
            "downloadTeacher" -> submit(teacherThread, "download", result) {
                teacherFile.parentFile?.mkdirs()
                TeacherDepth.download(teacherFile) { p -> emit(mapOf("type" to "download", "progress" to p.toDouble())) }
                reply(result, true)
            }
            "teach" -> {
                val args = call.arguments as Map<*, *>
                teach(args["rgb"] as ByteArray, args["learn"] as Boolean, result)
            }
            "save" -> submitBrain("save", result) { brain?.save(); reply(result, null) }
            "reset" -> submitBrain("reset", result) { brain?.reset(); reply(result, null) }
            else -> result.notImplemented()
        }
    }

    private val pendingPermissions = HashMap<Int, MethodChannel.Result>()

    /** [name] is "microphone" or "camera"; replies true when granted. */
    private fun requestPermission(name: String, result: MethodChannel.Result) {
        val permission = when (name) {
            "microphone" -> android.Manifest.permission.RECORD_AUDIO
            "camera" -> android.Manifest.permission.CAMERA
            "notifications" -> {
                // Only Android 13+ asks; older versions always allow notifications.
                if (android.os.Build.VERSION.SDK_INT < 33) { result.success(true); return }
                android.Manifest.permission.POST_NOTIFICATIONS
            }
            else -> { result.error("permission", "unknown permission $name", null); return }
        }
        if (checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED) {
            result.success(true)
            return
        }
        val code = PERMISSION_REQUEST_BASE + pendingPermissions.size
        pendingPermissions[code] = result
        requestPermissions(arrayOf(permission), code)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        pendingPermissions.remove(requestCode)?.success(
            grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)
    }

    /**
     * Camera frame in -> camera depth (for the camera-assist picture). With [learn], the depth
     * also becomes a training label, paired with the echoes heard at that moment; the
     * continuous trainer then spends the steps that sample earned.
     * Teacher (GPU when available) and trainer (CPU) run on separate threads in parallel.
     */
    private fun teach(rgb: ByteArray, learn: Boolean, result: MethodChannel.Result) {
        val stack = latestStack
        // The angle the phone was held at when these echoes were heard.
        val pose = BrainHost.pose
        val tilt = BrainHost.tiltDeg
        if (stack == null) { result.error("sonar", "sonar not running", null); return }
        submit(teacherThread, "teacher", result) {
            val t = teacher ?: TeacherDepth(teacherFile).also { teacher = it }
            val depth = t.run(rgb)
            if (learn) submitBrain("training") { brain?.addSample(stack, pose, depth, tilt) }
            reply(result, mapOf("depth" to depth, "ms" to t.lastMs.toInt(), "backend" to t.backend))
        }
    }

    private fun onPing(profile: FloatArray, clean: FloatArray, stack: FloatArray, directLevel: Float) {
        latestStack = stack
        // Raw echoes go straight to the UI: RADAR must work even if the AI can't run.
        emit(mapOf("type" to "ping", "profile" to profile, "clean" to clean, "direct" to directLevel.toDouble()))
        // Inference rides along when the brain is idle; skipped otherwise.
        if (!inferBusy.compareAndSet(false, true)) return
        val queued = submitBrain("echo AI") {
            try {
                val b = brain ?: return@submitBrain
                val depth = b.infer(stack, BrainHost.pose)
                emit(mapOf(
                    "type" to "brain",
                    "depth" to depth,
                    "steps" to b.trainedSteps,
                    "loss" to b.lastLoss.toDouble(),
                    "samples" to b.totalSamples,
                ))
            } finally {
                inferBusy.set(false)
            }
        }
        if (!queued) inferBusy.set(false)
    }

    /**
     * Runs [task] on [executor]; any failure is logged, sent to Dart as an error event and,
     * when [result] is given, returned as a channel error. Returns false if not queued.
     */
    private fun submit(
        executor: ExecutorService,
        what: String,
        result: MethodChannel.Result? = null,
        task: () -> Unit,
    ): Boolean = try {
        executor.execute {
            try {
                task()
            } catch (t: Throwable) {
                Log.e(TAG, "$what failed", t)
                reportError("$what failed: ${t.message ?: t.javaClass.simpleName}")
                if (result != null) main.post { result.error(what, t.message, null) }
            }
        }
        true
    } catch (e: RejectedExecutionException) {
        false // activity is shutting down
    }

    @Volatile private var lastError = ""
    @Volatile private var lastErrorAt = 0L

    /** Sends an error to the UI, at most once per 3 s per message (inference runs ~20x/s). */
    private fun reportError(message: String) {
        val now = System.currentTimeMillis()
        if (message == lastError && now - lastErrorAt < 3000) return
        lastError = message; lastErrorAt = now
        emit(mapOf("type" to "error", "message" to message))
    }

    private fun emit(event: Map<String, Any>) = main.post { sink?.success(event) }
    private fun reply(result: MethodChannel.Result, value: Any?) = main.post { result.success(value) }

    // ── Grip / tilt, for the GRIP indicator ─────────────────────────────────────────────
    // The app runs upside down (reversePortrait) so the bottom-edge speaker is at the TOP and
    // a comfortable viewing angle aims it away from the user.
    // Gravity in device coordinates -> tilt and roll *as the user sees the screen*:
    //   tilt: 0 = upright; POSITIVE = displayed top (the speaker end) tilted AWAY from the
    //         user, screen turning up toward their face; 90 = flat, screen up.
    //   roll: sideways lean; about ±90 when held landscape.
    // At rest the sensor reads +g along the axis pointing up. Upside down, "up on screen" is
    // the device's -y axis, so upright gives gy = -g; screen-up gives +gz.
    private var lastPose = 0L
    private val poseListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) {
            val now = System.currentTimeMillis()
            if (now - lastPose < 200) return
            lastPose = now
            val (gx, gy, gz) = e.values
            val tilt = Math.toDegrees(kotlin.math.atan2(gz.toDouble(), -gy.toDouble()))
            val roll = Math.toDegrees(kotlin.math.atan2(-gx.toDouble(), kotlin.math.hypot(gy.toDouble(), gz.toDouble())))
            BrainHost.setPose(tilt.toFloat(), roll.toFloat())   // the echo AI's angle input
            emit(mapOf("type" to "pose", "tilt" to tilt, "roll" to roll))
        }

        override fun onAccuracyChanged(s: Sensor?, accuracy: Int) {}
    }

    override fun onResume() {
        super.onResume()
        val sm = getSystemService(SENSOR_SERVICE) as SensorManager
        (sm.getDefaultSensor(Sensor.TYPE_GRAVITY) ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER))?.let {
            sm.registerListener(poseListener, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    override fun onPause() {
        super.onPause()
        (getSystemService(SENSOR_SERVICE) as SensorManager).unregisterListener(poseListener)
        sonar?.stop()
        submitBrain("save") { brain?.save() }   // also persists the training queue
    }

    override fun onDestroy() {
        sonar?.stop()
        // The brain stays alive in BrainHost: a running LearningService may still be using it.
        BrainHost.onError = null
        submit(teacherThread, "close") { teacher?.close() }
        teacherThread.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "Murdock"
        private const val PERMISSION_REQUEST_BASE = 4100
    }
}
