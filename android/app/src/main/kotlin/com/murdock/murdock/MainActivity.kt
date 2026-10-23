package com.murdock.murdock

import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.FlutterInjector
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.nio.ByteBuffer
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
 * Work runs on two single-thread executors. An exception escaping an executor task kills
 * the whole app on Android, so every task goes through [submit], which reports failures to
 * Dart instead.
 */
class MainActivity : FlutterActivity() {
    private val main = Handler(Looper.getMainLooper())
    private val brainThread = Executors.newSingleThreadExecutor()   // owns EchoBrain
    private val teacherThread = Executors.newSingleThreadExecutor() // owns TeacherDepth
    private val inferBusy = AtomicBoolean(false)

    private var sink: EventChannel.EventSink? = null
    private var sonar: SonarEngine? = null
    private var brain: EchoBrain? = null
    private var teacher: TeacherDepth? = null
    @Volatile private var latestStack: FloatArray? = null
    @Volatile private var training = false
    @Volatile private var brainState = "loading"

    private val prefs by lazy { getSharedPreferences("murdock", MODE_PRIVATE) }
    private val teacherFile by lazy { File(filesDir, "models/midas_v21_small.tflite") }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        EventChannel(messenger, "murdock/frames").setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink) { sink = events }
            override fun onCancel(arguments: Any?) { sink = null }
        })
        MethodChannel(messenger, "murdock/sonar").setMethodCallHandler(::onCall)
        submit(brainThread, "load model") {
            try {
                brain = loadBrain()
                brainState = "ready"
            } catch (t: Throwable) {
                brainState = "failed: ${t.message ?: t.javaClass.simpleName}"
                throw t
            }
        }
    }

    private fun loadBrain(): EchoBrain {
        val key = FlutterInjector.instance().flutterLoader().getLookupKeyForAsset("assets/models/echo_student.tflite")
        val bytes = assets.open(key).use { it.readBytes() }
        val buf = ByteBuffer.allocateDirect(bytes.size).put(bytes)
        return EchoBrain(buf, File(filesDir, "brain/head.weights"), File(getExternalFilesDir(null), "samples"))
    }

    private fun onCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "start" -> {
                if (sonar == null) {
                    sonar = SonarEngine(this, ::onPing, ::reportError) { stats ->
                        emit(stats + ("type" to "stats") + ("brain" to brainState))
                    }
                }
                sonar!!.start()
                result.success(null)
            }
            "stop" -> { sonar?.stop(); result.success(null) }
            "calibrate" -> { sonar?.calibrate(); result.success(null) }
            "status" -> submit(brainThread, "status", result) {
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
                    "samplesDir" to File(getExternalFilesDir(null), "samples").path,
                ))
            }
            "requestPermission" -> requestPermission(call.arguments as String, result)
            "setTraining" -> { training = call.arguments as Boolean; result.success(null) }
            "teacherReady" -> result.success(teacherFile.exists())
            "getFlag" -> result.success(prefs.getBoolean(call.arguments as String, false))
            "setFlag" -> {
                prefs.edit().putBoolean(call.arguments as String, true).apply()
                result.success(null)
            }
            "downloadTeacher" -> submit(teacherThread, "download", result) {
                teacherFile.parentFile?.mkdirs()
                TeacherDepth.download(teacherFile) { p -> emit(mapOf("type" to "download", "progress" to p.toDouble())) }
                reply(result, true)
            }
            "teach" -> teach(call.arguments as ByteArray, result)
            "save" -> submit(brainThread, "save", result) { brain?.save(); reply(result, null) }
            "reset" -> submit(brainThread, "reset", result) { brain?.reset(); reply(result, null) }
            else -> result.notImplemented()
        }
    }

    private val pendingPermissions = HashMap<Int, MethodChannel.Result>()

    /** [name] is "microphone" or "camera"; replies true when granted. */
    private fun requestPermission(name: String, result: MethodChannel.Result) {
        val permission = when (name) {
            "microphone" -> android.Manifest.permission.RECORD_AUDIO
            "camera" -> android.Manifest.permission.CAMERA
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

    /** Camera frame in -> teacher depth label, paired with the echoes heard at that moment. */
    private fun teach(rgb: ByteArray, result: MethodChannel.Result) {
        val stack = latestStack
        if (stack == null) { result.error("sonar", "sonar not running", null); return }
        submit(teacherThread, "teacher", result) {
            val t = teacher ?: TeacherDepth(teacherFile).also { teacher = it }
            val depth = t.run(rgb)
            submit(brainThread, "training") {
                val b = brain ?: return@submit
                b.addSample(stack, depth)
                b.train(TRAIN_STEPS_PER_SAMPLE)
            }
            reply(result, depth)
        }
    }

    private fun onPing(profile: FloatArray, clean: FloatArray, stack: FloatArray, directLevel: Float) {
        latestStack = stack
        // Raw echoes go straight to the UI: RADAR must work even if the AI can't run.
        emit(mapOf("type" to "ping", "profile" to profile, "clean" to clean, "direct" to directLevel.toDouble()))
        // Inference rides along when the brain is idle; skipped otherwise.
        if (!inferBusy.compareAndSet(false, true)) return
        val queued = submit(brainThread, "echo AI") {
            try {
                val b = brain ?: return@submit
                val depth = b.infer(stack)
                // Keep learning from the replay buffer between camera frames.
                if (training) b.train(1)
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

    override fun onPause() {
        super.onPause()
        sonar?.stop()
        submit(brainThread, "save") { brain?.save() }
    }

    override fun onDestroy() {
        sonar?.stop()
        submit(brainThread, "close") { brain?.close() }
        submit(teacherThread, "close") { teacher?.close() }
        brainThread.shutdown(); teacherThread.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "Murdock"
        private const val TRAIN_STEPS_PER_SAMPLE = 4
        private const val PERMISSION_REQUEST_BASE = 4100
    }
}
