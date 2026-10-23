package com.murdock.murdock

import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import io.flutter.FlutterInjector
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Bridges the native sonar + models to Flutter.
 *   MethodChannel "murdock/sonar": requestPermission, start, stop, teach, setTraining, save,
 *                                  reset, teacherReady, downloadTeacher, status
 *   EventChannel  "murdock/frames": one map per ping {profile, depth?, direct, ...}
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

    private val teacherFile by lazy { File(filesDir, "models/midas_v21_small.tflite") }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        EventChannel(messenger, "murdock/frames").setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink) { sink = events }
            override fun onCancel(arguments: Any?) { sink = null }
        })
        MethodChannel(messenger, "murdock/sonar").setMethodCallHandler(::onCall)
        brainThread.execute { brain = loadBrain() }
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
                if (sonar == null) sonar = SonarEngine(this, ::onPing)
                sonar!!.start()
                result.success(null)
            }
            "stop" -> { sonar?.stop(); result.success(null) }
            "status" -> brainThread.execute {
                val b = brain
                val s = sonar
                reply(result, mapOf(
                    "stereo" to (s?.stereo ?: false),
                    "source" to (s?.sourceName ?: ""),
                    "samples" to (b?.sampleCount ?: 0),
                    "steps" to (b?.trainedSteps ?: 0),
                    "loss" to (b?.lastLoss?.toDouble() ?: Double.NaN),
                    "teacherReady" to teacherFile.exists(),
                    "samplesDir" to File(getExternalFilesDir(null), "samples").path,
                ))
            }
            "requestPermission" -> requestPermission(call.arguments as String, result)
            "setTraining" -> { training = call.arguments as Boolean; result.success(null) }
            "teacherReady" -> result.success(teacherFile.exists())
            "downloadTeacher" -> teacherThread.execute {
                try {
                    teacherFile.parentFile?.mkdirs()
                    TeacherDepth.download(teacherFile) { p -> emit(mapOf("type" to "download", "progress" to p.toDouble())) }
                    reply(result, true)
                } catch (e: Exception) {
                    main.post { result.error("download", e.message, null) }
                }
            }
            "teach" -> teach(call.arguments as ByteArray, result)
            "save" -> brainThread.execute { brain?.save(); reply(result, null) }
            "reset" -> brainThread.execute { brain?.reset(); reply(result, null) }
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
        teacherThread.execute {
            try {
                val t = teacher ?: TeacherDepth(teacherFile).also { teacher = it }
                val depth = t.run(rgb)
                brainThread.execute {
                    val b = brain ?: return@execute
                    b.addSample(stack, depth)
                    b.train(TRAIN_STEPS_PER_SAMPLE)
                }
                reply(result, depth)
            } catch (e: Exception) {
                main.post { result.error("teacher", e.message, null) }
            }
        }
    }

    private fun onPing(profile: FloatArray, stack: FloatArray, directLevel: Float) {
        latestStack = stack
        val event = hashMapOf<String, Any>(
            "type" to "ping",
            "profile" to profile,
            "direct" to directLevel.toDouble(),
        )
        // Inference rides along with the ping when the brain is idle; dropped otherwise.
        if (!inferBusy.compareAndSet(false, true)) { emit(event); return }
        brainThread.execute {
            try {
                val b = brain
                if (b != null) {
                    event["depth"] = b.infer(stack)
                    event["steps"] = b.trainedSteps
                    event["loss"] = b.lastLoss.toDouble()
                    event["samples"] = b.sampleCount
                    // Keep learning from the replay buffer between camera frames.
                    if (training) b.train(1)
                }
            } finally {
                inferBusy.set(false)
            }
            emit(event)
        }
    }

    private fun emit(event: Map<String, Any>) = main.post { sink?.success(event) }
    private fun reply(result: MethodChannel.Result, value: Any?) = main.post { result.success(value) }

    override fun onPause() {
        super.onPause()
        sonar?.stop()
        brainThread.execute { brain?.save() }
    }

    override fun onDestroy() {
        sonar?.stop()
        brainThread.execute { brain?.close() }
        teacherThread.execute { teacher?.close() }
        brainThread.shutdown(); teacherThread.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val TRAIN_STEPS_PER_SAMPLE = 4
        private const val PERMISSION_REQUEST_BASE = 4100
    }
}
