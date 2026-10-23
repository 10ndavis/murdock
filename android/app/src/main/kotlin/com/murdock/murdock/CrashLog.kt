package com.murdock.murdock

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Remembers why the app last died, so the next launch can show it (and the user can copy it
 * to us) instead of the crash vanishing: Kotlin crashes are written to a file as they happen;
 * native crashes, freezes and out-of-memory kills come from Android's exit history (API 30+).
 */
object CrashLog {
    private const val PREFS = "crashlog"
    private const val SHOWN_UP_TO = "shownUpTo"
    private var installed = false

    private fun file(context: Context) = File(context.filesDir, "crash.txt")

    @Synchronized
    fun install(context: Context) {
        if (installed) return
        installed = true
        val out = file(context.applicationContext)
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, e ->
            try {
                val trace = StringWriter().also { e.printStackTrace(PrintWriter(it)) }.toString()
                out.writeText("${System.currentTimeMillis()}\nThread: ${thread.name}\n${trace.take(6000)}")
            } catch (_: Throwable) {
            }
            previous?.uncaughtException(thread, e)
        }
    }

    /** What ended the last session abnormally (not shown before), or null. Marks it shown. */
    fun takeLast(context: Context): String? {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val shownUpTo = prefs.getLong(SHOWN_UP_TO, 0L)
        var newest = shownUpTo
        val parts = ArrayList<String>()

        val f = file(context)
        if (f.exists()) {
            val text = f.readText()
            val at = text.substringBefore('\n').toLongOrNull() ?: 0L
            if (at > shownUpTo) {
                parts += "App error (${stamp(at)}):\n${text.substringAfter('\n')}"
                newest = maxOf(newest, at)
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val info = am.getHistoricalProcessExitReasons(context.packageName, 0, 1).firstOrNull()
            if (info != null && info.timestamp > shownUpTo && info.reason in REPORTED) {
                val mb = info.pss / 1024
                val lines = mutableListOf(
                    "Android ended the app (${stamp(info.timestamp)}): ${reasonName(info.reason)}",
                    "Details: ${info.description ?: "none"}",
                    "Memory in use: $mb MB",
                )
                // A freeze comes with the threads' stacks as text; keep the main thread's.
                if (info.reason == ApplicationExitInfo.REASON_ANR) {
                    info.traceInputStream?.bufferedReader()?.use { r ->
                        val trace = r.readText()
                        val main = trace.indexOf("\"main\"")
                        if (main >= 0) lines += trace.substring(main).take(3000)
                    }
                }
                parts += lines.joinToString("\n")
                newest = maxOf(newest, info.timestamp)
            }
        }

        if (newest > shownUpTo) prefs.edit().putLong(SHOWN_UP_TO, newest).apply()
        return parts.takeIf { it.isNotEmpty() }?.joinToString("\n\n")
    }

    private val REPORTED = setOf(
        ApplicationExitInfo.REASON_CRASH,
        ApplicationExitInfo.REASON_CRASH_NATIVE,
        ApplicationExitInfo.REASON_ANR,
        ApplicationExitInfo.REASON_LOW_MEMORY,
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE,
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE,
    )

    private fun reasonName(r: Int) = when (r) {
        ApplicationExitInfo.REASON_CRASH -> "app error"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "native crash (AI / graphics / audio library)"
        ApplicationExitInfo.REASON_ANR -> "app froze (not responding)"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "phone ran out of memory"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "used too much CPU or battery"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "failed to start"
        else -> "reason $r"
    }

    private fun stamp(ms: Long) = SimpleDateFormat("MMM d, h:mm:ss a", Locale.US).format(Date(ms))
}
