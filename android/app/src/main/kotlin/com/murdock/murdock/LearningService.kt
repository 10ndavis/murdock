package com.murdock.murdock

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager

/**
 * FINISH LEARNING as a foreground service: keeps training through the queue with a
 * persistent notification (progress, time left, Stop), even if the app is closed or the
 * screen is off, then saves and posts "Learning finished".
 *
 * The brain lives in [BrainHost] (process-wide), so the service and the activity share it.
 */
class LearningService : Service() {
    private val main = Handler(Looper.getMainLooper())
    private var wakeLock: PowerManager.WakeLock? = null
    private var total = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopLearning(finished = false)
            return START_NOT_STICKY
        }
        CrashLog.install(this)
        createChannel()
        val notification = progressNotification(pending = null)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFY_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFY_ID, notification)
        }
        // Keep the CPU running with the screen off; capped so a stuck run can't drain the battery.
        if (wakeLock == null) {
            wakeLock = (getSystemService(Context.POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "murdock:learning")
                .apply { acquire(MAX_RUN_MS) }
        }
        BrainHost.ensureLoaded(this)
        BrainHost.onFinished = { stopLearning(finished = true) }
        BrainHost.finishing = true
        main.removeCallbacks(updater)
        main.post(updater)
        // Restarted by Android after being killed: carry on (the queue is saved with the brain).
        return START_STICKY
    }

    private val updater = object : Runnable {
        override fun run() {
            val pending = BrainHost.brain?.pendingSteps
            if (pending != null && pending > total) total = pending
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .notify(NOTIFY_ID, progressNotification(pending))
            main.postDelayed(this, 1000)
        }
    }

    private fun stopLearning(finished: Boolean) {
        main.removeCallbacks(updater)
        BrainHost.finishing = false
        BrainHost.onFinished = null
        // Save whatever was learned so far (also when stopped early).
        BrainHost.submit("save") { BrainHost.brain?.save() }
        stopForeground(STOP_FOREGROUND_REMOVE)
        if (finished) {
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(DONE_ID, doneNotification())
        }
        stopSelf()
    }

    override fun onDestroy() {
        main.removeCallbacks(updater)
        BrainHost.finishing = false
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
        super.onDestroy()
    }

    // ── Notifications ───────────────────────────────────────────────────────────────────

    private fun createChannel() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, "Learning", NotificationManager.IMPORTANCE_LOW).apply {
                description = "Progress while Murdock finishes learning in the background"
            })
    }

    private fun openApp(): PendingIntent = PendingIntent.getActivity(
        this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_IMMUTABLE)

    private fun progressNotification(pending: Int?): Notification {
        val stop = PendingIntent.getService(
            this, 1, Intent(this, LearningService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE)
        val rate = BrainHost.stepsPerSec
        val text = when {
            pending == null -> "Loading the echo AI…"
            rate < 1 -> "$pending steps left"
            else -> {
                val s = (pending / rate).toInt()
                "$pending steps left · ${if (s < 60) "~$s s" else "~${(s + 59) / 60} min"}"
            }
        }
        val done = if (total > 0 && pending != null) total - pending else 0
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("Murdock is finishing learning")
            .setContentText(text)
            .setProgress(maxOf(total, 1), done, pending == null)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openApp())
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }

    private fun doneNotification(): Notification = Notification.Builder(this, CHANNEL)
        .setSmallIcon(R.mipmap.ic_launcher)
        .setContentTitle("Learning finished")
        .setContentText("Everything it collected has been learned and saved.")
        .setContentIntent(openApp())
        .setAutoCancel(true)
        .build()

    companion object {
        private const val CHANNEL = "learning"
        private const val NOTIFY_ID = 7101
        private const val DONE_ID = 7102
        private const val ACTION_STOP = "com.murdock.murdock.STOP_LEARNING"
        private const val MAX_RUN_MS = 60 * 60 * 1000L

        fun start(context: Context) {
            context.startForegroundService(Intent(context, LearningService::class.java))
        }

        fun stop(context: Context) {
            context.startService(Intent(context, LearningService::class.java).setAction(ACTION_STOP))
        }
    }
}
