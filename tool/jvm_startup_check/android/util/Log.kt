package android.util
object Log {
    @JvmStatic fun w(t: String, m: String): Int { println("W/$t: $m"); return 0 }
    @JvmStatic fun i(t: String, m: String): Int { println("I/$t: $m"); return 0 }
    @JvmStatic fun e(t: String, m: String, e: Throwable): Int { println("E/$t: $m $e"); return 0 }
}
