package com.murdock.murdock

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Scene primitives (floor, three wall bands, two nearest obstacles) fitted to a 32x32
 * nearness map: the shape labels the echo AI learns from the camera teacher.
 *
 * Mirrors ml/shapes.py exactly (ml/check_shapes_kotlin.py compares them); see there for the
 * layout of the [SHAPE_DIM] values and why these fits.
 */
object ShapeFit {
    const val N = 32
    const val SHAPE_DIM = 24
    const val FLOOR = 0
    const val WALLS = 3
    const val BOXES = 12
    private val WALL_BANDS = arrayOf(intArrayOf(0, 11), intArrayOf(11, 21), intArrayOf(21, 32))
    private const val FLOOR_FROM = 20
    private const val MIN_WALL_ROWS = 8
    private const val BOX_RISE = 0.12
    private const val MIN_BOX_CELLS = 6
    const val MAX_BOXES = 2

    private fun clamp(v: Double, lo: Double = 0.0, hi: Double = 1.0) = min(max(v, lo), hi)

    private fun median(v: DoubleArray): Double {
        val s = v.sortedArray()
        val m = s.size / 2
        return if (s.size % 2 == 1) s[m] else 0.5 * (s[m - 1] + s[m])
    }

    private class Line(val a: Double, val b: Double, val rmse: Double, val inliers: Double)

    private fun lsq(xs: DoubleArray, ys: DoubleArray, keep: BooleanArray?): Pair<Double, Double> {
        var n = 0; var xm = 0.0; var ym = 0.0
        for (i in xs.indices) if (keep == null || keep[i]) { xm += xs[i]; ym += ys[i]; n++ }
        xm /= n; ym /= n
        var sxx = 0.0; var sxy = 0.0
        for (i in xs.indices) if (keep == null || keep[i]) {
            sxx += (xs[i] - xm) * (xs[i] - xm); sxy += (xs[i] - xm) * (ys[i] - ym)
        }
        val b = if (sxx > 0) sxy / sxx else 0.0
        return Pair(ym - b * xm, b)
    }

    /** Robust line: least squares, then once more without the outliers (ml/shapes.py _line). */
    private fun line(xs: DoubleArray, ys: DoubleArray): Line {
        var (a, b) = lsq(xs, ys, null)
        var ss = 0.0
        for (i in xs.indices) { val r = a + b * xs[i] - ys[i]; ss += r * r }
        val tol = max(2.0 * sqrt(ss / xs.size), 0.03)
        val keep = BooleanArray(xs.size) { abs(a + b * xs[it] - ys[it]) <= tol }
        val kept = keep.count { it }
        if (kept >= max(3, xs.size / 2) && kept < xs.size) {
            val ab = lsq(xs, ys, keep); a = ab.first; b = ab.second
        }
        var inl = 0; var ssIn = 0.0
        for (i in xs.indices) {
            val r = abs(a + b * xs[i] - ys[i])
            if (r <= tol) { inl++; ssIn += r * r }
        }
        val rmse = if (inl > 0) sqrt(ssIn / inl) else 1.0
        return Line(a, b, rmse, inl.toDouble() / xs.size)
    }

    /** [N*N] nearness (row-major, 1 = near) -> [SHAPE_DIM] primitives in [0, 1]. */
    fun fit(nearness: FloatArray): FloatArray {
        require(nearness.size == N * N)
        val m = DoubleArray(N * N) { val v = nearness[it].toDouble(); if (v.isFinite()) v else 0.0 }
        val out = DoubleArray(SHAPE_DIM)

        // Floor: the bottom rows' median nearness rises linearly towards the phone.
        val rows = DoubleArray(N) { y -> median(DoubleArray(N) { x -> m[y * N + x] }) }
        val ys = DoubleArray(N - FLOOR_FROM) { (FLOOR_FROM + it).toDouble() }
        val fl = line(ys, DoubleArray(ys.size) { rows[FLOOR_FROM + it] })
        val rise = fl.b * (N - FLOOR_FROM)
        val floorP = clamp((rise - 0.15) / 0.35) * clamp(1.0 - fl.rmse / 0.08) * fl.inliers
        val upper = median(rows.copyOfRange(0, MIN_WALL_ROWS))
        val horizon = clamp(if (fl.b > 1e-6) (upper - fl.a) / fl.b else (N - 1).toDouble(), 0.0, (N - 1).toDouble())
        out[FLOOR] = floorP; out[FLOOR + 1] = horizon / (N - 1); out[FLOOR + 2] = clamp(fl.a + fl.b * (N - 1))
        val wallRows = max(MIN_WALL_ROWS, horizon.toInt())

        // Walls: per band, a straight line across its columns (rows above the horizon).
        val background = DoubleArray(N * N)
        for ((k, band) in WALL_BANDS.withIndex()) {
            val x0 = band[0]; val x1 = band[1]
            val cols = DoubleArray(x1 - x0) { i -> median(DoubleArray(wallRows) { y -> m[y * N + x0 + i] }) }
            val xs = DoubleArray(x1 - x0) { i -> (x0 + i - (x0 + x1 - 1) / 2.0) / (x1 - x0) }
            val w = line(xs, cols)
            val o = WALLS + 3 * k
            out[o] = clamp(1.0 - w.rmse / 0.12) * w.inliers; out[o + 1] = clamp(w.a); out[o + 2] = clamp(0.5 + 0.5 * w.b)
            for (y in 0 until N) for (i in xs.indices) background[y * N + x0 + i] = w.a + w.b * xs[i]
        }
        for (y in wallRows until N) for (x in 0 until N) background[y * N + x] = fl.a + fl.b * y

        // Obstacles: 4-connected blobs clearly nearer than the floor+walls background.
        val mask = BooleanArray(N * N) { m[it] - background[it] > BOX_RISE }
        val seen = BooleanArray(N * N)
        val boxes = ArrayList<DoubleArray>()   // near, presence, cx, cy, w, h
        val stack = IntArray(N * N)
        for (start in 0 until N * N) {
            if (!mask[start] || seen[start]) continue
            var sp = 0
            stack[sp++] = start; seen[start] = true
            var cells = 0; var sumNear = 0.0; var sumRise = 0.0
            var y0 = N; var y1 = 0; var bx0 = N; var bx1 = 0
            while (sp > 0) {
                val c = stack[--sp]
                val cy = c / N; val cx = c % N
                cells++; sumNear += m[c]; sumRise += m[c] - background[c]
                y0 = min(y0, cy); y1 = max(y1, cy + 1); bx0 = min(bx0, cx); bx1 = max(bx1, cx + 1)
                // Same neighbour order as the NumPy version: up, down, left, right.
                if (cy > 0) push(c - N, mask, seen, stack, sp).also { sp = it }
                if (cy < N - 1) push(c + N, mask, seen, stack, sp).also { sp = it }
                if (cx > 0) push(c - 1, mask, seen, stack, sp).also { sp = it }
                if (cx < N - 1) push(c + 1, mask, seen, stack, sp).also { sp = it }
            }
            if (cells < MIN_BOX_CELLS) continue
            val area = (y1 - y0) * (bx1 - bx0) / (N * N).toDouble()
            val presence = clamp(cells / 16.0) * clamp(sumRise / cells / 0.25) * clamp((0.6 - area) / 0.3)
            boxes.add(doubleArrayOf(sumNear / cells, presence, (bx0 + bx1) / 2.0 / N, (y0 + y1) / 2.0 / N,
                (bx1 - bx0).toDouble() / N, (y1 - y0).toDouble() / N))
        }
        boxes.sortByDescending { it[0] }   // nearest first; stable like Python's sort
        for (k in 0 until min(MAX_BOXES, boxes.size)) {
            val b = boxes[k]
            val o = BOXES + 6 * k
            out[o] = b[1]; out[o + 1] = b[0]
            for (j in 2 until 6) out[o + j] = b[j]
        }
        return FloatArray(SHAPE_DIM) { out[it].toFloat() }
    }

    private fun push(i: Int, mask: BooleanArray, seen: BooleanArray, stack: IntArray, sp: Int): Int {
        if (!mask[i] || seen[i]) return sp
        seen[i] = true
        stack[sp] = i
        return sp + 1
    }
}
