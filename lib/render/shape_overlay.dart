import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import 'palette.dart';
import 'shapes.dart';

/// Wireframe floor grid, wall panels and obstacle boxes drawn over the echo picture, in the
/// same square as [VisionView]. Each shape fades with its confidence, and the lines wobble
/// and break up a little: a sensed outline, never a perfect CAD drawing.
class ShapeOverlay extends StatefulWidget {
  const ShapeOverlay({super.key, required this.tracker, required this.palette, required this.feet});

  final ShapeTracker tracker;
  final SightPalette palette;

  /// Distance of the nearest clear echo, in feet (labels the nearest shape), or null.
  final double? Function() feet;

  @override
  State<ShapeOverlay> createState() => _ShapeOverlayState();
}

class _ShapeOverlayState extends State<ShapeOverlay> with SingleTickerProviderStateMixin {
  final _time = ValueNotifier<double>(0);
  late final Ticker _ticker = createTicker((e) => _time.value = e.inMicroseconds / 1e6)..start();

  @override
  void dispose() {
    _ticker.dispose();
    _time.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => IgnorePointer(
        child: CustomPaint(
          size: Size.infinite,
          painter: ShapePainter(widget.tracker, widget.palette, _time, widget.feet),
        ),
      );
}

class ShapePainter extends CustomPainter {
  ShapePainter(this.tracker, this.palette, this.time, this.feet) : super(repaint: time);

  final ShapeTracker tracker;
  final SightPalette palette;
  final ValueNotifier<double> time;
  final double? Function() feet;

  static const double _maxAlpha = 0.85;

  @override
  void paint(Canvas canvas, Size size) {
    final s = tracker.shapes;
    if (s == null) return;
    final t = time.value;
    // Same square as VisionView: the map fitted to the width, centred vertically.
    final side = size.width;
    final top = (size.height - side) / 2;
    Offset p(double x, double y) => Offset(x * side, top + y.clamp(-0.05, 1.05) * side);
    canvas.save();
    canvas.clipRect(Rect.fromLTWH(0, top, side, side));

    final floor = s.floor;
    final horizon = floor.presence > 0.5 ? floor.horizon : 0.45;
    final vp = Offset(0.5 + wobble(t, 1, amp: 0.004), horizon);

    // Floor: perspective grid, lines at equal depth steps (rows ~ 1/distance) to the horizon.
    if (floor.presence > 0.3) {
      final a = _alpha(floor.presence, 0.6);
      final span = 1 - vp.dy;
      for (var k = -5; k <= 5; k++) {
        final bx = 0.5 + k * 0.17;
        final from = Offset(vp.dx + (bx - vp.dx) * 0.06, vp.dy + span * 0.06);
        _line(canvas, p(from.dx, from.dy), p(bx, 1.02), a * 0.8, 10 + k, t);
      }
      for (var j = 1; j <= 7; j++) {
        final y = vp.dy + span / j;
        _line(canvas, p(0, y), p(1, y), a * (0.25 + 0.75 / j), 30 + j, t);
      }
    }

    // Walls: one panel per band, taller where nearer; slope tilts its top and bottom edges.
    for (final (k, w) in s.walls.indexed) {
      if (w.presence < 0.4) continue;
      final a = _alpha(w.presence, 0.5 * (0.4 + 0.6 * w.nearness));
      double half(double x) => 0.07 + 0.4 * w.nearnessAt(x);
      final x0 = w.x0 + (k == 0 ? 0.0 : 0.01), x1 = w.x1 - (k == 2 ? 0.0 : 0.01);
      final cy = horizon + wobble(t, 50 + k, amp: 0.004);
      final tl = p(x0, cy - half(x0)), tr = p(x1, cy - half(x1));
      final bl = p(x0, cy + half(x0)), br = p(x1, cy + half(x1));
      _line(canvas, tl, tr, a, 60 + k, t);
      _line(canvas, bl, br, a, 70 + k, t);
      _line(canvas, tl, bl, a * 0.7, 80 + k, t);
      _line(canvas, tr, br, a * 0.7, 90 + k, t);
      for (final f in [1 / 3, 2 / 3]) {
        final x = x0 + (x1 - x0) * f;
        _line(canvas, p(x, cy - half(x)), p(x, cy + half(x)), a * 0.35, 100 + k * 3 + (f * 3).round(), t);
      }
    }

    // Obstacles: wireframe boxes, the back face pulled towards the vanishing point (deeper
    // for farther objects), with corner brackets on the front face.
    for (final (k, b) in s.boxes.indexed) {
      if (b.presence < 0.35) continue;
      final a = _alpha(b.presence, 0.55 + 0.45 * b.nearness);
      final jx = wobble(t, 120 + k, amp: 0.006), jy = wobble(t, 130 + k, amp: 0.006);
      final l = b.cx - b.w / 2 + jx, r = b.cx + b.w / 2 + jx, tp = b.cy - b.h / 2 + jy, bt = b.cy + b.h / 2 + jy;
      final depth = 0.06 + 0.18 * (1 - b.nearness);
      Offset back(double x, double y) => Offset(x + (vp.dx - x) * depth, y + (vp.dy - y) * depth);
      final front = [Offset(l, tp), Offset(r, tp), Offset(r, bt), Offset(l, bt)];
      final rear = [for (final c in front) back(c.dx, c.dy)];
      for (var i = 0; i < 4; i++) {
        final n = (i + 1) % 4;
        _line(canvas, p(rear[i].dx, rear[i].dy), p(rear[n].dx, rear[n].dy), a * 0.45, 140 + k * 20 + i, t);
        _line(canvas, p(front[i].dx, front[i].dy), p(rear[i].dx, rear[i].dy), a * 0.6, 150 + k * 20 + i, t);
        _line(canvas, p(front[i].dx, front[i].dy), p(front[n].dx, front[n].dy), a, 160 + k * 20 + i, t);
      }
      _brackets(canvas, p(l, tp), p(r, bt), a);
    }

    final near = s.nearest;
    final ft = feet();
    if (near != null && ft != null) {
      final tp = TextPainter(
        text: TextSpan(
          text: '≈ ${ft.toStringAsFixed(ft < 10 ? 1 : 0)} FT',
          style: TextStyle(
              color: palette.accent.withValues(alpha: 0.9),
              fontSize: 11,
              letterSpacing: 2,
              fontWeight: FontWeight.w600,
              fontFamily: 'monospace'),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final at = p(near.cx, near.cy) - Offset(tp.width / 2, tp.height + 6);
      tp.paint(canvas, Offset(at.dx.clamp(4, side - tp.width - 4), at.dy.clamp(top + 4, top + side - tp.height - 4)));
    }
    canvas.restore();
  }

  double _alpha(double presence, double strength) =>
      (((presence - 0.3) / 0.7).clamp(0.0, 1.0) * strength).clamp(0.0, _maxAlpha);

  /// A slightly hand-drawn line: endpoints wobble and a few pieces drop out, changing
  /// a couple of times a second (like an outline that keeps being re-sensed).
  void _line(Canvas canvas, Offset a, Offset b, double alpha, int seed, double t) {
    if (alpha <= 0.02) return;
    final glow = Paint()
      ..color = palette.accent.withValues(alpha: alpha * 0.35)
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3);
    final core = Paint()
      ..color = Color.lerp(palette.accent, palette.hot, 0.35)!.withValues(alpha: alpha)
      ..strokeWidth = 1.3
      ..strokeCap = StrokeCap.round;
    const pieces = 7;
    final epoch = (t * 2).floor();
    final n = Offset(-(b - a).dy, (b - a).dx);
    final unit = n.distance > 0 ? n / n.distance : Offset.zero;
    Offset at(int i) => Offset.lerp(a, b, i / pieces)! + unit * wobble(t, seed * 7 + i, amp: 1.2);
    for (var i = 0; i < pieces; i++) {
      if (_hash(seed, i, epoch) < 0.12) continue;
      final p0 = at(i), p1 = at(i + 1);
      canvas.drawLine(p0, p1, glow);
      canvas.drawLine(p0, p1, core);
    }
  }

  void _brackets(Canvas canvas, Offset tl, Offset br, double alpha) {
    final paint = Paint()
      ..color = palette.hot.withValues(alpha: alpha)
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.square;
    final l = math.min(10.0, (br.dx - tl.dx).abs() / 3);
    for (final (x, y, dx, dy) in [
      (tl.dx, tl.dy, 1.0, 1.0),
      (br.dx, tl.dy, -1.0, 1.0),
      (tl.dx, br.dy, 1.0, -1.0),
      (br.dx, br.dy, -1.0, -1.0),
    ]) {
      canvas.drawLine(Offset(x, y), Offset(x + l * dx, y), paint);
      canvas.drawLine(Offset(x, y), Offset(x, y + l * dy), paint);
    }
  }

  static double _hash(int a, int b, int c) {
    var h = a * 73856093 ^ b * 19349663 ^ c * 83492791;
    h = (h ^ (h >> 13)) * 1274126177;
    return ((h ^ (h >> 16)) & 0xffff) / 0xffff;
  }

  @override
  bool shouldRepaint(ShapePainter old) => old.tracker != tracker || old.palette != palette;
}
