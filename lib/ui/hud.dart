import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../render/palette.dart';

/// Shared look: black glass panels, thin neon strokes, corner brackets, mono caps.
class Hud {
  static const bg = Color(0xFF05060A);
  static const glass = Color(0x1AFFFFFF);
  static const text = Color(0xFFE8ECF4);
  static const dim = Color(0x99E8ECF4);

  static TextStyle label(Color c, {double size = 10}) =>
      TextStyle(color: c, fontSize: size, letterSpacing: 2.2, fontWeight: FontWeight.w600);

  static TextStyle title(Color c, {double size = 22}) =>
      TextStyle(color: c, fontSize: size, letterSpacing: 4, fontWeight: FontWeight.w300);

  static const body = TextStyle(color: dim, fontSize: 14, height: 1.5);
}

/// Frosted panel with neon corner brackets.
class HudPanel extends StatelessWidget {
  const HudPanel({super.key, required this.palette, required this.child, this.padding = const EdgeInsets.all(14)});

  final SightPalette palette;
  final Widget child;
  final EdgeInsets padding;

  @override
  Widget build(BuildContext context) => ClipRRect(
        borderRadius: BorderRadius.circular(4),
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 12, sigmaY: 12),
          child: CustomPaint(
            foregroundPainter: _BracketPainter(palette.accent),
            child: Container(
              padding: padding,
              decoration: BoxDecoration(
                color: Hud.glass,
                border: Border.all(color: palette.accent.withValues(alpha: 0.18)),
              ),
              child: child,
            ),
          ),
        ),
      );
}

class _BracketPainter extends CustomPainter {
  _BracketPainter(this.color);

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    const l = 12.0;
    for (final (x, y, dx, dy) in [
      (0.0, 0.0, 1.0, 1.0),
      (size.width, 0.0, -1.0, 1.0),
      (0.0, size.height, 1.0, -1.0),
      (size.width, size.height, -1.0, -1.0),
    ]) {
      canvas.drawLine(Offset(x, y), Offset(x + l * dx, y), p);
      canvas.drawLine(Offset(x, y), Offset(x, y + l * dy), p);
    }
  }

  @override
  bool shouldRepaint(_BracketPainter old) => old.color != color;
}

/// Neon outlined button.
class GlowButton extends StatelessWidget {
  const GlowButton({super.key, required this.label, required this.onTap, required this.palette, this.icon, this.filled = false});

  final String label;
  final VoidCallback? onTap;
  final SightPalette palette;
  final IconData? icon;
  final bool filled;

  @override
  Widget build(BuildContext context) {
    final c = palette.accent;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(30),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 13),
        decoration: BoxDecoration(
          color: filled ? c.withValues(alpha: 0.9) : Colors.transparent,
          borderRadius: BorderRadius.circular(30),
          border: Border.all(color: c, width: 1.4),
          boxShadow: [BoxShadow(color: c.withValues(alpha: 0.45), blurRadius: 18)],
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          if (icon != null) ...[Icon(icon, size: 18, color: filled ? Colors.black : c), const SizedBox(width: 8)],
          Text(label, style: Hud.label(filled ? Colors.black : c, size: 12)),
        ]),
      ),
    );
  }
}

/// Five-bar meter, like a phone's signal icon, with a caption.
class SignalBars extends StatelessWidget {
  const SignalBars({super.key, required this.value, required this.palette, required this.caption});

  final double value;
  final SightPalette palette;
  final String caption;

  @override
  Widget build(BuildContext context) {
    final lit = (value * 5).ceil();
    return Row(mainAxisSize: MainAxisSize.min, children: [
      for (var i = 0; i < 5; i++)
        Container(
          width: 3,
          height: 5.0 + i * 2.5,
          margin: const EdgeInsets.only(right: 2),
          decoration: BoxDecoration(
            color: i < lit ? palette.accent : palette.accent.withValues(alpha: 0.18),
            boxShadow: i < lit ? [BoxShadow(color: palette.accent.withValues(alpha: 0.6), blurRadius: 4)] : null,
          ),
        ),
      const SizedBox(width: 6),
      Text(caption, style: Hud.label(palette.accent, size: 9)),
    ]);
  }
}

/// Circular progress with a big number in the middle.
class ProgressRing extends StatelessWidget {
  const ProgressRing({super.key, required this.value, required this.palette, required this.center, required this.caption, this.size = 120});

  final double value;
  final SightPalette palette;
  final String center;
  final String caption;
  final double size;

  @override
  Widget build(BuildContext context) => SizedBox(
        width: size,
        height: size,
        child: CustomPaint(
          painter: _RingPainter(value.clamp(0, 1), palette.accent),
          child: Center(
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              Text(center, style: Hud.title(Hud.text, size: size * 0.2)),
              Text(caption, style: Hud.label(palette.accent, size: 8)),
            ]),
          ),
        ),
      );
}

class _RingPainter extends CustomPainter {
  _RingPainter(this.value, this.color);

  final double value;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final r = rect.deflate(6);
    canvas.drawArc(r, 0, math.pi * 2, false,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 4
          ..color = color.withValues(alpha: 0.15));
    // Tick marks around the dial.
    final tick = Paint()
      ..color = color.withValues(alpha: 0.35)
      ..strokeWidth = 1;
    final c = rect.center;
    for (var i = 0; i < 60; i++) {
      final a = i / 60 * math.pi * 2;
      final o = size.width / 2;
      canvas.drawLine(c + Offset(math.cos(a), math.sin(a)) * (o - 1), c + Offset(math.cos(a), math.sin(a)) * (o - (i % 5 == 0 ? 5 : 3)), tick);
    }
    final arc = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round
      ..color = color;
    canvas.drawArc(r, -math.pi / 2, math.pi * 2 * value, false,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 10
          ..color = color.withValues(alpha: 0.25)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6));
    canvas.drawArc(r, -math.pi / 2, math.pi * 2 * value, false, arc);
  }

  @override
  bool shouldRepaint(_RingPainter old) => old.value != value || old.color != color;
}

/// A sweeping radar arm, used as an idle/"listening" animation.
class ScanSweep extends StatefulWidget {
  const ScanSweep({super.key, required this.palette, this.size = 140});

  final SightPalette palette;
  final double size;

  @override
  State<ScanSweep> createState() => _ScanSweepState();
}

class _ScanSweepState extends State<ScanSweep> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this, duration: const Duration(seconds: 2))..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
        animation: _c,
        builder: (_, _) => CustomPaint(size: Size.square(widget.size), painter: _SweepPainter(_c.value, widget.palette.accent)),
      );
}

class _SweepPainter extends CustomPainter {
  _SweepPainter(this.t, this.color);

  final double t;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    final r = size.width / 2;
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..color = color.withValues(alpha: 0.3);
    for (final f in [0.33, 0.66, 1.0]) {
      canvas.drawCircle(c, r * f, ring);
    }
    final a = t * math.pi * 2;
    canvas.drawCircle(
        c,
        r,
        Paint()
          ..shader = SweepGradient(
            startAngle: 0,
            endAngle: math.pi * 2,
            colors: [Colors.transparent, color.withValues(alpha: 0.5)],
            stops: const [0.75, 1.0],
            transform: GradientRotation(a - math.pi * 2),
          ).createShader(Rect.fromCircle(center: c, radius: r)));
    canvas.drawLine(c, c + Offset(math.cos(a), math.sin(a)) * r, Paint()..color = color..strokeWidth = 1.5);
  }

  @override
  bool shouldRepaint(_SweepPainter old) => old.t != t;
}
