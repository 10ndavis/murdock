import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../render/palette.dart';
import 'hud.dart';

/// Grip. The app runs UPSIDE DOWN, so the speaker (normally the bottom edge) is at the top;
/// holding it like you're reading it aims the speaker ahead.
///
/// There is no single "right" angle: the phone's tilt and roll are inputs to the echo AI, so
/// it learns how the room sounds and looks at each angle it has been taught. What matters is
/// (1) physics: the speaker must not be covered or aimed at you/the floor, and (2) whether
/// it has learned at the current angle yet. kIdealTilt is only the suggested starting grip.
/// Tilt sign (from MainActivity): positive = top (speaker end) away from you; 90 = flat.
const double kIdealTilt = 35;

/// Coverage bins, matching EchoBrain.TILT_* (Kotlin).
const double kTiltFromDeg = -30;
const double kTiltBinDeg = 10;

/// Samples at an angle before it counts as "learned there".
const int kAngleLearned = 40;

enum GripLevel { ok, warn, bad }

class GripAdvice {
  const GripAdvice(this.level, this.title, this.detail);

  final GripLevel level;
  final String title;
  final String detail;

  /// [tilt]/[roll] from the motion sensor; [directDb] = speaker loudness vs calibration;
  /// [coverage] = samples learned per tilt bin (checked only when [checkCoverage]).
  static GripAdvice of({
    required double tilt,
    required double roll,
    required double directDb,
    List<int> coverage = const [],
    bool checkCoverage = false,
  }) {
    if (!directDb.isNaN && directDb < -8) {
      return const GripAdvice(GripLevel.bad, 'SPEAKER OR MIC COVERED?',
          'The speaker sounds much quieter than when calibrated. Keep fingers off the top edge.');
    }
    if (roll.abs() > 35) {
      return const GripAdvice(GripLevel.bad, 'HOLD IT UPRIGHT', 'Hold it in portrait (this app runs upside down).');
    }
    if (tilt > 75) {
      return const GripAdvice(GripLevel.bad, 'TOO FLAT',
          'The speaker aims at the ceiling and the camera at the floor. Raise the top a bit.');
    }
    if (tilt < -10) {
      return const GripAdvice(GripLevel.bad, 'SPEAKER AIMED AT YOU',
          'The top is leaning toward you. Tilt it away, like reading the phone.');
    }
    if (checkCoverage && coverage.isNotEmpty) {
      final here = coverage[_bin(tilt, coverage.length)];
      if (here < kAngleLearned) {
        var best = 0;
        for (var i = 1; i < coverage.length; i++) {
          if (coverage[i] > coverage[best]) best = i;
        }
        final bestDeg = (kTiltFromDeg + (best + 0.5) * kTiltBinDeg).round();
        final hint = coverage[best] >= kAngleLearned
            ? 'Keep camera assist on to teach it, or go back to about $bestDeg°, where it knows the room best.'
            : 'Keep camera assist on with the lights on to teach it.';
        return GripAdvice(GripLevel.warn, 'NEW ANGLE · ${tilt.round()}°',
            'Only $here frames learned at this angle so far, so the picture is less reliable. $hint');
      }
      return GripAdvice(GripLevel.ok, 'GRIP OK · LEARNED HERE', '${tilt.round()}° · $here frames learned at this angle');
    }
    return GripAdvice(GripLevel.ok, 'GRIP OK', '${tilt.round()}° · speaker clear and aimed ahead');
  }

  static int _bin(double tilt, int n) => ((tilt - kTiltFromDeg) / kTiltBinDeg).floor().clamp(0, n - 1);
}

Color gripColour(GripLevel l) => switch (l) {
      GripLevel.ok => const Color(0xFF3DFFB5),
      GripLevel.warn => const Color(0xFFFFC14D),
      GripLevel.bad => const Color(0xFFFF4D6A),
    };

/// Compact strip: side-view tilt gauge + what to do.
class GripBadge extends StatelessWidget {
  const GripBadge({super.key, required this.advice, required this.tilt, required this.palette, this.coverage = const []});

  final GripAdvice advice;
  final double tilt;
  final SightPalette palette;
  final List<int> coverage;

  @override
  Widget build(BuildContext context) {
    final c = gripColour(advice.level);
    return HudPanel(
      palette: palette,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(children: [
        SizedBox(width: 44, height: 44, child: CustomPaint(painter: TiltGaugePainter(tilt, c, coverage))),
        const SizedBox(width: 12),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
            Text(advice.title, style: Hud.label(c, size: 10)),
            const SizedBox(height: 2),
            Text(advice.detail, style: Hud.body.copyWith(fontSize: 11.5, height: 1.3)),
          ]),
        ),
      ]),
    );
  }
}

/// The phone seen from the side (you are on the left, the room on the right), the angles it
/// has learned (green, brighter = more), and the speaker's sound beam leaving the top edge.
class TiltGaugePainter extends CustomPainter {
  TiltGaugePainter(this.tilt, this.colour, [this.coverage = const []]);

  final double tilt;
  final Color colour;
  final List<int> coverage;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    final r = size.shortestSide / 2;
    double rad(double deg) => deg * math.pi / 180;
    // Learned angles; before any learning, the suggested starting grip.
    final bins = coverage.isEmpty || coverage.every((v) => v == 0)
        ? [for (var i = 0; i < 12; i++) ((kTiltFromDeg + (i + 0.5) * kTiltBinDeg) - kIdealTilt).abs() < 15 ? kAngleLearned : 0]
        : coverage;
    for (var i = 0; i < bins.length; i++) {
      if (bins[i] == 0) continue;
      final a = (bins[i] / kAngleLearned).clamp(0.15, 1.0);
      canvas.drawArc(Rect.fromCircle(center: c, radius: r), -math.pi / 2 + rad(kTiltFromDeg + i * kTiltBinDeg),
          rad(kTiltBinDeg), true, Paint()..color = const Color(0xFF3DFFB5).withValues(alpha: 0.28 * a));
    }
    canvas.save();
    canvas.translate(c.dx, c.dy);
    canvas.rotate(rad(tilt.clamp(-90, 90)));
    final body = RRect.fromRectAndRadius(Rect.fromCenter(center: Offset.zero, width: 7, height: r * 1.7), const Radius.circular(2));
    canvas.drawRRect(body, Paint()..color = colour);
    // Sound beam from the top edge (the speaker, since the app runs upside down).
    final beam = Paint()
      ..color = colour.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    for (final f in [0.35, 0.6]) {
      canvas.drawArc(Rect.fromCircle(center: Offset(0, -r * 0.85), radius: r * f), -math.pi / 2 - 0.5, 1.0, false, beam);
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(TiltGaugePainter old) => old.tilt != tilt || old.colour != colour || old.coverage != coverage;
}
