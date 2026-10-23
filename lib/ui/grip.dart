import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../render/palette.dart';
import 'hud.dart';

/// Recommended grip: the app runs UPSIDE DOWN, so the speaker (normally the bottom edge) is
/// at the top. Hold it like you're reading it, top tilted away from you by about this much:
/// the screen faces you comfortably, the speaker aims ahead and up, the back camera faces the
/// room (angled slightly down).
/// Tilt sign (from MainActivity): positive = top (speaker end) away from you; 90 = flat.
const double kIdealTilt = 35;
const double kTiltTolerance = 15;

enum GripLevel { ok, warn, bad }

class GripAdvice {
  const GripAdvice(this.level, this.title, this.detail);

  final GripLevel level;
  final String title;
  final String detail;

  /// [tilt]/[roll] from the motion sensor; [directDb] = speaker loudness vs calibration;
  /// [taughtTilt] = average tilt while learning (only checked when [checkTaught]).
  static GripAdvice of({
    required double tilt,
    required double roll,
    required double directDb,
    double? taughtTilt,
    bool checkTaught = false,
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
          'The camera is aiming at the floor. Raise the top of the phone a bit.');
    }
    if (tilt < -10) {
      return const GripAdvice(GripLevel.bad, 'TOP LEANING TOWARD YOU',
          'The speaker is aiming at you. Tilt the top of the phone away from you, like reading it.');
    }
    if (checkTaught && taughtTilt != null && (tilt - taughtTilt).abs() > 15) {
      return GripAdvice(GripLevel.warn, 'NOT HOW YOU TAUGHT IT',
          'You taught it at ${taughtTilt.round()}°; now ${tilt.round()}°. Match the grip for a better picture.');
    }
    if ((tilt - kIdealTilt).abs() > kTiltTolerance) {
      return GripAdvice(GripLevel.warn, tilt < kIdealTilt ? 'TILT THE TOP AWAY A LITTLE' : 'RAISE THE TOP A LITTLE',
          'Aim for about ${kIdealTilt.round()}° so the top speaker points ahead. Now ${tilt.round()}°.');
    }
    return GripAdvice(GripLevel.ok, 'GRIP OK', '${tilt.round()}° · fingers clear of the top edge');
  }
}

Color gripColour(GripLevel l) => switch (l) {
      GripLevel.ok => const Color(0xFF3DFFB5),
      GripLevel.warn => const Color(0xFFFFC14D),
      GripLevel.bad => const Color(0xFFFF4D6A),
    };

/// Compact strip: side-view tilt gauge + what to do.
class GripBadge extends StatelessWidget {
  const GripBadge({super.key, required this.advice, required this.tilt, required this.palette});

  final GripAdvice advice;
  final double tilt;
  final SightPalette palette;

  @override
  Widget build(BuildContext context) {
    final c = gripColour(advice.level);
    return HudPanel(
      palette: palette,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(children: [
        SizedBox(width: 44, height: 44, child: CustomPaint(painter: TiltGaugePainter(tilt, c))),
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

/// The phone seen from the side (you are on the left, the room on the right), the green
/// target wedge, and the speaker's sound beam leaving the top edge.
class TiltGaugePainter extends CustomPainter {
  TiltGaugePainter(this.tilt, this.colour);

  final double tilt;
  final Color colour;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    final r = size.shortestSide / 2;
    double rad(double deg) => deg * math.pi / 180;
    // Target zone around the ideal tilt (angles measured from vertical, top toward you).
    final zone = Paint()..color = const Color(0xFF3DFFB5).withValues(alpha: 0.18);
    canvas.drawArc(Rect.fromCircle(center: c, radius: r), -math.pi / 2 + rad(kIdealTilt - kTiltTolerance),
        rad(2 * kTiltTolerance), true, zone);
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
  bool shouldRepaint(TiltGaugePainter old) => old.tilt != tilt || old.colour != colour;
}
