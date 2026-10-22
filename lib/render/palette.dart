import 'package:flutter/painting.dart';

/// Colour ramp black -> [mid] -> [hot] used by every view.
class SightPalette {
  const SightPalette(this.name, this.mid, this.hot, this.accent);

  final String name;
  final Color mid;
  final Color hot;
  final Color accent;

  /// Daredevil's "world on fire": crimson outlines on black.
  static const radar = SightPalette('RADAR', Color(0xFFB0002E), Color(0xFFFFB3C4), Color(0xFFFF2457));

  /// The Dark Knight's sonar phones: cold electric blue.
  static const sonar = SightPalette('SONAR', Color(0xFF1546C8), Color(0xFFCFE6FF), Color(0xFF5AA8FF));

  /// Writes the colour for intensity [i] in [0, 1] into [out] at [o] as RGBA.
  void write(double i, List<int> out, int o) {
    double r, g, b;
    if (i < 0.6) {
      final t = i / 0.6;
      r = mid.r * t;
      g = mid.g * t;
      b = mid.b * t;
    } else {
      final t = (i - 0.6) / 0.4;
      r = mid.r + (hot.r - mid.r) * t;
      g = mid.g + (hot.g - mid.g) * t;
      b = mid.b + (hot.b - mid.b) * t;
    }
    out[o] = (r * 255).round();
    out[o + 1] = (g * 255).round();
    out[o + 2] = (b * 255).round();
    out[o + 3] = 255;
  }
}
