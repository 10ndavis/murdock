import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import '../sonar/sonar_channel.dart';
import 'palette.dart';

/// Turns a coarse nearness map into the Daredevil / Dark Knight look:
/// glowing object edges, iso-depth contour lines, a sonar wavefront sweeping
/// outwards with each pulse, and sparks where edges are strong.
class VisionRenderer {
  static const int size = 112;

  final Float32List _smooth = Float32List(kDepthSize * kDepthSize);
  final Float32List _up = Float32List(size * size);
  final Uint8List _pixels = Uint8List(size * size * 4);
  // "Sonar painting": each surface lights up as a ping's wavefront reaches it and then
  // fades, so the scene is continuously re-painted by sound.
  final Float32List _glow = Float32List(size * size);
  double _lastSeconds = 0;
  final math.Random _rng = math.Random(7);
  bool _hasMap = false;

  /// Feed a new model output; smoothed over time because single pings are noisy.
  void update(Float32List depth) {
    for (var i = 0; i < _smooth.length; i++) {
      // Never let an invalid value in: one NaN would poison the smoothed map for good.
      final d = depth[i].isFinite ? depth[i].clamp(0.0, 1.0) : 0.0;
      _smooth[i] = _hasMap ? _smooth[i] * 0.6 + d * 0.4 : d;
    }
    _hasMap = true;
  }

  bool get hasMap => _hasMap;

  Future<ui.Image> render(double seconds, SightPalette palette) {
    _upsample();
    // The wavefront travels from near (1) to far (0) once per 1.4 s.
    final wave = 1.0 - (seconds % 1.4) / 1.4;
    final dt = (seconds - _lastSeconds).clamp(0.0, 0.2);
    _lastSeconds = seconds;
    final fade = math.exp(-dt / 0.55);
    const n = size;
    for (var y = 0; y < n; y++) {
      for (var x = 0; x < n; x++) {
        final i = y * n + x;
        final v = _up[i];
        final gx = _up[y * n + math.min(x + 1, n - 1)] - _up[y * n + math.max(x - 1, 0)];
        final gy = _up[math.min(y + 1, n - 1) * n + x] - _up[math.max(y - 1, 0) * n + x];
        final edge = math.min(1.0, math.sqrt(gx * gx + gy * gy) * 10);

        final bands = v * 12;
        final f = bands - bands.floorToDouble();
        final d = math.min(f, 1 - f) / 0.07;
        final contour = math.exp(-d * d);

        final w = (v - wave) / 0.045;
        final pulse = math.exp(-w * w) * (0.25 + 0.75 * v);
        final glow = math.max(_glow[i] * fade, pulse);
        _glow[i] = glow;

        // Edges and contours are revealed mostly while the glow from the last ping lasts.
        final lit = 0.45 + 0.55 * glow;
        var intensity = (edge * 1.1 + contour * 0.28 * v) * lit + glow * 0.6 + pulse * 0.35 + 0.03 * v;
        if (edge > 0.35 && _rng.nextDouble() < 0.025) intensity += 0.9;
        palette.write(intensity.clamp(0.0, 1.0), _pixels, i * 4);
      }
    }
    final done = Completer<ui.Image>();
    ui.decodeImageFromPixels(_pixels, n, n, ui.PixelFormat.rgba8888, done.complete);
    return done.future;
  }

  void _upsample() {
    const s = kDepthSize;
    const scale = s / size;
    for (var y = 0; y < size; y++) {
      final fy = ((y + 0.5) * scale - 0.5).clamp(0.0, s - 1.0);
      final y0 = fy.floor();
      final y1 = math.min(y0 + 1, s - 1);
      final ty = fy - y0;
      for (var x = 0; x < size; x++) {
        final fx = ((x + 0.5) * scale - 0.5).clamp(0.0, s - 1.0);
        final x0 = fx.floor();
        final x1 = math.min(x0 + 1, s - 1);
        final tx = fx - x0;
        final top = _smooth[y0 * s + x0] * (1 - tx) + _smooth[y0 * s + x1] * tx;
        final bot = _smooth[y1 * s + x0] * (1 - tx) + _smooth[y1 * s + x1] * tx;
        _up[y * size + x] = top * (1 - ty) + bot * ty;
      }
    }
  }
}
