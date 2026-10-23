import 'dart:typed_data';

import '../sonar/sonar_channel.dart';

/// Most of the SEE picture the camera may ever provide. The echo AI always contributes the
/// rest, so camera assist sharpens the echo view instead of replacing it.
const double kMaxCameraShare = 0.8;

/// Softens the camera's depth so it reads like a very well-trained echo sense rather than a
/// photo: a slight blur (echoes don't resolve fine texture).
Float32List softenCameraDepth(Float32List t) {
  const n = kDepthSize;
  final out = Float32List(t.length);
  for (var y = 0; y < n; y++) {
    for (var x = 0; x < n; x++) {
      var sum = 0.0, w = 0.0;
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          final yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= n || xx < 0 || xx >= n) continue;
          final k = (dx == 0 && dy == 0) ? 2.0 : 1.0;
          sum += t[yy * n + xx] * k;
          w += k;
        }
      }
      out[y * n + x] = sum / w;
    }
  }
  return out;
}

/// The SEE picture is always the echo view. Where the camera can see ([level] 0..1), its
/// (softened) depth pulls the echo AI's map toward the truth: strongly up close, weakly far
/// away (like real echolocation, distant things stay vague) and never more than
/// kMaxCameraShare, so it never looks like a filtered photo. (Per-pixel noise was tried and
/// dropped: the renderer's edge detector turns it into crawling texture.)
Float32List fuseEchoCamera(Float32List echo, Float32List? camera, double level) {
  final a = level.clamp(0.0, 1.0) * kMaxCameraShare;
  if (a <= 0 || camera == null) return echo;
  final out = Float32List(echo.length);
  for (var i = 0; i < echo.length; i++) {
    final w = a * (0.5 + 0.5 * camera[i]); // camera = nearness: near -> trusted more
    out[i] = (w * camera[i] + (1 - w) * echo[i]).clamp(0.0, 1.0);
  }
  return out;
}
