import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';

/// One camera frame for the teacher: upright 256x256 RGB plus how bright the scene was.
class TeacherFrame {
  const TeacherFrame(this.rgb, this.luma);

  final Uint8List rgb;

  /// Mean brightness 0..255 *before* the low-light boost (how much the camera really sees).
  final double luma;
}

/// Streams the back camera and hands out upright, centre-cropped 256x256 RGB frames for
/// the native MiDaS teacher. Frames are taken as fast as the teacher keeps up (one in
/// flight), capped at [maxFps].
class CameraTeacher {
  CameraTeacher({this.maxFps = 12});

  static const int side = 256;
  final double maxFps;

  CameraController? controller;
  bool _busy = false;
  DateTime _last = DateTime.fromMillisecondsSinceEpoch(0);

  bool get running => controller != null;

  Future<void> start(Future<void> Function(TeacherFrame frame) onFrame) async {
    if (controller != null) return;
    final cams = await availableCameras();
    final back = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cams.first);
    final c = CameraController(back, ResolutionPreset.low,
        enableAudio: false, imageFormatGroup: ImageFormatGroup.yuv420);
    await c.initialize();
    controller = c;
    await c.startImageStream((img) async {
      final now = DateTime.now();
      if (_busy || now.difference(_last).inMicroseconds < 1e6 / maxFps) return;
      _busy = true;
      _last = now;
      try {
        await onFrame(toRgb(img, back.sensorOrientation));
      } catch (_) {
        // A dropped label is harmless; the next frame will try again.
      } finally {
        _busy = false;
      }
    });
  }

  Future<void> stop() async {
    final c = controller;
    controller = null;
    if (c == null) return;
    if (c.value.isStreamingImages) await c.stopImageStream();
    await c.dispose();
  }

  /// YUV420 -> upright RGB, centre-cropped square, nearest-neighbour resampled, with a
  /// contrast stretch in dim light so the depth teacher still has something to work with.
  static TeacherFrame toRgb(CameraImage img, int rotation) {
    final w = img.width, h = img.height;
    final yP = img.planes[0], uP = img.planes[1], vP = img.planes[2];
    final uPixel = uP.bytesPerPixel ?? 1, vPixel = vP.bytesPerPixel ?? 1;
    final upW = rotation % 180 == 0 ? w : h;
    final upH = rotation % 180 == 0 ? h : w;
    final crop = math.min(upW, upH);
    final ox = (upW - crop) ~/ 2, oy = (upH - crop) ~/ 2;
    final out = Uint8List(side * side * 3);
    final hist = List<int>.filled(256, 0);
    var o = 0;
    for (var y = 0; y < side; y++) {
      final uy = oy + y * crop ~/ side;
      for (var x = 0; x < side; x++) {
        final ux = ox + x * crop ~/ side;
        // Map the upright pixel back to sensor coordinates.
        int sx, sy;
        switch (rotation) {
          case 90:
            sx = uy;
            sy = h - 1 - ux;
          case 180:
            sx = w - 1 - ux;
            sy = h - 1 - uy;
          case 270:
            sx = w - 1 - uy;
            sy = ux;
          default:
            sx = ux;
            sy = uy;
        }
        final yy = yP.bytes[sy * yP.bytesPerRow + sx];
        hist[yy]++;
        final u = uP.bytes[(sy >> 1) * uP.bytesPerRow + (sx >> 1) * uPixel] - 128;
        final v = vP.bytes[(sy >> 1) * vP.bytesPerRow + (sx >> 1) * vPixel] - 128;
        out[o++] = (yy + 1.402 * v).round().clamp(0, 255);
        out[o++] = (yy - 0.344 * u - 0.714 * v).round().clamp(0, 255);
        out[o++] = (yy + 1.772 * u).round().clamp(0, 255);
      }
    }
    // Mean and 2nd/98th percentile brightness.
    var sum = 0, acc = 0, lo = 0, hi = 255;
    const n = side * side;
    for (var v = 0; v < 256; v++) {
      sum += v * hist[v];
    }
    for (var v = 0; v < 256; v++) {
      acc += hist[v];
      if (acc >= n * 0.02) {
        lo = v;
        break;
      }
    }
    acc = 0;
    for (var v = 255; v >= 0; v--) {
      acc += hist[v];
      if (acc >= n * 0.02) {
        hi = v;
        break;
      }
    }
    final mean = sum / n;
    // Dim scene: stretch the used brightness range to full scale (noise rises too, which is
    // why very dark frames aren't used for learning; see home_page.dart).
    if (mean < 90 && hi - lo > 8) {
      final g = 255 / (hi - lo);
      for (var i = 0; i < out.length; i++) {
        out[i] = ((out[i] - lo) * g).round().clamp(0, 255);
      }
    }
    return TeacherFrame(out, mean);
  }
}
