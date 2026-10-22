import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';

/// Streams the back camera and hands out upright, centre-cropped 256x256 RGB
/// frames for the native MiDaS teacher, at most [maxFps] times a second.
class CameraTeacher {
  CameraTeacher({this.maxFps = 4});

  static const int side = 256;
  final double maxFps;

  CameraController? controller;
  bool _busy = false;
  DateTime _last = DateTime.fromMillisecondsSinceEpoch(0);

  Future<void> start(Future<void> Function(Uint8List rgb) onFrame) async {
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

  /// YUV420 -> upright RGB, centre-cropped square, nearest-neighbour resampled.
  static Uint8List toRgb(CameraImage img, int rotation) {
    final w = img.width, h = img.height;
    final yP = img.planes[0], uP = img.planes[1], vP = img.planes[2];
    final uPixel = uP.bytesPerPixel ?? 1, vPixel = vP.bytesPerPixel ?? 1;
    final upW = rotation % 180 == 0 ? w : h;
    final upH = rotation % 180 == 0 ? h : w;
    final crop = math.min(upW, upH);
    final ox = (upW - crop) ~/ 2, oy = (upH - crop) ~/ 2;
    final out = Uint8List(side * side * 3);
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
        final u = uP.bytes[(sy >> 1) * uP.bytesPerRow + (sx >> 1) * uPixel] - 128;
        final v = vP.bytes[(sy >> 1) * vP.bytesPerRow + (sx >> 1) * vPixel] - 128;
        out[o++] = (yy + 1.402 * v).round().clamp(0, 255);
        out[o++] = (yy - 0.344 * u - 0.714 * v).round().clamp(0, 255);
        out[o++] = (yy + 1.772 * u).round().clamp(0, 255);
      }
    }
    return out;
  }
}
