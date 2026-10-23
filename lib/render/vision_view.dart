import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import 'palette.dart';
import 'vision_renderer.dart';

/// Animated full-bleed view of a [VisionRenderer] with a bloom pass on top.
class VisionView extends StatefulWidget {
  const VisionView({super.key, required this.renderer, required this.palette, this.placeholder});

  final VisionRenderer renderer;
  final SightPalette palette;
  final String? placeholder;

  @override
  State<VisionView> createState() => _VisionViewState();
}

class _VisionViewState extends State<VisionView> with SingleTickerProviderStateMixin {
  late final Ticker _ticker;
  ui.Image? _image;
  bool _rendering = false;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_tick)..start();
  }

  Future<void> _tick(Duration elapsed) async {
    if (_rendering || !widget.renderer.hasMap) return;
    _rendering = true;
    final ui.Image img;
    try {
      img = await widget.renderer.render(elapsed.inMicroseconds / 1e6, widget.palette);
    } catch (e) {
      // A failed frame must not freeze the view on its placeholder: skip it, try the next.
      debugPrint('VisionView: frame failed: $e');
      return;
    } finally {
      _rendering = false;
    }
    if (!mounted) {
      img.dispose();
      return;
    }
    setState(() {
      _image?.dispose();
      _image = img;
    });
  }

  @override
  void dispose() {
    _ticker.dispose();
    _image?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final img = _image;
    if (img == null) {
      return Center(
        child: Text(widget.placeholder ?? 'LISTENING…',
            textAlign: TextAlign.center,
            style: TextStyle(color: widget.palette.accent, letterSpacing: 4, fontSize: 12)),
      );
    }
    return CustomPaint(painter: _VisionPainter(img, widget.palette), size: Size.infinite);
  }
}

class _VisionPainter extends CustomPainter {
  _VisionPainter(this.image, this.palette);

  final ui.Image image;
  final SightPalette palette;

  @override
  void paint(Canvas canvas, Size size) {
    // The map is square (centre crop of the camera's view), so fit it to the width.
    final side = size.width;
    final dst = Rect.fromLTWH(0, (size.height - side) / 2, side, side);
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    canvas.drawRect(Offset.zero & size, Paint()..color = Colors.black);
    canvas.drawImageRect(image, src, dst, Paint()..filterQuality = FilterQuality.medium);
    // Bloom: a blurred additive copy makes thin lines glow like the films.
    canvas.drawImageRect(
      image,
      src,
      dst,
      Paint()
        ..filterQuality = FilterQuality.low
        ..blendMode = BlendMode.plus
        ..imageFilter = ui.ImageFilter.blur(sigmaX: 9, sigmaY: 9),
    );
    // Horizon ticks, as in the reference shot.
    final hud = Paint()
      ..color = palette.accent.withValues(alpha: 0.8)
      ..strokeWidth = 1.5;
    final hy = dst.top + side * 0.2;
    canvas.drawLine(Offset(0, hy), Offset(size.width * 0.1, hy), hud);
    canvas.drawLine(Offset(size.width * 0.9, hy), Offset(size.width, hy), hud);
  }

  @override
  bool shouldRepaint(_VisionPainter old) => old.image != image || old.palette != palette;
}
