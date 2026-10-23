import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../sonar/sonar_channel.dart';
import 'palette.dart';

/// Raw sonar, no AI: what the phone actually hears.
///  * Top: echogram waterfall (time down, distance across).
///  * Bottom: a radar fan. Ring radius = echo distance; brightness = echo strength,
///    with static clutter subtracted so moving things pop. When the phone records in
///    stereo, the left/right level difference skews each echo towards its side.
class ScopeView extends StatefulWidget {
  const ScopeView({super.key, required this.frames, required this.palette});

  final Stream<SonarFrame> frames;
  final SightPalette palette;

  @override
  State<ScopeView> createState() => _ScopeViewState();
}

class _ScopeViewState extends State<ScopeView> {
  static const _rows = 160;
  StreamSubscription<SonarFrame>? _sub;
  final Float32List _background = Float32List(kRangeBins * kChannels);
  final Float32List _current = Float32List(kRangeBins * kChannels);
  final Uint8List _waterfall = Uint8List(kRangeBins * _rows * 4);
  ui.Image? _image;
  bool _decoding = false;
  bool _primed = false;

  @override
  void initState() {
    super.initState();
    _sub = widget.frames.listen(_onFrame);
  }

  void _onFrame(SonarFrame f) {
    final p = f.profile;
    for (var i = 0; i < p.length; i++) {
      _background[i] = _primed ? _background[i] * 0.97 + p[i] * 0.03 : p[i];
      _current[i] = p[i];
    }
    _primed = true;
    // Scroll the waterfall down one row and paint the newest ping on top.
    _waterfall.setRange(kRangeBins * 4, _waterfall.length, _waterfall);
    for (var b = 0; b < kRangeBins; b++) {
      final v = (p[b * 2] + p[b * 2 + 1]) / 2;
      final moving = math.max(0.0, v - (_background[b * 2] + _background[b * 2 + 1]) / 2);
      final level = (math.pow(v, 3) * 0.8 + moving * 4).clamp(0.0, 1.0).toDouble();
      widget.palette.write(level, _waterfall, b * 4);
    }
    if (_decoding) return;
    _decoding = true;
    ui.decodeImageFromPixels(Uint8List.fromList(_waterfall), kRangeBins, _rows, ui.PixelFormat.rgba8888,
        (img) {
      _decoding = false;
      if (!mounted) {
        img.dispose();
        return;
      }
      setState(() {
        _image?.dispose();
        _image = img;
      });
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    _image?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => CustomPaint(
        painter: _ScopePainter(_image, _current, _background, widget.palette),
        size: Size.infinite,
      );
}

class _ScopePainter extends CustomPainter {
  _ScopePainter(this.waterfall, Float32List current, Float32List background, this.palette)
      : current = Float32List.fromList(current),
        background = Float32List.fromList(background);

  final ui.Image? waterfall;
  final Float32List current;
  final Float32List background;
  final SightPalette palette;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = Colors.black);
    final split = size.height * 0.38;
    final label = TextStyle(color: palette.accent, fontSize: 10, letterSpacing: 2, fontFamily: 'monospace');

    final img = waterfall;
    if (img != null) {
      canvas.drawImageRect(
        img,
        Rect.fromLTWH(0, 0, img.width.toDouble(), img.height.toDouble()),
        Rect.fromLTWH(0, 0, size.width, split),
        Paint()..filterQuality = FilterQuality.low,
      );
    }
    for (final m in [1, 2, 3, 4, 5]) {
      final x = m / (kRangeBins * kMetresPerBin) * size.width;
      _text(canvas, '${m}m', Offset(x + 2, split - 14), label);
      canvas.drawLine(Offset(x, split - 16), Offset(x, split), Paint()..color = palette.accent);
    }

    // Radar fan from the phone (bottom centre) outwards.
    final origin = Offset(size.width / 2, size.height - 24);
    final maxR = math.min(size.width / 2, size.height - split - 40);
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = maxR / kRangeBins * 1.6
      ..strokeCap = StrokeCap.round;
    // Skip the first bins: they are swamped by the direct speaker->mic sound.
    for (var b = 6; b < kRangeBins; b++) {
      final l = current[b * 2], r = current[b * 2 + 1];
      final v = (l + r) / 2;
      final moving = math.max(0.0, v - (background[b * 2] + background[b * 2 + 1]) / 2);
      final strength = (math.pow(v, 3) * 0.7 + moving * 5).clamp(0.0, 1.0).toDouble();
      if (strength < 0.04) continue;
      final skew = ((l - r) * 6).clamp(-0.7, 0.7);
      final centre = -math.pi / 2 + skew;
      const halfWidth = 0.55;
      final radius = b / kRangeBins * maxR;
      ring.color = Color.lerp(palette.mid, palette.hot, strength)!.withValues(alpha: strength);
      canvas.drawArc(Rect.fromCircle(center: origin, radius: radius), centre - halfWidth, halfWidth * 2, false, ring);
    }
    final grid = Paint()
      ..style = PaintingStyle.stroke
      ..color = palette.accent.withValues(alpha: 0.25);
    for (final m in [1, 2, 3, 4, 5]) {
      final radius = m / (kRangeBins * kMetresPerBin) * maxR;
      canvas.drawArc(Rect.fromCircle(center: origin, radius: radius), -math.pi, math.pi, false, grid);
      _text(canvas, '${m}m', origin + Offset(radius + 2, -12), label);
    }
  }

  void _text(Canvas canvas, String s, Offset at, TextStyle style) {
    final tp = TextPainter(text: TextSpan(text: s, style: style), textDirection: TextDirection.ltr)..layout();
    tp.paint(canvas, at);
  }

  @override
  bool shouldRepaint(_ScopePainter old) => true;
}
