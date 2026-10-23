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
    final p = f.clean;
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

/// Distance markers, in feet.
const _ticks = [3, 6, 9, 12, 15];

/// Leftover speaker sound may remain this close even after cancellation.
const double kSpeakerZoneFeet = 0.5;

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
    final label = TextStyle(color: palette.accent, fontSize: 10, letterSpacing: 2, fontFamily: 'monospace');
    final traceBottom = size.height * 0.26;
    final split = size.height * 0.46;

    // 1. Echo trace ("A-scan"): echo strength against distance for the latest ping.
    //    The speaker's own sound is cancelled natively; what's left near 0 is residue,
    //    shaded as the SPEAKER zone. Every peak beyond it is something reflecting.
    _text(canvas, 'ECHO TRACE', const Offset(0, 0), label);
    final trace = Path();
    final fill = Path()..moveTo(0, traceBottom);
    for (var b = 0; b < kRangeBins; b++) {
      final v = (current[b * 2] + current[b * 2 + 1]) / 2;
      final x = b / (kRangeBins - 1) * size.width;
      final y = traceBottom - v * (traceBottom - 16);
      b == 0 ? trace.moveTo(x, y) : trace.lineTo(x, y);
      fill.lineTo(x, y);
    }
    fill
      ..lineTo(size.width, traceBottom)
      ..close();
    canvas.drawPath(fill, Paint()..color = palette.accent.withValues(alpha: 0.12));
    final deadX = kSpeakerZoneFeet / kMaxFeet * size.width;
    canvas.drawRect(Rect.fromLTRB(0, 14, deadX, traceBottom), Paint()..color = Colors.white.withValues(alpha: 0.06));
    _text(canvas, 'SPKR', Offset(2, traceBottom - 12), label.copyWith(fontSize: 8, color: Colors.white38));
    canvas.drawPath(
        trace,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 4
          ..color = palette.accent.withValues(alpha: 0.35)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4));
    canvas.drawPath(
        trace,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5
          ..color = palette.hot);

    // 2. Waterfall: the same trace over time, newest on top.
    final wfTop = traceBottom + 22;
    _text(canvas, 'ECHO HISTORY', Offset(0, traceBottom + 6), label);
    final img = waterfall;
    if (img != null) {
      canvas.drawImageRect(
        img,
        Rect.fromLTWH(0, 0, img.width.toDouble(), img.height.toDouble()),
        Rect.fromLTWH(0, wfTop, size.width, split - wfTop),
        Paint()..filterQuality = FilterQuality.low,
      );
    }
    for (final ft in _ticks) {
      final x = ft / kMaxFeet * size.width;
      _text(canvas, '$ft ft', Offset(x + 2, split - 14), label);
      final tick = Paint()..color = palette.accent.withValues(alpha: 0.35);
      canvas.drawLine(Offset(x, 14), Offset(x, traceBottom), tick);
      canvas.drawLine(Offset(x, split - 16), Offset(x, split), Paint()..color = palette.accent);
    }

    // Radar fan from the phone (bottom centre) outwards.
    final origin = Offset(size.width / 2, size.height - 24);
    final maxR = math.min(size.width / 2, size.height - split - 40);
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = maxR / kRangeBins * 1.6
      ..strokeCap = StrokeCap.round;
    // Skip the speaker zone: leftover speaker sound, not echoes.
    for (var b = (kSpeakerZoneFeet / kFeetPerBin).ceil(); b < kRangeBins; b++) {
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
    for (final ft in _ticks) {
      final radius = ft / kMaxFeet * maxR;
      canvas.drawArc(Rect.fromCircle(center: origin, radius: radius), -math.pi, math.pi, false, grid);
      _text(canvas, '$ft', origin + Offset(radius + 2, -12), label);
    }
  }

  void _text(Canvas canvas, String s, Offset at, TextStyle style) {
    final tp = TextPainter(text: TextSpan(text: s, style: style), textDirection: TextDirection.ltr)..layout();
    tp.paint(canvas, at);
  }

  @override
  bool shouldRepaint(_ScopePainter old) => true;
}
