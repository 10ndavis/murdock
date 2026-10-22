import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'render/palette.dart';
import 'render/scope_view.dart';
import 'render/vision_renderer.dart';
import 'render/vision_view.dart';
import 'sonar/sonar_channel.dart';
import 'train/camera_teacher.dart';

enum Mode { vision, scope, train }

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with WidgetsBindingObserver {
  final _sonar = SonarChannel();
  final _vision = VisionRenderer();
  final _teacherView = VisionRenderer();
  final _camera = CameraTeacher();
  StreamSubscription<SonarFrame>? _sub;

  Mode _mode = Mode.vision;
  SightPalette _palette = SightPalette.radar;
  SonarStatus? _status;
  SonarFrame? _last;
  String? _error;
  double? _download;
  int _pings = 0;
  double _pingRate = 0;
  Timer? _rateTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _sub = _sonar.frames.listen((f) {
      _pings++;
      _last = f;
      final d = f.depth;
      if (d != null) _vision.update(d);
    });
    _sonar.downloadProgress.listen((p) => setState(() => _download = p));
    _rateTimer = Timer.periodic(const Duration(seconds: 1), (_) async {
      _pingRate = _pings.toDouble();
      _pings = 0;
      if (!mounted) return;
      _status = await _sonar.status();
      if (mounted) setState(() {});
    });
    _startSonar();
  }

  Future<void> _startSonar() async {
    if (!await Permission.microphone.request().isGranted) {
      setState(() => _error = 'Microphone permission is needed: the mic is the sensor.');
      return;
    }
    await WakelockPlus.enable();
    await _sonar.start();
    setState(() => _error = null);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Native side stops the speaker on pause; bring it back on resume.
    if (state == AppLifecycleState.resumed) {
      _startSonar();
      if (_mode == Mode.train) _startTraining();
    } else if (state == AppLifecycleState.paused) {
      _stopTraining();
    }
  }

  Future<void> _setMode(Mode m) async {
    if (m == _mode) return;
    if (_mode == Mode.train) await _stopTraining();
    setState(() => _mode = m);
    if (m == Mode.train) await _startTraining();
  }

  Future<void> _startTraining() async {
    if (!await _sonar.teacherReady()) return; // UI offers the download.
    if (!await Permission.camera.request().isGranted) {
      setState(() => _error = 'Camera permission is needed to teach the echo model.');
      return;
    }
    await _sonar.setTraining(true);
    await _camera.start((Uint8List rgb) async {
      final depth = await _sonar.teach(rgb);
      _teacherView.update(depth);
    });
    if (mounted) setState(() {});
  }

  Future<void> _stopTraining() async {
    await _sonar.setTraining(false);
    await _camera.stop();
    await _sonar.save();
    if (mounted) setState(() {});
  }

  Future<void> _downloadTeacher() async {
    setState(() => _download = 0);
    try {
      await _sonar.downloadTeacher();
      await _startTraining();
    } catch (e) {
      setState(() => _error = 'Download failed: $e');
    }
    setState(() => _download = null);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _sub?.cancel();
    _rateTimer?.cancel();
    _camera.stop();
    _sonar.stop();
    WakelockPlus.disable();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final accent = _palette.accent;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _topBar(accent),
            Expanded(child: _body()),
            _statusLine(accent),
          ],
        ),
      ),
    );
  }

  Widget _topBar(Color accent) => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        child: Row(
          children: [
            for (final m in Mode.values)
              TextButton(
                onPressed: () => _setMode(m),
                child: Text(m.name.toUpperCase(),
                    style: TextStyle(
                      color: m == _mode ? accent : accent.withValues(alpha: 0.4),
                      letterSpacing: 3,
                      fontWeight: m == _mode ? FontWeight.bold : FontWeight.normal,
                    )),
              ),
            const Spacer(),
            IconButton(
              tooltip: 'Switch look',
              icon: Icon(Icons.blur_on, color: accent),
              onPressed: () => setState(() =>
                  _palette = _palette == SightPalette.radar ? SightPalette.sonar : SightPalette.radar),
            ),
          ],
        ),
      );

  Widget _body() {
    final err = _error;
    if (err != null) return _message(err);
    switch (_mode) {
      case Mode.vision:
        final untrained = (_status?.steps ?? 0) == 0;
        return Stack(children: [
          VisionView(
            renderer: _vision,
            palette: _palette,
            placeholder: 'LISTENING…\nTurn media volume up.',
          ),
          if (untrained)
            Positioned(
              left: 16,
              right: 16,
              bottom: 8,
              child: Text(
                'Model not trained on this phone yet: the picture is noise until you use TRAIN '
                'with the lights on (or load a model trained on a PC). SCOPE shows the raw echoes.',
                textAlign: TextAlign.center,
                style: TextStyle(color: _palette.accent.withValues(alpha: 0.7), fontSize: 11),
              ),
            ),
        ]);
      case Mode.scope:
        return ScopeView(frames: _sonar.frames, palette: _palette);
      case Mode.train:
        return _trainBody();
    }
  }

  Widget _trainBody() {
    final status = _status;
    final cam = _camera.controller;
    if (status != null && !status.teacherReady) {
      final p = _download;
      return _message(
        'Training uses the camera as a teacher: a depth model (MiDaS v2.1 small, ~66 MB) '
        'labels what the camera sees, and the echo model learns to predict that from sound.',
        action: p == null
            ? OutlinedButton(onPressed: _downloadTeacher, child: const Text('DOWNLOAD DEPTH TEACHER'))
            : LinearProgressIndicator(value: p, color: _palette.accent),
      );
    }
    return Column(
      children: [
        Expanded(
          child: Row(children: [
            Expanded(child: _panel('CAMERA', cam != null && cam.value.isInitialized
                ? ClipRect(child: AspectRatio(aspectRatio: 1, child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: cam.value.previewSize?.height ?? 1,
                      height: cam.value.previewSize?.width ?? 1,
                      child: CameraPreview(cam)))))
                : const SizedBox())),
            Expanded(child: _panel('TEACHER (CAMERA DEPTH)',
                VisionView(renderer: _teacherView, palette: _palette, placeholder: 'WAITING…'))),
          ]),
        ),
        Expanded(child: _panel('STUDENT (ECHO ONLY)', VisionView(renderer: _vision, palette: _palette))),
        Padding(
          padding: const EdgeInsets.all(8),
          child: Text(
            'Walk slowly around a lit room pointing the phone at walls, furniture and people. '
            'When the student panel resembles the teacher, switch the lights off and use VISION.\n'
            'Samples also saved for PC training: ${status?.samplesDir ?? ''}',
            style: TextStyle(color: _palette.accent.withValues(alpha: 0.7), fontSize: 11),
          ),
        ),
        Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
          TextButton(onPressed: _sonar.save, child: const Text('SAVE BRAIN')),
          TextButton(
            onPressed: () async {
              await _sonar.reset();
              _vision.update(Float32List(kDepthSize * kDepthSize));
            },
            child: const Text('RESET BRAIN'),
          ),
        ]),
      ],
    );
  }

  Widget _panel(String title, Widget child) => Container(
        margin: const EdgeInsets.all(4),
        decoration: BoxDecoration(border: Border.all(color: _palette.accent.withValues(alpha: 0.35))),
        child: Stack(children: [
          Positioned.fill(child: child),
          Positioned(
            left: 6,
            top: 4,
            child: Text(title, style: TextStyle(color: _palette.accent, fontSize: 10, letterSpacing: 2)),
          ),
        ]),
      );

  Widget _message(String text, {Widget? action}) => Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Text(text, textAlign: TextAlign.center, style: TextStyle(color: _palette.accent, height: 1.5)),
            if (action != null) ...[const SizedBox(height: 20), action],
          ]),
        ),
      );

  Widget _statusLine(Color accent) {
    final s = _status;
    final f = _last;
    final loss = f?.loss ?? double.nan;
    final parts = [
      '${_pingRate.toStringAsFixed(0)} PING/S',
      if (f != null) 'DIRECT ${f.direct.toStringAsFixed(2)}',
      if (s != null) s.stereo ? 'STEREO' : 'MONO',
      if (s != null && s.source.isNotEmpty) s.source.toUpperCase(),
      'STEPS ${f?.steps ?? s?.steps ?? 0}',
      if (!loss.isNaN) 'LOSS ${loss.toStringAsFixed(3)}',
      'SAMPLES ${f?.samples ?? s?.samples ?? 0}',
    ];
    return Padding(
      padding: const EdgeInsets.all(6),
      child: Text(parts.join('  ·  '),
          textAlign: TextAlign.center,
          style: TextStyle(color: accent.withValues(alpha: 0.75), fontSize: 10, letterSpacing: 1)),
    );
  }
}
