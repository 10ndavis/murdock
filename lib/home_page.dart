import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'render/fusion.dart';
import 'render/palette.dart';
import 'render/scope_view.dart';
import 'render/vision_renderer.dart';
import 'render/vision_view.dart';
import 'sonar/sonar_channel.dart';
import 'train/camera_teacher.dart';
import 'ui/grip.dart';
import 'ui/guide.dart';
import 'ui/hud.dart';
import 'ui/system_check.dart';

enum Mode { see, radar }

/// Camera-labelled samples that count as "fully taught" (learning continues past it).
/// Simulated on BatVision: 600 samples get 23% lower error, 1500 only 25%.
const int kTeachGoal = 600;

/// Camera brightness (mean luma 0..255, before the low-light boost):
/// below kAssistLuma the camera is ignored (sound only); the camera's share of the SEE
/// picture ramps up to full at kAssistFullLuma; frames at or above kLearnLuma also train the
/// echo AI (darker frames make the depth teacher guess, and bad labels would hurt).
const double kAssistLuma = 12;
const double kAssistFullLuma = 60;
const double kLearnLuma = 30;

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with WidgetsBindingObserver {
  final _sonar = SonarChannel();
  final _vision = VisionRenderer();
  final _camera = CameraTeacher();
  final List<StreamSubscription<Object?>> _subs = [];

  Mode _mode = Mode.see;
  SightPalette _palette = SightPalette.radar;
  SonarStatus? _status;
  String? _blocker;
  double? _download;
  double _signal = 0;
  int _samples = 0;
  SonarStats? _stats;
  String? _lastError;
  DateTime? _lastBrainAt;
  final DateTime _startedAt = DateTime.now();

  // Camera assist + continuous learning.
  bool _assist = true;
  bool _showCamera = false;
  bool _skipDownload = false;
  double _cameraLuma = 0;
  Float32List? _teacherMap;
  DateTime _teacherAt = DateTime.fromMillisecondsSinceEpoch(0);
  int _teacherMs = 0;
  String _teacherBackend = '';
  bool _lastFrameLearned = false;

  // Grip: live tilt from the motion sensor, and the average tilt while learning (saved).
  double _tilt = kIdealTilt;
  double _roll = 0;
  double? _taughtTilt;
  double _sessionTiltSum = 0;
  int _sessionTiltN = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _subs.add(_sonar.frames.listen((f) => _signal = _signal * 0.9 + f.signal * 0.1));
    _subs.add(_sonar.brain.listen((b) {
      _samples = b.samples;
      _lastBrainAt = DateTime.now();
      _vision.update(_fuse(b.depth));
    }));
    _subs.add(_sonar.pose.listen((p) => setState(() {
          _tilt = p.tilt;
          _roll = p.roll;
        })));
    _sonar.getNumber('taughtTilt3').then((t) => _taughtTilt = t);
    _sonar.getFlag('assistOff').then((off) => setState(() => _assist = !off));
    _subs.add(_sonar.stats.listen((s) => setState(() {
          _stats = s;
          if (s.pingsPerSec == 0) _signal = 0;
        })));
    _subs.add(_sonar.downloadProgress.listen((p) => setState(() => _download = p)));
    // Errors land in the System Check panel; only a new kind of error pops a message.
    _subs.add(_sonar.errors.listen((e) {
      if (e != _lastError) _toast(e);
      setState(() => _lastError = e);
    }));
    _subs.add(Stream.periodic(const Duration(milliseconds: 700)).listen((_) => _refresh()));
    _boot();
  }

  Future<void> _boot() async {
    if (!await _sonar.getFlag('guideSeen')) {
      await _openGuide();
      await _sonar.setFlag('guideSeen');
    }
    await _startSonar();
    final s = await _sonar.status();
    if (mounted) setState(() => _status = s);
    await _syncCamera();
  }

  Future<void> _openGuide() =>
      Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => GuidePage(palette: _palette)));

  Future<void> _refresh() async {
    if (!mounted) return;
    final s = await _sonar.status();
    if (mounted) setState(() => _status = s);
  }

  void _toast(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      backgroundColor: const Color(0xFF16080C),
      behavior: SnackBarBehavior.floating,
      content: Text(message, style: Hud.label(_palette.hot, size: 11)),
    ));
  }

  Future<void> _startSonar() async {
    if (!await _sonar.requestPermission('microphone')) {
      setState(() => _blocker = 'Murdock needs the microphone: it is the sensor that hears the echoes.\n\n'
          'Allow microphone access in Android settings, then reopen the app.');
      return;
    }
    // Keeping the screen on is a nicety; never let it block the sonar.
    await WakelockPlus.enable().catchError((Object _) {});
    await _sonar.start();
    if (mounted) setState(() => _blocker = null);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // The native side stops the speaker when the app is backgrounded; resume it here.
    if (state == AppLifecycleState.resumed) {
      _startSonar().then((_) => _syncCamera());
    } else if (state == AppLifecycleState.paused) {
      _stopCamera();
    }
  }

  Future<void> _setMode(Mode m) async {
    if (m == _mode) return;
    setState(() => _mode = m);
    await _syncCamera();
  }

  Future<void> _setAssist(bool on) async {
    setState(() {
      _assist = on;
      if (!on) _showCamera = false;
    });
    on ? await _sonar.clearFlag('assistOff') : await _sonar.setFlag('assistOff');
    await _syncCamera();
  }

  // ───────────────────────────── camera + learning ─────────────────────────────

  /// The camera runs in SEE while camera assist is on (it also feeds continuous learning).
  bool get _wantCamera => (_status?.teacherReady ?? false) && _mode == Mode.see && _assist;

  Future<void> _syncCamera() async {
    if (!mounted) return;
    if (_wantCamera && !_camera.running) {
      if (!await _sonar.requestPermission('camera')) {
        _toast('Camera permission is needed for camera assist and learning.');
        await _setAssist(false);
        return;
      }
      _sessionTiltSum = 0;
      _sessionTiltN = 0;
      try {
        await _camera.start(_onCameraFrame);
      } catch (e) {
        _toast('Camera failed: $e');
      }
    } else if (!_wantCamera && _camera.running) {
      await _stopCamera();
    }
    if (mounted) setState(() {});
  }

  Future<void> _stopCamera() async {
    await _camera.stop();
    _teacherMap = null;
    await _sonar.save();
    // Remember how the phone was held while learning, so SEE can ask for the same grip.
    if (_sessionTiltN > 25) {
      final session = _sessionTiltSum / _sessionTiltN;
      final prev = _taughtTilt;
      _taughtTilt = prev == null ? session : (prev + session) / 2;
      await _sonar.setNumber('taughtTilt3', _taughtTilt!);
      _sessionTiltN = 0;
    }
  }

  /// Every camera frame: if the camera can see, get its depth (to sharpen the echo view) and,
  /// if bright enough, turn it into a training sample. Learning never "finishes".
  Future<void> _onCameraFrame(TeacherFrame f) async {
    _cameraLuma = _cameraLuma * 0.7 + f.luma * 0.3;
    if (f.luma < kAssistLuma) {
      _lastFrameLearned = false;
      return; // too dark to help; save the GPU/battery
    }
    final learn = f.luma >= kLearnLuma;
    final r = await _sonar.teach(f.rgb, learn: learn);
    _teacherMap = softenCameraDepth(r.depth);
    _teacherAt = DateTime.now();
    _teacherMs = r.ms;
    _teacherBackend = r.backend;
    _lastFrameLearned = learn;
    if (learn) {
      _sessionTiltSum += _tilt;
      _sessionTiltN++;
    }
  }

  /// How much the camera can currently help, 0 (sound only) .. 1 (well lit).
  double get _assistLevel {
    if (_mode != Mode.see || !_assist || _teacherMap == null) return 0;
    if (DateTime.now().difference(_teacherAt).inMilliseconds > 1500) return 0;
    return ((_cameraLuma - kAssistLuma) / (kAssistFullLuma - kAssistLuma)).clamp(0.0, 1.0);
  }

  Float32List _fuse(Float32List echo) => fuseEchoCamera(echo, _teacherMap, _assistLevel);

  Future<void> _downloadTeacher() async {
    setState(() => _download = 0);
    try {
      await _sonar.downloadTeacher();
      _status = await _sonar.status();
      await _syncCamera();
    } catch (e) {
      _toast('Download failed: $e');
    }
    if (mounted) setState(() => _download = null);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    for (final s in _subs) {
      s.cancel();
    }
    _camera.stop();
    _sonar.stop();
    WakelockPlus.disable().catchError((Object _) {});
    super.dispose();
  }

  double get _taught => (_status?.totalSamples ?? _samples) / kTeachGoal;

  @override
  Widget build(BuildContext context) => Scaffold(
        backgroundColor: Hud.bg,
        // A plain column inside the safe area: the bars take their real height on every phone
        // (status bar, 3-button or gesture navigation, upside-down), so nothing can overlap.
        body: SafeArea(
          child: Column(children: [
            _topBar(),
            Expanded(child: _blocker != null ? _centerCard(_blocker!) : _body()),
            _navBar(),
          ]),
        ),
      );

  // ───────────────────────────── chrome ─────────────────────────────

  Widget _topBar() {
    final p = _palette;
    final signal = _signal;
    final signalText = signal > 0.6 ? 'ECHO STRONG' : signal > 0.3 ? 'ECHO OK' : signal > 0.08 ? 'ECHO WEAK' : 'NO ECHO';
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 8, 0),
      child: Row(children: [
        Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('MURDOCK', style: Hud.title(p.accent, size: 18).copyWith(fontWeight: FontWeight.w500, letterSpacing: 6)),
          const SizedBox(height: 4),
          Row(children: [
            GestureDetector(
              onTap: _openSystemCheck,
              child: SignalBars(value: signal, palette: p, caption: signalText),
            ),
            const SizedBox(width: 14),
            Text('BRAIN ${(_taught.clamp(0, 1) * 100).round()}%', style: Hud.label(p.accent, size: 9)),
          ]),
        ]),
        const Spacer(),
        _iconButton(Icons.palette_outlined, 'Switch colour',
            () => setState(() => _palette = _palette == SightPalette.radar ? SightPalette.sonar : SightPalette.radar)),
        _iconButton(Icons.monitor_heart_outlined, 'System check', _openSystemCheck),
        _iconButton(Icons.help_outline, 'How it works', _openGuide),
      ]),
    );
  }

  Widget _iconButton(IconData icon, String tip, VoidCallback onTap) =>
      IconButton(tooltip: tip, onPressed: onTap, icon: Icon(icon, color: _palette.accent.withValues(alpha: 0.85)));

  Widget _navBar() {
    const items = [
      (Mode.see, Icons.visibility_outlined, 'SEE'),
      (Mode.radar, Icons.radar, 'RADAR'),
    ];
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
      child: HudPanel(
        palette: _palette,
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(children: [
          for (final (mode, icon, label) in items)
            Expanded(
              child: InkWell(
                onTap: () => _setMode(mode),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Column(children: [
                    Icon(icon,
                        color: mode == _mode ? _palette.accent : Hud.dim,
                        shadows: mode == _mode ? [Shadow(color: _palette.accent, blurRadius: 12)] : null),
                    const SizedBox(height: 4),
                    Text(label, style: Hud.label(mode == _mode ? _palette.accent : Hud.dim, size: 10)),
                  ]),
                ),
              ),
            ),
        ]),
      ),
    );
  }

  /// One-line "what is this / what do I do" banner under the top bar.
  Widget _hint(String what, String todo) => Positioned(
        left: 16,
        right: 16,
        top: 8,
        child: _hintPanel(what, todo),
      );

  Widget _hintPanel(String what, String todo) => HudPanel(
        palette: _palette,
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(what, style: Hud.label(_palette.accent, size: 10)),
          const SizedBox(height: 4),
          Text(todo, style: Hud.body.copyWith(fontSize: 12.5)),
        ]),
      );

  Widget _centerCard(String text, {Widget? action, Widget? top}) => Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: HudPanel(
            palette: _palette,
            padding: const EdgeInsets.all(22),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              if (top != null) ...[top, const SizedBox(height: 18)],
              Text(text, textAlign: TextAlign.center, style: Hud.body),
              if (action != null) ...[const SizedBox(height: 20), action],
            ]),
          ),
        ),
      );

  // ───────────────────────────── screens ─────────────────────────────

  Widget _body() => switch (_mode) {
        Mode.see => _seeScreen(),
        Mode.radar => _radarScreen(),
      };

  Widget _seeScreen() {
    final status = _status;
    final teacherReady = status?.teacherReady ?? false;
    if (status != null && !teacherReady && !_skipDownload) return _downloadCard();

    final stalled = _seeStalled;
    if (stalled != null) return _stalledCard(stalled);

    final assisting = _assist && teacherReady;
    final level = _assistLevel;
    final untaught = _taught < 0.15;
    final String title, body;
    if (_showCamera) {
      title = 'CAMERA VIEW';
      body = 'What the camera sees right now. Tap ECHO VIEW to go back.';
    } else if (!assisting) {
      title = 'SEE · SOUND ONLY';
      body = untaught
          ? 'Generic BatVision head start until it learns your rooms. Turn on camera assist to teach it.'
          : 'Brighter = closer. Move the phone slowly.';
    } else if (_cameraLuma < kAssistLuma) {
      title = 'SEE · SOUND ONLY · TOO DARK FOR CAMERA';
      body = 'The picture comes from echoes alone. Brighter = closer.';
    } else {
      title = 'SEE · CAMERA ASSIST ${(level * 100).round()}%${_lastFrameLearned ? ' · LEARNING' : ''}';
      body = _lastFrameLearned
          ? 'Echo view sharpened by the camera, and every frame teaches the echo AI for the dark.'
          : 'Dim: the camera helps a little but is too noisy to learn from.';
    }
    final cam = _camera.controller;
    return Stack(children: [
      Positioned.fill(
        child: _showCamera && cam != null && cam.value.isInitialized
            ? FittedBox(
                fit: BoxFit.cover,
                clipBehavior: Clip.hardEdge,
                child: SizedBox(
                  width: cam.value.previewSize?.height ?? 1,
                  height: cam.value.previewSize?.width ?? 1,
                  child: CameraPreview(cam),
                ),
              )
            : VisionView(renderer: _vision, palette: _palette, placeholder: 'LISTENING…'),
      ),
      _hint(title, body),
      Positioned(
        left: 16,
        right: 16,
        bottom: 8,
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          if (assisting) _learningStrip(),
          const SizedBox(height: 8),
          Row(children: [
            if (teacherReady)
              _chip(
                _assist ? Icons.videocam : Icons.videocam_off_outlined,
                _assist ? 'ASSIST ON' : 'ASSIST OFF',
                () => _setAssist(!_assist),
              ),
            const Spacer(),
            if (assisting)
              _chip(
                _showCamera ? Icons.graphic_eq : Icons.photo_camera_outlined,
                _showCamera ? 'ECHO VIEW' : 'CAMERA VIEW',
                () => setState(() => _showCamera = !_showCamera),
              ),
          ]),
          const SizedBox(height: 8),
          _grip(checkTaught: true),
        ]),
      ),
    ]);
  }

  /// Why SEE has nothing to draw, or null if it's fine. Never leave the user staring at
  /// "LISTENING…": say which link of the chain is missing.
  String? get _seeStalled {
    final now = DateTime.now();
    if (now.difference(_startedAt).inSeconds < 5) return null; // still starting up
    final last = _lastBrainAt;
    if (last != null && now.difference(last).inSeconds < 3) return null;
    final st = _stats;
    if (st == null || !st.running) return 'The sonar isn\'t running.';
    if (st.pingsPerSec == 0) {
      return 'No pings: the app can\'t hear its own chirp, so the echo AI has nothing to work with.';
    }
    if (st.brain == 'loading') return 'The echo AI is still loading…';
    if (st.brain.startsWith('failed')) return 'The echo AI failed to start: ${st.brain}';
    return 'Pings are arriving but the echo AI isn\'t producing pictures (state: ${st.brain}).';
  }

  Widget _stalledCard(String why) => Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
        child: SingleChildScrollView(
          child: HudPanel(
            palette: _palette,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
              Text('NOTHING TO SHOW YET', style: Hud.label(_palette.accent, size: 11)),
              const SizedBox(height: 6),
              Text(why, style: Hud.body),
              const SizedBox(height: 14),
              _systemCheck(),
            ]),
          ),
        ),
      );

  Widget _chip(IconData icon, String label, VoidCallback onTap) => InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: Hud.bg.withValues(alpha: 0.6),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: _palette.accent.withValues(alpha: 0.6)),
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(icon, size: 16, color: _palette.accent),
            const SizedBox(width: 6),
            Text(label, style: Hud.label(_palette.accent, size: 10)),
          ]),
        ),
      );

  /// Thin progress bar: how much this phone has learned, plus where the camera AI runs.
  Widget _learningStrip() {
    final taught = _taught.clamp(0.0, 1.0);
    final pending = _status?.pendingSteps ?? 0;
    final info = [
      taught >= 1 ? 'LEARNED · STILL IMPROVING' : 'LEARNING ${(taught * 100).round()}%',
      if (_teacherBackend.isNotEmpty) 'CAMERA AI ${_teacherBackend.toUpperCase()} $_teacherMs MS',
      if (pending > 0) '$pending STEPS QUEUED',
    ].join(' · ');
    return HudPanel(
      palette: _palette,
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
        Text(info, style: Hud.label(_palette.accent, size: 9)),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: LinearProgressIndicator(
            value: taught,
            minHeight: 3,
            color: _palette.accent,
            backgroundColor: _palette.accent.withValues(alpha: 0.15),
          ),
        ),
        if (taught < 1) ...[
          const SizedBox(height: 6),
          Text('With lights on, walk slowly past walls, doors, furniture and people.',
              style: Hud.body.copyWith(fontSize: 11)),
        ],
      ]),
    );
  }

  Widget _downloadCard() {
    final d = _download;
    return _centerCard(
      'Murdock learns your rooms with a camera depth AI: wherever the camera can see, it sharpens '
      'the echo view and teaches the echo AI for when it gets dark.\n\n'
      'One-time 66 MB download. It runs fully on your phone afterwards.',
      top: Icon(Icons.download_for_offline_outlined, color: _palette.accent, size: 48),
      action: d == null
          ? Column(children: [
              GlowButton(label: 'DOWNLOAD CAMERA AI', icon: Icons.download, palette: _palette, filled: true, onTap: _downloadTeacher),
              const SizedBox(height: 10),
              TextButton(
                onPressed: () => setState(() => _skipDownload = true),
                child: Text('SOUND ONLY FOR NOW', style: Hud.label(_palette.accent, size: 11)),
              ),
            ])
          : Column(children: [
              LinearProgressIndicator(value: d, color: _palette.accent, backgroundColor: _palette.accent.withValues(alpha: 0.15)),
              const SizedBox(height: 8),
              Text('${(d * 100).round()}%', style: Hud.label(_palette.accent)),
            ]),
    );
  }

  /// True when the sonar clearly isn't working, so RADAR shows the check instead of an empty scope.
  bool get _broken {
    final st = _stats;
    return st != null && (!st.running || st.pingsPerSec == 0 || st.micDb < -85);
  }

  Widget _radarScreen() => Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
        child: Column(children: [
          _hintPanel('RADAR · RAW ECHOES, NO AI',
              'Hold the phone 2 ft from a wall: a bump appears in the ECHO TRACE at about 2 ft. '
              'Step back and it slides right. That bump is the echo. The phone\'s own speaker sound is '
              'filtered out.'),
          const SizedBox(height: 10),
          Expanded(
            child: _broken
                ? SingleChildScrollView(child: HudPanel(palette: _palette, child: _systemCheck()))
                : ScopeView(frames: _sonar.frames, palette: _palette),
          ),
          const SizedBox(height: 8),
          _grip(),
          const SizedBox(height: 8),
          _calibrateButton(),
        ]),
      );

  /// Live GRIP coaching: tilt, sideways, covered speaker, and (in SEE) "not how you taught it".
  Widget _grip({bool checkTaught = false}) => GripBadge(
        advice: GripAdvice.of(
          tilt: _tilt,
          roll: _roll,
          directDb: _stats?.directDb ?? double.nan,
          taughtTilt: _taughtTilt,
          checkTaught: checkTaught,
        ),
        tilt: _tilt,
        palette: _palette,
      );

  /// Re-learn the speaker's own sound so it can be filtered out of the echoes.
  Widget _calibrateButton() {
    final busy = _stats?.calibration == 'calibrating';
    return GlowButton(
      label: busy ? 'LEARNING SPEAKER SOUND…' : 'RECALIBRATE SPEAKER',
      icon: Icons.tune,
      palette: _palette,
      onTap: busy
          ? null
          : () async {
              final go = await showDialog<bool>(
                context: context,
                builder: (c) => AlertDialog(
                  backgroundColor: const Color(0xFF0B0D14),
                  title: Text('RECALIBRATE SPEAKER', style: Hud.label(_palette.accent, size: 13)),
                  content: const Text(
                    'Murdock filters out the sound going straight from your speaker to your mic, so only '
                    'real echoes are left.\n\nHold the phone in your normal grip (GRIP OK), pointing at open '
                    'space with nothing within 3 ft in front (not toward a wall or table). Keep still for '
                    '2 seconds. Your own body then gets filtered out too.',
                    style: Hud.body,
                  ),
                  actions: [
                    TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('CANCEL')),
                    TextButton(onPressed: () => Navigator.pop(c, true), child: const Text('START')),
                  ],
                ),
              );
              if (go == true) await _sonar.calibrate();
            },
    );
  }

  Widget _systemCheck() => SystemCheck(stats: _stats, signal: _signal, lastError: _lastError, palette: _palette);

  Future<void> _openSystemCheck() => showModalBottomSheet<void>(
        context: context,
        backgroundColor: const Color(0xF20B0D14),
        isScrollControlled: true,
        builder: (sheet) => SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 12),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              // Rebuild with live stats while the sheet is open.
              StreamBuilder<SonarStats>(stream: _sonar.stats, builder: (_, _) => _systemCheck()),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton(
                  onPressed: () {
                    Navigator.pop(sheet);
                    _confirmReset();
                  },
                  child: Text('FORGET EVERYTHING IT LEARNED', style: Hud.label(Hud.dim, size: 9)),
                ),
              ),
            ]),
          ),
        ),
      );

  Future<void> _confirmReset() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        backgroundColor: const Color(0xFF0B0D14),
        title: Text('FORGET EVERYTHING?', style: Hud.label(_palette.accent, size: 13)),
        content: const Text(
            'The echo AI loses everything it learned on this phone, and the recorded samples are deleted. '
            'It goes back to the BatVision head start.',
            style: Hud.body),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('CANCEL')),
          TextButton(onPressed: () => Navigator.pop(c, true), child: const Text('FORGET')),
        ],
      ),
    );
    if (ok != true) return;
    await _sonar.reset();
    _vision.update(Float32List(kDepthSize * kDepthSize));
    _toast('Brain reset.');
  }
}
