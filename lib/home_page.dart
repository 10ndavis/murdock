import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'render/palette.dart';
import 'render/scope_view.dart';
import 'render/vision_renderer.dart';
import 'render/vision_view.dart';
import 'sonar/sonar_channel.dart';
import 'train/camera_teacher.dart';
import 'ui/guide.dart';
import 'ui/hud.dart';
import 'ui/system_check.dart';

enum Mode { see, radar, teach }

/// Camera-labelled samples that count as "fully taught" for one room (~6 min at 4 fps).
const int kTeachGoal = 1500;

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
  final List<StreamSubscription<Object?>> _subs = [];

  Mode _mode = Mode.radar;
  SightPalette _palette = SightPalette.radar;
  SonarStatus? _status;
  String? _blocker;
  double? _download;
  bool _teaching = false;
  double _signal = 0;
  int _samples = 0;
  SonarStats? _stats;
  bool _previewUntaught = false;
  String? _lastError;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _subs.add(_sonar.frames.listen((f) => _signal = _signal * 0.9 + f.signal * 0.1));
    _subs.add(_sonar.brain.listen((b) {
      _samples = b.samples;
      _vision.update(b.depth);
    }));
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
    // Send new users to the place they need to start.
    if (s.totalSamples >= kTeachGoal ~/ 3 && mounted) setState(() => _mode = Mode.see);
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
      _startSonar();
    } else if (state == AppLifecycleState.paused) {
      _setTeaching(false);
    }
  }

  Future<void> _setMode(Mode m) async {
    if (m == _mode) return;
    if (_mode == Mode.teach) await _setTeaching(false);
    setState(() => _mode = m);
  }

  Future<void> _setTeaching(bool on) async {
    if (on == _teaching) return;
    if (on) {
      if (!await _sonar.requestPermission('camera')) {
        _toast('Camera permission is needed to teach.');
        return;
      }
      setState(() => _teaching = true);
      await _sonar.setTraining(true);
      try {
        await _camera.start((Uint8List rgb) async => _teacherView.update(await _sonar.teach(rgb)));
      } catch (e) {
        _toast('Camera failed: $e');
        await _setTeaching(false);
        return;
      }
    } else {
      setState(() => _teaching = false);
      await _sonar.setTraining(false);
      await _camera.stop();
      await _sonar.save();
    }
    if (mounted) setState(() {});
  }

  Future<void> _downloadTeacher() async {
    setState(() => _download = 0);
    try {
      await _sonar.downloadTeacher();
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
        body: Stack(children: [
          Positioned.fill(child: _blocker != null ? _centerCard(_blocker!) : _body()),
          Positioned(left: 0, right: 0, top: 0, child: SafeArea(child: _topBar())),
          Positioned(left: 0, right: 0, bottom: 0, child: SafeArea(child: _navBar())),
        ]),
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
      (Mode.teach, Icons.school_outlined, 'TEACH'),
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
        top: MediaQuery.paddingOf(context).top + 76,
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
        Mode.teach => _teachScreen(),
      };

  Widget _seeScreen() {
    final untaught = _taught < 0.15;
    if (untaught && !_previewUntaught) {
      return _centerCard(
        'The echo AI comes pre-trained on the BatVision dataset (about 11,000 real echo recordings paired with depth '
        'photos), so it knows what rooms typically look like. It hasn\'t learned YOUR rooms yet, so '
        'expect a generic room shape that only loosely follows what\'s really there.\n\n'
        'For a picture that follows your surroundings, use TEACH in a lit room for a few minutes.',
        top: ScanSweep(palette: _palette, size: 110),
        action: Column(children: [
          GlowButton(label: 'TEACH NOW', icon: Icons.school_outlined, palette: _palette, filled: true,
              onTap: () => _setMode(Mode.teach)),
          const SizedBox(height: 10),
          TextButton(
            onPressed: () => setState(() => _previewUntaught = true),
            child: Text('PREVIEW ANYWAY', style: Hud.label(_palette.accent, size: 11)),
          ),
        ]),
      );
    }
    return Stack(children: [
      Positioned.fill(child: VisionView(renderer: _vision, palette: _palette, placeholder: 'LISTENING…')),
      _hint(untaught ? 'SEE · PREVIEW (NOT TAUGHT YET)' : 'SEE · SOUND-ONLY VISION',
          untaught
              ? 'Generic BatVision head start: a typical room shape. TEACH makes it follow your room.'
              : 'Lights off. Brighter = closer. Move the phone slowly.'),
    ]);
  }

  /// True when the sonar clearly isn't working, so RADAR shows the check instead of an empty scope.
  bool get _broken {
    final st = _stats;
    return st != null && (!st.running || st.pingsPerSec == 0 || st.micDb < -85);
  }

  Widget _radarScreen() => Padding(
        padding: EdgeInsets.fromLTRB(16, MediaQuery.paddingOf(context).top + 76, 16, 100),
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
          _calibrateButton(),
        ]),
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
                    'real echoes are left.\n\nHold the phone up at arm\'s length, pointing at open space with '
                    'nothing within 3 ft (not toward a wall or table). Keep still for 2 seconds.',
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
        builder: (_) => SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 28),
            // Rebuild with live stats while the sheet is open.
            child: StreamBuilder<SonarStats>(
              stream: _sonar.stats,
              builder: (_, _) => _systemCheck(),
            ),
          ),
        ),
      );

  Widget _teachScreen() {
    final p = _palette;
    final status = _status;
    if (status != null && !status.teacherReady) {
      final d = _download;
      return _centerCard(
        'To learn, Murdock uses the camera as a teacher: a vision AI works out how far '
        'everything is, and the echo AI learns to match it.\n\n'
        'This needs a one-time 66 MB download. It runs fully on your phone afterwards.',
        top: Icon(Icons.download_for_offline_outlined, color: p.accent, size: 48),
        action: d == null
            ? GlowButton(label: 'DOWNLOAD TEACHER', icon: Icons.download, palette: p, filled: true, onTap: _downloadTeacher)
            : Column(children: [
                LinearProgressIndicator(value: d, color: p.accent, backgroundColor: p.accent.withValues(alpha: 0.15)),
                const SizedBox(height: 8),
                Text('${(d * 100).round()}%', style: Hud.label(p.accent)),
              ]),
      );
    }
    final cam = _camera.controller;
    final taught = _taught.clamp(0.0, 1.0);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 76, 16, 100),
      child: Column(children: [
        HudPanel(
          palette: p,
          child: Row(children: [
            ProgressRing(
              value: taught,
              palette: p,
              size: 96,
              center: '${(taught * 100).round()}%',
              caption: 'TAUGHT',
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(_teaching ? 'TEACHING…' : taught >= 1 ? 'READY FOR THE DARK' : 'HOW TO TEACH',
                    style: Hud.label(p.accent, size: 11)),
                const SizedBox(height: 6),
                Text(
                  _teaching
                      ? 'Walk slowly. Point at walls, doors, furniture, people. Keep the lights on.'
                      : taught >= 1
                          ? 'Turn the lights off and open SEE. Teach again in each new room.'
                          : '1  Lights on\n2  Press START\n3  Walk slowly around the room until 100%',
                  style: Hud.body.copyWith(fontSize: 12.5),
                ),
              ]),
            ),
          ]),
        ),
        const SizedBox(height: 12),
        Expanded(
          child: Row(children: [
            Expanded(
              child: _feed('WHAT THE CAMERA SEES', _teaching && cam != null && cam.value.isInitialized
                  ? FittedBox(
                      fit: BoxFit.cover,
                      clipBehavior: Clip.hardEdge,
                      child: SizedBox(
                        width: cam.value.previewSize?.height ?? 1,
                        height: cam.value.previewSize?.width ?? 1,
                        child: CameraPreview(cam),
                      ),
                    )
                  : Center(child: Icon(Icons.videocam_off_outlined, color: Hud.dim))),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _feed('WHAT THE ECHO AI SEES',
                  VisionView(renderer: _vision, palette: p, placeholder: 'LISTENING…')),
            ),
          ]),
        ),
        const SizedBox(height: 14),
        GlowButton(
          label: _teaching ? 'PAUSE' : 'START TEACHING',
          icon: _teaching ? Icons.pause : Icons.play_arrow,
          filled: !_teaching,
          palette: p,
          onTap: () => _setTeaching(!_teaching),
        ),
        TextButton(
          onPressed: _confirmReset,
          child: Text('FORGET EVERYTHING', style: Hud.label(Hud.dim, size: 9)),
        ),
      ]),
    );
  }

  Widget _feed(String title, Widget child) => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(title, style: Hud.label(_palette.accent, size: 9)),
        const SizedBox(height: 6),
        Expanded(
          child: Container(
            decoration: BoxDecoration(border: Border.all(color: _palette.accent.withValues(alpha: 0.3))),
            child: ClipRect(child: SizedBox.expand(child: child)),
          ),
        ),
      ]);

  Future<void> _confirmReset() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        backgroundColor: const Color(0xFF0B0D14),
        title: Text('FORGET EVERYTHING?', style: Hud.label(_palette.accent, size: 13)),
        content: const Text('The echo AI loses everything it learned on this phone.', style: Hud.body),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('CANCEL')),
          TextButton(onPressed: () => Navigator.pop(c, true), child: const Text('FORGET')),
        ],
      ),
    );
    if (ok != true) return;
    await _setTeaching(false);
    await _sonar.reset();
    _vision.update(Float32List(kDepthSize * kDepthSize));
    _toast('Brain reset.');
  }
}
