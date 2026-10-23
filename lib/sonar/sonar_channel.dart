import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Must match SonarConfig.kt / ml/murdock_model.py.
const int kRangeBins = 128;
const int kChannels = 2;
const int kDepthSize = 32;

/// Feet per range bin: 12 samples at 48 kHz, round trip (343 m/s = 1125 ft/s).
const double kFeetPerBin = 12 / 48000 * 1125.3 / 2;

/// Furthest distance the sonar covers (~18 ft).
const double kMaxFeet = kRangeBins * kFeetPerBin;

/// One sonar ping: the raw echo profile. Sent for every ping, AI or not.
class SonarFrame {
  SonarFrame(Map<Object?, Object?> m)
      : profile = m['profile']! as Float32List,
        clean = (m['clean'] as Float32List?) ?? m['profile']! as Float32List,
        direct = (m['direct'] as num?)?.toDouble() ?? 0;

  /// [kRangeBins x kChannels] interleaved, log-scaled echo strength in [0, 1]. Raw: this
  /// is what the AI sees, including the loud speaker->mic sound at distance 0.
  final Float32List profile;

  /// Same layout, with the speaker's own sound cancelled: only real echoes. For display.
  final Float32List clean;
  final double direct;

  /// How clearly the chirp is heard, 0 (nothing) to 1 (strong).
  ///
  /// Each profile is normalised to the direct speaker->mic peak, so when the chirp is
  /// heard well the far range bins sit far below it; when it isn't, the "peak" is just
  /// noise and every bin looks the same.
  double get signal {
    final far = <double>[for (var b = kRangeBins ~/ 2; b < kRangeBins; b++) profile[b * kChannels]]..sort();
    final median = far[far.length ~/ 2];
    return ((0.85 - median) / 0.4).clamp(0.0, 1.0);
  }
}

/// The echo AI's output for a recent ping.
class BrainFrame {
  BrainFrame(Map<Object?, Object?> m)
      : depth = m['depth']! as Float32List,
        steps = (m['steps'] as int?) ?? 0,
        loss = (m['loss'] as num?)?.toDouble() ?? double.nan,
        samples = (m['samples'] as int?) ?? 0;

  /// [kDepthSize x kDepthSize] predicted nearness (1 = close).
  final Float32List depth;
  final int steps;
  final double loss;

  /// Lifetime number of camera-labelled samples.
  final int samples;
}

/// Once-a-second health report from the native audio engine.
class SonarStats {
  SonarStats(Map<Object?, Object?> m)
      : running = m['running'] as bool? ?? false,
        speaker = m['speaker'] as String? ?? '',
        mic = m['mic'] as String? ?? '',
        source = m['source'] as String? ?? '',
        stereo = m['stereo'] as bool? ?? false,
        micDb = (m['micDb'] as num?)?.toDouble() ?? -140,
        pingsPerSec = m['pingsPerSec'] as int? ?? 0,
        calibrations = m['calibrations'] as int? ?? 0,
        volume = (m['volume'] as num?)?.toDouble() ?? 0,
        calibration = m['calibration'] as String? ?? 'none',
        directDb = (m['directDb'] as num?)?.toDouble() ?? double.nan,
        loudness = (m['loudness'] as num?)?.toDouble() ?? kDefaultLoudness,
        clarityDb = (m['clarityDb'] as num?)?.toDouble() ?? double.nan,
        tuning = m['tuning'] as bool? ?? false,
        brain = m['brain'] as String? ?? '';

  final bool running;
  final String speaker;
  final String mic;
  final String source;
  final bool stereo;

  /// Microphone loudness in dB full scale (silence ≈ -90 or lower, talking ≈ -30).
  final double micDb;
  final int pingsPerSec;
  final int calibrations;

  /// Media volume, 0..1.
  final double volume;

  /// Speaker-cancellation state: none, calibrating, done (just now), saved (from disk).
  final String calibration;

  /// Speaker->mic loudness relative to calibration, in dB (NaN until calibrated). Strongly
  /// negative means the speaker or mic is probably covered.
  final double directDb;

  /// Chirp loudness, 0.2..1 of full scale (on top of the media volume).
  final double loudness;

  /// How far the strongest echo stands out of the noise, in dB (NaN until heard).
  final double clarityDb;

  /// AUTO-TUNE is running.
  final bool tuning;
  final String brain;
}

/// Must match SonarConfig.kt.
const double kDefaultLoudness = 0.6;
const double kMinLoudness = 0.2;
const double kMaxLoudness = 1.0;

/// AUTO-TUNE progress or result, from SonarEngine.
class TuneEvent {
  TuneEvent(Map<Object?, Object?> m)
      : state = m['state'] as String? ?? 'running',
        phase = m['phase'] as String? ?? '',
        progress = (m['progress'] as num?)?.toDouble() ?? 0,
        levels = [for (final v in m['levels'] as List<Object?>? ?? const []) (v! as num).toDouble()],
        clarity = [for (final v in m['clarity'] as List<Object?>? ?? const []) (v as num?)?.toDouble() ?? double.nan],
        clipped = (m['clipped'] as List<Object?>? ?? const []).cast<bool>(),
        testing = (m['testing'] as num?)?.toDouble() ?? kDefaultLoudness,
        best = (m['best'] as num?)?.toDouble() ?? double.nan,
        bestDb = (m['bestDb'] as num?)?.toDouble() ?? double.nan,
        message = m['message'] as String? ?? '';

  /// running, done, failed or cancelled.
  final String state;

  /// settle, measure, final_settle or calibrate.
  final String phase;
  final double progress;

  /// Loudness levels tried, and the echo clarity (dB, NaN = not yet) and clipping at each.
  final List<double> levels;
  final List<double> clarity;
  final List<bool> clipped;

  /// Loudness being played right now.
  final double testing;

  /// The chosen loudness and its clarity (NaN until chosen).
  final double best;
  final double bestDb;

  /// Why it failed, in plain words.
  final String message;

  bool get running => state == 'running';
}

class SonarStatus {
  SonarStatus(Map<Object?, Object?> m)
      : stereo = m['stereo'] as bool? ?? false,
        source = m['source'] as String? ?? '',
        samples = m['samples'] as int? ?? 0,
        steps = m['steps'] as int? ?? 0,
        totalSamples = m['totalSamples'] as int? ?? 0,
        teacherReady = m['teacherReady'] as bool? ?? false,
        samplesDir = m['samplesDir'] as String? ?? '',
        pendingSteps = m['pendingSteps'] as int? ?? 0,
        teacherBackend = m['teacherBackend'] as String? ?? '',
        teacherMs = m['teacherMs'] as int? ?? 0,
        stepsPerSec = (m['stepsPerSec'] as num?)?.toDouble() ?? 0,
        finishing = m['finishing'] as bool? ?? false,
        tiltCoverage = (m['tiltCoverage'] as List<Object?>? ?? const []).cast<int>(),
        coverageHere = m['coverageHere'] as int? ?? 0;

  final bool stereo;
  final String source;
  final int samples;
  final int steps;

  /// Camera-labelled samples this phone has ever learned from (persists across restarts).
  final int totalSamples;
  final bool teacherReady;
  final String samplesDir;

  /// Training steps still owed to recent samples (0 = caught up).
  final int pendingSteps;

  /// Where the camera depth teacher runs ("gpu"/"cpu", empty until first use) and how long a frame takes.
  final String teacherBackend;
  final int teacherMs;

  /// Training speed over the last second, and whether FINISH LEARNING is running.
  final double stepsPerSec;
  final bool finishing;

  /// Samples learned per phone-tilt bin (10° bins from -30°), and at the current tilt.
  final List<int> tiltCoverage;
  final int coverageHere;
}

class TeachResult {
  TeachResult(Map<Object?, Object?> m)
      : depth = m['depth']! as Float32List,
        ms = m['ms'] as int? ?? 0,
        backend = m['backend'] as String? ?? '';

  final Float32List depth;
  final int ms;
  final String backend;
}

/// Thin wrapper over the native sonar/model bridge in MainActivity.kt.
class SonarChannel {
  static const _method = MethodChannel('murdock/sonar');
  static const _events = EventChannel('murdock/frames');

  late final Stream<Map<Object?, Object?>> _raw =
      _events.receiveBroadcastStream().cast<Map<Object?, Object?>>().asBroadcastStream();

  Stream<SonarFrame> get frames =>
      _raw.where((m) => m['type'] == 'ping').map(SonarFrame.new);

  Stream<BrainFrame> get brain => _raw.where((m) => m['type'] == 'brain').map(BrainFrame.new);

  /// Phone tilt in degrees: 0 = upright portrait, positive = top tilted toward you (speaker
  /// aims forward), -90 = lying flat screen-up. Roll ~±90 = landscape.
  Stream<({double tilt, double roll})> get pose => _raw
      .where((m) => m['type'] == 'pose')
      .map((m) => (tilt: (m['tilt']! as num).toDouble(), roll: (m['roll']! as num).toDouble()));

  Stream<SonarStats> get stats => _raw.where((m) => m['type'] == 'stats').map(SonarStats.new);

  Stream<TuneEvent> get tune => _raw.where((m) => m['type'] == 'tune').map(TuneEvent.new);

  Stream<String> get errors => _raw.where((m) => m['type'] == 'error').map((m) => m['message']! as String);

  Stream<double> get downloadProgress =>
      _raw.where((m) => m['type'] == 'download').map((m) => (m['progress']! as num).toDouble());

  /// [name] is 'microphone' or 'camera'.
  Future<bool> requestPermission(String name) async =>
      await _method.invokeMethod<bool>('requestPermission', name) ?? false;

  Future<bool> getFlag(String name) async => await _method.invokeMethod<bool>('getFlag', name) ?? false;
  Future<double?> getNumber(String name) async {
    final v = await _method.invokeMethod<double>('getNumber', name);
    return v == null || v.isNaN ? null : v;
  }

  Future<void> setNumber(String name, double value) => _method.invokeMethod('setNumber', [name, value]);

  Future<void> clearFlag(String name) => _method.invokeMethod('clearFlag', name);
  Future<void> setFlag(String name) => _method.invokeMethod('setFlag', name);

  Future<void> start() => _method.invokeMethod('start');
  /// Re-learn the speaker's own sound (~2 s). Hold the phone away from everything.
  Future<void> calibrate() => _method.invokeMethod('calibrate');

  /// Chirp loudness, [kMinLoudness]..[kMaxLoudness]. Applied live and remembered.
  Future<void> setLoudness(double v) => _method.invokeMethod('setLoudness', v);

  /// AUTO-TUNE: tries several loudness levels against a wall and keeps the clearest (~10 s).
  Future<void> autoTune() => _method.invokeMethod('autoTune');
  Future<void> cancelTune() => _method.invokeMethod('cancelTune');

  /// FINISH LEARNING: stop the sonar and spend all the effort on the training queue.
  Future<void> finishLearning() => _method.invokeMethod('finishLearning');
  Future<void> endFinishLearning() => _method.invokeMethod('endFinishLearning');

  Future<void> stop() => _method.invokeMethod('stop');
  Future<void> setTraining(bool on) => _method.invokeMethod('setTraining', on);
  Future<void> save() => _method.invokeMethod('save');
  Future<void> reset() => _method.invokeMethod('reset');
  Future<bool> teacherReady() async => await _method.invokeMethod<bool>('teacherReady') ?? false;
  Future<void> downloadTeacher() => _method.invokeMethod('downloadTeacher');

  Future<SonarStatus> status() async =>
      SonarStatus((await _method.invokeMethod<Map<Object?, Object?>>('status'))!);

  /// Sends an upright 256x256 RGB frame; returns the camera teacher's nearness map. With
  /// [learn], the frame also becomes a training sample for the echo AI.
  Future<TeachResult> teach(Uint8List rgb, {required bool learn}) async =>
      TeachResult((await _method.invokeMethod<Map<Object?, Object?>>('teach', {'rgb': rgb, 'learn': learn}))!);
}
