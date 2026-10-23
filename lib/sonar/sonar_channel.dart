import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Must match SonarConfig.kt / ml/murdock_model.py.
const int kRangeBins = 128;
const int kChannels = 2;
const int kDepthSize = 32;

/// Metres per range bin: 12 samples at 48 kHz, round trip.
const double kMetresPerBin = 12 / 48000 * 343 / 2;

/// One sonar ping: the raw echo profile. Sent for every ping, AI or not.
class SonarFrame {
  SonarFrame(Map<Object?, Object?> m)
      : profile = m['profile']! as Float32List,
        direct = (m['direct'] as num?)?.toDouble() ?? 0;

  /// [kRangeBins x kChannels] interleaved, log-scaled echo strength in [0, 1].
  final Float32List profile;
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
  final String brain;
}

class SonarStatus {
  SonarStatus(Map<Object?, Object?> m)
      : stereo = m['stereo'] as bool? ?? false,
        source = m['source'] as String? ?? '',
        samples = m['samples'] as int? ?? 0,
        steps = m['steps'] as int? ?? 0,
        totalSamples = m['totalSamples'] as int? ?? 0,
        teacherReady = m['teacherReady'] as bool? ?? false,
        samplesDir = m['samplesDir'] as String? ?? '';

  final bool stereo;
  final String source;
  final int samples;
  final int steps;

  /// Camera-labelled samples this phone has ever learned from (persists across restarts).
  final int totalSamples;
  final bool teacherReady;
  final String samplesDir;
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

  Stream<SonarStats> get stats => _raw.where((m) => m['type'] == 'stats').map(SonarStats.new);

  Stream<String> get errors => _raw.where((m) => m['type'] == 'error').map((m) => m['message']! as String);

  Stream<double> get downloadProgress =>
      _raw.where((m) => m['type'] == 'download').map((m) => (m['progress']! as num).toDouble());

  /// [name] is 'microphone' or 'camera'.
  Future<bool> requestPermission(String name) async =>
      await _method.invokeMethod<bool>('requestPermission', name) ?? false;

  Future<bool> getFlag(String name) async => await _method.invokeMethod<bool>('getFlag', name) ?? false;
  Future<void> setFlag(String name) => _method.invokeMethod('setFlag', name);

  Future<void> start() => _method.invokeMethod('start');
  Future<void> stop() => _method.invokeMethod('stop');
  Future<void> setTraining(bool on) => _method.invokeMethod('setTraining', on);
  Future<void> save() => _method.invokeMethod('save');
  Future<void> reset() => _method.invokeMethod('reset');
  Future<bool> teacherReady() async => await _method.invokeMethod<bool>('teacherReady') ?? false;
  Future<void> downloadTeacher() => _method.invokeMethod('downloadTeacher');

  Future<SonarStatus> status() async =>
      SonarStatus((await _method.invokeMethod<Map<Object?, Object?>>('status'))!);

  /// Sends an upright 256x256 RGB frame; returns the camera teacher's nearness map.
  Future<Float32List> teach(Uint8List rgb) async =>
      (await _method.invokeMethod<Float32List>('teach', rgb))!;
}
