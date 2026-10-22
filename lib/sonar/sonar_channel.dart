import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Must match SonarConfig.kt / ml/murdock_model.py.
const int kRangeBins = 128;
const int kChannels = 2;
const int kDepthSize = 32;

/// Metres per range bin: 12 samples at 48 kHz, round trip.
const double kMetresPerBin = 12 / 48000 * 343 / 2;

class SonarFrame {
  SonarFrame(Map<Object?, Object?> m)
      : profile = m['profile']! as Float32List,
        depth = m['depth'] as Float32List?,
        direct = (m['direct'] as num?)?.toDouble() ?? 0,
        steps = (m['steps'] as int?) ?? 0,
        loss = (m['loss'] as num?)?.toDouble() ?? double.nan,
        samples = (m['samples'] as int?) ?? 0;

  /// [kRangeBins x kChannels] interleaved, log-scaled echo strength in [0, 1].
  final Float32List profile;

  /// [kDepthSize x kDepthSize] predicted nearness (1 = close), when the model ran.
  final Float32List? depth;
  final double direct;
  final int steps;
  final double loss;
  final int samples;
}

class SonarStatus {
  SonarStatus(Map<Object?, Object?> m)
      : stereo = m['stereo'] as bool? ?? false,
        source = m['source'] as String? ?? '',
        samples = m['samples'] as int? ?? 0,
        steps = m['steps'] as int? ?? 0,
        teacherReady = m['teacherReady'] as bool? ?? false,
        samplesDir = m['samplesDir'] as String? ?? '';

  final bool stereo;
  final String source;
  final int samples;
  final int steps;
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

  Stream<double> get downloadProgress =>
      _raw.where((m) => m['type'] == 'download').map((m) => (m['progress']! as num).toDouble());

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
