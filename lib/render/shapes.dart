import 'dart:math' as math;
import 'dart:typed_data';

import '../sonar/sonar_channel.dart';
import 'fusion.dart';
import 'scope_view.dart';

/// Scene primitives from the echo AI's shapes head. Layout (ml/shapes.py, ShapeFit.kt), all
/// in the square view (x: 0 left .. 1 right, y: 0 top .. 1 bottom), values in [0, 1]:
///   [0..2]   floor: presence, horizon row, nearness at the bottom edge
///   [3..11]  walls left / ahead / right: presence, nearness, slope (0.5 = facing the phone)
///   [12..23] the two nearest obstacles: presence, nearness, cx, cy, w, h
const int kShapeDim = 24;
const int kFloor = 0, kWalls = 3, kBoxes = 12, kMaxBoxes = 2;

/// Wall bands' column ranges in the 32-wide map: [from, to).
const List<(int, int)> kWallBands = [(0, 11), (11, 21), (21, 32)];

class Floor {
  const Floor(this.presence, this.horizon, this.nearness);
  final double presence, horizon, nearness;
}

class Wall {
  const Wall(this.presence, this.nearness, this.slope, this.x0, this.x1);
  final double presence, nearness, slope;

  /// Band edges in the view, 0..1.
  final double x0, x1;

  /// Nearness across the band at view x (the fitted line).
  double nearnessAt(double x) {
    final rel = (x - (x0 + x1) / 2) / (x1 - x0);
    return (nearness + (slope - 0.5) * 2 * rel).clamp(0.0, 1.0);
  }
}

class Box {
  const Box(this.presence, this.nearness, this.cx, this.cy, this.w, this.h);
  final double presence, nearness, cx, cy, w, h;
}

class SceneShapes {
  SceneShapes(this.v) : assert(v.length == kShapeDim);

  final Float32List v;

  Floor get floor => Floor(v[kFloor], v[kFloor + 1], v[kFloor + 2]);

  List<Wall> get walls => [
        for (var k = 0; k < 3; k++)
          Wall(v[kWalls + 3 * k], v[kWalls + 3 * k + 1], v[kWalls + 3 * k + 2], kWallBands[k].$1 / kDepthSize,
              kWallBands[k].$2 / kDepthSize),
      ];

  List<Box> get boxes => [
        for (var k = 0; k < kMaxBoxes; k++)
          Box(v[kBoxes + 6 * k], v[kBoxes + 6 * k + 1], v[kBoxes + 6 * k + 2], v[kBoxes + 6 * k + 3],
              v[kBoxes + 6 * k + 4], v[kBoxes + 6 * k + 5]),
      ];

  /// The nearest thing that is really there: an obstacle, else the wall ahead (or null).
  ({double cx, double cy, double nearness})? get nearest {
    final b = boxes.where((b) => b.presence > 0.5).toList()..sort((a, b) => b.nearness.compareTo(a.nearness));
    if (b.isNotEmpty) return (cx: b.first.cx, cy: b.first.cy - b.first.h / 2, nearness: b.first.nearness);
    final ahead = walls[1];
    if (ahead.presence > 0.5 && ahead.nearness > 0.1) {
      final f = floor;
      return (cx: 0.5, cy: f.presence > 0.5 ? f.horizon * 0.8 : 0.35, nearness: ahead.nearness);
    }
    return null;
  }
}

/// Sanitises a raw shapes vector (never lets NaN or out-of-range values through).
Float32List? cleanShapes(Float32List? s) {
  if (s == null || s.length != kShapeDim) return null;
  return Float32List.fromList([for (final x in s) x.isFinite ? x.clamp(0.0, 1.0) : 0.0]);
}

/// Puts [b]'s obstacles in the slots that best match [a]'s, so smoothing and blending move a
/// box instead of cross-fading two different ones when their nearness order flips.
Float32List _matchBoxes(Float32List a, Float32List b) {
  double cost(int i, int j) {
    final pa = a[kBoxes + 6 * i], pb = b[kBoxes + 6 * j];
    if (pa < 0.1 || pb < 0.1) return 0.5;
    return (a[kBoxes + 6 * i + 2] - b[kBoxes + 6 * j + 2]).abs() + (a[kBoxes + 6 * i + 3] - b[kBoxes + 6 * j + 3]).abs();
  }

  if (cost(0, 1) + cost(1, 0) >= cost(0, 0) + cost(1, 1)) return b;
  final out = Float32List.fromList(b);
  for (var j = 0; j < 6; j++) {
    out[kBoxes + j] = b[kBoxes + 6 + j];
    out[kBoxes + 6 + j] = b[kBoxes + j];
  }
  return out;
}

/// Camera assist for shapes, like [fuseEchoCamera] for the depth picture: where the camera
/// can see, its shapes pull the echo AI's toward the truth, by at most kMaxCameraShare.
Float32List fuseShapes(Float32List echo, Float32List? camera, double level) {
  final a = level.clamp(0.0, 1.0) * kMaxCameraShare;
  if (a <= 0 || camera == null) return echo;
  final cam = _matchBoxes(echo, camera);
  return Float32List.fromList([for (var i = 0; i < kShapeDim; i++) echo[i] * (1 - a) + cam[i] * a]);
}

/// Smooths shapes over time (single pings are noisy) without letting boxes swap places.
class ShapeTracker {
  Float32List? _s;

  SceneShapes? get shapes => _s == null ? null : SceneShapes(_s!);

  void update(Float32List? raw) {
    final s = cleanShapes(raw);
    if (s == null) return;
    final prev = _s;
    if (prev == null) {
      _s = s;
      return;
    }
    final m = _matchBoxes(prev, s);
    _s = Float32List.fromList([for (var i = 0; i < kShapeDim; i++) prev[i] * 0.7 + m[i] * 0.3]);
  }
}

/// Distance in feet of the nearest clear echo (speaker zone skipped), or null when nothing
/// stands out. The shapes are in the camera's view and only know "nearer / farther"; the
/// sonar knows how far, so this gives the overlay its distance label.
double? nearestEchoFeet(Float32List clean) {
  final first = (kSpeakerZoneFeet / kFeetPerBin).ceil();
  final v = <double>[
    for (var b = first; b < kRangeBins; b++) (clean[b * kChannels] + clean[b * kChannels + 1]) / 2,
  ];
  if (v.isEmpty) return null;
  final sorted = [...v]..sort();
  final median = sorted[sorted.length ~/ 2];
  final peak = sorted.last;
  if (peak - median < 0.15) return null; // no echo clearly above the background
  final i = v.indexWhere((x) => x >= peak - 0.08);
  return (first + i) * kFeetPerBin;
}

/// Deterministic per-shape wobble so the wireframe breathes like a sensed outline, not a
/// rendered CAD model.
double wobble(double t, int seed, {double amp = 1}) =>
    amp * (math.sin(t * 1.7 + seed * 2.3) * 0.6 + math.sin(t * 3.1 + seed * 0.7) * 0.4);
