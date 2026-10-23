import 'dart:math' as math;
import 'dart:typed_data';

import '../sonar/sonar_channel.dart';
import 'fusion.dart';

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

/// Shape slots: floor, walls left / ahead / right, obstacles 1 and 2.
const int kShapeSlots = 6;
const List<int> _presenceAt = [kFloor, kWalls, kWalls + 3, kWalls + 6, kBoxes, kBoxes + 6];

/// How sure (slow average of presence) a shape must be before it is drawn at all, and when
/// it is drawn at full strength. Shapes must be reported steadily for ~2 s to appear.
const double kShapeSureFrom = 0.8, kShapeSureFull = 0.92;

/// Most of a pixel a sure shape may set: the echo picture always keeps the rest.
const double kShapeShare = 0.6;

/// Smooths shapes over time (single pings are noisy) without letting boxes swap places, and
/// keeps a slow per-shape confidence so only steadily reported shapes are drawn.
class ShapeTracker {
  Float32List? _s;
  final Float32List _sure = Float32List(kShapeSlots);

  SceneShapes? get shapes => _s == null ? null : SceneShapes(_s!);

  /// 0 (not drawn) .. 1 (drawn fully) per slot.
  double strength(int slot) => ((_sure[slot] - kShapeSureFrom) / (kShapeSureFull - kShapeSureFrom)).clamp(0.0, 1.0);

  void update(Float32List? raw) {
    final s = cleanShapes(raw);
    if (s == null) return;
    final prev = _s;
    if (prev == null) {
      _s = s;
    } else {
      final m = _matchBoxes(prev, s);
      _s = Float32List.fromList([for (var i = 0; i < kShapeDim; i++) prev[i] * 0.7 + m[i] * 0.3]);
    }
    // ~1 s time constant at the brain's frame rate: a flicker never builds confidence.
    for (var k = 0; k < kShapeSlots; k++) {
      _sure[k] = _sure[k] * 0.96 + s[_presenceAt[k]] * 0.04;
    }
  }
}

/// Draws the shapes the tracker is sure of into the nearness map, so the normal echo
/// renderer shows them in its own laser look: walls as planes, the floor as a ramp, and
/// obstacles as solid blocks. Shapes it isn't sure of leave the map untouched.
Float32List paintShapes(Float32List depth, ShapeTracker tracker) {
  final sc = tracker.shapes;
  if (sc == null || depth.length != kDepthSize * kDepthSize) return depth;
  const n = kDepthSize;
  final target = Float32List(n * n);
  final weight = Float32List(n * n);
  void put(int x, int y, double v, double w) {
    final i = y * n + x;
    if (w >= weight[i]) {
      target[i] = v.clamp(0.0, 1.0);
      weight[i] = w;
    }
  }

  final floor = sc.floor;
  final fw = tracker.strength(0);
  final horizon = fw > 0 ? (floor.horizon * (n - 1)).round().clamp(0, n - 1) : n;
  final walls = sc.walls;

  // Walls: rows above the horizon, the fitted line across each band.
  for (var k = 0; k < 3; k++) {
    final w = tracker.strength(1 + k);
    if (w <= 0) continue;
    final band = kWallBands[k];
    final wall = walls[k];
    for (var x = band.$1; x < band.$2; x++) {
      final v = wall.nearnessAt((x + 0.5) / n);
      for (var y = 0; y < horizon; y++) {
        put(x, y, v, w);
      }
    }
  }

  // Floor: a ramp from the horizon (as far as the wall ahead, else far) to the bottom edge.
  if (fw > 0 && horizon < n - 1) {
    final far = tracker.strength(2) > 0 ? walls[1].nearness : 0.0;
    for (var y = horizon; y < n; y++) {
      final v = far + (floor.nearness - far) * (y - horizon) / (n - 1 - horizon);
      for (var x = 0; x < n; x++) {
        put(x, y, v, fw);
      }
    }
  }

  // Obstacles last: they stand in front of walls and floor.
  for (final (k, b) in sc.boxes.indexed) {
    final w = tracker.strength(4 + k);
    if (w <= 0) continue;
    final x0 = ((b.cx - b.w / 2) * n).round().clamp(0, n), x1 = ((b.cx + b.w / 2) * n).round().clamp(0, n);
    final y0 = ((b.cy - b.h / 2) * n).round().clamp(0, n), y1 = ((b.cy + b.h / 2) * n).round().clamp(0, n);
    for (var y = y0; y < y1; y++) {
      for (var x = x0; x < x1; x++) {
        final i = y * n + x;
        target[i] = b.nearness;
        weight[i] = math.max(weight[i], w * 1.0001); // over walls/floor
      }
    }
  }

  var any = false;
  final out = Float32List(n * n);
  for (var i = 0; i < n * n; i++) {
    final a = weight[i].clamp(0.0, 1.0) * kShapeShare;
    if (a > 0) any = true;
    final d = depth[i].isFinite ? depth[i] : 0.0;
    out[i] = (d * (1 - a) + target[i] * a).clamp(0.0, 1.0);
  }
  return any ? out : depth;
}
