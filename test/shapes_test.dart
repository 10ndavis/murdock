import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:murdock/render/fusion.dart';
import 'package:murdock/render/shapes.dart';
import 'package:murdock/sonar/sonar_channel.dart';

Float32List _shapes({double box1x = 0.3, double box2x = 0.7}) {
  final s = Float32List(kShapeDim);
  s[kFloor] = 0.9;
  s[kFloor + 1] = 0.5;
  s[kBoxes] = 1;
  s[kBoxes + 1] = 0.8;
  s[kBoxes + 2] = box1x;
  s[kBoxes + 3] = 0.5;
  s[kBoxes + 4] = 0.2;
  s[kBoxes + 5] = 0.3;
  s[kBoxes + 6] = 1;
  s[kBoxes + 7] = 0.6;
  s[kBoxes + 8] = box2x;
  s[kBoxes + 9] = 0.5;
  return s;
}

void main() {
  test('invalid shapes never get through', () {
    final s = Float32List(kShapeDim)..[0] = double.nan;
    s[1] = 7;
    final c = cleanShapes(s)!;
    expect(c[0], 0);
    expect(c[1], 1);
    expect(cleanShapes(Float32List(3)), isNull);
    expect(cleanShapes(null), isNull);
  });

  test('tracking keeps boxes in place when their order flips', () {
    final tracker = ShapeTracker()..update(_shapes());
    // Same two obstacles, reported in the other order (their nearness ranking flipped).
    tracker.update(_shapes(box1x: 0.7, box2x: 0.3));
    final xs = tracker.shapes!.boxes.map((b) => b.cx).toList();
    expect(xs[0], closeTo(0.3, 1e-6));
    expect(xs[1], closeTo(0.7, 1e-6));
  });

  test('camera assist moves shapes toward the camera, but never all the way', () {
    final echo = Float32List(kShapeDim);
    final camera = Float32List(kShapeDim)..fillRange(0, kShapeDim, 1);
    final full = fuseShapes(echo, camera, 1);
    expect(full[0], closeTo(kMaxCameraShare, 1e-6));
    expect(identical(fuseShapes(echo, camera, 0), echo), isTrue);
    expect(identical(fuseShapes(echo, null, 1), echo), isTrue);
  });

  test('a shape is only drawn after it has been reported steadily', () {
    final depth = Float32List(kDepthSize * kDepthSize);
    final tracker = ShapeTracker()..update(_shapes());
    // One confident report is not enough: the picture is untouched.
    expect(identical(paintShapes(depth, tracker), depth), isTrue);
    for (var i = 0; i < 80; i++) {
      tracker.update(_shapes());
    }
    final painted = paintShapes(depth, tracker);
    expect(identical(painted, depth), isFalse);
    // Obstacle 1 at x 0.3, y 0.5 (nearness 0.8) now shows, but never fully.
    final at = (kDepthSize ~/ 2) * kDepthSize + (0.3 * kDepthSize).round();
    expect(painted[at], greaterThan(0.3));
    expect(painted[at], lessThanOrEqualTo(0.8 * kShapeShare + 1e-6));
  });

  test('a flickering shape never builds confidence', () {
    final depth = Float32List(kDepthSize * kDepthSize);
    final tracker = ShapeTracker();
    for (var i = 0; i < 200; i++) {
      tracker.update(i.isEven ? _shapes() : Float32List(kShapeDim));
    }
    for (var k = 0; k < kShapeSlots; k++) {
      expect(tracker.strength(k), 0);
    }
    expect(identical(paintShapes(depth, tracker), depth), isTrue);
  });

  test('scene shapes layout', () {
    final s = SceneShapes(_shapes());
    expect(s.floor.horizon, closeTo(0.5, 1e-6));
    expect(s.walls.length, 3);
    expect(s.boxes.first.cx, closeTo(0.3, 1e-6));
  });
}
