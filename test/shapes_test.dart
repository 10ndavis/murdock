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

  test('nearest echo distance in feet', () {
    final clean = Float32List(kRangeBins * kChannels)..fillRange(0, kRangeBins * kChannels, 0.2);
    expect(nearestEchoFeet(clean), isNull); // nothing stands out
    for (final (bin, v) in [(30, 0.9), (60, 0.95)]) {
      clean[bin * 2] = v;
      clean[bin * 2 + 1] = v;
    }
    // The first echo within ~6 dB of the strongest one: bin 30.
    expect(nearestEchoFeet(clean), closeTo(30 * kFeetPerBin, 1e-9));
  });

  test('nearest shape prefers a present obstacle', () {
    final s = SceneShapes(_shapes());
    expect(s.nearest!.cx, closeTo(0.3, 1e-6));
    expect(s.floor.horizon, closeTo(0.5, 1e-6));
    expect(s.walls.length, 3);
  });
}
