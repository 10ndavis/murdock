import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:murdock/render/palette.dart';
import 'package:murdock/render/vision_renderer.dart';
import 'package:murdock/sonar/sonar_channel.dart';
import 'package:murdock/ui/grip.dart';

void main() {
  test('palette ramps from black to hot', () {
    final px = List<int>.filled(8, 0);
    SightPalette.radar.write(0, px, 0);
    SightPalette.radar.write(1, px, 4);
    expect(px.sublist(0, 3), [0, 0, 0]);
    expect(px[4], greaterThan(200));
  });

  testWidgets('vision renderer produces an image from a depth map', (tester) async {
    final r = VisionRenderer();
    expect(r.hasMap, isFalse);
    final depth = Float32List(kDepthSize * kDepthSize);
    for (var i = 0; i < depth.length; i++) {
      depth[i] = (i % kDepthSize) / kDepthSize; // a ramp: edges and contours to draw
    }
    r.update(depth);
    final img = await tester.runAsync(() => r.render(0.3, SightPalette.sonar));
    expect(img!.width, VisionRenderer.size);
    img.dispose();
  });

  group('grip advice', () {
    GripLevel level(double tilt, {double roll = 0, double db = double.nan, double? taught, bool see = false}) =>
        GripAdvice.of(tilt: tilt, roll: roll, directDb: db, taughtTilt: taught, checkTaught: see).level;

    test('ideal tilt is OK', () => expect(level(18), GripLevel.ok));
    test('bolt upright: tilt the top toward you', () => expect(level(0), GripLevel.warn));
    test('speaker aimed at the floor', () => expect(level(-20), GripLevel.warn));
    test('lying flat is bad', () => expect(level(70), GripLevel.bad));
    test('landscape is bad', () => expect(level(18, roll: 80), GripLevel.bad));
    test('covered speaker beats everything', () => expect(level(18, db: -12), GripLevel.bad));
    test('a few dB quieter is fine', () => expect(level(18, db: -3), GripLevel.ok));
    test('SEE warns when the grip differs from teaching',
        () => expect(level(25, taught: 5, see: true), GripLevel.warn));
    test('TEACH ignores the taught grip', () => expect(level(25, taught: 5), GripLevel.ok));
  });
}
