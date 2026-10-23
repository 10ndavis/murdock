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

  group('grip advice (upside-down layout: positive tilt = top away from you)', () {
    GripLevel level(double tilt, {double roll = 0, double db = double.nan, double? taught, bool see = false}) =>
        GripAdvice.of(tilt: tilt, roll: roll, directDb: db, taughtTilt: taught, checkTaught: see).level;
    String title(double t) => GripAdvice.of(tilt: t, roll: 0, directDb: double.nan).title;

    test('reading angle is OK', () => expect(level(35), GripLevel.ok));
    test('bolt upright: tilt the top away a little', () => expect(title(5), 'TILT THE TOP AWAY A LITTLE'));
    test('top toward you: speaker aims at you', () => expect(title(-20), 'TOP LEANING TOWARD YOU'));
    test('nearly flat: raise the top', () => expect(title(60), 'RAISE THE TOP A LITTLE'));
    test('flat is bad', () => expect(level(85), GripLevel.bad));
    test('landscape is bad', () => expect(level(35, roll: 80), GripLevel.bad));
    test('covered speaker beats everything', () => expect(level(35, db: -12), GripLevel.bad));
    test('a few dB quieter is fine', () => expect(level(35, db: -3), GripLevel.ok));
    test('SEE warns when the grip differs from learning', () => expect(level(45, taught: 25, see: true), GripLevel.warn));
    test('without that check the same grip is fine', () => expect(level(45, taught: 25), GripLevel.ok));
  });

  testWidgets('invalid AI output never breaks the renderer', (tester) async {
    final r = VisionRenderer();
    final bad = Float32List(kDepthSize * kDepthSize)..fillRange(0, 100, double.nan);
    bad[200] = double.infinity;
    r.update(bad);
    final img = await tester.runAsync(() => r.render(0.5, SightPalette.radar));
    expect(img!.width, VisionRenderer.size);
    img.dispose();
  });
}
