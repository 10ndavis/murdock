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

  group('grip advice (angle is an AI input; only physics and coverage matter)', () {
    GripLevel level(double tilt, {double roll = 0, double db = double.nan, List<int> cov = const [], bool see = false}) =>
        GripAdvice.of(tilt: tilt, roll: roll, directDb: db, coverage: cov, checkCoverage: see).level;
    String title(double tilt, {List<int> cov = const [], bool see = false}) =>
        GripAdvice.of(tilt: tilt, roll: 0, directDb: double.nan, coverage: cov, checkCoverage: see).title;
    // 12 bins of 10° from -30°: bin 6 = 30..40°, bin 9 = 60..70°.
    final learnedAt35 = List<int>.filled(12, 0)..[6] = 500;

    test('any reasonable angle is fine physically', () {
      for (final t in [0.0, 20.0, 35.0, 55.0, 70.0]) {
        expect(level(t), GripLevel.ok, reason: '$t°');
      }
    });
    test('speaker aimed at you is bad', () => expect(title(-20), 'SPEAKER AIMED AT YOU'));
    test('flat is bad', () => expect(level(85), GripLevel.bad));
    test('landscape is bad', () => expect(level(35, roll: 80), GripLevel.bad));
    test('covered speaker beats everything', () => expect(level(35, db: -12), GripLevel.bad));
    test('learned angle in SEE: OK', () => expect(title(35, cov: learnedAt35, see: true), 'GRIP OK · LEARNED HERE'));
    test('new angle in SEE: warn, and point back to the learned one', () {
      final a = GripAdvice.of(tilt: 65, roll: 0, directDb: double.nan, coverage: learnedAt35, checkCoverage: true);
      expect(a.level, GripLevel.warn);
      expect(a.title, 'NEW ANGLE · 65°');
      expect(a.detail, contains('about 35°'));
    });
    test('RADAR ignores coverage', () => expect(level(65, cov: learnedAt35), GripLevel.ok));
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
