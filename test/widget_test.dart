import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:murdock/render/palette.dart';
import 'package:murdock/render/vision_renderer.dart';
import 'package:murdock/sonar/sonar_channel.dart';

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
}
