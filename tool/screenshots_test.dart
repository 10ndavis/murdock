// Renders the main screens with fake sonar data into docs/screenshots/.
//
//   flutter test tool/screenshots_test.dart --update-goldens
//
// Lives outside test/ so CI doesn't treat the PNGs as goldens to compare against.
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:murdock/home_page.dart';
import 'package:murdock/main.dart';
import 'package:murdock/render/palette.dart';
import 'package:murdock/sonar/sonar_channel.dart';
import 'package:murdock/ui/guide.dart';

const _method = MethodChannel('murdock/sonar');
const _events = EventChannel('murdock/frames');

Future<void> _loadFonts() async {
  Future<void> load(String family, String path) async {
    final loader = FontLoader(family)..addFont(Future.value(ByteData.sublistView(File(path).readAsBytesSync())));
    await loader.load();
  }

  final flutterRoot = Platform.environment['FLUTTER_ROOT']!;
  await load('monospace', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf');
  await load('MaterialIcons', '$flutterRoot/bin/cache/artifacts/material_fonts/MaterialIcons-Regular.otf');
}

/// A fake room: floor rising toward the viewer, a wall, a doorway and a person-ish blob.
Float32List _room() {
  final d = Float32List(kDepthSize * kDepthSize);
  for (var y = 0; y < kDepthSize; y++) {
    for (var x = 0; x < kDepthSize; x++) {
      var v = 0.25 + 0.5 * math.max(0, (y - 16) / 16);
      if (x > 20 && x < 27 && y > 6 && y < 22) v = 0.12; // doorway
      final dx = (x - 10) / 4.0, dy = (y - 15) / 9.0;
      if (dx * dx + dy * dy < 1) v = 0.8; // person
      d[y * kDepthSize + x] = v;
    }
  }
  return d;
}

Float32List _profile(int t) {
  final p = Float32List(kRangeBins * kChannels);
  for (var b = 0; b < kRangeBins; b++) {
    var v = 0.35 * math.exp(-b / 60);
    for (final (at, a) in [(18, 0.55), (47, 0.45), (80 + (t % 10), 0.35)]) {
      v += a * math.exp(-math.pow((b - at) / 2.5, 2));
    }
    p[b * 2] = v.clamp(0, 1);
    p[b * 2 + 1] = (v * 0.92).clamp(0, 1);
  }
  return p;
}

void _mockNative({required int samples}) {
  final messenger = TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  messenger.setMockMethodCallHandler(_method, (call) async => switch (call.method) {
        'getFlag' => true,
        'requestPermission' => true,
        'teacherReady' => true,
        'status' => {
            'stereo': true,
            'source': 'unprocessed',
            'samples': samples,
            'steps': samples * 4,
            'totalSamples': samples,
            'teacherReady': true,
            'samplesDir': '/sdcard/Android/data/com.murdock.murdock/files/samples',
          },
        _ => null,
      });
  messenger.setMockStreamHandler(
    _events,
    MockStreamHandler.inline(onListen: (args, sink) {
      for (var t = 0; t < 40; t++) {
        sink.success({'type': 'ping', 'profile': _profile(t), 'direct': 3.2, 'depth': _room(), 'samples': samples});
      }
    }),
  );
}

Future<void> _settle(WidgetTester tester) async {
  for (var i = 0; i < 12; i++) {
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 40)));
    await tester.pump(const Duration(milliseconds: 100));
  }
}

void main() {
  setUpAll(_loadFonts);

  Future<void> shoot(WidgetTester tester, String name, {required int samples, String? tab}) async {
    tester.view.physicalSize = const Size(1080, 2340);
    tester.view.devicePixelRatio = 2.75;
    _mockNative(samples: samples);
    await tester.pumpWidget(const MurdockApp());
    await _settle(tester);
    if (tab != null) {
      await tester.tap(find.text(tab));
      await _settle(tester);
    }
    await expectLater(find.byType(HomePage), matchesGoldenFile('../docs/screenshots/$name.png'));
  }

  testWidgets('radar', (t) => shoot(t, 'radar', samples: 0));
  testWidgets('see (untrained)', (t) => shoot(t, 'see_untrained', samples: 0, tab: 'SEE'));
  testWidgets('see (trained)', (t) => shoot(t, 'see', samples: 1400, tab: 'SEE'));
  testWidgets('teach', (t) => shoot(t, 'teach', samples: 600, tab: 'TEACH'));
  testWidgets('guide', (tester) async {
    tester.view.physicalSize = const Size(1080, 2340);
    tester.view.devicePixelRatio = 2.75;
    await tester.pumpWidget(MaterialApp(
      theme: ThemeData(fontFamily: 'monospace', brightness: Brightness.dark),
      home: const GuidePage(palette: SightPalette.radar),
    ));
    await tester.pump();
    await expectLater(find.byType(GuidePage), matchesGoldenFile('../docs/screenshots/guide.png'));
  });
}
