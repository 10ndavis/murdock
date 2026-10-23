import 'package:flutter/material.dart';

import '../render/palette.dart';
import 'hud.dart';

/// "How does this work / what do I do" walkthrough. Shown on first launch and from the ? button.
class GuidePage extends StatefulWidget {
  const GuidePage({super.key, required this.palette});

  final SightPalette palette;

  @override
  State<GuidePage> createState() => _GuidePageState();
}

class _Step {
  const _Step(this.icon, this.kicker, this.title, this.body);

  final IconData icon;
  final String kicker;
  final String title;
  final String body;
}

const _steps = [
  _Step(Icons.graphic_eq, 'HOW IT WORKS', 'Your phone becomes a bat',
      'Murdock plays a very high-pitched chirp from your speaker about 20 times a second. '
          'It bounces off walls, furniture and people, and your microphones hear the echoes.\n\n'
          'Most adults can\'t hear it. Kids and pets might.\n\n'
          'Turn your MEDIA VOLUME UP, and don\'t cover the speaker or mics.'),
  _Step(Icons.back_hand_outlined, 'HOLD IT LIKE THIS', 'Grip matters',
      'At these high pitches the speaker sends sound out in a narrow beam, mostly out of the BOTTOM edge.\n\n'
          '•  Portrait, like taking a photo, with the back camera facing the room.\n'
          '•  Arm partly out, about a foot from your chest.\n'
          '•  Tilt the top toward you about 15–20°, so the bottom speaker points ahead instead of at the floor.\n'
          '•  Hold it by the sides. Fingers off the bottom edge (speaker and mic).\n'
          '•  Use the SAME grip when you teach and when you see.\n\n'
          'The GRIP strip on screen checks all of this live and tells you what to fix.'),
  _Step(Icons.radar, 'STEP 1 · RADAR', 'Check your phone can hear',
      'Open RADAR. This is the raw sonar, with no AI involved.\n\n'
          'The ECHO TRACE graph shows echo strength by distance. Hold the phone 2 ft from a wall: '
          'a bump appears at about 2 ft. Step back and it slides right. That bump is the echo.\n\n'
          'The first time, Murdock spends 2 seconds learning the sound that goes straight from your '
          'speaker to your mic, so it can filter it out. If it ever looks wrong, tap RECALIBRATE SPEAKER.\n\n'
          'If the ECHO meter at the top stays empty, your phone\'s speaker or mic can\'t do high '
          'frequencies and the rest of the app won\'t work well.'),
  _Step(Icons.school, 'STEP 2 · LEARN', 'Let it learn with the lights on',
      'Echoes alone don\'t say what things look like, so Murdock learns your rooms.\n\n'
          'Open SEE with CAMERA ASSIST on. While the camera can see, a camera depth AI works out how far '
          'everything is, and the echo AI learns to match it from sound. Everything stays on your phone.\n\n'
          'Lights on, walk slowly around for a few minutes, pointing the phone at walls, doorways, '
          'furniture and people, until LEARNING reaches 100%. It keeps improving after that: the more it '
          'sees, the better it gets.'),
  _Step(Icons.visibility, 'STEP 3 · SEE', 'See in the dark',
      'SEE always shows the echo view. Where the camera can still see (even dim light), it makes that '
          'view more accurate, like a highly trained echo sense, without turning into a camera image. As '
          'it gets darker, the picture comes from sound alone. The banner shows the mix (CAMERA ASSIST %).\n\n'
          'Brighter means closer. Each sonar ping sweeps outward and lights up surfaces as it reaches them.\n\n'
          'CAMERA VIEW shows the real camera; ASSIST OFF gives pure sound and saves battery.'),
  _Step(Icons.info_outline, 'BE REALISTIC', 'What to expect',
      'Distance is the strong part: echoes measure it to within about 2 inches, out to roughly 18 ft.\n\n'
          'Shapes are blurry, and left/right is rough. A phone has only a couple of microphones, '
          'so the AI fills in detail from what it learned in your rooms.\n\n'
          'Out of the box, the AI is pre-trained on BatVision (real echo recordings with depth photos), '
          'so it knows what rooms typically look like, but not yours. It works best in rooms it has learned. '
          'In total darkness, expect blurry outlines even after lots of learning: two microphones '
          'can only hear so much.'),
];

class _GuidePageState extends State<GuidePage> {
  final _pages = PageController();
  int _index = 0;

  @override
  void dispose() {
    _pages.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final p = widget.palette;
    final last = _index == _steps.length - 1;
    return Scaffold(
      backgroundColor: Hud.bg,
      body: SafeArea(
        child: Column(children: [
          Align(
            alignment: Alignment.centerRight,
            child: TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text('SKIP', style: Hud.label(p.accent.withValues(alpha: 0.7))),
            ),
          ),
          Expanded(
            child: PageView.builder(
              controller: _pages,
              itemCount: _steps.length,
              onPageChanged: (i) => setState(() => _index = i),
              itemBuilder: (_, i) {
                final s = _steps[i];
                return SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 28),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    const SizedBox(height: 24),
                    Container(
                      padding: const EdgeInsets.all(18),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: p.accent, width: 1.5),
                        boxShadow: [BoxShadow(color: p.accent.withValues(alpha: 0.5), blurRadius: 24)],
                      ),
                      child: Icon(s.icon, color: p.accent, size: 36),
                    ),
                    const SizedBox(height: 28),
                    Text(s.kicker, style: Hud.label(p.accent, size: 11)),
                    const SizedBox(height: 10),
                    Text(s.title, style: Hud.title(Hud.text, size: 28)),
                    const SizedBox(height: 20),
                    Text(s.body, style: Hud.body.copyWith(fontSize: 15)),
                  ]),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(24),
            child: Row(children: [
              for (var i = 0; i < _steps.length; i++)
                AnimatedContainer(
                  duration: const Duration(milliseconds: 250),
                  width: i == _index ? 22 : 6,
                  height: 6,
                  margin: const EdgeInsets.only(right: 6),
                  decoration: BoxDecoration(
                    color: i == _index ? p.accent : p.accent.withValues(alpha: 0.25),
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),
              const Spacer(),
              GlowButton(
                label: last ? 'START' : 'NEXT',
                icon: last ? Icons.bolt : Icons.arrow_forward,
                filled: last,
                palette: p,
                onTap: () => last
                    ? Navigator.pop(context)
                    : _pages.nextPage(duration: const Duration(milliseconds: 300), curve: Curves.easeOut),
              ),
            ]),
          ),
        ]),
      ),
    );
  }
}
