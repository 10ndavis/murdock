import 'package:flutter/material.dart';

import '../render/palette.dart';
import '../sonar/sonar_channel.dart';
import 'hud.dart';

enum _Level { ok, warn, bad }

class _Row {
  const _Row(this.label, this.value, this.level, [this.fix]);

  final String label;
  final String value;
  final _Level level;
  final String? fix;
}

/// Plain-language health check of every link in the chain: speaker → room → mic → DSP → AI.
class SystemCheck extends StatelessWidget {
  const SystemCheck({super.key, required this.stats, required this.signal, required this.lastError, required this.palette});

  final SonarStats? stats;
  final double signal;
  final String? lastError;
  final SightPalette palette;

  List<_Row> _rows() {
    final s = stats;
    if (s == null) {
      return const [_Row('SONAR', 'starting…', _Level.warn, 'If this never changes, close and reopen the app.')];
    }
    final rows = <_Row>[];
    rows.add(s.speaker == 'playing'
        ? const _Row('SPEAKER', 'chirping', _Level.ok)
        : _Row('SPEAKER', s.speaker, _Level.bad, 'Another app may be using audio. Close it and reopen Murdock.'));
    final vol = (s.volume * 100).round();
    rows.add(vol >= 70
        ? _Row('MEDIA VOLUME', '$vol%', _Level.ok)
        : _Row('MEDIA VOLUME', '$vol%', vol == 0 ? _Level.bad : _Level.warn,
            'Press volume UP while Murdock is open. The chirp needs to be loud to echo.'));
    if (s.mic.startsWith('failed')) {
      rows.add(_Row('MICROPHONE', s.mic, _Level.bad, 'Check microphone permission in Android settings.'));
    } else {
      final heard = s.micDb > -85;
      rows.add(_Row('MICROPHONE', '${s.mic} · ${s.source}${s.stereo ? '' : ' (1 mic)'}',
          s.mic == 'listening' ? _Level.ok : _Level.warn));
      rows.add(_Row('MIC LEVEL', '${s.micDb.toStringAsFixed(0)} dB', heard ? _Level.ok : _Level.bad,
          heard ? null : 'The mic is returning silence. Another app may be recording, or the mic is blocked.'));
    }
    rows.add(s.pingsPerSec >= 15
        ? _Row('PINGS', '${s.pingsPerSec}/s', _Level.ok)
        : _Row('PINGS', '${s.pingsPerSec}/s', s.pingsPerSec == 0 ? _Level.bad : _Level.warn,
            'The app can\'t lock onto its own chirp. Turn the volume up and don\'t cover the bottom of the phone.'));
    final pct = (signal * 100).round();
    rows.add(signal > 0.3
        ? _Row('ECHO CLARITY', '$pct%', _Level.ok)
        : _Row('ECHO CLARITY', '$pct%', signal > 0.08 ? _Level.warn : _Level.bad,
            'Echoes are buried in noise. Raise the volume and try a quiet room. Some phones can\'t play or '
            'record this high a pitch at all.'));
    rows.add(s.brain == 'ready'
        ? const _Row('ECHO AI', 'loaded', _Level.ok)
        : _Row('ECHO AI', s.brain, s.brain == 'loading' ? _Level.warn : _Level.bad));
    return rows;
  }

  @override
  Widget build(BuildContext context) {
    final rows = _rows();
    Color colour(_Level l) => switch (l) {
          _Level.ok => const Color(0xFF3DFFB5),
          _Level.warn => const Color(0xFFFFC14D),
          _Level.bad => const Color(0xFFFF4D6A),
        };
    return Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
      Text('SYSTEM CHECK', style: Hud.label(palette.accent, size: 11)),
      const SizedBox(height: 10),
      for (final r in rows) ...[
        Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Container(
            width: 8,
            height: 8,
            margin: const EdgeInsets.only(top: 4, right: 10),
            decoration: BoxDecoration(
              color: colour(r.level),
              shape: BoxShape.circle,
              boxShadow: [BoxShadow(color: colour(r.level), blurRadius: 6)],
            ),
          ),
          SizedBox(width: 118, child: Text(r.label, style: Hud.label(Hud.dim, size: 10))),
          Expanded(child: Text(r.value, style: const TextStyle(color: Hud.text, fontSize: 12))),
        ]),
        if (r.fix != null)
          Padding(
            padding: const EdgeInsets.only(left: 18, top: 2, bottom: 4),
            child: Text(r.fix!, style: TextStyle(color: colour(r.level), fontSize: 11.5, height: 1.35)),
          ),
        const SizedBox(height: 6),
      ],
      if (lastError != null) ...[
        const SizedBox(height: 4),
        Text('LAST ERROR', style: Hud.label(const Color(0xFFFF4D6A), size: 10)),
        const SizedBox(height: 2),
        SelectableText(lastError!, style: const TextStyle(color: Hud.dim, fontSize: 11)),
      ],
    ]);
  }
}
