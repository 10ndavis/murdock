import 'package:flutter/material.dart';

import '../render/palette.dart';
import '../sonar/sonar_channel.dart';
import 'hud.dart';

/// "+14 dB", or "–" before anything was heard.
String formatDb(double db) => db.isNaN ? '–' : '${db >= 0 ? '+' : ''}${db.toStringAsFixed(0)} dB';

String _pct(double v) => '${(v * 100).round()}%';

/// RADAR's sound controls: chirp LOUDNESS (live, with the echo clarity it gives) and
/// AUTO-TUNE, which tries loudness levels, each speaker alone and every mic pair against a
/// wall, and keeps the clearest.
///
/// Only loudness is adjustable: the echo AI was trained on one exact chirp (17.5–21.5 kHz,
/// 5 ms, 23 a second), so changing its pitch or timing would make what it learned stop
/// matching.
class SoundPanel extends StatelessWidget {
  const SoundPanel({
    super.key,
    required this.palette,
    required this.loudness,
    required this.clarityDb,
    required this.tune,
    required this.onLoudness,
    required this.onAutoTune,
    required this.onCancel,
  });

  final SightPalette palette;
  final double loudness;
  final double clarityDb;

  /// Latest AUTO-TUNE event (null = never run this session).
  final TuneEvent? tune;
  final ValueChanged<double> onLoudness;
  final VoidCallback onAutoTune;
  final VoidCallback onCancel;

  static const _warn = Color(0xFFFFC14D);

  @override
  Widget build(BuildContext context) {
    final t = tune;
    return HudPanel(
      palette: palette,
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: t != null && t.running ? _running(t) : _idle(t),
      ),
    );
  }

  List<Widget> _idle(TuneEvent? t) => [
        Row(children: [
          Text('LOUDNESS', style: Hud.label(Hud.dim, size: 9)),
          Expanded(
            child: SliderTheme(
              data: SliderThemeData(
                trackHeight: 2,
                activeTrackColor: palette.accent,
                inactiveTrackColor: palette.accent.withValues(alpha: 0.2),
                thumbColor: palette.accent,
                overlayColor: palette.accent.withValues(alpha: 0.15),
                thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
                overlayShape: const RoundSliderOverlayShape(overlayRadius: 16),
              ),
              child: Slider(
                value: loudness.clamp(kMinLoudness, kMaxLoudness),
                min: kMinLoudness,
                max: kMaxLoudness,
                onChanged: onLoudness,
              ),
            ),
          ),
          SizedBox(width: 38, child: Text(_pct(loudness), style: Hud.label(Hud.text, size: 11))),
          _tuneChip(),
        ]),
        Text(
          'ECHO CLARITY ${formatDb(clarityDb)} · HIGHER = CLEARER',
          style: Hud.label(palette.accent, size: 8.5),
        ),
        if (t != null && t.state == 'done') ...[
          const SizedBox(height: 4),
          Text('AUTO-TUNED · ${t.bestLabel.isNotEmpty ? t.bestLabel : 'BEST ${_pct(t.best)}'} · CLARITY ${formatDb(t.bestDb)}',
              style: Hud.label(Hud.text, size: 9)),
        ] else if (t != null && t.state == 'failed') ...[
          const SizedBox(height: 4),
          Text('AUTO-TUNE FAILED · ${t.message}', style: const TextStyle(color: _warn, fontSize: 11, height: 1.3)),
        ],
      ];

  List<Widget> _running(TuneEvent t) {
    final step = t.clarity.where((c) => !c.isNaN).length;
    final what = switch (t.phase) {
      'final_settle' || 'calibrate' => 'RELEARNING SPEAKER SOUND',
      _ => 'TESTING ${t.testingLabel.isNotEmpty ? t.testingLabel : _pct(t.testing)} · '
          '${(step + 1).clamp(1, t.levels.length)}/${t.levels.length}',
    };
    return [
      Row(children: [
        Icon(Icons.auto_awesome, size: 14, color: palette.accent),
        const SizedBox(width: 6),
        Expanded(child: Text('AUTO-TUNE · $what', style: Hud.label(palette.accent, size: 9.5))),
        InkWell(
          onTap: onCancel,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
            child: Text('CANCEL', style: Hud.label(Hud.dim, size: 9)),
          ),
        ),
      ]),
      const SizedBox(height: 6),
      ClipRRect(
        borderRadius: BorderRadius.circular(2),
        child: LinearProgressIndicator(
          value: t.progress,
          minHeight: 3,
          color: palette.accent,
          backgroundColor: palette.accent.withValues(alpha: 0.15),
        ),
      ),
      const SizedBox(height: 6),
      Text('Keep the top edge aimed at the wall 4–6 ft away. Keep still.', style: Hud.body.copyWith(fontSize: 12)),
      const SizedBox(height: 6),
      _levels(t),
    ];
  }

  /// One bar per loudness tried: taller = clearer echo.
  Widget _levels(TuneEvent t) {
    final known = t.clarity.where((c) => !c.isNaN);
    final top = known.isEmpty ? 1.0 : known.reduce((a, b) => a > b ? a : b).clamp(1.0, 60.0);
    return Row(children: [
      for (var i = 0; i < t.levels.length; i++)
        Expanded(
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Container(
              height: 3 + 22 * (t.clarity[i].isNaN ? 0 : (t.clarity[i] / top).clamp(0.0, 1.0)),
              width: 14,
              alignment: Alignment.bottomCenter,
              decoration: BoxDecoration(
                color: (i < t.clipped.length && t.clipped[i] ? _warn : palette.accent)
                    .withValues(alpha: t.clarity[i].isNaN ? 0.2 : 0.85),
                boxShadow: t.clarity[i].isNaN ? null : [BoxShadow(color: palette.accent.withValues(alpha: 0.4), blurRadius: 4)],
              ),
            ),
            const SizedBox(height: 2),
            Text(
              t.clarity[i].isNaN ? (i < t.labels.length ? t.labels[i] : _pct(t.levels[i])) : formatDb(t.clarity[i]),
              style: Hud.label(Hud.dim, size: 7.5),
            ),
          ]),
        ),
    ]);
  }

  Widget _tuneChip() => InkWell(
        onTap: onAutoTune,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: palette.accent.withValues(alpha: 0.7)),
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.auto_awesome, size: 13, color: palette.accent),
            const SizedBox(width: 5),
            Text('AUTO-TUNE', style: Hud.label(palette.accent, size: 9)),
          ]),
        ),
      );
}
