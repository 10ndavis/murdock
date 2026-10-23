# Murdock — see in the dark with sound

A Flutter Android app that turns your phone into an active sonar. The speaker sends
near-ultrasonic chirps, both mics record the echoes, and a small on-device neural network
turns those echoes into a depth picture. The picture is drawn in a Daredevil "radar sense"
style (crimson) or a *Dark Knight* sonar style (blue). The camera is used only while the
lights are on: it teaches the network what the room looks like.

```
speaker ──chirp 17.5–21.5 kHz──► room ──echoes──► mic L/R
                                                     │ matched filter, 128 range bins × 2 mics
                                                     ▼
                     echo stack (16 pings × 128 × 2) ──► student CNN ──► 32×32 nearness ──► renderer
                                                                ▲
            lights on: camera ──► MiDaS depth (teacher) ────────┘ trains the student on device
```

## Is this possible? (short answer)

**Partly.** Everything in this repo runs. The honest limits are physical:

| What you get | Why |
|---|---|
| **Distance is real and good.** Echo range resolution is about 4 cm out to about 5.5 m. | 4 kHz of chirp bandwidth gives c / 2B ≈ 4.3 cm. You can see this live in the **SCOPE** tab, with no AI involved. |
| **Direction is weak.** You get at most a rough left/right cue. | A phone has one speaker and 2–3 mics a few cm apart, and Android often exposes only one or two of them. That isn't enough aperture to form an image the way the eye or a sonar array does. |
| **The "picture" is partly learned guesswork.** | The network learns the link between "echo pattern" and "what the camera saw" in places you trained it. In familiar rooms it produces believable, blurry depth layouts, like the published research. In a new room it falls back on priors. It won't render a crisp silhouette like the film. |
| **Some phones can't do it.** | Some phones low-pass audio above about 20 kHz, have weak tweeters, or run noise suppression. The `DIRECT` level on the status line and the SCOPE view show quickly whether yours works. |

Research reports similar results. BatVision predicted coarse depth maps from a binaural mic
and chirps on a robot. On a phone, AIM (MobiSys '18) got real acoustic images, but only by
sweeping the phone to build a synthetic aperture. Both are listed below.

## Using the app

1. Turn the **media volume up**. Chirps at 17.5–21.5 kHz are near-inaudible to most adults,
   but children and pets may hear them. Don't hold the speaker against anyone's ear.
   On RADAR, **AUTO-TUNE** (stand 4–6 ft from a wall, keep still ~10 s) tries five chirp
   loudness levels and keeps the quietest one within 1 dB of the clearest echo, then relearns
   the speaker filter at it. The **LOUDNESS** slider does the same by hand, with live ECHO
   CLARITY in dB. Only loudness is adjustable: the echo AI was trained on one exact chirp
   (17.5–21.5 kHz, 5 ms, ~23/s), so its pitch and timing stay fixed.
2. **SCOPE**: raw sonar. The top half is an echogram (distance → across, time ↓). The bottom
   half is a radar fan. Static clutter is subtracted, so moving things light up. Walk toward a
   wall and watch the ring move in.
3. **TRAIN**: on first use, downloads the camera depth teacher (MiDaS v2.1 small, about 66 MB,
   from the MiDaS GitHub release). With the lights on, walk slowly around the room. Every camera
   frame becomes a label, the echoes heard at that moment become the input, and the student's
   dense head trains on the phone. Each sample is also saved to
   `Android/data/com.murdock.murdock/files/samples/*.mdk` for PC training.
4. **VISION**: lights off. The echo-only network's output is drawn as glowing edges,
   iso-depth contours, and a pulse wavefront that sweeps outward.
5. The ✱ button switches between the red and blue looks. **SAVE/RESET BRAIN** keeps or
   discards on-device learning. The app also saves automatically when it is backgrounded.

## Local AI: what runs where

* **Student** (`assets/models/echo_student.tflite`, about 18 MB): a conv trunk that turns the
  echo stack into a 512-d latent, plus a dense head that outputs a 32×32 nearness map. It runs
  on LiteRT (TensorFlow Lite) on the phone and exports four signatures: `infer`, `train`,
  `get_weights`, and `set_weights`.
  * On-device training updates **only the head**. Training the conv layers needs TensorFlow
    "flex" ops (for example `Conv2DBackpropFilter`), which stock LiteRT doesn't include.
    The head's activations use hand-written gradients so the `train` signature uses only
    builtin ops. `ml/export_tflite.py` enforces this.
  * The shipped model is **pre-trained on BatVision** (see below); its weights are kept in
    `ml/weights/batvision_student.weights.h5` for further training on a PC. When the app ships
    a new model, on-device learning saved against the old one is discarded automatically.
* **PC training** (`ml/`):
  ```bash
  cd ml && pip install -r requirements.txt
  adb pull /sdcard/Android/data/com.murdock.murdock/files/samples ./phone
  python train.py --data ./phone --pretrained weights/batvision_student.weights.h5 --epochs 40
  python export_tflite.py --weights student.weights.h5 # -> assets/models/echo_student.tflite
  ```

### BatVision pre-training: what it does and doesn't do

`ml/stream_batvision.py` streams the 168 GB dataset from its data.gouv mirror and reduces it
on the fly (nothing large is stored): 41,474 train / 8,243 val / 5,622 test samples using the
official splits. Two problems had to be solved first:

* **Alignment.** BatVision is hardware-synchronised, but in the phone's 17.5–21.5 kHz band its
  direct speaker sound is weaker than nearby echoes, so "loudest peak" put echoes at the
  wrong distance in most recordings (25% correct on BV1, ~10% on BV2). Each dataset is now
  aligned at its measured emission point.
* **Scaling.** The phone and BatVision now share one input definition (`echo_features`):
  speaker sound removed, first ~7 in masked, scaled by the strongest echo.

Results on the held-out test rooms (`ml/evaluate.py`), compared with an audio-free guess:

| | L1 error (lower) | per-room layout correlation (higher) |
|---|---|---|
| no audio: average room | 0.220 | 0.545 |
| **shipped: phone band (17.5–21.5 kHz)** | **0.210** | 0.531 |
| diagnostic: full band (audible chirp) | 0.197 | 0.556 |

So BatVision mainly teaches the model what a **typical room** looks like, plus a little
distance sense. In the phone's near-silent band it doesn't learn to read a specific room;
even full-band audio only does modestly better (`ml/diagnose_fullband.py`). The per-room
picture has to come from TEACH on your own phone.

That on-device learning works on top of the pre-trained model: `ml/simulate_teach.py` runs the
shipped `.tflite` on the phone's runtime (LiteRT 1.4.0) through a simulated 10-minute TEACH
session (12,000 head-training steps on frames from a set of rooms) and cuts L1 error on
held-out frames from the same rooms by 25% (0.188 → 0.142).

On-device training is guarded: gradients are clipped (norm 1.0), the learning rate is an
input to the `train` signature, and if a step or a prediction is ever invalid the app restores
the last good weights (snapshotted every 500 steps) and halves the rate. Invalid weights are
never saved, and saved weights that produce invalid output are discarded at startup.

The phone's **tilt and roll are inputs** to the head (`pose_features`), recorded with every
sample (`.mdp` files). The same room sounds and looks different at other angles, so the AI
learns per angle instead of needing one fixed grip; the app tracks how much it has learned
per 10° of tilt and says when you're at an angle it hasn't learned yet. Old pose-free
weights load with the new pose rows at zero (identical output at pose 0). In a simulated
two-grip session on the phone runtime, the angle input cut held-out error from 0.215 to
0.162.

A second, small **shapes head** (`ml/shapes.py`) predicts a few scene primitives from the same
latent and pose: the floor (horizon, nearness), wall panels to the left, ahead and right
(nearness, slant), and the two nearest obstacles as boxes, each with a confidence. Only shapes
reported steadily for about 2 seconds are used: SEE then paints them into the nearness map
(`paintShapes` in `lib/render/shapes.dart`, at most 60% of a pixel), so the normal echo renderer
draws them in the same laser look. The labels are fitted to the camera teacher's depth
(`ShapeFit.kt`, a line-for-line copy of `shapes.py`; `ml/check_shapes_kotlin.py` checks they
agree, to 3e-8 on 2,500 maps), so learning them needs no extra work from you and samples
recorded before the shapes head get labels too. The depth head is untouched: same output and
the same training step as before (exact on the phone runtime), and learning saved by the
previous model carries over. Honest numbers (`ml/train_shapes.py`, BatVision test rooms,
presence-weighted L1, lower is better):

| | shape error |
|---|---|
| no audio: average shapes | 0.196 |
| shapes fitted to the depth head's picture | 0.257 |
| **shapes head (pretrained on BatVision)** | **0.186** |

Like the depth head, it starts as mostly a room prior. On-device learning
(`ml/simulate_shapes.py`, 600 samples) improves it 4% in the rooms it was taught, and it
costs 3% in unseen rooms (depth: +16% / −1%). The shapes head learns at a tenth of the depth
head's rate; at the full rate it memorised the taught rooms (−20% in unseen ones).
* **Teacher**: MiDaS v2.1 small, 256×256, downloaded at runtime. Depth Anything V2 Small is
  more accurate and could replace it later. I didn't find a verified TFLite build of it, but
  Qualcomm AI Hub and Arm's ExecuTorch guide both run it on Android.

`ml/features.py` is a NumPy copy of the phone's DSP in `SonarEngine.kt`, so recordings from
any source can be turned into the same features.

## Open datasets and prior work that help

| Resource | What it is | How it helps here |
|---|---|---|
| [BatVision (BV1/BV2)](https://arxiv.org/abs/2303.07257), [project page](https://amandinebtto.github.io/Batvision-Dataset/) | Real chirp echoes from a binaural mic on a robot, plus synchronized RGB-D, indoors and outdoors. | The closest real-world match. Pretrain the trunk with `ml/prepare_batvision.py`, then fine-tune on phone data. Caveat: its chirp tops out at 20 kHz, so only the 17.5–20 kHz overlap carries over, and the mics differ. |
| [VisualEchoes (ECCV 2020)](https://vision.cs.utexas.edu/projects/visualEchoes) / SoundSpaces | Simulated echoes (3 ms, 20 Hz–20 kHz chirps) in Replica and Matterport3D scans. | Large synthetic pretraining set. You can re-simulate with the phone's chirp band and mic geometry. |
| [Beyond Image to Depth (CVPR 2021)](https://arxiv.org/abs/2103.08468) | Echo + image depth benchmark on Replica and Matterport3D (77 scenes). | Benchmarks and baselines for echo-based depth. |
| [Beyond Visual Field of View (2022)](https://arxiv.org/abs/2207.01136) | Echoes + vision for 3D beyond the camera's field of view. | Shows echoes can add depth outside the camera's view. |
| [AIM: Acoustic Imaging on a Mobile (MobiSys '18)](https://par.nsf.gov/servlets/purl/10063070) | Real acoustic images from an unmodified phone (Galaxy S7) using synthetic aperture. | The best route to sharper images: sweep the phone and fuse pings with the IMU (future work). |
| [Deep Room Recognition with inaudible echoes](https://arxiv.org/abs/1809.00531), [Echoic smartphone SLAM](https://arxiv.org/abs/2210.08493) | Phone chirp echoes plus deep learning for room ID and localization. | Confirms phone speakers and mics produce usable learned echo features. |

## Building

```bash
flutter pub get
flutter build apk --release      # or: flutter run
```

Requires Android 8.0+ (API 26). Tested here only with `flutter analyze`, `flutter test`, and
an APK build. It hasn't been run on a real phone yet, so expect to tune the chirp volume and
timing per device.

## Layout

```
lib/sonar/       Dart bridge to the native sonar/model
lib/render/      Daredevil/Batman renderer, raw sonar scope
lib/train/       camera → 256×256 RGB for the teacher
android/.../     SonarEngine.kt (speaker/mic DSP), EchoBrain.kt (student, on-device training),
                 TeacherDepth.kt (MiDaS), MainActivity.kt (channels)
ml/              model definition, PC training, TFLite export, BatVision converter
```
