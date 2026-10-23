"""Train the full student (trunk + head) on a PC, then export for the phone.

  python train.py --data ~/murdock_dump --epochs 40
  python train.py --data data/bv/train.mdk --val data/bv/val.mdk      # BatVision splits
  python export_tflite.py --weights student.weights.h5

--data / --val take .mdk files or folders of them: pulled from the phone
(adb pull /sdcard/Android/data/com.murdock.murdock/files/samples) and/or produced by
stream_batvision.py / prepare_batvision.py. Use --pretrained to fine-tune from an earlier run.
"""

import argparse

import numpy as np
import tensorflow as tf

from dataset import load_dir
from murdock_model import build_student, depth_loss, load_student_weights


def augment(inputs, depth):
    """Label-preserving variations so the model learns echo patterns, not rooms.

    inputs: (echo [STACK, RANGE_BINS, 2] log-scaled, pose [tilt, roll]); depth: [D, D].
    """
    echo, pose = inputs
    # Mirror the world: swap left/right mics, flip the depth map, and lean the other way.
    if tf.random.uniform([]) < 0.5:
        echo = echo[..., ::-1]
        depth = depth[:, ::-1]
        pose = pose * tf.constant([1.0, -1.0])
    # Overall loudness (log domain) and per-mic imbalance.
    echo = echo + tf.random.uniform([], -0.06, 0.06) + tf.random.uniform([1, 1, 2], -0.02, 0.02)
    # Background noise.
    echo = echo + tf.random.normal(tf.shape(echo), stddev=0.02)
    # +-1 range bin of alignment jitter.
    shift = tf.random.uniform([], -1, 2, dtype=tf.int32)
    echo = tf.roll(echo, shift, axis=1)
    return (tf.clip_by_value(echo, 0.0, 1.0), pose), depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", help="held-out set; default: last 10%% of --data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained")
    ap.add_argument("--out", default="student.weights.h5")
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    def load(dirs):
        parts = [load_dir(d) for d in dirs]
        echo, pose, depth = (np.concatenate([p[i] for p in parts]) for i in range(3))
        return (echo, pose), depth

    train = load(args.data)
    if args.val:
        val = load(args.val)
    else:
        # Hold out a contiguous tail: neighbouring pings are near-duplicates, so a random
        # split would leak and make validation look far better than reality.
        (echo, pose), depth = train
        split = int(len(echo) * 0.9)
        train = (echo[:split], pose[:split]), depth[:split]
        val = (echo[split:], pose[split:]), depth[split:]
    model, _, _ = build_student()
    if args.pretrained:
        load_student_weights(model, args.pretrained)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=depth_loss)
    ds = tf.data.Dataset.from_tensor_slices(train).shuffle(min(len(train[0]), 50000))
    if not args.no_augment:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)
    model.fit(
        ds,
        validation_data=val,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ],
    )
    print(f"best weights -> {args.out}")


if __name__ == "__main__":
    main()
