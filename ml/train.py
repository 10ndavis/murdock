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
from murdock_model import build_student, depth_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", help="held-out set; default: last 10%% of --data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained")
    ap.add_argument("--out", default="student.weights.h5")
    args = ap.parse_args()

    def load(dirs):
        parts = [load_dir(d) for d in dirs]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    echo, depth = load(args.data)
    if args.val:
        val = load(args.val)
        train = (echo, depth)
    else:
        # Hold out a contiguous tail: neighbouring pings are near-duplicates, so a random
        # split would leak and make validation look far better than reality.
        split = int(len(echo) * 0.9)
        train, val = (echo[:split], depth[:split]), (echo[split:], depth[split:])
    model, _, _ = build_student()
    if args.pretrained:
        model.load_weights(args.pretrained)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=depth_loss)
    model.fit(
        *train,
        validation_data=val,
        epochs=args.epochs, batch_size=args.batch, shuffle=True,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ],
    )
    print(f"best weights -> {args.out}")


if __name__ == "__main__":
    main()
