"""Train the full student (trunk + head) on a PC, then export for the phone.

  python train.py --data ~/murdock_dump --epochs 40
  python export_tflite.py --weights student.weights.h5

--data takes any folder of .mdk files: pulled from the phone
(adb pull /sdcard/Android/data/com.murdock.murdock/files/samples) and/or
produced by prepare_batvision.py. Use --pretrained to fine-tune from an earlier run.
"""

import argparse

import numpy as np
import tensorflow as tf

from dataset import load_dir
from murdock_model import build_student, depth_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained")
    ap.add_argument("--out", default="student.weights.h5")
    args = ap.parse_args()

    parts = [load_dir(d) for d in args.data]
    echo = np.concatenate([p[0] for p in parts])
    depth = np.concatenate([p[1] for p in parts])

    # Hold out a contiguous tail: neighbouring pings are near-duplicates, so a random
    # split would leak and make validation look far better than reality.
    split = int(len(echo) * 0.9)
    model, _, _ = build_student()
    if args.pretrained:
        model.load_weights(args.pretrained)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=depth_loss)
    model.fit(
        echo[:split], depth[:split],
        validation_data=(echo[split:], depth[split:]),
        epochs=args.epochs, batch_size=args.batch, shuffle=True,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4),
        ],
    )
    print(f"best weights -> {args.out}")


if __name__ == "__main__":
    main()
