"""Pretrain the shapes head on a frozen trunk, and check it beats trivial baselines.

  python train_shapes.py --weights weights/batvision_student.weights.h5 \
      --train data/bv/train.mdk --val data/bv/val.mdk --test data/bv/test.mdk \
      --out weights/batvision_shapes.weights.h5

Labels come from shapes.fit_shapes on each sample's depth map (as the phone does with the
camera teacher). The trunk is frozen (same as on the phone), so its 512-d latents are
computed once, for each echo and for its left/right mirror image.

Compared on the test set (lower = better, presence-weighted L1 as in shape_loss):
  head       : the shapes head
  depth->fit : shapes fitted to the depth head's predicted map (no shapes head at all)
  mean       : the training set's average shapes (sees no audio)
"""

import argparse

import numpy as np
import tensorflow as tf

from dataset import load_dir
from murdock_model import SHAPE_DIM, build_shape_head, build_student, load_student_weights, shape_loss
from shapes import BOXES, FLOOR, MAX_BOXES, WALLS, fit_batch, presence_mask


def mirror_shapes(s):
    """Shapes of the left/right mirrored scene."""
    s = s.copy()
    left, right = s[:, WALLS:WALLS + 3].copy(), s[:, WALLS + 6:WALLS + 9].copy()
    s[:, WALLS:WALLS + 3], s[:, WALLS + 6:WALLS + 9] = right, left
    for k in range(3):
        s[:, WALLS + 3 * k + 2] = 1.0 - s[:, WALLS + 3 * k + 2]
    for k in range(MAX_BOXES):
        o = BOXES + 6 * k
        s[:, o + 2] = np.where(s[:, o] > 0, 1.0 - s[:, o + 2], 0.0)
    return s


def latents(trunk, echo, batch=256):
    return np.concatenate([trunk(echo[i:i + batch], training=False).numpy() for i in range(0, len(echo), batch)])


def score(pred, true):
    """Presence-weighted L1 plus a few readable numbers."""
    w = presence_mask(true)
    loss = float((w * np.abs(pred - true)).sum() / w.sum())
    out = {"loss": loss}
    for name, o in [("floor", FLOOR), ("wallL", WALLS), ("ahead", WALLS + 3), ("wallR", WALLS + 6), ("box1", BOXES)]:
        out[name + " present-acc"] = float(((pred[:, o] > 0.5) == (true[:, o] > 0.5)).mean())
    has = true[:, BOXES] > 0.5
    out["box1 x-err"] = float(np.abs(pred[has, BOXES + 2] - true[has, BOXES + 2]).mean()) if has.any() else float("nan")
    out["box1 near-err"] = float(np.abs(pred[has, BOXES + 1] - true[has, BOXES + 1]).mean()) if has.any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="trunk + depth head (train.py output)")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="shapes.weights.h5")
    args = ap.parse_args()

    model, trunk, head = build_student()
    load_student_weights(model, args.weights)

    def load(paths, mirror):
        parts = [load_dir(p) for p in paths]
        echo, pose, depth = (np.concatenate([p[i] for p in parts]) for i in range(3))
        shapes = fit_batch(depth)
        lat = latents(trunk, echo)
        if not mirror:
            return (lat, pose), shapes, echo, depth
        lat_m = latents(trunk, echo[..., ::-1].copy())
        pose_m = pose * np.array([1.0, -1.0], np.float32)
        return (np.concatenate([lat, lat_m]), np.concatenate([pose, pose_m])), np.concatenate([shapes, mirror_shapes(shapes)]), echo, depth

    (tr_x, tr_y, _, _) = load(args.train, mirror=True)
    (va_x, va_y, _, _) = load(args.val, mirror=False)
    (te_x, te_y, te_echo, _) = load(args.test, mirror=False)
    print(f"train {len(tr_y)} (incl. mirrored), val {len(va_y)}, test {len(te_y)}; label means {np.round(tr_y.mean(0), 2)}")

    shape_head = build_shape_head()
    shape_head.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=shape_loss)
    shape_head.fit(
        tf.data.Dataset.from_tensor_slices(tr_x + (tr_y,)).map(lambda a, b, y: ((a, b), y))
        .shuffle(len(tr_y)).batch(64).prefetch(tf.data.AUTOTUNE),
        validation_data=(va_x, va_y),
        epochs=args.epochs,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3),
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        ],
    )
    shape_head.load_weights(args.out)

    pred = shape_head.predict(te_x, batch_size=256, verbose=0)
    depth_pred = head.predict(te_x, batch_size=256, verbose=0)
    mean = np.broadcast_to(tr_y.mean(0), te_y.shape)
    rows = [("head", pred), ("depth->fit", fit_batch(depth_pred)), ("mean", mean)]
    keys = list(score(pred, te_y).keys())
    print(f"\n{'':11s}" + "".join(f"{k:>18s}" for k in keys))
    for name, p in rows:
        s = score(p, te_y)
        print(f"{name:11s}" + "".join(f"{s[k]:18.4f}" for k in keys))
    print(f"\nshapes head -> {args.out}")
    assert pred.shape[1] == SHAPE_DIM


if __name__ == "__main__":
    main()
