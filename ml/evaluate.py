"""Does the echo model actually know anything? Compare it with trivial baselines.

  python evaluate.py --weights student.weights.h5 --train data/bv/train.mdk --test data/bv/test.mdk

Baselines (they see no audio at all):
  mean-map : the average nearness map of the training set (the "typical room")
  constant : a flat grey map at the training mean
If the model doesn't clearly beat mean-map, it is not reading anything from the echoes.
"""

import argparse

import numpy as np

from dataset import load_dir
from murdock_model import build_student


def metrics(pred, true):
    l1 = np.abs(pred - true).mean()
    # Edge agreement: what the renderer draws.
    gx = lambda a: np.diff(a, axis=2)
    gy = lambda a: np.diff(a, axis=1)
    edge = (np.abs(gx(pred) - gx(true)).mean() + np.abs(gy(pred) - gy(true)).mean()) / 2
    # Does it know near from far? Correlation of each map with its target, averaged.
    p = pred.reshape(len(pred), -1); t = true.reshape(len(true), -1)
    p = p - p.mean(1, keepdims=True); t = t - t.mean(1, keepdims=True)
    corr = ((p * t).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-9)).mean()
    return l1, edge, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    args = ap.parse_args()

    _, d_train = zip(*[load_dir(d) for d in args.train])
    d_train = np.concatenate(d_train)
    e_test, d_test = zip(*[load_dir(d) for d in args.test])
    e_test, d_test = np.concatenate(e_test), np.concatenate(d_test)

    model, _, _ = build_student()
    model.load_weights(args.weights)
    pred = model.predict(e_test, batch_size=64, verbose=0)

    mean_map = np.broadcast_to(d_train.mean(0), d_test.shape)
    const = np.full_like(d_test, d_train.mean())
    print(f"{'':10s} {'L1 (lower=better)':>18s} {'edges (lower)':>14s} {'per-map corr (higher)':>22s}")
    for name, p in [("model", pred), ("mean-map", mean_map), ("constant", const)]:
        l1, edge, corr = metrics(p, d_test)
        print(f"{name:10s} {l1:18.4f} {edge:14.4f} {corr:22.3f}")


if __name__ == "__main__":
    main()
