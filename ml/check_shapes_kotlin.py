"""Checks that ShapeFit.kt (on the phone) and shapes.py (training on a PC) agree.

  KOTLINC=/path/to/kotlinc python check_shapes_kotlin.py [--maps data/bv/val.mdk]

Runs both on synthetic scenes, random noise and (optionally) real depth maps.
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np

from dataset import load_dir
from shapes import N, SHAPE_DIM, fit_batch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def synthetic(rng, n):
    maps = []
    y = np.arange(N)[:, None] * np.ones((1, N))
    for _ in range(n):
        h = rng.integers(6, 24)
        m = np.where(y >= h, rng.uniform(0, .3) + (y - h) / (N - h) * rng.uniform(.3, 1), rng.uniform(0, .4))
        m = m + (np.arange(N)[None] - N / 2) / N * rng.uniform(-.5, .5) * (y < h)
        for _ in range(rng.integers(0, 3)):
            y0, x0 = rng.integers(0, 28, 2)
            m[y0:y0 + rng.integers(3, 14), x0:x0 + rng.integers(3, 14)] = rng.uniform(.5, 1)
        maps.append(np.clip(m + rng.normal(0, .02, m.shape), 0, 1))
    return np.array(maps, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", help=".mdk/.mdp file or folder with real depth maps")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    sets = {"synthetic": synthetic(rng, 400), "noise": rng.uniform(0, 1, (100, N, N)).astype(np.float32)}
    if args.maps:
        depth = load_dir(args.maps)[2]
        sets["real"] = depth[:: max(1, len(depth) // args.limit)][: args.limit].astype(np.float32)

    kotlinc = os.environ.get("KOTLINC", "kotlinc")
    tmp = tempfile.mkdtemp()
    jar = os.path.join(tmp, "parity.jar")
    subprocess.run([kotlinc, "-nowarn", "-include-runtime", "-d", jar,
                    os.path.join(ROOT, "android/app/src/main/kotlin/com/murdock/murdock/ShapeFit.kt"),
                    os.path.join(ROOT, "tool/shape_parity/ShapeParity.kt")], check=True)
    worst = 0.0
    for name, maps in sets.items():
        src, dst = os.path.join(tmp, f"{name}.f32"), os.path.join(tmp, f"{name}.out")
        maps.astype("<f4").tofile(src)
        subprocess.run(["java", "-cp", jar, "ShapeParityKt", src, dst], check=True)
        kt = np.fromfile(dst, "<f4").reshape(-1, SHAPE_DIM)
        py = fit_batch(maps)
        diff = np.abs(kt - py).max(1)
        worst = max(worst, float(diff.max()))
        print(f"{name:10s} {len(maps):5d} maps  max |kotlin - numpy| = {diff.max():.2e}  "
              f"maps differing > 1e-4: {(diff > 1e-4).sum()}")
    print("OK" if worst < 1e-4 else "MISMATCH")
    raise SystemExit(0 if worst < 1e-4 else 1)


if __name__ == "__main__":
    main()
