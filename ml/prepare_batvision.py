"""Convert BatVision (or any chirp-echo + depth dataset) into .mdk training files.

BatVision: https://amandinebtto.github.io/Batvision-Dataset/ (request access there).
Its layout differs between BV1/BV2 releases, so this script takes a CSV you build
with two columns, `audio,depth`: a binaural .wav per sample and its depth map
(.npy in metres, or 16-bit .png in millimetres).

  python prepare_batvision.py --pairs bv1_train.csv --out data/batvision/bv1.mdk

Caveats, so expectations stay honest:
  * BatVision's chirp is 20 Hz-20 kHz; only the 17.5-20 kHz overlap with the phone
    chirp survives our matched filter, and the mics are a dummy head, not a phone.
    Treat it as pretraining for the trunk, then fine-tune on your own phone data.
  * Each recording is one ping, so it is tiled across the STACK axis.
"""

import argparse
import csv

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly

from dataset import write
from features import normalize_nearness, ping_features
from murdock_model import DEPTH_SIZE, SAMPLE_RATE, STACK


def load_audio(path):
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g, axis=0)
    return x


def load_depth(path):
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d = np.asarray(Image.open(path), dtype=np.float32) / 1000.0
    d = np.squeeze(d)
    inv = np.where(d > 0.05, 1.0 / np.maximum(d, 0.05), 0.0)   # invalid -> far
    img = Image.fromarray(inv.astype(np.float32), mode="F").resize((DEPTH_SIZE, DEPTH_SIZE), Image.BOX)
    return normalize_nearness(np.asarray(img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    echoes, depths = [], []
    with open(args.pairs) as f:
        for row in csv.DictReader(f):
            feats = ping_features(load_audio(row["audio"]))
            echoes.append(np.repeat(feats[None], STACK, axis=0))
            depths.append(load_depth(row["depth"]))
    write(args.out, np.stack(echoes), np.stack(depths))
    print(f"wrote {len(echoes)} samples -> {args.out}")


if __name__ == "__main__":
    main()
