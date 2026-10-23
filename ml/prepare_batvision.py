"""Convert the BatVision dataset into .mdk training files.

Code: https://github.com/AmandineBtto/Batvision-Dataset  (layout taken from its dataloaders)
Data: https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M
      mirror: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/HYLZNL
License: CC-BY-SA-4.0.

  # BV1: one root folder with train.csv / val.csv / test.csv
  python prepare_batvision.py --bv1 /data/BatvisionV1 --split train --out data/bv1_train.mdk
  # BV2: one sub-folder per location, each with its own train.csv / val.csv / test.csv
  python prepare_batvision.py --bv2 /data/BatvisionV2 --split train --out data/bv2_train.mdk

Layout (from the official dataloaders):
  BV1 csv columns: "depth path", "audio path left", "audio path right"
      audio = one .npy waveform per ear, depth = .npy in millimetres (nan/inf = invalid)
  BV2 csv columns: "depth path", "depth file name", "audio path", "audio file name"
      audio = binaural .wav, depth = .npy in millimetres
  Both: 44.1 kHz, ~0.5 s per recording, paths relative to the dataset root.

Caveats, so expectations stay honest:
  * BatVision's chirp tops out at 20 kHz; only the 17.5-20 kHz overlap with the phone
    chirp survives our matched filter, and the mics are a dummy head, not a phone.
    Treat it as pretraining for the trunk, then fine-tune on your own phone data.
  * Each recording is one ping, so it is tiled across the STACK axis.
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly

from dataset import write
from features import normalize_nearness, ping_features
from murdock_model import DEPTH_SIZE, SAMPLE_RATE, STACK

BATVISION_RATE = 44100


def _resample(x, sr):
    if sr == SAMPLE_RATE:
        return x.astype(np.float32)
    g = np.gcd(sr, SAMPLE_RATE)
    return resample_poly(x, SAMPLE_RATE // g, sr // g, axis=0).astype(np.float32)


def load_wav(path):
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    return _resample(np.asarray(x, np.float32), sr)


def load_npy_pair(left, right):
    x = np.stack([np.load(left).astype(np.float32).ravel(), np.load(right).astype(np.float32).ravel()], axis=1)
    return _resample(x, BATVISION_RATE)


def load_depth(path):
    return depth_to_nearness(np.load(path))


def depth_to_nearness(depth_mm):
    d = np.squeeze(depth_mm.astype(np.float32)) / 1000.0           # mm -> m
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    # Centre square crop, matching the camera teacher's view in the app.
    h, w = d.shape
    s = min(h, w)
    d = d[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    inv = np.where(d > 0.05, 1.0 / np.maximum(d, 0.05), 0.0)      # invalid -> far
    img = Image.fromarray(inv.astype(np.float32), mode="F").resize((DEPTH_SIZE, DEPTH_SIZE), Image.BOX)
    return normalize_nearness(np.asarray(img))


def bv1_rows(root, split):
    with open(os.path.join(root, f"{split}.csv")) as f:
        for row in csv.DictReader(f):
            yield (lambda r=row: load_npy_pair(os.path.join(root, r["audio path left"]),
                                               os.path.join(root, r["audio path right"])),
                   os.path.join(root, row["depth path"]))


def bv2_rows(root, split):
    for location in sorted(os.listdir(root)):
        path = os.path.join(root, location, f"{split}.csv")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                yield (lambda r=row: load_wav(os.path.join(root, r["audio path"], r["audio file name"])),
                       os.path.join(root, row["depth path"], row["depth file name"]))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bv1", help="BatVision V1 root")
    src.add_argument("--bv2", help="BatVision V2 root")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--search", type=int, default=2000,
                    help="samples (48 kHz) to search for the direct speaker->mic peak")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = bv1_rows(args.bv1, args.split) if args.bv1 else bv2_rows(args.bv2, args.split)
    echoes, depths, skipped = [], [], 0
    for load_audio, depth_path in rows:
        try:
            feats = ping_features(load_audio(), search=args.search)
            depth = load_depth(depth_path)
        except (OSError, ValueError) as e:
            skipped += 1
            print(f"skip {depth_path}: {e}")
            continue
        echoes.append(np.repeat(feats[None], STACK, axis=0))
        depths.append(depth)
        if len(echoes) % 500 == 0:
            print(f"{len(echoes)} samples...")
    write(args.out, np.stack(echoes), np.stack(depths))
    print(f"wrote {len(echoes)} samples -> {args.out} ({skipped} skipped)")


if __name__ == "__main__":
    main()
