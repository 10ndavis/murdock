"""Stream BatVision straight from the data.gouv mirror into .mdk files, never storing it.

The dataset is ~168 GB (five independent .tar.gz parts); what training needs is ~1 KB of
echo features plus a 32x32 depth map per sample. So each part is downloaded, gunzipped and
untarred as a stream, every recording/depth frame is reduced on the fly, and the official
train/val/test CSVs are used afterwards to pair them.

  python stream_batvision.py fetch --out data/bv --parts 0 1 2 3 4   # resumable, per part
  python stream_batvision.py join  --out data/bv                     # -> data/bv/{train,val,test}.mdk
"""

import argparse
import csv
import io
import os
import sys
import tarfile
import time
import urllib.request

import numpy as np
from scipy.io import wavfile

from dataset import write
from murdock_model import STACK
from prepare_batvision import BATVISION_RATE, _resample, depth_to_nearness
from features import ping_features

API = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/"
PART_IDS = [627779, 627781, 627782, 627790, 627791]   # Batvision_Dataset-{0..4}.tar.gz
SEARCH = 2000


class RangeStream(io.RawIOBase):
    """Read-only HTTP stream that transparently resumes (Range) after network errors."""

    def __init__(self, url):
        self.url, self.pos, self.resp = url, 0, None

    def readable(self):
        return True

    def _open(self):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-"})
        self.resp = urllib.request.urlopen(req, timeout=60)

    def readinto(self, b):
        for attempt in range(20):
            try:
                if self.resp is None:
                    self._open()
                n = self.resp.readinto(b)
                self.pos += n
                return n
            except Exception as e:  # noqa: BLE001 - any network hiccup: reopen at self.pos
                print(f"  network error at {self.pos / 1e9:.2f} GB ({e}); retry {attempt + 1}", flush=True)
                self.resp = None
                time.sleep(min(60, 2 ** attempt))
        raise IOError("too many network errors")


def _audio_wav(data):
    sr, x = wavfile.read(io.BytesIO(data))
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    return _resample(np.asarray(x, np.float32), sr)


def fetch_part(part, out):
    done_flag = os.path.join(out, f"part{part}.done")
    if os.path.exists(done_flag):
        print(f"part {part}: already done")
        return
    feats, depths, pending = {}, {}, {}
    stream = io.BufferedReader(RangeStream(API + str(PART_IDS[part])), buffer_size=1 << 22)
    start = time.time()
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for i, m in enumerate(tar):
            if not m.isfile():
                continue
            name = m.name
            try:
                if name.endswith(".csv"):
                    dest = os.path.join(out, "csv", name)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(tar.extractfile(m).read())
                elif "/audio" in name and name.endswith((".npy", ".wav")):
                    data = tar.extractfile(m).read()
                    if name.endswith(".wav"):                      # BV2: binaural wav
                        feats[name] = ping_features(_audio_wav(data), search=SEARCH)
                    else:                                         # BV1: one .npy per ear
                        key, side = name.rsplit("_", 1)
                        half = pending.setdefault(key, {})
                        half[side] = np.load(io.BytesIO(data)).astype(np.float32).ravel()
                        if len(half) == 2:
                            x = np.stack([half["left.npy"], half["right.npy"]], axis=1)
                            feats[key + "_left.npy"] = ping_features(_resample(x, BATVISION_RATE), search=SEARCH)
                            del pending[key]
                elif name.endswith(".npy"):                       # every other .npy is a depth map
                    d = np.load(io.BytesIO(tar.extractfile(m).read()))
                    depths[name] = depth_to_nearness(d)
            except Exception as e:  # noqa: BLE001 - one corrupt member shouldn't kill a 37 GB pass
                print(f"  skip {name}: {e}")
            if i % 20000 == 0:
                gb = stream.raw.pos / 1e9
                print(f"part {part}: {gb:.1f} GB, {gb * 1e3 / max(time.time() - start, 1):.0f} MB/s, "
                      f"{len(feats)} echoes, {len(depths)} depth maps", flush=True)
    np.savez(os.path.join(out, f"part{part}_echo.npz"), **{k.replace("/", "|"): v for k, v in feats.items()})
    np.savez(os.path.join(out, f"part{part}_depth.npz"), **{k.replace("/", "|"): v for k, v in depths.items()})
    open(done_flag, "w").close()
    print(f"part {part}: done, {len(feats)} echoes, {len(depths)} depth maps, {len(pending)} unpaired ears")


def _load_all(out, kind):
    table = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(f"_{kind}.npz"):
            with np.load(os.path.join(out, f)) as z:
                table.update({k.replace("|", "/"): z[k] for k in z.files})
    return table


def join(out):
    echoes, depths = _load_all(out, "echo"), _load_all(out, "depth")
    print(f"{len(echoes)} echoes, {len(depths)} depth maps")
    csv_root = os.path.join(out, "csv")
    for split in ("train", "val", "test"):
        e_out, d_out, missing = [], [], 0
        for dirpath, _, files in os.walk(csv_root):
            if f"{split}.csv" not in files:
                continue
            rel = os.path.relpath(dirpath, csv_root)
            version = rel.split(os.sep)[0]                 # BatvisionV1 / BatvisionV2
            root = version
            with open(os.path.join(dirpath, f"{split}.csv")) as f:
                for row in csv.DictReader(f):
                    if "audio path left" in row:           # BV1
                        a = os.path.normpath(os.path.join(root, row["audio path left"]))
                        d = os.path.normpath(os.path.join(root, row["depth path"]))
                    else:                                  # BV2
                        a = os.path.normpath(os.path.join(root, row["audio path"], row["audio file name"]))
                        d = os.path.normpath(os.path.join(root, row["depth path"], row["depth file name"]))
                    if a in echoes and d in depths:
                        e_out.append(np.repeat(echoes[a][None], STACK, axis=0))
                        d_out.append(depths[d])
                    else:
                        missing += 1
        if e_out:
            write(os.path.join(out, f"{split}.mdk"), np.stack(e_out), np.stack(d_out))
        print(f"{split}: {len(e_out)} samples ({missing} csv rows without both files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "join"])
    ap.add_argument("--out", default="data/bv")
    ap.add_argument("--parts", type=int, nargs="+", default=list(range(len(PART_IDS))))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "fetch":
        for p in args.parts:
            fetch_part(p, args.out)
    else:
        join(args.out)


if __name__ == "__main__":
    sys.exit(main())
