"""Diagnostic: same network, but fed BatVision's FULL frequency range.

  python diagnose_fullband.py data/bv        # after stream_batvision.py fetch
  python train.py --data data/bv/fullband/train.mdk --val data/bv/fullband/val.mdk
  python evaluate.py --weights student.weights.h5 --train data/bv/fullband/train.mdk --test data/bv/fullband/test.mdk

Answers "is a weak phone-band result the band's fault or the pipeline's?" Not used by the app.

Features: 16 frequency bands (0.5-20 kHz, log-spaced) x 128 range bins x 2 mics, packed into
the model's [STACK=16, 128, 2] input shape. Envelope per band, aligned at the dataset's
broadband emission point, pooled like the phone features, scaled by the strongest return.
"""
import os, sys, csv, numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from dataset import write
from murdock_model import STACK, RANGE_BINS, RANGE_SAMPLES
from stream_batvision import _load_all, _session

S = sys.argv[1]
edges = np.geomspace(500, 20000, STACK + 1)
sos = [butter(4, [lo, hi], btype='band', fs=48000, output='sos') for lo, hi in zip(edges[:-1], edges[1:])]
audio = _load_all(S, 'audio')
depths = _load_all(S, 'depth')

def band_envs(x):
    return np.stack([np.abs(hilbert(sosfiltfilt(s, x, axis=0), axis=0)) for s in sos])  # [B, n, 2]

# Emission point per dataset: median over sessions of the broadband envelope's first rise.
rise = {}
for k, a in list(audio.items()):
    rise.setdefault(_session(k), []).append(np.abs(a[:3000, 0].astype(np.float32)))
zero = {}
for v in ('BatvisionV1', 'BatvisionV2'):
    rs = []
    for s, xs in rise.items():
        if not s.startswith(v): continue
        m = np.median([np.convolve(x, np.ones(24) / 24, 'same') for x in xs], 0)
        rs.append(int(np.argmax(m > 4 * np.median(m[:200]))))
    zero[v] = int(np.median(rs)); print(v, 'broadband emission at', zero[v], rs)

pool = RANGE_SAMPLES // RANGE_BINS
feats = {}
for i, (k, a) in enumerate(audio.items()):
    e = band_envs(a.astype(np.float32))
    o = zero[k.split('/')[0]]
    seg = e[:, o:o + RANGE_SAMPLES]
    seg = np.pad(seg, ((0, 0), (0, RANGE_SAMPLES - seg.shape[1]), (0, 0)))
    p = seg.reshape(STACK, RANGE_BINS, pool, 2).max(2)
    p = p / p.max(axis=(1, 2), keepdims=True).clip(1e-9)      # per-band scale
    feats[k] = np.clip((np.log10(p + 1e-4) + 4) / 4, 0, 1).astype(np.float32)
    if i % 2000 == 0: print(i, 'recordings')

for split in ('train', 'val', 'test'):
    E, D = [], []
    for dirpath, _, files in os.walk(os.path.join(S, 'csv')):
        if f'{split}.csv' not in files: continue
        root = os.path.relpath(dirpath, os.path.join(S, 'csv')).split(os.sep)[0]
        for r in csv.DictReader(open(os.path.join(dirpath, f'{split}.csv'))):
            if 'audio path left' in r:
                a = os.path.normpath(os.path.join(root, r['audio path left'])); d = os.path.normpath(os.path.join(root, r['depth path']))
            else:
                a = os.path.normpath(os.path.join(root, r['audio path'], r['audio file name'])); d = os.path.normpath(os.path.join(root, r['depth path'], r['depth file name']))
            if a in feats and d in depths: E.append(feats[a]); D.append(depths[d])
    write(os.path.join(S, 'fullband', f'{split}.mdk'), np.stack(E), np.stack(D)); print(split, len(E))
