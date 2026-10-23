"""Staring at one thing: does the echo AI start seeing it everywhere?

Simulates a phone session on BatVision: some varied teaching, then a long stretch of
near-identical camera frames of one view (pointing at a box), under the on-device sample
policy. "current" adds every frame and gives each STEPS_PER_SAMPLE steps (as before the
fix); "gated" applies EchoBrain's novelty gate (a view already in the replay buffer
NEAR_LIMIT times is skipped and earns no steps).

  python simulate_fixation.py ../assets/models/echo_student.tflite data/bv/val.mdk

Reports held-out error and how close predictions on other scenes are to the fixated view.
"""
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

STEPS_PER_SAMPLE = 20
NEAR_DIST, NEAR_LIMIT = 0.035, 3   # must match EchoBrain.kt

model, val = sys.argv[1], sys.argv[2]
# "weak": echoes carry little scene information (heavy noise), as on a phone early on;
# the AI then mostly predicts the average of what it was taught.
WEAK = len(sys.argv) > 3 and sys.argv[3] == 'weak'
REC = 16 * 128 * 2 + 32 * 32
raw = np.fromfile(val, '<f4')
raw = raw[: len(raw) // REC * REC].reshape(-1, REC)
echo = raw[:, :4096].reshape(-1, 16, 128, 2)
depth = raw[:, 4096:].reshape(-1, 32, 32)
POSE = np.zeros((8, 2), np.float32)


def signature(d):
    """8x8 average-pooled depth: EchoBrain.signature."""
    return d.reshape(8, 4, 8, 4).mean(axis=(1, 3)).ravel()


def run(policy, n_varied=300, n_fixed=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(echo))
    hold, varied, box = idx[:500], idx[500:500 + n_varied], idx[500 + n_varied]
    it = Interpreter(model_path=model)
    infer, train = it.get_signature_runner('infer'), it.get_signature_runner('train')

    def predict(ids):
        return np.stack([infer(echo=noisy[i:i + 1], pose=np.zeros((1, 2), np.float32))['depth'][0] for i in ids])

    def report(tag):
        p = predict(hold)
        l1 = np.abs(p - depth[hold]).mean()
        to_box = np.abs(p - depth[box]).mean()
        print(f'  {"weak " if WEAK else ""}{policy:8s} {tag:22s} held-out L1 {l1:.4f}  distance to the box view {to_box:.4f}', flush=True)

    if WEAK:
        noisy = echo + rng.normal(0, 1.5, echo.shape).astype(np.float32) * echo.std()
    else:
        noisy = echo
    replay_e, replay_d, sigs = [], [], []
    budget = 0

    def add(e, d):
        nonlocal budget
        s = signature(d)
        if policy == 'gated' and sigs:
            near = int((np.abs(np.asarray(sigs) - s).mean(axis=1) < NEAR_DIST).sum())
            if near >= NEAR_LIMIT:
                return
        replay_e.append(e); replay_d.append(d); sigs.append(s)
        budget += STEPS_PER_SAMPLE

    def spend():
        nonlocal budget
        n = len(replay_e)
        for _ in range(budget):
            b = rng.integers(0, n, 8)
            train(echo=np.stack([replay_e[i] for i in b]), pose=POSE,
                  depth=np.stack([replay_d[i] for i in b]), shapes=np.zeros((8, 24), np.float32),
                  lr=np.float32(0.1))
        budget = 0

    report('start')
    for i in varied:
        add(noisy[i], depth[i])
    before = len(replay_e)
    spend()
    report(f'after {n_varied} varied')
    # Fixation: the same view, with camera-depth jitter and echo noise from hand shake.
    for _ in range(n_fixed):
        e = noisy[box] + rng.normal(0, 0.05, echo[box].shape).astype(np.float32) * echo[box].std()
        d = np.clip(depth[box] + rng.normal(0, 0.015, depth[box].shape), 0, 1).astype(np.float32)
        add(e, d)
    kept = len(replay_e) - before
    spend()
    report(f'after staring ({kept} kept)')


for policy in ['current', 'gated']:
    run(policy)
