"""Checks the two-head model on the phone's runtime and simulates learning shapes on device.

  python simulate_shapes.py <previous.tflite> <new.tflite> data/bv/test.mdk 600 12000

1-3: the depth output, the depth head's training step and restoring depth-head-only saved
learning are unchanged from the previous (pre-shapes) model. 4: a simulated TEACH session
(N samples, M steps, like simulate_teach.py) in the same rooms and in new rooms.
Needs ai-edge-litert (the runtime the app uses) and NumPy, not TensorFlow.
"""
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from shapes import fit_batch, presence_mask

old_m, new_m, test = sys.argv[1:4]
REC = 16 * 128 * 2 + 32 * 32
raw = np.fromfile(test, '<f4'); raw = raw[: len(raw) // REC * REC].reshape(-1, REC)
echo = raw[:, :4096].reshape(-1, 16, 128, 2); depth = raw[:, 4096:].reshape(-1, 32, 32)
rng = np.random.default_rng(0)

old_it, new_it = Interpreter(model_path=old_m), Interpreter(model_path=new_m)
print("signatures:", {k: (sorted(v['inputs']), sorted(v['outputs'])) for k, v in new_it.get_signature_list().items()
                      if k in ("infer", "train")})
old_inf, new_inf = old_it.get_signature_runner('infer'), new_it.get_signature_runner('infer')

# 1. Depth output identical to the current model (random poses too).
d = 0.0
for i in range(30):
    pose = rng.uniform(-1, 1, (1, 2)).astype(np.float32)
    a = old_inf(echo=echo[i:i + 1], pose=pose)['depth']
    o = new_inf(echo=echo[i:i + 1], pose=pose)
    d = max(d, float(np.abs(a - o['depth']).max()))
    assert o['shapes'].shape == (1, 24) and np.all((o['shapes'] >= 0) & (o['shapes'] <= 1))
print(f"1. depth vs current model: max difference {d:.2e}")

# 2. One training step on the same batch: the depth head ends up identical too.
b = rng.choice(len(echo), 8)
pose = np.zeros((8, 2), np.float32)
o_tr = old_it.get_signature_runner('train')(echo=echo[b], pose=pose, depth=depth[b], lr=np.float32(0.1))
n_tr = new_it.get_signature_runner('train')(echo=echo[b], pose=pose, depth=depth[b], shapes=fit_batch(depth[b]), lr=np.float32(0.1))
wo = old_it.get_signature_runner('get_weights')(dummy=np.float32(0))
wn = new_it.get_signature_runner('get_weights')(dummy=np.float32(0))
dw = max(float(np.abs(wo[f'w{i}'] - wn[f'w{i}']).max()) for i in range(4))
print(f"2. after a train step: depth loss {float(o_tr['loss']):.5f} vs {float(n_tr['loss']):.5f}, shape loss "
      f"{float(n_tr['shape_loss']):.4f}; depth head weights max difference {dw:.2e}; {len(wn)} weight tensors")

# 3. Old saved learning (depth head only) restored into the new model -> same depth output.
fresh = Interpreter(model_path=new_m)
mine = fresh.get_signature_runner('get_weights')(dummy=np.float32(0))
fresh.get_signature_runner('set_weights')(**{**mine, **{f'w{i}': wo[f'w{i}'] for i in range(4)}})
x = echo[100:101]
dd = float(np.abs(fresh.get_signature_runner('infer')(echo=x, pose=pose[:1])['depth'] - old_inf(echo=x, pose=pose[:1])['depth']).max())
print(f"3. legacy depth-head weights restored: depth difference vs old model {dd:.2e}")

# 4. On-device learning simulation: teach 600 samples from one part of the test split (20 steps
#    each, as on the phone), evaluate on a later, held-out stretch.
shapes_all = fit_batch(depth)
n_teach, steps = int(sys.argv[4]), int(sys.argv[5])
perm = np.random.default_rng(0).permutation(len(echo))
splits = {
    "same rooms (random split, as simulate_teach.py)": (perm[:n_teach], perm[2000:2600]),
    "new rooms (teach early frames, test a later stretch)": (np.arange(n_teach) * (2000 // n_teach), np.arange(2400, len(echo), 4)),
}
for name, (teach, hold) in splits.items():
    sim = Interpreter(model_path=new_m)
    inf, tr = sim.get_signature_runner('infer'), sim.get_signature_runner('train')

    def err():
        outs = [inf(echo=echo[i][None], pose=np.zeros((1, 2), np.float32)) for i in hold]
        pred = np.concatenate([o['shapes'] for o in outs])
        t = shapes_all[hold]
        w = presence_mask(t)
        dl1 = float(np.mean([np.abs(o['depth'][0] - depth[i]).mean() for o, i in zip(outs, hold)]))
        return float((w * np.abs(pred - t)).sum() / w.sum()), dl1

    w = presence_mask(shapes_all[hold])
    mean_err = float((w * np.abs(shapes_all[teach].mean(0) - shapes_all[hold])).sum() / w.sum())
    before = err()
    for s in range(steps):
        bb = rng.choice(teach, 8)
        tr(echo=echo[bb], pose=np.zeros((8, 2), np.float32), depth=depth[bb], shapes=shapes_all[bb], lr=np.float32(0.1))
    after = err()
    print(f"4. {name}, {n_teach} samples, {steps} steps:\n"
          f"   shapes {before[0]:.4f} -> {after[0]:.4f} ({(1 - after[0] / before[0]) * 100:+.0f}%), "
          f"taught samples' average shapes {mean_err:.4f}; depth L1 {before[1]:.4f} -> {after[1]:.4f} "
          f"({(1 - after[1] / before[1]) * 100:+.0f}%)")
