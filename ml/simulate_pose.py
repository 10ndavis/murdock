"""Does the angle input help? Phone runtime (ai-edge-litert), two simulated grips.

  python simulate_pose.py OLD_MODEL.tflite ../assets/models/echo_student.tflite data/bv/val.mdk

1. The pose model must match the pre-pose model exactly at pose 0.
2. Two grips whose camera views differ (tilt 0 vs 60 deg, scene shifted 8 rows): learn with
   and without the angle input, compare held-out error. Measured: 0.215 without, 0.162 with.
"""
import sys, numpy as np
from ai_edge_litert.interpreter import Interpreter
old_m, new_m, val = sys.argv[1:4]
REC = 16 * 128 * 2 + 32 * 32
raw = np.fromfile(val, '<f4'); raw = raw[: len(raw) // REC * REC].reshape(-1, REC)
echo = raw[:, :4096].reshape(-1, 16, 128, 2); depth = raw[:, 4096:].reshape(-1, 32, 32)

# 1. identical at pose 0
old = Interpreter(model_path=old_m).get_signature_runner('infer')
new = Interpreter(model_path=new_m).get_signature_runner('infer')
x = echo[:20]
d = max(np.abs(old(echo=x[i:i+1])['depth'] - new(echo=x[i:i+1], pose=np.zeros((1, 2), np.float32))['depth']).max() for i in range(20))
print(f'pose 0: max output difference vs current model = {d:.2e}')

# 2. does it learn the angle? Two "grips": tilt 0 (scene as is) and tilt 60 deg (camera aimed
#    lower, so the scene sits 8 rows higher in the depth map). Same echoes, different labels.
rng = np.random.default_rng(0)
idx = rng.permutation(len(echo))[:2400]
grip = rng.integers(0, 2, len(idx))
labels = np.where(grip[:, None, None] == 1, np.roll(depth[idx], -8, axis=1), depth[idx])
poses = np.stack([np.where(grip == 1, 60 / 90, 0.0), np.zeros(len(idx))], 1).astype(np.float32)
teach, hold = np.arange(0, 1800), np.arange(1800, 2400)

def run(use_pose):
    it = Interpreter(model_path=new_m)
    inf, tr = it.get_signature_runner('infer'), it.get_signature_runner('train')
    p = poses if use_pose else np.zeros_like(poses)
    for s in range(6000):
        b = rng.choice(teach, 8)
        tr(echo=echo[idx[b]], pose=p[b], depth=labels[b], lr=np.float32(0.1))
    return np.mean([np.abs(inf(echo=echo[idx[i]][None], pose=p[i][None])['depth'][0] - labels[i]).mean() for i in hold])

print(f'held-out error after learning two grips: without angle input {run(False):.4f}, with angle input {run(True):.4f}')
