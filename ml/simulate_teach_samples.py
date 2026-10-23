"""How many TEACH samples are enough? Fixed step budget, varying number of camera samples.

  python simulate_teach_samples.py ../assets/models/echo_student.tflite data/bv/val.mdk 600 12000 0.1

Args: model, data, samples, steps, learning rate. Needs ai-edge-litert (the phone runtime).
Results are in EchoBrain (STEPS_PER_SAMPLE, DEFAULT_LR).
"""
import sys, numpy as np
from ai_edge_litert.interpreter import Interpreter
model, val, n_teach, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lr = np.float32(sys.argv[5]) if len(sys.argv) > 5 else None
REC = 16 * 128 * 2 + 32 * 32
raw = np.fromfile(val, '<f4'); raw = raw[: len(raw) // REC * REC].reshape(-1, REC)
echo = raw[:, :4096].reshape(-1, 16, 128, 2); depth = raw[:, 4096:].reshape(-1, 32, 32)
rng = np.random.default_rng(0); idx = rng.permutation(len(echo))
hold, teach = idx[:600], idx[600:600 + n_teach]
it = Interpreter(model_path=model)
infer, train = it.get_signature_runner('infer'), it.get_signature_runner('train')
def l1(ids): return float(np.mean([np.abs(infer(echo=echo[i:i+1])['depth'][0] - depth[i]).mean() for i in ids]))
before = l1(hold)
for s in range(steps):
    b = rng.choice(teach, 8)
    kw = {'lr': lr} if lr is not None else {}
    loss = train(echo=echo[b], depth=depth[b], **kw)['loss']
    assert np.isfinite(loss), f'NaN at step {s}'
print(f'lr={lr} samples={n_teach:5d} steps={steps}: held-out L1 {before:.4f} -> {l1(hold):.4f} ({(1 - l1(hold) / before) * 100:.0f}% better)', flush=True)
