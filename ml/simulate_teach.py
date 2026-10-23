"""Simulate a TEACH session with the phone runtime: on-device head training on some frames,
then error on held-out frames of the same rooms.

  python simulate_teach.py ../assets/models/echo_student.tflite data/bv/val.mdk 12000

Needs ai-edge-litert (the same runtime the app uses), not TensorFlow.
"""
import sys, numpy as np
from ai_edge_litert.interpreter import Interpreter
model, val = sys.argv[1], sys.argv[2]
REC = 16 * 128 * 2 + 32 * 32
raw = np.fromfile(val, '<f4'); raw = raw[: len(raw) // REC * REC].reshape(-1, REC)
echo = raw[:, :4096].reshape(-1, 16, 128, 2); depth = raw[:, 4096:].reshape(-1, 32, 32)
rng = np.random.default_rng(0); idx = rng.permutation(len(echo))
teach, hold = idx[:2000], idx[2000:2600]       # "rooms you walked around" vs held-out frames
it = Interpreter(model_path=model)
infer, train = it.get_signature_runner('infer'), it.get_signature_runner('train')
def l1(ids): return float(np.mean([np.abs(infer(echo=echo[i:i+1])['depth'][0] - depth[i]).mean() for i in ids]))
out = infer(echo=echo[:1])['depth']
print('output finite:', np.isfinite(out).all(), 'range', round(float(out.min()), 3), round(float(out.max()), 3))
print('held-out L1 before on-device training:', round(l1(hold), 4))
for step in range(int(sys.argv[3]) if len(sys.argv) > 3 else 300):
    b = rng.choice(teach, 8)
    loss = train(echo=echo[b], depth=depth[b], lr=np.float32(0.1))['loss']
print('train loss after N steps:', round(float(loss), 4))
print('held-out L1 after on-device training: ', round(l1(hold), 4))
