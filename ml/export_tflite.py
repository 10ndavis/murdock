"""Export the student as an on-device-trainable LiteRT (TFLite) model.

Signatures (used by EchoBrain.kt):
  infer(echo[1,T,R,C])                 -> depth[1,32,32]
  train(echo[B,T,R,C], depth[B,32,32], lr[]) -> loss[]   (updates the head only)
  get_weights(dummy[])                 -> w0..wN     (head weights, persist after training)
  set_weights(w0..wN)                  -> ok[]       (restore; also zeroes momentum)

Only builtin LiteRT ops are allowed (no flex runtime on the phone), which is
why the conv trunk is frozen on device; see murdock_model.py.

Persistence uses plain variable reads/assigns instead of tf.raw_ops.Save/Restore
so the model doesn't need the Select-TF-ops (flex) runtime for checkpointing.

Usage:
  python export_tflite.py                         # random init
  python export_tflite.py --weights student.keras # from train.py output
"""

import argparse

import tensorflow as tf

from murdock_model import INPUT_SHAPE, OUTPUT_SHAPE, build_student, depth_loss

TRAIN_BATCH = 8


MOMENTUM = 0.9
CLIP_NORM = 1.0   # max gradient norm per step: one bad batch can't blow the weights up


class TrainableStudent(tf.Module):
    """On-device training uses a hand-written momentum SGD instead of a Keras optimizer:
    the learning rate is a signature input (the app lowers it if training ever goes
    unstable) and gradients are clipped by global norm. Only builtin LiteRT ops."""

    def __init__(self, model, trunk, head):
        super().__init__()
        self.model, self.trunk, self.head = model, trunk, head
        self.momentum = [tf.Variable(tf.zeros(v.shape), trainable=False, name=f"mom{i}")
                         for i, v in enumerate(self.head.trainable_variables)]
        # Keras 3 variables aren't tf.Module-trackable; track the backing tf.Variables
        # explicitly so the SavedModel (and hence the .tflite) carries their values.
        self.tracked_vars = [v.value for v in model.variables]

    @tf.function(input_signature=[tf.TensorSpec([1, *INPUT_SHAPE], tf.float32, name="echo")])
    def infer(self, echo):
        return {"depth": self.model(echo, training=False)}

    @tf.function(input_signature=[
        tf.TensorSpec([TRAIN_BATCH, *INPUT_SHAPE], tf.float32, name="echo"),
        tf.TensorSpec([TRAIN_BATCH, *OUTPUT_SHAPE], tf.float32, name="depth"),
        tf.TensorSpec([], tf.float32, name="lr"),
    ])
    def train(self, echo, depth, lr):
        latent = self.trunk(echo, training=False)
        weights = self.head.trainable_variables
        with tf.GradientTape() as tape:
            pred = self.head(latent, training=True)
            loss = depth_loss(depth, pred)
        grads = tape.gradient(loss, weights)
        norm = tf.sqrt(tf.add_n([tf.reduce_sum(g * g) for g in grads]))
        scale = tf.minimum(1.0, CLIP_NORM / (norm + 1e-6))
        for v, m, g in zip(weights, self.momentum, grads):
            m.assign(MOMENTUM * m + scale * g)
            v.assign_sub(lr * m)
        return {"loss": loss}

    def _weights(self):
        return self.head.trainable_variables

    def make_io_signatures(self):
        weights = self._weights()

        # LiteRT's Java runSignature rejects an empty input map, hence the unused scalar.
        @tf.function(input_signature=[tf.TensorSpec([], tf.float32, name="dummy")])
        def get_weights(dummy):
            return {f"w{i}": tf.identity(v.value) for i, v in enumerate(weights)}

        @tf.function(input_signature=[
            tf.TensorSpec(v.shape, v.dtype, name=f"w{i}") for i, v in enumerate(weights)
        ])
        def set_weights(*values):
            for v, value in zip(weights, values):
                v.assign(value)
            for m in self.momentum:   # stale momentum would undo a rollback
                m.assign(tf.zeros_like(m))
            return {"ok": tf.constant(1.0)}

        return get_weights, set_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="Keras weights from train.py")
    ap.add_argument("--out", default="../assets/models/echo_student.tflite")
    args = ap.parse_args()

    model, trunk, head = build_student()
    if args.weights:
        model.load_weights(args.weights)
    module = TrainableStudent(model, trunk, head)
    get_weights, set_weights = module.make_io_signatures()

    signatures = {
        "infer": module.infer.get_concrete_function(),
        "train": module.train.get_concrete_function(),
        "get_weights": get_weights.get_concrete_function(),
        "set_weights": set_weights.get_concrete_function(),
    }
    converter = tf.lite.TFLiteConverter.from_saved_model(_save(module, signatures))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_enable_resource_variables = True
    tflite = converter.convert()
    with open(args.out, "wb") as f:
        f.write(tflite)
    print(f"wrote {args.out} ({len(tflite) / 1e6:.1f} MB), {len(head.trainable_variables)} trainable head tensors")


def _save(module, signatures, path="/tmp/murdock_saved_model"):
    tf.saved_model.save(module, path, signatures=signatures)
    return path


if __name__ == "__main__":
    main()
