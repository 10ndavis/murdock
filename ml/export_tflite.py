"""Export the student as an on-device-trainable LiteRT (TFLite) model.

Signatures (used by EchoBrain.kt):
  infer(echo[1,T,R,C], pose[1,2])      -> depth[1,32,32], shapes[1,24]
  train(echo[B,T,R,C], pose[B,2], depth[B,32,32], shapes[B,24], lr[]) -> loss[], shape_loss[]
                                          (updates both heads; the trunk stays frozen)
  get_weights(dummy[])                 -> w0..w7     (w0..w3 depth head, w4..w7 shapes head)
  set_weights(w0..w7)                  -> ok[]       (restore; also zeroes momentum)

The depth head and its training step are exactly as before the shapes head existed
(gradients are clipped per head), so the depth picture is unchanged by adding it.

Only builtin LiteRT ops are allowed (no flex runtime on the phone), which is
why the conv trunk is frozen on device; see murdock_model.py.

Persistence uses plain variable reads/assigns instead of tf.raw_ops.Save/Restore
so the model doesn't need the Select-TF-ops (flex) runtime for checkpointing.

Usage:
  python export_tflite.py                         # random init
  python export_tflite.py --weights student.weights.h5 --shape-weights shapes.weights.h5
"""

import argparse

import tensorflow as tf

from murdock_model import (INPUT_SHAPE, OUTPUT_SHAPE, POSE_SHAPE, SHAPE_DIM, build_shape_head, build_student,
                           depth_loss, load_student_weights, shape_loss)

TRAIN_BATCH = 8


MOMENTUM = 0.9
CLIP_NORM = 1.0   # max gradient norm per step: one bad batch can't blow the weights up
# The shapes head learns slower than the depth head: at the full rate it memorised the taught
# rooms' layout and got worse elsewhere (simulated on BatVision, see the commit message).
SHAPE_LR_SCALE = 0.1


class TrainableStudent(tf.Module):
    """On-device training uses a hand-written momentum SGD instead of a Keras optimizer:
    the learning rate is a signature input (the app lowers it if training ever goes
    unstable) and gradients are clipped by global norm. Only builtin LiteRT ops."""

    def __init__(self, model, trunk, head, shape_head, shape_lr_scale=SHAPE_LR_SCALE):
        super().__init__()
        self.model, self.trunk, self.head, self.shape_head = model, trunk, head, shape_head
        self.shape_lr_scale = float(shape_lr_scale)
        self.momentum = [tf.Variable(tf.zeros(v.shape), trainable=False, name=f"mom{i}")
                         for i, v in enumerate(self._weights())]
        # Keras 3 variables aren't tf.Module-trackable; track the backing tf.Variables
        # explicitly so the SavedModel (and hence the .tflite) carries their values.
        self.tracked_vars = [v.value for v in model.variables + shape_head.variables]

    @tf.function(input_signature=[
        tf.TensorSpec([1, *INPUT_SHAPE], tf.float32, name="echo"),
        tf.TensorSpec([1, *POSE_SHAPE], tf.float32, name="pose"),
    ])
    def infer(self, echo, pose):
        latent = self.trunk(echo, training=False)
        return {"depth": self.head([latent, pose], training=False),
                "shapes": self.shape_head([latent, pose], training=False)}

    @tf.function(input_signature=[
        tf.TensorSpec([TRAIN_BATCH, *INPUT_SHAPE], tf.float32, name="echo"),
        tf.TensorSpec([TRAIN_BATCH, *POSE_SHAPE], tf.float32, name="pose"),
        tf.TensorSpec([TRAIN_BATCH, *OUTPUT_SHAPE], tf.float32, name="depth"),
        tf.TensorSpec([TRAIN_BATCH, SHAPE_DIM], tf.float32, name="shapes"),
        tf.TensorSpec([], tf.float32, name="lr"),
    ])
    def train(self, echo, pose, depth, shapes, lr):
        latent = self.trunk(echo, training=False)
        n_depth = len(self.head.trainable_variables)
        losses = []
        for net, label, loss_fn, mom, rate in [
            (self.head, depth, depth_loss, self.momentum[:n_depth], lr),
            (self.shape_head, shapes, shape_loss, self.momentum[n_depth:], lr * self.shape_lr_scale),
        ]:
            weights = net.trainable_variables
            with tf.GradientTape() as tape:
                loss = loss_fn(label, net([latent, pose], training=True))
            grads = tape.gradient(loss, weights)
            # Clipped per head: the depth head trains exactly as it did on its own.
            norm = tf.sqrt(tf.add_n([tf.reduce_sum(g * g) for g in grads]))
            scale = tf.minimum(1.0, CLIP_NORM / (norm + 1e-6))
            for v, m, g in zip(weights, mom, grads):
                m.assign(MOMENTUM * m + scale * g)
                v.assign_sub(rate * m)
            losses.append(loss)
        return {"loss": losses[0], "shape_loss": losses[1]}

    def _weights(self):
        # Depth head first (w0..w3, as before the shapes head existed), then the shapes head.
        return self.head.trainable_variables + self.shape_head.trainable_variables

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
    ap.add_argument("--shape-weights", help="shapes head weights from train_shapes.py")
    ap.add_argument("--shape-lr-scale", type=float, default=SHAPE_LR_SCALE)
    ap.add_argument("--out", default="../assets/models/echo_student.tflite")
    args = ap.parse_args()

    model, trunk, head = build_student()
    if args.weights:
        load_student_weights(model, args.weights)
    shape_head = build_shape_head()
    if args.shape_weights:
        shape_head.load_weights(args.shape_weights)
    module = TrainableStudent(model, trunk, head, shape_head, args.shape_lr_scale)
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
    print(f"wrote {args.out} ({len(tflite) / 1e6:.1f} MB), {len(module._weights())} trainable head tensors")


def _save(module, signatures, path="/tmp/murdock_saved_model"):
    tf.saved_model.save(module, path, signatures=signatures)
    return path


if __name__ == "__main__":
    main()
