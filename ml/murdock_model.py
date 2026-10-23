"""Echo -> depth student network shared by training, export and the app.

Keep these constants in sync with android/app/src/main/kotlin/com/murdock/murdock/SonarConfig.kt.

The network is split in two so it can be fine-tuned on the phone:
  trunk: conv encoder, echo stack -> 512-d latent. Trained on a PC (train.py),
         frozen on device (conv weight gradients need TF "flex" ops LiteRT lacks).
  head:  two dense layers, (latent + phone pose) -> 32x32 nearness map. Trainable on
         device using only builtin LiteRT ops (activations use hand-written gradients).

A second, small head predicts scene primitives (floor, walls, nearest obstacles; see
shapes.py) from the same latent + pose. It is separate from the depth head, so the depth
output is unchanged by it, and it is trained on device next to the depth head with labels
fitted to the camera teacher's depth.

Phone pose (tilt, roll) is a head input: the same room sounds and looks different at other
angles, so the AI learns per angle instead of needing one fixed grip. BatVision (a robot)
has no tilt; its samples use pose 0 and the on-device learning fills in the rest.
"""

import tensorflow as tf

SAMPLE_RATE = 48000
PING_PERIOD = 2048          # samples between chirps (~23 pings/s)
RANGE_SAMPLES = 1536        # samples kept after the direct path (~5.5 m one way)
RANGE_BINS = 128            # RANGE_SAMPLES / 12 -> ~4.3 cm per bin
STACK = 16                  # pings stacked into one model input (~0.7 s)
CHANNELS = 2                # stereo mics (mono phones duplicate the channel)
DEPTH_SIZE = 32             # output nearness map is DEPTH_SIZE x DEPTH_SIZE
LATENT = 512
POSE_DIM = 2                # (tilt / 90°, roll / 90°), see pose_features()
SHAPE_DIM = 24              # scene primitives, see shapes.py
SHAPE_HIDDEN = 256

INPUT_SHAPE = (STACK, RANGE_BINS, CHANNELS)
OUTPUT_SHAPE = (DEPTH_SIZE, DEPTH_SIZE)
POSE_SHAPE = (POSE_DIM,)


def pose_features(tilt_deg, roll_deg):
    """Phone angle -> head input. Mirrors EchoBrain.poseFeatures (Kotlin)."""
    import numpy as np
    return np.clip(np.array([tilt_deg / 90.0, roll_deg / 90.0], np.float32), -1.5, 1.5)


@tf.custom_gradient
def relu_b(x):
    """ReLU whose gradient uses builtin ops only (tf.ReluGrad is not a LiteRT builtin)."""
    mask = tf.cast(x > 0.0, x.dtype)

    def grad(dy):
        return dy * mask

    return x * mask, grad


@tf.custom_gradient
def sigmoid_b(x):
    y = 1.0 / (1.0 + tf.exp(-x))

    def grad(dy):
        return dy * y * (1.0 - y)

    return y, grad


def build_trunk() -> tf.keras.Model:
    L = tf.keras.layers
    inp = L.Input(INPUT_SHAPE, name="echo")
    x = L.Conv2D(32, (3, 7), strides=(1, 2), padding="same", activation="relu")(inp)   # 16x64
    x = L.Conv2D(64, (3, 5), strides=(2, 2), padding="same", activation="relu")(x)     # 8x32
    x = L.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)    # 4x16
    x = L.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)    # 2x8
    x = L.Flatten()(x)
    x = L.Dropout(0.3)(x)
    x = L.Dense(LATENT, activation="relu", name="latent")(x)
    x = L.Dropout(0.3)(x)
    return tf.keras.Model(inp, x, name="trunk")


def build_head(pose_dim: int = POSE_DIM) -> tf.keras.Model:
    L = tf.keras.layers
    latent = L.Input((LATENT,), name="latent")
    inputs = [latent]
    x = latent
    if pose_dim:
        pose = L.Input((pose_dim,), name="pose")
        inputs.append(pose)
        x = L.Concatenate()([latent, pose])
    x = L.Dense(1024, activation=relu_b)(x)
    x = L.Dense(DEPTH_SIZE * DEPTH_SIZE, activation=sigmoid_b)(x)
    out = L.Reshape(OUTPUT_SHAPE, name="depth")(x)
    return tf.keras.Model(inputs, out, name="head")


def build_shape_head(pose_dim: int = POSE_DIM) -> tf.keras.Model:
    """(latent + pose) -> SHAPE_DIM primitives in [0, 1] (shapes.py layout)."""
    L = tf.keras.layers
    latent = L.Input((LATENT,), name="latent")
    pose = L.Input((pose_dim,), name="pose")
    x = L.Concatenate()([latent, pose])
    x = L.Dense(SHAPE_HIDDEN, activation=relu_b)(x)
    out = L.Dense(SHAPE_DIM, activation=sigmoid_b, name="shapes")(x)
    return tf.keras.Model([latent, pose], out, name="shape_head")


def _presence_matrix():
    """(SHAPE_DIM, SHAPE_DIM) P and bias c so that y @ P + c gives each value's loss weight:
    1 for presences, the shape's presence for its other values (see shapes.presence_mask).
    A matmul keeps the loss to builtin LiteRT ops."""
    import numpy as np
    from shapes import BOXES, FLOOR, MAX_BOXES, WALLS
    P = np.zeros((SHAPE_DIM, SHAPE_DIM), np.float32)
    c = np.zeros(SHAPE_DIM, np.float32)
    for o, n in [(FLOOR, 3)] + [(WALLS + 3 * k, 3) for k in range(3)] + [(BOXES + 6 * k, 6) for k in range(MAX_BOXES)]:
        c[o] = 1.0
        P[o, o + 1:o + n] = 1.0
    return tf.constant(P), tf.constant(c)


_P, _C = _presence_matrix()


def shape_loss(y_true, y_pred):
    """Presence-weighted L1: where a wall or box isn't there, its position doesn't matter."""
    w = tf.matmul(y_true, _P) + _C
    return tf.reduce_sum(w * tf.abs(y_true - y_pred)) / (tf.reduce_sum(w) + 1e-6)


def build_student(pose_dim: int = POSE_DIM):
    """Returns (model, trunk, head). model inputs: [echo, pose] (or just echo if pose_dim=0)."""
    trunk, head = build_trunk(), build_head(pose_dim)
    echo = tf.keras.layers.Input(INPUT_SHAPE, name="echo")
    if pose_dim:
        pose = tf.keras.layers.Input((pose_dim,), name="pose")
        model = tf.keras.Model([echo, pose], head([trunk(echo), pose]), name="murdock_student")
    else:
        model = tf.keras.Model(echo, head(trunk(echo)), name="murdock_student")
    return model, trunk, head


def load_student_weights(model, path):
    """Loads weights; also accepts weights saved before the pose input existed (the new
    pose rows of the first head layer start at zero, so behaviour is unchanged until the
    model learns to use the angle)."""
    try:
        model.load_weights(path)
        return
    except Exception:  # noqa: BLE001 - shape mismatch: legacy, pose-free weights
        pass
    legacy, _, _ = build_student(pose_dim=0)
    legacy.load_weights(path)
    old = legacy.get_weights()
    new = model.get_weights()
    assert len(old) == len(new), "unexpected architecture"
    for i, (o, n) in enumerate(zip(old, new)):
        if o.shape == n.shape:
            new[i] = o
        else:  # first head kernel: (LATENT, 1024) -> (LATENT + POSE_DIM, 1024)
            assert o.shape[0] == LATENT and n.shape[0] == LATENT + POSE_DIM, (o.shape, n.shape)
            new[i] = n * 0
            new[i][:LATENT] = o
    model.set_weights(new)


def _diff_matrix(n):
    # (n, n-1) matrix D so that (x @ D)[..., j] = x[..., j+1] - x[..., j].
    # Matmul keeps the edge loss differentiable with builtin ops (slicing grads aren't).
    eye = tf.eye(n)
    return eye[:, 1:] - eye[:, :-1]


_DX = _diff_matrix(DEPTH_SIZE)


def depth_loss(y_true, y_pred):
    """L1 plus an image-gradient term so object edges (what we render) stay sharp.

    y_* are [B, DEPTH_SIZE, DEPTH_SIZE] nearness maps.
    """
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    dx = tf.abs(tf.matmul(y_true, _DX) - tf.matmul(y_pred, _DX))
    t_true = tf.transpose(y_true, [0, 2, 1])
    t_pred = tf.transpose(y_pred, [0, 2, 1])
    dy = tf.abs(tf.matmul(t_true, _DX) - tf.matmul(t_pred, _DX))
    return l1 + 0.5 * (tf.reduce_mean(dx) + tf.reduce_mean(dy))
