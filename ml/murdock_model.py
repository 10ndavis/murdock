"""Echo -> depth student network shared by training, export and the app.

Keep these constants in sync with android/app/src/main/kotlin/com/murdock/murdock/SonarConfig.kt.

The network is split in two so it can be fine-tuned on the phone:
  trunk: conv encoder, echo stack -> 512-d latent. Trained on a PC (train.py),
         frozen on device (conv weight gradients need TF "flex" ops LiteRT lacks).
  head:  two dense layers, latent -> 32x32 nearness map. Trainable on device using
         only builtin LiteRT ops (activations use hand-written gradients below).
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

INPUT_SHAPE = (STACK, RANGE_BINS, CHANNELS)
OUTPUT_SHAPE = (DEPTH_SIZE, DEPTH_SIZE)


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
    x = L.Dense(LATENT, activation="relu", name="latent")(x)
    return tf.keras.Model(inp, x, name="trunk")


def build_head() -> tf.keras.Model:
    L = tf.keras.layers
    inp = L.Input((LATENT,), name="latent")
    x = L.Dense(1024, activation=relu_b)(inp)
    x = L.Dense(DEPTH_SIZE * DEPTH_SIZE, activation=sigmoid_b)(x)
    out = L.Reshape(OUTPUT_SHAPE, name="depth")(x)
    return tf.keras.Model(inp, out, name="head")


def build_student():
    trunk, head = build_trunk(), build_head()
    inp = tf.keras.layers.Input(INPUT_SHAPE, name="echo")
    model = tf.keras.Model(inp, head(trunk(inp)), name="murdock_student")
    return model, trunk, head


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
