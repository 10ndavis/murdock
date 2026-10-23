"""Reads/writes training samples: flat little-endian float32 records.

  .mdk  echo [STACK, RANGE_BINS, CHANNELS] + depth [DEPTH_SIZE, DEPTH_SIZE]
        (BatVision and early phone builds; no phone angle -> pose 0)
  .mdp  echo + pose [POSE_DIM] + depth   (phone samples since the pose input; EchoBrain.kt)

echo uses the model-input layout (row-major); depth is camera-teacher nearness (1 = near);
pose is murdock_model.pose_features(tilt, roll).
"""

import glob
import os

import numpy as np

from murdock_model import INPUT_SHAPE, OUTPUT_SHAPE, POSE_DIM

ECHO_FLOATS = int(np.prod(INPUT_SHAPE))
DEPTH_FLOATS = int(np.prod(OUTPUT_SHAPE))
RECORD_FLOATS = ECHO_FLOATS + DEPTH_FLOATS            # .mdk
POSE_RECORD_FLOATS = ECHO_FLOATS + POSE_DIM + DEPTH_FLOATS  # .mdp


def _records(path, width):
    raw = np.fromfile(path, dtype="<f4")
    n = len(raw) // width   # a partly written last record is dropped
    return raw[: n * width].reshape(n, width)


def load_dir(path: str):
    """Loads one .mdk/.mdp file, or all of them under a folder. Returns (echo, pose, depth)."""
    if path.endswith((".mdk", ".mdp")):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "**", "*.md[kp]"), recursive=True))
    if not files:
        raise SystemExit(f"no .mdk/.mdp files under {path}")
    echoes, poses, depths = [], [], []
    for f in files:
        if f.endswith(".mdp"):
            r = _records(f, POSE_RECORD_FLOATS)
            echoes.append(r[:, :ECHO_FLOATS])
            poses.append(r[:, ECHO_FLOATS:ECHO_FLOATS + POSE_DIM])
            depths.append(r[:, ECHO_FLOATS + POSE_DIM:])
        else:
            r = _records(f, RECORD_FLOATS)
            echoes.append(r[:, :ECHO_FLOATS])
            poses.append(np.zeros((len(r), POSE_DIM), np.float32))
            depths.append(r[:, ECHO_FLOATS:])
    echo = np.concatenate(echoes).reshape(-1, *INPUT_SHAPE)
    pose = np.concatenate(poses)
    depth = np.concatenate(depths).reshape(-1, *OUTPUT_SHAPE)
    print(f"loaded {len(echo)} samples from {len(files)} files")
    return echo, pose, depth


def write(path: str, echo: np.ndarray, depth: np.ndarray, pose: np.ndarray | None = None):
    """Writes .mdk (no pose) or, when pose is given, .mdp records."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    parts = [echo.reshape(len(echo), ECHO_FLOATS)]
    if pose is not None:
        assert path.endswith(".mdp"), "pose samples go in .mdp files"
        parts.append(pose.reshape(len(pose), POSE_DIM))
    parts.append(depth.reshape(len(depth), DEPTH_FLOATS))
    np.concatenate(parts, axis=1).astype("<f4").tofile(path)
