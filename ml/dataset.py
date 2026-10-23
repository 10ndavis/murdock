"""Reads/writes the .mdk sample files the app records in "Train" mode.

A .mdk file is a flat sequence of little-endian float32 records:
  echo  [STACK, RANGE_BINS, CHANNELS]   (row-major, same layout as the model input)
  depth [DEPTH_SIZE, DEPTH_SIZE]        (camera-teacher nearness, 1 = near)
"""

import glob
import os

import numpy as np

from murdock_model import INPUT_SHAPE, OUTPUT_SHAPE

ECHO_FLOATS = int(np.prod(INPUT_SHAPE))
DEPTH_FLOATS = int(np.prod(OUTPUT_SHAPE))
RECORD_FLOATS = ECHO_FLOATS + DEPTH_FLOATS


def load_dir(path: str):
    """Loads one .mdk file, or every .mdk file under a folder."""
    files = [path] if path.endswith(".mdk") else sorted(glob.glob(os.path.join(path, "**", "*.mdk"), recursive=True))
    if not files:
        raise SystemExit(f"no .mdk files under {path}")
    chunks = []
    for f in files:
        raw = np.fromfile(f, dtype="<f4")
        n = len(raw) // RECORD_FLOATS
        chunks.append(raw[: n * RECORD_FLOATS].reshape(n, RECORD_FLOATS))
    data = np.concatenate(chunks)
    echo = data[:, :ECHO_FLOATS].reshape(-1, *INPUT_SHAPE)
    depth = data[:, ECHO_FLOATS:].reshape(-1, *OUTPUT_SHAPE)
    print(f"loaded {len(data)} samples from {len(files)} files")
    return echo, depth


def write(path: str, echo: np.ndarray, depth: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    echo = echo.reshape(len(echo), ECHO_FLOATS)
    depth = depth.reshape(len(depth), DEPTH_FLOATS)
    np.concatenate([echo, depth], axis=1).astype("<f4").tofile(path)
