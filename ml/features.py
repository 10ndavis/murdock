"""NumPy reference of the on-phone sonar DSP (SonarEngine.kt). Keep the two in sync.

Pipeline per ping:
  recording --(complex matched filter against the chirp)--> echo envelope
  -> align to the direct speaker->mic peak -> keep RANGE_SAMPLES after it
  -> max-pool into RANGE_BINS -> normalise by the direct-path peak -> log scale [0, 1]
"""

import numpy as np

from murdock_model import CHANNELS, RANGE_BINS, RANGE_SAMPLES, SAMPLE_RATE

CHIRP_F0 = 17500.0
CHIRP_F1 = 21500.0
CHIRP_LEN = 240            # 5 ms
POOL = RANGE_SAMPLES // RANGE_BINS
LOG_FLOOR = 1e-4           # -80 dB relative to the direct path maps to 0


def chirp_iq():
    n = np.arange(CHIRP_LEN)
    t = n / SAMPLE_RATE
    dur = CHIRP_LEN / SAMPLE_RATE
    phase = 2 * np.pi * (CHIRP_F0 * t + (CHIRP_F1 - CHIRP_F0) * t * t / (2 * dur))
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / (CHIRP_LEN - 1))
    return (w * np.cos(phase)).astype(np.float32), (w * np.sin(phase)).astype(np.float32)


def envelope(x: np.ndarray) -> np.ndarray:
    """Matched-filter envelope of a mono signal (valid correlation lags only)."""
    ci, cq = chirp_iq()
    i = np.correlate(x, ci, mode="valid")
    q = np.correlate(x, cq, mode="valid")
    return np.sqrt(i * i + q * q)


def profile(env: np.ndarray, direct: int) -> np.ndarray:
    """Envelope -> RANGE_BINS log-scaled features starting at the direct-path peak."""
    seg = env[direct:direct + RANGE_SAMPLES]
    if len(seg) < RANGE_SAMPLES:
        seg = np.pad(seg, (0, RANGE_SAMPLES - len(seg)))
    ref = max(float(env[direct]), 1e-9)
    pooled = seg.reshape(RANGE_BINS, POOL).max(axis=1) / ref
    return np.clip((np.log10(pooled + LOG_FLOOR) + 4.0) / 4.0, 0.0, 1.0).astype(np.float32)


def ping_features(stereo: np.ndarray, search: int = 600) -> np.ndarray:
    """stereo: [n, C] float recording that starts at (or shortly before) a chirp.

    Returns [RANGE_BINS, CHANNELS]. The direct path is the strongest peak in the
    first `search` lags of the first channel; both channels share that alignment so
    the inter-mic delay (a left/right cue) is preserved.
    """
    if stereo.ndim == 1:
        stereo = stereo[:, None]
    if stereo.shape[1] < CHANNELS:
        stereo = np.repeat(stereo[:, :1], CHANNELS, axis=1)
    envs = [envelope(stereo[:, c]) for c in range(CHANNELS)]
    direct = int(np.argmax(envs[0][:search]))
    return np.stack([profile(e, direct) for e in envs], axis=-1)


def normalize_nearness(inv_depth: np.ndarray) -> np.ndarray:
    """Relative inverse depth -> [0, 1] nearness, robust per-frame min/max.

    Same normalisation the app applies to the MiDaS camera teacher (TeacherDepth.kt).
    """
    lo, hi = np.percentile(inv_depth, [2, 98])
    return np.clip((inv_depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
