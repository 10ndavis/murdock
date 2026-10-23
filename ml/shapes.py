"""Scene primitives fitted to a nearness map: the echo AI's second output.

A few big, simple shapes are much easier to learn from echoes than 1024 depth pixels, and
they are what echoes are best at: flat floors and walls return strong, clean reflections,
and large obstacles stand out by range. The camera teacher's depth is turned into these
primitives automatically, so on-device learning needs no hand labelling.

Mirrored in android/app/src/main/kotlin/com/murdock/murdock/ShapeFit.kt; keep the two in sync
(ml/check_shapes_kotlin.py compares them).

All geometry is in the 32x32 view (row 0 = top, column 0 = left), every value in [0, 1]:
  [0:3]    floor : presence, horizon row (0 top .. 1 bottom), nearness at the bottom edge
  [3:12]   walls : left / ahead / right bands, each presence, nearness, slope
                   (slope 0.5 = facing the phone; <0.5 nearer on the band's left side)
  [12:24]  boxes : the two nearest obstacles, each presence, nearness, cx, cy, w, h

Why these fits: on a ground plane, inverse depth is exactly linear in image row, and the
nearness maps are an affine transform of inverse depth, so the floor is a straight line in
the row profile; a wall is a straight line across its band's columns. Obstacles are what
sticks out of that floor+walls background.
"""

import numpy as np

N = 32
SHAPE_DIM = 24
FLOOR, WALLS, BOXES = 0, 3, 12
WALL_BANDS = ((0, 11), (11, 21), (21, 32))   # left, ahead, right columns [from, to)
FLOOR_ROWS = (20, 32)        # rows the floor line is fitted on
MIN_WALL_ROWS = 8
BOX_RISE = 0.12              # nearer than the background by this much -> obstacle
MIN_BOX_CELLS = 6
MAX_BOXES = 2


def _median(v):
    s = np.sort(np.asarray(v, np.float64))
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _lsq(xs, ys):
    xm, ym = xs.mean(), ys.mean()
    sxx = ((xs - xm) ** 2).sum()
    b = ((xs - xm) * (ys - ym)).sum() / sxx if sxx > 0 else 0.0
    return ym - b * xm, b


def _line(xs, ys):
    """Robust y = a + b*x: least squares, then once more without the outliers (a door in a
    wall, a chair leg on the floor). Returns (a, b, fit quality in [0, 1] given the noise
    scale used by the caller: inlier rmse and inlier fraction, see _quality)."""
    xs = np.asarray(xs, np.float64); ys = np.asarray(ys, np.float64)
    a, b = _lsq(xs, ys)
    res = np.abs(a + b * xs - ys)
    tol = max(2.0 * np.sqrt((res ** 2).mean()), 0.03)
    keep = res <= tol
    if keep.sum() >= max(3, len(xs) // 2) and not keep.all():
        a, b = _lsq(xs[keep], ys[keep])
    res = np.abs(a + b * xs - ys)
    inl = res <= tol
    rmse = np.sqrt((res[inl] ** 2).mean()) if inl.any() else 1.0
    return a, b, rmse, inl.mean()


def _clamp(v, lo=0.0, hi=1.0):
    return min(max(v, lo), hi)


def fit_shapes(nearness: np.ndarray) -> np.ndarray:
    """[32, 32] nearness (1 = near) -> [SHAPE_DIM] primitives in [0, 1]."""
    m = np.nan_to_num(np.asarray(nearness, np.float64).reshape(N, N))
    out = np.zeros(SHAPE_DIM, np.float64)

    # ── Floor: the bottom rows' median nearness rises linearly towards the phone.
    rows = np.array([_median(m[y]) for y in range(N)])
    ys = np.arange(*FLOOR_ROWS)
    fa, fb, rmse, frac = _line(ys, rows[ys])
    rise = fb * (FLOOR_ROWS[1] - FLOOR_ROWS[0])         # nearness gained over the fit rows
    floor_p = _clamp((rise - 0.15) / 0.35) * _clamp(1.0 - rmse / 0.08) * frac
    upper = _median(rows[:MIN_WALL_ROWS])
    horizon = (upper - fa) / fb if fb > 1e-6 else float(N - 1)
    horizon = _clamp(horizon, 0.0, N - 1.0)
    out[FLOOR:FLOOR + 3] = floor_p, horizon / (N - 1), _clamp(fa + fb * (N - 1))
    # Rows above the horizon belong to walls (always at least the top MIN_WALL_ROWS), the
    # rest to the floor line, even a faint one: otherwise the floor itself reads as a box.
    wall_rows = max(MIN_WALL_ROWS, int(horizon))

    # ── Walls: per band, a straight line across its columns (rows above the horizon).
    background = np.zeros((N, N))
    for k, (x0, x1) in enumerate(WALL_BANDS):
        cols = np.array([_median(m[:wall_rows, x]) for x in range(x0, x1)])
        xs = (np.arange(x0, x1) - (x0 + x1 - 1) / 2.0) / (x1 - x0)   # band-relative, ~[-0.5, 0.5]
        a, b, rmse, frac = _line(xs, cols)
        o = WALLS + 3 * k
        out[o:o + 3] = _clamp(1.0 - rmse / 0.12) * frac, _clamp(a), _clamp(0.5 + 0.5 * b)
        background[:, x0:x1] = a + b * xs
    for y in range(wall_rows, N):
        background[y, :] = fa + fb * y

    # ── Obstacles: 4-connected blobs clearly nearer than the floor+walls background.
    mask = (m - background) > BOX_RISE
    seen = np.zeros((N, N), bool)
    boxes = []
    for y in range(N):
        for x in range(N):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]; seen[y, x] = True; cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < N and 0 <= nx < N and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; stack.append((ny, nx))
            if len(cells) < MIN_BOX_CELLS:
                continue
            cy = [c[0] for c in cells]; cx = [c[1] for c in cells]
            near = float(np.mean([m[c] for c in cells]))
            rise = float(np.mean([m[c] - background[c] for c in cells]))
            y0, y1, x0, x1 = min(cy), max(cy) + 1, min(cx), max(cx) + 1
            # Small blobs are noise; one covering most of the view is a background misfit.
            area = (y1 - y0) * (x1 - x0) / (N * N)
            presence = _clamp(len(cells) / 16.0) * _clamp(rise / 0.25) * _clamp((0.6 - area) / 0.3)
            boxes.append((near, presence, (x0 + x1) / 2 / N, (y0 + y1) / 2 / N, (x1 - x0) / N, (y1 - y0) / N))
    boxes.sort(key=lambda bx: -bx[0])   # nearest first (stable: ties keep scan order)
    for k, (near, presence, cx, cy, w, h) in enumerate(boxes[:MAX_BOXES]):
        o = BOXES + 6 * k
        out[o:o + 6] = presence, near, cx, cy, w, h
    return out.astype(np.float32)


def fit_batch(depth: np.ndarray) -> np.ndarray:
    return np.stack([fit_shapes(d) for d in depth]) if len(depth) else np.zeros((0, SHAPE_DIM), np.float32)


def presence_mask(shapes):
    """Per-value loss weights: presences always count, the other values only as much as the
    shape is really there (a missing wall's 'slope' is meaningless)."""
    shapes = np.asarray(shapes)
    w = np.ones_like(shapes)
    for o, n in [(FLOOR, 3)] + [(WALLS + 3 * k, 3) for k in range(3)] + [(BOXES + 6 * k, 6) for k in range(MAX_BOXES)]:
        w[..., o + 1:o + n] = shapes[..., o:o + 1]
    return w
