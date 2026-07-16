"""VERIFIED numpy reference for task138 (5daaa586) — 0/2000 fresh (seed 31337).
The ray fill IS a single directional prefix-max (== directional MaxPool / dir_reach).
Build the ONNX from THIS. The one hard part is the data-dependent crop+translate of
the frame region [up..down]x[left..right] to the top-left of the 30x30 output.
"""
import numpy as np


def mode(a):
    v, c = np.unique(a, return_counts=True)
    return int(v[c.argmax()])


def solve(gi):
    g = np.array(gi); H, W = g.shape
    # frame lines = high nonzero-density cols/rows (>=70%); color = mode of the line.
    fc = [c for c in range(W) if (g[:, c] > 0).sum() >= 0.7 * H]
    fr = [r for r in range(H) if (g[r, :] > 0).sum() >= 0.7 * W]
    if len(fc) < 2 or len(fr) < 2:
        return None
    left, right, up, down = min(fc), max(fc), min(fr), max(fr)
    cL = mode(g[:, left][g[:, left] > 0]); cR = mode(g[:, right][g[:, right] > 0])
    cU = mode(g[up, :][g[up, :] > 0]); cD = mode(g[down, :][g[down, :] > 0])
    interior = g[up + 1:down, left + 1:right]
    vals = [v for v in np.unique(interior[interior > 0]) if v in (cL, cR, cU, cD)]
    if not vals:
        return g[up:down + 1, left:right + 1].copy()
    dc = vals[0]
    dr, dcx = (0, -1) if dc == cL else (0, 1) if dc == cR else (-1, 0) if dc == cU else (1, 0)
    seed = np.zeros((H, W), bool)
    seed[up + 1:down, left + 1:right] = (g[up + 1:down, left + 1:right] == dc)
    fill = np.zeros((H, W), bool)
    if dcx != 0:  # horizontal ray -> directional prefix-max along rows
        for r in range(up + 1, down):
            on = False
            rng = range(left + 1, right) if dcx > 0 else range(right - 1, left, -1)
            for c in rng:
                on = on or seed[r, c]
                if on:
                    fill[r, c] = True
    else:         # vertical ray -> along cols
        for c in range(left + 1, right):
            on = False
            rng = range(up + 1, down) if dr > 0 else range(down - 1, up, -1)
            for r in rng:
                on = on or seed[r, c]
                if on:
                    fill[r, c] = True
    out = g[up:down + 1, left:right + 1].copy()   # frame region incl. corners (draw-order free)
    out[fill[up:down + 1, left:right + 1]] = dc
    return out
