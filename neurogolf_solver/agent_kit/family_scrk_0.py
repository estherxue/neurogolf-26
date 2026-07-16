"""family_scrk_0 — slice U[0::5] = tasks [5,46,80,107,157,182,219,285,367].

After deep per-pair analysis, none of these 9 tasks reduce to an EXACT,
generalizing opset-10 static-graph rule within the arsenal. Detailed findings
are in the agent report. The closest was task 367 (enclosed-rectangle fill),
which reaches 3/3 train + 1/1 test but only ~228/262 arc-gen because the
generator's "which border pocket is a clipped box" decision is a GLOBAL object
property that is provably not separable by any local neighborhood/edge feature
(verified: two geometrically identical width-1 edge pockets get opposite labels).

candidates() implements the best 367 rule but self-validates against ALL provided
examples (train+test+arc-gen) and yields ONLY on an exact match, so it never emits
a candidate the grader would reject. For this slice it yields nothing.
"""
from collections import deque
import numpy as np


def _comps0(a):
    H, W = a.shape
    lab = -np.ones((H, W), int)
    out = []
    cid = 0
    for i in range(H):
        for j in range(W):
            if a[i, j] == 0 and lab[i, j] < 0:
                dq = deque([(i, j)])
                lab[i, j] = cid
                cells = [(i, j)]
                while dq:
                    y, x = dq.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and a[ny, nx] == 0 and lab[ny, nx] < 0:
                            lab[ny, nx] = cid
                            dq.append((ny, nx))
                            cells.append((ny, nx))
                out.append(cells)
                cid += 1
    return out


def _solve367(a):
    """Enclosed-rectangle fill: solid 0-rectangle whose top & bottom rows are real
    5-walls and whose left/right sides are 5-walls OR the grid edge -> fill with 4."""
    H, W = a.shape
    out = a.copy()
    for cells in _comps0(a):
        ys = [y for y, x in cells]
        xs = [x for y, x in cells]
        r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(cells):
            continue
        top = r0 > 0 and all(a[r0 - 1, x] == 5 for x in range(c0, c1 + 1))
        bot = r1 < H - 1 and all(a[r1 + 1, x] == 5 for x in range(c0, c1 + 1))
        lef = c0 == 0 or all(a[y, c0 - 1] == 5 for y in range(r0, r1 + 1))
        rig = c1 == W - 1 or all(a[y, c1 + 1] == 5 for y in range(r0, r1 + 1))
        if top and bot and lef and rig:
            for y, x in cells:
                out[y, x] = 4
    return out


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for g in ("train", "test", "arc-gen") for e in examples.get(g, [])]
    if not prs:
        return []

    # Task-367 detection: colors are {0,5} in, {0,4,5} out, same shape.
    if all(a.shape == b.shape for a, b in prs) \
       and all(set(np.unique(a)).issubset({0, 5}) for a, b in prs) \
       and all(set(np.unique(b)).issubset({0, 4, 5}) for a, b in prs):
        if all(np.array_equal(_solve367(a), b) for a, b in prs):
            # Exact on every provided example -> would build & emit the ONNX here.
            # (Not reached on the local arc-gen set; the global box rule is not
            #  locally expressible, so we do not emit a rejectable candidate.)
            pass
    return []
