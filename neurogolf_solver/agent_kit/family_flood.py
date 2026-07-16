"""Flood-fill / region-propagation family (origin-anchored, unrolled-CA).

Rule covered
------------
"Fill the ENCLOSED background holes with a fixed color."  A cell is *enclosed*
when it has the background color 0 and is NOT 4-connected to the image border
through other background cells -- i.e. it sits inside a wall of non-background
colors.  Every enclosed background cell is recolored to a single fixed color
`dst`; everything else (walls, border-connected background) is left untouched.

Why this is origin-safe under top-left zero-padding
---------------------------------------------------
In the one-hot tensor a real background(0) cell has channel-0 == 1 while a
PADDING cell is all-zero on every channel.  Define

    open = 1 - sum(channels 1..9)          (1x1 Conv, 1 output channel)

`open` is 1 on real background AND on padding, 0 on walls.  The padding region
is therefore "open" and is contiguous with the grid's bottom/right edges, so it
behaves exactly like the outside.  We seed a reachability mask from the 30x30
tensor border (a fixed constant, origin-invariant) and flood it INWARD through
`open` cells.  Padding + every border-connected background cell becomes
reachable; the holes do not.  enclosed = open AND NOT reachable -- which can
only be real background (padding is always reachable), so the padding stays 0.

Unrolled cellular automaton (Loop is banned)
--------------------------------------------
The flood is a fixed chain of identical 3x3 conv steps:

    reach_0   = open * border_mask
    reach_k   = min( Conv_plus(reach_{k-1}), open )          k = 1..N

`Conv_plus` is a 4-neighbourhood + self kernel; min(.,open) re-clamps to {0,1}
and confines growth to open cells.  After N >= (max geodesic distance from the
border to any outside cell) steps, reach == the true outside.  N is chosen from
the actual examples (the harness validates exactness on train+test+arc-gen).

Output assembly
---------------
    enc      = open - reach_N                                ( {0,1} mask )
    addmap   = Conv_1x1(enc)  with weight -1 on ch0, +1 on ch dst
    output   = input + addmap

so channel 0 loses the hole cells and channel `dst` gains them; all other
channels are untouched.  After the grader's (output>0) threshold this is the
exact one-hot target.

Cost: per step two [1,1,30,30] intermediates (3600 B each); ~12-13 pts.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# graph helper                                                                #
# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "flood", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _border_mask():
    m = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    m[:, :, 0, :] = 1.0
    m[:, :, HEIGHT - 1, :] = 1.0
    m[:, :, :, 0] = 1.0
    m[:, :, :, WIDTH - 1] = 1.0
    return m


def _build(dst, n_steps, conn8=False):
    """ONNX graph: recolor enclosed background-0 holes to color `dst`."""
    nodes, inits = [], []

    # open = 1 - sum(channels 1..9)   (1 on bg/padding, 0 on walls)
    w_open = np.zeros((1, CHANNELS, 1, 1), np.float32)
    w_open[0, 1:, 0, 0] = -1.0
    inits.append(oh.make_tensor("w_open", DATA_TYPE, [1, CHANNELS, 1, 1],
                                w_open.ravel().tolist()))
    inits.append(oh.make_tensor("b_open", DATA_TYPE, [1], [1.0]))
    nodes.append(oh.make_node("Conv", ["input", "w_open", "b_open"], ["open"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    # reach_0 = open * border_mask
    inits.append(oh.make_tensor("bmask", DATA_TYPE, [1, 1, HEIGHT, WIDTH],
                                _border_mask().ravel().tolist()))
    nodes.append(oh.make_node("Mul", ["open", "bmask"], ["r0"]))

    # propagation kernel (4-neigh + self, or full 8-neigh + self)
    if conn8:
        k = np.ones((1, 1, 3, 3), np.float32)
    else:
        k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float32).reshape(1, 1, 3, 3)
    inits.append(oh.make_tensor("kprop", DATA_TYPE, [1, 1, 3, 3], k.ravel().tolist()))

    prev = "r0"
    for s in range(1, n_steps + 1):
        nb = f"nb{s}"
        rk = f"r{s}"
        nodes.append(oh.make_node("Conv", [prev, "kprop"], [nb],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Min", [nb, "open"], [rk]))
        prev = rk

    # enc = open - reach_N
    nodes.append(oh.make_node("Sub", ["open", prev], ["enc"]))

    # addmap: -enc on channel 0, +enc on channel dst
    w_add = np.zeros((CHANNELS, 1, 1, 1), np.float32)
    w_add[0, 0, 0, 0] = -1.0
    w_add[dst, 0, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_add", DATA_TYPE, [CHANNELS, 1, 1, 1],
                                w_add.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["enc", "w_add"], ["addmap"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    nodes.append(oh.make_node("Add", ["input", "addmap"], ["output"]))
    return _model(nodes, inits)


def _build_seed(seed, n_steps, conn8=False):
    """ONNX graph: flood seed color `seed` through background-0 cells until it
    hits a wall (any non-background colour).  Padding never participates because
    the medium is channel-0 ONLY (padding is all-zero, not channel-0==1)."""
    nodes, inits = [], []

    # ch0 = channel 0 (the real background = flood medium)
    w_ch0 = np.zeros((1, CHANNELS, 1, 1), np.float32)
    w_ch0[0, 0, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_ch0", DATA_TYPE, [1, CHANNELS, 1, 1],
                                w_ch0.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["input", "w_ch0"], ["ch0"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    # r0 = channel `seed` (the seed cells = initial reach)
    w_chs = np.zeros((1, CHANNELS, 1, 1), np.float32)
    w_chs[0, seed, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_chs", DATA_TYPE, [1, CHANNELS, 1, 1],
                                w_chs.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["input", "w_chs"], ["r0"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    # allowed = bg + seed  (cells that may end up coloured `seed`)
    nodes.append(oh.make_node("Add", ["ch0", "r0"], ["allowed"]))

    if conn8:
        k = np.ones((1, 1, 3, 3), np.float32)
    else:
        k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float32).reshape(1, 1, 3, 3)
    inits.append(oh.make_tensor("kprop", DATA_TYPE, [1, 1, 3, 3], k.ravel().tolist()))

    prev = "r0"
    for s in range(1, n_steps + 1):
        nb, rk = f"nb{s}", f"r{s}"
        nodes.append(oh.make_node("Conv", [prev, "kprop"], [nb],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Min", [nb, "allowed"], [rk]))
        prev = rk

    # fill = reach AND background  (bg cells newly coloured `seed`)
    nodes.append(oh.make_node("Min", [prev, "ch0"], ["fill"]))

    w_add = np.zeros((CHANNELS, 1, 1, 1), np.float32)
    w_add[0, 0, 0, 0] = -1.0
    w_add[seed, 0, 0, 0] = 1.0
    inits.append(oh.make_tensor("w_add", DATA_TYPE, [CHANNELS, 1, 1, 1],
                                w_add.ravel().tolist()))
    nodes.append(oh.make_node("Conv", ["fill", "w_add"], ["addmap"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    nodes.append(oh.make_node("Add", ["input", "addmap"], ["output"]))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# reference simulator (mirrors the ONNX exactly) -- used for detection + N     #
# --------------------------------------------------------------------------- #
def _padded_open(grid):
    """30x30 int mask: 1 on real bg(0) and on padding, 0 on walls."""
    g = np.asarray(grid, int)
    h, w = g.shape
    op = np.ones((HEIGHT, WIDTH), np.int64)
    op[:h, :w] = (g == 0).astype(np.int64)
    return op


def _flood_outside(op, conn8, n_steps=None):
    """Iteratively grow 'reach' from the tensor border through open cells.
    Returns (reach, steps_used). If n_steps is None, runs to convergence."""
    reach = np.zeros((HEIGHT, WIDTH), np.int64)
    reach[0, :] = op[0, :]
    reach[-1, :] = op[-1, :]
    reach[:, 0] = op[:, 0]
    reach[:, -1] = op[:, -1]
    if conn8:
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    steps = 0
    while True:
        nb = np.zeros_like(reach)
        for dy, dx in offs:
            ys0, ys1 = max(0, dy), HEIGHT + min(0, dy)
            xs0, xs1 = max(0, dx), WIDTH + min(0, dx)
            yd0, yd1 = max(0, -dy), HEIGHT + min(0, -dy)
            xd0, xd1 = max(0, -dx), WIDTH + min(0, -dx)
            nb[yd0:yd1, xd0:xd1] += reach[ys0:ys1, xs0:xs1]
        new = np.minimum(nb, op)
        steps += 1
        if np.array_equal(new, reach):
            steps -= 1
            break
        reach = new
        if n_steps is not None and steps >= n_steps:
            break
    return reach, steps


def _simulate(grid, dst, conn8):
    """Apply the enclosed-hole fill exactly as the ONNX would, to a raw grid."""
    g = np.asarray(grid, int).copy()
    op = _padded_open(g)
    reach, _ = _flood_outside(op, conn8)
    h, w = g.shape
    enc = (op == 1) & (reach == 0)          # enclosed = open & not reached
    enc_grid = enc[:h, :w]
    g[enc_grid] = dst
    return g


def _flood_seed(g, seed, conn8, n_steps=None):
    """Grow 'reach' from seed cells through background-0 cells (raw grid).
    Returns (reach_bool, steps_used)."""
    g = np.asarray(g, int)
    h, w = g.shape
    medium = (g == 0)
    reach = (g == seed)
    if conn8:
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    allowed = medium | reach
    steps = 0
    while True:
        nb = np.zeros((h, w), np.int64)
        ri = reach.astype(np.int64)
        for dy, dx in offs:
            ys0, ys1 = max(0, dy), h + min(0, dy)
            xs0, xs1 = max(0, dx), w + min(0, dx)
            yd0, yd1 = max(0, -dy), h + min(0, -dy)
            xd0, xd1 = max(0, -dx), w + min(0, -dx)
            nb[yd0:yd1, xd0:xd1] += ri[ys0:ys1, xs0:xs1]
        new = (nb > 0) & allowed
        steps += 1
        if np.array_equal(new, reach):
            steps -= 1
            break
        reach = new
        if n_steps is not None and steps >= n_steps:
            break
    return reach, steps


def _simulate_seed(grid, seed, conn8):
    """Apply seed flood-fill exactly as the ONNX would, to a raw grid."""
    g = np.asarray(grid, int).copy()
    reach, _ = _flood_seed(g, seed, conn8)
    g[reach & (np.asarray(grid, int) == 0)] = seed
    return g


# --------------------------------------------------------------------------- #
# detection                                                                    #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits=("train", "test", "arc-gen")):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _infer_dst(prs):
    """All changed cells go 0 -> single color; return that color or None."""
    dst = None
    for a, b in prs:
        if a.shape != b.shape:
            return None
        d = a != b
        if not d.any():
            continue
        if (a[d] != 0).any():           # only background cells may change
            return None
        vals = np.unique(b[d])
        if vals.size != 1:
            return None
        if dst is None:
            dst = int(vals[0])
        elif dst != int(vals[0]):
            return None
    return dst


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    # flood-fill preserves grid size; bail on any shape change
    if not all(a.shape == b.shape for a, b in prs):
        return []
    # need at least one example that actually fills something
    if all((a == b).all() for a, b in prs):
        return []

    dst = _infer_dst(prs)
    if dst is None or dst == 0:
        return []

    out = []

    # ---- rule 1: fill enclosed background holes with a fixed colour ----------
    for conn8 in (False, True):
        if not all(np.array_equal(_simulate(a, dst, conn8), b) for a, b in prs):
            continue
        max_steps = 0
        for a, _ in prs:
            _, st = _flood_outside(_padded_open(a), conn8)
            max_steps = max(max_steps, st)
        n_steps = max(1, max_steps) + 3
        tag = "enc8" if conn8 else "enc4"
        try:
            out.append((f"flood_{tag}_dst{dst}_n{n_steps}",
                        _build(dst, n_steps, conn8)))
        except Exception:
            pass

    # ---- rule 2: flood a seed colour through background until it hits walls ---
    for conn8 in (False, True):
        if not all(np.array_equal(_simulate_seed(a, dst, conn8), b) for a, b in prs):
            continue
        max_steps = 0
        for a, _ in prs:
            _, st = _flood_seed(a, dst, conn8)
            max_steps = max(max_steps, st)
        n_steps = max(1, max_steps) + 3
        tag = "seed8" if conn8 else "seed4"
        try:
            out.append((f"flood_{tag}_c{dst}_n{n_steps}",
                        _build_seed(dst, n_steps, conn8)))
        except Exception:
            pass

    return out
