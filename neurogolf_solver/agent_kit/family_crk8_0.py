"""family_crk8_0 -- flood-fill + checkerboard recolour family.

Rule covered
------------
The grid has a background colour 0, a "wall" colour, and TWO sparse "seed"
colours.  Each seed colour occupies cells of a SINGLE (r+c) parity, and the two
seeds sit on OPPOSITE parities.  Starting from the seed cells, a 4-connected
flood spreads through background(0) cells.  Every flooded background cell is
recoloured by parity: even-(r+c) cells get the even-parity seed's colour, odd
cells get the odd-parity seed's colour.  Walls block the flood; seeds and walls
keep their colour; unreached background stays 0.

This is the rule of task 286 (verified EXACT on all train/test/arc-gen, 265/265).
Key point about the I/O contract: a PADDING cell is all-zero on every channel,
while a real background(0) cell has channel-0 == 1.  Therefore channel-0 is the
real background ONLY, the flood started there can never leak into padding, and
no grid-extent / bounding-box reasoning is needed (an earlier bbox attempt was
in fact harmful, severing real grid-border cells that happened to be
content-free).

ONNX construction (opset-10, static [1,10,30,30])
-------------------------------------------------
All masks are 1-channel [1,1,30,30] floats; per-channel scalars are [1,10,1,1].

  passable = Conv_1x1(input, ch0)                   real background (NOT padding)
  evencnt  = ReduceSum_{h,w}(input * EVEN)          ([1,10,1,1])
  oddcnt   = ReduceSum_{h,w}(input) - evencnt
  gE[c]    = (evencnt>0) & (oddcnt==0)              even-parity seed channel
  gO[c]    = (oddcnt>0) & (evencnt==0)              odd-parity  seed channel
  seedmask = ReduceSum_c(input * (gE+gO))           seed cells, 1-channel
  reach_0  = seedmask;  reach_k = min(Conv_plus(reach_{k-1}), passable)
  fill     = reach_N                                flooded background cells
  out      = input + gE*fill*EVEN + gO*fill*(1-EVEN) - bg*fill

The family validates the EXACT rule (incl. arc-gen) in numpy before emitting,
so it stays silent unless an instance truly matches.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT


# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "crk8", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _even_checker():
    ii, jj = np.mgrid[0:HEIGHT, 0:WIDTH]
    return ((ii + jj) % 2 == 0).astype(np.float32).reshape(1, 1, HEIGHT, WIDTH)


# --------------------------------------------------------------------------- #
# numpy mirror of the exact ONNX graph (used for detection + N selection)      #
# --------------------------------------------------------------------------- #
def _conv_plus(r):
    d = np.zeros_like(r)
    d[1:, :] += r[:-1, :]; d[:-1, :] += r[1:, :]
    d[:, 1:] += r[:, :-1]; d[:, :-1] += r[:, 1:]
    d += r
    return d


def _onehot(grid):
    """Faithful encoding: real grid cells get a one-hot channel; PADDING is
    all-zero on every channel (so channel-0 == real background ONLY)."""
    g = np.asarray(grid, int); H, W = g.shape
    oh_ = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    for c in range(CHANNELS):
        oh_[c, :H, :W] = (g[:H, :W] == c)
    return oh_, H, W


def _gates(oh_):
    E = _even_checker()[0, 0]
    evenc = (oh_ * E).reshape(CHANNELS, -1).sum(1)
    total = oh_.reshape(CHANNELS, -1).sum(1)
    oddc = total - evenc
    gE = ((evenc > 0.5) & (oddc < 0.5)).astype(np.float32)
    gO = ((oddc > 0.5) & (evenc < 0.5)).astype(np.float32)
    return E, gE, gO


def _simulate(grid, nsteps):
    """Apply the family rule exactly as the ONNX would, to a raw HxW grid.
    passable = channel-0 (real background only); padding is all-zero so the
    flood cannot leak past the grid -- no bounding box needed."""
    oh_, H, W = _onehot(grid)
    E, gE, gO = _gates(oh_)
    seedmask = (oh_ * (gE + gO)[:, None, None]).sum(0)
    passable = oh_[0]
    r = seedmask.copy()
    for _ in range(nsteps):
        r = np.minimum(_conv_plus(r), passable)
    fill = r
    fe = fill * E; fo = fill - fe
    out = oh_.copy()
    for c in range(CHANNELS):
        out[c] = out[c] + gE[c] * fe + gO[c] * fo
    out[0] = out[0] - fill
    # decode like the grader: the unique channel that is > 0
    pos = out > 0
    res = np.full((H, W), 0, int)
    for r0 in range(H):
        for c0 in range(W):
            idx = np.where(pos[:, r0, c0])[0]
            res[r0, c0] = idx[0] if idx.size == 1 else -1
    return res


def _converge_steps(grid):
    """Steps for the flood to stop changing (with this exact scheme)."""
    oh_, H, W = _onehot(grid)
    E, gE, gO = _gates(oh_)
    seedmask = (oh_ * (gE + gO)[:, None, None]).sum(0)
    passable = oh_[0]
    r = seedmask.copy()
    for k in range(400):
        nr = np.minimum(_conv_plus(r), passable)
        if np.array_equal(nr, r):
            return k
        r = nr
    return 400


# --------------------------------------------------------------------------- #
# ONNX builder                                                                 #
# --------------------------------------------------------------------------- #
def _const(name, arr, dtype=FLOAT):
    arr = np.asarray(arr)
    return oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist())


def _build(nsteps):
    nodes, inits = [], []
    E = _even_checker()
    inits.append(_const("EVEN", E))
    inits.append(_const("kplus", np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                          np.float32).reshape(1, 1, 3, 3)))
    inits.append(_const("half", np.array([0.5], np.float32)))

    w_ch0 = np.zeros((1, CHANNELS, 1, 1), np.float32); w_ch0[0, 0, 0, 0] = 1.0
    inits.append(_const("w_ch0", w_ch0))
    bgvec = np.zeros((1, CHANNELS, 1, 1), np.float32); bgvec[0, 0, 0, 0] = 1.0
    inits.append(_const("bgvec", bgvec))

    n = nodes.append
    # passable = channel-0 (real background only; padding is all-zero)
    n(oh.make_node("Conv", ["input", "w_ch0"], ["passable"],
                   kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    # parity counts
    n(oh.make_node("Mul", ["input", "EVEN"], ["even_m"]))
    n(oh.make_node("ReduceSum", ["even_m"], ["evencnt"], axes=[2, 3], keepdims=1))
    n(oh.make_node("ReduceSum", ["input"], ["total"], axes=[2, 3], keepdims=1))
    n(oh.make_node("Sub", ["total", "evencnt"], ["oddcnt"]))

    # gE = (even>0.5)&(odd<0.5); gO = (odd>0.5)&(even<0.5)
    n(oh.make_node("Greater", ["evencnt", "half"], ["e_gt"]))
    n(oh.make_node("Less", ["evencnt", "half"], ["e_lt"]))
    n(oh.make_node("Greater", ["oddcnt", "half"], ["o_gt"]))
    n(oh.make_node("Less", ["oddcnt", "half"], ["o_lt"]))
    n(oh.make_node("Cast", ["e_gt"], ["e_gt_f"], to=FLOAT))
    n(oh.make_node("Cast", ["e_lt"], ["e_lt_f"], to=FLOAT))
    n(oh.make_node("Cast", ["o_gt"], ["o_gt_f"], to=FLOAT))
    n(oh.make_node("Cast", ["o_lt"], ["o_lt_f"], to=FLOAT))
    n(oh.make_node("Mul", ["e_gt_f", "o_lt_f"], ["gE"]))   # [1,10,1,1]
    n(oh.make_node("Mul", ["o_gt_f", "e_lt_f"], ["gO"]))
    n(oh.make_node("Add", ["gE", "gO"], ["gSum"]))

    # seedmask = ReduceSum_c(input * gSum)
    n(oh.make_node("Mul", ["input", "gSum"], ["seed_m"]))
    n(oh.make_node("ReduceSum", ["seed_m"], ["seedmask"], axes=[1], keepdims=1))

    # flood
    prev = "seedmask"
    for s in range(1, nsteps + 1):
        n(oh.make_node("Conv", [prev, "kplus"], [f"nb{s}"],
                       kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        n(oh.make_node("Min", [f"nb{s}", "passable"], [f"r{s}"]))
        prev = f"r{s}"
    n(oh.make_node("Identity", [prev], ["fill"]))

    # recolour
    n(oh.make_node("Mul", ["fill", "EVEN"], ["fill_e"]))
    n(oh.make_node("Sub", ["fill", "fill_e"], ["fill_o"]))
    n(oh.make_node("Mul", ["gE", "fill_e"], ["addE"]))   # [1,10,30,30]
    n(oh.make_node("Mul", ["gO", "fill_o"], ["addO"]))
    n(oh.make_node("Mul", ["bgvec", "fill"], ["remBg"]))
    n(oh.make_node("Add", ["input", "addE"], ["t1"]))
    n(oh.make_node("Add", ["t1", "addO"], ["t2"]))
    n(oh.make_node("Sub", ["t2", "remBg"], ["output"]))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# detection                                                                    #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > HEIGHT or max(b.shape) > WIDTH:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    try:
        prs = _pairs(ex)
    except Exception:
        return []
    if len(prs) < 2:
        return []
    # quick gate: same shape, only 0 -> {two colours} changes, something changes
    changed_any = False
    for a, b in prs:
        if a.shape != b.shape:
            return []
        d = a != b
        if d.any():
            changed_any = True
            if (a[d] != 0).any():
                return []
            if np.unique(b[d]).size > 2:
                return []
    if not changed_any:
        return []

    # choose flood length from convergence over all examples (+margin), capped
    try:
        steps = max(_converge_steps(a) for a, _ in prs)
    except Exception:
        return []
    nsteps = min(int(steps) + 8, 120)
    if nsteps < 1:
        nsteps = 1

    # validate the EXACT family rule on every available example before emitting
    try:
        if not all(np.array_equal(_simulate(a, nsteps), b) for a, b in prs):
            return []
    except Exception:
        return []

    try:
        model = _build(nsteps)
    except Exception:
        return []
    return [(f"crk8_floodcheck_n{nsteps}", model)]
