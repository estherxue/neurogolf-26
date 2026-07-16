"""family_golf_1 : cheaper EXACT rebuilds of selected golf-target tasks.

Each candidate is re-derived structurally from the train/test/arc-gen pairs and
verified EXACT in numpy before emitting a minimal opset-10 graph.  The integrator
auto-picks the cheapest correct solver, so we only need to be cheaper than the
incumbent while staying exact.

cost = params + intermediate-tensor memory.  Tricks used here:
  * operate on single-channel [1,1,30,30] intermediates where possible (3600 B)
  * keep spatial extent small (work in the 9x9 / 3x3 sub-grid for fractals)
  * let the final write go to "output" (free) via Concat / Pad
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > HEIGHT or max(b.shape) > HEIGHT:
                continue
            out.append((a, b))
    return out


# --------------------------------------------------------------------------- #
# T217  fractal3 : 9x9 -> 9x9 self-fractal  out[3br+r,3bc+c] = P[r,c]*(P[br,bc]!=0)
# --------------------------------------------------------------------------- #
def _fractal_apply(a):
    if a.shape != (9, 9):
        return None
    P = a.reshape(3, 3, 3, 3).max(axis=(0, 2))      # collapse blocks -> 3x3 pattern
    out = np.zeros((9, 9), int)
    for br in range(3):
        for bc in range(3):
            if P[br, bc] != 0:
                out[3 * br:3 * br + 3, 3 * bc:3 * bc + 3] = P
    return out


def _build_fractal():
    inits = []

    def cst(name, dims, vals, dt=INT64):
        inits.append(oh.make_tensor(name, dt, dims, list(vals)))

    # slice top-left 9x9
    cst("s9_s", [2], [0, 0]); cst("s9_e", [2], [9, 9]); cst("s9_a", [2], [2, 3])
    s9 = oh.make_node("Slice", ["input", "s9_s", "s9_e", "s9_a"], ["s9"])
    # reshape 9x9 -> (br,r,bc,c)
    cst("rsh", [6], [1, CHANNELS, 3, 3, 3, 3])
    rsh = oh.make_node("Reshape", ["s9", "rsh"], ["r6"])
    # collapse blocks -> P [1,10,3,3]
    pm = oh.make_node("ReduceMax", ["r6"], ["P"], axes=[2, 4], keepdims=0)
    # colour channels of P
    cst("p1_s", [1], [1]); cst("p1_e", [1], [CHANNELS]); cst("p1_a", [1], [1])
    p1 = oh.make_node("Slice", ["P", "p1_s", "p1_e", "p1_a"], ["P1"])
    # block-nonzero mask (3x3) then upscale x3 -> 9x9
    nz = oh.make_node("ReduceSum", ["P1"], ["nz"], axes=[1], keepdims=1)
    cst("sc", [4], [1.0, 1.0, 3.0, 3.0], dt=DATA_TYPE)
    B = oh.make_node("Resize", ["nz", "sc"], ["B"], mode="nearest")
    # tile the colour channels to 9x9 then mask
    cst("rep", [4], [1, 1, 3, 3])
    tp = oh.make_node("Tile", ["P1", "rep"], ["tP1"])
    fg = oh.make_node("Mul", ["tP1", "B"], ["fg"])
    fgs = oh.make_node("ReduceSum", ["fg"], ["fgs"], axes=[1], keepdims=1)
    inits.append(oh.make_tensor("ones9", DATA_TYPE, [1, 1, 9, 9], [1.0] * 81))
    out0 = oh.make_node("Sub", ["ones9", "fgs"], ["out0"])
    cat = oh.make_node("Concat", ["out0", "fg"], ["out9"], axis=1)
    pad = oh.make_node("Pad", ["out9"], ["output"], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, HEIGHT - 9, WIDTH - 9])
    return _model([s9, rsh, pm, p1, nz, B, tp, fg, fgs, out0, cat, pad], inits)


# --------------------------------------------------------------------------- #
# T81  local2 : fill the missing corner of an L-tromino of colour X with Y
# --------------------------------------------------------------------------- #
def _corner_apply(a, X, Y):
    h, w = a.shape
    out = a.copy()
    for r in range(h):
        for c in range(w):
            if a[r, c] != 0:
                continue
            hit = False
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                ok = True
                for rr, cc in ((r + dr, c), (r, c + dc), (r + dr, c + dc)):
                    if not (0 <= rr < h and 0 <= cc < w and a[rr, cc] == X):
                        ok = False
                        break
                if ok:
                    hit = True
                    break
            if hit:
                out[r, c] = Y
    return out


def _build_corner(X, Y):
    inits = []

    def cst(name, dims, vals, dt=INT64):
        inits.append(oh.make_tensor(name, dt, dims, list(vals)))

    # channel X (foreground) and channel 0 (background)
    cst("cx_s", [1], [X]); cst("cx_e", [1], [X + 1]); cst("cx_a", [1], [1])
    cx = oh.make_node("Slice", ["input", "cx_s", "cx_e", "cx_a"], ["cx"])
    cst("c0_s", [1], [0]); cst("c0_e", [1], [1]); cst("c0_a", [1], [1])
    c0 = oh.make_node("Slice", ["input", "c0_s", "c0_e", "c0_a"], ["bg0"])
    # 4 L-tromino sum kernels, bias -2 so relu>0 iff all 3 neighbours == X
    W1 = np.zeros((4, 1, 3, 3), np.float32)
    for u, (dr, dc) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        for (kr, kc) in ((dr + 1, 1), (1, dc + 1), (dr + 1, dc + 1)):
            W1[u, 0, kr, kc] = 1.0
    inits.append(oh.make_tensor("W1", DATA_TYPE, [4, 1, 3, 3], W1.ravel().tolist()))
    inits.append(oh.make_tensor("B1", DATA_TYPE, [4], [-2.0] * 4))
    conv = oh.make_node("Conv", ["cx", "W1", "B1"], ["blk"],
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    rel = oh.make_node("Relu", ["blk"], ["blkr"])
    cand = oh.make_node("ReduceSum", ["blkr"], ["cand"], axes=[1], keepdims=1)
    newc = oh.make_node("Mul", ["cand", "bg0"], ["newc"])      # new Y cells
    cnew0 = oh.make_node("Sub", ["bg0", "newc"], ["cbg"])      # remaining bg
    zero = oh.make_node("Sub", ["cx", "cx"], ["z"])            # [1,1,30,30] zeros
    chans = []
    for i in range(CHANNELS):
        if i == 0:
            chans.append("cbg")
        elif i == Y:
            chans.append("newc")
        elif i == X:
            chans.append("cx")
        else:
            chans.append("z")
    cat = oh.make_node("Concat", chans, ["output"], axis=1)
    return _model([cx, c0, conv, rel, cand, newc, cnew0, zero, cat], inits)


# --------------------------------------------------------------------------- #
# T196  loopfill : recolor 1-components that ENCLOSE a hole from 1 -> 3
#   (single-channel masked floods; incumbent used 10-channel intermediates)
# --------------------------------------------------------------------------- #
def _flood4(a, seed, mask):
    reach = seed & mask
    while True:
        nb = np.zeros_like(reach)
        nb[1:, :] |= reach[:-1, :]; nb[:-1, :] |= reach[1:, :]
        nb[:, 1:] |= reach[:, :-1]; nb[:, :-1] |= reach[:, 1:]
        new = (nb & mask) | reach
        if (new == reach).all():
            return reach
        reach = new


def _loopfill_apply(a):
    h, w = a.shape
    bg = (a == 0)
    seed = np.zeros_like(bg)
    seed[0, :] |= bg[0, :]; seed[-1, :] |= bg[-1, :]
    seed[:, 0] |= bg[:, 0]; seed[:, -1] |= bg[:, -1]
    reach = _flood4(a, seed, bg)
    holes = bg & ~reach
    one = (a == 1)
    hd = np.zeros_like(holes)
    hd[1:, :] |= holes[:-1, :]; hd[:-1, :] |= holes[1:, :]
    hd[:, 1:] |= holes[:, :-1]; hd[:, :-1] |= holes[:, 1:]
    recolor = _flood4(a, one & hd, one)
    out = a.copy()
    out[recolor] = 3
    return out


def _build_loopfill(N1=18, N2=5):
    inits = []

    def cst(name, dims, vals, dt=INT64):
        inits.append(oh.make_tensor(name, dt, dims, list(vals)))

    cst("k5", [1, 1, 3, 3], [0, 1, 0, 1, 1, 1, 0, 1, 0], dt=DATA_TYPE)
    cst("k4", [1, 1, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1, 0], dt=DATA_TYPE)
    cst("four", [1], [4.0], dt=DATA_TYPE)
    cst("c0s", [1], [0]); cst("c0e", [1], [1]); cst("ax1", [1], [1])
    cst("c1s", [1], [1]); cst("c1e", [1], [2])

    nodes = []
    nodes.append(oh.make_node("Slice", ["input", "c0s", "c0e", "ax1"], ["bg"]))
    nodes.append(oh.make_node("Slice", ["input", "c1s", "c1e", "ax1"], ["one"]))
    nodes.append(oh.make_node("ReduceMax", ["input"], ["region"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Conv", ["region", "k4"], ["rn"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(oh.make_node("Less", ["rn", "four"], ["lt"]))
    nodes.append(oh.make_node("Cast", ["lt"], ["ltf"], to=DATA_TYPE))
    nodes.append(oh.make_node("Mul", ["bg", "ltf"], ["reach0"]))
    prev = "reach0"
    for i in range(N1):
        nodes.append(oh.make_node("Conv", [prev, "k5"], [f"bc{i}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Mul", [f"bc{i}", "bg"], [f"br{i}"]))
        prev = f"br{i}"
    nodes.append(oh.make_node("Sub", ["bg", prev], ["holesraw"]))
    nodes.append(oh.make_node("Relu", ["holesraw"], ["holes"]))
    nodes.append(oh.make_node("Conv", ["holes", "k4"], ["hd"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(oh.make_node("Mul", ["one", "hd"], ["s1"]))
    prev = "s1"
    for i in range(N2):
        nodes.append(oh.make_node("Conv", [prev, "k5"], [f"cc{i}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Mul", [f"cc{i}", "one"], [f"cr{i}"]))
        prev = f"cr{i}"
    nodes.append(oh.make_node("Sub", ["one", prev], ["ch1"]))
    nodes.append(oh.make_node("Sub", ["bg", "bg"], ["z"]))
    chans = ["bg", "ch1", "z", prev, "z", "z", "z", "z", "z", "z"]
    nodes.append(oh.make_node("Concat", chans, ["output"], axis=1))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    # ---- T217 fractal ----
    if all(a.shape == (9, 9) and b.shape == (9, 9) for a, b in prs):
        good = True
        for a, b in prs:
            o = _fractal_apply(a)
            if o is None or not np.array_equal(o, b):
                good = False
                break
        if good:
            out.append(("fractal9_kron", _build_fractal()))

    # ---- T196 loopfill ----
    if all(a.shape == b.shape for a, b in prs):
        in_cols = set(int(v) for a, _ in prs for v in np.unique(a))
        out_new = set(int(v) for _, b in prs for v in np.unique(b)) - in_cols
        if in_cols <= {0, 1} and out_new == {3}:
            good = True
            for a, b in prs:
                if not np.array_equal(_loopfill_apply(a), b):
                    good = False
                    break
            if good:
                out.append(("loopfill_holes", _build_loopfill()))

    # ---- T81 corner fill ----
    if all(a.shape == b.shape for a, b in prs):
        in_nz = set(int(v) for a, _ in prs for v in np.unique(a) if v != 0)
        new_cols = set(int(v) for a, b in prs for v in np.unique(b)) - \
            set(int(v) for a, b in prs for v in np.unique(a))
        if len(in_nz) == 1 and len(new_cols) == 1:
            X = in_nz.pop()
            Y = new_cols.pop()
            if Y != 0 and X != 0:
                good = True
                for a, b in prs:
                    if not np.array_equal(_corner_apply(a, X, Y), b):
                        good = False
                        break
                if good:
                    out.append((f"corner_X{X}_Y{Y}", _build_corner(X, Y)))

    return out
