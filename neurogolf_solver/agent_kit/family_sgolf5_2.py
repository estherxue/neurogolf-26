"""family_sgolf5_2 -- cheaper EXACT solvers, GRID-AGNOSTIC and NO-CROP.

Every model here works on the full 30x30 one-hot canvas end to end (input
[1,10,30,30] -> output [1,10,30,30]); nothing is Sliced to a smaller work area
and Padded back.  The cheapenings are the allowed grid-agnostic kind:

  * intermediates are collapsed to SINGLE-CHANNEL [1,1,30,30] fields (10x memory
    cut vs a [1,10,30,30] intermediate), and the boolean masks stay 1-byte;
  * the final answer is written straight into the FREE `output` tensor with a
    single `Where(mask, colour_onehot[1,10,1,1], input)` -- no [1,10,30,30]
    intermediate is ever materialised, and `output` bytes are not charged.

Currently implemented rule (origin-anchored, translation-equivariant):

  frame_dominoes (task 278)
  -------------------------
  A "domino" is a pair of orthogonally-adjacent cells of colour `d`.  Every
  background cell that is 8-adjacent to a domino cell is painted colour `f`;
  isolated `d` cells and everything else are left untouched.  This is a local
  rule of radius 2 but it is NOT linearly separable (an OR of ANDs), so the
  single-Conv LUT family cannot express it -- here it is a 3-op chain:

     conv1 = Conv(input, K1)      # 4*[cell==d] + (#orthogonal d-neighbours)
     domino = Cast(conv1 > 4.5)   # a d-cell WITH >=1 orthogonal d-neighbour
     dil    = Conv(domino, ones3) # 8-neighbour dilation count
     mask   = (dil > 0.5) AND (input-is-background)
     output = Where(mask, onehot(f), input)

  xdiag (task 375)
  ----------------
  The grid is a solid colour `C` with exactly one interior background hole at
  (hr,hc).  The output blanks (-> colour 0) the two full diagonals through the
  hole, i.e. every cell with r-c==hr-hc or r+c==hr+hc; the rest stays colour C.
  Instead of an O(grid) diagonal propagation we read the hole coordinate
  algebraically: the hole is exactly input's channel-0 plane, so
  hm = sum(H0*(r-c)) and hp = sum(H0*(r+c)) recover its two diagonal offsets,
  and diag = (|(r-c)-hm|<.5) OR (|(r+c)-hp|<.5).  Gated to the real grid
  (sum of channels > 0) it is written into the free `output` via
  Where(diag_on_grid, onehot(0), input) -- colour C is never hard-coded.

  All weights are analytic; the harness is the final exactness judge over
  train+test+arc-gen.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX arithmetic exactly)                        #
# --------------------------------------------------------------------------- #
def _frame_dominoes_np(a, d, f):
    H, W = a.shape
    dm = (a == d).astype(np.int32)
    orth = np.zeros((H, W), np.int32)
    orth[1:] += dm[:-1]
    orth[:-1] += dm[1:]
    orth[:, 1:] += dm[:, :-1]
    orth[:, :-1] += dm[:, 1:]
    domino = ((dm == 1) & (orth >= 1)).astype(np.int32)
    dil = np.zeros((H, W), np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            dil[r0:r1, c0:c1] += domino[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    frame = (dil >= 1) & (a == 0)
    out = a.copy()
    out[frame] = f
    return out


# --------------------------------------------------------------------------- #
# ONNX builder                                                                 #
# --------------------------------------------------------------------------- #
def _build_frame_dominoes(d, f):
    # conv1: reads only channel d.  centre weight 4, the 4 orthogonal cells 1.
    K1 = np.zeros((1, CHANNELS, 3, 3), np.float32)
    K1[0, d, 1, 1] = 4.0
    K1[0, d, 0, 1] = 1.0
    K1[0, d, 2, 1] = 1.0
    K1[0, d, 1, 0] = 1.0
    K1[0, d, 1, 2] = 1.0
    ones3 = np.ones((1, 1, 3, 3), np.float32)
    onehot_f = np.zeros((1, CHANNELS, 1, 1), np.float32)
    onehot_f[0, f, 0, 0] = 1.0

    w1 = oh.make_tensor("K1", DATA_TYPE, [1, CHANNELS, 3, 3], K1.ravel().tolist())
    w2 = oh.make_tensor("K2", DATA_TYPE, [1, 1, 3, 3], ones3.ravel().tolist())
    wf = oh.make_tensor("OHF", DATA_TYPE, [1, CHANNELS, 1, 1], onehot_f.ravel().tolist())
    # background plane extractor: channel 0
    Kbg = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Kbg[0, 0, 0, 0] = 1.0
    wbg = oh.make_tensor("KBG", DATA_TYPE, [1, CHANNELS, 1, 1], Kbg.ravel().tolist())

    t45 = oh.make_tensor("t45", DATA_TYPE, [1], [4.5])
    t05 = oh.make_tensor("t05", DATA_TYPE, [1], [0.5])

    nodes = [
        oh.make_node("Conv", ["input", "K1"], ["conv1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        oh.make_node("Greater", ["conv1", "t45"], ["dom_b"]),
        oh.make_node("Cast", ["dom_b"], ["dom_f"], to=DATA_TYPE),
        oh.make_node("Conv", ["dom_f", "K2"], ["dil"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        oh.make_node("Greater", ["dil", "t05"], ["dil_b"]),
        oh.make_node("Conv", ["input", "KBG"], ["bg"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node("Greater", ["bg", "t05"], ["bg_b"]),
        oh.make_node("And", ["dil_b", "bg_b"], ["mask"]),
        oh.make_node("Where", ["mask", "OHF", "input"], ["output"]),
    ]
    inits = [w1, w2, wf, wbg, t45, t05]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# xdiag                                                                        #
# --------------------------------------------------------------------------- #
def _xdiag_np(a):
    H, W = a.shape
    vals, cnt = np.unique(a, return_counts=True)
    C = vals[np.argmax(cnt)]
    holes = np.argwhere(a != C)
    if len(holes) != 1:
        return None
    if a[tuple(holes[0])] != 0:
        return None
    hr, hc = holes[0]
    out = np.full((H, W), C)
    for r in range(H):
        for c in range(W):
            if r - c == hr - hc or r + c == hr + hc:
                out[r, c] = 0
    return out


def _build_xdiag():
    rmc = np.array([[float(r - c) for c in range(WIDTH)] for r in range(HEIGHT)], np.float32)
    rpc = np.array([[float(r + c) for c in range(WIDTH)] for r in range(HEIGHT)], np.float32)
    Kc0 = np.zeros((1, CHANNELS, 1, 1), np.float32); Kc0[0, 0, 0, 0] = 1.0
    Kall = np.ones((1, CHANNELS, 1, 1), np.float32)
    onehot0 = np.zeros((1, CHANNELS, 1, 1), np.float32); onehot0[0, 0, 0, 0] = 1.0

    inits = [
        oh.make_tensor("RMC", DATA_TYPE, [1, 1, HEIGHT, WIDTH], rmc.ravel().tolist()),
        oh.make_tensor("RPC", DATA_TYPE, [1, 1, HEIGHT, WIDTH], rpc.ravel().tolist()),
        oh.make_tensor("KC0", DATA_TYPE, [1, CHANNELS, 1, 1], Kc0.ravel().tolist()),
        oh.make_tensor("KALL", DATA_TYPE, [1, CHANNELS, 1, 1], Kall.ravel().tolist()),
        oh.make_tensor("OH0", DATA_TYPE, [1, CHANNELS, 1, 1], onehot0.ravel().tolist()),
        oh.make_tensor("half", DATA_TYPE, [1], [0.5]),
    ]
    nodes = [
        oh.make_node("Conv", ["input", "KC0"], ["H0"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node("Mul", ["RMC", "H0"], ["mM"]),
        oh.make_node("ReduceSum", ["mM"], ["hm"], axes=[2, 3], keepdims=1),
        oh.make_node("Mul", ["RPC", "H0"], ["mP"]),
        oh.make_node("ReduceSum", ["mP"], ["hp"], axes=[2, 3], keepdims=1),
        oh.make_node("Sub", ["RMC", "hm"], ["s1"]),
        oh.make_node("Abs", ["s1"], ["a1"]),
        oh.make_node("Less", ["a1", "half"], ["b1"]),
        oh.make_node("Sub", ["RPC", "hp"], ["s2"]),
        oh.make_node("Abs", ["s2"], ["a2"]),
        oh.make_node("Less", ["a2", "half"], ["b2"]),
        oh.make_node("Or", ["b1", "b2"], ["diag"]),
        oh.make_node("Conv", ["input", "KALL"], ["within"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node("Greater", ["within", "half"], ["withb"]),
        oh.make_node("And", ["diag", "withb"], ["mask"]),
        oh.make_node("Where", ["mask", "OH0", "input"], ["output"]),
    ]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# detection                                                                    #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
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


def _detect_frame_dominoes(allp):
    if not all(a.shape == b.shape for a, b in allp):
        return None
    in_cols, out_cols = set(), set()
    for a, b in allp:
        in_cols |= set(np.unique(a).tolist())
        out_cols |= set(np.unique(b).tolist())
    # background must be 0 and present
    if 0 not in in_cols:
        return None
    # colours that appear only in outputs -> candidate frame colour f
    only_out = (out_cols - in_cols) - {0}
    # input non-background colours -> candidate domino colour d
    in_fg = in_cols - {0}
    if len(only_out) != 1 or len(in_fg) != 1:
        return None
    f = only_out.pop()
    d = in_fg.pop()
    if not (1 <= d <= 9) or not (1 <= f <= 9) or d == f:
        return None
    for a, b in allp:
        if not np.array_equal(_frame_dominoes_np(a, d, f), b):
            return None
    return d, f


def candidates(ex):
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not allp:
        return []
    out = []
    fd = _detect_frame_dominoes(allp)
    if fd is not None:
        d, f = fd
        try:
            out.append((f"framedom_d{d}f{f}", _build_frame_dominoes(d, f)))
        except Exception:
            pass

    if all(a.shape == b.shape for a, b in allp):
        okx = True
        for a, b in allp:
            o = _xdiag_np(a)
            if o is None or not np.array_equal(o, b):
                okx = False
                break
        if okx:
            try:
                out.append(("xdiag", _build_xdiag()))
            except Exception:
                pass
    return out
