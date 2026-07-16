"""GOLF wave-3 slice [2::6]: cheaper EXACT re-solves of already-solved targets.

Two rules are implemented, each chosen for a *much* cheaper ONNX realisation than
the family that currently owns the task (integrator auto-picks the cheapest):

  * t41 ('connect_RLUD')  -> per-colour HORIZONTAL SPAN FILL.
    For every colour, in every row, paint every cell between that colour's
    leftmost and rightmost occurrence.  Currently solved by the Conv->Clip
    doubling chain of family_connectdots (cost ~1.8e5).  We do it with two
    triangular [30,30] MatMuls (prefix/suffix counts over the width axis) and a
    Min -- span_c = min(prefix_c, suffix_c) > 0 exactly between the arms.  All
    intermediates are [1,9,30,30] (background channel dropped) so the cost is
    ~1.4e5 -> ~13.1 pts.

  * t136 ('diag_rays') -> 1-wide diagonal rays from block corners.
    Colour-1 blocks shoot a ray up-left from their top-left corner; colour-2
    blocks shoot a ray down-right from their bottom-right corner.  Corners are
    extracted with two neighbour Convs + Relu, the ray is an OR (Max) of a
    Hillis-Steele DOUBLING chain of diagonal-shift Convs.  Every intermediate is
    a single channel [1,1,30,30] (3600 B) -> ~13.1 pts (vs 12.62 current).

Both detections mirror the ONNX arithmetic in numpy and only emit after
reproducing EVERY available pair (train+test+arc-gen) exactly, so wrong
hypotheses never reach the grader.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# numpy references                                                            #
# --------------------------------------------------------------------------- #
def _percolor_hspan(a):
    out = a.copy()
    H, W = a.shape
    for color in range(1, 10):
        m = (a == color)
        for r in range(H):
            cc = np.where(m[r])[0]
            if len(cc):
                out[r, cc.min():cc.max() + 1] = color
    return out


def _shift(m, dr, dc):
    H, W = m.shape
    out = np.zeros_like(m)
    r0, r1 = max(0, dr), min(H, H + dr)
    c0, c1 = max(0, dc), min(W, W + dc)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def _diag_rays(a):
    H, W = a.shape
    out = a.copy()
    b1 = (a == 1).astype(np.int64)
    b2 = (a == 2).astype(np.int64)
    # bottom-right corners of colour 2 (no right & no below neighbour)
    c2 = b2 & (1 - _shift(b2, 0, -1)) & (1 - _shift(b2, -1, 0))
    # top-left corners of colour 1 (no left & no above neighbour)
    c1 = b1 & (1 - _shift(b1, 0, 1)) & (1 - _shift(b1, 1, 0))
    ray2 = np.zeros_like(a)
    cur = c2.copy()
    for _ in range(1, H + W):
        cur = _shift(cur, 1, 1)
        ray2 = np.maximum(ray2, cur)
    ray1 = np.zeros_like(a)
    cur = c1.copy()
    for _ in range(1, H + W):
        cur = _shift(cur, -1, -1)
        ray1 = np.maximum(ray1, cur)
    out[(ray2 > 0) & (a == 0)] = 2
    out[(ray1 > 0) & (a == 0)] = 1
    return out


# --------------------------------------------------------------------------- #
# ONNX builders                                                                #
# --------------------------------------------------------------------------- #
def _slice(data, out, lo, hi, axis):
    starts = oh.make_tensor(out + "_s", INT64, [1], [lo])
    ends = oh.make_tensor(out + "_e", INT64, [1], [hi])
    axes = oh.make_tensor(out + "_a", INT64, [1], [axis])
    node = oh.make_node("Slice", [data, out + "_s", out + "_e", out + "_a"], [out])
    return node, [starts, ends, axes]


def build_hspan():
    """Per-colour horizontal span fill via triangular MatMuls on the width axis."""
    nodes, inits = [], []
    # nobg = input[:, 1:10]  -> [1,9,30,30]
    n, i = _slice("input", "nobg", 1, CHANNELS, 1)
    nodes.append(n); inits += i
    # in0 = input[:, 0:1]
    n, i = _slice("input", "in0", 0, 1, 1)
    nodes.append(n); inits += i
    # L[w,k] = 1 if w<=k  (prefix), U[w,k] = 1 if w>=k (suffix)
    L = np.triu(np.ones((WIDTH, WIDTH), np.float32))      # L[w,k]=1 for w<=k
    U = np.tril(np.ones((WIDTH, WIDTH), np.float32))      # U[w,k]=1 for w>=k
    inits.append(oh.make_tensor("L", DATA_TYPE, [WIDTH, WIDTH], L.ravel().tolist()))
    inits.append(oh.make_tensor("U", DATA_TYPE, [WIDTH, WIDTH], U.ravel().tolist()))
    nodes.append(oh.make_node("MatMul", ["nobg", "L"], ["pref"]))
    nodes.append(oh.make_node("MatMul", ["nobg", "U"], ["suff"]))
    nodes.append(oh.make_node("Min", ["pref", "suff"], ["color"]))  # [1,9,30,30]
    nodes.append(oh.make_node("ReduceSum", ["color"], ["csum"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Sub", ["in0", "csum"], ["bg_raw"]))
    nodes.append(oh.make_node("Relu", ["bg_raw"], ["ch0"]))
    nodes.append(oh.make_node("Concat", ["ch0", "color"], ["output"], axis=1))
    return _model(nodes, inits)


def _conv_shift(data, out, kh, kw, pads, oh_i, ow_i):
    """Single-tap Conv: out[r,c] = in_padded[r+oh_i, c+ow_i].
    `pads` is ONNX 2-D order [top, left, bottom, right]."""
    w = np.zeros((1, 1, kh, kw), np.float32)
    w[0, 0, oh_i, ow_i] = 1.0
    wt = oh.make_tensor(out + "_w", DATA_TYPE, [1, 1, kh, kw], w.ravel().tolist())
    node = oh.make_node("Conv", [data, out + "_w"], [out],
                        kernel_shape=[kh, kw], pads=list(pads))
    return node, [wt]


def _shift_dr(data, out, s):
    # out[r,c] = in[r-s, c-s]: pad top=s,left=s, kernel (s+1)^2, tap (0,0)
    return _conv_shift(data, out, s + 1, s + 1, [s, s, 0, 0], 0, 0)


def _shift_ul(data, out, s):
    # out[r,c] = in[r+s, c+s]: pad bottom=s,right=s, kernel (s+1)^2, tap (s,s)
    return _conv_shift(data, out, s + 1, s + 1, [0, 0, s, s], s, s)


def build_diag_rays():
    nodes, inits = [], []
    # b1, b2, in0
    for ch, nm in ((0, "in0"), (1, "b1"), (2, "b2")):
        n, i = _slice("input", nm, ch, ch + 1, 1)
        nodes.append(n); inits += i

    # --- corner of colour 2 (bottom-right): b2 - rightnb - belownb, relu ---
    # right neighbour: out[r,c]=in[r,c+1] -> pad right1, kernel 1x2 tap (0,1)
    n, i = _conv_shift("b2", "rnb", 1, 2, [0, 0, 0, 1], 0, 1)
    nodes.append(n); inits += i
    # below neighbour: out[r,c]=in[r+1,c] -> pad bottom1, kernel 2x1 tap (1,0)
    n, i = _conv_shift("b2", "bnb", 2, 1, [0, 0, 1, 0], 1, 0)
    nodes.append(n); inits += i
    nodes.append(oh.make_node("Add", ["rnb", "bnb"], ["nb2"]))
    nodes.append(oh.make_node("Sub", ["b2", "nb2"], ["cr2"]))
    nodes.append(oh.make_node("Relu", ["cr2"], ["c2"]))

    # --- corner of colour 1 (top-left): b1 - leftnb - abovenb, relu ---
    # left neighbour: out[r,c]=in[r,c-1] -> pad left1, kernel 1x2 tap (0,0)
    n, i = _conv_shift("b1", "lnb", 1, 2, [0, 1, 0, 0], 0, 0)
    nodes.append(n); inits += i
    # above neighbour: out[r,c]=in[r-1,c] -> pad top1, kernel 2x1 tap (0,0)
    n, i = _conv_shift("b1", "anb", 2, 1, [1, 0, 0, 0], 0, 0)
    nodes.append(n); inits += i
    nodes.append(oh.make_node("Add", ["lnb", "anb"], ["nb1"]))
    nodes.append(oh.make_node("Sub", ["b1", "nb1"], ["cr1"]))
    nodes.append(oh.make_node("Relu", ["cr1"], ["c1"]))

    offs = [1, 1, 2, 4, 8, 16]
    # ray2 (down-right) doubling
    n, i = _shift_dr("c2", "r2_0", offs[0])
    nodes.append(n); inits += i
    prev = "r2_0"
    for k in range(1, len(offs)):
        sh = f"r2_sh{k}"
        n, i = _shift_dr(prev, sh, offs[k])
        nodes.append(n); inits += i
        nxt = f"r2_{k}"
        nodes.append(oh.make_node("Max", [prev, sh], [nxt]))
        prev = nxt
    ray2 = prev
    # ray1 (up-left) doubling
    n, i = _shift_ul("c1", "r1_0", offs[0])
    nodes.append(n); inits += i
    prev = "r1_0"
    for k in range(1, len(offs)):
        sh = f"r1_sh{k}"
        n, i = _shift_ul(prev, sh, offs[k])
        nodes.append(n); inits += i
        nxt = f"r1_{k}"
        nodes.append(oh.make_node("Max", [prev, sh], [nxt]))
        prev = nxt
    ray1 = prev

    # rays must stop at the REAL grid edge, not the 30x30 frame edge: mask by the
    # in-grid rectangle (=1 where any input channel is set, 0 in the zero pad).
    nodes.append(oh.make_node("ReduceSum", ["input"], ["ingrid"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Mul", [ray2, "ingrid"], ["ray2m"]))
    nodes.append(oh.make_node("Mul", [ray1, "ingrid"], ["ray1m"]))
    ray2, ray1 = "ray2m", "ray1m"

    # assemble channels
    nodes.append(oh.make_node("Max", ["b2", ray2], ["och2"]))
    nodes.append(oh.make_node("Max", ["b1", ray1], ["och1"]))
    nodes.append(oh.make_node("Add", [ray1, ray2], ["raysum"]))
    nodes.append(oh.make_node("Sub", ["in0", "raysum"], ["bg_raw"]))
    nodes.append(oh.make_node("Relu", ["bg_raw"], ["och0"]))
    nodes.append(oh.make_node("Sub", ["b1", "b1"], ["zero"]))
    chans = ["och0", "och1", "och2"] + ["zero"] * (CHANNELS - 3)
    nodes.append(oh.make_node("Concat", chans, ["output"], axis=1))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# candidates                                                                   #
# --------------------------------------------------------------------------- #
def _pairs(examples):
    out = []
    for key in ("train", "test", "arc-gen"):
        for e in examples.get(key, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if max(a.shape) <= 30 and max(b.shape) <= 30:
                out.append((a, b))
    return out


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return []
    cands = []

    # --- t41: per-colour horizontal span fill ---
    if all(a.shape == b.shape for a, b in prs):
        if all((_percolor_hspan(a) == b).all() for a, b in prs):
            try:
                cands.append(("hspan", build_hspan()))
            except Exception:
                pass

    # --- t136: diagonal corner rays (colour1 UL, colour2 DR) ---
    if all(a.shape == b.shape for a, b in prs):
        cset = set()
        for a, b in prs:
            cset |= set(np.unique(a).tolist()) | set(np.unique(b).tolist())
        if cset <= {0, 1, 2} and all((_diag_rays(a) == b).all() for a, b in prs):
            try:
                cands.append(("diagrays", build_diag_rays()))
            except Exception:
                pass

    return cands
