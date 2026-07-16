"""family_ms_1 — MAX-effort rebuilds for high-memory algorithmic incumbents.

Routing mirrors the other family modules: a numpy ``_ref`` reproduces the TRUE
generator rule for every train+test pair (the fingerprint); the matching ONNX
model is only yielded when the fingerprint reproduces the task exactly.

Task 349 (db93a21d "deathstars"):
    Maroon squares (side 2r) each emit a downward blue beam and a green halo
    that is the square's bounding box grown by r = width//2 on all sides
    (a 2x-scaled square, centred). Priority maroon(9) > green(3) > blue(1) > bg.

    The incumbent (out_blend6/onnx/task349.onnx) composites its uint8 colour
    masks with Min/Max on uint8 tensors. ORT 1.23.2 does NOT implement
    Min/Max for uint8 (verified), so the incumbent FAILS TO LOAD -> 0 pts under
    the pinned runtime. This rebuild reuses the incumbent's proven size-detect /
    halo-stamp QLinearConv weights but replaces the non-loadable uint8-Max
    compositing with a loadable Where/Equal chain. Verified exact on the local
    task + >2500 fresh generator samples; loads under ORT 1.23.2.

    (Task 25 / 1a07d186 "cling" is reported FLOOR — its incumbent is already a
    near-minimal collapsed fp16 line-slot representation.)
"""
from __future__ import annotations

from collections import deque

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

from ng_utils_shim import GRID_SHAPE

F = TP.FLOAT
U8 = TP.UINT8
I8 = TP.INT8
I32 = TP.INT32
I64 = TP.INT64
B = TP.BOOL


# --------------------------------------------------------------------------- #
# task 349 — numpy reference (fingerprint + gate)                             #
# --------------------------------------------------------------------------- #
def _ref349(I):
    I = np.asarray(I, int)
    if I.ndim != 2:
        return None
    H, W = I.shape
    maroon = (I == 9)
    if not maroon.any():
        return None
    # 8-connected components of maroon (each is a solid square)
    visited = np.zeros_like(maroon, bool)
    objs = []
    for i in range(H):
        for j in range(W):
            if maroon[i, j] and not visited[i, j]:
                q = deque([(i, j)])
                visited[i, j] = True
                cells = []
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and maroon[nr, nc] \
                                    and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                objs.append(cells)
    # blue beam: straight down from every maroon cell (prefix, incl. its row)
    beam = np.zeros((H, W), bool)
    for r, c in zip(*np.where(maroon)):
        beam[r:, c] = True
    # green halo: bounding box grown by r = width//2 on all sides
    halo = np.zeros((H, W), bool)
    for cells in objs:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        t, b, l, ri = min(rs), max(rs), min(cs), max(cs)
        rad = (ri - l + 1) // 2
        halo[max(0, t - rad):b + rad + 1, max(0, l - rad):ri + rad + 1] = True
    out = I.copy()
    for i in range(H):
        for j in range(W):
            if maroon[i, j]:
                out[i, j] = 9
            elif halo[i, j] and I[i, j] == 0:
                out[i, j] = 3
            elif beam[i, j] and I[i, j] == 0:
                out[i, j] = 1
    return out


# --------------------------------------------------------------------------- #
# task 349 — proven QLinearConv weights (reconstructed, not embedded verbatim) #
# --------------------------------------------------------------------------- #
def _h_kernel():
    # [5,1,1,12] int8: radius r=k+1 detects a horizontal maroon run of length
    # exactly 2r bordered by non-maroon (the -100 taps); bias -(2r-1) -> fires 1.
    w = np.zeros((5, 1, 1, 12), np.int8)
    for k in range(5):
        r = k + 1
        w[k, 0, 0, 0] = -100
        w[k, 0, 0, 1:1 + 2 * r] = 1
        w[k, 0, 0, 1 + 2 * r] = -100
    return w


def _h_bias():
    return np.array([-1, -3, -5, -7, -9], np.int32)


def _halo_weight():
    # [1,5,11,20] int8: per input channel k (radius r=k+1) a filled value-3 block
    # rows [5-r,5+r], cols [15-3r,14+r]; convolved with the 2r-tall anchor line it
    # covers the full 4r x 4r halo, value 3 (green).
    w = np.zeros((1, 5, 11, 20), np.int8)
    for k in range(5):
        r = k + 1
        w[0, k, 5 - r:5 + r + 1, 15 - 3 * r:14 + r + 1] = 3
    return w


def _t(name, dt, arr):
    a = np.asarray(arr)
    return oh.make_tensor(name, dt, list(a.shape), a.ravel().tolist())


def _s(name, dt, val):
    return oh.make_tensor(name, dt, [], [val])


def _build349():
    inp = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    out = oh.make_tensor_value_info("output", B, GRID_SHAPE)

    inits = [
        _t("ch9_gidx", I64, [9]),
        _s("scale", F, 1.0),
        _s("xz_u8", U8, 0),
        _s("wz_i8", I8, 0),
        _t("hk", I8, _h_kernel()),
        _t("hb", I32, _h_bias()),
        _t("hw", I8, _halo_weight()),
        _s("three", U8, 3),
        _s("nine", U8, 9),
        _s("sent", U8, 250),
        _s("zero_f", F, 0.0),
        _t("k_u8", U8, np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
        _t("ax13", I64, [1, 3]),
        _t("ax12", I64, [1, 2]),
    ]

    n = []
    # maroon mask (channel 9) as float, then u8 (conv/pool) and bool (compositing)
    n.append(oh.make_node("Gather", ["input", "ch9_gidx"], ["ch9"], axis=1))
    n.append(oh.make_node("Cast", ["ch9"], ["ch9_u8"], to=U8))
    n.append(oh.make_node("Cast", ["ch9"], ["M_bool"], to=B))
    # size detection: 5-channel one-hot of radius at the object's left edge
    n.append(oh.make_node(
        "QLinearConv",
        ["ch9_u8", "scale", "xz_u8", "hk", "scale", "wz_i8", "scale", "xz_u8", "hb"],
        ["h_pos_u8"], kernel_shape=[1, 12], pads=[0, 1, 0, 10], strides=[1, 1]))
    # halo stamp -> value 3 over the green box (incl. maroon cells)
    n.append(oh.make_node(
        "QLinearConv",
        ["h_pos_u8", "scale", "xz_u8", "hw", "scale", "wz_i8", "scale", "xz_u8"],
        ["halo_u8"], kernel_shape=[11, 20], pads=[5, 14, 5, 5], strides=[1, 1]))
    n.append(oh.make_node("Cast", ["halo_u8"], ["H_bool"], to=B))
    # blue beam: prefix-max downward of maroon
    n.append(oh.make_node("MaxPool", ["ch9_u8"], ["beam_u8"],
                          kernel_shape=[30, 1], pads=[29, 0, 0, 0], strides=[1, 1]))
    # composite by priority: maroon(9) > green(3) > blue(1) > bg(0)
    n.append(oh.make_node("Where", ["H_bool", "three", "beam_u8"], ["inner"]))
    n.append(oh.make_node("Where", ["M_bool", "nine", "inner"], ["color"]))
    # padding sentinel: out-of-grid rows/cols -> 250 (matches no colour in k_u8)
    n.append(oh.make_node("ReduceMax", ["input", "ax13"], ["vr_f"], keepdims=1))
    n.append(oh.make_node("ReduceMax", ["input", "ax12"], ["vc_f"], keepdims=1))
    n.append(oh.make_node("LessOrEqual", ["vr_f", "zero_f"], ["rp_b"]))
    n.append(oh.make_node("LessOrEqual", ["vc_f", "zero_f"], ["cp_b"]))
    n.append(oh.make_node("Cast", ["rp_b"], ["rp_u8"], to=U8))
    n.append(oh.make_node("Cast", ["cp_b"], ["cp_u8"], to=U8))
    n.append(oh.make_node("Mul", ["rp_u8", "sent"], ["row_sent"]))
    n.append(oh.make_node("Mul", ["cp_u8", "sent"], ["col_sent"]))
    n.append(oh.make_node("Add", ["color", "row_sent"], ["cf1"]))
    n.append(oh.make_node("Add", ["cf1", "col_sent"], ["color_final"]))
    n.append(oh.make_node("Equal", ["color_final", "k_u8"], ["output"]))

    g = oh.make_graph(n, "ms1_349", [inp], [out], inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 18)])
    m.ir_version = 10
    return m


# --------------------------------------------------------------------------- #
# dispatch                                                                     #
# --------------------------------------------------------------------------- #
_TASKS = [
    (_ref349, _build349),
]


def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return []
            if max(a.shape) > 30 or max(b.shape) > 30:
                return []
            out.append((a, b))
    return out


def _matches(ref, prs):
    for a, b in prs:
        o = ref(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    for ref, build in _TASKS:
        if _matches(ref, prs):
            try:
                yield (build.__name__[6:], build())
            except Exception:
                pass
            return
