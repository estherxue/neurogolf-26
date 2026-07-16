"""Two-marker CROSS-PROJECTION on a fixed NxN grid (cheap, single small tensors).

Rule (exact, structural):
  The grid contains exactly two single-cell markers of distinct colours v1, v2.
  Each marker projects a full cross = {its whole row} U {its whole column}.
  Output colours:
    * cell covered by BOTH crosses        -> intersection colour `inter`
    * cell covered by the v1 cross only    -> v1
    * cell covered by the v2 cross only    -> v2
    * otherwise                            -> background (0)

Every example (train/test/arc-gen) is the SAME square size NxN, so we work in a
static top-left NxN window and zero-pad back to 30x30 -- no data-dependent grid
width is needed and the padding gotcha is handled by construction.

Cheapness: the two crosses are obtained by ReduceMax projections of a *channel-
sliced* [1,C',N,N] tensor (C' = span of the two marker channels, typically 2),
so every intermediate is tiny (N is small).  A single 1x1 Conv + Clip turns the
two cross masks into the full one-hot output; a Pad restores 30x30.
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
# numpy reference (mirrors the ONNX projection arithmetic exactly)            #
# --------------------------------------------------------------------------- #
def _cross(a, v1, v2, inter):
    m1 = (a == v1)
    m2 = (a == v2)
    c1 = m1.any(1)[:, None] | m1.any(0)[None, :]
    c2 = m2.any(1)[:, None] | m2.any(0)[None, :]
    return np.where(c1 & c2, inter, np.where(c1, v1, np.where(c2, v2, 0)))


# --------------------------------------------------------------------------- #
# ONNX construction                                                            #
# --------------------------------------------------------------------------- #
def _build(N, v1, v2, inter):
    lo, hi = min(v1, v2), max(v1, v2)
    Cp = hi - lo + 1
    i1, i2 = v1 - lo, v2 - lo  # local channel indices inside the slice

    inits = []

    def _int(name, vals):
        t = oh.make_tensor(name, INT64, [len(vals)], list(vals))
        inits.append(t)
        return name

    s_starts = _int("s_starts", [lo, 0, 0])
    s_ends = _int("s_ends", [hi + 1, N, N])
    s_axes = _int("s_axes", [1, 2, 3])

    W = np.zeros((CHANNELS, Cp, 1, 1), np.float32)
    B = np.zeros((CHANNELS,), np.float32)
    # background
    W[0, i1, 0, 0] += -1.0
    W[0, i2, 0, 0] += -1.0
    B[0] = 1.0
    # intersection
    W[inter, i1, 0, 0] += 1.0
    W[inter, i2, 0, 0] += 1.0
    B[inter] = -1.0
    # v1 only
    W[v1, i1, 0, 0] += 1.0
    W[v1, i2, 0, 0] += -1.0
    # v2 only
    W[v2, i1, 0, 0] += -1.0
    W[v2, i2, 0, 0] += 1.0

    inits.append(oh.make_tensor("cw", DATA_TYPE, [CHANNELS, Cp, 1, 1], W.ravel().tolist()))
    inits.append(oh.make_tensor("cb", DATA_TYPE, [CHANNELS], B.tolist()))

    nodes = [
        oh.make_node("Slice", ["input", s_starts, s_ends, s_axes], ["s"]),
        oh.make_node("ReduceMax", ["s"], ["rp"], axes=[3], keepdims=1),
        oh.make_node("ReduceMax", ["s"], ["cp"], axes=[2], keepdims=1),
        oh.make_node("Max", ["rp", "cp"], ["cross"]),
        oh.make_node("Conv", ["cross", "cw", "cb"], ["pre"],
                     kernel_shape=[1, 1], pads=[0, 0, 0, 0]),
        oh.make_node("Clip", ["pre"], ["clipped"], min=0.0, max=1.0),
        oh.make_node("Pad", ["clipped"], ["output"], mode="constant",
                     pads=[0, 0, 0, 0, 0, 0, HEIGHT - N, WIDTH - N], value=0.0),
    ]

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "cross", [x], [y], inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# detection / entry                                                            #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim == 2 and b.ndim == 2 and a.size and b.size:
                out.append((a, b))
    return out


def candidates(ex):
    tt = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not tt:
        return []
    # fixed square NxN across every example (in == out == NxN)
    shapes = {a.shape for a, _ in allp} | {b.shape for _, b in allp}
    if len(shapes) != 1:
        return []
    (H0, W0), = shapes
    if H0 != W0:
        return []
    N = H0
    if not (1 <= N <= min(HEIGHT, WIDTH)):
        return []

    # derive marker colours v1,v2 and intersection colour `inter` from train
    inv = set()
    for a, _ in tt:
        inv |= set(np.unique(a).tolist())
    inv.discard(0)
    if len(inv) != 2:
        return []
    v1, v2 = sorted(inv)
    interset = set()
    for _, b in tt:
        interset |= set(np.unique(b).tolist())
    interset -= {0, v1, v2}
    if len(interset) != 1:
        return []
    inter = interset.pop()
    if not all(0 <= c < CHANNELS for c in (v1, v2, inter)):
        return []

    # verify the structural rule EXACTLY on everything available
    for a, b in allp:
        if a.shape != (N, N):
            return []
        if not np.array_equal(_cross(a, v1, v2, inter), b):
            return []

    try:
        model = _build(N, v1, v2, inter)
    except Exception:
        return []
    return [(f"cross_{N}_{v1}_{v2}_{inter}", model)]
