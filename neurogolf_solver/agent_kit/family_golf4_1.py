"""family_golf4_1 — cheaper EXACT solvers for selected low-scoring golf targets.

Slice [1::6] of golf_targets.json. The integrator keeps the cheapest exact model
per task (cost = params + intermediate_memory; points = 25 - ln cost), so we only
need to produce a correct graph that is CHEAPER than the current best.

Each solver re-derives the rule from train+test (+arc-gen) in numpy and only fires
when the rule holds EXACTLY on every available pair.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(examples, secs=("train", "test")):
    out = []
    for s in secs:
        for e in examples.get(s, []):
            out.append((np.array(e["input"], int), np.array(e["output"], int)))
    return out


def _allpairs(examples):
    return _pairs(examples, ("train", "test", "arc-gen"))


# ==========================================================================
# Task 108 — odd-lattice subsample then 4x block upscale.
#   out[r,c] = in[(r//4)*2 + 1, (c//4)*2 + 1]   (10x10 -> 20x20)
# Implementation: Slice odd rows/cols (start 1, step 2) -> 5x5, ConvTranspose
# stride-4 ones kernel (groups=10) -> 20x20, Pad -> 30x30.
# ==========================================================================
def _is_108(P):
    for a, b in P:
        H, W = a.shape
        OH, OW = b.shape
        if OH != 2 * H or OW != 2 * W:
            return False
        pred = np.zeros_like(b)
        for r in range(OH):
            for c in range(OW):
                sr = (r // 4) * 2 + 1
                sc = (c // 4) * 2 + 1
                if sr < H and sc < W:
                    pred[r, c] = a[sr, sc]
        if not np.array_equal(pred, b):
            return False
    return True


def _model_108():
    # Slice rows/cols 1,3,5,7,9  -> [1,10,5,5]
    s = oh.make_tensor("s_st", INT64, [2], [1, 1])
    e = oh.make_tensor("s_en", INT64, [2], [11, 11])
    a = oh.make_tensor("s_ax", INT64, [2], [2, 3])
    st = oh.make_tensor("s_sp", INT64, [2], [2, 2])
    sl = oh.make_node("Slice", ["input", "s_st", "s_en", "s_ax", "s_sp"], ["sub"])
    # grouped ConvTranspose, ones 4x4 kernel, stride 4 -> each cell becomes 4x4 block
    w = oh.make_tensor("W", DATA_TYPE, [10, 1, 4, 4], [1.0] * (10 * 16))
    ct = oh.make_node("ConvTranspose", ["sub", "W"], ["big"],
                      strides=[4, 4], group=10, kernel_shape=[4, 4], pads=[0, 0, 0, 0])
    pad = oh.make_node("Pad", ["big"], ["output"], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, HEIGHT - 20, WIDTH - 20])
    return _model([sl, ct, pad], [s, e, a, st, w])


# ==========================================================================
# Task 215 — vertical period-3 tiling, masked to the (variable) grid region.
#   The input has one 3-row band; the output repeats it with period 3 over the
#   whole grid.  output[r,c] = band_row[(r-r0)%3][c]  within the grid.
# Implementation entirely from data:
#   inGrid = sum_ch X                (1 inside grid, 0 in padding)
#   bandRow[r'] = 1 if row r' holds an object cell (non-background)
#   C[r,r'] = 1 iff (r-r')%3==0       (constant)
#   T = C * bandRow ;  tiled = T @ X ;  output = tiled * inGrid
# ==========================================================================
def _sim_period(a, period):
    H, W = a.shape
    X = np.zeros((10, 30, 30))
    for r in range(H):
        for c in range(W):
            X[a[r, c], r, c] = 1.0
    inGrid = X.sum(0)
    objMask = inGrid - X[0]
    bandCol = np.clip(objMask.sum(1), 0, 1)
    R = np.arange(30)
    C = ((R[:, None] - R[None, :]) % period == 0).astype(float)
    T = C * bandCol[None, :]
    tiled = np.einsum('rs,csx->crx', T, X)
    out = tiled * inGrid[None, :, :]
    return out > 0


def _oh(g):
    H, W = g.shape
    X = np.zeros((10, 30, 30), bool)
    for r in range(H):
        for c in range(W):
            X[g[r, c], r, c] = True
    return X


def _is_period(P, period):
    for a, b in P:
        if a.shape != b.shape:
            return False
        if not np.array_equal(_sim_period(a, period), _oh(b)):
            return False
    return True


def _model_period(period):
    R = np.arange(30)
    C = ((R[:, None] - R[None, :]) % period == 0).astype(np.float32)
    Ct = oh.make_tensor("C", DATA_TYPE, [1, 1, 30, 30], C.ravel().tolist())
    c0s = oh.make_tensor("c0s", INT64, [1], [0])
    c0e = oh.make_tensor("c0e", INT64, [1], [1])
    c0a = oh.make_tensor("c0a", INT64, [1], [1])
    nodes = [
        oh.make_node("ReduceSum", ["input"], ["inGrid"], axes=[1], keepdims=1),
        oh.make_node("Slice", ["input", "c0s", "c0e", "c0a"], ["ch0"]),
        oh.make_node("Sub", ["inGrid", "ch0"], ["objMask"]),
        oh.make_node("ReduceSum", ["objMask"], ["bandRow"], axes=[3], keepdims=1),
        oh.make_node("Clip", ["bandRow"], ["bandRowC"], min=0.0, max=1.0),
        oh.make_node("Transpose", ["bandRowC"], ["bandRowT"], perm=[0, 1, 3, 2]),
        oh.make_node("Mul", ["C", "bandRowT"], ["T"]),
        oh.make_node("MatMul", ["T", "input"], ["tiled"]),
        oh.make_node("Mul", ["tiled", "inGrid"], ["output"]),
    ]
    return _model(nodes, [Ct, c0s, c0e, c0a])


def candidates(examples):
    P = _pairs(examples)
    cands = []
    if not P:
        return cands
    if _is_108(P):
        cands.append(("blockup108", _model_108()))
    for period in (2, 3, 4):
        if all(a.shape == b.shape for a, b in P) and _is_period(P, period):
            cands.append((f"vtile{period}", _model_period(period)))
            break
    return cands
