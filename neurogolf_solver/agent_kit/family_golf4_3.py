"""family_golf4_3 — cheaper EXACT solvers for selected low-scoring golf targets.

Targets (slice [3::6] of golf_targets.json). The integrator auto-picks the cheapest
exact model per task, so we only need to BEAT the current points by producing a
graph with lower cost (cost = params + intermediate_memory; points = 25 - ln cost).

Key technique used here: a DYNAMIC anti-diagonal MatMul.  For a grid anchored
top-left with a (variable) occupied height H, the row-reversal matrix
    M[i,j] = 1  iff  i + j == H - 1
maps row j -> row H-1-j, i.e. an EXACT vertical flip that stays anchored at the
origin for any H.  We build M at runtime from the data (no [900,900] matrix, no
per-size constants):  M = Relu(1 - |K - S|) where K[i,j] = i+j+1 (a [30,30]
constant) and S = number of occupied rows.  Then output = MatMul(M, input).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            out.append((np.array(e["input"]), np.array(e["output"])))
    return out


# --------------------------------------------------------------------------
# vertical flip within the (variable-height) content, anchored top-left.
# --------------------------------------------------------------------------
def _vflip_model():
    # K[i,j] = i + j + 1   (constant [1,1,30,30])
    K = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            K[0, 0, i, j] = i + j + 1
    Kt = oh.make_tensor("K", DATA_TYPE, [1, 1, HEIGHT, WIDTH], K.ravel().tolist())
    one = oh.make_tensor("one", DATA_TYPE, [1, 1, 1, 1], [1.0])

    nodes = [
        # rowCount[r] = #real cells in row r  ([1,1,30,1]); >0 for real rows only.
        oh.make_node("ReduceSum", ["input"], ["rowCount"], axes=[1, 3], keepdims=1),
        # rowOcc = clip(rowCount,0,1)  -> 1 for occupied rows
        oh.make_node("Clip", ["rowCount"], ["rowOcc"], min=0.0, max=1.0),
        # S = H = number of occupied rows  ([1,1,1,1])
        oh.make_node("ReduceSum", ["rowOcc"], ["S"], axes=[2], keepdims=1),
        # M = Relu(1 - |K - S|)  -> anti-diagonal reversal restricted to rows < H
        oh.make_node("Sub", ["K", "S"], ["D"]),
        oh.make_node("Abs", ["D"], ["AD"]),
        oh.make_node("Sub", ["one", "AD"], ["om"]),
        oh.make_node("Relu", ["om"], ["M"]),
        # output[:, :, i, :] = sum_j M[i,j] input[:, :, j, :]
        oh.make_node("MatMul", ["M", "input"], ["output"]),
    ]
    return _model(nodes, [Kt, one])


def _is_vflip(P):
    if not P:
        return False
    for a, b in P:
        if a.shape != b.shape:
            return False
        if not np.array_equal(b, a[::-1, :]):
            return False
    return True


# --------------------------------------------------------------------------
# column gravity (objects fall straight down), monochromatic columns.
#   output channel k>0  = (col has color k) AND (row in bottom `count_c` rows)
#   output channel 0    = real cell AND not filled by foreground
# All built from per-column counts + occupied H/W -- no scatter, no big matrix.
# --------------------------------------------------------------------------
def _gravity_model():
    rowIdx = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    colIdx = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    RI = oh.make_tensor("RI", DATA_TYPE, [1, 1, HEIGHT, 1], rowIdx.ravel().tolist())
    CI = oh.make_tensor("CI", DATA_TYPE, [1, 1, 1, WIDTH], colIdx.ravel().tolist())
    one = oh.make_tensor("one", DATA_TYPE, [1, 1, 1, 1], [1.0])
    s_st = oh.make_tensor("s_st", INT64, [1], [1])
    s_en = oh.make_tensor("s_en", INT64, [1], [10])
    s_ax = oh.make_tensor("s_ax", INT64, [1], [1])

    nodes = [
        # per-column, per-channel counts  [1,10,1,30]
        oh.make_node("ReduceSum", ["input"], ["cnt"], axes=[2], keepdims=1),
        # foreground channels 1..9, and "this column uses color k" mask
        oh.make_node("Slice", ["cnt", "s_st", "s_en", "s_ax"], ["cnt19"]),
        oh.make_node("Clip", ["cnt19"], ["mask19"], min=0.0, max=1.0),
        # number of non-bg cells per column  [1,1,1,30]
        oh.make_node("ReduceSum", ["cnt19"], ["nonbg"], axes=[1], keepdims=1),
        # occupied height H  [1,1,1,1]
        oh.make_node("ReduceSum", ["input"], ["rowCount"], axes=[1, 3], keepdims=1),
        oh.make_node("Clip", ["rowCount"], ["rowOcc"], min=0.0, max=1.0),
        oh.make_node("ReduceSum", ["rowOcc"], ["H"], axes=[2], keepdims=1),
        # occupied width W  [1,1,1,1]
        oh.make_node("ReduceSum", ["input"], ["colCount"], axes=[1, 2], keepdims=1),
        oh.make_node("Clip", ["colCount"], ["colOcc"], min=0.0, max=1.0),
        oh.make_node("ReduceSum", ["colOcc"], ["W"], axes=[3], keepdims=1),
        # fill threshold: bottom rows  thr = H - nonbg ; A = (r >= thr)
        oh.make_node("Sub", ["H", "nonbg"], ["thr"]),
        oh.make_node("Sub", ["thr", "one"], ["thrm1"]),       # thr-1
        oh.make_node("Sub", ["RI", "thrm1"], ["subA"]),       # r - thr + 1
        oh.make_node("Clip", ["subA"], ["A"], min=0.0, max=1.0),
        # C = (r >= H)  [1,1,30,1]
        oh.make_node("Sub", ["H", "one"], ["Hm1"]),
        oh.make_node("Sub", ["RI", "Hm1"], ["subC"]),
        oh.make_node("Clip", ["subC"], ["C"], min=0.0, max=1.0),
        # fgAll = (thr<=r<H)
        oh.make_node("Sub", ["A", "C"], ["fgAll"]),
        # foreground output channels 1..9
        oh.make_node("Mul", ["mask19", "fgAll"], ["prod19"]),
        # background channel 0 = realcell AND not foreground
        oh.make_node("Sub", ["one", "C"], ["rLT"]),           # r < H
        oh.make_node("Sub", ["W", "one"], ["Wm1"]),
        oh.make_node("Sub", ["CI", "Wm1"], ["subCcol"]),
        oh.make_node("Clip", ["subCcol"], ["cGE"], min=0.0, max=1.0),
        oh.make_node("Sub", ["one", "cGE"], ["cLT"]),         # c < W
        oh.make_node("Sub", ["one", "fgAll"], ["notfg"]),
        oh.make_node("Mul", ["notfg", "cLT"], ["tmp0"]),
        oh.make_node("Mul", ["tmp0", "rLT"], ["ch0"]),
        oh.make_node("Concat", ["ch0", "prod19"], ["output"], axis=1),
    ]
    return _model(nodes, [RI, CI, one, s_st, s_en, s_ax])


def _is_gravity_down(P):
    if not P:
        return False
    for a, b in P:
        if a.shape != b.shape:
            return False
        H, W = a.shape
        for c in range(W):
            col = [a[r, c] for r in range(H) if a[r, c] != 0]
            if len(set(col)) > 1:           # column must be monochromatic
                return False
            want = np.zeros(H, dtype=a.dtype)
            for i, v in enumerate(col[::-1]):
                want[H - 1 - i] = v
            if not np.array_equal(want, b[:, c]):
                return False
    return True


def candidates(examples):
    P = _pairs(examples)
    cands = []
    if _is_vflip(P):
        cands.append(("vflip_matmul", _vflip_model()))
    if _is_gravity_down(P):
        cands.append(("gravity_down", _gravity_model()))
    return cands
