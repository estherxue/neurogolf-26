"""family_golf5_2: cheaper EXACT golf solvers for targets in slice [2::6].

Implemented families
--------------------
* lattice_complete (task 314, linecomplete_3): the 8x8 grid is a fixed 3x3
  lattice of 2x2 cells (separator rows/cols at indices 2 and 5).  The base fill
  is colour 1; "special" colours (2..9) appear in some cells.  Whenever a
  horizontal or vertical line of three aligned cells (same local 2x2 position,
  spaced 3 apart) contains >=2 cells of the same special colour, the whole line
  of three is filled with that colour.  Lone specials are kept.

  This is expressed with two depth-wise convolutions (one horizontal, one
  vertical) whose kernels tap every 3 pixels -> each special channel's line-sum;
  threshold >=2, OR the two directions, keep originals, and pack the output with
  a single Concat (the [1,10,30,30] output tensor itself is free).

Each family auto-detects with a numpy reference over train+test; the model is
proposed only when the reference reproduces every pair exactly, so wrong guesses
cost nothing (the grader still validates arc-gen for EXACTness).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------
# lattice_complete (task 314)
# --------------------------------------------------------------------------

_TAPS = [-6, -3, 0, 3, 6]   # offsets every 3, covering the 3 lattice cells


def _lattice_solve_np(a):
    a = np.asarray(a)
    H, W = a.shape
    if (H, W) != (8, 8):
        return None
    X = np.stack([(a == c).astype(float) for c in range(10)])   # [10,8,8]
    Xs = X[2:10]                                                 # [8,8,8]

    def line_sum(m, axis):
        out = np.zeros_like(m, dtype=float)
        for off in _TAPS:
            sh = np.zeros_like(m, dtype=float)
            if axis == 1:   # vertical (rows)
                if off > 0:
                    sh[:, off:, :] = m[:, :H - off, :]
                elif off < 0:
                    sh[:, :off, :] = m[:, -off:, :]
                else:
                    sh = m.copy()
            else:           # horizontal (cols)
                if off > 0:
                    sh[:, :, off:] = m[:, :, :W - off]
                elif off < 0:
                    sh[:, :, :off] = m[:, :, -off:]
                else:
                    sh = m.copy()
            out += sh
        return out

    LH = line_sum(Xs, 2)
    LV = line_sum(Xs, 1)
    L = np.maximum(LH, LV)
    fill = (L >= 2).astype(float)
    Sp = np.maximum(Xs, fill)
    hs = Sp.sum(axis=0)
    out = np.zeros((10, H, W))
    out[0] = X[0]
    out[1] = X[1] * (1.0 - hs)
    out[2:10] = Sp
    return np.argmax(out, axis=0)


def _build_lattice():
    inits = []
    nodes = []

    def cst(name, arr, dtype=FLOAT):
        arr = np.asarray(arr)
        inits.append(oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist()))
        return name

    # slice the 8 special channels (colours 2..9)
    cst("c2", np.array([2], np.int64), INT64)
    cst("c10", np.array([10], np.int64), INT64)
    cst("cax", np.array([1], np.int64), INT64)
    nodes.append(oh.make_node("Slice", ["input", "c2", "c10", "cax"], ["Xs"]))

    # depth-wise horizontal / vertical line-sum kernels (taps every 3)
    kh = np.zeros((8, 1, 1, 13), np.float32)
    kv = np.zeros((8, 1, 13, 1), np.float32)
    for off in _TAPS:
        kh[:, 0, 0, 6 + off] = 1.0
        kv[:, 0, 6 + off, 0] = 1.0
    cst("kh", kh)
    cst("kv", kv)
    nodes.append(oh.make_node("Conv", ["Xs", "kh"], ["LH"],
                              kernel_shape=[1, 13], pads=[0, 6, 0, 6], group=8))
    nodes.append(oh.make_node("Conv", ["Xs", "kv"], ["LV"],
                              kernel_shape=[13, 1], pads=[6, 0, 6, 0], group=8))
    nodes.append(oh.make_node("Max", ["LH", "LV"], ["L"]))

    cst("thr", np.array([1.5], np.float32))
    nodes.append(oh.make_node("Greater", ["L", "thr"], ["G"]))
    nodes.append(oh.make_node("Cast", ["G"], ["F"], to=FLOAT))
    nodes.append(oh.make_node("Max", ["Xs", "F"], ["Sp"]))           # special channels

    nodes.append(oh.make_node("ReduceSum", ["Sp"], ["hs"], axes=[1], keepdims=1))

    cst("c0", np.array([0], np.int64), INT64)
    cst("c1", np.array([1], np.int64), INT64)
    nodes.append(oh.make_node("Slice", ["input", "c0", "c1", "cax"], ["X0"]))
    nodes.append(oh.make_node("Slice", ["input", "c1", "c2", "cax"], ["X1"]))

    cst("one", np.array([1.0], np.float32))
    nodes.append(oh.make_node("Sub", ["one", "hs"], ["om"]))
    nodes.append(oh.make_node("Mul", ["X1", "om"], ["new1"]))

    nodes.append(oh.make_node("Concat", ["X0", "new1", "Sp"], ["output"], axis=1))
    return _model(nodes, inits)


def _try_lattice(prs):
    saw = False
    for a, b in prs:
        if a.shape != (8, 8):
            return None
        pred = _lattice_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
        if (pred != a).any():
            saw = True
    return _build_lattice() if saw else None


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    m = _try_lattice(prs)
    if m is not None:
        out.append(("lattice_complete", m))
    return out
