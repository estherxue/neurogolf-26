"""family_golf4_2: cheap EXACT golf solvers for a few targets in slice [2::6].

Implemented families
--------------------
* cone (task 348, crk6_3_cone): a vertical line of color 7 whose BOTTOM cell is
  the apex of an upward-opening cone.  The cone is drawn with colours that
  alternate 7/8 by horizontal distance parity from the line column.  We build
  the triangular cone purely with single-channel [1,1,30,30] convolutions that
  propagate a reachability mask upward (a fixed cone-shaped kernel applied a
  constant number of times -> O(1) intermediates, no [900,900] matrices), then
  colour by column parity and pack the 3 single-channel masks into the 10-channel
  output with one 1x1 Conv.

Each family auto-detects by running a numpy reference over the train+test pairs;
the model is only proposed when the reference reproduces every pair exactly, so
wrong guesses cost nothing (the grader still validates arc-gen for EXACTness).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------
# cone (task 348)
# --------------------------------------------------------------------------

_CONE_H = 11          # cone kernel height; 3 applications reach 3*(H-1)=30 rows
_CONE_STEPS = 3


def _cone_kernel(h):
    w = 2 * (h - 1) + 1
    K = np.zeros((h, w), np.float32)
    for dr in range(h):
        for kj in range(w):
            if abs(kj - (h - 1)) <= dr:
                K[dr, kj] = 1.0
    return K


def _cone_solve_np(a):
    """numpy reference for the cone transform."""
    a = np.asarray(a)
    Hh, Ww = a.shape
    seed = (a == 7).astype(np.float32)
    if seed.sum() == 0:
        return None
    cols = np.where(seed.sum(axis=0) > 0)[0]
    if len(cols) != 1:                 # must be a single vertical line
        return None
    K = _cone_kernel(_CONE_H)
    h, w = K.shape
    pl = w // 2
    cur = seed.copy()
    for _ in range(_CONE_STEPS):
        P = np.pad(cur, ((0, h - 1), (pl, pl)))
        nxt = np.zeros_like(cur)
        for r in range(Hh):
            for c in range(Ww):
                nxt[r, c] = (P[r:r + h, c:c + w] * K).sum()
        cur = nxt
    cone = cur > 0
    c0 = cols[0]
    out = np.zeros((Hh, Ww), int)
    for r in range(Hh):
        for c in range(Ww):
            if cone[r, c]:
                out[r, c] = 7 if ((c - c0) % 2 == 0) else 8
    return out


def _build_cone():
    h = _CONE_H
    w = 2 * (h - 1) + 1
    pl = w // 2
    K = _cone_kernel(h).reshape(1, 1, h, w)

    inits = []
    nodes = []

    def cst(name, arr, dtype=DATA_TYPE):
        arr = np.asarray(arr)
        t = oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist())
        inits.append(t)
        return name

    # seed = channel 7
    cst("s7s", np.array([7], np.int64), INT64)
    cst("s7e", np.array([8], np.int64), INT64)
    cst("s7a", np.array([1], np.int64), INT64)
    nodes.append(oh.make_node("Slice", ["input", "s7s", "s7e", "s7a"], ["seed"]))

    # cone propagation: K applied _CONE_STEPS times (no clip; non-neg weights
    # keep support == reachable set)
    cst("Kc", K)
    pads = [0, pl, h - 1, pl]
    prev = "seed"
    for i in range(_CONE_STEPS):
        nm = f"c{i}"
        nodes.append(oh.make_node("Conv", [prev, "Kc"], [nm],
                                  kernel_shape=[h, w], pads=pads))
        prev = nm
    cone = prev  # counts > 0 exactly on the cone

    # real-cell mask (1 at any real cell, 0 at padding)
    nodes.append(oh.make_node("ReduceSum", ["input"], ["real"], axes=[1], keepdims=1))

    # column parity of the line column c0
    nodes.append(oh.make_node("ReduceMax", ["seed"], ["colmask"], axes=[2], keepdims=1))
    evencol = np.array([1.0 if j % 2 == 0 else 0.0 for j in range(30)], np.float32).reshape(1, 1, 1, 30)
    oddcol = 1.0 - evencol
    cst("evencol", evencol)
    cst("oddcol", oddcol)
    cst("one", np.array([1.0], np.float32))
    nodes.append(oh.make_node("Mul", ["colmask", "evencol"], ["cm_e"]))
    nodes.append(oh.make_node("ReduceSum", ["cm_e"], ["Ec0"], axes=[3], keepdims=1))
    nodes.append(oh.make_node("Sub", ["one", "Ec0"], ["Eo"]))
    nodes.append(oh.make_node("Mul", ["evencol", "Ec0"], ["p7a"]))
    nodes.append(oh.make_node("Mul", ["oddcol", "Eo"], ["p7b"]))
    nodes.append(oh.make_node("Add", ["p7a", "p7b"], ["P7"]))      # [1,1,1,30]
    nodes.append(oh.make_node("Sub", ["one", "P7"], ["P8"]))       # [1,1,1,30]

    # restrict cone to the real grid, colour by parity
    nodes.append(oh.make_node("Mul", [cone, "real"], ["Mr"]))
    nodes.append(oh.make_node("Sub", ["real", "Mr"], ["out0"]))
    nodes.append(oh.make_node("Mul", ["Mr", "P7"], ["M7"]))
    nodes.append(oh.make_node("Mul", ["Mr", "P8"], ["M8"]))

    nodes.append(oh.make_node("Concat", ["out0", "M7", "M8"], ["feat"], axis=1))

    # pack 3 single-channel masks -> 10-channel output via 1x1 conv
    Wout = np.zeros((10, 3, 1, 1), np.float32)
    Wout[0, 0, 0, 0] = 1.0   # channel 0  <- out0
    Wout[7, 1, 0, 0] = 1.0   # channel 7  <- M7
    Wout[8, 2, 0, 0] = 1.0   # channel 8  <- M8
    cst("Wout", Wout)
    nodes.append(oh.make_node("Conv", ["feat", "Wout"], ["output"],
                              kernel_shape=[1, 1], pads=[0, 0, 0, 0]))

    return _model(nodes, inits)


def _try_cone(prs):
    for a, b in prs:
        if a.shape[0] > 30 or a.shape[1] > 30:
            continue
        pred = _cone_solve_np(a)
        if pred is None or pred.shape != b.shape or not (pred == b).all():
            return None
    return _build_cone()


# --------------------------------------------------------------------------
# hollow (task 98, hollow_box): erode interior of filled shapes (a fg cell whose
# 4 orthogonal neighbours are all fg becomes background).  Single-channel convs;
# the 10-channel output is assembled with one Concat so no full [1,10,30,30]
# intermediate is materialised.
# --------------------------------------------------------------------------

def _hollow_solve_np(a):
    a = np.asarray(a)
    fg = (a != 0).astype(int)
    nb = np.zeros_like(fg)
    nb[1:, :] += fg[:-1, :]
    nb[:-1, :] += fg[1:, :]
    nb[:, 1:] += fg[:, :-1]
    nb[:, :-1] += fg[:, 1:]
    interior = (fg == 1) & (nb == 4)
    out = a.copy()
    out[interior] = 0
    return out


def _build_hollow():
    inits = []
    nodes = []

    def cst(name, arr, dtype=DATA_TYPE):
        arr = np.asarray(arr)
        t = oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist())
        inits.append(t)
        return name

    cst("h1s", np.array([1], np.int64), INT64)
    cst("h10e", np.array([10], np.int64), INT64)
    cst("hca", np.array([1], np.int64), INT64)
    nodes.append(oh.make_node("Slice", ["input", "h1s", "h10e", "hca"], ["in19"]))
    nodes.append(oh.make_node("ReduceSum", ["in19"], ["fg"], axes=[1], keepdims=1))

    plus5 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float32).reshape(1, 1, 3, 3)
    cst("plus5", plus5)
    nodes.append(oh.make_node("Conv", ["fg", "plus5"], ["s"],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    cst("thr", np.array([4.5], np.float32))
    cst("one", np.array([1.0], np.float32))
    nodes.append(oh.make_node("Less", ["s", "thr"], ["lt"]))
    nodes.append(oh.make_node("Cast", ["lt"], ["keep"], to=onnx.TensorProto.FLOAT))
    nodes.append(oh.make_node("Sub", ["one", "keep"], ["interior"]))
    nodes.append(oh.make_node("Mul", ["in19", "keep"], ["masked19"]))

    cst("h0s", np.array([0], np.int64), INT64)
    cst("h1e", np.array([1], np.int64), INT64)
    nodes.append(oh.make_node("Slice", ["input", "h0s", "h1e", "hca"], ["ch0"]))
    nodes.append(oh.make_node("Add", ["ch0", "interior"], ["out0"]))
    nodes.append(oh.make_node("Concat", ["out0", "masked19"], ["output"], axis=1))
    return _model(nodes, inits)


def _try_hollow(prs):
    saw = False
    for a, b in prs:
        if a.shape[0] > 30 or a.shape[1] > 30:
            continue
        pred = _hollow_solve_np(a)
        if pred.shape != b.shape or not (pred == b).all():
            return None
        if (pred != a).any():
            saw = True
    return _build_hollow() if saw else None


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    m = _try_cone(prs)
    if m is not None:
        out.append(("cone", m))
    m = _try_hollow(prs)
    if m is not None:
        out.append(("hollow", m))
    return out
