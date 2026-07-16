"""family_crk9_6 — cracked-hard tasks (slice U[6::7]).

Rules implemented:
  * marker_frames (task 328): single-pixel colour markers (each a distinct colour, in
    corners). Each cell takes the colour of the Manhattan-nearest marker (ties -> bg) and
    is filled iff the Chebyshev distance to that marker is even (concentric square frames
    at even distance). Restricted to the HxW grid region.
"""
from __future__ import annotations
import numpy as np
import onnx
from onnx import helper as oh

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ----------------------------------------------------------------------------
# numpy mirror for marker_frames
# ----------------------------------------------------------------------------
def _mf_solve(a):
    H, W = a.shape
    ys, xs = np.where(a != 0)
    markers = [(int(y), int(x), int(a[y, x])) for y, x in zip(ys, xs)]
    out = np.zeros_like(a)
    if not markers:
        return out
    R = np.arange(H)[:, None]
    C = np.arange(W)[None, :]
    chebs, manhs, cols = [], [], []
    for (y, x, col) in markers:
        dr = np.abs(R - y)
        dc = np.abs(C - x)
        chebs.append(np.maximum(dr, dc))
        manhs.append(dr + dc)
        cols.append(col)
    Ch = np.stack(chebs, 0)
    Mn = np.stack(manhs, 0)
    mmin = Mn.min(0)
    ismin = (Mn == mmin[None]).astype(int)
    cnt = ismin.sum(0)
    for k, (y, x, col) in enumerate(markers):
        sel = (ismin[k] == 1) & (cnt == 1) & (Ch[k] % 2 == 0)
        out[sel] = col
    return out


def _mf_detect(prs):
    if not prs:
        return False
    for a, b in prs:
        if a.shape != b.shape:
            return False
        # markers must be single-cell, distinct colours
        ys, xs = np.where(a != 0)
        cols = [int(a[y, x]) for y, x in zip(ys, xs)]
        if len(cols) == 0 or len(cols) != len(set(cols)):
            return False
        if not (_mf_solve(a) == b).all():
            return False
    return True


def _const(name, arr, dtype=F):
    arr = np.asarray(arr)
    t = oh.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist())
    return oh.make_node("Constant", [], [name], value=t), name


def _mf_build():
    nodes = []
    inits = []

    # coordinate ramps [1,1,30,30]
    Rg = np.arange(HEIGHT, dtype=np.float32)[None, None, :, None] * np.ones((1, 1, 1, WIDTH), np.float32)
    Cg = np.ones((1, 1, HEIGHT, 1), np.float32) * np.arange(WIDTH, dtype=np.float32)[None, None, None, :]
    inits.append(oh.make_tensor("Rgrid", F, [1, 1, HEIGHT, WIDTH], Rg.flatten().tolist()))
    inits.append(oh.make_tensor("Cgrid", F, [1, 1, HEIGHT, WIDTH], Cg.flatten().tolist()))
    inits.append(oh.make_tensor("BIG", F, [1, 1, 1, 1], [1e6]))
    inits.append(oh.make_tensor("HALF", F, [1, 1, 1, 1], [0.5]))
    inits.append(oh.make_tensor("ONE", F, [1, 1, 1, 1], [1.0]))
    inits.append(oh.make_tensor("TWO", F, [1, 1, 1, 1], [2.0]))
    inits.append(oh.make_tensor("ONEHALF", F, [1, 1, 1, 1], [1.5]))
    inits.append(oh.make_tensor("sl1", INT64, [1], [1]))
    inits.append(oh.make_tensor("sl10", INT64, [1], [10]))
    inits.append(oh.make_tensor("slax", INT64, [1], [1]))

    N = oh.make_node
    # grid mask: 1 inside HxW grid, 0 padding
    nodes.append(N("ReduceSum", ["input"], ["gmask"], axes=[1], keepdims=1))
    # markers = channels 1..9
    nodes.append(N("Slice", ["input", "sl1", "sl10", "slax"], ["inp9"]))
    nodes.append(N("ReduceSum", ["inp9"], ["present"], axes=[2, 3], keepdims=1))   # [1,9,1,1]
    nodes.append(N("Mul", ["inp9", "Rgrid"], ["rmul"]))
    nodes.append(N("ReduceSum", ["rmul"], ["rsum"], axes=[2, 3], keepdims=1))       # [1,9,1,1]
    nodes.append(N("Mul", ["inp9", "Cgrid"], ["cmul"]))
    nodes.append(N("ReduceSum", ["cmul"], ["csum"], axes=[2, 3], keepdims=1))
    # distances [1,9,30,30]
    nodes.append(N("Sub", ["Rgrid", "rsum"], ["dR"]))
    nodes.append(N("Sub", ["Cgrid", "csum"], ["dC"]))
    nodes.append(N("Abs", ["dR"], ["aR"]))
    nodes.append(N("Abs", ["dC"], ["aC"]))
    nodes.append(N("Max", ["aR", "aC"], ["cheb"]))
    nodes.append(N("Add", ["aR", "aC"], ["manh"]))
    # penalty for absent colours: (1-present)*BIG  [1,9,1,1]
    nodes.append(N("Sub", ["ONE", "present"], ["absent"]))
    nodes.append(N("Mul", ["absent", "BIG"], ["pen"]))
    nodes.append(N("Add", ["manh", "pen"], ["manjadj"]))
    nodes.append(N("ReduceMin", ["manjadj"], ["mmin"], axes=[1], keepdims=1))        # [1,1,30,30]
    nodes.append(N("Add", ["mmin", "HALF"], ["mmin_h"]))
    nodes.append(N("Less", ["manjadj", "mmin_h"], ["isminb"]))
    nodes.append(N("Cast", ["isminb"], ["ismin"], to=F))
    nodes.append(N("ReduceSum", ["ismin"], ["cnt"], axes=[1], keepdims=1))          # [1,1,30,30]
    nodes.append(N("Less", ["cnt", "ONEHALF"], ["uniqb"]))
    nodes.append(N("Cast", ["uniqb"], ["uniq"], to=F))
    # evenness of cheb
    nodes.append(N("Mod", ["cheb", "TWO"], ["modv"], fmod=1))
    nodes.append(N("Less", ["modv", "HALF"], ["evenb"]))
    nodes.append(N("Cast", ["evenb"], ["evenm"], to=F))
    # fill = ismin * evenm * uniq * gmask   (broadcast uniq,gmask over channels)
    nodes.append(N("Mul", ["ismin", "evenm"], ["f1"]))
    nodes.append(N("Mul", ["f1", "uniq"], ["f2"]))
    nodes.append(N("Mul", ["f2", "gmask"], ["fill9"]))                              # [1,9,30,30]
    # channel 0 background = gmask*(1-anyfill)
    nodes.append(N("ReduceSum", ["fill9"], ["anyfill"], axes=[1], keepdims=1))
    nodes.append(N("Sub", ["ONE", "anyfill"], ["nofill"]))
    nodes.append(N("Mul", ["gmask", "nofill"], ["ch0"]))
    nodes.append(N("Concat", ["ch0", "fill9"], ["output"], axis=1))
    return _model(nodes, inits)


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    out = []
    if _mf_detect(prs):
        try:
            out.append(("marker_frames", _mf_build()))
        except Exception:
            pass
    return out
