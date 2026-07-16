"""family_golf8_3 — aggressive-golf reformulations of low-scoring solvers.

Strategy: for FIXED-size tasks, Slice the one-hot input down to the real HxW
region so every intermediate is ~9x smaller ([1,10,10,10]=4000B vs 36000B),
run the transform, then Pad back to [1,10,30,30] (padding=0 == all-channels<=0,
which is exactly what the grader wants for padding cells). Keep intermediates
single-channel [1,1,h,w] wherever possible.

Each candidate is gated by re-running the numpy rule on train+test; only emit if
it reproduces every pair exactly (anti-overfit + correctness).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _onehot(a):
    o = np.zeros((10,) + a.shape, np.float32)
    for c in range(10):
        o[c] = (a == c)
    return o


# ----------------------------------------------------------------------------
# Task 40 — "two opposite full-line poles; recolor interior markers to the
# nearer pole colour (by row-half if horizontal poles, col-half if vertical)".
# ----------------------------------------------------------------------------

def _t40_rule(a):
    H, W = a.shape
    x = _onehot(a)
    bg = x[0]
    ha = 1.0 - bg[0, :].max()          # row0 fully nonzero -> horizontal poles
    va = 1.0 - bg[:, 0].max()          # col0 fully nonzero -> vertical poles
    mid, midr = W // 2, H // 2
    Hf = np.zeros_like(x); Hf[:, :midr, :] = x[:, 0:1, :]; Hf[:, midr:, :] = x[:, H - 1:H, :]
    Vf = np.zeros_like(x); Vf[:, :, :mid] = x[:, :, 0:1]; Vf[:, :, mid:] = x[:, :, W - 1:W]
    field = Hf * ha + Vf * va
    same = (x * field).sum(0)
    M = (1.0 - bg) * (1.0 - same)
    out = x + M[None] * (field - x)
    res = np.zeros((H, W), int)
    for c in range(1, 10):
        res[out[c] > 0.5] = c
    return res


def _t40_ok(pairs):
    for a, b in pairs:
        if a.shape != (10, 10) or b.shape != (10, 10):
            return False
        p = _t40_rule(a)
        if p.shape != b.shape or not (p == b).all():
            return False
    return True


def _t40_model():
    n = 10
    inits = []

    def const(name, arr, dtype=INT64):
        arr = np.asarray(arr)
        t = oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist())
        inits.append(t)
        return name

    def fconst(name, arr):
        arr = np.asarray(arr, np.float32)
        t = oh.make_tensor(name, DATA_TYPE, list(arr.shape), arr.ravel().tolist())
        inits.append(t)
        return name

    nodes = []
    # x = input[:, :, 0:10, 0:10]
    const("sl_s", [0, 0]); const("sl_e", [n, n]); const("sl_a", [2, 3])
    nodes.append(oh.make_node("Slice", ["input", "sl_s", "sl_e", "sl_a"], ["x"]))
    # bg = x[:, 0:1]
    const("bg_s", [0]); const("bg_e", [1]); const("bg_a", [1])
    nodes.append(oh.make_node("Slice", ["x", "bg_s", "bg_e", "bg_a"], ["bg"]))
    fconst("one", np.ones((1, 1, 1, 1), np.float32))
    # ha
    const("r0_s", [0]); const("r0_e", [1]); const("r0_a", [2])
    nodes.append(oh.make_node("Slice", ["bg", "r0_s", "r0_e", "r0_a"], ["row0bg"]))
    nodes.append(oh.make_node("ReduceMax", ["row0bg"], ["hmax"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Sub", ["one", "hmax"], ["ha"]))
    # va
    const("c0_s", [0]); const("c0_e", [1]); const("c0_a", [3])
    nodes.append(oh.make_node("Slice", ["bg", "c0_s", "c0_e", "c0_a"], ["col0bg"]))
    nodes.append(oh.make_node("ReduceMax", ["col0bg"], ["vmax"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Sub", ["one", "vmax"], ["va"]))
    # Hfield
    const("top_s", [0]); const("top_e", [1]); const("top_a", [2])
    nodes.append(oh.make_node("Slice", ["x", "top_s", "top_e", "top_a"], ["top"]))
    const("bot_s", [n - 1]); const("bot_e", [n]); const("bot_a", [2])
    nodes.append(oh.make_node("Slice", ["x", "bot_s", "bot_e", "bot_a"], ["bot"]))
    const("rep_h", [1, 1, n // 2, 1])
    nodes.append(oh.make_node("Tile", ["top", "rep_h"], ["topT"]))
    nodes.append(oh.make_node("Tile", ["bot", "rep_h"], ["botT"]))
    nodes.append(oh.make_node("Concat", ["topT", "botT"], ["Hf"], axis=2))
    # Vfield
    const("lf_s", [0]); const("lf_e", [1]); const("lf_a", [3])
    nodes.append(oh.make_node("Slice", ["x", "lf_s", "lf_e", "lf_a"], ["left"]))
    const("rt_s", [n - 1]); const("rt_e", [n]); const("rt_a", [3])
    nodes.append(oh.make_node("Slice", ["x", "rt_s", "rt_e", "rt_a"], ["right"]))
    const("rep_v", [1, 1, 1, n // 2])
    nodes.append(oh.make_node("Tile", ["left", "rep_v"], ["leftT"]))
    nodes.append(oh.make_node("Tile", ["right", "rep_v"], ["rightT"]))
    nodes.append(oh.make_node("Concat", ["leftT", "rightT"], ["Vf"], axis=3))
    # field = Hf*ha + Vf*va
    nodes.append(oh.make_node("Mul", ["Hf", "ha"], ["Hfg"]))
    nodes.append(oh.make_node("Mul", ["Vf", "va"], ["Vfg"]))
    nodes.append(oh.make_node("Add", ["Hfg", "Vfg"], ["field"]))
    # M = (1-bg)*(1-sum(x*field))
    nodes.append(oh.make_node("Mul", ["x", "field"], ["prod"]))
    nodes.append(oh.make_node("ReduceSum", ["prod"], ["same"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Sub", ["one", "same"], ["diff"]))
    nodes.append(oh.make_node("Sub", ["one", "bg"], ["nz"]))
    nodes.append(oh.make_node("Mul", ["nz", "diff"], ["M"]))
    # res = x + M*(field - x)
    nodes.append(oh.make_node("Sub", ["field", "x"], ["fmx"]))
    nodes.append(oh.make_node("Mul", ["M", "fmx"], ["Mfmx"]))
    nodes.append(oh.make_node("Add", ["x", "Mfmx"], ["res"]))
    # pad back to 30x30
    nodes.append(oh.make_node("Pad", ["res"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, 30 - n, 30 - n]))
    return _model(nodes, inits)


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    out = []
    if prs and _t40_ok(prs):
        out.append(("golf8_t40", _t40_model()))
    return out
