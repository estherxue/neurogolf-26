"""family_pgolf_0: cheap generalizing solvers for a subset of golf targets.

Fires only on tasks whose true rule it can encode grid-agnostically and cheaply.
Each candidate is validated for EXACTNESS on train+test+arc-gen by the harness.
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
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _slice(inp, out, starts, ends, axes, name):
    ini = [
        oh.make_tensor(f"{name}_s", INT64, [len(starts)], list(starts)),
        oh.make_tensor(f"{name}_e", INT64, [len(ends)], list(ends)),
        oh.make_tensor(f"{name}_a", INT64, [len(axes)], list(axes)),
    ]
    node = oh.make_node("Slice", [inp, f"{name}_s", f"{name}_e", f"{name}_a"], [out])
    return node, ini


# --------------------------------------------------------------------------
# t389: recolor color-5 cells to the (unique) other present color, rest -> 0.
# --------------------------------------------------------------------------

def _detect_recolor5(prs):
    for a, b in prs:
        cols = set(int(c) for c in np.unique(a)) - {0, 5}
        if len(cols) != 1:
            return False
        x = cols.pop()
        if not np.array_equal(np.where(a == 5, x, 0), b):
            return False
        if a.shape != b.shape:
            return False
    return True


def _build_recolor5():
    nodes, inits = [], []
    # m5 = channel 5 of input (the mask of 5-cells)
    n, i = _slice("input", "m5", [5], [6], [1], "m5")
    nodes.append(n); inits += i
    # realmask = 1 where any channel is on (real cell), 0 on padding
    nodes.append(oh.make_node("ReduceMax", ["input"], ["realmask"], axes=[1], keepdims=1))
    # present[c] = 1 if color c appears anywhere
    nodes.append(oh.make_node("ReduceMax", ["input"], ["present"], axes=[2, 3], keepdims=1))
    # present9 = present over channels 1..9
    n, i = _slice("present", "present9", [1], [10], [1], "p9")
    nodes.append(n); inits += i
    # w9 = present9 with the color-5 slot (index 4 within 1..9) zeroed
    # channels 1..9 -> slice indices 0..8; color 5 lives at index 4 and must be dropped
    k9 = oh.make_tensor("k9", DATA_TYPE, [1, 9, 1, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1])
    inits.append(k9)
    nodes.append(oh.make_node("Mul", ["present9", "k9"], ["w9"]))
    # colored9 = m5 broadcast into channels 1..9, kept only in the other-color slot
    nodes.append(oh.make_node("Mul", ["m5", "w9"], ["colored9"]))
    # ch0 = real background of output = real cells that are not 5-cells
    nodes.append(oh.make_node("Sub", ["realmask", "m5"], ["ch0"]))
    nodes.append(oh.make_node("Concat", ["ch0", "colored9"], ["output"], axis=1))
    return _model(nodes, inits)


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples["train"] + examples["test"]]
    if not prs:
        return
    if _detect_recolor5(prs):
        yield "recolor5", _build_recolor5()
