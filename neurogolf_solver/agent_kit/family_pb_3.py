"""family_pb_3 -- minimal-cost ONNX recompilations for a small set of tasks.

Each candidate is gated on a numpy reference (the task's true verifier) matching
EXACTLY on train+test+arc-gen, so a model only fires for the task it targets.

Targets (task -> hash -> rule):
  265 a8d7556c : fill every 2x2 all-color-0 block with color 2 (fixed 18x18)
"""
from __future__ import annotations

import sys, os
import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rearc"))
import verifiers as _V  # noqa: E402

INT64 = TP.INT64
UINT8 = TP.UINT8
BOOL = TP.BOOL
FLOAT = TP.FLOAT


def _mk(nodes, inits, opset=13):
    x = oh.make_tensor_value_info("input", FLOAT, [1, 10, 30, 30])
    y = oh.make_tensor_value_info("output", FLOAT, [1, 10, 30, 30])
    g = oh.make_graph(nodes, "g", [x], [y], inits)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)])
    m.ir_version = 9
    return m


def _pairs(ex, splits=("train", "test", "arc-gen")):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            out.append((np.array(e["input"]), np.array(e["output"])))
    return out


def _ref_matches(ex, fn):
    prs = _pairs(ex)
    if not prs:
        return False
    for a, b in prs:
        try:
            r = fn(tuple(map(tuple, a.tolist())))
            r = np.array(r)
        except Exception:
            return False
        if r.shape != b.shape or not np.array_equal(r, b):
            return False
    return True


# --------------------------------------------------------------------------- #
# task265 : 2x2 all-color-0 blocks -> color 2   (fixed 18x18 grids)
# --------------------------------------------------------------------------- #
def _model_265():
    n = []
    inits = []
    # slice channel-0 plane over the 18x18 content region -> [1,1,18,18] fp32
    inits.append(oh.make_tensor("s0", INT64, [4], [0, 0, 0, 0]))
    inits.append(oh.make_tensor("e0", INT64, [4], [1, 1, 18, 18]))
    n.append(oh.make_node("Slice", ["input", "s0", "e0"], ["z"]))
    # notz = (z < 0.5) : True where cell is NOT color-0
    inits.append(oh.make_tensor("half", FLOAT, [1], [0.5]))
    n.append(oh.make_node("Less", ["z", "half"], ["notz_b"]))
    n.append(oh.make_node("Cast", ["notz_b"], ["notz"], to=UINT8))
    # pool: 1 if any not-color-0 in the 2x2 block
    n.append(oh.make_node("MaxPool", ["notz"], ["pool"], kernel_shape=[2, 2], strides=[1, 1]))
    # B = (pool == 0) : the 2x2 block is all color-0
    inits.append(oh.make_tensor("z0", UINT8, [1], [0]))
    n.append(oh.make_node("Equal", ["pool", "z0"], ["B_b"]))
    n.append(oh.make_node("Cast", ["B_b"], ["B"], to=UINT8))
    # F = dilate B over 2x2 footprint -> covered cells   [1,1,18,18]
    n.append(oh.make_node("MaxPool", ["B"], ["F"], kernel_shape=[2, 2], strides=[1, 1],
                          pads=[1, 1, 1, 1]))
    n.append(oh.make_node("Cast", ["F"], ["F_b"], to=BOOL))
    # pad the fill-mask up to 30x30 (bool) for the terminal Where condition
    inits.append(oh.make_tensor("pp", INT64, [8], [0, 0, 0, 0, 0, 0, 12, 12]))
    n.append(oh.make_node("Pad", ["F_b", "pp"], ["F30"], mode="constant"))
    # output = where(fill, color-2 one-hot, input)
    red = [0.0] * 10
    red[2] = 1.0
    inits.append(oh.make_tensor("red", FLOAT, [1, 10, 1, 1], red))
    n.append(oh.make_node("Where", ["F30", "red", "input"], ["output"]))
    return _mk(n, inits)


def candidates(ex):
    out = []
    prs = _pairs(ex)
    shapes = set(a.shape for a, _ in prs)
    # task265: fixed 18x18, colors subset of {0, 5}, true rule matches
    if shapes == {(18, 18)} and _ref_matches(ex, _V.verify_a8d7556c):
        try:
            out.append(("pb3_265", _model_265()))
        except Exception:
            pass
    return out
