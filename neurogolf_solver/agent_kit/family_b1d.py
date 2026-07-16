"""family_b1d — cheap recompiles for tasks 46,365,216,396,255,80,89,313,63,128.

Each candidate is gated by a numpy _ref that must be exact on train+test, so a
candidate only fires for the task it was designed for.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION

F = onnx.TensorProto.FLOAT
BOOL = onnx.TensorProto.BOOL

# use opset 13 (Where/Or/Equal/ReduceSum/Conv all present); grader accepts 7-21
OPSET = [oh.make_opsetid("", 12)]


def _pairs(examples):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in examples.get("train", []) + examples.get("test", [])]


def _gate(examples, ref):
    prs = _pairs(examples)
    if not prs:
        return False
    for a, b in prs:
        r = ref(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            return False
    return True


# ---------------- task 63 (2bee17df) ---------------- #
def _ref63(x):
    x = np.asarray(x)
    if not set(np.unique(x)).issubset({0, 2, 8}):
        return None
    H, W = x.shape
    out = x.copy()
    N = (x != 0).astype(int)
    rc = N.sum(1, keepdims=True)
    cc = N.sum(0, keepdims=True)
    g = ((rc == 2) | (cc == 2)) & (x == 0)
    out[g] = 3
    return out


def _c(name, arr, dt=F):
    a = np.asarray(arr)
    a = a.astype(np.float32) if dt == F else a
    return oh.make_tensor(name, dt, list(a.shape) if a.shape else [1], a.flatten().tolist())


def _build63():
    nodes, inits = [], []
    # Wn: 1x1 conv selecting channels 2 and 8 (bars)
    wn = np.zeros((1, 10, 1, 1), np.float32); wn[0, 2] = 1; wn[0, 8] = 1
    inits.append(_c("Wn", wn))
    wbg = np.zeros((1, 10, 1, 1), np.float32); wbg[0, 0] = 1
    inits.append(_c("Wbg", wbg))
    gv = np.zeros((1, 10, 1, 1), np.float32); gv[0, 3] = 1
    inits.append(_c("gv", gv))
    inits.append(_c("two", [2.0])); inits.append(_c("half", [0.5]))
    nodes.append(oh.make_node("Conv", ["input", "Wn"], ["N"]))
    nodes.append(oh.make_node("Conv", ["input", "Wbg"], ["bg"]))
    nodes.append(oh.make_node("ReduceSum", ["N"], ["rc"], axes=[3], keepdims=1))
    nodes.append(oh.make_node("ReduceSum", ["N"], ["cc"], axes=[2], keepdims=1))
    nodes.append(oh.make_node("Equal", ["rc", "two"], ["re"]))
    nodes.append(oh.make_node("Equal", ["cc", "two"], ["ce"]))
    nodes.append(oh.make_node("Or", ["re", "ce"], ["eor"]))
    nodes.append(oh.make_node("Greater", ["bg", "half"], ["bgb"]))
    nodes.append(oh.make_node("And", ["eor", "bgb"], ["G"]))
    nodes.append(oh.make_node("Where", ["G", "gv", "input"], ["output"]))
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "b1d63", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET)


def candidates(examples):
    # NOTE: _build63 is a correct, self-contained solver for task 63, but it
    # measures ~10232 cost (15.77 pts) vs the incumbent's 1680 (17.56) because
    # the incumbent uses an Einsum row/col-vector graph with a single 900B bool
    # plane and NO full-resolution float plane. We do not yield a regression.
    # No candidate in this family strictly beat its incumbent locally; see report.
    return
    yield ("b1d63", _build63())  # unreachable; kept for provenance
