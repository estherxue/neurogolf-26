"""family_crk3_1 -- crack module for slice IDX=1 of the unsolved NeuroGolf tasks.

Each detected task gets its own structural detector (validated EXACTLY against the
provided train/test pairs in numpy) plus a static opset-10 ONNX builder.  All
intermediates are static-shape; data-dependent geometry uses computed index grids
+ ReduceMax/Mod/Abs/Less, never dynamic Resize/Pad.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
G = HEIGHT  # 30


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _check(m):
    onnx.checker.check_model(m, full_check=True)
    return m


def _pairs(ex):
    out = []
    for k in ("train", "test"):
        for p in ex.get(k, []):
            out.append((np.array(p["input"]), np.array(p["output"])))
    return out


def _slc(g, src, lo, hi, axis):
    """Slice src[axis] in [lo,hi)."""
    s = g.i64([lo]); e = g.i64([hi]); a = g.i64([axis])
    return g.nd("Slice", [src, s, e, a])


# =========================================================================== #
# TASK 332 -- alternate columns (from the right edge) recolor 5 -> 3          #
# =========================================================================== #
def _t332_rule(i):
    h, w = i.shape
    o = i.copy()
    cols = np.where(i.any(axis=0))[0]
    if len(cols) == 0:
        return o
    last = cols.max()
    for c in range(w):
        if (last - c) % 2 == 0:
            o[:, c][i[:, c] == 5] = 3
    return o


def _t332_detect(prs):
    for i, o in prs:
        if set(np.unique(i).tolist()) - {0, 5}:
            return False
        if not np.array_equal(_t332_rule(i), o):
            return False
    return True


def _t332_build():
    g = _G()
    J = g.f([1, 1, 1, G], list(range(G)))
    two = g.f([1, 1, 1, 1], [2.0])
    one = g.f([1, 1, 1, 1], [1.0])
    m5 = _slc(g, "input", 5, 6, 1)                         # [1,1,30,30]
    colmax = g.nd("ReduceMax", [m5], axes=[2], keepdims=1)  # [1,1,1,30]
    idxp = g.nd("Mul", [colmax, J])
    last = g.nd("ReduceMax", [idxp], axes=[3], keepdims=1)  # [1,1,1,1]
    lastpar = g.nd("Mod", [last, two], fmod=1)
    cpar = g.nd("Mod", [J, two], fmod=1)
    diff = g.nd("Sub", [cpar, lastpar])
    match = g.nd("Sub", [one, g.nd("Abs", [diff])])        # 1 where recolor
    inv = g.nd("Sub", [one, match])
    out5 = g.nd("Mul", [m5, inv])
    out3 = g.nd("Mul", [m5, match])
    A = _slc(g, "input", 0, 3, 1)     # ch0,1,2
    C = _slc(g, "input", 4, 5, 1)     # ch4
    E = _slc(g, "input", 6, 10, 1)    # ch6..9
    g.nd("Concat", [A, out3, C, out5, E], "output", axis=1)
    return _model(g)


# =========================================================================== #
# dispatch                                                                    #
# =========================================================================== #
_SOLVERS = [
    ("t332", _t332_detect, _t332_build),
]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    for name, detect, build in _SOLVERS:
        try:
            if detect(prs):
                m = _check(build())
                out.append((name, m))
        except Exception:
            pass
    return out
