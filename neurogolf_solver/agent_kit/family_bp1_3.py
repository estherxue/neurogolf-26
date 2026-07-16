"""family_bp1_3 — memory-dominated rebuilds for tasks 173/145/149/138.

Each solver is a hand-built ONNX (onnx.helper) implementing the TRUE minimal rule,
aiming far cheaper than the memory-heavy incumbents in out_blend4. Every family
yields only when its pure-numpy mirror is bit-exact on train+test.

task149 (verify_6773b310): a hollywood_squares 3x3-of-3x3 board (fixed 11x11),
    cyan(8) gridlines, pink(6) dots, 1 or 2 dots per mini-cell.  Output is a 3x3
    grid: blue(1) where the mini-cell holds 2 dots, black(0) where it holds 1.
    ONNX: one stride-4 Conv sums pink per mini-cell -> two comparisons (==1, ==2)
    give the black/blue channels -> Concat+Pad writes the free output.  Fixed
    input size makes this a tiny, near-zero-memory graph.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE


class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def c(self, dtype, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, dtype, list(dims),
                                          np.asarray(vals).ravel().tolist()))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g, name):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, name, [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


# --------------------------------------------------------------------------- #
# task149 — 6773b310                                                           #
# --------------------------------------------------------------------------- #
def build_149():
    g = _G()
    # weight: pick channel 6 (pink), 3x3 window of ones; stride 4 = per mini-cell
    W = np.zeros((1, 10, 3, 3), np.float32)
    W[0, 6, :, :] = 1.0
    w = g.c(F, [1, 10, 3, 3], W)
    conv = g.nd("Conv", ["input", w], strides=[4, 4], kernel_shape=[3, 3])  # [1,1,7,7]

    c05 = g.c(F, [1, 1, 1, 1], [0.5])
    c15 = g.c(F, [1, 1, 1, 1], [1.5])
    blue = g.nd("Greater", [conv, c15])        # count >= 2  -> channel 1
    geq1 = g.nd("Greater", [conv, c05])        # count >= 1  (all mini-cells)
    nblue = g.nd("Not", [blue])
    black = g.nd("And", [geq1, nblue])         # count == 1  -> channel 0

    cat = g.nd("Concat", [black, blue], axis=1)          # bool [1,2,7,7]
    catf = g.nd("Cast", [cat], to=F)                     # float [1,2,7,7]
    g.nd("Pad", [catf], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 8, 23, 23])                # [1,10,30,30]
    return _model(g, "bp1_149")


def _mirror_149(a):
    a = np.asarray(a, int)
    if a.shape != (11, 11):
        return None
    out = np.zeros((3, 3), int)
    for r in range(3):
        for c in range(3):
            cnt = int((a[r * 4:r * 4 + 3, c * 4:c * 4 + 3] == 6).sum())
            if cnt < 1 or cnt > 2:
                return None
            out[r, c] = 1 if cnt == 2 else 0
    return out


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return []
            if max(a.shape) > 30 or max(b.shape) > 30:
                return []
            out.append((a, b))
    return out


def _matches(prs, fn):
    for a, b in prs:
        try:
            o = fn(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    if _matches(prs, _mirror_149):
        try:
            yield ("bp1_6773b310", build_149())
        except Exception:
            pass
