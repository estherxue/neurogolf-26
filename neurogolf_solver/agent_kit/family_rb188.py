"""family_rb188 — task188 (verify 7b7f7511): "de-duplicate a doubled tile".

Generator: a height x width tile (both dims in [2,4]) of colours drawn from a
3-4 colour palette is duplicated either VERTICALLY (stacked twice -> input shape
(2h, w)) or HORIZONTALLY (side by side twice -> input shape (h, 2w)).  The output
is the single tile = the top-left h x w block of the input.

Recovering the split from the input alone:
  * vertical  <=> R in {4,6,8}, C in {2,3,4}, and top R/2 rows == bottom R/2 rows
  * horizontal otherwise (a genuine horizontal sample has C in {4,6,8}, R in {2,3,4})
The two interpretations can only coexist dimensionally at shape 4x4, and only when
the grid is 2x2-periodic.  Such inputs are produced by the generator with BOTH vert
values (identical bytes, different output 2x4 vs 4x2) at EXACTLY 50/50 probability,
so they are information-theoretically unsolvable; we tie-break to vertical (matches
every known-fail sample).  Residual error on fresh generator samples is therefore
~0.044% (the horizontal-truth half of the 4x4-2x2-periodic inputs) and cannot be
reduced by any deterministic function.

Since the grader compares the full [1,10,30,30] one-hot, the model is simply:
  output = input masked to the top-left H_out x W_out region, where
  (H_out, W_out) = (R/2, C) if vertical else (R, C/2).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
H30 = 30


# --------------------------------------------------------------------------- #
# numpy reference (the TRUE generator rule + tie-break)                         #
# --------------------------------------------------------------------------- #
def _ref(a):
    a = np.asarray(a, int)
    if a.ndim != 2:
        return None
    R, C = a.shape
    if R > 30 or C > 30:
        return None
    Rcond = R in (4, 6, 8)
    Ccond = C in (2, 3, 4)
    half_eq = False
    if Rcond and Ccond:
        h = R // 2
        half_eq = np.array_equal(a[:h], a[h:2 * h])
    if Rcond and Ccond and half_eq:          # vertical (tie-break wins here too)
        return a[:R // 2, :C]
    return a[:R, :C // 2]                     # horizontal


# --------------------------------------------------------------------------- #
# graph accumulator                                                            #
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
        self.inits.append(oh.make_tensor(
            n, F, list(dims), [float(v) for v in np.asarray(vals, np.float64).ravel()]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _eqm(g, a, b, half):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), half)


def _model(g, name):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, name, [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


def build_model():
    g = _G()
    rowidx = g.f([1, 1, H30, 1], list(range(H30)))   # i along axis 2
    colidx = g.f([1, 1, 1, H30], list(range(H30)))   # j along axis 3
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    four = g.f([1, 1, 1, 1], [4.0])
    six = g.f([1, 1, 1, 1], [6.0])
    eight = g.f([1, 1, 1, 1], [8.0])
    three = g.f([1, 1, 1, 1], [3.0])

    # grid extent R, C (grid is a dense rectangle at the top-left) -------------
    occ = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    rowhas = g.nd("ReduceMax", [occ], axes=[3], keepdims=1)         # [1,1,30,1]
    colhas = g.nd("ReduceMax", [occ], axes=[2], keepdims=1)         # [1,1,1,30]
    R = g.nd("ReduceSum", [rowhas], axes=[2], keepdims=1)           # [1,1,1,1]
    C = g.nd("ReduceSum", [colhas], axes=[3], keepdims=1)           # [1,1,1,1]
    halfR = g.nd("Mul", [R, half])
    halfC = g.nd("Mul", [C, half])

    # Rcond = R in {4,6,8} ; Ccond = C in {2,3,4} -----------------------------
    Rcond = g.nd("Add", [g.nd("Add", [_eqm(g, R, four, half),
                                      _eqm(g, R, six, half)]),
                         _eqm(g, R, eight, half)])
    Ccond = g.nd("Add", [g.nd("Add", [_eqm(g, C, two, half),
                                      _eqm(g, C, three, half)]),
                         _eqm(g, C, four, half)])

    # half_eq = (top R/2 rows == bottom R/2 rows) over cols < C ----------------
    # shift matrix Sh[i,j] = (j == i + R/2) -> shifted row i = input row i+R/2
    tgt = g.nd("Add", [rowidx, halfR])                             # [1,1,30,1]
    Sh = _eqm(g, colidx, tgt, half)                               # [1,1,30,30]
    shifted = g.nd("MatMul", [Sh, "input"])                       # [1,10,30,30]
    reg = g.nd("Mul", [_lt(g, rowidx, halfR), _lt(g, colidx, C)])  # [1,1,30,30]
    diff = g.nd("Mul", [g.nd("Abs", [g.nd("Sub", ["input", shifted])]), reg])
    d = g.nd("ReduceSum", [diff], axes=[1, 2, 3], keepdims=1)     # [1,1,1,1]
    half_eq = _lt(g, d, half)

    use_vert = g.nd("Mul", [g.nd("Mul", [Rcond, Ccond]), half_eq])  # [1,1,1,1] 0/1
    not_vert = g.nd("Sub", [one, use_vert])

    H_out = g.nd("Add", [g.nd("Mul", [use_vert, halfR]),
                         g.nd("Mul", [not_vert, R])])
    W_out = g.nd("Add", [g.nd("Mul", [use_vert, C]),
                         g.nd("Mul", [not_vert, halfC])])

    maskH = _lt(g, rowidx, H_out)                                 # [1,1,30,1]
    maskW = _lt(g, colidx, W_out)                                 # [1,1,1,30]
    mask = g.nd("Mul", [maskH, maskW])                            # [1,1,30,30]
    g.nd("Mul", ["input", mask], "output")                        # [1,10,30,30]
    return _model(g, "rb188")


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


def _matches(prs):
    for a, b in prs:
        o = _ref(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(examples):
    prs = _pairs(examples)
    if not prs or not _matches(prs):
        return
    try:
        yield ("family_rb188", build_model())
    except Exception:
        pass
