"""family_rsc_0 — from-scratch minimal-cost rebuilds of low-scoring golf targets.

task124 / verify_53b68214  (bg=0): a mono/poly-colour motif that is periodic under a
    single translation vector v=(dr,dc).  Output is a fixed 10-row (x input-width W)
    canvas holding the union of shift(motif, k*v) for k=0..9.  The vector is the one
    maximising the self-overlap |fg ∩ shift(fg,v)| over dr in 1..5, dc in -10..9 among
    vectors whose overlap is colour-consistent; ties -> max dr*dc, then max |v|^2.

    ONNX: value/one-hot image; self-overlap and same-colour-overlap maps computed as
    two runtime-weight Conv autocorrelations (weight = the fg mask itself); validity =
    (overlap>0 & overlap==samecolour); an exact integer rank picks the argmax vector;
    the selected (dr,dc) drive a runtime translate (MatMul shift matrices) doubling
    union (v,2v,4v,8v -> k=0..15) which is clipped to the 10 x W region.  Fully static,
    opset-10, validated numpy==onnx and against the grader on train+test+arc-gen.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh
from collections import Counter

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64
H30 = 30


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

    def f1(self, v):
        return self.f([1, 1, 1, 1], [v])

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
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


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _eqm(g, a, b):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.f1(0.5))


# =========================================================================== #
# task124 — 53b68214                                                          #
# =========================================================================== #
def build_124():
    g = _G()
    rowidx = g.f([1, 1, H30, 1], list(range(H30)))
    colidx = g.f([1, 1, 1, H30], list(range(H30)))
    arange10 = g.f([1, 10, 1, 1], list(range(10)))
    one = g.f1(1.0)

    # value image + occupancy
    V = g.nd("Conv", ["input", arange10], kernel_shape=[1, 1])          # [1,1,30,30]
    occ = g.nd("ReduceMax", ["input"], axes=[1], keepdims=1)            # [1,1,30,30]

    # background colour = argmax channel sum
    sums = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)        # [1,10,1,1]
    maxsum = g.nd("ReduceMax", [sums], axes=[1], keepdims=1)
    bg_oh = _eqm(g, sums, maxsum)                                       # [1,10,1,1]
    bg_plane = g.nd("ReduceSum", [g.nd("Mul", ["input", bg_oh])], axes=[1], keepdims=1)
    bg_color = g.nd("ReduceSum", [g.nd("Mul", [arange10, bg_oh])], axes=[1], keepdims=1)

    fgmask = g.nd("Mul", [occ, g.nd("Sub", [one, bg_plane])])           # [1,1,30,30]
    FGOH = g.nd("Mul", ["input", fgmask])                              # [1,10,30,30]
    encV = g.nd("Mul", [g.nd("Add", [V, one]), fgmask])                # fg -> colour+1 ; gap 0

    # ---- self-overlap autocorrelation maps via runtime-weight Conv --------- #
    pads = [0, 0, 0, 10, 0, 0, 5, 9]        # top,left / bot,right  -> out [1,1,6,20]
    fg_pad = g.nd("Pad", [fgmask], mode="constant", value=0.0, pads=pads)
    O = g.nd("Conv", [fg_pad, fgmask], kernel_shape=[H30, H30])         # overlap map
    oh_pad = g.nd("Pad", [FGOH], mode="constant", value=0.0, pads=pads)
    S = g.nd("Conv", [oh_pad, FGOH], kernel_shape=[H30, H30])           # same-colour overlap

    # index grids over the [1,1,6,20] map:  vr=oi (0..5), vc=oj-10 (-10..9)
    vr = g.f([1, 1, 6, 1], list(range(6)))
    vc = g.f([1, 1, 1, 20], [j - 10 for j in range(20)])
    drdc = g.nd("Mul", [vr, vc])
    mag = g.nd("Add", [g.nd("Mul", [vr, vr]), g.nd("Mul", [vc, vc])])

    valid = g.nd("Mul", [_eqm(g, O, S), _gt(g, O, g.f1(0.5))])
    valid = g.nd("Mul", [valid, _gt(g, vr, g.f1(0.5))])                 # require dr>=1

    # exact integer rank:  O*12000 + (dr*dc+45)*128 + mag   (all < 2^24)
    rank = g.nd("Add", [
        g.nd("Add", [g.nd("Mul", [O, g.f1(12000.0)]),
                     g.nd("Mul", [g.nd("Add", [drdc, g.f1(45.0)]), g.f1(128.0)])]),
        mag])
    rank_eff = g.nd("Sub", [g.nd("Mul", [rank, valid]),
                            g.nd("Mul", [g.nd("Sub", [one, valid]), g.f1(1.0e6)])])
    maxrank = g.nd("ReduceMax", [rank_eff], axes=[2, 3], keepdims=1)
    sel = g.nd("Mul", [_eqm(g, rank_eff, maxrank), valid])             # one-hot [1,1,6,20]
    dr_sel = g.nd("ReduceSum", [g.nd("Mul", [sel, vr])], axes=[2, 3], keepdims=1)
    dc_sel = g.nd("ReduceSum", [g.nd("Mul", [sel, vc])], axes=[2, 3], keepdims=1)

    # ---- doubling union of shift(fg, k*v), k=0..15 ------------------------- #
    rowdiff = g.nd("Sub", [rowidx, colidx])     # [1,1,30,30]  value i-j
    coldiff = g.nd("Sub", [colidx, rowidx])     # value j-i
    P = encV
    for s in (1.0, 2.0, 4.0, 8.0):
        a = g.nd("Mul", [dr_sel, g.f1(s)])
        b = g.nd("Mul", [dc_sel, g.f1(s)])
        Trow = _eqm(g, rowdiff, a)
        Tcol = _eqm(g, coldiff, b)
        shifted = g.nd("MatMul", [g.nd("MatMul", [Trow, P]), Tcol])
        P = g.nd("Max", [P, shifted])

    # ---- clip to 10 x W region -------------------------------------------- #
    Wm1 = g.nd("ReduceMax", [g.nd("Mul", [occ, colidx])], axes=[2, 3], keepdims=1)
    region = g.nd("Mul", [_lt(g, rowidx, g.f1(9.5)),
                          _lt(g, colidx, g.nd("Add", [Wm1, g.f1(0.5)]))])
    Pc = g.nd("Mul", [P, region])
    present = _gt(g, Pc, g.f1(0.5))
    finalV = g.nd("Add", [g.nd("Mul", [present, g.nd("Sub", [Pc, one])]),
                          g.nd("Mul", [g.nd("Sub", [one, present]), bg_color])])
    OH = _eqm(g, finalV, arange10)                                    # [1,10,30,30]
    g.nd("Mul", [OH, region], "output")
    return _model(g, "rsc_124")


# =========================================================================== #
# numpy references                                                             #
# =========================================================================== #
def _ref124(a):
    a = np.array(a, int)
    H, W = a.shape
    bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
    fg = {(r, c): int(a[r, c]) for r in range(H) for c in range(W) if a[r, c] != bg}
    if not fg:
        return None
    idx = set(fg)
    valid = {}
    for dr in range(1, 6):
        for dc in range(-10, 10):
            sh = {(r + dr, c + dc): v for (r, c), v in fg.items()}
            inter = idx & set(sh)
            if not inter:
                continue
            if not all(fg[p] == sh[p] for p in inter):
                continue
            valid[(dr, dc)] = len(inter)
    if not valid:
        return None
    best_ov = max(valid.values())
    cands = [v for v, ov in valid.items() if ov == best_ov]
    dr, dc = max(cands, key=lambda t: (t[0] * t[1], t[0] * t[0] + t[1] * t[1]))
    out = np.full((10, W), bg, int)
    for k in range(10):
        for (r, c), v in fg.items():
            rr, cc = r + k * dr, c + k * dc
            if 0 <= rr < 10 and 0 <= cc < W:
                out[rr, cc] = v
    return out


# =========================================================================== #
# harness                                                                      #
# =========================================================================== #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _matches(prs, fn):
    if not prs:
        return False
    for a, b in prs:
        try:
            o = fn(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    out = []
    if _matches(prs, _ref124):
        try:
            out.append(("rsc_124", build_124()))
        except Exception:
            pass
    return out
