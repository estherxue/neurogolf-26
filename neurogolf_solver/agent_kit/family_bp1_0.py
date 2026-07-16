"""family_bp1_0 — memory-dominated task recompiles via fp16 working canvases.

The deployed out_blend4 net for task209 (8a004b2b, "magnify sprite into box") is a
2836-node graph whose intermediates are ALL fp32 [1,1,30,30] / [1,10,30,30]
canvases (~2.49M bytes of memory, 10.27 pts — by far the worst task in the
portfolio). Every value that flows through those canvases is a small integer
(colours 0..9, coordinates 0..30, area/count sums <=1000), which is represented
exactly in fp16 (integers <=2048 are exact). Recompiling the identical algorithm
with an fp16 working dtype halves the memory of every named intermediate with no
change in numerics.

The ONE place fp16 is unsafe is the 45-way candidate-selection KEY,
  key = mag*10000 + Or*100 + Oc   (sentinel 1e6 for invalid),
whose magnitudes (up to ~1e6) exceed fp16's exact-integer range. Those tensors
are all [1,1,1,1] scalars, so keeping just that selection block in fp32 costs a
handful of bytes while preserving the exact lexicographic tie-break. Everything
spatial (the big canvases) stays fp16.

Input value_info stays fp32 (the grader always feeds an fp32 one-hot); the graph
casts it to fp16 in its first node. Output is fp16 (free; grader compares >0).

Algorithm, numpy reference, and train+test/arc-gen gate are identical to the
deployed rb209 solver (this is a pure dtype recompile), so exactness is preserved
on every recoverable input and it matches the incumbent on the generator's
~0.5% inherently-ambiguous inputs.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

FIN = onnx.TensorProto.FLOAT       # input/output boundary dtype (grader feeds fp32)
F = onnx.TensorProto.FLOAT16       # working dtype
F32 = onnx.TensorProto.FLOAT       # selection-key dtype (needs exact large ints)
INT64 = onnx.TensorProto.INT64
H30 = 30


# --------------------------------------------------------------------------- #
# graph accumulator + helpers                                                  #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals, dt=F):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, dt, list(dims), [float(v) for v in np.asarray(vals, np.float64).ravel()]))
        return n

    def f1(self, v):
        return self.f([1, 1, 1, 1], [v])

    def f32(self, dims, vals):
        return self.f(dims, vals, dt=F32)

    def f321(self, v):
        return self.f([1, 1, 1, 1], [v], dt=F32)

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g, name):
    x = oh.make_tensor_value_info("input", FIN, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, name, [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


def _consts(g):
    g.rowidx = g.f([1, 1, H30, 1], list(range(H30)))
    g.colidx = g.f([1, 1, 1, H30], list(range(H30)))
    g.half = g.f1(0.5)
    g.one = g.f1(1.0)
    g.cbig = g.f1(1000.0)
    g.valvec = g.f([1, 10, 1, 1], list(range(10)))
    g.half32 = g.f321(0.5)


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _eqm(g, a, b):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


def _ge(g, a, b):
    return _gt(g, a, g.nd("Sub", [b, g.half]))


def _le(g, a, b):
    return _lt(g, a, g.nd("Add", [b, g.half]))


# fp32 selection helpers (small [1,1,1,1] scalars only) ---------------------- #
def _eqm32(g, a, b):
    d = g.nd("Abs", [g.nd("Sub", [a, b])])
    return g.nd("Cast", [g.nd("Less", [d, g.half32])], to=F32)


def _tof32(g, a):
    return g.nd("Cast", [a], to=F32)


def _value_img(g, x):
    w = g.f([1, 10, 1, 1], list(range(10)))
    return g.nd("Conv", [x, w], kernel_shape=[1, 1])


def _minmax(g, has, idx, axis):
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], axes=[axis], keepdims=1)
    inv = g.nd("Mul", [has, g.nd("Sub", [g.cbig, idx])])
    mn = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [inv], axes=[axis], keepdims=1)])
    return mn, mx


def _rowspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)
    return _minmax(g, has, g.rowidx, 2)


def _colspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)
    return _minmax(g, has, g.colidx, 3)


def _crop(g, src, r0, c0, h, w):
    Rrow = g.nd("Mul", [_eqm(g, g.colidx, g.nd("Add", [g.rowidx, r0])),
                        _lt(g, g.rowidx, g.nd("Sub", [h, g.half]))])
    Rcol = g.nd("Mul", [_eqm(g, g.rowidx, g.nd("Add", [g.colidx, c0])),
                        _lt(g, g.colidx, g.nd("Sub", [w, g.half]))])
    return g.nd("MatMul", [Rrow, g.nd("MatMul", [src, Rcol])])


def _place(g, src, Or, Oc):
    Prow = _eqm(g, g.rowidx, g.nd("Add", [g.colidx, Or]))
    Pcol = _eqm(g, g.colidx, g.nd("Add", [g.rowidx, Oc]))
    return g.nd("MatMul", [Prow, g.nd("MatMul", [src, Pcol])])


# =========================================================================== #
# ONNX builder                                                                 #
# =========================================================================== #
def build_209():
    g = _G()
    _consts(g)
    xf = g.nd("Cast", ["input"], to=F)                 # fp32 one-hot -> fp16
    V = _value_img(g, xf)                              # per-cell colour value

    # ---- yellow box bbox ----
    Ymask = _eqm(g, V, g.f1(4.0))
    br, mr = _rowspan(g, Ymask)
    bc, mc = _colspan(g, Ymask)
    tall = g.nd("Add", [g.nd("Sub", [mr, br]), g.one])
    wide = g.nd("Add", [g.nd("Sub", [mc, bc]), g.one])

    # ---- colour / inbox / sprite / shown masks ----
    col = g.nd("Mul", [_gt(g, V, g.half),
                       _gt(g, g.nd("Abs", [g.nd("Sub", [V, g.f1(4.0)])]), g.half)])
    inrow = g.nd("Mul", [_ge(g, g.rowidx, br), _le(g, g.rowidx, mr)])
    incol = g.nd("Mul", [_ge(g, g.colidx, bc), _le(g, g.colidx, mc)])
    inbox = g.nd("Mul", [inrow, incol])
    spr = g.nd("Mul", [col, g.nd("Sub", [g.one, inbox])])
    shown = g.nd("Mul", [col, inbox])

    # ---- sprite -> P (origin-anchored value grid), small canvas, magnified ----
    Vspr = g.nd("Mul", [V, spr])
    sr0, sr1 = _rowspan(g, spr)
    sc0, sc1 = _colspan(g, spr)
    tp = g.nd("Add", [g.nd("Sub", [sr1, sr0]), g.one])
    wp = g.nd("Add", [g.nd("Sub", [sc1, sc0]), g.one])
    Pfull = _crop(g, Vspr, sr0, sc0, tp, wp)
    Psmall = g.nd("Slice", [Pfull, g.i64([0, 0]), g.i64([4, 6]), g.i64([2, 3])])
    Tm = {}
    for mag in (2, 3, 4):
        sc = g.f32([4], [1, 1, mag, mag])   # Resize 'scales' must be fp32
        up = g.nd("Resize", [Psmall, sc], mode="nearest")
        Tm[mag] = g.nd("Pad", [up], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, H30 - 4 * mag, H30 - 6 * mag])

    # ---- shown blocks box-relative + (minR,minC) ----
    Vsh = g.nd("Mul", [V, shown])
    Sbox = _crop(g, Vsh, br, bc, tall, wide)
    Sm = _gt(g, Sbox, g.half)
    mR0, _ = _rowspan(g, Sm)
    mC0, _ = _colspan(g, Sm)

    # ---- 45-way candidate search (spatial fp16; key/selection fp32) ----
    keys, valids32, Or32s, Oc32s, magc = [], [], [], [], []
    for mag in (2, 3, 4):
        mh = g.nd("Mul", [tp, g.f1(mag)])
        mw = g.nd("Mul", [wp, g.f1(mag)])
        for rt in range(3):
            for cl in range(5):
                Or = g.nd("Sub", [mR0, g.f1(rt * mag)])
                Oc = g.nd("Sub", [mC0, g.f1(cl * mag)])
                Tpl = _place(g, Tm[mag], Or, Oc)
                diff = g.nd("Abs", [g.nd("Sub", [Sbox, Tpl])])
                mism = g.nd("ReduceSum", [g.nd("Mul", [Sm, _gt(g, diff, g.half)])],
                            axes=[2, 3], keepdims=1)
                fits = g.nd("Mul", [g.nd("Mul", [_ge(g, Or, g.one), _ge(g, Oc, g.one)]),
                                    g.nd("Mul", [_le(g, g.nd("Add", [Or, mh]), tall),
                                                 _le(g, g.nd("Add", [Oc, mw]), wide)])])
                valid = g.nd("Mul", [fits, _lt(g, mism, g.half)])   # fp16 0/1

                # --- fp32 key arithmetic (exact large ints) ---
                Or_32 = _tof32(g, Or)
                Oc_32 = _tof32(g, Oc)
                valid32 = _tof32(g, valid)
                key = g.nd("Add", [g.nd("Add", [g.f321(mag * 10000.0),
                                                g.nd("Mul", [Or_32, g.f321(100.0)])]), Oc_32])
                keyf = g.nd("Add", [g.nd("Mul", [valid32, key]),
                                    g.nd("Mul", [g.nd("Sub", [g.f321(1.0), valid32]),
                                                 g.f321(1e6)])])
                keys.append(keyf)
                valids32.append(valid32)
                Or32s.append(Or_32)
                Oc32s.append(Oc_32)
                magc.append(mag)

    minkey = g.nd("Min", keys)
    Or_sel32 = Oc_sel32 = mag_sel32 = None
    for i in range(len(keys)):
        ind = g.nd("Mul", [valids32[i], _eqm32(g, keys[i], minkey)])
        tOr = g.nd("Mul", [ind, Or32s[i]])
        tOc = g.nd("Mul", [ind, Oc32s[i]])
        tMg = g.nd("Mul", [ind, g.f321(magc[i])])
        Or_sel32 = tOr if Or_sel32 is None else g.nd("Add", [Or_sel32, tOr])
        Oc_sel32 = tOc if Oc_sel32 is None else g.nd("Add", [Oc_sel32, tOc])
        mag_sel32 = tMg if mag_sel32 is None else g.nd("Add", [mag_sel32, tMg])

    Or_sel = g.nd("Cast", [Or_sel32], to=F)            # back to fp16 (exact int)
    Oc_sel = g.nd("Cast", [Oc_sel32], to=F)
    mag_sel = g.nd("Cast", [mag_sel32], to=F)

    # ---- place the winning magnified creature ----
    Tfinal = None
    for mag in (2, 3, 4):
        pl = g.nd("Mul", [_eqm(g, mag_sel, g.f1(mag)), _place(g, Tm[mag], Or_sel, Oc_sel)])
        Tfinal = pl if Tfinal is None else g.nd("Add", [Tfinal, pl])

    # ---- corners + compose + one-hot ----
    cr = g.nd("Add", [_eqm(g, g.rowidx, g.f1(0.0)),
                      _eqm(g, g.rowidx, g.nd("Sub", [tall, g.one]))])
    cc = g.nd("Add", [_eqm(g, g.colidx, g.f1(0.0)),
                      _eqm(g, g.colidx, g.nd("Sub", [wide, g.one]))])
    corners = g.nd("Mul", [g.nd("Mul", [cr, cc]), g.f1(4.0)])
    OUTv = g.nd("Add", [Tfinal, g.nd("Mul", [corners, _lt(g, Tfinal, g.half)])])

    ingrid = g.nd("Mul", [_lt(g, g.rowidx, g.nd("Sub", [tall, g.half])),
                          _lt(g, g.colidx, g.nd("Sub", [wide, g.half]))])
    OH = _eqm(g, OUTv, g.valvec)
    g.nd("Mul", [OH, ingrid], "output")
    return _model(g, "bp1_209")


# =========================================================================== #
# numpy reference (identical to rb209)                                         #
# =========================================================================== #
def _magnify(P, mag):
    return np.repeat(np.repeat(P, mag, 0), mag, 1)


def _parse(a):
    a = np.array(a, int)
    ys, xs = np.where(a == 4)
    if len(ys) != 4:
        return None
    brow, bcol, mrow, mcol = ys.min(), xs.min(), ys.max(), xs.max()
    tall, wide = mrow - brow + 1, mcol - bcol + 1
    if tall < 2 or wide < 2:
        return None
    col = (a != 0) & (a != 4)
    inbox = np.zeros_like(col)
    inbox[brow:mrow + 1, bcol:mcol + 1] = True
    spr = col & ~inbox
    shb = col & inbox
    if not spr.any() or not shb.any():
        return None
    sy, sx = np.where(spr)
    sr0, sc0 = sy.min(), sx.min()
    tp, wp = sy.max() - sr0 + 1, sx.max() - sc0 + 1
    P = np.zeros((tp, wp), int)
    for r, c in zip(sy, sx):
        P[r - sr0, c - sc0] = a[r, c]
    shy, shx = np.where(shb)
    shcells = [(int(r - brow), int(c - bcol), int(a[r, c])) for r, c in zip(shy, shx)]
    minR = min(x[0] for x in shcells)
    minC = min(x[1] for x in shcells)
    return P, tall, wide, shcells, minR, minC


def _valid_place(P, tall, wide, shcells, mag, Or, Oc):
    M = _magnify(P, mag)
    mh, mw = M.shape
    if Or < 1 or Oc < 1 or Or + mh > tall or Oc + mw > wide:
        return False
    for (R, C, v) in shcells:
        i, j = R - Or, C - Oc
        if not (0 <= i < mh and 0 <= j < mw and M[i, j] == v):
            return False
    return True


def _build(P, tall, wide, mag, Or, Oc):
    M = _magnify(P, mag)
    mh, mw = M.shape
    o = np.zeros((tall, wide), int)
    for r, c in [(0, 0), (0, wide - 1), (tall - 1, 0), (tall - 1, wide - 1)]:
        o[r, c] = 4
    o[Or:Or + mh, Oc:Oc + mw] = np.where(M > 0, M, o[Or:Or + mh, Oc:Oc + mw])
    return o


def _ref(a):
    p = _parse(a)
    if p is None:
        return None
    P, tall, wide, shcells, minR, minC = p
    tp, wp = P.shape
    if tp > 3 or wp > 5:
        return None
    for mag in (2, 3, 4):
        best = None
        for rt in range(3):
            for cl in range(5):
                Or = minR - rt * mag
                Oc = minC - cl * mag
                if _valid_place(P, tall, wide, shcells, mag, Or, Oc):
                    if best is None or (Or, Oc) < best:
                        best = (Or, Oc)
        if best is not None:
            return _build(P, tall, wide, mag, best[0], best[1])
    return None


# =========================================================================== #
# detection / candidates                                                       #
# =========================================================================== #
def _pairs(ex):
    out = []
    for s in ("train", "test"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _matches(prs):
    if not prs:
        return False
    for a, b in prs:
        try:
            o = _ref(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not _matches(prs):
        return []
    try:
        return [("bp1_209", build_209())]
    except Exception:
        return []
