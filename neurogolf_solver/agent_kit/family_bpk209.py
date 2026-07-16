"""family_bpk209 — task209 / 8a004b2b : "magnify sprite into box" (bit-packed / low-mem recompile).

Same transform as family_rb209 (see that file for the full spec), recompiled for
minimal scored memory.  The 1.3 MB of family_rb209 came from a 45-way candidate
search that materialised ~690 fp16 [1,1,30,30] intermediates.  Here the search is
done on a FIXED 12x12 window anchored at the shown-block top-left (aR,aC):

    for candidate (mag,rt,cl):
        Or = aR-brow-rt*mag,  Oc = aC-bcol-cl*mag
        the placed magnified sprite, restricted to the window, is exactly the
        STATIC crop  Mmag[rt*mag : rt*mag+12, cl*mag : cl*mag+12]
        (because window cell (aR+i,aC+j) -> sprite-mag cell (i+rt*mag, j+cl*mag)).

So all 45 candidate templates are static slices of the 3 magnified sprites, and the
whole colour-mismatch check runs on tiny batched uint8 tensors [45,1,12,12].  The
`fits`, key, arg-min and final placement are scalar / single-grid ops.

Design levers used:  (1) 12x12 downsampled work window for the search (the shown
region is provably <= mh x mw <= 12x12);  (2) all geometry in fp16 (halves the
30x30 masks);  (3) the batched candidate scoring in uint8 with Equal/ReduceMax
(opset 13);  (4) opset 13 so Equal/ReduceMax accept uint8.  No banned ops.

Exact on all local train+test and byte-identical to family_rb209._ref on ARC-GEN
(so exact on every recoverable sample, deterministic on the ~0.5% ambiguous ones).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import GRID_SHAPE, IR_VERSION

F32 = onnx.TensorProto.FLOAT
F16 = onnx.TensorProto.FLOAT16
U8 = onnx.TensorProto.UINT8
BOOL = onnx.TensorProto.BOOL
INT64 = onnx.TensorProto.INT64
H30 = 30
S = 20      # parsing canvas: generator caps width/height at 20, so content is < 20
WIN = 12
OPSET = [oh.make_opsetid("", 13)]


# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def h(self, dims, vals):  # fp16 initializer
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, F16, list(dims),
            np.asarray(vals, np.float16).astype(np.float16).tobytes(), raw=True))
        return n

    def h1(self, v):
        return self.h([1, 1, 1, 1], [v])

    def f32(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, F32, list(dims), [float(v) for v in np.asarray(vals, np.float64).ravel()]))
        return n

    def u8(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, U8, list(dims), np.asarray(vals, np.uint8).tobytes(), raw=True))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


# ---- fp16 comparison helpers -------------------------------------------------
def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F16)


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F16)


def _eqm(g, a, b):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


def _ge(g, a, b):
    return _gt(g, a, g.nd("Sub", [b, g.half]))


def _le(g, a, b):
    return _lt(g, a, g.nd("Add", [b, g.half]))


def _consts(g):
    g.rowidx = g.h([1, 1, H30, 1], list(range(H30)))   # [1,1,30,1] output canvas
    g.colidx = g.h([1, 1, 1, H30], list(range(H30)))   # [1,1,1,30]
    g.r20 = g.h([1, 1, S, 1], list(range(S)))           # [1,1,20,1] parse canvas
    g.c20 = g.h([1, 1, 1, S], list(range(S)))           # [1,1,1,20]
    g.winr = g.h([1, 1, WIN, 1], list(range(WIN)))      # [1,1,12,1]
    g.winc = g.h([1, 1, 1, WIN], list(range(WIN)))      # [1,1,1,12]
    g.half = g.h1(0.5)
    g.one = g.h1(1.0)
    g.cbig = g.h1(1000.0)


def _minmax(g, has, idx, axis):
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], axes=[axis], keepdims=1)
    inv = g.nd("Mul", [has, g.nd("Sub", [g.cbig, idx])])
    mn = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [inv], axes=[axis], keepdims=1)])
    return mn, mx


def _rowspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)
    return _minmax(g, has, g.r20, 2)


def _colspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)
    return _minmax(g, has, g.c20, 3)


def _crop20(g, src, r0, c0, h, w):
    """Crop region [r0:r0+h, c0:c0+w] of a 20x20 src to top-left origin (20x20)."""
    Rrow = g.nd("Mul", [_eqm(g, g.c20, g.nd("Add", [g.r20, r0])),
                        _lt(g, g.r20, g.nd("Sub", [h, g.half]))])
    Rcol = g.nd("Mul", [_eqm(g, g.r20, g.nd("Add", [g.c20, c0])),
                        _lt(g, g.c20, g.nd("Sub", [w, g.half]))])
    return g.nd("MatMul", [Rrow, g.nd("MatMul", [src, Rcol])])


def _win(g, src, r0, c0):
    """Return src[r0:r0+12, c0:c0+12] as [1,1,12,12] from a 20x20 src (dynamic r0,c0)."""
    Grow = _eqm(g, g.nd("Add", [g.winr, r0]), g.c20)      # [1,1,12,20]
    Gcol = _eqm(g, g.r20, g.nd("Add", [g.winc, c0]))      # [1,1,20,12]
    return g.nd("MatMul", [Grow, g.nd("MatMul", [src, Gcol])])


def _place(g, src, Or, Oc):
    """out[r,c] = src[r-Or, c-Oc] on a 20x20 canvas (dynamic Or,Oc)."""
    Prow = _eqm(g, g.r20, g.nd("Add", [g.c20, Or]))       # [1,1,20,20]
    Pcol = _eqm(g, g.c20, g.nd("Add", [g.r20, Oc]))
    return g.nd("MatMul", [Prow, g.nd("MatMul", [src, Pcol])])


# =========================================================================== #
def build_209():
    g = _G()
    _consts(g)

    # ---- one-hot input -> colour value image (fp32 Conv), cropped to 20x20 fp16 ----
    wconv = g.f32([1, 10, 1, 1], list(range(10)))
    Vf = g.nd("Conv", ["input", wconv], kernel_shape=[1, 1])   # [1,1,30,30] fp32
    V = g.nd("Cast", [g.nd("Slice", [Vf, g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])],
             to=F16)                                            # [1,1,20,20] fp16

    # ---- yellow box bbox ----
    Ym = _eqm(g, V, g.h1(4.0))
    br, mr = _rowspan(g, Ym)
    bc, mc = _colspan(g, Ym)
    tall = g.nd("Add", [g.nd("Sub", [mr, br]), g.one])
    wide = g.nd("Add", [g.nd("Sub", [mc, bc]), g.one])

    # ---- colour / inbox / sprite / shown masks ----
    col = g.nd("Mul", [_gt(g, V, g.half),
                       _gt(g, g.nd("Abs", [g.nd("Sub", [V, g.h1(4.0)])]), g.half)])
    inrow = g.nd("Mul", [_ge(g, g.r20, br), _le(g, g.r20, mr)])
    incol = g.nd("Mul", [_ge(g, g.c20, bc), _le(g, g.c20, mc)])
    inbox = g.nd("Mul", [inrow, incol])
    spr = g.nd("Mul", [col, g.nd("Sub", [g.one, inbox])])
    shown = g.nd("Mul", [col, inbox])

    # ---- sprite -> origin-anchored value grid P; magnified sprites Tm[mag] ----
    Vspr = g.nd("Mul", [V, spr])
    sr0, sr1 = _rowspan(g, spr)
    sc0, sc1 = _colspan(g, spr)
    tp = g.nd("Add", [g.nd("Sub", [sr1, sr0]), g.one])
    wp = g.nd("Add", [g.nd("Sub", [sc1, sc0]), g.one])
    Pfull = _crop20(g, Vspr, sr0, sc0, tp, wp)
    Psmall = g.nd("Slice", [Pfull, g.i64([0, 0]), g.i64([4, 6]), g.i64([2, 3])])  # [1,1,4,6]
    Tm, Tu = {}, {}
    for mag in (2, 3, 4):
        sc = g.f32([4], [1, 1, mag, mag])
        up = g.nd("Resize", [Psmall, "", sc], mode="nearest")          # [1,1,4mag,6mag]
        pad = g.i64([0, 0, 0, 0, 0, 0, H30 - 4 * mag, H30 - 6 * mag])
        Tm[mag] = g.nd("Pad", [up, pad, g.h1(0.0)], mode="constant")   # [1,1,30,30] fp16
        Tu[mag] = g.nd("Cast", [Tm[mag]], to=U8)

    # ---- shown blocks: absolute top-left (aR,aC), window crops ----
    Vsh = g.nd("Mul", [V, shown])
    aR, _ = _rowspan(g, shown)
    aC, _ = _colspan(g, shown)
    Wsbox = g.nd("Cast", [_win(g, Vsh, aR, aC)], to=U8)     # [1,1,12,12] u8
    zeroU = g.u8([1, 1, 1, 1], [0])
    shownB = g.nd("Greater", [Wsbox, zeroU])                # bool, shown iff colour>0

    # ---- 45 static candidate templates -> [45,1,12,12] uint8 ----
    order = [(mag, rt, cl) for mag in (2, 3, 4) for rt in range(3) for cl in range(5)]
    slices = []
    for (mag, rt, cl) in order:
        RM, CM = rt * mag, cl * mag
        slices.append(g.nd("Slice", [Tu[mag], g.i64([RM, CM]),
                                     g.i64([RM + WIN, CM + WIN]), g.i64([2, 3])]))
    Wtpl = g.nd("Concat", slices, axis=0)                  # [45,1,12,12] u8

    # ---- batched colour-match check: per shown cell need Wsbox==Wtpl ----
    neq = g.nd("Not", [g.nd("Equal", [Wsbox, Wtpl])])          # [45,1,12,12] bool mismatch
    mm = g.nd("Cast", [g.nd("And", [shownB, neq])], to=U8)     # 1 where shown & mismatch
    bad = g.nd("ReduceMax", [mm], axes=[2, 3], keepdims=1)      # [45,1,1,1] u8
    matchok = g.nd("Cast", [g.nd("Equal", [bad, zeroU])], to=F32)

    # ---- fits + key + arg-min (fp32 scalars, batched over 45) ----
    def c32(a):
        return g.nd("Cast", [a], to=F32)

    aRb = g.nd("Sub", [c32(aR), c32(br)])
    aCb = g.nd("Sub", [c32(aC), c32(bc)])
    tp3, wp3 = c32(tp), c32(wp)
    tall3, wide3 = c32(tall), c32(wide)
    MAG = g.f32([45, 1, 1, 1], [m for (m, r, c) in order])
    RMv = g.f32([45, 1, 1, 1], [r * m for (m, r, c) in order])
    CMv = g.f32([45, 1, 1, 1], [c * m for (m, r, c) in order])
    half3 = g.f32([1, 1, 1, 1], [0.5])
    one3 = g.f32([1, 1, 1, 1], [1.0])

    def ge3(a, b):
        return g.nd("Cast", [g.nd("Greater", [a, g.nd("Sub", [b, half3])])], to=F32)

    def le3(a, b):
        return g.nd("Cast", [g.nd("Less", [a, g.nd("Add", [b, half3])])], to=F32)

    Or = g.nd("Sub", [aRb, RMv])                # [45,1,1,1]
    Oc = g.nd("Sub", [aCb, CMv])
    mh = g.nd("Mul", [tp3, MAG])
    mw = g.nd("Mul", [wp3, MAG])
    fits = g.nd("Mul", [g.nd("Mul", [ge3(Or, one3), ge3(Oc, one3)]),
                        g.nd("Mul", [le3(g.nd("Add", [Or, mh]), tall3),
                                     le3(g.nd("Add", [Oc, mw]), wide3)])])
    valid = g.nd("Mul", [fits, matchok])
    key = g.nd("Add", [g.nd("Add", [g.nd("Mul", [MAG, g.f32([1, 1, 1, 1], [10000.0])]),
                                    g.nd("Mul", [Or, g.f32([1, 1, 1, 1], [100.0])])]), Oc])
    keyf = g.nd("Add", [g.nd("Mul", [valid, key]),
                        g.nd("Mul", [g.nd("Sub", [one3, valid]), g.f32([1, 1, 1, 1], [1e6])])])
    minkey = g.nd("ReduceMin", [keyf], axes=[0], keepdims=1)
    sel = g.nd("Mul", [valid, g.nd("Cast", [g.nd("Less", [
        g.nd("Abs", [g.nd("Sub", [keyf, minkey])]), half3])], to=F32)])
    Or_sel = g.nd("Cast", [g.nd("ReduceMax", [g.nd("Mul", [sel, Or])], axes=[0], keepdims=1)], to=F16)
    Oc_sel = g.nd("Cast", [g.nd("ReduceMax", [g.nd("Mul", [sel, Oc])], axes=[0], keepdims=1)], to=F16)
    mag_sel = g.nd("Cast", [g.nd("ReduceMax", [g.nd("Mul", [sel, MAG])], axes=[0], keepdims=1)], to=F16)

    # ---- select magnified sprite + place (all on the 20x20 canvas) ----
    Msel = None
    for mag in (2, 3, 4):
        Tm20 = g.nd("Slice", [Tm[mag], g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])
        term = g.nd("Mul", [_eqm(g, mag_sel, g.h1(mag)), Tm20])
        Msel = term if Msel is None else g.nd("Add", [Msel, term])
    Placed = _place(g, Msel, Or_sel, Oc_sel)

    # ---- corners + compose + one-hot ----
    cr = g.nd("Add", [_eqm(g, g.r20, g.h1(0.0)),
                      _eqm(g, g.r20, g.nd("Sub", [tall, g.one]))])
    cc = g.nd("Add", [_eqm(g, g.c20, g.h1(0.0)),
                      _eqm(g, g.c20, g.nd("Sub", [wide, g.one]))])
    corners = g.nd("Mul", [g.nd("Mul", [cr, cc]), g.h1(4.0)])
    OUTv = g.nd("Add", [Placed, g.nd("Mul", [corners, _lt(g, Placed, g.half)])])
    ingrid = g.nd("Mul", [_lt(g, g.r20, g.nd("Sub", [tall, g.half])),
                          _lt(g, g.c20, g.nd("Sub", [wide, g.half]))])
    # cells outside the box -> sentinel -1 so no colour channel matches
    OUTv2 = g.nd("Sub", [g.nd("Mul", [OUTv, ingrid]),
                         g.nd("Sub", [g.one, ingrid])])                 # [1,1,20,20]
    # pad 20x20 -> 30x30 output canvas with sentinel -1 (no colour there)
    padded = g.nd("Pad", [OUTv2, g.i64([0, 0, 0, 0, 0, 0, H30 - S, H30 - S]),
                          g.h1(-1.0)], mode="constant")                 # [1,1,30,30]
    valvec = g.h([1, 10, 1, 1], list(range(10)))
    g.nd("Cast", [g.nd("Equal", [g.nd("Cast", [padded], to=F32),
                                 g.nd("Cast", [valvec], to=F32)])], to=F32, out="output")

    x = oh.make_tensor_value_info("input", F32, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F32, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, "bpk209", [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET)
    onnx.checker.check_model(m, full_check=True)
    return m


# =========================================================================== #
# detection / candidates  (reuse the reference numerics from family_rb209)      #
# =========================================================================== #
from family_rb209 import _ref, _pairs, _matches  # noqa: E402


def candidates(ex):
    prs = _pairs(ex)
    if not _matches(prs):
        return []
    try:
        return [("bpk209", build_209())]
    except Exception:
        return []
