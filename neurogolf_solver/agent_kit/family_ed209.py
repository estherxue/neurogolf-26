"""family_ed209 — task209 / 8a004b2b : "magnify sprite into box" (enriched-arsenal golf).

Byte-identical transform to family_bpk209 / family_rb209 (see family_rb209 for the
full spec + numpy reference `_ref`), recompiled to cut scored intermediate memory.

The generator (task_8a004b2b) fixes these bounds, which this rebuild exploits:
  * input W,H in [15,20]                         -> parse canvas 20x20
  * box height  tall <= height-4 <= 16           -> OUTPUT canvas only 16 rows
  * magnified creature = mag*t x mag*w, with
    t<=3, w<=7-mag  =>  mag*w<=12, mag*t<=12      -> 12x12 search window
So the 45-way candidate search runs on tiny [45,1,12,12] uint8 batches, and the
whole output side (placement / corners / one-hot) lives on a 16x20 canvas.

Golf levers vs bpk209 (all value-exact — output is byte-identical to bpk209 on
every input, verified against the incumbent ONNX on local + 3000+ fresh gen):
  1. FREE-OUTPUT one-hot tail: one-hot on the small work canvas (Equal ->
     [1,10,16,20] bool), then Pad straight to the FREE [1,10,30,30] `output`.
     Kills the fp16 30x30 `padded`, both f32 30x30 casts, and the 9000-byte
     [1,10,30,30] bool Equal.
  2. BATCH mask-trick: the shown-cell colour check collapses from 5 to 4
     [45,1,12,12] tensors — Where-mask the templates to 0 off the shown cells
     (Wsbox is already 0 there), Equal, then a single ReduceMin(uint8) instead
     of Not+And+Cast+ReduceMax.
  3. 16-row output canvas (tall<=16) shrinks every placement/compose tensor.

opset 13 (Equal/ReduceMax/ReduceMin/Where on uint8), static [1,10,30,30], IR 10,
no banned ops.
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
S = 20       # parse canvas (input W,H <= 20)
OH = 16      # output canvas rows (box height tall <= height-4 <= 16)
WIN = 12     # search window (magnified creature <= 12x12)
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

    def i8(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, onnx.TensorProto.INT8, list(dims), np.asarray(vals, np.int8).tobytes(), raw=True))
        return n

    def f32s(self, v):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F32, [], [float(v)]))
        return n

    def boolc(self, v):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, BOOL, [], [bool(v)]))
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
    g.r20 = g.h([1, 1, S, 1], list(range(S)))           # [1,1,20,1] parse rows
    g.c20 = g.h([1, 1, 1, S], list(range(S)))           # [1,1,1,20] parse cols
    g.or16 = g.h([1, 1, OH, 1], list(range(OH)))        # [1,1,16,1] out rows
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


def _place16(g, src20, Or, Oc):
    """out[r,c] = src20[r-Or, c-Oc] on a 16x20 canvas (dynamic Or,Oc).

    Prow[16,20]: out-row r picks src-row (r-Or).  Pcol[20,20]: col c picks (c-Oc)."""
    Prow = _eqm(g, g.or16, g.nd("Add", [g.c20, Or]))      # [1,1,16,20]  (r == k+Or)
    Pcol = _eqm(g, g.c20, g.nd("Add", [g.r20, Oc]))       # [1,1,20,20]  (k == c-Oc)
    return g.nd("MatMul", [Prow, g.nd("MatMul", [src20, Pcol])])   # [1,1,16,20]


# =========================================================================== #
def build_209(mode="conv"):
    g = _G()
    _consts(g)

    # ---- one-hot input -> colour value image (fp32 Conv), cropped to 20x20 fp16 ----
    wconv = g.f32([1, 10, 1, 1], list(range(10)))
    Vf = g.nd("Conv", ["input", wconv], kernel_shape=[1, 1])   # [1,1,30,30] fp32
    V = g.nd("Cast", [g.nd("Slice", [Vf, g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])],
             to=F16)                                            # [1,1,20,20] fp16

    # ---- yellow box bbox ---- (d4 = |V-4| shared by the yellow mask and colour mask)
    c4 = g.h1(4.0)
    d4 = g.nd("Abs", [g.nd("Sub", [V, c4])])
    Ym = _lt(g, d4, g.half)                            # V == 4  (yellow)
    br, mr = _rowspan(g, Ym)
    bc, mc = _colspan(g, Ym)
    tall = g.nd("Add", [g.nd("Sub", [mr, br]), g.one])
    wide = g.nd("Add", [g.nd("Sub", [mc, bc]), g.one])

    # ---- colour / inbox / sprite / shown masks ----
    col = g.nd("Mul", [_gt(g, V, g.half), _gt(g, d4, g.half)])   # colour: V>0 and V!=4
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
        # pad to 20 rows (place canvas) x 28 cols (max template slice col CM+12=28)
        pad = g.i64([0, 0, 0, 0, 0, 0, S - 4 * mag, 28 - 6 * mag])
        Tm[mag] = g.nd("Pad", [up, pad, g.h1(0.0)], mode="constant")   # [1,1,20,28] fp16
        Tu[mag] = g.nd("Cast", [Tm[mag]], to=U8)

    # ---- shown blocks: absolute top-left (aR,aC), window crops ----
    Vsh = g.nd("Mul", [V, shown])
    aR, _ = _rowspan(g, shown)
    aC, _ = _colspan(g, shown)
    Wsbox = g.nd("Cast", [_win(g, Vsh, aR, aC)], to=U8)     # [1,1,12,12] u8
    order = [(mag, rt, cl) for mag in (2, 3, 4) for rt in range(3) for cl in range(5)]

    if mode == "conv":
        # ---- RUNTIME-BIAS QLinearConv BINARY DETECT (arsenal #1) ----------------
        # One-hot the shown query over the 4 possible sprite colours {1,2,3,8}; the
        # match count at every candidate offset is a single cross-correlation
        # (QLinearConv, int8 runtime weight = the query, uint8 template input).
        # corr[offset] == shown_count  <=>  every shown cell matches the template.
        colv = g.u8([1, 4, 1, 1], [1, 2, 3, 8])
        Qb = g.nd("Equal", [Wsbox, colv])                     # [1,4,12,12] bool
        Qw = g.nd("Cast", [Qb], to=onnx.TensorProto.INT8)     # weight [1,4,12,12] i8
        # shown_count = #shown cells (each has exactly one colvec colour)
        cnt = g.nd("ReduceSum", [g.nd("Cast", [Qb], to=F16), g.i64([0, 1, 2, 3])],
                   keepdims=1)                                # [1,1,1,1] f16
        scu = g.nd("Cast", [cnt], to=U8)                      # [1,1,1,1] u8
        xs, xz = g.f32s(1.0), g.u8([], [0])
        ws, wz = g.f32s(1.0), g.i8([], [0])
        ys, yz = g.f32s(1.0), g.u8([], [0])
        WQ = 20 - WIN + 1     # 9   (template rows 20)
        HQ = 28 - WIN + 1     # 17  (template cols 28)
        gathers = []
        for mag in (2, 3, 4):
            Tb = g.nd("Equal", [Tu[mag], colv])               # [1,4,20,28] bool
            Toh = g.nd("Cast", [Tb], to=U8)                   # [1,4,20,28] u8
            ymap = g.nd("QLinearConv", [Toh, xs, xz, Qw, ws, wz, ys, yz])  # [1,1,9,17] u8 corr
            mmap = g.nd("Cast", [g.nd("Equal", [ymap, scu])], to=F32)      # [1,1,9,17] match
            flat = g.nd("Reshape", [mmap, g.i64([WQ * HQ])])              # [153]
            idx = [rt * mag * HQ + cl * mag for rt in range(3) for cl in range(5)]
            gathers.append(g.nd("Gather", [flat, g.i64(idx)], axis=0))    # [15]
        matchok = g.nd("Reshape", [g.nd("Concat", gathers, axis=0), g.i64([45, 1, 1, 1])])
    else:
        # ---- mask-trick fallback (uint8, no QLinearConv) ------------------------
        zeroU = g.u8([1, 1, 1, 1], [0])
        shownB = g.nd("Greater", [Wsbox, zeroU])
        slices = []
        for (mag, rt, cl) in order:
            RM, CM = rt * mag, cl * mag
            slices.append(g.nd("Slice", [Tu[mag], g.i64([RM, CM]),
                                         g.i64([RM + WIN, CM + WIN]), g.i64([2, 3])]))
        Wtpl = g.nd("Concat", slices, axis=0)                  # [45,1,12,12] u8
        Wtpl_m = g.nd("Where", [shownB, Wtpl, zeroU])          # [45,1,12,12] u8
        eqc = g.nd("Cast", [g.nd("Equal", [Wsbox, Wtpl_m])], to=U8)
        matchok = g.nd("Cast", [g.nd("ReduceMin", [eqc], axes=[2, 3], keepdims=1)], to=F32)

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

    # ---- select magnified sprite (20x20) + place onto 16x20 out canvas ----
    Msel = None
    for mag in (2, 3, 4):
        Tm20 = g.nd("Slice", [Tm[mag], g.i64([0, 0]), g.i64([S, S]), g.i64([2, 3])])
        term = g.nd("Mul", [_eqm(g, mag_sel, g.h1(mag)), Tm20])
        Msel = term if Msel is None else g.nd("Add", [Msel, term])
    Placed = _place16(g, Msel, Or_sel, Oc_sel)             # [1,1,16,20]

    # ---- corners + compose + one-hot on the 16x20 canvas ----
    cr = g.nd("Add", [_eqm(g, g.or16, g.h1(0.0)),
                      _eqm(g, g.or16, g.nd("Sub", [tall, g.one]))])
    cc = g.nd("Add", [_eqm(g, g.c20, g.h1(0.0)),
                      _eqm(g, g.c20, g.nd("Sub", [wide, g.one]))])
    corners = g.nd("Mul", [g.nd("Mul", [cr, cc]), g.h1(4.0)])
    OUTv = g.nd("Add", [Placed, g.nd("Mul", [corners, _lt(g, Placed, g.half)])])
    ingrid = g.nd("Mul", [_lt(g, g.or16, g.nd("Sub", [tall, g.half])),
                          _lt(g, g.c20, g.nd("Sub", [wide, g.half]))])
    # cells outside the box -> sentinel -1 so no colour channel matches
    OUTv2 = g.nd("Sub", [g.nd("Mul", [OUTv, ingrid]),
                         g.nd("Sub", [g.one, ingrid])])    # [1,1,16,20]

    # ---- FREE-OUTPUT one-hot: Equal on 16x20 -> Pad bool straight to 30x30 output ----
    valvec = g.h([1, 10, 1, 1], list(range(10)))
    onehot = g.nd("Equal", [OUTv2, valvec])                # [1,10,16,20] bool
    g.nd("Pad", [onehot, g.i64([0, 0, 0, 0, 0, 0, H30 - OH, H30 - S]), g.boolc(False)],
         out="output", mode="constant")                    # [1,10,30,30] bool

    x = oh.make_tensor_value_info("input", F32, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", BOOL, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, "ed209", [x], [y], inits),
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
    out = []
    for tag, mode in (("ed209", "conv"), ("ed209m", "mask")):
        try:
            out.append((tag, build_209(mode)))
        except Exception:
            pass
    return out
