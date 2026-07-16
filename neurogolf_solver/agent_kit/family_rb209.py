"""family_rb209 — task209 / 8a004b2b : "magnify sprite into box".

INPUT (variable W x H) contains:
  * a small scale-1 colour creature (colours in {1,2,3,8}) sitting BELOW a box;
    the creature always contains pixel (0,0) (its bbox top-left is filled).
  * a YELLOW(4) box = 4 corner pixels of a wide x tall rectangle.
  * inside the box, a SUBSET of the creature's pixels drawn MAGNIFIED by `mag`
    (each pixel -> mag x mag solid block) at box offset (irow, icol).
OUTPUT = the box interior (size wide x tall): the 4 yellow corners + the FULL
creature magnified by mag at (irow, icol) (ALL pixels, mag x mag blocks).

Transform:
  1. yellow bbox -> box (brow,bcol,tall,wide).
  2. colour pixels outside the box (below it) -> the small creature P (value grid,
     bbox tp x wp, tp<=3, wp<=5), cropped to origin.
  3. colour pixels inside the box -> the shown magnified blocks (box-relative);
     (minR,minC) = top-left of the shown region.
  4. mag,(Or,Oc): every valid placement has its top-most shown block starting at a
     creature-row rt and its left-most shown block at creature-col cl, so
     Or = minR - rt*mag, Oc = minC - cl*mag with rt in [0,tp), cl in [0,wp).
     Enumerate mag in {2,3,4}, rt in [0,3), cl in [0,5); a candidate is VALID iff
     the full magnified creature placed at (Or,Oc) covers every shown block with
     matching colour and fits inside the box.  Pick the smallest mag, then the
     lexicographically-smallest (Or,Oc)  (== brute-force min-mag + lexmin offset).
  5. OUTPUT = corners + magnify(P,mag) placed at (Or,Oc), one-hot, clipped to box.

The generator is ~0.5% inherently AMBIGUOUS (a solid / row- or column-constant
creature revealed by a symmetric subset admits several distinct outputs — even a
perfect solver cannot recover the generator's random irow/icol).  On every input
whose output IS determined by the grid, this solver is exact; it is exact on all
266 local train+test+arc-gen pairs and all 8 recorded known-fail inputs.

ONNX: value image via 1x1 Conv; yellow/sprite/shown bboxes via masked min/max;
dyncrop MatMuls crop the sprite and the box to the origin; Resize(nearest) builds
the x2/x3/x4 magnified creature; a 45-way static candidate search (dynamic
MatMul placement + masked colour-mismatch count) selects (mag,Or,Oc) by an
integer key arg-min; the winner is placed with another dynamic MatMul and
one-hot encoded.  opset-10, static [1,10,30,30], no banned ops.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
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


def _consts(g):
    g.rowidx = g.f([1, 1, H30, 1], list(range(H30)))
    g.colidx = g.f([1, 1, 1, H30], list(range(H30)))
    g.half = g.f1(0.5)
    g.one = g.f1(1.0)
    g.cbig = g.f1(1000.0)
    g.valvec = g.f([1, 10, 1, 1], list(range(10)))


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
    """Crop region [r0:r0+h, c0:c0+w] of src to the top-left origin."""
    Rrow = g.nd("Mul", [_eqm(g, g.colidx, g.nd("Add", [g.rowidx, r0])),
                        _lt(g, g.rowidx, g.nd("Sub", [h, g.half]))])
    Rcol = g.nd("Mul", [_eqm(g, g.rowidx, g.nd("Add", [g.colidx, c0])),
                        _lt(g, g.colidx, g.nd("Sub", [w, g.half]))])
    return g.nd("MatMul", [Rrow, g.nd("MatMul", [src, Rcol])])


def _place(g, src, Or, Oc):
    """Shift origin-anchored src so its (0,0) lands at (Or,Oc): out[r,c]=src[r-Or,c-Oc]."""
    Prow = _eqm(g, g.rowidx, g.nd("Add", [g.colidx, Or]))    # [r == k+Or]
    Pcol = _eqm(g, g.colidx, g.nd("Add", [g.rowidx, Oc]))    # [c == k+Oc]
    return g.nd("MatMul", [Prow, g.nd("MatMul", [src, Pcol])])


# =========================================================================== #
# ONNX builder                                                                 #
# =========================================================================== #
def build_209():
    g = _G()
    _consts(g)
    V = _value_img(g, "input")                        # per-cell colour value

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
        sc = g.f([4], [1, 1, mag, mag])
        up = g.nd("Resize", [Psmall, sc], mode="nearest")        # [1,1,4*mag,6*mag]
        Tm[mag] = g.nd("Pad", [up], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, H30 - 4 * mag, H30 - 6 * mag])

    # ---- shown blocks box-relative + (minR,minC) ----
    Vsh = g.nd("Mul", [V, shown])
    Sbox = _crop(g, Vsh, br, bc, tall, wide)
    Sm = _gt(g, Sbox, g.half)
    mR0, _ = _rowspan(g, Sm)
    mC0, _ = _colspan(g, Sm)

    # ---- 45-way candidate search ----
    keys, valids, Ors, Ocs, magc = [], [], [], [], []
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
                valid = g.nd("Mul", [fits, _lt(g, mism, g.half)])
                key = g.nd("Add", [g.nd("Add", [g.f1(mag * 10000.0),
                                                g.nd("Mul", [Or, g.f1(100.0)])]), Oc])
                keys.append(g.nd("Add", [g.nd("Mul", [valid, key]),
                                         g.nd("Mul", [g.nd("Sub", [g.one, valid]),
                                                      g.f1(1e6)])]))
                valids.append(valid)
                Ors.append(Or)
                Ocs.append(Oc)
                magc.append(mag)

    minkey = g.nd("Min", keys)
    Or_sel = None
    Oc_sel = None
    mag_sel = None
    for i in range(len(keys)):
        ind = g.nd("Mul", [valids[i], _eqm(g, keys[i], minkey)])
        tOr = g.nd("Mul", [ind, Ors[i]])
        tOc = g.nd("Mul", [ind, Ocs[i]])
        tMg = g.nd("Mul", [ind, g.f1(magc[i])])
        Or_sel = tOr if Or_sel is None else g.nd("Add", [Or_sel, tOr])
        Oc_sel = tOc if Oc_sel is None else g.nd("Add", [Oc_sel, tOc])
        mag_sel = tMg if mag_sel is None else g.nd("Add", [mag_sel, tMg])

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
    return _model(g, "rb209")


# =========================================================================== #
# numpy reference  (mirrors the ONNX numerics exactly)                         #
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
        return [("rb209", build_209())]
    except Exception:
        return []
