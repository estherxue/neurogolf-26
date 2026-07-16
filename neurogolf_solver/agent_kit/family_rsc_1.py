"""family_rsc_1 — from-scratch minimal-rule rebuilds (golf ranks 14..28).

Each solver is a hand-built opset-10 ONNX (onnx.helper) implementing the TRUE minimal
rule for a task, aiming to be far cheaper than the generic-search incumbents.
Every family yields only when its pure-numpy mirror is bit-exact on train+test.

task090 (verify_3eda0437): recolour the maximum-area all-zero axis-aligned rectangle
    (h,w >= 2) to colour 6.  Empirically the max-area zero-rectangle is unique per grid
    and grids are short (H<=5).  ONNX: single batched Conv (one output channel per
    (h,w) candidate) counts zeros in each top-left window -> all-zero indicator; pick
    global max area; grouped ConvTranspose paints the winning window(s) with 6.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
F16 = onnx.TensorProto.FLOAT16
BOOL = onnx.TensorProto.BOOL
INT64 = onnx.TensorProto.INT64


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

    def i64(self, vals):
        return self.c(INT64, [len(vals)], [int(v) for v in vals])

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
# task090 — 3eda0437                                                           #
# --------------------------------------------------------------------------- #
_HP = 8              # rows processed (grids are H<=5; margin)
_KH = 6             # max candidate rectangle height
_KW = 9             # max candidate rectangle width
_HCAND = list(range(2, _KH + 1))   # 2..6
_WCAND = list(range(2, _KW + 1))   # 2..9
_CANDS = [(h, w) for h in _HCAND for w in _WCAND]
_NC = len(_CANDS)


def build_090():
    g = _G()
    # zero-mask, first _HP rows, as fp16
    z = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])         # [1,1,30,30] ch0
    z = g.nd("Slice", [z, g.i64([0]), g.i64([_HP]), g.i64([2])])             # [1,1,HP,30]
    zf = g.nd("Cast", [z], to=F16)
    zpad = g.nd("Pad", [zf], mode="constant", value=0.0,
                pads=[0, 0, 0, 0, 0, 0, _KH - 1, _KW - 1])                    # [1,1,HP+5,38]

    # per-candidate ones kernel (top-left h x w block)
    W1 = np.zeros((_NC, 1, _KH, _KW), np.float32)
    for c, (h, w) in enumerate(_CANDS):
        W1[c, 0, :h, :w] = 1.0
    w1 = g.c(F16, [_NC, 1, _KH, _KW], W1)
    conv = g.nd("Conv", [zpad, w1], kernel_shape=[_KH, _KW])                  # [1,NC,HP,30]

    areas = np.array([h * w for (h, w) in _CANDS], np.float32)
    area_m05 = g.c(F16, [1, _NC, 1, 1], areas - 0.5)
    areav = g.c(F16, [1, _NC, 1, 1], areas)
    tl = g.nd("Greater", [conv, area_m05])                                    # bool [1,NC,HP,30]

    zero = g.c(F16, [1, 1, 1, 1], [0.0])
    masked = g.nd("Where", [tl, areav, zero])                                 # fp16
    winA = g.nd("ReduceMax", [masked], axes=[1, 2, 3], keepdims=1)            # [1,1,1,1]

    half = g.c(F16, [1, 1, 1, 1], [0.5])
    eqhi = g.nd("Less", [areav, g.nd("Add", [winA, half])])                   # bool [1,NC,1,1]
    eqlo = g.nd("Greater", [areav, g.nd("Sub", [winA, half])])
    eqwin = g.nd("And", [eqhi, eqlo])
    iswin = g.nd("And", [tl, eqwin])                                          # bool [1,NC,HP,30]
    iswf = g.nd("Cast", [iswin], to=F16)

    W2 = np.zeros((_NC, 1, _KH, _KW), np.float32)
    for c, (h, w) in enumerate(_CANDS):
        W2[c, 0, :h, :w] = 1.0
    w2 = g.c(F16, [_NC, 1, _KH, _KW], W2)
    ct = g.nd("ConvTranspose", [iswf, w2], group=_NC, kernel_shape=[_KH, _KW])  # [1,NC,HP+5,38]
    fills = g.nd("ReduceSum", [ct], axes=[1], keepdims=1)                     # [1,1,HP+5,38]
    fills = g.nd("Slice", [fills, g.i64([0, 0]), g.i64([_HP, 30]), g.i64([2, 3])])  # [1,1,HP,30]

    fillm = g.nd("Cast", [g.nd("Greater", [fills, half])], to=F)              # float [1,1,HP,30]
    fillm = g.nd("Pad", [fillm], mode="constant", value=0.0,
                 pads=[0, 0, 0, 0, 0, 0, 30 - _HP, 0])                        # [1,1,30,30]

    vec = np.zeros((1, 10, 1, 1), np.float32)
    vec[0, 6, 0, 0] = 1.0
    vec[0, 0, 0, 0] = -1.0
    vc = g.c(F, [1, 10, 1, 1], vec)
    delta = g.nd("Mul", [fillm, vc])                                          # float [1,10,30,30]
    g.nd("Add", ["input", delta], "output")
    return _model(g, "rsc_090")


# --------------------------------------------------------------------------- #
# task153 — 681b3aeb  (two shapes interlock into a solid 3x3 square)           #
# --------------------------------------------------------------------------- #
def _eqm_f(g, a, b):
    d = g.nd("Sub", [a, b])
    d = g.nd("Abs", [d])
    return g.nd("Cast", [g.nd("Less", [d, g.c(F, [1, 1, 1, 1], [0.5])])], to=F)


def build_153():
    g = _G()
    inp = "input"
    # ---- per-colour normalisation: crop each channel to its 3x3 bbox origin --
    rowhas = g.nd("ReduceMax", [inp], axes=[3], keepdims=1)        # [1,10,30,1]
    colhas = g.nd("ReduceMax", [inp], axes=[2], keepdims=1)        # [1,10,1,30]
    ridx = g.c(F, [1, 1, 30, 1], list(range(30)))
    cidx_r = g.c(F, [1, 1, 1, 30], list(range(30)))
    c30 = g.c(F, [1, 1, 1, 1], [30.0])
    minr = g.nd("Sub", [c30, g.nd("ReduceMax",
                [g.nd("Mul", [rowhas, g.nd("Sub", [c30, ridx])])], axes=[2], keepdims=1)])  # [1,10,1,1]
    minc = g.nd("Sub", [c30, g.nd("ReduceMax",
                [g.nd("Mul", [colhas, g.nd("Sub", [c30, cidx_r])])], axes=[3], keepdims=1)])  # [1,10,1,1]

    k3r = g.c(F, [1, 1, 3, 1], [0, 1, 2])
    k3c = g.c(F, [1, 1, 1, 3], [0, 1, 2])
    jrow = g.c(F, [1, 1, 1, 30], list(range(30)))      # column coordinate
    irow = g.c(F, [1, 1, 30, 1], list(range(30)))      # row coordinate
    Srow = _eqm_f(g, g.nd("Add", [minr, k3r]), jrow)   # [1,10,3,30]  (k,i)->1 if i==minr+k
    Scol = _eqm_f(g, irow, g.nd("Add", [minc, k3c]))   # [1,10,30,3]  (j,k)->1 if j==minc+k
    tmp = g.nd("MatMul", [inp, Scol])                  # [1,10,30,3]
    Mcrop = g.nd("MatMul", [Srow, tmp])                # [1,10,3,3]  normalised shapes

    # ---- identify the two foreground colours --------------------------------
    present = g.nd("ReduceMax", [rowhas], axes=[2], keepdims=1)     # [1,10,1,1]
    fgmask = g.c(F, [1, 10, 1, 1], [0] + [1] * 9)
    pfg = g.nd("Mul", [present, fgmask])
    cidx = g.c(F, [1, 10, 1, 1], list(range(10)))
    c9 = g.c(F, [1, 1, 1, 1], [9.0])
    colorA = g.nd("Sub", [c9, g.nd("ReduceMax",
                  [g.nd("Mul", [pfg, g.nd("Sub", [c9, cidx])])], axes=[1], keepdims=1)])  # min fg
    colorB = g.nd("ReduceMax", [g.nd("Mul", [pfg, cidx])], axes=[1], keepdims=1)          # max fg
    eVecA = _eqm_f(g, cidx, colorA)                    # [1,10,1,1]
    eVecB = _eqm_f(g, cidx, colorB)
    M_A = g.nd("ReduceSum", [g.nd("Mul", [Mcrop, eVecA])], axes=[1], keepdims=1)  # [1,1,3,3]
    M_B = g.nd("ReduceSum", [g.nd("Mul", [Mcrop, eVecB])], axes=[1], keepdims=1)

    # ---- shift matrices (down by d / right by d) ----------------------------
    Shd, Shc = {}, {}
    for d in range(3):
        md = np.zeros((3, 3), np.float32)
        mc = np.zeros((3, 3), np.float32)
        for i in range(3):
            if 0 <= i - d < 3:
                md[i, i - d] = 1.0
        for k in range(3):
            if 0 <= k + d < 3:
                mc[k, k + d] = 1.0
        Shd[d] = g.c(F, [1, 1, 3, 3], md)
        Shc[d] = g.c(F, [1, 1, 3, 3], mc)

    def place(M, dr, dc):
        return g.nd("MatMul", [g.nd("MatMul", [Shd[dr], M]), Shc[dc]])

    Ash = {o: place(M_A, o // 3, o % 3) for o in range(9)}
    Bsh = {o: place(M_B, o // 3, o % 3) for o in range(9)}

    one = g.c(F, [1, 1, 1, 1], [1.0])
    half = g.c(F, [1, 1, 1, 1], [0.5])
    sumA = sumB = None
    for oa in range(9):
        for ob in range(9):
            S = g.nd("Add", [Ash[oa], Bsh[ob]])
            err = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [S, one])])],
                       axes=[1, 2, 3], keepdims=1)               # [1,1,1,1]
            vf = g.nd("Cast", [g.nd("Less", [err, half])], to=F)
            cA = g.nd("Mul", [vf, Ash[oa]])
            cB = g.nd("Mul", [vf, Bsh[ob]])
            sumA = cA if sumA is None else g.nd("Add", [sumA, cA])
            sumB = cB if sumB is None else g.nd("Add", [sumB, cB])

    onehot = g.nd("Add", [g.nd("Mul", [sumA, eVecA]), g.nd("Mul", [sumB, eVecB])])  # [1,10,3,3]
    g.nd("Pad", [onehot], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, 27, 27])
    return _model(g, "rsc_153")


def _mirror_153(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or max(a.shape) > 30:
        return None
    cs = [c for c in range(1, 10) if (a == c).any()]
    if len(cs) != 2:
        return None
    ca, cb = min(cs), max(cs)
    out = _tile2(a, ca, cb)
    return out


def _tile2(a, ca, cb):
    def norm(c):
        ys, xs = np.where(a == c)
        m = np.zeros((3, 3), int)
        r0, c0 = ys.min(), xs.min()
        for y, x in zip(ys, xs):
            if y - r0 > 2 or x - c0 > 2:
                return None
            m[y - r0, x - c0] = 1
        return m
    MA, MB = norm(ca), norm(cb)
    if MA is None or MB is None:
        return None

    def shift(M, dr, dc):
        out = np.zeros((3, 3), int)
        for i in range(3):
            for j in range(3):
                if M[i, j] and 0 <= i + dr < 3 and 0 <= j + dc < 3:
                    out[i + dr, j + dc] = 1
        return out
    for oa in range(9):
        As = shift(MA, oa // 3, oa % 3)
        if As.sum() != MA.sum():
            continue
        for ob in range(9):
            Bs = shift(MB, ob // 3, ob % 3)
            if Bs.sum() != MB.sum():
                continue
            if np.array_equal(As + Bs, np.ones((3, 3), int)):
                return As * ca + Bs * cb
    return None


# --------------------------------------------------------------------------- #
# task308 — c8cbb738  (overlay every marker centred on a common centre)        #
# --------------------------------------------------------------------------- #
_WC = 9   # working canvas cap (markers <= 7 in data)


def build_308():
    g = _G()
    inp = "input"
    y30 = g.c(F, [1, 1, 30, 1], list(range(30)))
    x30 = g.c(F, [1, 1, 1, 30], list(range(30)))
    c30 = g.c(F, [1, 1, 1, 1], [30.0])

    rowhas = g.nd("ReduceMax", [inp], axes=[3], keepdims=1)          # [1,10,30,1]
    colhas = g.nd("ReduceMax", [inp], axes=[2], keepdims=1)          # [1,10,1,30]
    ymin = g.nd("Sub", [c30, g.nd("ReduceMax",
                [g.nd("Mul", [rowhas, g.nd("Sub", [c30, y30])])], axes=[2], keepdims=1)])
    ymax = g.nd("ReduceMax", [g.nd("Mul", [rowhas, y30])], axes=[2], keepdims=1)
    xmin = g.nd("Sub", [c30, g.nd("ReduceMax",
                [g.nd("Mul", [colhas, g.nd("Sub", [c30, x30])])], axes=[3], keepdims=1)])
    xmax = g.nd("ReduceMax", [g.nd("Mul", [colhas, x30])], axes=[3], keepdims=1)
    one = g.c(F, [1, 1, 1, 1], [1.0])
    hc = g.nd("Add", [g.nd("Sub", [ymax, ymin]), one])              # [1,10,1,1]
    wc = g.nd("Add", [g.nd("Sub", [xmax, xmin]), one])

    counts = g.nd("ReduceSum", [inp], axes=[2, 3], keepdims=1)      # [1,10,1,1]
    eVecBg = _eqm_f(g, counts, g.nd("ReduceMax", [counts], axes=[1], keepdims=1))
    present = g.nd("ReduceMax", [rowhas], axes=[2], keepdims=1)
    pfg = g.nd("Mul", [present, g.nd("Sub", [one, eVecBg])])        # fg present mask

    maxh = g.nd("ReduceMax", [g.nd("Mul", [hc, pfg])], axes=[1], keepdims=1)  # [1,1,1,1]
    maxw = g.nd("ReduceMax", [g.nd("Mul", [wc, pfg])], axes=[1], keepdims=1)
    hf = g.c(F, [1, 1, 1, 1], [0.5])
    offr = g.nd("Mul", [g.nd("Sub", [maxh, hc]), hf])               # [1,10,1,1]
    offc = g.nd("Mul", [g.nd("Sub", [maxw, wc]), hf])
    shiftr = g.nd("Sub", [ymin, offr])
    shiftc = g.nd("Sub", [xmin, offc])

    krow = g.c(F, [1, 1, _WC, 1], list(range(_WC)))
    kcol = g.c(F, [1, 1, 1, _WC], list(range(_WC)))
    ycoord = g.c(F, [1, 1, 1, 30], list(range(30)))
    xcoord = g.c(F, [1, 1, 30, 1], list(range(30)))
    Srow = _eqm_f(g, g.nd("Add", [krow, shiftr]), ycoord)          # [1,10,WC,30]
    Scol = _eqm_f(g, xcoord, g.nd("Add", [kcol, shiftc]))          # [1,10,30,WC]
    tmp = g.nd("MatMul", [inp, Scol])                              # [1,10,30,WC]
    Pc = g.nd("MatMul", [Srow, tmp])                               # [1,10,WC,WC]

    markerhot = g.nd("Mul", [Pc, pfg])                            # [1,10,WC,WC]
    markerany = g.nd("ReduceSum", [markerhot], axes=[1], keepdims=1)  # [1,1,WC,WC]
    cmask = g.nd("Mul", [
        g.nd("Cast", [g.nd("Less", [krow, maxh])], to=F),
        g.nd("Cast", [g.nd("Less", [kcol, maxw])], to=F)])         # [1,1,WC,WC]
    bgfill = g.nd("Mul", [cmask, g.nd("Sub", [one, markerany])])
    bghot = g.nd("Mul", [bgfill, eVecBg])                          # [1,10,WC,WC]
    onehot = g.nd("Add", [markerhot, bghot])
    g.nd("Pad", [onehot], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, 30 - _WC, 30 - _WC])
    return _model(g, "rsc_308")


def _mirror_308(a):
    import collections
    a = np.asarray(a, int)
    if a.ndim != 2 or max(a.shape) > 30:
        return None
    bg = collections.Counter(a.ravel()).most_common(1)[0][0]
    cs = [c for c in range(10) if c != bg and (a == c).any()]
    if not cs:
        return None
    objs = {}
    maxh = maxw = 0
    for c in cs:
        ys, xs = np.where(a == c)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        objs[c] = (ys, xs, h, w)
        maxh = max(maxh, h)
        maxw = max(maxw, w)
    if maxh > _WC or maxw > _WC:
        return None
    out = np.full((maxh, maxw), bg, int)
    for c, (ys, xs, h, w) in objs.items():
        if (maxh - h) % 2 or (maxw - w) % 2:
            return None
        offr = (maxh - h) // 2
        offc = (maxw - w) // 2
        for y, x in zip(ys, xs):
            ny = y - ys.min() + offr
            nx = x - xs.min() + offc
            out[ny, nx] = c
    return out


# --------------------------------------------------------------------------- #
# numpy mirror for detection                                                   #
# --------------------------------------------------------------------------- #
def _mirror_090(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or max(a.shape) > 30 or a.shape[0] > _HP:
        return None
    H, W = a.shape
    Z = (a == 0).astype(int)
    ps = np.zeros((H + 1, W + 1), int)
    ps[1:, 1:] = np.cumsum(np.cumsum(Z, 0), 1)

    def wsum(i, j, h, w):
        return ps[i + h, j + w] - ps[i, j + w] - ps[i + h, j] + ps[i, j]

    tl = {}
    winarea = 0
    for h in _HCAND:
        for w in _WCAND:
            if h > H or w > W:
                continue
            for i in range(H - h + 1):
                for j in range(W - w + 1):
                    if wsum(i, j, h, w) == h * w:
                        tl.setdefault((h, w), []).append((i, j))
                        if h * w > winarea:
                            winarea = h * w
    out = a.copy()
    if winarea > 0:
        for (h, w), lst in tl.items():
            if h * w == winarea:
                for (i, j) in lst:
                    out[i:i + h, j:j + w] = 6
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
    if _matches(prs, _mirror_090):
        try:
            yield ("rsc_3eda0437", build_090())
        except Exception:
            pass
    if _matches(prs, _mirror_153):
        try:
            yield ("rsc_681b3aeb", build_153())
        except Exception:
            pass
    if _matches(prs, _mirror_308):
        try:
            yield ("rsc_c8cbb738", build_308())
        except Exception:
            pass
