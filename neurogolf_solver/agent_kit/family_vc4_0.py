"""family_vc4_0 — task233 (verify_97a05b5b): marker-hole exact-cover stamp.

Rule (bgc=0, sqc=2): a solid sqc square has several `obj`-shaped holes (bgc)
carved into it; scattered 3x3 markers (unique colour `col` + `obj` cells drawn in
sqc) sit in the background.  Output = the square cropped to the top-left, every
hole filled to sqc and its matching marker's full 3x3 stamped so obj aligns onto
the hole (obj->sqc, the col border painted `col`).

The assignment is a global exact-cover but is ALWAYS FORCED, so it is realised by
K fixed constraint-propagation passes: each pass, for every colour slot, count
exact full-3x3-window matches (obj cells on the REMAINING holes, col cells on the
square's sqc) over all 8 D4 orientations; a marker whose covered obj-cell-set is
UNIQUE (union size == |obj|) claims those cells (removed from the remaining holes)
and paints its border with the first orientation in the canonical D4 order.

Static ONNX construction (verified exact vs `_ref233`, the numpy mirror of the
graph numerics, 266/266):
  * value image V; sqmask=(V==2); square found by 3x3 erosion (zero-padded min-
    pool) -> interior seeds -> 8-conn flood within sqmask -> bbox (minr..maxr,..);
  * per colour slot c: extract the marker 3x3 by a bounded flood from the (V==c)
    seeds through the outside-square non-bg mask; centred 3x3 obj/col kernels via
    dyncrop MatMul + 3x3 slice; 8 D4 orientations via fixed 9x9 permutation MatMul;
  * matching by correlation Conv (obj vs remaining holes, col vs square sqc) ==
    counts; obj cover / border stamp by ConvTranspose; union-size uniqueness test;
  * remaining-hole mask + painted-colour image threaded through 6 passes x 8 slots;
  * final square cropped to origin by dyncrop MatMul, one-hot, in-grid clip.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64
H30 = 30
SLOTS = [1, 3, 4, 5, 6, 7, 8, 9]
_D4 = [("r", 0), ("r", 1), ("r", 2), ("r", 3),
       ("f", 0), ("f", 1), ("f", 2), ("f", 3)]
K_PASS = 6


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
    g.half = g.f([1, 1, 1, 1], [0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])
    g.cbig = g.f([1, 1, 1, 1], [1000.0])


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _eqm(g, a, b):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


def _sub1(g, a):
    return g.nd("Sub", [g.one, a])


def _minmax(g, has, idx, axis):
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], axes=[axis], keepdims=1)
    inv = g.nd("Mul", [has, g.nd("Sub", [g.cbig, idx])])
    mn = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [inv], axes=[axis], keepdims=1)])
    return mn, mx


def _value_img(g):
    return g.nd("Conv", ["input", g.f([1, 10, 1, 1], list(range(10)))], kernel_shape=[1, 1])


def _rowmin(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)
    return _minmax(g, has, g.rowidx, 2)


def _colmin(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)
    return _minmax(g, has, g.colidx, 3)


def _crop_matmul(g, src, r0, c0, h, w):
    Rrow = g.nd("Mul", [_eqm(g, g.colidx, g.nd("Add", [g.rowidx, r0])),
                        _lt(g, g.rowidx, g.nd("Sub", [h, g.half]))])
    Rcol = g.nd("Mul", [_eqm(g, g.rowidx, g.nd("Add", [g.colidx, c0])),
                        _lt(g, g.colidx, g.nd("Sub", [w, g.half]))])
    return g.nd("MatMul", [Rrow, g.nd("MatMul", [src, Rcol])])


def _maxpool3(g, x):
    return g.nd("MaxPool", [x], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])


def _perm_mat(key):
    base = np.arange(9).reshape(3, 3)
    t, k = key
    r = np.rot90(base, k)
    r = np.fliplr(r) if t == "f" else r
    oriented = r.ravel()          # oriented[o] = source flat index
    P = np.zeros((9, 9), np.float32)
    for o in range(9):
        P[oriented[o], o] = 1.0
    return P


def _orient_kernel(g, K3, Pk):
    flat = g.nd("Reshape", [K3, g.i64([1, 9])])
    o = g.nd("MatMul", [flat, Pk])
    return g.nd("Reshape", [o, g.i64([1, 1, 3, 3])])


# =========================================================================== #
# builder                                                                     #
# =========================================================================== #
def build_233():
    g = _G()
    _consts(g)
    three = g.f1(3.0)
    two = g.f1(2.0)

    V = _value_img(g)                                    # [1,1,30,30]
    sqmask = _eqm(g, V, g.f1(2.0))
    nonbg = _gt(g, V, g.half)

    # ---- find the square: erosion (zero-pad min-pool) -> flood -------------
    comp = _sub1(g, sqmask)
    comp_p = g.nd("Pad", [comp], mode="constant", value=1.0,
                  pads=[0, 0, 1, 1, 0, 0, 1, 1])          # outside sqmask = 0
    dil = g.nd("MaxPool", [comp_p], kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    seeds = _sub1(g, dil)                                 # eroded sqmask
    sq = seeds
    for _ in range(30):
        sq = g.nd("Min", [_maxpool3(g, sq), sqmask])
    minr, maxr = _rowmin(g, sq)
    minc, maxc = _colmin(g, sq)
    Hh = g.nd("Add", [g.nd("Sub", [maxr, minr]), g.one])
    Ww = g.nd("Add", [g.nd("Sub", [maxc, minc]), g.one])

    inrow = g.nd("Mul", [_lt(g, g.nd("Sub", [minr, g.half]), g.rowidx),
                         _lt(g, g.rowidx, g.nd("Add", [maxr, g.half]))])   # [1,1,30,1]
    incol = g.nd("Mul", [_lt(g, g.nd("Sub", [minc, g.half]), g.colidx),
                         _lt(g, g.colidx, g.nd("Add", [maxc, g.half]))])   # [1,1,1,30]
    inside = g.nd("Mul", [inrow, incol])                                   # [1,1,30,30]

    holemask = g.nd("Mul", [_eqm(g, V, g.f1(0.0)), inside])
    sqc_region = g.nd("Mul", [sqmask, inside])
    outside = _sub1(g, inside)
    region = g.nd("Mul", [nonbg, outside])

    Pk = [g.f([9, 9], _perm_mat(key)) for key in _D4]

    # ---- per-slot precompute (pass-independent) ---------------------------
    slots = []
    for c in SLOTS:
        colseed = _eqm(g, V, g.f1(float(c)))
        present = _gt(g, g.nd("ReduceSum", [colseed], axes=[2, 3], keepdims=1), g.half)
        blk = colseed
        for _ in range(2):
            blk = g.nd("Min", [_maxpool3(g, blk), region])
        objm = g.nd("Mul", [sqmask, blk])
        colm = colseed
        mr0, _ = _rowmin(g, blk)
        mc0, _ = _colmin(g, blk)
        Kobj = g.nd("Slice", [_crop_matmul(g, objm, mr0, mc0, three, three),
                              g.i64([0, 0]), g.i64([3, 3]), g.i64([2, 3])])
        Kcol = g.nd("Slice", [_crop_matmul(g, colm, mr0, mc0, three, three),
                              g.i64([0, 0]), g.i64([3, 3]), g.i64([2, 3])])
        objcnt = g.nd("ReduceSum", [Kobj], axes=[2, 3], keepdims=1)
        colcnt = g.nd("ReduceSum", [Kcol], axes=[2, 3], keepdims=1)
        present = g.nd("Mul", [present, _gt(g, objcnt, g.half)])
        oris = []
        for k in range(8):
            Ok = _orient_kernel(g, Kobj, Pk[k])
            Ck = _orient_kernel(g, Kcol, Pk[k])
            bd = g.nd("Conv", [sqc_region, Ck], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
            bdmatch = _eqm(g, bd, colcnt)
            oris.append((Ok, Ck, bdmatch))
        slots.append((c, present, objcnt, oris))

    # ---- propagation: thread (remaining, paintv) --------------------------
    remaining = holemask
    paintv = g.nd("Mul", [inside, g.f1(0.0)])            # [1,1,30,30] zeros
    for _p in range(K_PASS):
        for (c, present, objcnt, oris) in slots:
            mmlist = []
            U = None
            for (Ok, Ck, bdmatch) in oris:
                co = g.nd("Conv", [remaining, Ok], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
                mm = g.nd("Mul", [_eqm(g, co, objcnt), bdmatch])
                mmlist.append(mm)
                os_ = _gt(g, g.nd("ConvTranspose", [mm, Ok], kernel_shape=[3, 3],
                                  strides=[1, 1], pads=[1, 1, 1, 1]), g.half)
                U = os_ if U is None else g.nd("Max", [U, os_])
            usize = g.nd("ReduceSum", [U], axes=[2, 3], keepdims=1)
            unique = g.nd("Mul", [present, _eqm(g, usize, objcnt)])       # [1,1,1,1]
            remaining = g.nd("Mul", [remaining, _sub1(g, g.nd("Mul", [U, unique]))])
            # first-orientation border
            prev_any = g.f([1, 1, 1, 1], [0.0])
            border = None
            for k in range(8):
                mm = mmlist[k]
                Ck = oris[k][1]
                has = _gt(g, g.nd("ReduceSum", [mm], axes=[2, 3], keepdims=1), g.half)
                take = g.nd("Mul", [has, _sub1(g, prev_any)])
                bst = _gt(g, g.nd("ConvTranspose", [mm, Ck], kernel_shape=[3, 3],
                                  strides=[1, 1], pads=[1, 1, 1, 1]), g.half)
                contrib = g.nd("Mul", [bst, take])
                border = contrib if border is None else g.nd("Add", [border, contrib])
                prev_any = _sub1(g, g.nd("Mul", [_sub1(g, prev_any), _sub1(g, has)]))
            bpaint = g.nd("Mul", [border, unique])                        # [1,1,30,30]
            paintv = g.nd("Add", [g.nd("Mul", [paintv, _sub1(g, bpaint)]),
                                  g.nd("Mul", [bpaint, g.f1(float(c))])])

    # ---- assemble output value image & crop -------------------------------
    paintmask = _gt(g, paintv, g.half)
    OUTv = g.nd("Add", [g.nd("Mul", [g.nd("Mul", [inside, two]), _sub1(g, paintmask)]),
                        paintv])
    RV = _crop_matmul(g, OUTv, minr, minc, Hh, Ww)
    OH = _eqm(g, RV, g.f([1, 10, 1, 1], list(range(10))))
    ingrid = g.nd("Mul", [_lt(g, g.rowidx, g.nd("Sub", [Hh, g.half])),
                          _lt(g, g.colidx, g.nd("Sub", [Ww, g.half]))])
    g.nd("Mul", [OH, ingrid], "output")
    return _model(g, "vc4_233")


# =========================================================================== #
# numpy mirror of the graph numerics (detection gate)                          #
# =========================================================================== #
def _dil3(m):
    p = np.pad(m, 1)
    o = np.zeros_like(m)
    for di in range(3):
        for dj in range(3):
            o = np.maximum(o, p[di:di + m.shape[0], dj:dj + m.shape[1]])
    return o


def _ero3(m):
    p = np.pad(m, 1)
    o = np.ones_like(m)
    for di in range(3):
        for dj in range(3):
            o = np.minimum(o, p[di:di + m.shape[0], dj:dj + m.shape[1]])
    return o


def _orient(m, key):
    t, k = key
    r = np.rot90(m, k)
    return np.fliplr(r) if t == "f" else r


def _corr(field, K):
    fp = np.pad(field, 1)
    H, W = field.shape
    o = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            o[i, j] = (K * fp[i:i + 3, j:j + 3]).sum()
    return o


def _stamp(anchors, K):
    H, W = anchors.shape
    o = np.zeros((H, W))
    ys, xs = np.where(anchors > 0.5)
    for i, j in zip(ys, xs):
        for ki in range(3):
            for kj in range(3):
                y, x = i + ki - 1, j + kj - 1
                if 0 <= y < H and 0 <= x < W:
                    o[y, x] += K[ki, kj]
    return o


def _ref233(a, K_PASS=K_PASS):
    a = np.asarray(a, int)
    if a.ndim != 2 or max(a.shape) > 30:
        return None
    H0, W0 = a.shape
    V = np.zeros((30, 30), int)
    V[:H0, :W0] = a
    sqmask = (V == 2).astype(int)
    nonbg = (V > 0).astype(int)
    sq = _ero3(sqmask)
    for _ in range(30):
        sq = np.minimum(_dil3(sq), sqmask)
    if sq.sum() == 0:
        return None
    ys, xs = np.where(sq > 0)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    inside = np.zeros((30, 30), int)
    inside[r0:r1 + 1, c0:c1 + 1] = 1
    holemask = ((V == 0) & (inside > 0)).astype(int)
    sqc_region = ((V == 2) & (inside > 0)).astype(int)
    H, W = r1 - r0 + 1, c1 - c0 + 1
    region = nonbg * (1 - inside)

    slot = []
    for c in SLOTS:
        colseed = (V == c).astype(int)
        if colseed.sum() == 0:
            slot.append(None)
            continue
        blk = colseed.copy()
        for _ in range(2):
            blk = np.minimum(_dil3(blk), region)
        if blk.sum() == 0:
            slot.append(None)
            continue
        objm = ((V == 2) & (blk > 0)).astype(int)
        yy, xx = np.where(blk > 0)
        mr0, mc0 = yy.min(), xx.min()
        Kobj = np.zeros((3, 3))
        Kcol = np.zeros((3, 3))
        for ki in range(3):
            for kj in range(3):
                y, x = mr0 + ki, mc0 + kj
                if 0 <= y < 30 and 0 <= x < 30:
                    Kobj[ki, kj] = objm[y, x]
                    Kcol[ki, kj] = colseed[y, x]
        if Kobj.sum() < 0.5:
            slot.append(None)
            continue
        slot.append((c, Kobj, Kcol))

    remaining = holemask.copy()
    paintv = np.zeros((30, 30), int)
    placed = [False] * len(SLOTS)
    for _p in range(K_PASS):
        for si, s in enumerate(slot):
            if s is None or placed[si]:
                continue
            c, Kobj, Kcol = s
            objcount = Kobj.sum()
            colcount = Kcol.sum()
            oris = [(_orient(Kobj, key), _orient(Kcol, key)) for key in _D4]
            U = np.zeros((30, 30))
            mmlist = []
            for Ko, Kc in oris:
                mm = ((np.abs(_corr(remaining, Ko) - objcount) < 0.5) &
                      (np.abs(_corr(sqc_region, Kc) - colcount) < 0.5)).astype(int)
                mmlist.append(mm)
                U = np.maximum(U, (_stamp(mm, Ko) > 0.5).astype(int))
            unique = abs(U.sum() - objcount) < 0.5
            if not unique:
                continue
            remaining = (remaining * (1 - U)).astype(int)
            bstamp = np.zeros((30, 30), int)
            any_prev = 0
            for (Ko, Kc), mm in zip(oris, mmlist):
                has = 1 if mm.sum() > 0 else 0
                if has and not any_prev:
                    bstamp = (_stamp(mm, Kc) > 0.5).astype(int)
                any_prev = max(any_prev, has)
            paintv[bstamp > 0] = c
            placed[si] = True
    if not all(p for p, s in zip(placed, slot) if s is not None):
        return None
    OUTv = np.zeros((30, 30), int)
    OUTv[inside > 0] = 2
    OUTv[paintv > 0] = paintv[paintv > 0]
    return OUTv[r0:r0 + H, c0:c0 + W]


# =========================================================================== #
# entry point                                                                 #
# =========================================================================== #
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
    if not prs:
        return False
    for a, b in prs:
        try:
            o = _ref233(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(examples):
    prs = _pairs(examples)
    if not _matches(prs):
        return []
    try:
        return [("vc4_97a05b5b", build_233())]
    except Exception:
        return []
