"""family_r233 — static ONNX for task233 (verify_97a05b5b).

Rule (see family_vc3_1 for the exact numpy reference solve233, 266/266):
a solid sqc(=2) square (<=20x20) has bgc(=0) "anti-shape" holes carved in the
orientations of scattered 3x3 markers (unique minority colour `col` per marker,
obj cells left sqc, base orientation).  Output = the square cropped top-left with
every hole filled by the matching marker's 3x3 (obj->sqc, rest->col) in the hole's
orientation.  Placement is a FORCED exact cover.

This module builds the whole thing as a static feed-forward graph:
  * per-colour SLOT unroll (8 fixed slots, colours [1,3,4,5,6,7,8,9]); marker 3x3
    kernels recovered by a bounded 8-flood + MatMul crop;
  * D4 all-8 orientations via Rev/Transpose;
  * K fixed constraint-propagation passes: per (slot,orientation) correlation Conv
    counts exact obj-matches vs the REMAINING hole mask; a slot is CLAIMED when its
    total match count equals its obj's free-shape D4 stabiliser size s (<=> exactly
    one distinct covered cell-set); claimed obj cells subtracted, col frame stamped
    (ConvTranspose) using the canonical-FIRST matched orientation (tie-break);
  * square bbox + top-left crop via dyncrop MatMul; one-hot + ingrid clip.

The detection gate (`_ref`, mirror of the graph numerics) is validated EXACT on
train+test+arc-gen before the model is emitted.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
FT = onnx.TensorProto.FLOAT16   # fp16 intermediates (all values are small exact ints)
INT64 = onnx.TensorProto.INT64
H30 = 30
COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
S = 8   # slots
O = 8   # orientations
NPASS = 6


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
            n, FT, list(dims), [float(v) for v in np.asarray(vals, np.float64).ravel()]))
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


# elementwise helpers ------------------------------------------------------- #
def gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=FT)


def lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=FT)


def le(g, a, b):
    return lt(g, a, g.nd("Add", [b, g.half]))


def ge(g, a, b):
    return gt(g, a, g.nd("Sub", [b, g.half]))


def eqm(g, a, b):
    return lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


def mul(g, *xs):
    r = xs[0]
    for x in xs[1:]:
        r = g.nd("Mul", [r, x])
    return r


def rs(g, x, shape):
    return g.nd("Reshape", [x, g.i64(shape)])


def _minmax(g, has, idx, axis):
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], axes=[axis], keepdims=1)
    inv = g.nd("Mul", [has, g.nd("Sub", [g.cbig, idx])])
    mn = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [inv], axes=[axis], keepdims=1)])
    return mn, mx


# --------------------------------------------------------------------------- #
def build233():
    g = _G()
    g.half = g.f1(0.5)
    g.one = g.f1(1.0)
    g.cbig = g.f1(1000.0)
    g.two = g.f1(2.0)
    ri = g.f([1, 1, H30, 1], list(range(H30)))     # rowidx
    ci = g.f([1, 1, 1, H30], list(range(H30)))     # colidx
    Rev = g.f([1, 1, 3, 3], np.fliplr(np.eye(3)))
    ri3 = g.f([1, 1, 3, 1], [0, 1, 2])
    ci3 = g.f([1, 1, 1, 3], [0, 1, 2])
    chvals = g.f([1, 10, 1, 1], list(range(10)))

    # value image, key channels ------------------------------------------
    xin = g.nd("Cast", ["input"], to=FT)                               # fp16 one-hot
    V = g.nd("Conv", [xin, chvals], kernel_shape=[1, 1])              # [1,1,30,30]
    ch2 = g.nd("Slice", [xin, g.i64([2]), g.i64([3]), g.i64([1])])     # [1,1,30,30]
    # per-slot col masks via a select conv
    selW = np.zeros((S, 10, 1, 1), np.float32)
    for k, c in enumerate(COLORS):
        selW[k, c, 0, 0] = 1.0
    colc = g.nd("Conv", [xin, g.f([S, 10, 1, 1], selW)], kernel_shape=[1, 1])  # [1,8,30,30]
    thru = g.nd("Add", [colc, ch2])                                   # [1,8,30,30]

    # bounded 8-flood: block masks per slot ------------------------------
    m = colc
    for _ in range(3):
        dil = g.nd("MaxPool", [m], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
        m = g.nd("Min", [dil, thru])
    block = gt(g, m, g.half)                                          # [1,8,30,30] 0/1
    objc = g.nd("Mul", [block, ch2])                                  # [1,8,30,30]
    objunion = gt(g, g.nd("ReduceSum", [objc], axes=[1], keepdims=1), g.half)  # [1,1,30,30]
    present = g.nd("ReduceMax", [colc], axes=[2, 3], keepdims=1)      # [1,8,1,1]

    # marker 3x3 kernels K3 [1,8,3,3] via MatMul crop --------------------
    rowhas = g.nd("ReduceMax", [block], axes=[3], keepdims=1)         # [1,8,30,1]
    colhas = g.nd("ReduceMax", [block], axes=[2], keepdims=1)         # [1,8,1,30]
    minr, _ = _minmax(g, rowhas, ri, 2)                              # [1,8,1,1]
    minc, _ = _minmax(g, colhas, ci, 3)
    tgt_i = g.nd("Add", [minr, ri3])                                 # [1,8,3,1]
    Srow = eqm(g, ci, tgt_i)                                         # [1,8,3,30]
    tgt_j = g.nd("Add", [minc, ci3])                                 # [1,8,1,3]
    Scol = eqm(g, ri, tgt_j)                                         # [1,8,30,3]
    Vblock = g.nd("Mul", [V, block])                                 # [1,8,30,30]
    K3 = g.nd("MatMul", [Srow, g.nd("MatMul", [Vblock, Scol])])      # [1,8,3,3]

    # D4 orientations ----------------------------------------------------
    def T(x):
        return g.nd("Transpose", [x], perm=[0, 1, 3, 2])

    def Hf(x):
        return g.nd("MatMul", [Rev, x])

    def Wf(x):
        return g.nd("MatMul", [x, Rev])

    def orient(x, o):
        if o == 0:
            return x
        if o == 1:
            return Hf(T(x))
        if o == 2:
            return Hf(Wf(x))
        if o == 3:
            return Wf(T(x))
        if o == 4:
            return Wf(x)
        if o == 5:
            return Wf(Hf(T(x)))
        if o == 6:
            return Hf(x)
        return T(x)                                                  # o == 7

    obj5_parts, col5_parts = [], []
    for o in range(O):
        Ko = orient(K3, o)                                          # [1,8,3,3]
        objko = eqm(g, Ko, g.two)                                   # [1,8,3,3]
        colko = g.nd("Mul", [Ko, g.nd("Sub", [g.one, objko])])     # value at col cells
        obj5_parts.append(rs(g, objko, [1, S, 1, 3, 3]))
        col5_parts.append(rs(g, colko, [1, S, 1, 3, 3]))
    OBJ5 = g.nd("Concat", obj5_parts, axis=2)                       # [1,8,8,3,3]
    COL5 = g.nd("Concat", col5_parts, axis=2)                       # [1,8,8,3,3]
    OBJw = rs(g, OBJ5, [S * O, 1, 3, 3])                            # conv weight
    objk0 = eqm(g, K3, g.two)                                       # base obj [1,8,3,3]
    nobj = g.nd("ReduceSum", [objk0], axes=[2, 3], keepdims=1)      # [1,8,1,1]
    nobj5 = rs(g, nobj, [1, S, 1, 1, 1])

    # free-shape stabiliser size s per slot ------------------------------
    plane = g.nd("Pad", [objk0], mode="constant", value=0.0,
                 pads=[0, 0, 2, 2, 0, 0, 2, 2])                     # [1,8,7,7]
    plane64 = g.nd("Tile", [rs(g, plane, [1, S, 1, 7, 7]),
                            g.i64([1, 1, O, 1, 1])])                # [1,8,8,7,7]
    plane64 = rs(g, plane64, [S * O, 1, 7, 7])
    acorr = g.nd("Conv", [rs(g, plane64, [1, S * O, 7, 7]), OBJw],
                 kernel_shape=[3, 3], group=S * O, pads=[2, 2, 2, 2])  # [1,64,9,9]
    amax = g.nd("ReduceMax", [acorr], axes=[2, 3], keepdims=1)      # [1,64,1,1]
    sind = eqm(g, rs(g, amax, [1, S, O, 1, 1]), nobj5)             # [1,8,8,1,1]
    sfree = g.nd("ReduceSum", [sind], axes=[2], keepdims=0)        # [1,8,1,1]

    # square bbox --------------------------------------------------------
    sq2 = g.nd("Mul", [ch2, g.nd("Sub", [g.one, objunion])])       # [1,1,30,30]
    R0, R1 = _minmax(g, g.nd("ReduceMax", [sq2], axes=[3], keepdims=1), ri, 2)
    C0, C1 = _minmax(g, g.nd("ReduceMax", [sq2], axes=[2], keepdims=1), ci, 3)
    # window mask: topleft oy in [R0,R1-2], ox in [C0,C1-2]
    R1m2 = g.nd("Sub", [R1, g.two])
    C1m2 = g.nd("Sub", [C1, g.two])
    winmask = mul(g, ge(g, ri, R0), le(g, ri, R1m2),
                  ge(g, ci, C0), le(g, ci, C1m2))                   # [1,1,30,30]

    # priority weights for canonical-first orientation
    prio5 = g.f([1, 1, O, 1, 1], [O - o for o in range(O)])

    remaining = g.nd("Mul", [g.nd("Sub", [g.one, ch2]),
                             mul(g, ge(g, ri, R0), le(g, ri, R1),
                                 ge(g, ci, C0), le(g, ci, C1))])     # holes inside bbox [1,1,30,30]
    # (V==0 inside bbox); ch2==1 are sqc so 1-ch2 excludes them; holes are the 0-cells.
    placed = g.f([1, S, 1, 1], [0] * S)
    colval = g.f([1, 1, H30, H30], [0] * (H30 * H30))

    for _p in range(NPASS):
        in64 = g.nd("Tile", [remaining, g.i64([1, S * O, 1, 1])])   # [1,64,30,30]
        corr = g.nd("Conv", [in64, OBJw], kernel_shape=[3, 3],
                    group=S * O, pads=[0, 0, 2, 2])                 # [1,64,30,30]
        full5 = g.nd("Mul", [eqm(g, rs(g, corr, [1, S, O, H30, H30]), nobj5),
                             rs(g, winmask, [1, 1, 1, H30, H30])])  # [1,8,8,30,30]
        cnt = g.nd("ReduceSum", [full5], axes=[3, 4], keepdims=0)   # [1,8,8,1]? -> reduce spatial
        cnt = g.nd("ReduceSum", [g.nd("ReduceSum", [full5], axes=[4], keepdims=0)],
                   axes=[3], keepdims=0)                            # [1,8,8]
        total = g.nd("ReduceSum", [cnt], axes=[2], keepdims=0)      # [1,8]
        total = rs(g, total, [1, S, 1, 1])
        unique = mul(g, eqm(g, total, sfree), gt(g, total, g.half),
                     gt(g, present, g.half), g.nd("Sub", [g.one, placed]))  # [1,8,1,1]

        # claimed obj footprint --------------------------------------
        u5 = rs(g, unique, [1, S, 1, 1, 1])
        claim64 = rs(g, g.nd("Mul", [full5, u5]), [1, S * O, H30, H30])
        objfoot = g.nd("ConvTranspose", [claim64, OBJw], kernel_shape=[3, 3],
                       group=S * O, strides=[1, 1], pads=[0, 0, 2, 2])  # [1,64,30,30]
        objfoot = gt(g, g.nd("ReduceSum", [objfoot], axes=[1], keepdims=1), g.half)  # [1,1,30,30]
        remaining = g.nd("Mul", [remaining, g.nd("Sub", [g.one, objfoot])])

        # canonical-first col stamp ----------------------------------
        hasm = gt(g, rs(g, cnt, [1, S, O, 1, 1]), g.half)          # [1,8,8,1,1]
        score = g.nd("Mul", [hasm, prio5])
        mx = g.nd("ReduceMax", [score], axes=[2], keepdims=1)      # [1,8,1,1,1]
        winsel = mul(g, eqm(g, score, mx), hasm, u5)              # [1,8,8,1,1]
        matchsel = g.nd("ReduceSum", [g.nd("Mul", [full5, winsel])], axes=[2], keepdims=0)  # [1,8,30,30]
        colsel = g.nd("ReduceSum", [g.nd("Mul", [COL5, winsel])], axes=[2], keepdims=0)     # [1,8,3,3]
        colsel = rs(g, colsel, [S, 1, 3, 3])
        stamp = g.nd("ConvTranspose", [matchsel, colsel], kernel_shape=[3, 3],
                     group=S, strides=[1, 1], pads=[0, 0, 2, 2])   # [1,8,30,30]
        colstamp = g.nd("ReduceSum", [stamp], axes=[1], keepdims=1)  # [1,1,30,30]
        colval = g.nd("Add", [colval, colstamp])

        placed = g.nd("Add", [placed, unique])

    # build value image over square, crop top-left, one-hot --------------
    inside = mul(g, ge(g, ri, R0), le(g, ri, R1), ge(g, ci, C0), le(g, ci, C1))  # [1,1,30,30]
    colpres = gt(g, colval, g.half)
    OUTv = g.nd("Mul", [inside,
                        g.nd("Add", [colval,
                                     g.nd("Mul", [g.two, g.nd("Sub", [g.one, colpres])])])])
    sgh = g.nd("Add", [g.nd("Sub", [R1, R0]), g.one])
    sgw = g.nd("Add", [g.nd("Sub", [C1, C0]), g.one])
    Rrow = g.nd("Mul", [eqm(g, ci, g.nd("Add", [ri, R0])), lt(g, ri, sgh)])   # [1,1,30,30]
    Rcol = g.nd("Mul", [eqm(g, ri, g.nd("Add", [ci, C0])), lt(g, ci, sgw)])
    cropped = g.nd("MatMul", [Rrow, g.nd("MatMul", [OUTv, Rcol])])            # [1,1,30,30]
    ingrid = g.nd("Mul", [lt(g, ri, sgh), lt(g, ci, sgw)])
    OH = eqm(g, cropped, chvals)
    outf16 = g.nd("Mul", [OH, ingrid])
    g.nd("Cast", [outf16], "output", to=F)
    return _model(g, "r233")


# =========================================================================== #
# numpy reference (mirror of the graph numerics) — detection gate             #
# =========================================================================== #
def _orient3(m, o):
    return np.rot90(m, o) if o < 4 else np.fliplr(np.rot90(m, o - 4))


def _dil8(m):
    p = np.pad(m.astype(bool), 1)
    out = np.zeros_like(m, bool)
    for di in range(3):
        for dj in range(3):
            out |= p[di:di + m.shape[0], dj:dj + m.shape[1]]
    return out


def _tight(x):
    ys, xs = np.where(x)
    return x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _ref(a):
    V = np.asarray(a, int)
    if V.ndim != 2 or max(V.shape) > 30:
        return None
    H, W = V.shape
    kernels = {}
    objunion = np.zeros((H, W), bool)
    for c in COLORS:
        colc = (V == c)
        if not colc.any():
            continue
        thru = (V == 2) | (V == c)
        m = colc.copy()
        for _ in range(3):
            m = _dil8(m) & thru
        objunion |= m & (V == 2)
        ys, xs = np.where(m)
        if ys.max() - ys.min() > 2 or xs.max() - xs.min() > 2:
            return None
        K = np.zeros((3, 3), int)
        K[ys - ys.min(), xs - xs.min()] = V[ys, xs]
        kernels[c] = K
    if not kernels:
        return None
    sq2 = (V == 2) & ~objunion
    if not sq2.any():
        return None
    ys, xs = np.where(sq2)
    R0, R1, C0, C1 = ys.min(), ys.max(), xs.min(), xs.max()
    sgh, sgw = R1 - R0 + 1, C1 - C0 + 1
    inside = np.zeros((H, W), bool)
    inside[R0:R1 + 1, C0:C1 + 1] = True
    remaining = (V == 0) & inside
    info = {}
    for c, K in kernels.items():
        objb = (K == 2)
        tb = _tight(objb)
        s = sum(1 for o in range(8)
                if _orient3(objb, o).shape == objb.shape and np.array_equal(_tight(_orient3(objb, o)), tb))
        info[c] = s
    colimg = np.zeros((H, W), int)
    placed = set()
    for _p in range(NPASS):
        claimed = []
        for c, K in kernels.items():
            if c in placed:
                continue
            wins = []
            for o in range(8):
                objk = (_orient3(K, o) == 2)
                rel = list(zip(*np.where(objk)))
                ws = [(oy, ox) for oy in range(R0, R1 - 1) for ox in range(C0, C1 - 1)
                      if all(remaining[oy + ry, ox + rx] for ry, rx in rel)]
                wins.append(ws)
            tot = sum(len(w) for w in wins)
            if tot > 0 and tot == info[c]:
                claimed.append((c, K, wins))
        if not claimed:
            break
        for c, K, wins in claimed:
            placed.add(c)
            for o in range(8):
                ob = _orient3(K, o)
                for oy, ox in wins[o]:
                    for ry in range(3):
                        for rx in range(3):
                            if ob[ry, rx] == 2:
                                remaining[oy + ry, ox + rx] = False
            fo = next(o for o in range(8) if wins[o])
            oy, ox = wins[fo][0]
            ob = _orient3(K, fo)
            for ry in range(3):
                for rx in range(3):
                    if ob[ry, rx] != 2:
                        colimg[oy + ry, ox + rx] = c
    if len(placed) != len(kernels) or remaining.any():
        return None
    out = np.full((sgh, sgw), 2, int)
    ci = colimg[R0:R1 + 1, C0:C1 + 1]
    out[ci != 0] = ci[ci != 0]
    return out


# =========================================================================== #
def candidates(examples):
    prs = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return []
            if max(a.shape) > 30 or max(b.shape) > 30:
                return []
            prs.append((a, b))
    if not prs:
        return []
    for a, b in prs:
        try:
            o = _ref(a)
        except Exception:
            return []
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return []
    try:
        return [("r233_97a05b5b", build233())]
    except Exception:
        return []
