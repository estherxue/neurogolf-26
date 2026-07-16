"""family_vc2_3 — static ONNX solvers (retry wave).

task173 / verify_72322fa7  (bg=0): objects are 2-colour tiny stamps (bbox<=3x3);
    scattered single-colour fragments match ONE of the two colour subparts of a
    template.  At every occurrence of either subpart, paste the full 2-colour
    template aligned.  ONNX: 36 colour-pair pipelines (a<b).  Per pair: isolate the
    (single) 2-colour object by flooding the pair-mask from bi-colour 8-adjacency
    seeds; anchor its bbox to the origin and slice 5x5 runtime kernels Ka (colour-a
    cells) / Kb (colour-b cells); correlation Conv finds occurrences of each subpart
    (count==|K|, off-grid cells auto-fail via zero padding); ConvTranspose stamps
    Ka and Kb back at each occurrence.  Absent pairs self-mask (empty kernels ->
    zero stamps).  Output value image = input overwritten by stamps, one-hot + clip.

Detection gate is a numpy mirror validated 266/266 vs train+test+arc-gen.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64
H30 = 30
KS = 5           # runtime kernel side (template bbox <= 3, margin to 5)
P1 = KS - 1      # correlation / conv-transpose padding


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
    g.nhalf = g.f([1, 1, 1, 1], [-0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])
    g.cbig = g.f([1, 1, 1, 1], [1000.0])


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


def _minmax(g, has, idx, axis):
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], axes=[axis], keepdims=1)
    inv = g.nd("Mul", [has, g.nd("Sub", [g.cbig, idx])])
    mn = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [inv], axes=[axis], keepdims=1)])
    return mn, mx


def _rowmin(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)
    mn, _ = _minmax(g, has, g.rowidx, 2)
    return mn


def _colmin(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)
    mn, _ = _minmax(g, has, g.colidx, 3)
    return mn


def _dil8(g, m):
    return g.nd("MaxPool", [m], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])


def _eqm_scalar(g, a, b):
    """|a-b|<0.5 as float 0/1  (a tensor, b tensor broadcastable)."""
    return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [a, b])]), g.half])], to=F)


def _anchor5(g, X, minr, minc):
    """Shift content so (minr,minc)->origin, then slice to [1,1,5,5]."""
    Srow = _eqm_scalar(g, g.colidx, g.nd("Add", [g.rowidx, minr]))   # [1,1,30,30]
    Scol = _eqm_scalar(g, g.rowidx, g.nd("Add", [g.colidx, minc]))
    A = g.nd("MatMul", [Srow, g.nd("MatMul", [X, Scol])])
    return g.nd("Slice", [A, g.i64([0, 0]), g.i64([KS, KS]), g.i64([2, 3])])


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _rowspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)
    return _minmax(g, has, g.rowidx, 2)


def _colspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)
    return _minmax(g, has, g.colidx, 3)


def _shift(g, M, dy, dx):
    """Translate content by (dy,dx): out[p,q]=M[p-dy,q-dx] (zero fill, drop OOB)."""
    Srow = _eqm_scalar(g, g.colidx, g.nd("Sub", [g.rowidx, dy]))
    Scol = _eqm_scalar(g, g.rowidx, g.nd("Sub", [g.colidx, dx]))
    return g.nd("MatMul", [Srow, g.nd("MatMul", [M, Scol])])


def _upk(g, M, k):
    """Upscale by runtime factor k: out[r,c]=M[floor(r/k),floor(c/k)] (origin kept)."""
    D1 = g.nd("Sub", [g.rowidx, g.nd("Mul", [g.colidx, k])])
    Uk = g.nd("Mul", [_gt(g, D1, g.nhalf),
                      _lt(g, D1, g.nd("Sub", [k, g.half]))])
    D2 = g.nd("Sub", [g.colidx, g.nd("Mul", [g.rowidx, k])])
    UkcT = g.nd("Mul", [_gt(g, D2, g.nhalf),
                        _lt(g, D2, g.nd("Sub", [k, g.half]))])
    return g.nd("MatMul", [Uk, g.nd("MatMul", [M, UkcT])])


def _build_173():
    g = _G()
    _consts(g)
    allsum = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)      # in-grid mask

    # per-colour masks (channels are already one-hot)
    Ms = {c: g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])
          for c in range(1, 10)}

    stampCh = {c: [] for c in range(1, 10)}   # accumulate value-stamps per colour

    for a in range(1, 10):
        for b in range(a + 1, 10):
            Ma, Mb = Ms[a], Ms[b]
            Pab = g.nd("Add", [Ma, Mb])
            # seed: a-cells touching a b, or b-cells touching an a (8-adjacency)
            seed = g.nd("Add", [g.nd("Mul", [Ma, _dil8(g, Mb)]),
                                g.nd("Mul", [Mb, _dil8(g, Ma)])])
            # flood within Pab
            T = seed
            for _ in range(6):
                T = g.nd("Mul", [_dil8(g, T), Pab])
            T = _gt(g, T, g.half)                     # binarise template mask
            SA = g.nd("Mul", [T, Ma])
            SB = g.nd("Mul", [T, Mb])
            minr = _rowmin(g, T)
            minc = _colmin(g, T)
            Ka = _anchor5(g, SA, minr, minc)          # [1,1,5,5]
            Kb = _anchor5(g, SB, minr, minc)
            nA = g.nd("ReduceSum", [Ka], axes=[2, 3], keepdims=1)
            nB = g.nd("ReduceSum", [Kb], axes=[2, 3], keepdims=1)
            # occurrences of subpart a (correlate colour-a mask with Ka)
            corrA = g.nd("Conv", [Ma, Ka], kernel_shape=[KS, KS],
                         pads=[P1, P1, P1, P1])        # [1,1,34,34]
            corrB = g.nd("Conv", [Mb, Kb], kernel_shape=[KS, KS],
                         pads=[P1, P1, P1, P1])
            occA = g.nd("Mul", [_gt(g, corrA, g.nd("Sub", [nA, g.half])),
                                _gt(g, nA, g.half)])
            occB = g.nd("Mul", [_gt(g, corrB, g.nd("Sub", [nB, g.half])),
                                _gt(g, nB, g.half)])
            # stamp full object (Ka -> colour a cells, Kb -> colour b cells)
            for occ in (occA, occB):
                sA = g.nd("ConvTranspose", [occ, Ka], kernel_shape=[KS, KS],
                          pads=[P1, P1, P1, P1])        # [1,1,30,30]
                sB = g.nd("ConvTranspose", [occ, Kb], kernel_shape=[KS, KS],
                          pads=[P1, P1, P1, P1])
                stampCh[a].append(sA)
                stampCh[b].append(sB)

    # combine stamps into a value image, overwrite input, one-hot, clip to grid
    # per colour: binary stamp presence
    stampPres = {}
    for c in range(1, 10):
        s = g.nd("Sum", stampCh[c])
        stampPres[c] = _gt(g, s, g.half)              # [1,1,30,30]
    anyStamp = g.nd("Sum", [stampPres[c] for c in range(1, 10)])
    anyStamp = _gt(g, anyStamp, g.half)
    stampVal = g.nd("Sum", [g.nd("Mul", [stampPres[c], g.f([1, 1, 1, 1], [float(c)])])
                            for c in range(1, 10)])
    # input value image
    wV = g.f([1, 10, 1, 1], list(range(10)))
    V = g.nd("Conv", ["input", wV], kernel_shape=[1, 1])
    Vout = g.nd("Add", [g.nd("Mul", [stampVal, anyStamp]),
                        g.nd("Mul", [V, g.nd("Sub", [g.one, anyStamp])])])
    OH = _eqm_scalar(g, Vout, g.f([1, 10, 1, 1], list(range(10))))   # [1,10,30,30]
    g.nd("Mul", [OH, allsum], "output")
    return _model(g, "vc2_173")


# --------------------------------------------------------------------------- #
# numpy reference (detection gate)                                             #
# --------------------------------------------------------------------------- #
def _dil8_np(m):
    P = np.pad(m, 1)
    out = np.zeros_like(m)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out = out | P[dr:dr + m.shape[0], dc:dc + m.shape[1]]
    return out


def _np_173(a):
    a = np.asarray(a, int)
    Hd, Wd = a.shape
    if Hd > 30 or Wd > 30:
        return None
    out = a.copy()
    for A in range(1, 10):
        for B in range(A + 1, 10):
            Ma = (a == A)
            Mb = (a == B)
            if not Ma.any() or not Mb.any():
                continue
            Pab = Ma | Mb
            seed = (Ma & _dil8_np(Mb)) | (Mb & _dil8_np(Ma))
            if not seed.any():
                continue
            T = seed.copy()
            for _ in range(6):
                T = _dil8_np(T) & Pab
            SA = T & Ma
            SB = T & Mb
            if not SA.any() or not SB.any():
                continue
            ys, xs = np.where(T)
            r0, c0 = ys.min(), xs.min()
            objcells = [(int(a[y, x]), int(y - r0), int(x - c0))
                        for y, x in zip(*np.where(T))]
            for (Smask, col) in [(SA, A), (SB, B)]:
                sy, sx = np.where(Smask)
                sr0, sc0 = sy.min(), sx.min()
                cells = [(int(y - sr0), int(x - sc0)) for y, x in zip(sy, sx)]
                maxi = max(i for i, j in cells)
                maxj = max(j for i, j in cells)
                mask = (a == col)
                oi = sr0 - r0
                oj = sc0 - c0
                for r in range(Hd - maxi):
                    for c in range(Wd - maxj):
                        if all(mask[r + i, c + j] for i, j in cells):
                            for v, ti, tj in objcells:
                                yy = r + ti - oi
                                xx = c + tj - oj
                                if 0 <= yy < Hd and 0 <= xx < Wd:
                                    out[yy, xx] = v
    return out


# =========================================================================== #
# task133 — 57aa92db                                                          #
# =========================================================================== #
def _build_133():
    g = _G()
    _consts(g)
    g.rc = g.f([1, 1, H30, H30],
               [i * H30 + j for i in range(H30) for j in range(H30)])
    g.cbig2 = g.f([1, 1, 1, 1], [1.0e6])
    colorvec = g.f([1, 10, 1, 1], list(range(10)))
    allsum = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)      # in-grid mask
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    NB = g.nd("Sub", [allsum, ch0])                                  # nonbg mask
    V = g.nd("Conv", ["input", colorvec], kernel_shape=[1, 1])       # value image

    # ---- anchor colour x7: colour adjacent to the most distinct colours ---- #
    scores = []
    for c in range(1, 10):
        chc = g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])
        dil = _dil8(g, chc)
        adj = g.nd("ReduceMax", [g.nd("Mul", [dil, "input"])], axes=[2, 3],
                   keepdims=1)                                       # [1,10,1,1]
        adjb = _gt(g, adj, g.half)
        # exclude bg(ch0) and self(ch c)
        sel = [1.0] * 10
        sel[0] = 0.0
        sel[c] = 0.0
        selc = g.f([1, 10, 1, 1], sel)
        sc = g.nd("ReduceSum", [g.nd("Mul", [adjb, selc])], axes=[1], keepdims=1)
        # if colour c absent, force score to -1
        present = _gt(g, g.nd("ReduceSum", [chc], axes=[2, 3], keepdims=1), g.half)
        sc = g.nd("Sub", [g.nd("Mul", [sc, present]),
                          g.nd("Sub", [g.one, present])])
        scores.append(sc)                                            # [1,1,1,1]
    scoreCh = g.nd("Concat", [g.f([1, 1, 1, 1], [-2.0])] + scores, axis=1)  # [1,10,1,1]
    maxsc = g.nd("ReduceMax", [scoreCh], axes=[1], keepdims=1)
    onmax = _eqm_scalar(g, scoreCh, maxsc)                           # ties
    idxv = g.f([1, 10, 1, 1], list(range(10)))
    cand = g.nd("Add", [g.nd("Mul", [idxv, onmax]),
                        g.nd("Mul", [g.cbig, g.nd("Sub", [g.one, onmax])])])
    x7i = g.nd("ReduceMin", [cand], axes=[1], keepdims=1)            # smallest colour
    x7oh = _eqm_scalar(g, idxv, x7i)                                 # [1,10,1,1]
    m7 = g.nd("ReduceSum", [g.nd("Mul", ["input", x7oh])], axes=[1], keepdims=1)
    x7val = g.nd("ReduceSum", [g.nd("Mul", [x7oh, colorvec])], axes=[1], keepdims=1)

    # ---- extract up to 4 connected components (raster-first flood) ---------- #
    comps = []
    rem = NB
    for _ in range(4):
        big = g.nd("Add", [g.nd("Mul", [rem, g.rc]),
                           g.nd("Mul", [g.nd("Sub", [g.one, rem]), g.cbig2])])
        mn = g.nd("ReduceMin", [big], axes=[2, 3], keepdims=1)
        seed = g.nd("Mul", [_eqm_scalar(g, big, mn), rem])
        T = seed
        for _ in range(32):
            T = g.nd("Mul", [_dil8(g, T), rem])
        T = _gt(g, T, g.half)
        comps.append(T)
        rem = g.nd("Mul", [rem, g.nd("Sub", [g.one, T])])

    # ---- template selection: min x7-count, tie max size, tie first slot ----- #
    keys = []
    infos = []
    for i, C in enumerate(comps):
        cnt = g.nd("ReduceSum", [g.nd("Mul", [C, m7])], axes=[2, 3], keepdims=1)
        size = g.nd("ReduceSum", [C], axes=[2, 3], keepdims=1)
        nonempty = _gt(g, size, g.half)
        # key = cnt*1e6 - size*100 + i  (empty -> +inf)
        key = g.nd("Add", [g.nd("Sub", [g.nd("Mul", [cnt, g.cbig2]),
                                        g.nd("Mul", [size, g.f([1, 1, 1, 1], [100.0])])],),
                           g.f([1, 1, 1, 1], [float(i)])])
        key = g.nd("Add", [g.nd("Mul", [key, nonempty]),
                           g.nd("Mul", [g.nd("Sub", [g.one, nonempty]),
                                        g.f([1, 1, 1, 1], [1.0e12])])])
        keys.append(key)
        infos.append((cnt, size, nonempty))
    keycat = g.nd("Concat", keys, axis=1)                           # [1,4,1,1]
    kmin = g.nd("ReduceMin", [keycat], axes=[1], keepdims=1)
    isT = [_eqm_scalar(g, keys[i], kmin) for i in range(4)]         # template gate

    # template mask + normalized template
    Tm = g.nd("Sum", [g.nd("Mul", [comps[i], isT[i]]) for i in range(4)])
    tr0, _ = _rowspan(g, Tm)
    tc0, _ = _colspan(g, Tm)
    Tval = _shift(g, g.nd("Mul", [Tm, V]),
                  g.nd("Sub", [g.nd("Sub", [tr0, tr0]), tr0]),      # dy = -tr0
                  g.nd("Sub", [g.nd("Sub", [tc0, tc0]), tc0]))      # dx = -tc0
    Tanchor = _eqm_scalar(g, Tval, x7val)
    Tanchor = g.nd("Mul", [Tanchor, _gt(g, Tval, g.half)])          # exclude bg
    Tshape = g.nd("Mul", [_gt(g, Tval, g.half),
                          g.nd("Sub", [g.one, _eqm_scalar(g, Tval, x7val)])])
    a7r, _ = _rowspan(g, Tanchor)
    a7c, _ = _colspan(g, Tanchor)

    # ---- per-seed stamping ------------------------------------------------- #
    anchAcc = []
    shapeAcc = []
    for i, C in enumerate(comps):
        gate = g.nd("Mul", [infos[i][2], g.nd("Sub", [g.one, isT[i]])])  # nonempty & !template
        s7 = g.nd("Mul", [C, m7])
        S7r, _ = _rowspan(g, s7)
        S7c, c7max = _colspan(g, s7)
        S7cmin = S7c
        k = g.nd("Add", [g.nd("Sub", [c7max, S7cmin]), g.one])      # width of x7-part
        shp = g.nd("Mul", [C, g.nd("Sub", [g.one, m7])])
        c_i = g.nd("ReduceMax", [g.nd("Mul", [V, shp])], axes=[2, 3], keepdims=1)
        UA = _upk(g, Tanchor, k)
        US = _upk(g, Tshape, k)
        offr = g.nd("Sub", [S7r, g.nd("Mul", [a7r, k])])
        offc = g.nd("Sub", [S7c, g.nd("Mul", [a7c, k])])
        UA = g.nd("Mul", [_shift(g, UA, offr, offc), gate])
        US = g.nd("Mul", [_shift(g, US, offr, offc), gate])
        anchAcc.append(UA)
        shapeAcc.append(g.nd("Mul", [US, c_i]))                     # coloured shape

    anchAll = _gt(g, g.nd("Sum", anchAcc), g.half)                  # x7 cells
    shapeVal = g.nd("Sum", shapeAcc)                                # coloured
    shapeAll = _gt(g, shapeVal, g.half)
    stampVal = g.nd("Add", [g.nd("Mul", [anchAll, x7val]), shapeVal])
    anyStamp = _gt(g, g.nd("Add", [anchAll, shapeAll]), g.half)
    Vout = g.nd("Add", [g.nd("Mul", [stampVal, anyStamp]),
                        g.nd("Mul", [V, g.nd("Sub", [g.one, anyStamp])])])
    OH = _eqm_scalar(g, Vout, colorvec)
    g.nd("Mul", [OH, allsum], "output")
    return _model(g, "vc2_133")


# --------------------------------------------------------------------------- #
def _dil8b(m):
    P = np.pad(m.astype(int), 1)
    out = np.zeros_like(m, int)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out = np.maximum(out, P[dr:dr + m.shape[0], dc:dc + m.shape[1]])
    return out > 0


def _np_133(a):
    a = np.asarray(a, int)
    H, W = a.shape
    if H > 30 or W > 30:
        return None
    A = np.zeros((30, 30), int)
    A[:H, :W] = a
    nb = A > 0
    RC = np.arange(30)[:, None] * 30 + np.arange(30)[None, :]
    rem = nb.copy()
    comps = []
    for _ in range(4):
        big = np.where(rem, RC, 10 ** 6)
        seed = (big == big.min()) & rem
        T = seed.copy()
        for _ in range(32):
            T = _dil8b(T) & rem
        comps.append(T)
        rem = rem & ~T
    score = np.full(10, -2.0)
    for c in range(1, 10):
        mc = (A == c)
        if not mc.any():
            score[c] = -1
            continue
        d = _dil8b(mc)
        s = 0
        for e in range(1, 10):
            if e != c and (d & (A == e)).any():
                s += 1
        score[c] = s
    x7 = int(max(range(1, 10), key=lambda c: (score[c], -c)))
    m7 = (A == x7)
    keys = []
    for i, C in enumerate(comps):
        if not C.any():
            keys.append(1e12)
            continue
        cnt = int((C & m7).sum())
        size = int(C.sum())
        keys.append(cnt * 1e6 - size * 100 + i)
    tsel = int(np.argmin(keys))
    Tm = comps[tsel]
    if not Tm.any():
        return A[:H, :W]
    ty, tx = np.where(Tm)
    tr0, tc0 = ty.min(), tx.min()
    Tval = np.zeros((30, 30), int)
    for y, x in zip(ty, tx):
        Tval[y - tr0, x - tc0] = A[y, x]
    Tanchor = (Tval == x7)
    Tshape = (Tval != x7) & (Tval != 0)
    ay, ax = np.where(Tanchor)
    a7r, a7c = ay.min(), ax.min()
    stampVal = np.zeros((30, 30), int)
    for i, C in enumerate(comps):
        if i == tsel or not C.any():
            continue
        s7 = C & m7
        if not s7.any():
            continue
        sy, sx = np.where(s7)
        S7r, S7c = sy.min(), sx.min()
        k = int(sx.max() - sx.min() + 1)
        shp = C & ~m7
        c_i = int(A[shp].max()) if shp.any() else x7
        offr = S7r - a7r * k
        offc = S7c - a7c * k
        for (msk, col) in [(Tanchor, x7), (Tshape, c_i)]:
            for (yy, xx) in zip(*np.where(msk)):
                for io in range(k):
                    for jo in range(k):
                        ny = yy * k + io + offr
                        nx = xx * k + jo + offc
                        if 0 <= ny < 30 and 0 <= nx < 30:
                            stampVal[ny, nx] = col
    out = np.where(stampVal > 0, stampVal, A)
    return out[:H, :W]


# --------------------------------------------------------------------------- #
def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
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
    if _matches(prs, _np_173):
        try:
            m = _build_173()
            yield ("vc2_stamp72322fa7", m)
        except Exception:
            pass
    if _matches(prs, _np_133):
        try:
            m = _build_133()
            yield ("vc2_scale57aa92db", m)
        except Exception:
            pass
