"""family_vc2_1 — RETRY wave: two static-ONNX rules the previous wave marked infeasible.

task319 (verify_ce602527): three fg-colour objects; one is a 2x-upscaled "template",
    the other two are 1x candidates.  Output = crop of the candidate whose 2x-upscale
    contains another object's exact fg/bg bbox pattern (occurrences>0); tie -> smaller.
    ONNX: per-channel origin-crop (MatMul selection), constant-matrix upscale x2, a
    single batched Conv computing correlation of every source-upscale against every
    template (exact-match via fg-overlap==|Tfg| and window-fg-sum==|Tfg|), argmax
    selection, MatMul crop of the winner.  (Colour-independent: recolour cancels.)

task076 (verify_36d67576): per-object D4-oriented multicolour template stamp.  (see
    build below; emitted only if its numpy mirror is exact on train+test.)
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


def _le(g, a, b):
    # a <= b for integer tensors -> a < b + 0.5
    return _lt(g, a, g.nd("Add", [b, g.half]))


def _eqm(g, a, b):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


def _minmax(g, has, idx, axis):
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], axes=[axis], keepdims=1)
    inv = g.nd("Mul", [has, g.nd("Sub", [g.cbig, idx])])
    mn = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [inv], axes=[axis], keepdims=1)])
    return mn, mx


# =========================================================================== #
# task319 — ce602527                                                          #
# =========================================================================== #
def build_319():
    g = _G()
    _consts(g)
    rowidx, colidx, half, one = g.rowidx, g.colidx, g.half, g.one

    # counts / background / foreground gates ---------------------------------
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    chidx = g.f([1, 10, 1, 1], list(range(10)))
    bg_arg = g.nd("ArgMax", [counts], axis=1, keepdims=1)                   # int64 [1,1,1,1]
    bggate = g.nd("Cast", [g.nd("Equal",
                  [bg_arg, g.nd("Cast", [chidx], to=INT64)])], to=F)        # [1,10,1,1]
    present = _gt(g, counts, half)                                          # [1,10,1,1]
    fg = g.nd("Mul", [present, g.nd("Sub", [one, bggate])])                 # [1,10,1,1]

    # per-channel origin crop -------------------------------------------------
    rowhas = g.nd("ReduceMax", ["input"], axes=[3], keepdims=1)            # [1,10,30,1]
    colhas = g.nd("ReduceMax", ["input"], axes=[2], keepdims=1)            # [1,10,1,30]
    minr, maxr = _minmax(g, rowhas, rowidx, 2)                             # [1,10,1,1]
    minc, maxc = _minmax(g, colhas, colidx, 3)
    Hc = g.nd("Add", [g.nd("Sub", [maxr, minr]), one])                     # [1,10,1,1]
    Wc = g.nd("Add", [g.nd("Sub", [maxc, minc]), one])

    match_r = _eqm(g, colidx, g.nd("Add", [rowidx, minr]))                 # [1,10,30,30]
    trunc_r = _lt(g, rowidx, Hc)                                           # [1,10,30,1]
    Srow = g.nd("Mul", [match_r, trunc_r])
    match_c = _eqm(g, g.nd("Add", [colidx, minc]), rowidx)
    trunc_c = _lt(g, colidx, Wc)                                           # [1,10,1,30]
    Scol = g.nd("Mul", [match_c, trunc_c])
    crC = g.nd("MatMul", [Srow, g.nd("MatMul", ["input", Scol])])         # [1,10,30,30]

    # upscale x2 (constant matrices)  UP = U @ crC @ U^T ----------------------
    Uv = np.zeros((30, 30), np.float32)
    for r in range(30):
        Uv[r, r // 2] = 1.0
    U = g.f([1, 1, 30, 30], Uv)
    Ut = g.f([1, 1, 30, 30], Uv.T)
    UP = g.nd("MatMul", [U, g.nd("MatMul", [crC, Ut])])                    # [1,10,30,30]

    # reshape to batch=channel, pad to 59x59 ---------------------------------
    UPb = g.nd("Reshape", [UP, g.i64([10, 1, 30, 30])])                    # [10,1,30,30]
    UP59 = g.nd("Pad", [UPb], mode="constant", value=0.0,
                pads=[0, 0, 0, 0, 0, 0, 29, 29])                          # [10,1,59,59]

    # template weights (source crops reused as templates) --------------------
    Tw = g.nd("Reshape", [crC, g.i64([10, 1, 30, 30])])                   # [10,1,30,30]
    Tfg = g.nd("ReduceSum", [crC], axes=[2, 3], keepdims=1)               # [1,10,1,1]
    bboxind = g.nd("Mul", [_lt(g, rowidx, Hc), _lt(g, colidx, Wc)])       # [1,10,30,30]
    Bw = g.nd("Reshape", [bboxind, g.i64([10, 1, 30, 30])])              # [10,1,30,30]

    # batched correlation: [src=10, tpl=10, 30, 30] --------------------------
    corr = g.nd("Conv", [UP59, Tw], kernel_shape=[30, 30])                # [10,10,30,30]
    winc = g.nd("Conv", [UP59, Bw], kernel_shape=[30, 30])                # [10,10,30,30]

    # Tfg indexed by template dim (dim1); broadcast [1,10,1,1] over [10,10,..]
    m_corr = _eqm(g, corr, Tfg)
    m_winc = _eqm(g, winc, Tfg)
    match_num = g.nd("Mul", [m_corr, m_winc])                             # [10,10,30,30]

    # valid offsets: i <= 2*Hc[src] - Hb[tpl] ; j <= 2*Wc[src] - Wb[tpl]
    Hc_src = g.nd("Reshape", [Hc, g.i64([10, 1, 1, 1])])                  # [10,1,1,1]
    Wc_src = g.nd("Reshape", [Wc, g.i64([10, 1, 1, 1])])
    two = g.f([1, 1, 1, 1], [2.0])
    upH = g.nd("Mul", [Hc_src, two])                                      # [10,1,1,1]
    upW = g.nd("Mul", [Wc_src, two])
    rhs_i = g.nd("Sub", [upH, Hc])                                        # [10,10,1,1]
    rhs_j = g.nd("Sub", [upW, Wc])
    cond_i = _le(g, rowidx, rhs_i)                                        # [10,10,30,1]
    cond_j = _le(g, colidx, rhs_j)                                        # [10,10,1,30]
    valid = g.nd("Mul", [cond_i, cond_j])                                # [10,10,30,30]
    match = g.nd("Mul", [match_num, valid])
    occ = _gt(g, g.nd("ReduceMax", [match], axes=[2, 3], keepdims=1), half)  # [10,10,1,1]

    # gate templates: fg[b] and b!=c -----------------------------------------
    fg_tpl = g.nd("Reshape", [fg, g.i64([1, 10, 1, 1])])                  # [1,10,1,1] (dim1=tpl)
    ne = 1.0 - np.eye(10, dtype=np.float32)
    Ine = g.f([10, 10, 1, 1], ne)
    occ_g = g.nd("Mul", [g.nd("Mul", [occ, fg_tpl]), Ine])               # [10,10,1,1]
    win_src = g.nd("ReduceMax", [occ_g], axes=[1], keepdims=1)            # [10,1,1,1]
    win = g.nd("Reshape", [win_src, g.i64([1, 10, 1, 1])])               # [1,10,1,1] (ch=src)

    # score / winner ----------------------------------------------------------
    BIG = g.f([1, 1, 1, 1], [1e6])
    INF = g.f([1, 1, 1, 1], [1e12])
    base = g.nd("Sub", [g.nd("Mul", [win, BIG]), counts])
    score = g.nd("Sub", [base, g.nd("Mul", [g.nd("Sub", [one, fg]), INF])])  # [1,10,1,1]
    win_arg = g.nd("ArgMax", [score], axis=1, keepdims=1)
    wgate = g.nd("Cast", [g.nd("Equal",
                 [win_arg, g.nd("Cast", [chidx], to=INT64)])], to=F)      # [1,10,1,1]

    # build output ------------------------------------------------------------
    fgorigin = g.nd("ReduceSum", [g.nd("Mul", [crC, wgate])], axes=[1], keepdims=1)  # [1,1,30,30]
    bh = g.nd("ReduceSum", [g.nd("Mul", [Hc, wgate])], axes=[1], keepdims=1)         # [1,1,1,1]
    bw = g.nd("ReduceSum", [g.nd("Mul", [Wc, wgate])], axes=[1], keepdims=1)
    bbox = g.nd("Mul", [_lt(g, rowidx, bh), _lt(g, colidx, bw)])          # [1,1,30,30]
    bgmask = g.nd("Mul", [bbox, g.nd("Sub", [one, fgorigin])])           # [1,1,30,30]
    out_fg = g.nd("Mul", [fgorigin, wgate])                              # [1,10,30,30]
    out_bg = g.nd("Mul", [bgmask, bggate])
    g.nd("Add", [out_fg, out_bg], "output")
    return _model(g, "vc2_319")


# =========================================================================== #
# task076 — 36d67576  (per-object D4-oriented multicolour template stamp)      #
# =========================================================================== #
_KS = 15
_CTR = 7


def _slice_ch(g, c):
    return g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])


def build_076():
    g = _G()
    _consts(g)
    rowidx, colidx, half, one = g.rowidx, g.colidx, g.half, g.one

    # value grid + non-background mask ---------------------------------------
    Vg = g.nd("Conv", ["input", g.f([1, 10, 1, 1], list(range(10)))], kernel_shape=[1, 1])
    nonbg = g.nd("Sub", [g.nd("ReduceSum", ["input"], axes=[1], keepdims=1),
                         _slice_ch(g, 0)])                                  # [1,1,30,30]

    # template = 8-connected flood from colour-1 seeds through non-bg --------
    T = _slice_ch(g, 1)
    for _ in range(30):
        dil = g.nd("MaxPool", [T], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
        T = g.nd("Min", [dil, nonbg])
    T = _gt(g, T, half)                                                     # [1,1,30,30]

    ch2, ch4 = _slice_ch(g, 2), _slice_ch(g, 4)
    a2 = g.nd("Mul", [ch2, T])                                              # template anchor cell
    ai = g.nd("ReduceSum", [g.nd("Mul", [a2, rowidx])], axes=[2, 3], keepdims=1)  # [1,1,1,1]
    aj = g.nd("ReduceSum", [g.nd("Mul", [a2, colidx])], axes=[2, 3], keepdims=1)
    S4 = g.nd("ReduceSum", [g.nd("Mul", [ch4, T])], axes=[2, 3], keepdims=1)

    # centred kernel Kc[ki,kj] = Vg where anchor -> centre -------------------
    VgT = g.nd("Mul", [Vg, T])
    ri15 = g.f([1, 1, _KS, 1], list(range(_KS)))
    ci15 = g.f([1, 1, 1, _KS], list(range(_KS)))
    coff = g.f([1, 1, 1, 1], [float(_CTR)])
    # Srow2[ki,i] = (i == ai - CTR + ki)   -> [1,1,15,30]
    tgt_i = g.nd("Add", [g.nd("Sub", [ri15, coff]), ai])                   # [1,1,15,1]
    Srow2 = _eqm(g, colidx, tgt_i)                                         # [1,1,15,30]
    # Scol2[j,kj] = (j == aj - CTR + kj)   -> [1,1,30,15]
    tgt_j = g.nd("Add", [g.nd("Sub", [ci15, coff]), aj])                   # [1,1,1,15]
    Scol2 = _eqm(g, rowidx, tgt_j)                                         # [1,1,30,15]
    Kc = g.nd("MatMul", [Srow2, g.nd("MatMul", [VgT, Scol2])])            # [1,1,15,15]

    # D4 orientations via constant anti-identity (reflect about centre) ------
    rev = np.zeros((_KS, _KS), np.float32)
    for i in range(_KS):
        rev[i, _KS - 1 - i] = 1.0
    Rev = g.f([1, 1, _KS, _KS], rev)

    def revW(x):
        return g.nd("MatMul", [x, Rev])

    def revH(x):
        return g.nd("MatMul", [Rev, x])

    Kt = g.nd("Transpose", [Kc], perm=[0, 1, 3, 2])
    orients = [Kc, revW(Kc), revH(Kc), revW(revH(Kc)),
               Kt, revW(Kt), revH(Kt), revW(revH(Kt))]

    marker2 = g.nd("Mul", [ch2, g.nd("Sub", [one, T])])
    marker4 = g.nd("Mul", [ch4, g.nd("Sub", [one, T])])

    stamp1 = None
    stamp3 = None
    for Ag in orients:
        K4 = _eqm(g, Ag, g.f([1, 1, 1, 1], [4.0]))                        # [1,1,15,15]
        corr = g.nd("Conv", [marker4, K4], kernel_shape=[_KS, _KS],
                    pads=[_CTR, _CTR, _CTR, _CTR])                        # [1,1,30,30]
        match = g.nd("Mul", [_eqm(g, corr, S4), marker2])                # matched anchors
        for c, acc in (("1", "s1"), ("3", "s3")):
            Kcc = _eqm(g, Ag, g.f([1, 1, 1, 1], [float(c)]))
            sc = g.nd("ConvTranspose", [match, Kcc], kernel_shape=[_KS, _KS],
                      strides=[1, 1], pads=[_CTR, _CTR, _CTR, _CTR])      # [1,1,30,30]
            if acc == "s1":
                stamp1 = sc if stamp1 is None else g.nd("Add", [stamp1, sc])
            else:
                stamp3 = sc if stamp3 is None else g.nd("Add", [stamp3, sc])

    s1 = _gt(g, stamp1, half)                                             # [1,1,30,30]
    s3 = _gt(g, stamp3, half)
    sany = g.nd("Add", [s1, s3])

    oh1 = g.f([1, 10, 1, 1], [1.0 if k == 1 else 0.0 for k in range(10)])
    oh3 = g.f([1, 10, 1, 1], [1.0 if k == 3 else 0.0 for k in range(10)])
    oh0 = g.f([1, 10, 1, 1], [1.0 if k == 0 else 0.0 for k in range(10)])
    add1 = g.nd("Mul", [s1, oh1])                                         # [1,10,30,30]
    add3 = g.nd("Mul", [s3, oh3])
    sub0 = g.nd("Mul", [sany, oh0])
    g.nd("Add", [g.nd("Sub", [g.nd("Add", ["input", add1]), sub0]), add3], "output")
    return _model(g, "vc2_076")


# --------------------------------------------------------------------------- #
# numpy mirror for task076 detection                                           #
# --------------------------------------------------------------------------- #
def _dilate8(m):
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for di in range(3):
        for dj in range(3):
            out = np.maximum(out, p[di:di + m.shape[0], dj:dj + m.shape[1]])
    return out


def _mirror_076(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or max(a.shape) > 30:
        return None
    Hd, Wd = a.shape
    oh_ = _onehot(a)
    if int(np.argmax(oh_.sum(axis=(1, 2)))) != 0:
        return None
    nonbg = oh_[1:].sum(0)
    T = oh_[1].copy()
    for _ in range(30):
        T = np.minimum(_dilate8(T), nonbg)
    T = (T > 0.5).astype(np.float32)
    Vg = sum(c * oh_[c] for c in range(10))
    a2 = oh_[2] * T
    ys, xs = np.where(a2 > 0.5)
    if ys.size != 1:
        return None
    ai, aj = ys[0], xs[0]
    KS, CTR = _KS, _CTR
    Kc = np.zeros((KS, KS), np.float32)
    for ii, jj in zip(*np.where(T > 0.5)):
        ki, kj = CTR + (ii - ai), CTR + (jj - aj)
        if 0 <= ki < KS and 0 <= kj < KS:
            Kc[ki, kj] = Vg[ii, jj]
    rev = np.zeros((KS, KS), np.float32)
    for i in range(KS):
        rev[i, KS - 1 - i] = 1.0
    Kt = Kc.T
    orients = [Kc, Kc @ rev, rev @ Kc, rev @ Kc @ rev,
               Kt, Kt @ rev, rev @ Kt, rev @ Kt @ rev]
    S4 = (oh_[4] * T).sum()
    marker2 = oh_[2] * (1 - T)
    marker4 = oh_[4] * (1 - T)
    Xp = np.pad(marker4, CTR)
    out = Vg.copy()
    for Ag in orients:
        K4 = (np.abs(Ag - 4) < 0.5).astype(np.float32)
        corr = np.zeros((30, 30), np.float32)
        for i in range(30):
            for j in range(30):
                corr[i, j] = (K4 * Xp[i:i + KS, j:j + KS]).sum()
        match = ((np.abs(corr - S4) < 0.5) & (marker2 > 0.5))
        if match.sum() < 0.5:
            continue
        for c in (1, 3):
            Kcc = (np.abs(Ag - c) < 0.5)
            for p_i, p_j in zip(*np.where(match)):
                for ki, kj in np.argwhere(Kcc):
                    qi, qj = p_i + ki - CTR, p_j + kj - CTR
                    if 0 <= qi < 30 and 0 <= qj < 30:
                        out[qi, qj] = c
    return out[:Hd, :Wd].astype(int)


# --------------------------------------------------------------------------- #
# numpy mirror for task319 detection                                           #
# --------------------------------------------------------------------------- #
_U = np.zeros((30, 30), np.float32)
for _r in range(30):
    _U[_r, _r // 2] = 1.0
_RI = np.arange(30).reshape(30, 1).astype(np.float32)
_CI = np.arange(30).reshape(1, 30).astype(np.float32)


def _onehot(a):
    Hd, Wd = a.shape
    o = np.zeros((10, 30, 30), np.float32)
    for c in range(10):
        o[c, :Hd, :Wd] = (a == c)
    return o


def _corr(src59, tpl):
    o = np.zeros((30, 30), np.float32)
    for i in range(30):
        for j in range(30):
            o[i, j] = (tpl * src59[i:i + 30, j:j + 30]).sum()
    return o


def _mirror_319(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or max(a.shape) > 30:
        return None
    oh_ = _onehot(a)
    counts = oh_.sum(axis=(1, 2))
    bg = int(np.argmax(counts))
    fg = np.array([counts[c] > 0.5 and c != bg for c in range(10)])
    if fg.sum() != 3:
        return None
    crC = np.zeros((10, 30, 30), np.float32)
    Hc = np.zeros(10)
    Wc = np.zeros(10)
    for c in range(10):
        m = oh_[c]
        if m.sum() < 0.5:
            continue
        ys, xs = np.where(m > 0.5)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        crC[c, :h, :w] = m[r0:r1 + 1, c0:c1 + 1]
        Hc[c], Wc[c] = h, w
    UP = np.stack([_U @ crC[c] @ _U.T for c in range(10)])
    UP59 = np.zeros((10, 59, 59), np.float32)
    UP59[:, :30, :30] = UP
    Tfg = crC.reshape(10, -1).sum(1)
    bboxind = np.zeros((10, 30, 30), np.float32)
    for b in range(10):
        if Hc[b] > 0:
            bboxind[b, :int(Hc[b]), :int(Wc[b])] = 1
    occ = np.zeros((10, 10))
    for c in range(10):
        if not fg[c]:
            continue
        for b in range(10):
            if not fg[b] or b == c:
                continue
            cc = _corr(UP59[c], crC[b])
            wu = _corr(UP59[c], bboxind[b])
            valid = (_RI <= 2 * Hc[c] - Hc[b]) & (_CI <= 2 * Wc[c] - Wc[b])
            match = (np.abs(cc - Tfg[b]) < 0.5) & (np.abs(wu - Tfg[b]) < 0.5) & valid
            occ[c, b] = 1.0 if match.any() else 0.0
    win = occ.max(axis=1)
    score = np.where(fg, win * 1e6 - counts, -1e18)
    wc = int(np.argmax(score))
    h, w = int(Hc[wc]), int(Wc[wc])
    out = np.full((h, w), bg, int)
    out[crC[wc, :h, :w] > 0.5] = wc
    return out


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
    if _matches(prs, _mirror_319):
        try:
            yield ("vc2_ce602527", build_319())
        except Exception:
            pass
    if _matches(prs, _mirror_076):
        try:
            yield ("vc2_36d67576", build_076())
        except Exception:
            pass
