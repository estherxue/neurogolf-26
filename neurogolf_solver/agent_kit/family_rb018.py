"""family_rb018 — task18 / 0e206a2e : "complete the rotated clone".

INPUT (square-ish W x H, 12-24) contains 1 or 2 SPRITES.  Each sprite is a small
4-connected creature whose body cells are one COMMON colour (the global mode =
most frequent coloured cell) and whose 3 remaining cells are 3 DISTINCT "marker"
colours.  Every sprite appears twice: an ORIGINAL drawn FULLY (body + 3 markers),
and a CLONE drawn ROTATED by one of 4 fixed transforms but showing ONLY its 3
marker cells (the common-colour body is hidden).  The 4 sprites/clones never come
within 3 cells of one another (generator spacing=3).

OUTPUT (decoded from the generator task_0e206a2e.py — the *actual* semantics, not
the task-card paraphrase):  the output contains ONLY the clone(s) drawn FULLY in
their rotated orientation (markers kept in place + the body filled in).  The
ORIGINAL sprites are DROPPED entirely; every other cell is background.

Transform:
  1. mode = most frequent nonbackground colour.
  2. Flood-fill (8-conn, through non-background) from the first two mode seeds ->
     the (<=2) ORIGINAL components (body + their 3 markers).  Clone markers = the
     non-background cells in neither component.
  3. For each component build a centred value-kernel (anchored on its first marker
     cell) and its 4 orientations:
        rot1 (r,c)->(wide-c-1,r) = +90       -> Rev @ Kc^T
        rot2 (r,c)->(c,tall-r-1) = -90       -> Kc^T @ Rev
        rot3 (r,c)->(tall-r-1,c) = vflip     -> Rev @ Kc
        rot4 (r,c)->(c,r)        = transpose -> Kc^T
     (Rev = anti-identity; the generator's 4 maps are exactly these 4 D4 elements.)
  4. For every (component, rotation) correlate the rotated MARKER pattern (one Conv
     per marker colour) against the clone-marker colour masks; a position is a valid
     placement iff all 3 markers land colour-exact (match-count == 3).  Stamp the
     FULL rotated sprite there with a per-colour ConvTranspose.
  5. Where two placements collide (an inherently symmetric clone, see below) pick
     the winner by MIN priority = rotation*10 + component-index, resolved per cell.

Irreducible ambiguity (~1.5% of *fresh* samples, provably not a function of the
input): the generator only forbids markers sharing a single row or column, so a
marker triple lying on a diagonal is symmetric under transpose (rot4) — two
different rotations of the same sprite then produce the *same* 3 marker cells but
different bodies, and a clone can even match BOTH sprites at once.  On such inputs
no solver can recover the generator's random rotation; this solver picks the
deterministic min-priority parse.  It is EXACT on every input whose output IS
determined by the grid — all 266 local train+test+arc-gen pairs and 100% of the
RECOVERABLE fresh samples (2954/3000, the 46 misses are exactly the ambiguous set).

ONNX (opset-10, static [1,10,30,30], no banned ops): value image via 1x1 Conv;
mode via per-channel ReduceSum + ArgMax; two flood-fills via MaxPool(3x3) chains;
first-cell seeds/anchors via flattened ArgMax; centred kernels via the Srow/Scol
MatMul crop; D4 orientations via a constant anti-identity + Transpose; matching via
data-dependent Conv, stamping via data-dependent ConvTranspose; per-cell
min-priority selection via Min + equality masks; one-hot via value-vector equality.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64
H30 = 30
KS = 13
CTR = 6
NFLOOD = 15
BIGP = 1e6

# ======================================================================== #
# numpy reference (mirrors the ONNX numerics exactly)                       #
# ======================================================================== #
_REV = np.zeros((KS, KS), np.float32)
for _i in range(KS):
    _REV[_i, KS - 1 - _i] = 1.0


def _dilate8(m):
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for di in range(3):
        for dj in range(3):
            out = np.maximum(out, p[di:di + m.shape[0], dj:dj + m.shape[1]])
    return out


def _flood(seed, nonbg):
    T = seed.copy()
    for _ in range(NFLOOD):
        T = np.minimum(_dilate8(T), nonbg)
    return (T > 0.5).astype(np.float32)


def _first_cell(mask):
    flat = mask.reshape(-1)
    if flat.max() < 0.5:
        return np.zeros_like(mask)
    idx = int(np.argmax(flat))
    oh_ = np.zeros_like(flat)
    oh_[idx] = 1.0
    return oh_.reshape(mask.shape)


def _corr(X, K):
    H, W = X.shape
    Xp = np.pad(X, CTR)
    out = np.zeros((H, W), np.float32)
    ys, xs = np.where(K != 0)
    for ki, kj in zip(ys, xs):
        out += K[ki, kj] * Xp[ki:ki + H, kj:kj + W]
    return out


def _stamp(centers, K):
    H, W = centers.shape
    acc = np.zeros((H + 2 * CTR, W + 2 * CTR), np.float32)
    ys, xs = np.where(K != 0)
    for ki, kj in zip(ys, xs):
        acc[ki:ki + H, kj:kj + W] += K[ki, kj] * centers
    return acc[CTR:CTR + H, CTR:CTR + W]


def _to30(a):
    H, W = a.shape
    g = np.zeros((H30, H30), np.float32)
    g[:H, :W] = a
    real = np.zeros((H30, H30), np.float32)
    real[:H, :W] = 1.0
    return g, real


def _centered_kernel(Vg, comp, ar, ac):
    Kc = np.zeros((KS, KS), np.float32)
    ys, xs = np.where(comp > 0.5)
    for r, c in zip(ys, xs):
        ki, kj = CTR + (r - ar), CTR + (c - ac)
        if 0 <= ki < KS and 0 <= kj < KS:
            Kc[ki, kj] = Vg[r, c]
    return Kc


def _ref(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or a.size == 0 or max(a.shape) > H30:
        return None
    H, W = a.shape
    Vg, real = _to30(a)
    nonbg = (Vg > 0.5).astype(np.float32) * real
    if nonbg.sum() < 0.5:
        return np.zeros((H, W), int)
    counts = np.array([((np.abs(Vg - c) < 0.5) * nonbg).sum() for c in range(10)])
    counts[0] = -1
    mode = int(np.argmax(counts))
    modeMask = (np.abs(Vg - mode) < 0.5).astype(np.float32) * nonbg

    seed0 = _first_cell(modeMask)
    comp0 = _flood(seed0, nonbg)
    rem = modeMask * (1.0 - comp0)
    seed1 = _first_cell(rem)
    comp1 = _flood(seed1, nonbg)
    comps = [comp0, comp1]

    cloneMk = nonbg * (1.0 - comp0) * (1.0 - comp1) * (1.0 - modeMask)
    cloneC = {c: cloneMk * (np.abs(Vg - c) < 0.5) for c in range(1, 10)}

    S_list, cov_list, pri_list = [], [], []
    for ci, comp in enumerate(comps):
        markers = comp * (1.0 - modeMask)
        hasComp = 1.0 if comp.sum() > 0.5 else 0.0
        nmark = markers.sum()
        an = _first_cell(markers)
        ar = int((an * np.arange(H30).reshape(H30, 1)).sum())
        ac = int((an * np.arange(H30).reshape(1, H30)).sum())
        Kc = _centered_kernel(Vg, comp, ar, ac)
        Krots = {1: _REV @ Kc.T, 2: Kc.T @ _REV, 3: _REV @ Kc, 4: Kc.T}
        for k in (1, 2, 3, 4):
            KR = Krots[k]
            mc = np.zeros((H30, H30), np.float32)
            for c in range(1, 10):
                Kc_c = (np.abs(KR - c) < 0.5).astype(np.float32)
                if Kc_c.sum() < 0.5:
                    continue
                mc += _corr(cloneC[c], Kc_c)
            valid = ((mc > 2.5) & (mc < 3.5)).astype(np.float32) * hasComp
            valid *= (1.0 if abs(nmark - 3) < 0.5 else 0.0)
            S = np.zeros((H30, H30), np.float32)
            cov = np.zeros((H30, H30), np.float32)
            for v in range(1, 10):
                Kv = (np.abs(KR - v) < 0.5).astype(np.float32)
                if Kv.sum() < 0.5:
                    continue
                pm = (_stamp(valid, Kv) > 0.5).astype(np.float32)
                S += v * pm
                cov = np.maximum(cov, pm)
            S_list.append(S)
            cov_list.append(cov)
            pri_list.append(float(k * 10 + ci))

    minpri = np.full((H30, H30), BIGP, np.float32)
    for cov, p in zip(cov_list, pri_list):
        minpri = np.minimum(minpri, cov * p + (1.0 - cov) * BIGP)
    out = np.zeros((H30, H30), np.float32)
    for S, cov, p in zip(S_list, cov_list, pri_list):
        pf = cov * p + (1.0 - cov) * BIGP
        out += S * (cov * (np.abs(pf - minpri) < 0.5))
    out *= real
    return out[:H, :W].astype(int)


# ======================================================================== #
# ONNX graph accumulator + helpers                                          #
# ======================================================================== #
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
    g.valvec = g.f([1, 10, 1, 1], list(range(10)))
    g.chidx = g.f([1, 10, 1, 1], list(range(10)))
    g.rev = g.f([1, 1, KS, KS], _REV)
    g.ri13 = g.f([1, 1, KS, 1], list(range(KS)))
    g.ci13 = g.f([1, 1, 1, KS], list(range(KS)))
    g.coff = g.f1(float(CTR))
    g.bigp = g.f1(BIGP)
    g.aranged = g.f([1, H30 * H30], [H30 * H30 - i for i in range(H30 * H30)])  # descending
    g.arange = g.f([1, H30 * H30], list(range(H30 * H30)))


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _eqm(g, a, b):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


def _first_cell_g(g, mask):
    """One-hot [1,1,30,30] of first row-major set cell; zero if mask empty."""
    flat = g.nd("Reshape", [mask, g.i64([1, H30 * H30])])         # [1,900]
    masked = g.nd("Mul", [flat, g.aranged])
    idx = g.nd("ArgMax", [masked], axis=1, keepdims=1)            # int64 [1,1]
    idxf = g.nd("Cast", [idx], to=F)
    oh_ = _eqm(g, g.arange, idxf)                                 # [1,900]
    has = _gt(g, g.nd("ReduceSum", [mask], axes=[2, 3], keepdims=1), g.half)  # [1,1,1,1]
    grid = g.nd("Reshape", [oh_, g.i64([1, 1, H30, H30])])
    return g.nd("Mul", [grid, has])


def _flood_g(g, seed, nonbg):
    T = seed
    for _ in range(NFLOOD):
        dil = g.nd("MaxPool", [T], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
        T = g.nd("Min", [dil, nonbg])
    return _gt(g, T, g.half)


def build_18():
    g = _G()
    _consts(g)

    Vg = g.nd("Conv", ["input", g.f([1, 10, 1, 1], list(range(10)))], kernel_shape=[1, 1])
    real = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    nonbg = g.nd("Sub", [real, ch0])                                     # [1,1,30,30]

    # mode
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)       # [1,10,1,1]
    ch0ind = g.f([1, 10, 1, 1], [1.0] + [0.0] * 9)
    counts_m = g.nd("Sub", [counts, g.nd("Mul", [ch0ind, g.f1(1e9)])])
    modeArg = g.nd("ArgMax", [counts_m], axis=1, keepdims=1)             # int64 [1,1,1,1]
    modegate = g.nd("Cast", [g.nd("Equal",
                    [modeArg, g.nd("Cast", [g.chidx], to=INT64)])], to=F)  # [1,10,1,1]
    modeVal = g.nd("ReduceSum", [g.nd("Mul", [modegate, g.valvec])], axes=[1], keepdims=1)  # [1,1,1,1]
    modeMask = g.nd("Mul", [_eqm(g, Vg, modeVal), nonbg])               # [1,1,30,30]

    # two flood components
    seed0 = _first_cell_g(g, modeMask)
    comp0 = _flood_g(g, seed0, nonbg)
    rem = g.nd("Mul", [modeMask, g.nd("Sub", [g.one, comp0])])
    seed1 = _first_cell_g(g, rem)
    comp1 = _flood_g(g, seed1, nonbg)
    comps = [comp0, comp1]

    notmode = g.nd("Sub", [g.one, modeMask])
    cloneMk = g.nd("Mul", [g.nd("Mul", [nonbg, g.nd("Sub", [g.one, comp0])]),
                           g.nd("Mul", [g.nd("Sub", [g.one, comp1]), notmode])])
    cloneC = {}
    for c in range(1, 10):
        cloneC[c] = g.nd("Mul", [cloneMk, _eqm(g, Vg, g.f1(float(c)))])

    S_list, cov_list, pri_list = [], [], []
    for ci, comp in enumerate(comps):
        markers = g.nd("Mul", [comp, notmode])
        hasComp = _gt(g, g.nd("ReduceSum", [comp], axes=[2, 3], keepdims=1), g.half)
        nmark = g.nd("ReduceSum", [markers], axes=[2, 3], keepdims=1)
        nmark3 = _eqm(g, nmark, g.f1(3.0))
        gate = g.nd("Mul", [hasComp, nmark3])                           # [1,1,1,1]
        an = _first_cell_g(g, markers)
        ar = g.nd("ReduceSum", [g.nd("Mul", [an, g.rowidx])], axes=[2, 3], keepdims=1)
        ac = g.nd("ReduceSum", [g.nd("Mul", [an, g.colidx])], axes=[2, 3], keepdims=1)
        VgC = g.nd("Mul", [Vg, comp])
        tgt_i = g.nd("Add", [g.nd("Sub", [g.ri13, g.coff]), ar])        # [1,1,KS,1]
        Srow2 = _eqm(g, g.colidx, tgt_i)                               # [1,1,KS,30]
        tgt_j = g.nd("Add", [g.nd("Sub", [g.ci13, g.coff]), ac])        # [1,1,1,KS]
        Scol2 = _eqm(g, g.rowidx, tgt_j)                               # [1,1,30,KS]
        Kc = g.nd("MatMul", [Srow2, g.nd("MatMul", [VgC, Scol2])])     # [1,1,KS,KS]
        KcT = g.nd("Transpose", [Kc], perm=[0, 1, 3, 2])
        Krots = {1: g.nd("MatMul", [g.rev, KcT]),
                 2: g.nd("MatMul", [KcT, g.rev]),
                 3: g.nd("MatMul", [g.rev, Kc]),
                 4: KcT}
        for k in (1, 2, 3, 4):
            KR = Krots[k]
            mc = None
            for c in range(1, 10):
                KR_c = _eqm(g, KR, g.f1(float(c)))
                term = g.nd("Conv", [cloneC[c], KR_c], kernel_shape=[KS, KS],
                            pads=[CTR, CTR, CTR, CTR])
                mc = term if mc is None else g.nd("Add", [mc, term])
            valid = g.nd("Mul", [_eqm(g, mc, g.f1(3.0)), gate])         # [1,1,30,30]
            S = None
            cov = None
            for v in range(1, 10):
                KR_v = _eqm(g, KR, g.f1(float(v)))
                pm = g.nd("ConvTranspose", [valid, KR_v], kernel_shape=[KS, KS],
                          strides=[1, 1], pads=[CTR, CTR, CTR, CTR])
                pm01 = _gt(g, pm, g.half)
                sv = g.nd("Mul", [pm01, g.f1(float(v))])
                S = sv if S is None else g.nd("Add", [S, sv])
                cov = pm01 if cov is None else g.nd("Max", [cov, pm01])
            S_list.append(S)
            cov_list.append(cov)
            pri_list.append(float(k * 10 + ci))

    # per-cell min-priority combine
    minpri = None
    pfs = []
    for cov, p in zip(cov_list, pri_list):
        pf = g.nd("Add", [g.nd("Mul", [cov, g.f1(p)]),
                          g.nd("Mul", [g.nd("Sub", [g.one, cov]), g.bigp])])
        pfs.append(pf)
        minpri = pf if minpri is None else g.nd("Min", [minpri, pf])
    outv = None
    for S, cov, pf in zip(S_list, cov_list, pfs):
        sel = g.nd("Mul", [cov, _eqm(g, pf, minpri)])
        contrib = g.nd("Mul", [S, sel])
        outv = contrib if outv is None else g.nd("Add", [outv, contrib])
    outv = g.nd("Mul", [outv, real])

    OH = _eqm(g, outv, g.valvec)                                       # [1,10,30,30]
    g.nd("Mul", [OH, real], "output")
    return _model(g, "rb018")


# ======================================================================== #
# detection / candidates                                                    #
# ======================================================================== #
def _pairs(ex):
    out = []
    for s in ("train", "test"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return []
            if max(a.shape) > H30 or max(b.shape) > H30:
                return []
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
        return [("rb018", build_18())]
    except Exception:
        return []
