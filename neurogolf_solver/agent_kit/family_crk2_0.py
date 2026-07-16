"""family_crk2_0 -- data-dependent ARC->ONNX solvers (opset 10).

Each sub-solver detects a structural rule from the train/test/arc-gen pairs,
mirrors the exact ONNX semantics in numpy, and only emits when the reconstruction
reproduces EVERY provided pair.  Heavy use of the data-dependent MatMul-selection
technique: build a STATIC [.,.,30,30] selection/shift matrix from a scalar computed
from the input, apply with MatMul -> data-dependent crop/translate/scale with fully
static tensor shapes (required for the cost shape-inference pass).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
G = HEIGHT  # 30


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
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
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > G or max(b.shape) > G:
                continue
            out.append((a, b))
    return out


def _check(model):
    onnx.checker.check_model(model, full_check=True)


QMAX = 15


# ---- shift / period / fill helpers (mirror of family_dynperiod) ------------ #
def _shift_plane(g, t, d, axis):
    """Shift [.,.,30,30] LEFT/UP by static d>=1 along axis (2 rows, 3 cols)."""
    if axis == 3:
        sl = g.nd("Slice", [t, g.i64([d]), g.i64([G]), g.i64([3])])
        return g.nd("Pad", [sl], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, d])
    sl = g.nd("Slice", [t, g.i64([d]), g.i64([G]), g.i64([2])])
    return g.nd("Pad", [sl], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, d, 0])


def _period_scalar(g, known, color, axis, halfc, validmask=None):
    """Smallest static d in 1..QMAX with zero pattern autocorrelation, as [1,1,1,1]."""
    qstar = None
    remaining = None
    one = g.f([1, 1, 1, 1], [1.0])
    for d in range(1, QMAX + 1):
        shK = _shift_plane(g, known, d, axis)
        shC = _shift_plane(g, color, d, axis)
        both = g.nd("Mul", [known, shK])
        if validmask is not None:
            both = g.nd("Mul", [both, validmask[d]])
        diff = g.nd("Abs", [g.nd("Sub", [color, shC])])
        mism = g.nd("Cast", [g.nd("Greater", [diff, halfc])], to=F)
        prod = g.nd("Mul", [both, mism])
        score = g.nd("ReduceSum", [prod], axes=[2, 3], keepdims=1)   # [1,1,1,1]
        zero = g.nd("Cast", [g.nd("Less", [score, halfc])], to=F)
        if remaining is None:
            gate = zero
            remaining = g.nd("Sub", [one, zero])
        else:
            gate = g.nd("Mul", [zero, remaining])
            remaining = g.nd("Mul", [remaining, g.nd("Sub", [one, zero])])
        contrib = g.nd("Mul", [gate, g.f([1, 1, 1, 1], [float(d)])])
        qstar = contrib if qstar is None else g.nd("Add", [qstar, contrib])
    return qstar


def _shift_mats(g, t1, d, half):
    s_a = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [t1, d])]), half])], to=F)
    s_b = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Add", [t1, d])]), half])], to=F)
    return s_a, s_b


def _fill_axis(g, Ft, qstar, axis, rowidx, colidx, half):
    """Doubling-OR fill along axis (3 horizontal, 2 vertical) by data-dependent qstar."""
    if axis == 3:
        t1 = g.nd("Sub", [colidx, rowidx])
    else:
        t1 = g.nd("Sub", [rowidx, colidx])
    cur = Ft
    for k in range(5):
        d = qstar if k == 0 else g.nd("Mul", [qstar, g.f([1, 1, 1, 1], [float(2 ** k)])])
        s_pos, s_neg = _shift_mats(g, t1, d, half)
        if axis == 3:
            a = g.nd("MatMul", [cur, s_pos])
            b = g.nd("MatMul", [cur, s_neg])
        else:
            a = g.nd("MatMul", [s_pos, cur])
            b = g.nd("MatMul", [s_neg, cur])
        cur = g.nd("Max", [cur, a, b])
    return cur


# ---- numpy mirrors of the above -------------------------------------------- #
def _onehot(a):
    Hh, Ww = a.shape
    X = np.zeros((CHANNELS, G, G), np.float32)
    for c in range(CHANNELS):
        X[c, :Hh, :Ww] = (a == c)
    return X


def _np_shift(T, d, axis, direction):
    out = np.zeros_like(T)
    if d == 0:
        return T.copy()
    if d >= G:
        return out
    if axis == 1:
        if direction > 0:
            out[..., :G - d] = T[..., d:]
        else:
            out[..., d:] = T[..., :G - d]
    else:
        if direction > 0:
            out[..., :G - d, :] = T[..., d:, :]
        else:
            out[..., d:, :] = T[..., :G - d, :]
    return out


def _np_period(known, color, axis, validmask=None):
    for d in range(1, QMAX + 1):
        shK = _np_shift(known, d, axis, +1)
        shC = _np_shift(color, d, axis, +1)
        both = known * shK
        if validmask is not None:
            both = both * validmask[d]
        mism = (np.abs(color - shC) > 0.5).astype(np.float32)
        if float((both * mism).sum()) < 0.5:
            return d
    return 0


def _np_fill(Ften, dstar, axis):
    cur = Ften
    for k in range(5):
        d = dstar * (2 ** k)
        cur = np.maximum(np.maximum(cur, _np_shift(cur, d, axis, +1)),
                         _np_shift(cur, d, axis, -1))
    return cur


# =========================================================================== #
# TASK 269: nearest upscale by k = number of non-background cells              #
# =========================================================================== #
def _sim_upk(a):
    k = int((a != 0).sum())
    if k < 1:
        return None
    H, W = a.shape
    if H * k > G or W * k > G:
        return None
    return np.kron(a, np.ones((k, k), int))


def _build_upk():
    g = _G()
    ch19 = g.nd("Slice", ["input", g.i64([1]), g.i64([CHANNELS]), g.i64([1])])  # [1,9,30,30]
    k = g.nd("ReduceSum", [ch19], axes=[1, 2, 3], keepdims=1)                    # [1,1,1,1]
    icol = g.f([1, 1, G, 1], list(range(G)))
    irow = g.f([1, 1, 1, G], list(range(G)))
    rk = g.nd("Mul", [irow, k])                                                  # [1,1,1,30]
    d = g.nd("Sub", [icol, rk])                                                  # [1,1,30,30]
    ge0 = g.nd("Cast", [g.nd("Greater", [d, g.f([1], [-0.5])])], to=F)
    km = g.nd("Sub", [k, g.f([1, 1, 1, 1], [0.5])])
    lt = g.nd("Cast", [g.nd("Less", [d, km])], to=F)
    rrow = g.nd("Mul", [ge0, lt])                                                # [1,1,30,30]
    rcol = g.nd("Transpose", [rrow], perm=[0, 1, 3, 2])
    y = g.nd("MatMul", [rrow, "input"])
    g.nd("MatMul", [y, rcol], "output")
    return _model(g)


def _try_upk(ex):
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    det = _pairs(ex, ("train", "test"))
    if not det:
        return None
    for a, b in det + allp:
        pred = _sim_upk(a)
        if pred is None or pred.shape != b.shape or not np.array_equal(pred, b):
            return None
    # require an actual upscale somewhere (k>1) to avoid trivial identity overfit
    if all(int((a != 0).sum()) <= 1 for a, _ in det):
        return None
    try:
        m = _build_upk(); _check(m)
    except Exception:
        return None
    return ("upk", m)


# =========================================================================== #
# TASK 362: a full +cross of colour C, plus n marker cells of colour 5 in a   #
# side column.  Output moves the cross down by n and left by n, drops markers. #
# =========================================================================== #
def _sim_cross(a, marker=5):
    H, W = a.shape
    n = int((a == marker).sum())
    xc = a.copy()
    xc[xc == marker] = 0
    mask = (xc != 0).astype(int)
    colsum = mask.sum(0)
    rowsum = mask.sum(1)
    vcol = colsum > 1
    hrow = rowsum > 1
    if vcol.sum() != 1 or hrow.sum() != 1:
        return None
    cols = np.where(vcol)[0]
    cs = set(int(v) for v in xc[xc != 0].tolist())
    if len(cs) != 1:
        return None
    C = cs.pop()
    real = np.ones((H, W), int)
    vline = real * vcol[None, :]
    hline = real * hrow[:, None]
    # shift vline left by n, hline down by n
    out = np.zeros((H, W), int)
    M = np.zeros((G, G), int)
    for i in range(G):
        for j in range(G):
            if i - j == n:
                M[i, j] = 1
    vl = np.zeros((G, G)); vl[:H, :W] = vline
    hl = np.zeros((G, G)); hl[:H, :W] = hline
    vshift = vl @ M
    hshift = M @ hl
    cross = np.maximum(vshift, hshift)[:H, :W]
    out[cross > 0.5] = C
    return out


def _build_cross(marker=5):
    g = _G()
    e_m = g.f([1, CHANNELS, 1, 1], [1.0 if c == marker else 0.0 for c in range(CHANNELS)])
    # keep only coloured cross cells: drop background (0) and marker
    e_cross = g.f([1, CHANNELS, 1, 1],
                  [0.0 if (c == marker or c == 0) else 1.0 for c in range(CHANNELS)])
    half = g.f([1], [0.5])

    xc = g.nd("Mul", ["input", e_cross])                               # only cross colours
    xmask = g.nd("ReduceSum", [xc], axes=[1], keepdims=1)              # [1,1,30,30]
    real = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)         # [1,1,30,30]
    colsum = g.nd("ReduceSum", [xmask], axes=[2], keepdims=1)         # [1,1,1,30]
    rowsum = g.nd("ReduceSum", [xmask], axes=[3], keepdims=1)         # [1,1,30,1]
    vcol = g.nd("Cast", [g.nd("Greater", [colsum, g.f([1], [1.5])])], to=F)
    hrow = g.nd("Cast", [g.nd("Greater", [rowsum, g.f([1], [1.5])])], to=F)
    vline = g.nd("Mul", [vcol, real])                                 # [1,1,30,30]
    hline = g.nd("Mul", [hrow, real])
    # n = number of markers
    n = g.nd("ReduceSum", [g.nd("Mul", ["input", e_m])], axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    # shift matrix M[i,j] = 1 iff i-j == n
    icol = g.f([1, 1, G, 1], list(range(G)))
    irow = g.f([1, 1, 1, G], list(range(G)))
    dM = g.nd("Sub", [icol, irow])                                    # [1,1,30,30]: i-j
    M = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [dM, n])]), half])], to=F)
    vshift = g.nd("MatMul", [vline, M])                               # cols left by n
    hshift = g.nd("MatMul", [M, hline])                              # rows down by n
    cross = g.nd("Max", [vshift, hshift])                            # [1,1,30,30]
    # colour selector (xc already excludes bg + marker)
    cnt = g.nd("ReduceSum", [xc], axes=[2, 3], keepdims=1)            # [1,10,1,1]
    sel = g.nd("Cast", [g.nd("Greater", [cnt, g.f([1], [0.5])])], to=F)
    colored = g.nd("Mul", [cross, sel])                              # [1,10,30,30]
    # restore background channel on real cells not covered by the cross
    one = g.f([1, 1, 1, 1], [1.0])
    bg = g.nd("Mul", [real, g.nd("Sub", [one, cross])])             # [1,1,30,30]
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    g.nd("Add", [colored, g.nd("Mul", [bg, e0])], "output")
    return _model(g)


def _try_cross(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    for a, b in det:
        if a.shape != b.shape:
            return None
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        pred = _sim_cross(a)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_cross(); _check(m)
    except Exception:
        return None
    return ("cross", m)


# =========================================================================== #
# Generic crop-to-bounding-box of non-background, optional bg-recolour to 0    #
# and optional static integer upscale.  (TASK 259, 384)                        #
# =========================================================================== #
def _sim_bboxcrop(a, bgc, scale):
    ar = a.copy()
    if bgc != 0:
        ar[ar == bgc] = 0
    fg = ar != 0
    if not fg.any():
        return None
    rows = np.where(fg.any(1))[0]
    cols = np.where(fg.any(0))[0]
    crop = ar[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    if scale > 1:
        crop = np.kron(crop, np.ones((scale, scale), int))
    if max(crop.shape) > G:
        return None
    return crop


def _build_bboxcrop(bgc, scale):
    g = _G()
    half = g.f([1], [0.5])
    BIG = 100.0
    src = "input"
    if bgc != 0:
        # recolour: colour bgc -> 0 (merge into channel 0) via 1x1 conv
        w = [0.0] * (CHANNELS * CHANNELS)  # [O,I,1,1]
        for i in range(CHANNELS):
            o = 0 if i == bgc else i
            w[o * CHANNELS + i] = 1.0
        wt = oh.make_tensor(g.nm("W"), F, [CHANNELS, CHANNELS, 1, 1], w)
        g.inits.append(wt)
        src = g.nd("Conv", [src, wt.name], kernel_shape=[1, 1], pads=[0, 0, 0, 0])

    e_nobg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    fg = g.nd("Cast", [g.nd("Greater",
              [g.nd("ReduceSum", [g.nd("Mul", [src, e_nobg])], axes=[1], keepdims=1),
               half])], to=F)                                       # [1,1,30,30]
    rowhas = g.nd("ReduceMax", [fg], axes=[3], keepdims=1)          # [1,1,30,1]
    colhas = g.nd("ReduceMax", [fg], axes=[2], keepdims=1)          # [1,1,1,30]
    ri = g.f([1, 1, G, 1], list(range(G)))
    ci = g.f([1, 1, 1, G], list(range(G)))
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [BIG])
    # rmin / rmax
    rpos = g.nd("Mul", [ri, rowhas])
    rmin = g.nd("ReduceMin", [g.nd("Add", [rpos, g.nd("Mul", [big, g.nd("Sub", [one, rowhas])])])],
                axes=[2], keepdims=1)                               # [1,1,1,1]
    rmax = g.nd("ReduceMax", [g.nd("Sub", [rpos, g.nd("Mul", [big, g.nd("Sub", [one, rowhas])])])],
                axes=[2], keepdims=1)
    cpos = g.nd("Mul", [ci, colhas])
    cmin = g.nd("ReduceMin", [g.nd("Add", [cpos, g.nd("Mul", [big, g.nd("Sub", [one, colhas])])])],
                axes=[3], keepdims=1)
    cmax = g.nd("ReduceMax", [g.nd("Sub", [cpos, g.nd("Mul", [big, g.nd("Sub", [one, colhas])])])],
                axes=[3], keepdims=1)
    h = g.nd("Add", [g.nd("Sub", [rmax, rmin]), one])
    w_ = g.nd("Add", [g.nd("Sub", [cmax, cmin]), one])
    # shift matrices
    gi = g.f([1, 1, G, 1], list(range(G)))
    gr = g.f([1, 1, 1, G], list(range(G)))
    dRA = g.nd("Sub", [gr, gi])                                     # [i,r] = r - i
    Srow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [dRA, rmin])]), half])], to=F)
    gc = g.f([1, 1, G, 1], list(range(G)))
    gj = g.f([1, 1, 1, G], list(range(G)))
    dCJ = g.nd("Sub", [gc, gj])                                     # [c,j] = c - j
    Scol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [dCJ, cmin])]), half])], to=F)
    tmp = g.nd("MatMul", [src, Scol])
    shifted = g.nd("MatMul", [Srow, tmp])                           # [1,10,30,30]
    # window mask
    rowsel = g.nd("Cast", [g.nd("Less", [ri, g.nd("Sub", [h, half])])], to=F)   # [1,1,30,1]
    colsel = g.nd("Cast", [g.nd("Less", [ci, g.nd("Sub", [w_, half])])], to=F)  # [1,1,1,30]
    wm = g.nd("Mul", [rowsel, colsel])                             # [1,1,30,30]
    cropped = g.nd("Mul", [shifted, wm])                           # [1,10,30,30]
    if scale > 1:
        scales = g.f([4], [1.0, 1.0, float(scale), float(scale)])
        up = g.nd("Resize", [cropped, scales], mode="nearest")
        g.nd("Slice", [up, g.i64([0, 0]), g.i64([G, G]), g.i64([2, 3])], "output")
    else:
        g.nd("Identity", [cropped], "output")
    return _model(g)


def _try_bboxcrop(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    cols = set()
    for a, _ in det:
        cols |= set(int(v) for v in np.unique(a).tolist())
    for scale in (1, 2, 3):
        for bgc in [0] + sorted(cols):
            ok = True
            changed = False
            for a, b in det + allp:
                pred = _sim_bboxcrop(a, bgc, scale)
                if pred is None or pred.shape != b.shape or not np.array_equal(pred, b):
                    ok = False
                    break
                if a.shape != b.shape:
                    changed = True
            if ok and changed:
                try:
                    m = _build_bboxcrop(bgc, scale); _check(m)
                except Exception:
                    continue
                return (f"bboxcrop_bg{bgc}_s{scale}", m)
    return None


# =========================================================================== #
# TASK 3: recolour + vertical period extension to a constant output height     #
# =========================================================================== #
def _recolor_np(a, cmap):
    out = a.copy()
    for c in range(CHANNELS):
        out[a == c] = cmap[c]
    return out


def _decode_onehot(X, Ho, Wo):
    sel = X > 0.5
    cnt = sel.sum(0)
    grid = np.zeros((Ho, Wo), int)
    for r in range(Ho):
        for c in range(Wo):
            if cnt[r, c] != 1:
                return None
            grid[r, c] = int(np.argmax(sel[:, r, c]))
    # everything outside the Ho x Wo window must be empty
    mask = np.ones((G, G), bool)
    mask[:Ho, :Wo] = False
    if sel[:, mask].any():
        return None
    return grid


def _sim_vtile(a, cmap, Hout):
    ar = _recolor_np(a, cmap)
    X = _onehot(ar)
    realmask = X.sum(0)
    color = sum(c * X[c] for c in range(CHANNELS))
    p = _np_period(realmask, color, 0)
    if p == 0:
        return None
    Vf = _np_fill(X, p, 0)
    rowsel = (np.arange(G) < Hout).astype(np.float32)[None, :, None]
    out = Vf * rowsel
    return _decode_onehot(out, Hout, a.shape[1])


def _build_vtile(cmap, Hout):
    g = _G()
    halfc = g.f([1, 1, 1, 1], [0.5])
    half = g.f([1], [0.5])
    # recolor 1x1 conv
    w = [0.0] * (CHANNELS * CHANNELS)
    for i in range(CHANNELS):
        w[cmap[i] * CHANNELS + i] = 1.0
    wt = oh.make_tensor(g.nm("W"), F, [CHANNELS, CHANNELS, 1, 1], w)
    g.inits.append(wt)
    Xr = g.nd("Conv", ["input", wt.name], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    color = g.nd("ReduceSum", [g.nd("Mul", [Xr, colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", [Xr], axes=[1], keepdims=1)
    q = _period_scalar(g, realmask, color, 2, halfc)
    rowidx = g.f([1, 1, G, 1], list(range(G)))
    colidx = g.f([1, 1, 1, G], list(range(G)))
    Vf = _fill_axis(g, Xr, q, 2, rowidx, colidx, half)
    rsel = g.f([1, 1, G, 1], [1.0 if r < Hout else 0.0 for r in range(G)])
    g.nd("Mul", [Vf, rsel], "output")
    return _model(g)


def _detect_cmap(prs):
    cmap = {}
    for a, b in prs:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        for r in range(h):
            for c in range(w):
                k = int(a[r, c]); v = int(b[r, c])
                if k in cmap and cmap[k] != v:
                    return None
                cmap[k] = v
    full = list(range(CHANNELS))
    for k, v in cmap.items():
        if 0 <= k < CHANNELS and 0 <= v < CHANNELS:
            full[k] = v
        else:
            return None
    return full


def _try_vtile(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    outH = set(b.shape[0] for _, b in det)
    if len(outH) != 1:
        return None
    Hout = outH.pop()
    if any(b.shape[1] != a.shape[1] for a, b in det):
        return None
    if any(b.shape[0] <= a.shape[0] for a, b in det):  # must extend vertically
        return None
    cmap = _detect_cmap(det)
    if cmap is None:
        return None
    for a, b in det + allp:
        if b.shape[0] != Hout or b.shape[1] != a.shape[1]:
            return None
        pred = _sim_vtile(a, cmap, Hout)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_vtile(cmap, Hout); _check(m)
    except Exception:
        return None
    return (f"vtile_h{Hout}", m)


# =========================================================================== #
# TASK 343: horizontal period tiling -- extend a left-anchored periodic        #
# pattern rightward to fill each row's real width.                             #
# =========================================================================== #
def _sim_htile(a):
    X = _onehot(a)
    realmask = X.sum(0)
    color = sum(c * X[c] for c in range(CHANNELS))
    colored = (X[1:].sum(0) > 0.5).astype(np.float32)  # [30,30]
    colhas = colored.max(0)                            # [30]
    if colhas.sum() < 1:
        return None
    L = int(np.max(np.where(colhas > 0.5)[0]))
    # validmask[d][.,c] = 1 if c + d <= L
    cidx = np.arange(G)
    vmask = {}
    for d in range(1, QMAX + 1):
        vmask[d] = ((cidx + d) <= L).astype(np.float32)[None, :]  # broadcast rows
        vmask[d] = np.broadcast_to(vmask[d], (G, G)).astype(np.float32)
    q = _np_period(realmask, color, 1, vmask)
    if q == 0:
        return None
    Xc = X.copy(); Xc[0] = 0.0
    Hf = _np_fill(Xc, q, 1)
    Hf = Hf * realmask[None]
    colored_present = (Hf.sum(0) > 0.5)
    out = Hf.copy()
    out[0] = realmask * (1.0 - colored_present)
    return _decode_onehot(out, a.shape[0], a.shape[1])


def _build_htile():
    g = _G()
    halfc = g.f([1, 1, 1, 1], [0.5])
    half = g.f([1], [0.5])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    color = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    e_nobg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    colored = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", [g.nd("Mul", ["input", e_nobg])], axes=[1], keepdims=1),
                   halfc])], to=F)                                    # [1,1,30,30]
    colhas = g.nd("ReduceMax", [colored], axes=[2], keepdims=1)       # [1,1,1,30]
    cidx = g.f([1, 1, 1, G], list(range(G)))
    L = g.nd("ReduceMax", [g.nd("Mul", [cidx, colhas])], axes=[3], keepdims=1)  # [1,1,1,1]
    # validmask[d]: c + d <= L  ->  Less(cidx + d, L + 0.5)
    vmask = {}
    for d in range(1, QMAX + 1):
        cd = g.nd("Add", [cidx, g.f([1, 1, 1, 1], [float(d)])])       # [1,1,1,30]
        vmask[d] = g.nd("Cast", [g.nd("Less", [cd, g.nd("Add", [L, halfc])])], to=F)
    q = _period_scalar(g, realmask, color, 3, halfc, vmask)
    e0z = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    Xc = g.nd("Mul", ["input", e0z])                                 # colored channels only
    rowidx = g.f([1, 1, G, 1], list(range(G)))
    colidx = g.f([1, 1, 1, G], list(range(G)))
    Hf = _fill_axis(g, Xc, q, 3, rowidx, colidx, half)
    Hf = g.nd("Mul", [Hf, realmask])                                 # within real region
    present = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", [Hf], axes=[1], keepdims=1), halfc])], to=F)
    one = g.f([1, 1, 1, 1], [1.0])
    bg = g.nd("Mul", [realmask, g.nd("Sub", [one, present])])
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    g.nd("Add", [Hf, g.nd("Mul", [bg, e0])], "output")
    return _model(g)


def _try_htile(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    if all(np.array_equal(a, b) for a, b in det):
        return None
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        pred = _sim_htile(a)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_htile(); _check(m)
    except Exception:
        return None
    return ("htile", m)


# =========================================================================== #
# TASK 295: 1xW left-justified run -> growing left-justified staircase,        #
# height = floor(W/2)+, row r filled for columns c < init+r.                    #
# =========================================================================== #
def _sim_stair(a):
    H, W = a.shape
    if H != 1:
        return None
    colored = a[0] != 0
    init = int(colored.sum())
    if init < 1:
        return None
    if not np.array_equal(np.where(colored)[0], np.arange(init)):
        return None
    cs = set(int(v) for v in a[0][colored].tolist())
    if len(cs) != 1:
        return None
    C = cs.pop()
    outH = sum(1 for r in range(G) if r < W / 2.0)
    if outH < 1:
        return None
    grid = np.zeros((outH, W), int)
    for r in range(outH):
        for c in range(W):
            if c - r < init:
                grid[r, c] = C
    return grid


def _build_stair():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    e_nobg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    W = g.nd("ReduceSum", [realmask], axes=[2, 3], keepdims=1)           # [1,1,1,1]
    ch19 = g.nd("Mul", ["input", e_nobg])
    init = g.nd("ReduceSum", [ch19], axes=[1, 2, 3], keepdims=1)         # [1,1,1,1]
    rowidx = g.f([1, 1, G, 1], list(range(G)))
    colidx = g.f([1, 1, 1, G], list(range(G)))
    D = g.nd("Sub", [colidx, rowidx])                                    # c - r
    mask1 = g.nd("Cast", [g.nd("Less", [D, g.nd("Sub", [init, half])])], to=F)
    whalf = g.nd("Mul", [W, half])
    rowsel = g.nd("Cast", [g.nd("Less", [rowidx, whalf])], to=F)         # [1,1,30,1]
    colsel = g.nd("Cast", [g.nd("Less", [colidx, g.nd("Sub", [W, half])])], to=F)  # [1,1,1,30]
    realout = g.nd("Mul", [rowsel, colsel])                              # [1,1,30,30]
    tri = g.nd("Mul", [mask1, realout])
    cnt = g.nd("ReduceSum", [ch19], axes=[2, 3], keepdims=1)             # [1,10,1,1]
    sel = g.nd("Cast", [g.nd("Greater", [cnt, half])], to=F)
    colored_out = g.nd("Mul", [tri, sel])
    bg = g.nd("Mul", [realout, g.nd("Sub", [one, mask1])])
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    g.nd("Add", [colored_out, g.nd("Mul", [bg, e0])], "output")
    return _model(g)


def _try_stair(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape[0] != 1 for a, _ in det):
        return None
    for a, b in det + allp:
        pred = _sim_stair(a)
        if pred is None or pred.shape != b.shape or not np.array_equal(pred, b):
            return None
    try:
        m = _build_stair(); _check(m)
    except Exception:
        return None
    return ("stair", m)


# =========================================================================== #
# TASK 313: 2-colour checkerboard (+ marker frame) -> full-grid checkerboard    #
# with the two colours swapped, drawn over the whole real region.              #
# =========================================================================== #
def _sim_checker(a, swap=True):
    H, W = a.shape
    if H < 2 or W < 2:
        return None
    A = int(a[0, 0]); B = int(a[0, 1])
    if A == 0 or B == 0 or A == B:
        return None
    Eg = ((np.arange(H)[:, None] + np.arange(W)[None, :]) % 2 == 0)
    if swap:
        out = np.where(Eg, B, A)
    else:
        out = np.where(Eg, A, B)
    return out.astype(int)


def _build_checker(swap=True):
    g = _G()
    Eg = g.f([1, 1, G, G], [1.0 if (r + c) % 2 == 0 else 0.0 for r in range(G) for c in range(G)])
    Og = g.f([1, 1, G, G], [1.0 if (r + c) % 2 == 1 else 0.0 for r in range(G) for c in range(G)])
    # one-hot colour of the two checkerboard corners (0,0) even-phase and (0,1) odd-phase
    Asel = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([1, 1]), g.i64([2, 3])])  # [1,10,1,1]
    Bsel = g.nd("Slice", ["input", g.i64([0, 1]), g.i64([1, 2]), g.i64([2, 3])])
    even_col = Bsel if swap else Asel
    odd_col = Asel if swap else Bsel
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    ER = g.nd("Mul", [Eg, realmask])
    OR_ = g.nd("Mul", [Og, realmask])
    out_even = g.nd("Mul", [ER, even_col])
    out_odd = g.nd("Mul", [OR_, odd_col])
    g.nd("Add", [out_even, out_odd], "output")
    return _model(g)


def _try_checker(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    if all(np.array_equal(a, b) for a, b in det):
        return None
    for swap in (True, False):
        ok = True
        for a, b in det + allp:
            if a.shape != b.shape:
                ok = False; break
            pred = _sim_checker(a, swap)
            if pred is None or not np.array_equal(pred, b):
                ok = False; break
        if ok:
            try:
                m = _build_checker(swap); _check(m)
            except Exception:
                continue
            return (f"checker_s{int(swap)}", m)
    return None


# =========================================================================== #
# TASK 354: blobs of one colour M get flood-filled with the colour of the      #
# single "legend" pixel sitting in (a column of) the blob.                     #
# =========================================================================== #
def _np_shift2(a, dr, dc):
    out = np.zeros_like(a)
    H, W = a.shape
    rs = slice(max(0, dr), H + min(0, dr)); rd = slice(max(0, -dr), H + min(0, -dr))
    cs = slice(max(0, dc), W + min(0, dc)); cd = slice(max(0, -dc), W + min(0, -dc))
    out[rd, cd] = a[rs, cs]
    return out


def _sim_flood(a, M, steps=30):
    V = a.astype(int)
    maskM = (V == M)
    legcolor = np.where((V != 0) & (V != M), V, 0)
    legproj = legcolor.max(0, keepdims=True)
    cur = np.where(maskM, np.broadcast_to(legproj, V.shape), 0)
    for _ in range(steps):
        nb = cur.copy()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = np.maximum(nb, _np_shift2(cur, dr, dc))
        cur = np.where(maskM, nb, 0)
    return np.where(maskM, cur, V)


def _shift_pos(g, t, d, axis):
    """Shift content toward higher index by d (out[i]=t[i-d])."""
    if axis == 3:
        sl = g.nd("Slice", [t, g.i64([0]), g.i64([G - d]), g.i64([3])])
        return g.nd("Pad", [sl], mode="constant", value=0.0, pads=[0, 0, 0, d, 0, 0, 0, 0])
    sl = g.nd("Slice", [t, g.i64([0]), g.i64([G - d]), g.i64([2])])
    return g.nd("Pad", [sl], mode="constant", value=0.0, pads=[0, 0, d, 0, 0, 0, 0, 0])


def _build_flood(M, steps=30):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)  # [1,1,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    Mc = g.f([1, 1, 1, 1], [float(M)])
    maskM = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [V, Mc])]), half])], to=F)
    notM = g.nd("Sub", [one, maskM])
    nonbg = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)
    legcolor = g.nd("Mul", [g.nd("Mul", [V, notM]), nonbg])
    legproj = g.nd("ReduceMax", [legcolor], axes=[2], keepdims=1)            # [1,1,1,30]
    cur = g.nd("Mul", [maskM, legproj])
    for _ in range(steps):
        d_ = _shift_plane(g, cur, 1, 2)     # from below: out[r]=cur[r+1]
        u_ = _shift_pos(g, cur, 1, 2)       # from above
        l_ = _shift_plane(g, cur, 1, 3)     # from right
        r_ = _shift_pos(g, cur, 1, 3)       # from left
        nb = g.nd("Max", [cur, d_, u_, l_, r_])
        cur = g.nd("Mul", [nb, maskM])
    Vp = g.nd("Add", [g.nd("Mul", [V, notM]), cur])
    chrange = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    close = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [Vp, chrange])]), half])], to=F)
    g.nd("Mul", [close, realmask], "output")
    return _model(g)


def _try_flood(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    inc = set(); outc = set()
    for a, b in det:
        inc |= set(int(v) for v in np.unique(a).tolist())
        outc |= set(int(v) for v in np.unique(b).tolist())
    M = sorted(inc - outc)
    if len(M) != 1:
        return None
    M = M[0]
    if M == 0:
        return None
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        if not np.array_equal(_sim_flood(a, M), b):
            return None
    try:
        m = _build_flood(M); _check(m)
    except Exception:
        return None
    return (f"flood_m{M}", m)


# =========================================================================== #
# TASK 32: per-column/row gravity toward a grid edge (parallel falling-sand CA) #
# =========================================================================== #
def _sim_gravity(a, dr, dc, steps=30):
    H, W = a.shape
    V = a.astype(float)
    real = np.ones((H, W))

    def toward(x):
        return _np_shift2(x, dr, dc)   # value of neighbour in gravity direction

    def frm(x):
        return _np_shift2(x, -dr, -dc)  # value of opposite neighbour

    for _ in range(steps):
        colored = (V > 0.5).astype(float)
        tw_col = toward(colored); tw_real = toward(real)
        fromV = frm(V); fr_col = (fromV > 0.5).astype(float)
        curBg = real * (1 - colored)
        leave = colored * (1 - tw_col) * tw_real
        receive = curBg * fr_col
        V = V * (1 - leave) + fromV * receive
    return V.astype(int)


def _build_gravity(dr, dc, steps=30):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)

    # toward = neighbour in (dr,dc); realised by shifting content opposite way
    def toward(x):
        if dr == 1:
            return _shift_plane(g, x, 1, 2)
        if dr == -1:
            return _shift_pos(g, x, 1, 2)
        if dc == 1:
            return _shift_plane(g, x, 1, 3)
        return _shift_pos(g, x, 1, 3)

    def frm(x):
        if dr == 1:
            return _shift_pos(g, x, 1, 2)
        if dr == -1:
            return _shift_plane(g, x, 1, 2)
        if dc == 1:
            return _shift_pos(g, x, 1, 3)
        return _shift_plane(g, x, 1, 3)

    for _ in range(steps):
        colored = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)
        tw_col = toward(colored); tw_real = toward(realmask)
        fromV = frm(V); fr_col = g.nd("Cast", [g.nd("Greater", [fromV, half])], to=F)
        curBg = g.nd("Mul", [realmask, g.nd("Sub", [one, colored])])
        leave = g.nd("Mul", [g.nd("Mul", [colored, g.nd("Sub", [one, tw_col])]), tw_real])
        receive = g.nd("Mul", [curBg, fr_col])
        V = g.nd("Add", [g.nd("Mul", [V, g.nd("Sub", [one, leave])]),
                         g.nd("Mul", [fromV, receive])])
    chrange = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    close = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [V, chrange])]), half])], to=F)
    g.nd("Mul", [close, realmask], "output")
    return _model(g)


def _try_gravity(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    if all(np.array_equal(a, b) for a, b in det):
        return None
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ok = True
        for a, b in det:
            if a.shape != b.shape or not np.array_equal(_sim_gravity(a, dr, dc), b):
                ok = False; break
        if not ok:
            continue
        for a, b in allp:
            if a.shape != b.shape or not np.array_equal(_sim_gravity(a, dr, dc), b):
                ok = False; break
        if ok:
            try:
                m = _build_gravity(dr, dc); _check(m)
            except Exception:
                continue
            return (f"gravity_{dr}_{dc}", m)
    return None


# =========================================================================== #
# TASK 126: each downward-opening "cup" drops a marker straight to the floor    #
# in the column of its opening.                                                 #
# =========================================================================== #
def _sim_cupdrop(a, M):
    V = a.astype(int); H, W = a.shape
    colored = (V > 0).astype(int)
    real = np.ones((H, W), int)
    bg = real * (1 - colored)
    up = _np_shift2(colored, -1, 0)
    left = _np_shift2(colored, 0, -1)
    right = _np_shift2(colored, 0, 1)
    opening = bg * up * left * right
    opcol = opening.max(0, keepdims=True)
    floor = np.zeros((H, W), int); floor[H - 1, :] = 1
    four = floor * np.broadcast_to(opcol, (H, W))
    out = V.copy()
    if (four * (V != 0)).any():    # would overwrite content -> not this rule
        return None
    out[four > 0] = M
    return out


def _build_cupdrop(M):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    colored = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)
    up = _shift_pos(g, colored, 1, 2)
    left = _shift_pos(g, colored, 1, 3)
    right = _shift_plane(g, colored, 1, 3)
    bg = g.nd("Mul", [realmask, g.nd("Sub", [one, colored])])
    opening = g.nd("Mul", [g.nd("Mul", [bg, up]), g.nd("Mul", [left, right])])
    opcol = g.nd("ReduceMax", [opening], axes=[2], keepdims=1)            # [1,1,1,30]
    rowidx = g.f([1, 1, G, 1], list(range(G)))
    rowhas = g.nd("ReduceMax", [realmask], axes=[3], keepdims=1)          # [1,1,30,1]
    last = g.nd("ReduceMax", [g.nd("Mul", [rowidx, rowhas])], axes=[2], keepdims=1)
    floor = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, last])]), half])], to=F)
    four = g.nd("Mul", [floor, opcol])                                    # [1,1,30,30]
    de = g.f([1, CHANNELS, 1, 1], [-1.0 if c == 0 else (1.0 if c == M else 0.0)
                                   for c in range(CHANNELS)])
    g.nd("Add", ["input", g.nd("Mul", [four, de])], "output")
    return _model(g)


def _try_cupdrop(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    inc = set(); outc = set()
    for a, b in det:
        inc |= set(int(v) for v in np.unique(a).tolist())
        outc |= set(int(v) for v in np.unique(b).tolist())
    M = sorted(outc - inc)
    if len(M) != 1 or M[0] == 0:
        return None
    M = M[0]
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        pred = _sim_cupdrop(a, M)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_cupdrop(M); _check(m)
    except Exception:
        return None
    return (f"cupdrop_m{M}", m)


# =========================================================================== #
# TASK 278: outline 4-connected clusters (size>=2) of colour C with a colour-M  #
# 8-neighbourhood frame.                                                        #
# =========================================================================== #
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _sim_frame(a, C, M):
    V = a.astype(int); H, W = a.shape
    is2 = (V == C).astype(int)
    nb2 = np.zeros((H, W), int)
    for dr, dc in _N4:
        nb2 = nb2 + _np_shift2(is2, dr, dc)
    active = is2 * (nb2 >= 1)
    nbA = np.zeros((H, W), int)
    for dr, dc in _N8:
        nbA = np.maximum(nbA, _np_shift2(active, dr, dc))
    bg = (V == 0).astype(int)
    frame = bg * nbA
    out = V.copy(); out[frame > 0] = M
    return out


def _onnx_shift(g, t, dr, dc):
    """out[r,c] = t[r+dr, c+dc] (zero outside)."""
    x = t
    if dr == 1:
        x = _shift_plane(g, x, 1, 2)
    elif dr == -1:
        x = _shift_pos(g, x, 1, 2)
    if dc == 1:
        x = _shift_plane(g, x, 1, 3)
    elif dc == -1:
        x = _shift_pos(g, x, 1, 3)
    return x


def _build_frame(C, M):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    Cc = g.f([1, 1, 1, 1], [float(C)])
    is2 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [V, Cc])]), half])], to=F)
    nb2 = None
    for dr, dc in _N4:
        s = _onnx_shift(g, is2, dr, dc)
        nb2 = s if nb2 is None else g.nd("Add", [nb2, s])
    active = g.nd("Mul", [is2, g.nd("Cast", [g.nd("Greater", [nb2, half])], to=F)])
    nbA = None
    for dr, dc in _N8:
        s = _onnx_shift(g, active, dr, dc)
        nbA = s if nbA is None else g.nd("Max", [nbA, s])
    colored = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)
    bg = g.nd("Mul", [realmask, g.nd("Sub", [one, colored])])
    frame = g.nd("Mul", [bg, nbA])
    de = g.f([1, CHANNELS, 1, 1], [-1.0 if c == 0 else (1.0 if c == M else 0.0)
                                   for c in range(CHANNELS)])
    g.nd("Add", ["input", g.nd("Mul", [frame, de])], "output")
    return _model(g)


def _try_frame(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    inc = set(); outc = set()
    for a, b in det:
        inc |= set(int(v) for v in np.unique(a).tolist())
        outc |= set(int(v) for v in np.unique(b).tolist())
    added = sorted(outc - inc)
    base = sorted((inc - {0}))
    if len(added) != 1 or added[0] == 0 or len(base) != 1:
        return None
    M = added[0]; C = base[0]
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        if not np.array_equal(_sim_frame(a, C, M), b):
            return None
    try:
        m = _build_frame(C, M); _check(m)
    except Exception:
        return None
    return (f"frame_c{C}_m{M}", m)


# =========================================================================== #
# TASK 151: stamp the 8-neighbourhood of a +cross intersection with colour M.   #
# =========================================================================== #
def _sim_cstamp(a, M):
    V = a.astype(int); H, W = a.shape
    colored = (V > 0).astype(int)
    hrow = (colored.sum(1) > 1).astype(int)
    vcol = (colored.sum(0) > 1).astype(int)
    if hrow.sum() != 1 or vcol.sum() != 1:
        return None
    inter = np.outer(hrow, vcol)
    ring = np.zeros((H, W), int)
    for dr, dc in _N8:
        ring = np.maximum(ring, _np_shift2(inter, dr, dc))
    ring = ring * (1 - inter)
    out = V.copy(); out[ring > 0] = M
    return out


def _build_cstamp(M):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    colored = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)
    rowsum = g.nd("ReduceSum", [colored], axes=[3], keepdims=1)
    colsum = g.nd("ReduceSum", [colored], axes=[2], keepdims=1)
    hrow = g.nd("Cast", [g.nd("Greater", [rowsum, g.f([1, 1, 1, 1], [1.5])])], to=F)
    vcol = g.nd("Cast", [g.nd("Greater", [colsum, g.f([1, 1, 1, 1], [1.5])])], to=F)
    inter = g.nd("Mul", [hrow, vcol])
    ring = None
    for dr, dc in _N8:
        s = _onnx_shift(g, inter, dr, dc)
        ring = s if ring is None else g.nd("Max", [ring, s])
    ring = g.nd("Mul", [g.nd("Mul", [ring, g.nd("Sub", [one, inter])]), realmask])
    eM = g.f([1, CHANNELS, 1, 1], [1.0 if c == M else 0.0 for c in range(CHANNELS)])
    keep = g.nd("Mul", ["input", g.nd("Sub", [one, ring])])
    g.nd("Add", [keep, g.nd("Mul", [ring, eM])], "output")
    return _model(g)


def _try_cstamp(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    inc = set(); outc = set()
    for a, b in det:
        inc |= set(int(v) for v in np.unique(a).tolist())
        outc |= set(int(v) for v in np.unique(b).tolist())
    added = sorted(outc - inc)
    if len(added) != 1 or added[0] == 0:
        return None
    M = added[0]
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        pred = _sim_cstamp(a, M)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_cstamp(M); _check(m)
    except Exception:
        return None
    return (f"cstamp_m{M}", m)


# =========================================================================== #
# TASK 55: two vertical + two horizontal full lines form a tic-tac-toe; the     #
# 5 plus-regions (centre/up/down/left/right) get filled with fixed colours.     #
# =========================================================================== #
def _ttt_lines(a, L):
    V = a.astype(int); H, W = a.shape
    is8 = (V == L)
    vcols = [c for c in range(W) if is8[:, c].all() and is8[:, c].any()]
    hrows = [r for r in range(H) if is8[r, :].all() and is8[r, :].any()]
    if len(vcols) != 2 or len(hrows) != 2:
        return None
    return min(vcols), max(vcols), min(hrows), max(hrows)


def _sim_ttt(a, L, cc, cu, cd, cl, cr):
    V = a.astype(int); H, W = a.shape
    ln = _ttt_lines(a, L)
    if ln is None:
        return None
    lc, rc, tr, br = ln
    out = V.copy()
    for r in range(H):
        for c in range(W):
            if V[r, c] != 0:
                continue
            bv = lc < c < rc; bh = tr < r < br; col = 0
            if bv and bh:
                col = cc
            elif bv and r < tr:
                col = cu
            elif bv and r > br:
                col = cd
            elif bh and c < lc:
                col = cl
            elif bh and c > rc:
                col = cr
            if col:
                out[r, c] = col
    return out


def _build_ttt(L, cc, cu, cd, cl, cr):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    BIG = g.f([1, 1, 1, 1], [100.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    Lc = g.f([1, 1, 1, 1], [float(L)])
    is8 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [V, Lc])]), half])], to=F)
    colored = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)
    # full vertical lines
    c8col = g.nd("ReduceSum", [is8], axes=[2], keepdims=1)        # [1,1,1,30]
    crcol = g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)
    vline = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [c8col, half])], to=F),
                         g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [c8col, crcol])]), half])], to=F)])
    c8row = g.nd("ReduceSum", [is8], axes=[3], keepdims=1)        # [1,1,30,1]
    crrow = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)
    hline = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [c8row, half])], to=F),
                         g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [c8row, crrow])]), half])], to=F)])
    colidx = g.f([1, 1, 1, G], list(range(G)))
    rowidx = g.f([1, 1, G, 1], list(range(G)))
    # lc/rc from vline columns
    lc = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [colidx, vline]),
              g.nd("Mul", [BIG, g.nd("Sub", [one, vline])])])], axes=[3], keepdims=1)
    rc = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [colidx, vline]),
              g.nd("Mul", [BIG, g.nd("Sub", [one, vline])])])], axes=[3], keepdims=1)
    tr = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [rowidx, hline]),
              g.nd("Mul", [BIG, g.nd("Sub", [one, hline])])])], axes=[2], keepdims=1)
    br = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [rowidx, hline]),
              g.nd("Mul", [BIG, g.nd("Sub", [one, hline])])])], axes=[2], keepdims=1)
    band_v = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [colidx, lc])], to=F),
                          g.nd("Cast", [g.nd("Less", [colidx, rc])], to=F)])   # [1,1,1,30]
    band_h = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [rowidx, tr])], to=F),
                          g.nd("Cast", [g.nd("Less", [rowidx, br])], to=F)])   # [1,1,30,1]
    r_lt_tr = g.nd("Cast", [g.nd("Less", [rowidx, tr])], to=F)
    r_gt_br = g.nd("Cast", [g.nd("Greater", [rowidx, br])], to=F)
    c_lt_lc = g.nd("Cast", [g.nd("Less", [colidx, lc])], to=F)
    c_gt_rc = g.nd("Cast", [g.nd("Greater", [colidx, rc])], to=F)
    bg = g.nd("Mul", [realmask, g.nd("Sub", [one, colored])])
    center = g.nd("Mul", [bg, g.nd("Mul", [band_v, band_h])])
    up = g.nd("Mul", [bg, g.nd("Mul", [band_v, r_lt_tr])])
    down = g.nd("Mul", [bg, g.nd("Mul", [band_v, r_gt_br])])
    left = g.nd("Mul", [bg, g.nd("Mul", [band_h, c_lt_lc])])
    right = g.nd("Mul", [bg, g.nd("Mul", [band_h, c_gt_rc])])

    def de(col):
        return g.f([1, CHANNELS, 1, 1], [-1.0 if c == 0 else (1.0 if c == col else 0.0)
                                         for c in range(CHANNELS)])
    acc = "input"
    for region, col in [(center, cc), (up, cu), (down, cd), (left, cl), (right, cr)]:
        acc = g.nd("Add", [acc, g.nd("Mul", [region, de(col)])])
    g.nd("Identity", [acc], "output")
    return _model(g)


def _try_ttt(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    # detect line colour L and the 5 region colours from train
    a0, b0 = det[0]
    H, W = a0.shape
    params = None
    for L in range(1, CHANNELS):
        ln = _ttt_lines(a0, L)
        if ln is None:
            continue
        lc, rc, tr, br = ln

        def sample(cond):
            for r in range(H):
                for c in range(W):
                    if a0[r, c] == 0 and cond(r, c):
                        return int(b0[r, c])
            return 0
        cc = sample(lambda r, c: lc < c < rc and tr < r < br)
        cu = sample(lambda r, c: lc < c < rc and r < tr)
        cd = sample(lambda r, c: lc < c < rc and r > br)
        cl = sample(lambda r, c: tr < r < br and c < lc)
        cr = sample(lambda r, c: tr < r < br and c > rc)
        params = (L, cc, cu, cd, cl, cr)
        break
    if params is None:
        return None
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        pred = _sim_ttt(a, *params)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_ttt(*params); _check(m)
    except Exception:
        return None
    return ("ttt", m)


# =========================================================================== #
# TASK 368: replace every solid blob of colour M5 with a copy of the (single)   #
# multicolour template found elsewhere in the grid (runtime template -> stamp    #
# via ConvTranspose at each blob's top-left corner).                            #
# =========================================================================== #
def _sim_stamp368(a, M5):
    V = a.astype(int); H, W = a.shape
    tmask = (V != 0) & (V != M5)
    if not tmask.any():
        return None
    rs = np.where(tmask.any(1))[0]; cs = np.where(tmask.any(0))[0]
    tr0, tr1, tc0, tc1 = rs.min(), rs.max(), cs.min(), cs.max()
    T = V[tr0:tr1 + 1, tc0:tc1 + 1].copy()
    if (T == M5).any():
        return None
    th, tw = T.shape
    is5 = (V == M5).astype(int)
    up = _np_shift2(is5, -1, 0); left = _np_shift2(is5, 0, -1)
    corner = is5 * (1 - up) * (1 - left)
    out = V.copy(); out[V == M5] = 0
    for r in range(H):
        for c in range(W):
            if corner[r, c] and r + th <= H and c + tw <= W:
                out[r:r + th, c:c + tw] = T
    return out


def _build_stamp368(M5):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5]); one = g.f([1, 1, 1, 1], [1.0]); BIG = g.f([1, 1, 1, 1], [100.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    M5c = g.f([1, 1, 1, 1], [float(M5)])
    is5 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [V, M5c])]), half])], to=F)
    ekeep = g.f([1, CHANNELS, 1, 1], [0.0 if c == M5 else 1.0 for c in range(CHANNELS)])
    Xno5 = g.nd("Mul", ["input", ekeep])
    up = _shift_pos(g, is5, 1, 2); left = _shift_pos(g, is5, 1, 3)
    corner = g.nd("Mul", [g.nd("Mul", [is5, g.nd("Sub", [one, up])]), g.nd("Sub", [one, left])])
    colored = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)
    tmask = g.nd("Mul", [colored, g.nd("Sub", [one, is5])])
    rowhas = g.nd("ReduceMax", [tmask], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [tmask], axes=[2], keepdims=1)
    ri = g.f([1, 1, G, 1], list(range(G))); ci = g.f([1, 1, 1, G], list(range(G)))
    rmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [ri, rowhas]),
               g.nd("Mul", [BIG, g.nd("Sub", [one, rowhas])])])], axes=[2], keepdims=1)
    rmax = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [ri, rowhas]),
               g.nd("Mul", [BIG, g.nd("Sub", [one, rowhas])])])], axes=[2], keepdims=1)
    cmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [ci, colhas]),
               g.nd("Mul", [BIG, g.nd("Sub", [one, colhas])])])], axes=[3], keepdims=1)
    cmax = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [ci, colhas]),
               g.nd("Mul", [BIG, g.nd("Sub", [one, colhas])])])], axes=[3], keepdims=1)
    h = g.nd("Add", [g.nd("Sub", [rmax, rmin]), one])
    w_ = g.nd("Add", [g.nd("Sub", [cmax, cmin]), one])
    gi = g.f([1, 1, G, 1], list(range(G))); gr = g.f([1, 1, 1, G], list(range(G)))
    dRA = g.nd("Sub", [gr, gi])
    Srow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [dRA, rmin])]), half])], to=F)
    gc = g.f([1, 1, G, 1], list(range(G))); gj = g.f([1, 1, 1, G], list(range(G)))
    dCJ = g.nd("Sub", [gc, gj])
    Scol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [dCJ, cmin])]), half])], to=F)
    shifted = g.nd("MatMul", [Srow, g.nd("MatMul", [Xno5, Scol])])
    rowsel = g.nd("Cast", [g.nd("Less", [ri, g.nd("Sub", [h, half])])], to=F)
    colsel = g.nd("Cast", [g.nd("Less", [ci, g.nd("Sub", [w_, half])])], to=F)
    Tw = g.nd("Mul", [shifted, g.nd("Mul", [rowsel, colsel])])     # [1,10,30,30] weight
    stamped = g.nd("ConvTranspose", [corner, Tw], strides=[1, 1], pads=[0, 0, 0, 0])
    cropped = g.nd("Slice", [stamped, g.i64([0, 0]), g.i64([G, G]), g.i64([2, 3])])
    g.nd("Add", [Xno5, cropped], "output")
    return _model(g)


def _try_stamp368(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    inc = set(); outc = set()
    for a, b in det:
        inc |= set(int(v) for v in np.unique(a).tolist())
        outc |= set(int(v) for v in np.unique(b).tolist())
    rem = sorted(inc - outc)
    if len(rem) != 1 or rem[0] == 0:
        return None
    M5 = rem[0]
    for a, b in det + allp:
        if a.shape != b.shape:
            return None
        pred = _sim_stamp368(a, M5)
        if pred is None or not np.array_equal(pred, b):
            return None
    try:
        m = _build_stamp368(M5); _check(m)
    except Exception:
        return None
    return (f"stamp_m{M5}", m)


# =========================================================================== #
# TASK 70: recolour cells of colour Cin to Cto inside the bounding box of       #
# colour B.                                                                     #
# =========================================================================== #
def _sim_boxrecolor(a, B, Cin, Cto):
    V = a.astype(int); H, W = a.shape
    isB = (V == B)
    if not isB.any():
        return None
    rs = np.where(isB.any(1))[0]; cs = np.where(isB.any(0))[0]
    r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
    out = V.copy()
    box = np.zeros((H, W), bool); box[r0:r1 + 1, c0:c1 + 1] = True
    out[box & (V == Cin)] = Cto
    return out


def _build_boxrecolor(B, Cin, Cto):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5]); one = g.f([1, 1, 1, 1], [1.0]); BIG = g.f([1, 1, 1, 1], [100.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    Bc = g.f([1, 1, 1, 1], [float(B)])
    isB = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [V, Bc])]), half])], to=F)
    rowhas = g.nd("ReduceMax", [isB], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [isB], axes=[2], keepdims=1)
    ri = g.f([1, 1, G, 1], list(range(G))); ci = g.f([1, 1, 1, G], list(range(G)))
    r0 = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [ri, rowhas]),
              g.nd("Mul", [BIG, g.nd("Sub", [one, rowhas])])])], axes=[2], keepdims=1)
    r1 = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [ri, rowhas]),
              g.nd("Mul", [BIG, g.nd("Sub", [one, rowhas])])])], axes=[2], keepdims=1)
    c0 = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [ci, colhas]),
              g.nd("Mul", [BIG, g.nd("Sub", [one, colhas])])])], axes=[3], keepdims=1)
    c1 = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [ci, colhas]),
              g.nd("Mul", [BIG, g.nd("Sub", [one, colhas])])])], axes=[3], keepdims=1)
    rin = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [ri, g.nd("Sub", [r0, half])])], to=F),
                       g.nd("Cast", [g.nd("Less", [ri, g.nd("Add", [r1, half])])], to=F)])  # [1,1,30,1]
    cin = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [ci, g.nd("Sub", [c0, half])])], to=F),
                       g.nd("Cast", [g.nd("Less", [ci, g.nd("Add", [c1, half])])], to=F)])  # [1,1,1,30]
    box = g.nd("Mul", [rin, cin])                                       # [1,1,30,30]
    Cinc = g.f([1, 1, 1, 1], [float(Cin)])
    isCin = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [V, Cinc])]), half])], to=F)
    tgt = g.nd("Mul", [box, isCin])                                     # [1,1,30,30]
    de = g.f([1, CHANNELS, 1, 1], [(-1.0 if c == Cin else (1.0 if c == Cto else 0.0))
                                   for c in range(CHANNELS)])
    g.nd("Add", ["input", g.nd("Mul", [tgt, de])], "output")
    return _model(g)


def _try_boxrecolor(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    frm = set(); to = set()
    for a, b in det:
        ch = (a != b)
        if ch.any():
            frm |= set(int(v) for v in a[ch].tolist())
            to |= set(int(v) for v in b[ch].tolist())
    if len(frm) != 1 or len(to) != 1:
        return None
    Cin = frm.pop(); Cto = to.pop()
    cols = set()
    for a, _ in det:
        cols |= set(int(v) for v in np.unique(a).tolist())
    for B in sorted(cols - {Cin, Cto, 0}):
        ok = True
        for a, b in det + allp:
            if a.shape != b.shape:
                ok = False; break
            pred = _sim_boxrecolor(a, B, Cin, Cto)
            if pred is None or not np.array_equal(pred, b):
                ok = False; break
        if ok:
            try:
                m = _build_boxrecolor(B, Cin, Cto); _check(m)
            except Exception:
                continue
            return (f"boxrecolor_b{B}_{Cin}to{Cto}", m)
    return None


# =========================================================================== #
# TASK 224: four marker cells (colour M) at the extremes define a rectangle;     #
# draw its outline (shrunk inward by 1) in the shape's colour C (runtime).       #
# =========================================================================== #
def _sim_markerframe(a, M):
    V = a.astype(int); H, W = a.shape
    cols = set(int(v) for v in np.unique(V).tolist()) - {0, M}
    if len(cols) != 1:
        return None
    C = cols.pop()
    isM = (V == M)
    if isM.sum() < 2:
        return None
    rs = np.where(isM.any(1))[0]; cs = np.where(isM.any(0))[0]
    rt, rb, cl, cr = rs.min() + 1, rs.max() - 1, cs.min() + 1, cs.max() - 1
    out = V.copy()
    for r in range(H):
        for c in range(W):
            if rt <= r <= rb and cl <= c <= cr and (r == rt or r == rb or c == cl or c == cr):
                if V[r, c] == 0:
                    out[r, c] = C
    return out


def _build_markerframe(M):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5]); one = g.f([1, 1, 1, 1], [1.0]); BIG = g.f([1, 1, 1, 1], [100.0])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    V = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    Mc = g.f([1, 1, 1, 1], [float(M)])
    isM = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [V, Mc])]), half])], to=F)
    rowhas = g.nd("ReduceMax", [isM], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [isM], axes=[2], keepdims=1)
    ri = g.f([1, 1, G, 1], list(range(G))); ci = g.f([1, 1, 1, G], list(range(G)))
    top = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [ri, rowhas]),
               g.nd("Mul", [BIG, g.nd("Sub", [one, rowhas])])])], axes=[2], keepdims=1)
    bot = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [ri, rowhas]),
               g.nd("Mul", [BIG, g.nd("Sub", [one, rowhas])])])], axes=[2], keepdims=1)
    lft = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [ci, colhas]),
               g.nd("Mul", [BIG, g.nd("Sub", [one, colhas])])])], axes=[3], keepdims=1)
    rgt = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [ci, colhas]),
               g.nd("Mul", [BIG, g.nd("Sub", [one, colhas])])])], axes=[3], keepdims=1)
    rt = g.nd("Add", [top, one]); rb = g.nd("Sub", [bot, one])
    cl = g.nd("Add", [lft, one]); cr = g.nd("Sub", [rgt, one])
    in_r = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [ri, g.nd("Sub", [rt, half])])], to=F),
                        g.nd("Cast", [g.nd("Less", [ri, g.nd("Add", [rb, half])])], to=F)])  # [1,1,30,1]
    in_c = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [ci, g.nd("Sub", [cl, half])])], to=F),
                        g.nd("Cast", [g.nd("Less", [ci, g.nd("Add", [cr, half])])], to=F)])  # [1,1,1,30]
    rb_mask = g.nd("Max", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ri, rt])]), half])], to=F),
                           g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ri, rb])]), half])], to=F)])
    cb_mask = g.nd("Max", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ci, cl])]), half])], to=F),
                           g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ci, cr])]), half])], to=F)])
    rb_full = g.nd("Mul", [rb_mask, in_c])    # top/bottom edges, within col range
    cb_full = g.nd("Mul", [cb_mask, in_r])    # left/right edges, within row range
    outline = g.nd("Max", [rb_full, cb_full])  # [1,1,30,30]
    colored = g.nd("Cast", [g.nd("Greater", [V, half])], to=F)
    bg = g.nd("Mul", [realmask, g.nd("Sub", [one, colored])])
    outbg = g.nd("Mul", [outline, bg])
    e_keep = g.f([1, CHANNELS, 1, 1], [0.0 if (c == 0 or c == M) else 1.0 for c in range(CHANNELS)])
    cnt = g.nd("ReduceSum", [g.nd("Mul", ["input", e_keep])], axes=[2, 3], keepdims=1)
    sel = g.nd("Cast", [g.nd("Greater", [cnt, half])], to=F)            # [1,10,1,1] shape colour
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    delta = g.nd("Mul", [outbg, g.nd("Sub", [sel, e0])])
    g.nd("Add", ["input", delta], "output")
    return _model(g)


def _try_markerframe(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return None
    if any(a.shape != b.shape for a, b in det):
        return None
    cols = set()
    for a, _ in det:
        cols |= set(int(v) for v in np.unique(a).tolist())
    for M in sorted(cols - {0}):
        ok = True
        for a, b in det + allp:
            if a.shape != b.shape:
                ok = False; break
            pred = _sim_markerframe(a, M)
            if pred is None or not np.array_equal(pred, b):
                ok = False; break
        if ok:
            try:
                m = _build_markerframe(M); _check(m)
            except Exception:
                continue
            return (f"markerframe_m{M}", m)
    return None


# =========================================================================== #
def candidates(ex):
    out = []
    for fn in (_try_upk, _try_cross, _try_bboxcrop, _try_vtile, _try_htile,
               _try_stair, _try_checker, _try_flood, _try_gravity, _try_cupdrop,
               _try_frame, _try_cstamp, _try_ttt, _try_stamp368, _try_boxrecolor,
               _try_markerframe):
        try:
            r = fn(ex)
        except Exception:
            r = None
        if r is not None:
            out.append(r)
    return out
