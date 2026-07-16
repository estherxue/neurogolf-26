"""Local-rule COMPOSITIONS: chain a learned local conv (denoise / outline /
neighbour-rule, exactly as ``family_localconv`` / ``family_local2`` fit it) with a
recolor and/or an origin-safe geometric op, building ONE end-to-end opset-10 graph.

What composes with a local conv -- and what does NOT
---------------------------------------------------
A "local rule" maps every output cell to a fixed function of its (2R+1)x(2R+1)
input neighbourhood.  We realise it as a single small ``Conv`` (when each output
colour is linearly separable in the neighbourhood) or, when it is not (XOR-like
neighbour logic, lone exact-match patterns), as a compact ``Conv->Relu->Conv`` --
the same two backends ``family_localconv`` and ``family_local2`` use.  Either way the
>0-thresholded final channels are the one-hot local-rule output, and out-of-grid /
padding neighbourhoods are driven to "no colour".

The grader thresholds ``output > 0`` only at the very END of the graph, so:

  * ``recolor o localrule`` and ``localrule o recolor`` are compositions of *linear*
    maps with a single trailing threshold, hence they FUSE into one local conv --
    fitting the conv directly on (input neighbourhood -> final colour) already
    expresses them.  This module emits that fused conv as the same-shape member
    (covering both recolor orders whenever the fused rule is separable).

  * ``transpose o localrule`` genuinely needs the ``Transpose``: output cell (i,j)
    depends on the neighbourhood around (j,i), which a position-local conv cannot
    see.  We fit the local rule to the *transposed* target (``b.T``), then
    ``Transpose`` (perm [0,1,3,2]).

  * ``upscale_k o localrule`` changes the shape (k*H x k*W) -- a same-shape conv
    cannot.  We fit the local rule to the k-subsampled target (``b[::k,::k]``), then
    block-replicate with ``Resize(nearest, k)``.  ``Resize`` only copies values, so
    thresholding before/after the upscale is identical -> still exact.

Both spatial ops are ORIGIN-SAFE (Transpose swaps axes; the upscale slices the
top-left feeder then Resizes + crops), so they generalise across the variable,
top-left-anchored, zero-padded grid sizes.  The conv pads exactly like the grader's
own conv, so out-of-grid neighbourhoods == zero padding and are forced to "no
colour" -> padding cells stay all-zero through the whole chain.

Anti-overfit: the rule is fit only from neighbourhood structure (a deterministic
local function; integer-margin half-spaces, each kept pure), capped to small radius
(K=3, else 5) and a bounded number of distinct neighbourhoods / hidden units -- no
per-example memorisation.  The integrator fits on 70% of arc-gen and grades on 100%;
a genuine local rule survives.  Every candidate is re-validated for EXACT equality.
"""
from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64

_MAX_UNIQ = 4000            # cap on distinct neighbourhoods (single-conv backend)
_PERC_ITERS = 400           # batch-perceptron sweeps (single conv)
_RELU_CAP = {3: 96}         # hidden-unit cap (relu backend); compact = high points
_RELU_MAX_UNIQ = 4000       # skip the relu backend above this many neighbourhoods
_GREEDY_ITERS = 16          # greedy half-space steps per colour before exact units
_PURE_ITERS = 120           # perceptron sweeps inside the relu half-space fitter


# --------------------------------------------------------------------------- #
# model / graph helpers                                                         #
# --------------------------------------------------------------------------- #
def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


class _G:
    """Node/initializer accumulator with auto-unique tensor names."""

    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init(self, dtype, dims, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, dtype, list(dims), list(vals)))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    def finish(self):
        self.nodes[-1].output[0] = "output"
        return _model(self.nodes, self.inits)


# --------------------------------------------------------------------------- #
# one-hot / neighbourhood helpers (identical conventions to family_localconv)  #
# --------------------------------------------------------------------------- #
def _onehot(grid):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = grid.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (grid == c)
    return t


def _neighborhoods(t, K):
    """t (C,30,30) -> (900, C*K*K) int8; feature index = c*K*K + kr*K + kc,
    matching Conv weight layout W[o,c,kr,kc] with offset (kr-K//2, kc-K//2)."""
    r = K // 2
    pad = np.zeros((CHANNELS, HEIGHT + 2 * r, WIDTH + 2 * r), np.float32)
    pad[:, r:r + HEIGHT, r:r + WIDTH] = t
    feats = np.empty((CHANNELS, K, K, HEIGHT, WIDTH), np.int8)
    for kr in range(K):
        for kc in range(K):
            feats[:, kr, kc] = pad[:, kr:kr + HEIGHT, kc:kc + WIDTH].astype(np.int8)
    return feats.reshape(CHANNELS * K * K, HEIGHT * WIDTH).T.copy()


def _target_colors(grid):
    """(900,) int8: colour 0..9 for real cells, -1 for padding cells."""
    lab = np.full((HEIGHT, WIDTH), -1, np.int8)
    h, w = grid.shape
    lab[:h, :w] = grid
    return lab.reshape(-1)


def _unique_rows(A):
    A = np.ascontiguousarray(A)
    v = A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
    _, idx = np.unique(v, return_index=True)
    return A[idx]


def _distinct(tins, tgts, K):
    """Distinct (neighbourhood, colour) at radius K, with an explicit all-zero
    deep-padding row mapped to 'no colour'. Returns (Xu[int8], colour[int]) or None
    if the rule is not a function at this radius / too large."""
    F = CHANNELS * K * K
    Xs, ys = [], []
    for ti, tg in zip(tins, tgts):
        Xs.append(_neighborhoods(ti, K))
        ys.append(_target_colors(tg))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    X = np.concatenate([X, np.zeros((1, F), np.int8)], 0)
    y = np.concatenate([y, np.array([-1], np.int8)], 0)
    xy = np.concatenate([X, (y + 1).astype(np.int8)[:, None]], 1)  # label -> 0..10
    uxy = _unique_rows(xy)
    ux = _unique_rows(X)
    if uxy.shape[0] != ux.shape[0]:
        return None                       # same neighbourhood -> two colours
    return uxy[:, :-1].astype(np.int8), (uxy[:, -1].astype(np.int64) - 1)


# --------------------------------------------------------------------------- #
# backend 1: single Conv (margin perceptron, linearly separable colours)       #
# --------------------------------------------------------------------------- #
def _fit_separator(X, z, max_iter=_PERC_ITERS):
    F = X.shape[1]
    w = np.zeros(F, np.float64)
    b = 0.0
    for _ in range(max_iter):
        s = X @ w + b
        viol = z * s < 1.0
        if not viol.any():
            break
        zv = z[viol]
        w += zv @ X[viol]
        b += zv.sum()
    s = X @ w + b
    if np.all(((z > 0) & (s > 0)) | ((z < 0) & (s <= 0))):
        return w, b
    return None


def _fit_conv(tins, tgts, K):
    """Fit one Conv (kernel K). Returns (W[10,10,K,K], B[10]) or None."""
    d = _distinct(tins, tgts, K)
    if d is None:
        return None
    Xu8, cu = d
    if Xu8.shape[0] > _MAX_UNIQ:
        return None
    Xu = Xu8.astype(np.float64)
    F = CHANNELS * K * K
    W = np.zeros((CHANNELS, F), np.float64)
    B = np.zeros(CHANNELS, np.float64)
    for o in range(CHANNELS):
        z = np.where(cu == o, 1.0, -1.0)
        if not (z > 0).any():
            W[o] = 0.0
            B[o] = -1.0                    # colour never produced -> never fires
            continue
        res = _fit_separator(Xu, z)
        if res is None:
            return None
        W[o], B[o] = res
    return W.reshape(CHANNELS, CHANNELS, K, K), B


# --------------------------------------------------------------------------- #
# backend 2: Conv->Relu->Conv (pure half-space cover; non-separable rules)      #
# --------------------------------------------------------------------------- #
def _fit_pure(P, N, max_iter=_PURE_ITERS):
    """Margin perceptron over P(+1) vs N(-1), then lower the bias so every negative
    is <= 0 (pure).  Integer-valued w, b."""
    Pf = P.astype(np.float64)
    Nf = N.astype(np.float64)
    F = Pf.shape[1]
    w = np.zeros(F)
    bb = 0.0
    for _ in range(max_iter):
        vp = (Pf @ w + bb) < 1.0
        vn = (Nf @ w + bb) > -1.0 if len(Nf) else np.zeros(0, bool)
        if not vp.any() and not (len(Nf) and vn.any()):
            break
        if vp.any():
            w = w + Pf[vp].sum(0)
            bb += float(vp.sum())
        if len(Nf) and vn.any():
            w = w - Nf[vn].sum(0)
            bb -= float(vn.sum())
    if len(Nf):
        b = -float((Nf @ w).max())         # negatives <= 0, positives above covered
    else:
        b = 1.0 - float((Pf @ w).min())
    return w, b


def _exact_unit(p):
    """Half-space firing on exactly binary neighbourhood p and on no other."""
    w = np.where(p > 0, 1.0, -2.0)
    b = 1.0 - float(p.sum())
    return w, b


def _cover_color(P, N, cap):
    """Cover positives P with pure ReLU half-spaces (<= cap units), else None."""
    units = []
    rem = np.ones(len(P), bool)
    Pf = P.astype(np.float64)
    greedy = 0
    while rem.any():
        idx = np.where(rem)[0]
        if greedy < _GREEDY_ITERS and len(N):
            w, b = _fit_pure(Pf[idx], N)
            cov = (Pf[idx] @ w + b) > 0.5
            if cov.any():
                units.append((w, b))
                rem[idx[cov]] = False
                greedy += 1
                if len(units) > cap:
                    return None
                continue
        for i in idx:                      # fall back to exact-match units
            units.append(_exact_unit(Pf[i]))
            if len(units) > cap:
                return None
        rem[idx] = False
    return units


def _fit_relu(tins, tgts, K):
    """Fit a Conv->Relu->Conv local rule. Returns units [(w,b,colour)] or None."""
    d = _distinct(tins, tgts, K)
    if d is None:
        return None
    pats, labels = d
    cap = _RELU_CAP.get(K)
    if cap is None or pats.shape[0] > _RELU_MAX_UNIQ:
        return None
    units = []
    for c in range(CHANNELS):
        pos = labels == c
        if not pos.any():
            continue
        cu = _cover_color(pats[pos], pats[~pos], cap)
        if cu is None:
            return None
        for w, b in cu:
            units.append((w, b, c))
        if len(units) > cap:
            return None
    if not units or len(units) > cap:
        return None
    return units


# --------------------------------------------------------------------------- #
# unified local-rule backend: fit / predict (numpy) / build (onnx fragment)     #
# --------------------------------------------------------------------------- #
def _fit_local(tins, tgts, ks=(3, 5), allow_relu=True):
    """Cheapest working local-rule backend. Returns (kind, params, K) or None.
    Geometric callers pass ks=(3,) -- K=5 neighbourhood extraction over the whole
    dataset is costly and a genuine compact rule never needs the larger window."""
    for K in ks:
        res = _fit_conv(tins, tgts, K)
        if res is not None:
            return ("conv", res, K)
    if allow_relu:
        units = _fit_relu(tins, tgts, 3)   # relu only at K=3 (compact, fast)
        if units is not None:
            return ("relu", units, 3)
    return None


def _scores(onehot, W, B, K):
    X = _neighborhoods(onehot, K).astype(np.float64)
    S = X @ W.reshape(CHANNELS, -1).T + B
    return S.reshape(HEIGHT, WIDTH, CHANNELS).transpose(2, 0, 1)


def _relu_scores(onehot, units, K):
    X = _neighborhoods(onehot, K).astype(np.float64)
    out = np.zeros((HEIGHT * WIDTH, CHANNELS))
    for w, b, c in units:
        out[:, c] += np.maximum(X @ w + b, 0.0)
    return out.reshape(HEIGHT, WIDTH, CHANNELS).transpose(2, 0, 1)


def _local_pred(kind, params, K, onehot):
    """Bool (10,30,30): the local rule's >0-thresholded one-hot output."""
    if kind == "conv":
        W, B = params
        return _scores(onehot, W, B, K) > 0
    return _relu_scores(onehot, params, K) > 0


def _local_frag(g, src, kind, params, K):
    r = K // 2
    if kind == "conv":
        W, B = params
        w = g.init(DATA_TYPE, [CHANNELS, CHANNELS, K, K],
                   W.astype(np.float32).ravel().tolist())
        bt = g.init(DATA_TYPE, [CHANNELS], B.astype(np.float32).tolist())
        return g.node("Conv", [src, w, bt], kernel_shape=[K, K], pads=[r, r, r, r])
    units = params
    H = len(units)
    W1 = np.zeros((H, CHANNELS, K, K), np.float32)
    B1 = np.zeros(H, np.float32)
    W2 = np.zeros((CHANNELS, H, 1, 1), np.float32)
    for u, (w, b, c) in enumerate(units):
        W1[u] = np.asarray(w, np.float32).reshape(CHANNELS, K, K)
        B1[u] = float(b)
        W2[c, u, 0, 0] = 1.0
    w1 = g.init(DATA_TYPE, list(W1.shape), W1.ravel().tolist())
    b1 = g.init(DATA_TYPE, [H], B1.tolist())
    w2 = g.init(DATA_TYPE, list(W2.shape), W2.ravel().tolist())
    h = g.node("Conv", [src, w1, b1], kernel_shape=[K, K], pads=[r, r, r, r])
    hr = g.node("Relu", [h])
    return g.node("Conv", [hr, w2], kernel_shape=[1, 1], pads=[0, 0, 0, 0])


# --------------------------------------------------------------------------- #
# geometric fragments                                                          #
# --------------------------------------------------------------------------- #
def _transpose_frag(g, src):
    return g.node("Transpose", [src], perm=[0, 1, 3, 2])


def _upscale_frag(g, src, k):
    nh = math.ceil(HEIGHT / k)
    nw = math.ceil(WIDTH / k)
    s0 = g.init(INT64, [2], [0, 0])
    se = g.init(INT64, [2], [nh, nw])
    sa = g.init(INT64, [2], [2, 3])
    sm = g.node("Slice", [src, s0, se, sa])                 # small top-left feeder
    sc = g.init(DATA_TYPE, [4], [1.0, 1.0, float(k), float(k)])
    up = g.node("Resize", [sm, sc], mode="nearest")         # nh*k x nw*k
    if nh * k == HEIGHT and nw * k == WIDTH:
        return up
    c0 = g.init(INT64, [2], [0, 0])
    ce = g.init(INT64, [2], [HEIGHT, WIDTH])
    ca = g.init(INT64, [2], [2, 3])
    return g.node("Slice", [up, c0, ce, ca])


# --------------------------------------------------------------------------- #
# per-case fit + verify + build                                                #
# --------------------------------------------------------------------------- #
def _try_same(prs):
    """recolor o localrule / localrule o recolor / pure localrule (same shape).
    Single conv only (the fused linear rule); non-separable same-shape rules are
    pure local rules and belong to family_local2, not a composition."""
    tins = [_onehot(a) for a, _ in prs]
    res = _fit_local(tins, [b for _, b in prs], ks=(3, 5), allow_relu=False)
    if res is None:
        return None
    kind, params, K = res
    for a, b in prs:
        if not np.array_equal(_local_pred(kind, params, K, _onehot(a)),
                              _onehot(b) > 0):
            return None
    g = _G()
    _local_frag(g, "input", kind, params, K)
    return (f"localrule_k{K}", g.finish())


def _try_transpose(prs):
    """transpose o localrule: fit the local rule to the transposed target (b.T),
    then Transpose (perm [0,1,3,2]).  Output cell (i,j) reflects the neighbourhood
    around (j,i), which a position-local conv alone cannot reach."""
    tins = [_onehot(a) for a, _ in prs]
    tgts = [b.T for _, b in prs]
    res = _fit_local(tins, tgts, ks=(3,))
    if res is None:
        return None
    kind, params, K = res
    for (a, b), ti in zip(prs, tins):
        pred = _local_pred(kind, params, K, ti).transpose(0, 2, 1)
        if not np.array_equal(pred, _onehot(b) > 0):
            return None
    g = _G()
    c = _local_frag(g, "input", kind, params, K)
    _transpose_frag(g, c)
    return (f"transpose_localrule_{kind}_k{K}", g.finish())


def _try_upscale(prs, k):
    """upscale_k o localrule: fit the local rule to the k-subsampled target."""
    for a, b in prs:
        if b.shape != (a.shape[0] * k, a.shape[1] * k):
            return None
        rep = b[::k, ::k]
        if not np.array_equal(b, np.kron(rep, np.ones((k, k), dtype=b.dtype))):
            return None
    tins = [_onehot(a) for a, _ in prs]
    tgts = [b[::k, ::k] for _, b in prs]
    res = _fit_local(tins, tgts, ks=(3,))
    if res is None:
        return None
    kind, params, K = res
    nh = math.ceil(HEIGHT / k)
    nw = math.ceil(WIDTH / k)
    for (a, b), ti in zip(prs, tins):
        pred = _local_pred(kind, params, K, ti)
        small = pred[:, :nh, :nw]
        up = np.kron(small, np.ones((1, k, k), dtype=bool)).astype(bool)[:, :HEIGHT, :WIDTH]
        if not np.array_equal(up, _onehot(b) > 0):
            return None
    g = _G()
    c = _local_frag(g, "input", kind, params, K)
    _upscale_frag(g, c, k)
    return (f"upscale{k}_localrule_{kind}_k{K}", g.finish())


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for e in ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", []):
        a = np.array(e["input"], int)
        b = np.array(e["output"], int)
        if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
            continue
        if max(a.shape) > HEIGHT or max(b.shape) > HEIGHT:
            continue
        out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []

    same_shape = all(b.shape == a.shape for a, b in prs)
    transpose_shape = all(b.shape == (a.shape[1], a.shape[0]) for a, b in prs)
    up_ks = [k for k in range(2, 6)
             if all(b.shape == (a.shape[0] * k, a.shape[1] * k) for a, b in prs)]
    if not (same_shape or transpose_shape or up_ks):
        return []

    out = []
    try:
        # same-shape fused recolor o localrule / localrule o recolor (separable).
        # A same-shape solution is the cheapest possible, so when it exists we are
        # done -- no need to also probe the (more expensive) transpose form, which
        # for square grids would otherwise run a wasteful relu fallback.
        if same_shape and not all(np.array_equal(a, b) for a, b in prs):
            r = _try_same(prs)
            if r is not None:
                return [r]

        # transpose o localrule (Conv then Transpose).  Only this order: the
        # reverse (Transpose then Conv) fits a different function that merely
        # coincides on the shown pairs and would not survive the held-out grade.
        if transpose_shape:
            r = _try_transpose(prs)
            if r is not None:
                out.append(r)

        # upscale_k o localrule
        for k in up_ks:
            r = _try_upscale(prs, k)
            if r is not None:
                out.append(r)
                break
    except Exception:
        pass

    seen, uniq = set(), []
    for nm, m in out:
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append((nm, m))
    return uniq
