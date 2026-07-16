"""family_sgolf3_7 -- cheap EXACT solvers for deterministic, origin-anchored LOCAL
rules, expressed as a SINGLE KxK Conv (no intermediates -> maximal points) whenever
the rule is linearly separable, else a compact identity-default neighbourhood LUT.

Rationale
---------
Several same-shape ARC tasks recolour each cell as a fixed function of its
(2R+1)x(2R+1) one-hot neighbourhood, padded exactly as the grader's Conv sees it
(out-of-grid == 0).  When that function is linearly separable it is a single Conv
`output = Conv(input, W, B)` with ZERO intermediate tensors -- the cheapest possible
non-trivial model (cost = params only), scoring far above the multi-Conv CA / learned
16-filter CNN baselines these tasks currently carry (e.g. t352, t122).  When it is
deterministic-local but not separable we fall back to a two-Conv identity-default LUT
(one hidden exact-match layer + a 1x1 delta), still cheap.

Exactness / anti-overfit
------------------------
We only fire when the mapping is a genuine deterministic local rule over EVERY
train+test+arc-gen pair (same neighbourhood -> same colour) at the smallest radius
that works, and the default is identity (copy input) so unseen neighbourhoods
generalise to "keep".  The shared harness then re-validates EXACTness on all splits
(and the grader on a private set) before the model is kept; wrong guesses are dropped.

Grader-shape notes
------------------
Output must be one-hot with exactly one channel > 0 on every real cell and <= 0 on
padding.  Padding neighbourhoods are all-zero, so the separator's `w.x + b` there is
`b`; we force every class score at the all-zero input to be <= 0 (via the fitted bias)
so padding stays blank.  For the LUT, an all-zero neighbourhood matches no exact-match
unit, so the identity default keeps padding at 0.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS,
                           CHANNELS, HEIGHT, WIDTH)

_MAX_U = 400            # cap LUT exceptions (file-size + cost sanity)
_MAX_FILE_BYTES = 1_300_000
_MAX_PATTERNS = 12000   # too many unique neighbourhoods -> give up on separator


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# neighbourhood extraction (matches Conv weight layout W[o, i, kr, kc])        #
# --------------------------------------------------------------------------- #
def _onehot(grid):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.int8)
    h, w = grid.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (grid == c)
    return t


def _neighborhoods(t, K):
    r = K // 2
    pad = np.zeros((CHANNELS, HEIGHT + 2 * r, WIDTH + 2 * r), np.int8)
    pad[:, r:r + HEIGHT, r:r + WIDTH] = t
    feats = np.empty((CHANNELS, K, K, HEIGHT, WIDTH), np.int8)
    for kr in range(K):
        for kc in range(K):
            feats[:, kr, kc] = pad[:, kr:kr + HEIGHT, kc:kc + WIDTH]
    return feats.reshape(CHANNELS * K * K, HEIGHT * WIDTH).T


def _labels(grid):
    lab = np.full((HEIGHT, WIDTH), -1, np.int8)
    h, w = grid.shape
    lab[:h, :w] = grid
    return lab.reshape(-1)


def _collect(prs, K, real_only=True):
    """Neighbourhood features X, target colours y, centre colours cen over the cells
    of every pair (radius K).  With real_only, keep only in-grid cells; otherwise keep
    all 900 cells so padding-region neighbourhoods (label -1) act as hard negatives."""
    Xs, ys, cs = [], [], []
    for a, b in prs:
        Xs.append(_neighborhoods(_onehot(a), K))
        ys.append(_labels(b))
        cs.append(_labels(a))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    c = np.concatenate(cs, 0)
    if real_only:
        m = y >= 0
        return X[m], y[m], c[m]
    return X, y, c


# --------------------------------------------------------------------------- #
# 1) single separable Conv  (no intermediates -> cheapest)                     #
# --------------------------------------------------------------------------- #
def _fit_separator(X, z, max_iter=4000):
    """Perceptron with margin 1; returns (w, b) with sign(w.x+b) == sign(z), or None."""
    Xf = X.astype(np.float64)
    w = np.zeros(Xf.shape[1], np.float64)
    b = 0.0
    for _ in range(max_iter):
        s = Xf @ w + b
        viol = z * s < 1.0
        if not viol.any():
            break
        zv = z[viol]
        w += zv @ Xf[viol]
        b += zv.sum()
    s = Xf @ w + b
    if np.all((z > 0) == (s > 0)):
        return w, b
    return None


def _fit_single_conv(X, y, K):
    # X, y span ALL 900 cells per grid (real + padding).  Padding cells carry label -1 and
    # act as hard negatives for every class, so the fitted Conv stays <= 0 on every
    # padding-region neighbourhood (incl. border windows overlapping the grid) -> the
    # grader "padding blank" requirement holds by construction, not by float luck.
    F = CHANNELS * K * K
    Xc = np.ascontiguousarray(X)
    v = Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()
    _, idx = np.unique(v, return_index=True)
    # determinism: unique neighbourhood -> unique label
    xy = np.concatenate([Xc, (y + 1).astype(np.int8)[:, None]], 1)
    vv = np.ascontiguousarray(xy).view(np.dtype((np.void, xy.itemsize * xy.shape[1])))
    if np.unique(vv).size != idx.size:
        return None
    if idx.size > _MAX_PATTERNS:
        return None
    Xu = X[idx].astype(np.float64)
    yu = y[idx].astype(np.int64)
    W = np.zeros((CHANNELS, F), np.float64)
    B = np.zeros(CHANNELS, np.float64)
    for o in range(CHANNELS):
        z = np.where(yu == o, 1.0, -1.0)
        if not (z > 0).any():
            B[o] = -1.0          # class never occurs -> always <= 0
            continue
        res = _fit_separator(Xu, z)
        if res is None:
            return None
        W[o], B[o] = res
    return W.reshape(CHANNELS, CHANNELS, K, K), B


def _build_single_conv(W, B, K):
    r = K // 2
    w = oh.make_tensor("W", DATA_TYPE, [CHANNELS, CHANNELS, K, K],
                       W.astype(np.float32).ravel().tolist())
    bt = oh.make_tensor("B", DATA_TYPE, [CHANNELS], B.astype(np.float32).tolist())
    node = oh.make_node("Conv", ["input", "W", "B"], ["output"],
                        kernel_shape=[K, K], pads=[r, r, r, r])
    return _model([node], [w, bt])


# --------------------------------------------------------------------------- #
# 2) identity-default LUT  (exact-match hidden layer + 1x1 delta)              #
# --------------------------------------------------------------------------- #
def _fit_lut(X, y, cen, K):
    Xc = np.ascontiguousarray(X)
    v = Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()
    order = np.argsort(v, kind="stable")
    vs = v[order]; Xo = Xc[order]; yo = y[order]; co = cen[order]
    exceptions = []
    i, n = 0, len(vs)
    while i < n:
        j = i
        while j < n and vs[j] == vs[i]:
            j += 1
        if np.unique(yo[i:j]).size > 1:
            return None                       # not deterministic at this radius
        o = int(yo[i]); c0 = int(co[i])
        if o != c0:
            exceptions.append((Xo[i].copy(), o, c0))
            if len(exceptions) > _MAX_U:
                return None
        i = j
    return exceptions


def _build_lut(exceptions, K):
    U = len(exceptions)
    r = K // 2
    W1 = np.zeros((U, CHANNELS, K, K), np.float32)
    B1 = np.zeros(U, np.float32)
    W2 = np.zeros((CHANNELS, U, 1, 1), np.float32)
    for u, (patt, o, c0) in enumerate(exceptions):
        p = patt.astype(np.float32)
        S = float(p.sum())
        W1[u] = (2.0 * p - 1.0).reshape(CHANNELS, K, K)
        B1[u] = -(S - 1.0)
        W2[o, u, 0, 0] = 1.0
        W2[c0, u, 0, 0] = -1.0
    w1 = oh.make_tensor("W1", DATA_TYPE, [U, CHANNELS, K, K], W1.ravel().tolist())
    b1 = oh.make_tensor("B1", DATA_TYPE, [U], B1.tolist())
    w2 = oh.make_tensor("W2", DATA_TYPE, [CHANNELS, U, 1, 1], W2.ravel().tolist())
    conv1 = oh.make_node("Conv", ["input", "W1", "B1"], ["h"],
                         kernel_shape=[K, K], pads=[r, r, r, r])
    relu = oh.make_node("Relu", ["h"], ["hr"])
    conv2 = oh.make_node("Conv", ["hr", "W2"], ["delta"],
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    add = oh.make_node("Add", ["input", "delta"], ["output"])
    return _model([conv1, relu, conv2, add], [w1, b1, w2])


# --------------------------------------------------------------------------- #
# entry                                                                        #
# --------------------------------------------------------------------------- #
_KS = (3, 5, 7)          # radii to try (r = 1,2,3); larger rarely helps and costs a lot


def _pairs(ex):
    out = []
    for sec in ("train", "test", "arc-gen"):
        for e in ex.get(sec, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return None
            if max(a.shape + b.shape) > 30:
                return None
            out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if not prs or not all(a.shape == b.shape for a, b in prs):
        return []
    if not any((a != b).any() for a, b in prs):
        return []                              # identity is not our job
    out = []
    for K in _KS:
        try:
            X, y, cen = _collect(prs, K, real_only=False)
        except Exception:
            continue
        m = y >= 0
        # deterministic local rule at this radius?  (same neighbourhood -> same colour)
        Xc = np.ascontiguousarray(X[m])
        v = Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()
        _, idx = np.unique(v, return_index=True)
        xy = np.concatenate([Xc, (y[m] + 1).astype(np.int8)[:, None]], 1)
        vv = np.ascontiguousarray(xy).view(np.dtype((np.void, xy.itemsize * xy.shape[1])))
        if np.unique(vv).size != idx.size:
            continue                           # not deterministic here -> try larger radius
        # (1) single separable Conv (no intermediates) -- cheapest when it exists.
        try:
            res = _fit_single_conv(X, y, K)
        except Exception:
            res = None
        if res is not None:
            out.append((f"sconv{K}", _build_single_conv(res[0], res[1], K)))
        # (2) identity-default LUT fallback (padding-safe by construction).
        try:
            exc = _fit_lut(X[m], y[m], cen[m], K)
        except Exception:
            exc = None
        if exc:
            U = len(exc)
            if U * CHANNELS * K * K * 4 <= _MAX_FILE_BYTES:
                out.append((f"lut{K}_{U}", _build_lut(exc, K)))
        break                                  # smallest deterministic radius wins
    return out
