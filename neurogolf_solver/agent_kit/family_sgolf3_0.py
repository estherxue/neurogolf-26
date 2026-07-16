"""family_sgolf3_0 -- cheap EXACT solvers for deterministic, origin-anchored LOCAL
rules via an IDENTITY-DEFAULT neighbourhood LUT.

Many ARC same-shape tasks are "keep the input, but overwrite a sparse set of cells"
where the new colour of a changed cell is a fixed function of its (2R+1)x(2R+1)
one-hot neighbourhood (padded exactly as the grader's Conv sees it: out-of-grid == 0).
Such a rule is deterministic + local at some small radius R, but NOT linearly
separable, so a single Conv (family_sgolf2_2 / family_localconv) cannot express it.

Construction (exact, origin-anchored, no data-size dependence):
  base   = input                                   (identity default: keep colour)
  For each UNIQUE neighbourhood pattern p whose target colour o(p) != centre(p)
  (an "exception"), add a hidden unit that fires iff the neighbourhood equals p,
  then in a 1x1 Conv turn that indicator into a delta (+1 on channel o, -1 on
  channel centre).  output = input + delta.

  Exact-match detector for a binary pattern p with S ones:
      w = 2p-1,  bias = -(S-1)     ->  ReLU(w.x + bias) == 1  iff x == p, else 0
  (w.x is integer; for x==p it equals S, else <= S-1, so the ReLU is 0/1.)

  Padding cells (all-zero neighbourhood, S_x=0) match no exception (every S>=1),
  so they stay <=0 on all channels.  Exactly one exception can match a given
  neighbourhood, so the delta touches only (o, centre); every real cell ends with
  exactly one channel >0.

Cost = params + intermediate memory of two small [1,U,30,30] tensors + one
[1,10,30,30] delta.  Cheaper than the multi-Conv CA baselines these tasks carry.

ANTI-OVERFIT: we require the rule to be a genuine deterministic local rule at
radius R over train+test+arc-gen (same neighbourhood -> same colour), the default
is the ubiquitous identity (copy input) so unseen neighbourhoods generalise to
"keep", and the harness re-validates EXACTness on every split before emitting.
We also try the linearly-separable single Conv (no intermediates) first; the
harness keeps whichever is cheapest.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS,
                           CHANNELS, HEIGHT, WIDTH)

_MAX_U = 400          # cap exceptions -> keep file < 1.44MB and cost sane
_MAX_FILE_BYTES = 1_350_000


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# neighbourhood extraction (matches Conv weight layout W[o,i,kr,kc])           #
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


# --------------------------------------------------------------------------- #
# identity-default LUT                                                         #
# --------------------------------------------------------------------------- #
def _collect(prs, K, real_only=True):
    """Return (X, y, cen) over cells of every pair, radius K.  With real_only,
    keep only in-grid cells; otherwise keep ALL 900 cells (padding label == -1,
    needed as negatives so the single Conv never fires on padding neighbourhoods)."""
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


def _fit_lut(prs, K):
    """Deterministic local rule at radius K?  If so return the exception table
    [(pattern int8[F], out_color, centre_color), ...] with identity default."""
    X, y, cen = _collect(prs, K)
    Xc = np.ascontiguousarray(X)
    v = Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()
    order = np.argsort(v, kind="stable")
    vs = v[order]
    Xo = Xc[order]
    yo = y[order]
    co = cen[order]
    exceptions = []
    i, n = 0, len(vs)
    while i < n:
        j = i
        while j < n and vs[j] == vs[i]:
            j += 1
        if np.unique(yo[i:j]).size > 1:
            return None                      # not deterministic at this radius
        o = int(yo[i]); c0 = int(co[i])
        if o != c0:
            exceptions.append((Xo[i].copy(), o, c0))
            if len(exceptions) > _MAX_U:
                return None
        i = j
    return exceptions


def _build_lut(exceptions, K):
    U = len(exceptions)
    F = CHANNELS * K * K
    r = K // 2
    # layer 1: exact-match detectors  h_u = ReLU(w_u.x + b_u)
    W1 = np.zeros((U, CHANNELS, K, K), np.float32)
    B1 = np.zeros(U, np.float32)
    # layer 2: 1x1 conv  delta[c] = sum_u W2[c,u] h_u
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
# linearly-separable single Conv (no intermediates) -- cheapest when it works  #
# --------------------------------------------------------------------------- #
def _fit_separator(X, z, max_iter=300):
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


def _fit_single_conv(prs, K):
    X, y, _ = _collect(prs, K, real_only=False)
    F = CHANNELS * K * K
    X = np.concatenate([X, np.zeros((1, F), np.int8)], 0)
    y = np.concatenate([y, np.array([-1], np.int8)], 0)
    Xc = np.ascontiguousarray(X)
    v = Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()
    _, idx = np.unique(v, return_index=True)
    Xu = X[idx].astype(np.float64)
    yu = y[idx].astype(np.int64)
    # determinism already guaranteed if unique rows carry unique labels
    xy = np.concatenate([Xc, (y + 1).astype(np.int8)[:, None]], 1)
    vv = np.ascontiguousarray(xy).view(np.dtype((np.void, xy.itemsize * xy.shape[1])))
    if np.unique(vv).size != idx.size:
        return None                        # not deterministic at this radius
    if idx.size > 6000:
        return None                        # too many patterns to separate cheaply
    W = np.zeros((CHANNELS, F), np.float64)
    B = np.zeros(CHANNELS, np.float64)
    for o in range(CHANNELS):
        z = np.where(yu == o, 1.0, -1.0)
        if not (z > 0).any():
            B[o] = -1.0
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
# entry                                                                       #
# --------------------------------------------------------------------------- #
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
    if not prs:
        return []
    same_shape = all(a.shape == b.shape for a, b in prs)
    out = []
    # 1) linearly-separable single Conv (no intermediates) -- cheapest.
    #    Small kernels first (fewer params -> more points).  Works for any task
    #    whose output cell is a deterministic local function of the input
    #    neighbourhood at the SAME position (incl. fixed shrink-to-corner like
    #    3x3 -> 1x1: every non-output cell is a padding negative).
    for K in (3, 5):
        try:
            res = _fit_single_conv(prs, K)
        except Exception:
            res = None
        if res is not None:
            W, B = res
            out.append((f"sconv{K}", _build_single_conv(W, B, K)))
            break
    if not same_shape:
        return out
    # 2) identity-default local LUT at the smallest deterministic radius
    for K in (3, 5):
        try:
            exc = _fit_lut(prs, K)
        except Exception:
            exc = None
        if exc is None:
            continue
        if not exc:
            break                      # deterministic but nothing changes: skip
        U = len(exc)
        if U * CHANNELS * K * K * 4 > _MAX_FILE_BYTES:
            break
        out.append((f"lut{K}_{U}", _build_lut(exc, K)))
        break
    return out
