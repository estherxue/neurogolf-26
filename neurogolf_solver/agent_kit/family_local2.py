"""Two-layer local learner: Conv(KxK) -> Relu -> Conv(1x1).

Many ARC local rules are NOT linearly separable in the one-hot neighborhood
(XOR-like neighbor logic: "fill a cell iff exactly two diagonal neighbours of
colour X", "outline only convex corners", etc.).  A single conv (see
``family_localconv``) cannot express those.  This family fits a COMPACT
conv->relu->conv that is EXACT on the fit data.

Idea
----
A local rule maps each output cell's colour to a fixed function of its
(2R+1)x(2R+1) input neighbourhood (the one-hot tensor, zero padded exactly the
way the grader's Conv pads).  We:

  1. Extract every cell's neighbourhood + target colour over ALL available pairs
     (train+test+arc-gen; the integrator shows us only 70% of arc-gen).
  2. Deduplicate (neighbourhood -> colour) and verify it is a *function* at this
     radius (same neighbourhood never maps to two colours, padding included).
     If not deterministic at K=3 we retry K=5.
  3. For each output colour ``c`` we cover its positive neighbourhoods with as
     FEW ReLU half-space "hidden units" as possible, each kept PURE (fires on no
     negative neighbourhood) so the per-channel sum is exactly the indicator of
     colour ``c``:
        * a margin perceptron finds a half-space separating the remaining
          positives from all negatives; its bias is lowered until no negative
          fires (purity), covering every positive that still scores above all
          negatives -> one hidden unit covers a whole linearly-separable chunk;
        * any positives left over fall back to an exact-match unit
          (w = +1 on the pattern's ones, -2 elsewhere, bias = 1 - |ones|), which
          provably outputs 1 on exactly that neighbourhood and 0 on every other
          binary input -> guarantees termination and exactness.

  Linearly separable colours need 1 unit; genuinely non-separable (XOR-like)
  colours expand to a handful.  Hidden width H is therefore minimised
  automatically -> low cost (cost = params + intermediate memory).

The first conv (KxK, ``pads=[r,r,r,r]``) is origin anchored exactly like the
grader's own conv, so zero padding == out-of-grid cells; the construction makes
every padding/out-of-grid neighbourhood map to all-zero output.  The whole net
is grid-size independent and re-validated for EXACT equality by the harness.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64

# width caps (respect the 1.44MB file limit + keep cost sane).
#   per unit params ~= (C*K*K + 1 + C) floats * 4 bytes
_CAP = {3: 2500, 5: 1000}
_GREEDY_ITERS = 24          # greedy half-space steps per colour before per-pattern


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# one-hot / neighbourhood helpers (same conventions as family_localconv)       #
# --------------------------------------------------------------------------- #
def _onehot(grid):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = grid.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (grid == c)
    return t


def _neighborhoods(t, K):
    """t (C,30,30) -> (900, C*K*K) int8, feature index = c*K*K + kr*K + kc,
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
    lab = np.full((HEIGHT, WIDTH), -1, np.int8)
    h, w = grid.shape
    lab[:h, :w] = grid
    return lab.reshape(-1)


def _unique_rows(A):
    A = np.ascontiguousarray(A)
    v = A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
    _, idx = np.unique(v, return_index=True)
    return A[idx]


def _extract(prs, K):
    """Return (patterns[D,F] int8, labels[D] int) of distinct neighbourhoods, or
    None if the rule is not a function at this radius / too large."""
    F = CHANNELS * K * K
    Xs, ys = [], []
    for a, b in prs:
        Xs.append(_neighborhoods(_onehot(a), K))
        ys.append(_target_colors(b))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    # an explicit all-zero (deep padding) neighbourhood maps to "no colour"
    X = np.concatenate([X, np.zeros((1, F), np.int8)], 0)
    y = np.concatenate([y, np.array([-1], np.int8)], 0)

    xy = np.concatenate([X, (y + 1).astype(np.int8)[:, None]], 1)  # label -> [0..10]
    uxy = _unique_rows(xy)
    ux = _unique_rows(X)
    if uxy.shape[0] != ux.shape[0]:
        return None                       # same neighbourhood -> two colours
    if uxy.shape[0] > _CAP[K]:
        return None
    return uxy[:, :-1].astype(np.int8), (uxy[:, -1].astype(np.int64) - 1)


# --------------------------------------------------------------------------- #
# pure half-space fitting                                                       #
# --------------------------------------------------------------------------- #
def _fit_pure(P, N, max_iter=200):
    """Margin perceptron over P(+1) vs N(-1) for a direction, then lower the bias
    until every negative is <= 0 (purity).  Integer-valued w, b."""
    Pf = P.astype(np.float64)
    Nf = N.astype(np.float64)
    F = Pf.shape[1]
    w = np.zeros(F)
    bb = 0.0
    for _ in range(max_iter):
        sp = Pf @ w + bb
        vp = sp < 1.0
        vn = (Nf @ w + bb) > -1.0 if len(Nf) else np.zeros(0, bool)
        if not vp.any() and not vn.any():
            break
        if vp.any():
            w = w + Pf[vp].sum(0)
            bb += float(vp.sum())
        if len(Nf) and vn.any():
            w = w - Nf[vn].sum(0)
            bb -= float(vn.sum())
    if len(Nf):
        # boundary right at the top negative: negatives map to <= 0 (off under
        # the grader's >0 threshold), positives strictly above are covered.
        b = -float((Nf @ w).max())
    else:
        b = 1.0 - float((Pf @ w).min())     # no negatives: cover all positives
    return w, b


def _exact_unit(p):
    """Half-space that outputs 1 on exactly neighbourhood p and 0 on every other
    binary input (proof in module docstring)."""
    w = np.where(p > 0, 1.0, -2.0)
    b = 1.0 - float(p.sum())
    return w, b


def _cover_color(P, N, cap):
    """Cover positives P with pure ReLU half-spaces; <= cap units, else None."""
    units = []
    rem = np.ones(len(P), bool)
    Pf = P.astype(np.float64)
    greedy = 0
    while rem.any():
        idx = np.where(rem)[0]
        if greedy < _GREEDY_ITERS and len(N):
            w, b = _fit_pure(Pf[idx], N)
            s = Pf[idx] @ w + b
            cov = s > 0.5
            if cov.any():
                units.append((w, b))
                rem[idx[cov]] = False
                greedy += 1
                if len(units) > cap:
                    return None
                continue
        # fall back to one exact-match unit per remaining pattern
        for i in idx:
            units.append(_exact_unit(Pf[i]))
            if len(units) > cap:
                return None
        rem[idx] = False
    return units


# --------------------------------------------------------------------------- #
# ONNX build                                                                    #
# --------------------------------------------------------------------------- #
def _build(units, K):
    H = len(units)
    r = K // 2
    W1 = np.zeros((H, CHANNELS, K, K), np.float32)
    B1 = np.zeros(H, np.float32)
    W2 = np.zeros((CHANNELS, H, 1, 1), np.float32)
    for u, (w, b, color) in enumerate(units):
        W1[u] = np.asarray(w, np.float32).reshape(CHANNELS, K, K)
        B1[u] = float(b)
        W2[color, u, 0, 0] = 1.0
    w1 = oh.make_tensor("W1", DATA_TYPE, list(W1.shape), W1.ravel().tolist())
    b1 = oh.make_tensor("B1", DATA_TYPE, [H], B1.tolist())
    w2 = oh.make_tensor("W2", DATA_TYPE, list(W2.shape), W2.ravel().tolist())
    conv1 = oh.make_node("Conv", ["input", "W1", "B1"], ["h"],
                         kernel_shape=[K, K], pads=[r, r, r, r])
    relu = oh.make_node("Relu", ["h"], ["hr"])
    conv2 = oh.make_node("Conv", ["hr", "W2"], ["output"],
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model([conv1, relu, conv2], [w1, b1, w2])


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


def _fit_K(prs, K):
    res = _extract(prs, K)
    if res is None:
        return None
    pats, labels = res
    cap = _CAP[K]
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


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    # local rule -> same shape; pure identity belongs to a cheaper family
    if not all(a.shape == b.shape for a, b in prs):
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []

    out = []
    for K in (3, 5):
        try:
            units = _fit_K(prs, K)
        except Exception:
            units = None
        if units is not None:
            out.append((f"local2_k{K}_h{len(units)}", _build(units, K)))
            # K=3 already exact & cheaper; no need for K=5
            break
    return out
