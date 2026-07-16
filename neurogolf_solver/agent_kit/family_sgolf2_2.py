"""family_sgolf2_2 -- cheaper EXACT solvers for LOCAL, origin-anchored rules via a
single small Conv LUT (params-only cost, ZERO intermediate tensors -> high points).

A "local rule" maps every output cell's colour to a fixed function of the input
(2R+1)x(2R+1) one-hot neighbourhood, zero-padded exactly as the grader's Conv sees
it (padding / out-of-grid == 0). For same-shape tasks whose rule is deterministic
and linearly separable at some small radius we fit, per output colour o, an integer
margin-1 perceptron w_o.x + b_o that is >0 iff the target colour is o (<=0 else).
If all ten colours separate we emit ONE Conv (weight [10,10,K,K] + bias): the single
Conv output IS the graph output, so there are NO intermediate tensors -> cost =
params only. That is far cheaper than the multi-Conv baselines these tasks carry.

The PADDING GOTCHA is handled for free: padding cells are all-zero neighbourhoods,
and we pin an explicit all-zero neighbourhood -> "no colour" so every bias stays
negative (padding output <= 0 on all channels). The rule is origin-anchored because
the Conv is translation-equivariant with SAME padding = 0.

ANTI-OVERFIT: we FIT over train+test+arc-gen and re-check determinism + exact
separation on all of them; a task that is not a genuine local rule at radius R
(same neighbourhood -> two colours, or not linearly separable) is rejected and no
model is emitted. Nothing depends on locally-seen grid sizes.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _onehot(grid):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = grid.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (grid == c)
    return t


def _neighborhoods(t, K):
    """t (C,30,30) -> (H*W, C*K*K) int8 in Conv weight layout W[o,i,kr,kc]."""
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


def _fit_separator(X, z, max_iter=600):
    """Batch perceptron, margin 1. Returns integer (w,b) with z*(X.w+b)>0 all rows."""
    N, F = X.shape
    w = np.zeros(F, np.float64)
    b = 0.0
    Xf = X.astype(np.float64)
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


def _unique_rows(A):
    A = np.ascontiguousarray(A)
    v = A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
    _, idx = np.unique(v, return_index=True)
    return A[idx]


def _fit_single_conv(tins, out_grids, K):
    F = CHANNELS * K * K
    Xs, ys = [], []
    for ti, og in zip(tins, out_grids):
        Xs.append(_neighborhoods(ti, K))
        ys.append(_target_colors(og))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    # pin the all-zero (padding) neighbourhood -> "no colour"
    X = np.concatenate([X, np.zeros((1, F), np.int8)], 0)
    y = np.concatenate([y, np.array([-1], np.int8)], 0)

    xy = np.concatenate([X, (y + 1).astype(np.int8)[:, None]], 1)
    uxy = _unique_rows(xy)
    ux = _unique_rows(X)
    if uxy.shape[0] != ux.shape[0]:
        return None  # not deterministic at this radius
    if uxy.shape[0] > 4000:
        return None  # too complex to be a compact rule

    Xu = uxy[:, :-1].astype(np.float64)
    cu = uxy[:, -1].astype(np.int64) - 1
    W = np.zeros((CHANNELS, F), np.float64)
    B = np.zeros(CHANNELS, np.float64)
    for o in range(CHANNELS):
        z = np.where(cu == o, 1.0, -1.0)
        if not (z > 0).any():
            W[o] = 0.0
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


def _pairs(ex):
    out = []
    for e in ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", []):
        a = np.array(e["input"], int)
        b = np.array(e["output"], int)
        if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
            return None
        if max(a.shape) > 30 or max(b.shape) > 30:
            continue
        out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if not all(a.shape == b.shape for a, b in prs):
        return []
    if not any((a != b).any() for a, b in prs):
        return []  # identity -> not our family

    tins = [_onehot(a) for a, _ in prs]
    out_grids = [b for _, b in prs]

    out = []
    for K in (3, 5, 7):
        try:
            res = _fit_single_conv(tins, out_grids, K)
        except Exception:
            res = None
        if res is not None:
            W, B = res
            out.append((f"lconv{K}", _build_single_conv(W, B, K)))
            break  # smallest kernel = fewest params = most points
    return out
