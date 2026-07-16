"""family_sgolf5_7 -- CHEAP, GRID-AGNOSTIC, NO-CROP exact solvers for two shapes of
same-size rules found in this slice.

Both emitters keep the canvas 30x30 end-to-end (input [1,10,30,30] -> output
[1,10,30,30]); nothing is sliced smaller and padded back, no work-area shrink.

(A) POSITION-PARITY RECOLOUR (task 252 "colorbypos_per1x2").
    Every foreground cell whose row (or column) index has a fixed parity is recoloured
    to a single colour T; all other cells are copied verbatim. The parity class is a
    static function of the absolute (origin-anchored) coordinate, so it generalises to
    any grid size. We build it with NO full 10-channel intermediate: a 1x1 Conv reduces
    the one-hot to a single-channel foreground mask [1,1,30,30], we AND it with a tiny
    broadcast parity mask, and a single `Where` writes the answer straight into the FREE
    `output` tensor (its bytes aren't charged). Only single-channel intermediates ->
    far cheaper than a multi-channel recolour pipeline.

(B) SINGLE-CONV LOCAL LUT (task 122). For same-shape tasks whose output colour is a
    deterministic, linearly-separable function of a small input neighbourhood we fit one
    Conv [10,10,K,K]+bias whose output IS the graph output -> ZERO intermediate memory,
    cost = params only. (Same idea as family_sgolf2_2, kept here so this module is
    self-contained.)

ANTI-OVERFIT: every rule is re-verified EXACTLY (numpy mirror == prediction) on all
train+test+arc-gen before a model is emitted; a task that doesn't match is skipped.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS,
                           CHANNELS, HEIGHT, WIDTH)

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


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


# --------------------------------------------------------------------------- #
# (A) position-parity recolour
# --------------------------------------------------------------------------- #
def _detect_parity(prs):
    """Return (axis, period, p, T) if foreground cells with coord%period==p all map to
    colour T and everything else is copied, uniquely, else None."""
    hits = []
    for axis in (0, 1):
        for period in (2, 3):
            for p in range(period):
                for T in range(1, CHANNELS):
                    ok = True
                    for a, b in prs:
                        if a.shape != b.shape:
                            ok = False
                            break
                        pred = a.copy()
                        coord = (np.arange(a.shape[0])[:, None] if axis == 0
                                 else np.arange(a.shape[1])[None, :])
                        sel = (a != 0) & ((coord % period) == p)
                        pred[sel] = T
                        if not np.array_equal(pred, b):
                            ok = False
                            break
                    if ok:
                        hits.append((axis, period, p, T))
    if len(hits) == 1:
        return hits[0]
    return None


def _build_parity(axis, period, p, T):
    # foreground reducer: 1x1 Conv, weight 0 for colour-0 channel, 1 for colours 1..9.
    wfg = np.zeros((1, CHANNELS, 1, 1), np.float32)
    wfg[0, 1:, 0, 0] = 1.0
    Wfg = oh.make_tensor("Wfg", DATA_TYPE, [1, CHANNELS, 1, 1], wfg.ravel().tolist())

    # tiny broadcast parity mask (row -> [1,1,30,30] via [1,1,H,1]; col via [1,1,1,W]).
    if axis == 0:
        m = np.zeros((1, 1, HEIGHT, 1), np.float32)
        m[0, 0, (np.arange(HEIGHT) % period) == p, 0] = 1.0
        mask = oh.make_tensor("MASK", DATA_TYPE, [1, 1, HEIGHT, 1], m.ravel().tolist())
    else:
        m = np.zeros((1, 1, 1, WIDTH), np.float32)
        m[0, 0, 0, (np.arange(WIDTH) % period) == p] = 1.0
        mask = oh.make_tensor("MASK", DATA_TYPE, [1, 1, 1, WIDTH], m.ravel().tolist())

    half = oh.make_tensor("HALF", DATA_TYPE, [], [0.5])
    eT = np.zeros((1, CHANNELS, 1, 1), np.float32)
    eT[0, T, 0, 0] = 1.0
    ET = oh.make_tensor("ET", DATA_TYPE, [1, CHANNELS, 1, 1], eT.ravel().tolist())

    nodes = [
        oh.make_node("Conv", ["input", "Wfg"], ["fg"], kernel_shape=[1, 1]),
        oh.make_node("Mul", ["fg", "MASK"], ["chg"]),
        oh.make_node("Greater", ["chg", "HALF"], ["cond"]),
        oh.make_node("Where", ["cond", "ET", "input"], ["output"]),
    ]
    return _model(nodes, [Wfg, mask, half, ET])


# --------------------------------------------------------------------------- #
# (B) single-Conv local LUT
# --------------------------------------------------------------------------- #
def _onehot(grid):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = grid.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (grid == c)
    return t


def _neighborhoods(t, K):
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
    X = np.concatenate([X, np.zeros((1, F), np.int8)], 0)
    y = np.concatenate([y, np.array([-1], np.int8)], 0)

    xy = np.concatenate([X, (y + 1).astype(np.int8)[:, None]], 1)
    uxy = _unique_rows(xy)
    ux = _unique_rows(X)
    if uxy.shape[0] != ux.shape[0]:
        return None
    if uxy.shape[0] > 4000:
        return None

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


# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if not all(a.shape == b.shape for a, b in prs):
        return []
    if not any((a != b).any() for a, b in prs):
        return []

    out = []

    # (A) position-parity recolour
    pr = _detect_parity(prs)
    if pr is not None:
        out.append(("parity_%d%d%d_c%d" % pr, _build_parity(*pr)))

    # (B) single-Conv local LUT
    tins = [_onehot(a) for a, _ in prs]
    out_grids = [b for _, b in prs]
    for K in (3, 5, 7):
        try:
            res = _fit_single_conv(tins, out_grids, K)
        except Exception:
            res = None
        if res is not None:
            W, B = res
            out.append((f"lconv{K}", _build_single_conv(W, B, K)))
            break
    return out
