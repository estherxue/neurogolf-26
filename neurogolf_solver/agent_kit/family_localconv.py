"""Local-neighborhood family: express common ARC local rules (denoise, outline,
neighbor-count, per-pixel recolor) as a SINGLE small origin-anchored Conv, or a
tiny Conv->Relu->Conv pair.

Strategy
--------
A "local rule" maps each output cell's color to a fixed function of the input
neighborhood (a (2R+1)x(2R+1) window of the one-hot tensor, zero-padded exactly
like the grader's Conv).  For same-shape tasks we:

  1. Extract every cell's neighborhood (over train+test+arc-gen, exactly as the
     Conv sees it: zero padding = padding/out-of-grid cells) and its target color.
  2. Check the mapping is deterministic (a real local rule at this radius).
  3. Fit, per output color o, an integer linear separator w_o.x + b_o that is
     >0 iff the target color is o and <=0 otherwise (a batch perceptron with
     margin 1 -> exact integer weights -> robust under the grader's strict >0).
     If all 10 colors separate, emit a single Conv (weight [10,10,K,K] + bias).

  This single Conv covers denoise-KEEP channels, outline, neighbor-count
  thresholds, per-pixel recolor, etc. -- anything linearly separable in the
  neighborhood.  Cost = params only (no intermediate tensors) -> high points.

Denoise (remove isolated single-cell specks -> background) needs the background
channel to fire on removed specks of ANY color, which is a cross-color AND and is
NOT linearly separable; for that we additionally emit an analytic
Conv->Relu->Conv pair.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# one-hot helpers                                                             #
# --------------------------------------------------------------------------- #
def _onehot(grid):
    """grid (H,W) ints -> (CHANNELS,30,30) float one-hot, top-left anchored."""
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = grid.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (grid == c)
    return t


def _neighborhoods(t, K):
    """t (C,30,30) -> features (900, C*K*K) int8, matching Conv weight layout
    W[o,i,kr,kc] with offset (kr-K//2, kc-K//2)."""
    r = K // 2
    pad = np.zeros((CHANNELS, HEIGHT + 2 * r, WIDTH + 2 * r), np.float32)
    pad[:, r:r + HEIGHT, r:r + WIDTH] = t
    feats = np.empty((CHANNELS, K, K, HEIGHT, WIDTH), np.int8)
    for kr in range(K):
        for kc in range(K):
            feats[:, kr, kc] = pad[:, kr:kr + HEIGHT, kc:kc + WIDTH].astype(np.int8)
    # -> (H*W, C*K*K)
    return feats.reshape(CHANNELS * K * K, HEIGHT * WIDTH).T.copy()


def _target_colors(grid):
    """(900,) int8: color 0..9 for real cells, -1 for padding cells."""
    lab = np.full((HEIGHT, WIDTH), -1, np.int8)
    h, w = grid.shape
    lab[:h, :w] = grid
    return lab.reshape(-1)


# --------------------------------------------------------------------------- #
# perceptron separator                                                        #
# --------------------------------------------------------------------------- #
def _fit_separator(X, z, max_iter=400):
    """Batch perceptron with margin 1. X:(N,F) int, z:(N,) +/-1.
    Returns (w,b) integer arrays with z*(X.w+b) > 0 for all rows, else None."""
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
    # strict check: positives >0, negatives <=0
    if np.all((z > 0) == (s > 0)):
        return w, b
    if np.all(((z > 0) & (s > 0)) | ((z < 0) & (s <= 0))):
        return w, b
    return None


def _unique_rows(A):
    """Fast unique-rows for a contiguous int8 array via void view."""
    A = np.ascontiguousarray(A)
    v = A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
    _, idx = np.unique(v, return_index=True)
    return A[idx]


def _fit_single_conv(tins, out_grids, K):
    """Try to fit one Conv of kernel K over the local neighborhoods.
    tins: list of (C,30,30) one-hot inputs; out_grids: list of raw (H,W) outputs.
    Returns (W[10,10,K,K], B[10]) ints or None."""
    F = CHANNELS * K * K
    Xs, ys = [], []
    for ti, og in zip(tins, out_grids):
        Xs.append(_neighborhoods(ti, K))
        ys.append(_target_colors(og))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    # force biases negative: an explicit all-zero (padding) neighborhood -> none
    X = np.concatenate([X, np.zeros((1, F), np.int8)], 0)
    y = np.concatenate([y, np.array([-1], np.int8)], 0)

    # deduplicate (neighborhood,label); verify determinism.  Pack label as an
    # extra column so a void-view unique gives unique (neighborhood,label) pairs.
    xy = np.concatenate([X, (y + 1).astype(np.int8)[:, None]], 1)  # +1 -> [0..10]
    uxy = _unique_rows(xy)
    ux = _unique_rows(X)
    if uxy.shape[0] != ux.shape[0]:
        return None  # same neighborhood -> two colors: not local at radius R
    if uxy.shape[0] > 4000:
        return None  # too complex to be a compact rule

    Xu = uxy[:, :-1].astype(np.float64)
    cu = uxy[:, -1].astype(np.int64) - 1
    W = np.zeros((CHANNELS, F), np.float64)
    B = np.zeros(CHANNELS, np.float64)
    for o in range(CHANNELS):
        z = np.where(cu == o, 1.0, -1.0)
        if not (z > 0).any():
            # color never an output: drive bias very negative
            W[o] = 0.0
            B[o] = -1.0
            # verify: all rows give Xu.0 + (-1) <= 0  -> holds
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
# analytic denoise: Conv(3x3,grouped) -> Relu -> Conv(1x1)                     #
# --------------------------------------------------------------------------- #
def _detect_denoise(prs):
    """Returns background color B if every pair is exactly 'remove isolated
    (no same-color 4-neighbour) non-B cells -> B', else None."""
    # infer B from changed cells
    B = None
    for a, b in prs:
        if a.shape != b.shape:
            return None
        diff = a != b
        if diff.any():
            vals = np.unique(b[diff])
            if vals.size != 1:
                return None
            if B is None:
                B = int(vals[0])
            elif B != int(vals[0]):
                return None
    if B is None:
        return None  # nothing ever changes -> not a denoise demo
    for a, b in prs:
        exp = a.copy()
        h, w = a.shape
        for i in range(h):
            for j in range(w):
                k = a[i, j]
                if k == B:
                    continue
                same = False
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and a[ni, nj] == k:
                        same = True
                        break
                if not same:
                    exp[i, j] = B
        if not np.array_equal(exp, b):
            return None
    return B


def _build_denoise(B):
    """Layer1 grouped Conv (group=10, 10->20): per input channel c emit
        keep_c = Relu(5*center_c + neigh4_c - 5)   (out 2c)
        pres_c = center_c                          (out 2c+1)
    Layer2 1x1 Conv (20->10):
        out_c (c!=B) = keep_c
        out_B = pres_B + sum_{c!=B}(pres_c - keep_c)
    """
    K = 3
    # grouped conv weight shape [out=20, in/group=1, 3, 3]
    W1 = np.zeros((2 * CHANNELS, 1, K, K), np.float32)
    B1 = np.zeros(2 * CHANNELS, np.float32)
    for c in range(CHANNELS):
        # keep_c
        W1[2 * c, 0, 1, 1] = 5.0
        W1[2 * c, 0, 0, 1] = 1.0
        W1[2 * c, 0, 2, 1] = 1.0
        W1[2 * c, 0, 1, 0] = 1.0
        W1[2 * c, 0, 1, 2] = 1.0
        B1[2 * c] = -5.0
        # pres_c
        W1[2 * c + 1, 0, 1, 1] = 1.0
    w1 = oh.make_tensor("W1", DATA_TYPE, list(W1.shape), W1.ravel().tolist())
    b1 = oh.make_tensor("B1", DATA_TYPE, [2 * CHANNELS], B1.tolist())
    conv1 = oh.make_node("Conv", ["input", "W1", "B1"], ["h"],
                         kernel_shape=[K, K], pads=[1, 1, 1, 1], group=CHANNELS)
    relu = oh.make_node("Relu", ["h"], ["hr"])
    # layer2: 1x1 conv  out[o] = sum_j W2[o,j]*hr[j]
    W2 = np.zeros((CHANNELS, 2 * CHANNELS, 1, 1), np.float32)
    for c in range(CHANNELS):
        if c != B:
            W2[c, 2 * c, 0, 0] = 1.0          # out_c = keep_c
    # out_B
    W2[B, 2 * B + 1, 0, 0] = 1.0              # + pres_B
    for c in range(CHANNELS):
        if c != B:
            W2[B, 2 * c + 1, 0, 0] = 1.0      # + pres_c
            W2[B, 2 * c, 0, 0] = -1.0         # - keep_c
    w2 = oh.make_tensor("W2", DATA_TYPE, list(W2.shape), W2.ravel().tolist())
    conv2 = oh.make_node("Conv", ["hr", "W2"], ["output"],
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model([conv1, relu, conv2], [w1, b1, w2])


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
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
    # same-shape requirement
    if not all(a.shape == b.shape for a, b in prs):
        # denoise is same-shape too; nothing else here applies
        return []

    # Fit on ALL examples: the consistency check then rejects non-local tasks,
    # and any emitted separator is exact on the whole train+test+arc-gen set the
    # grader validates against.  Fast thanks to void-view dedup of neighborhoods.
    fit_prs = prs
    tins = [_onehot(a) for a, _ in fit_prs]
    out_grids = [b for _, b in fit_prs]

    out = []
    # 1) single conv, smallest kernel first (fewest params -> most points)
    for K in (3, 5):
        try:
            res = _fit_single_conv(tins, out_grids, K)
        except Exception:
            res = None
        if res is not None:
            W, B = res
            out.append((f"localconv{K}", _build_single_conv(W, B, K)))
            break  # smaller K already best

    # 2) analytic denoise (handles the speck->background channel that a single
    #    conv cannot express)
    try:
        bg = _detect_denoise(fit_prs)
    except Exception:
        bg = None
    if bg is not None:
        out.append((f"denoise_bg{bg}", _build_denoise(bg)))

    return out
