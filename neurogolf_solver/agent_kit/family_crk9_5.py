"""family_crk9_5 — diagonal block-domino "8" extension (task 042) + helpers.

Task 042 rule (verified EXACT on train+test+arc-gen, 266/266):
A grid contains color-3 squares arranged as DIAGONAL DOMINOES: two equal bs x bs
solid blocks touching at a single corner (either main-diagonal: TL+BR filled, or
anti-diagonal: TR+BL filled).  For every such isolated 2bs x 2bs configuration the
output stamps two bs x bs blocks of color 8 at the EMPTY diagonal, shifted one
block OUTWARD along that diagonal.  Block sizes 1,2,3 all occur (single-cell
"dominoes" are bs=1).  Grids are always 10x10 so position-dependent shifts are safe.

Implementation (opset-10, static [1,1,30,30] masks):
  * blocksum_k via Conv(ones[1,1,k,k], pads bottom/right=k-1)  -> sum over the
    k x k block anchored top-left at each cell.
  * full = (blocksum_bs == bs^2),  empty = (blocksum_bs == 0).
  * ring_clean = (blocksum_{2bs+2}@(i-1,j-1) == blocksum_{2bs}@(i,j))  -> the
    2bs x 2bs bbox is isolated (kills cross-block-size false positives).
  * main[i,j] / anti[i,j] = the four-quadrant test (full/full/empty/empty) AND
    ring_clean.  (i,j) is the bbox top-left.
  * each of the 4 stamps = shift the anchor to its reference corner then dilate
    bs x bs (Conv-sum > 0).
  * output = input + out8 * K   with K[0]=-1, K[8]=+1 (out8 lands only on bg).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
H, W = HEIGHT, WIDTH


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

    def i64(self, vals, dims=None):
        n = self.nm("i")
        dims = dims if dims is not None else [len(vals)]
        self.inits.append(oh.make_tensor(n, INT64, list(dims),
                          [int(v) for v in np.asarray(vals).ravel()]))
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


def _shift(g, x, dr, dc):
    """out[i,j] = x[i-dr, j-dc] (content moves by (dr,dc)), zero fill."""
    if dr == 0 and dc == 0:
        return x
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0,
             pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
    ax = g.i64([2, 3])
    return g.nd("Slice", [p, st, en, ax])


def _look(g, x, di, dj):
    """out[i,j] = x[i+di, j+dj]."""
    return _shift(g, x, -di, -dj)


def _blocksum(g, M, k):
    """out[i,j] = sum of M over rows[i..i+k-1] cols[j..j+k-1]."""
    if k == 1:
        return M
    w = g.f([1, 1, k, k], np.ones(k * k))
    return g.nd("Conv", [M, w], kernel_shape=[k, k], pads=[0, 0, k - 1, k - 1])


def _ge(g, x, thr):
    t = g.f([1, 1, 1, 1], [thr])
    return g.nd("Cast", [g.nd("Greater", [x, t])], to=F)


def _le(g, x, thr):
    t = g.f([1, 1, 1, 1], [thr])
    return g.nd("Cast", [g.nd("Less", [x, t])], to=F)


def _muln(g, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = g.nd("Mul", [acc, x])
    return acc


def _stamp(g, anchor, ro_lo, co_lo, bs):
    """OR over ro in [ro_lo, ro_lo+bs-1], co in [co_lo, co_lo+bs-1] of anchor[p+ro,q+co]."""
    a1 = _look(g, anchor, ro_lo, co_lo)
    s = _blocksum(g, a1, bs)
    return _ge(g, s, 0.5)


def build():
    g = _G()
    M3 = g.nd("Slice", ["input", g.i64([3]), g.i64([4]), g.i64([1])])  # [1,1,30,30]
    out8 = None
    for bs in (1, 2, 3):
        bsum = _blocksum(g, M3, bs)
        full = _ge(g, bsum, bs * bs - 0.5)
        empty = _le(g, bsum, 0.5)
        inner = _blocksum(g, M3, 2 * bs)
        outer = _blocksum(g, M3, 2 * bs + 2)
        outer_s = _shift(g, outer, 1, 1)              # outer@(i-1,j-1)
        ring_clean = _le(g, g.nd("Sub", [outer_s, inner]), 0.5)

        main = _muln(g, [full,
                         _look(g, full, bs, bs),
                         _look(g, empty, 0, bs),
                         _look(g, empty, bs, 0),
                         ring_clean])
        anti = _muln(g, [_look(g, full, 0, bs),
                         _look(g, full, bs, 0),
                         empty,
                         _look(g, empty, bs, bs),
                         ring_clean])

        stamps = [
            _stamp(g, main, 1, -3 * bs + 1, bs),          # main TR
            _stamp(g, main, -3 * bs + 1, 1, bs),          # main BL
            _stamp(g, anti, 1, 1, bs),                    # anti TL
            _stamp(g, anti, -3 * bs + 1, -3 * bs + 1, bs),  # anti BR
        ]
        for s in stamps:
            out8 = s if out8 is None else g.nd("Max", [out8, s])

    # restrict stamps to the real grid region (padding cells must stay all-zero):
    # real cells have exactly one channel == 1, padding cells are all-zero.
    occ = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    grid = _ge(g, occ, 0.5)
    out8 = g.nd("Mul", [out8, grid])

    K = g.f([1, CHANNELS, 1, 1], [-1.0] + [0.0] * 7 + [1.0, 0.0])
    delta = g.nd("Mul", [out8, K])
    g.nd("Add", ["input", delta], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX numerics) for detection                    #
# --------------------------------------------------------------------------- #
def _np_blocksum(M, k):
    cs = np.zeros((H + 1, W + 1))
    cs[1:, 1:] = np.cumsum(np.cumsum(M, 0), 1)
    out = np.zeros_like(M)
    for i in range(H):
        for j in range(W):
            i2, j2 = min(i + k, H), min(j + k, W)
            out[i, j] = cs[i2, j2] - cs[i, j2] - cs[i2, j] + cs[i, j]
    return out


def _np_look(M, di, dj):
    out = np.zeros_like(M)
    for i in range(H):
        for j in range(W):
            ii, jj = i + di, j + dj
            if 0 <= ii < H and 0 <= jj < W:
                out[i, j] = M[ii, jj]
    return out


def _np_stamp(anchor, ro_lo, co_lo, bs):
    out = np.zeros_like(anchor)
    for ro in range(ro_lo, ro_lo + bs):
        for co in range(co_lo, co_lo + bs):
            out = np.maximum(out, _np_look(anchor, ro, co))
    return out


def _ref(a):
    h, w = a.shape
    M3 = np.zeros((H, W))
    M3[:h, :w] = (a == 3).astype(float)
    out8 = np.zeros((H, W))
    for bs in (1, 2, 3):
        bsum = _np_blocksum(M3, bs)
        full = (bsum > bs * bs - 0.5).astype(float)
        empty = (bsum < 0.5).astype(float)
        inner = _np_blocksum(M3, 2 * bs)
        outer_s = _np_look(_np_blocksum(M3, 2 * bs + 2), -1, -1)
        ring = ((outer_s - inner) < 0.5).astype(float)
        main = full * _np_look(full, bs, bs) * _np_look(empty, 0, bs) * _np_look(empty, bs, 0) * ring
        anti = _np_look(full, 0, bs) * _np_look(full, bs, 0) * empty * _np_look(empty, bs, bs) * ring
        out8 = np.maximum(out8, _np_stamp(main, 1, -3 * bs + 1, bs))
        out8 = np.maximum(out8, _np_stamp(main, -3 * bs + 1, 1, bs))
        out8 = np.maximum(out8, _np_stamp(anti, 1, 1, bs))
        out8 = np.maximum(out8, _np_stamp(anti, -3 * bs + 1, -3 * bs + 1, bs))
    res = a.copy()
    res[(out8[:h, :w] > 0.5)] = 8
    return res


def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    # our family: same-shape, adds color 8 to color-3 diagonal dominoes
    if any(a.shape != b.shape for a, b in prs):
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    for a, b in prs:
        if _ref(a).shape != b.shape or not np.array_equal(_ref(a), b):
            return []
    try:
        m = build()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("crk9_5_diag8", m)]
