"""Region-conditioned recolor with a STATIC, origin-anchored mask.

Rule family
-----------
    output[i,j] = map[input[i,j]]   if  M[i,j]      (recolor inside a fixed region)
                  input[i,j]        otherwise        (identity outside)

where `M` is a FIXED binary mask over absolute (top-left anchored) positions that
does NOT depend on the (variable, data-dependent) grid size, and `map` is a single
global per-color recolor.  Because the one-hot tensor is zero-padded to 30x30 and
grids are anchored at (0,0), a mask defined on absolute positions generalises:
"top-left k x k", "first k rows", "first k columns", "every other row/column",
"checkerboard", "(upper/lower) triangle", "main diagonal", and the complement of
each of those.  Masks whose correct location depends on the grid size (e.g. the
bottom/right border, an anti-diagonal) are NOT in this family and are never emitted.

Realisation (opset 10, origin-safe, cheap)
------------------------------------------
    recolored = Conv1x1(input, W)            # one [1,10,30,30] intermediate (36000 B)
    output    = Where(mask, recolored, input)

`W[o,i,0,0] = 1 iff map[i]==o` is the 1x1 recolor (100 params).  `mask` is a BOOL
initializer broadcast over the 10 channels; when the region depends on only the row
(or only the column) the mask collapses to [1,1,30,1] (or [1,1,1,30]) -> 30 params.
A Conv with no bias maps the all-zero padding columns to zero, and Where keeps the
zero on both branches, so padding stays <=0 on every channel.  cost ~= 36000 +
~130 params -> ~14.5 pts.

Detection fits the global recolor `map` from the cells that actually change, derives
hard "must-mask"/"must-not-mask" constraints, then searches a parametric bank of
size-independent masks, keeping only those that reproduce every train+test+arc-gen
pair EXACTLY (the grader's gate).  Degenerate full-grid masks (== a plain global
recolor) are skipped so this family stays region-conditioned.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


# --------------------------------------------------------------------------- #
# graph helper                                                                #
# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(ex):
    """All usable (input, output) raw int grids; skip >30 grids (grader ignores)."""
    out = []
    for e in ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", []):
        a = np.array(e["input"], int)
        b = np.array(e["output"], int)
        if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
            continue
        if max(a.shape) > 30 or max(b.shape) > 30:
            continue
        out.append((a, b))
    return out


# --------------------------------------------------------------------------- #
# parametric mask bank (functions of absolute row i / column j over 0..29)     #
# --------------------------------------------------------------------------- #
def _gen_masks():
    H, W = HEIGHT, WIDTH
    I, J = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out, seen = [], set()

    def add(name, M):
        M = np.asarray(M, bool)
        if not M.any() or M.all():           # empty / full -> not region-conditioned
            return
        key = M.tobytes()
        if key in seen:
            return
        seen.add(key)
        out.append((name, M))

    for k in range(1, H):                     # top-left k x k square
        add(f"tl{k}", (I < k) & (J < k))
    for k in range(1, H):                     # first k rows
        add(f"top{k}", I < k)
    for k in range(1, W):                     # first k columns
        add(f"left{k}", J < k)
    for p in (2, 3, 4):                       # every p-th row / column, phase r
        for r in range(p):
            add(f"rmod{p}_{r}", (I % p) == r)
            add(f"cmod{p}_{r}", (J % p) == r)
    for r in (0, 1):                          # checkerboard
        add(f"chk{r}", ((I + J) % 2) == r)
    add("diag", I == J)                       # main diagonal (origin-anchored)
    add("triU", I < J)                        # strict upper triangle
    add("triL", I > J)                        # strict lower triangle
    add("triUe", I <= J)
    add("triLe", I >= J)

    for name, M in list(out):                 # complements (also size-independent)
        add("not_" + name, ~M)
    return out


def _periodic_mask(P, N):
    """Smallest-period 2D mask (period pr x pc, tiled over 30x30) consistent with
    the must-mask (P) / must-not-mask (N) constraints, or None.  Captures size-
    independent periodic fills (e.g. 'recolor every other repeating motif').
    Unknown period-classes default to 0 (unmasked)."""
    known = P | N
    ii, jj = np.where(known)
    if ii.size == 0:
        return None
    val = P[ii, jj].astype(np.int64)          # 1 must-mask, 0 must-not-mask
    best = None
    for pr in range(1, 13):
        for pc in range(1, 16):
            cls = (ii % pr) * pc + (jj % pc)
            n = pr * pc
            tmin = np.full(n, 2, np.int64)
            tmax = np.full(n, -1, np.int64)
            np.minimum.at(tmin, cls, val)
            np.maximum.at(tmax, cls, val)
            if np.any((tmax == 1) & (tmin == 0)):
                continue                       # class wants both -> inconsistent
            tile = np.where(tmax == 1, 1, 0).reshape(pr, pc)
            if not tile.any():
                continue
            M = np.tile(tile, (HEIGHT // pr + 1, WIDTH // pc + 1))
            M = M[:HEIGHT, :WIDTH].astype(bool)
            best = (f"per{pr}x{pc}", M)
            return best
    return None


# --------------------------------------------------------------------------- #
# model construction                                                           #
# --------------------------------------------------------------------------- #
def _mask_init(M):
    """Smallest broadcast representation of a 30x30 bool mask (BOOL initializer)."""
    if (M == M[:, :1]).all():                 # depends on row only
        return [1, 1, HEIGHT, 1], M[:, 0].astype(int).tolist()
    if (M == M[:1, :]).all():                 # depends on column only
        return [1, 1, 1, WIDTH], M[0, :].astype(int).tolist()
    return [1, 1, HEIGHT, WIDTH], M.astype(int).ravel().tolist()


def _build(M, cmap):
    W = np.zeros((CHANNELS, CHANNELS, 1, 1), np.float32)
    for i, o in enumerate(cmap):
        W[o, i, 0, 0] = 1.0
    wt = oh.make_tensor("W", DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], W.ravel().tolist())
    conv = oh.make_node("Conv", ["input", "W"], ["recolored"],
                        kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    dims, vals = _mask_init(M)
    mt = oh.make_tensor("mask", BOOL, dims, vals)
    where = oh.make_node("Where", ["mask", "recolored", "input"], ["output"])
    return _model([conv, where], [wt, mt])


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):   # recolor preserves shape
        return []

    # global color map from cells that change -------------------------------- #
    cmap = list(range(CHANNELS))
    defined = [False] * CHANNELS
    P = np.zeros((HEIGHT, WIDTH), bool)            # must-be-masked positions
    for a, b in prs:
        d = a != b
        if not d.any():
            continue
        for ic, oc in zip(a[d].tolist(), b[d].tolist()):
            if defined[ic] and cmap[ic] != oc:
                return []                          # not a single global recolor
            cmap[ic] = oc
            defined[ic] = True
        di, dj = np.where(d)
        P[di, dj] = True
    if not P.any():
        return []                                  # nothing recolored
    cmap_arr = np.array(cmap)

    # must-NOT-be-masked positions (would have changed under map but didn't) -- #
    N = np.zeros((HEIGHT, WIDTH), bool)
    for a, b in prs:
        h, w = a.shape
        forced0 = (a == b) & (cmap_arr[a] != a)
        ii, jj = np.where(forced0)
        N[ii, jj] = True

    maxh = max(a.shape[0] for a, _ in prs)
    maxw = max(a.shape[1] for a, _ in prs)

    bank = _gen_masks()
    per = _periodic_mask(P, N)                  # size-independent periodic fill
    if per is not None:
        bank = bank + [per]

    results, seen_behaviour = [], set()
    for name, M in bank:
        if (P & ~M).any() or (N & M).any():        # constraint pre-filter
            continue
        # degenerate full-grid mask == plain global recolor -> not our family
        if all(M[:a.shape[0], :a.shape[1]].all() for a, _ in prs):
            continue
        ok = True
        for a, b in prs:
            h, w = a.shape
            exp = np.where(M[:h, :w], cmap_arr[a], a)
            if not np.array_equal(exp, b):
                ok = False
                break
        if not ok:
            continue
        beh = M[:maxh, :maxw].tobytes()            # dedupe equivalent masks
        if beh in seen_behaviour:
            continue
        seen_behaviour.add(beh)
        dims, _ = _mask_init(M)
        results.append((int(np.prod(dims)), name, M))

    if not results:
        return []
    results.sort(key=lambda r: (r[0], r[1]))        # cheapest mask first
    out = []
    for _, name, M in results[:4]:
        try:
            out.append((f"maskmap_{name}", _build(M, cmap)))
        except Exception:
            continue
    return out
