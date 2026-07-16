"""DATA-DEPENDENT CROP TO BOUNDING BOX (opset-10, static shapes via MatMul shifts).

Every rule here CROPS the input to the axis-aligned bounding box of some content
mask and re-anchors the cropped region at the TOP-LEFT origin.  Because the
one-hot tensor is zero-padded to 30x30 with the grid at (0,0) and grid sizes vary,
the destination of every cell depends on a DATA-DEPENDENT offset (minrow / mincol)
that a naive static Slice/Pad cannot express.  We realise the shift with the
"computed selection matrix + MatMul" trick:

    minrow/mincol/maxrow/maxcol  <-  position-weighted ReduceMax of the content
                                     mask (no Loop / NonZero needed).
    Srow[r,k] = 1  iff  k == r + minrow  and  r < bbox_h     ([1,1,30,30])
    Scol[w,j] = 1  iff  j == w - mincol  and  j < bbox_w     ([1,1,30,30])
    output = MatMul(Srow, MatMul(input, Scol))               ([1,10,30,30])

The two selection matrices are built from constant index grids with
Sub/Add/Abs/Less/Cast and carry the bbox truncation INSIDE the matrix (rows /
columns beyond the box are all-zero), so the second MatMul writes "output"
directly with no extra mask multiply.  All values stay exactly 0/1 (each output
cell sums a single 0/1 product), so the grader's `output > 0` threshold is exact.

Content variants (the matching one is inferred from train/test/arc-gen pairs)
----------------------------------------------------------------------------
  nonbg      bbox of ALL non-background cells (channels 1..9).
  mostfreq   bbox of the most-frequent non-background colour (ArgMax over per-
             channel counts, channel-0 suppressed, first-index tie-break).
  color C    bbox of a single fixed colour C (same C across every pair).
  largest    bbox of the LARGEST 4-connected non-background object.  The object
             labels are computed at inference time by a label-propagation cellular
             automaton (Pad/Slice shifts + Max), component sizes by a [900,900]
             equality matrix, and the max-size component's cells select the bbox.

In every variant the WHOLE input region inside the box is kept (re-anchored), so
background cells inside the box survive — matching the ARC "crop to bounding box"
semantics.  Detection mirrors the ONNX numerics exactly (ArgMax tie-break, the
"all cells of a max-size component" selection) and only emits a candidate when it
reproduces EVERY available pair, so wrong hypotheses are dropped before scoring.
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
INT32 = onnx.TensorProto.INT32
F = DATA_TYPE
H, W = HEIGHT, WIDTH
_CBIG = 1000.0


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

    def i64(self, vals, dims=None):
        n = self.nm("i")
        dims = dims if dims is not None else [len(vals)]
        self.inits.append(oh.make_tensor(n, INT64, list(dims), [int(v) for v in np.asarray(vals).ravel()]))
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


# --------------------------------------------------------------------------- #
# shared constants / masks                                                     #
# --------------------------------------------------------------------------- #
def _consts(g):
    g.rowidx = g.f([1, 1, H, 1], list(range(H)))     # i / r grid  [1,1,30,1]
    g.colidx = g.f([1, 1, 1, W], list(range(W)))     # j / k grid  [1,1,1,30]
    g.half = g.f([1, 1, 1, 1], [0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])
    g.cbig = g.f([1, 1, 1, 1], [_CBIG])


def _nonbg_mask(g):
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)            # [1,1,30,30]
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])      # [1,1,30,30]
    return g.nd("Sub", [realmask, ch0])                                     # [1,1,30,30]


# --------------------------------------------------------------------------- #
# content masks  ->  [1,1,30,30]  (1 on the cells that define the bbox)        #
# --------------------------------------------------------------------------- #
def _content_nonbg(g):
    return _nonbg_mask(g)


def _content_color(g, c):
    return g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])  # [1,1,30,30]


def _content_mostfreq(g):
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    bgneg = g.f([1, CHANNELS, 1, 1], [-_CBIG] + [0.0] * (CHANNELS - 1))
    sel = g.nd("Add", [counts, bgneg])
    amax = g.nd("ArgMax", [sel], axis=1, keepdims=1)                        # int64 [1,1,1,1]
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    gate = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)                 # [1,10,1,1]
    seloh = g.nd("Mul", ["input", gate])                                    # [1,10,30,30]
    return g.nd("ReduceSum", [seloh], axes=[1], keepdims=1)                 # [1,1,30,30]


def _content_largest(g, T):
    """Mask [1,1,30,30] of the cells of the largest 4-connected non-bg component."""
    M = _nonbg_mask(g)                                                      # [1,1,30,30]

    def shift(x, dr, dc):
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        p = g.nd("Pad", [x], mode="constant", value=0.0,
                 pads=[0, 0, pt, pl, 0, 0, pb, pr])
        st = g.i64([max(-dr, 0), max(-dc, 0)])
        en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
        ax = g.i64([2, 3])
        return g.nd("Slice", [p, st, en, ax])

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    P = g.f([1, 1, H, W], np.arange(1, H * W + 1))
    L = g.nd("Mul", [M, P])
    for _ in range(T):
        nbrs = [shift(L, dr, dc) for dr, dc in dirs]
        mx = g.nd("Max", [L] + nbrs)
        L = g.nd("Mul", [mx, M])

    Lcol = g.nd("Reshape", [L, g.i64([H * W, 1])])                          # [900,1]
    Lrow = g.nd("Reshape", [L, g.i64([1, H * W])])                          # [1,900]
    E = g.nd("Equal", [g.nd("Cast", [Lcol], to=INT32), g.nd("Cast", [Lrow], to=INT32)])
    Ef = g.nd("Cast", [E], to=F)                                           # [900,900]
    size_col = g.nd("ReduceSum", [Ef], axes=[1], keepdims=1)                # [900,1]
    size2d = g.nd("Reshape", [size_col, g.i64([1, 1, H, W])])              # [1,1,30,30]
    size2d = g.nd("Mul", [size2d, M])                                      # zero bg/padding
    maxsize = g.nd("ReduceMax", [size2d], axes=[2, 3], keepdims=1)         # [1,1,1,1]
    # cells whose component size equals the max (size2d >= maxsize, integer-valued)
    return g.nd("Cast", [g.nd("Greater", [size2d, g.nd("Sub", [maxsize, g.half])])], to=F)


# --------------------------------------------------------------------------- #
# bbox crop machinery (shared)                                                 #
# --------------------------------------------------------------------------- #
def _finish_crop(g, content):
    rowidx, colidx = g.rowidx, g.colidx
    half, one, cbig = g.half, g.one, g.cbig

    rowhas = g.nd("ReduceMax", [content], axes=[3], keepdims=1)             # [1,1,30,1]
    colhas = g.nd("ReduceMax", [content], axes=[2], keepdims=1)             # [1,1,1,30]

    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    bbox_h = g.nd("Add", [g.nd("Sub", [maxrow, minrow]), one])             # [1,1,1,1]
    bbox_w = g.nd("Add", [g.nd("Sub", [maxcol, mincol]), one])            # [1,1,1,1]

    # Scol[w,j] = (j == w - mincol) & (j < bbox_w)
    diff_c = g.nd("Sub", [g.nd("Add", [colidx, mincol]), rowidx])         # [1,1,30,30]
    match_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_c]), half])], to=F)
    trunc_c = g.nd("Cast", [g.nd("Less", [colidx, bbox_w])], to=F)        # [1,1,1,30]
    Scol = g.nd("Mul", [match_c, trunc_c])                                # [1,1,30,30]

    # Srow[r,k] = (k == r + minrow) & (r < bbox_h)
    diff_r = g.nd("Sub", [colidx, g.nd("Add", [rowidx, minrow])])        # [1,1,30,30]
    match_r = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_r]), half])], to=F)
    trunc_r = g.nd("Cast", [g.nd("Less", [rowidx, bbox_h])], to=F)        # [1,1,30,1]
    Srow = g.nd("Mul", [match_r, trunc_r])                                # [1,1,30,30]

    shift1 = g.nd("MatMul", ["input", Scol])                              # [1,10,30,30]
    g.nd("MatMul", [Srow, shift1], "output")                             # [1,10,30,30]


def build_crop(content_kind, color=None, T=30):
    g = _G()
    _consts(g)
    if content_kind == "nonbg":
        content = _content_nonbg(g)
    elif content_kind == "mostfreq":
        content = _content_mostfreq(g)
    elif content_kind == "largest":
        content = _content_largest(g, T)
    else:
        content = _content_color(g, color)
    _finish_crop(g, content)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX numerics for detection)                    #
# --------------------------------------------------------------------------- #
def _bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _crop(a, mask):
    bb = _bbox(mask)
    if bb is None:
        return None
    r0, r1, c0, c1 = bb
    return a[r0:r1 + 1, c0:c1 + 1].copy()


def _ref_nonbg(a):
    return _crop(a, a != 0)


def _ref_color(a, c):
    return _crop(a, a == c)


def _mostfreq_color(a):
    cnt = np.array([int((a == c).sum()) for c in range(CHANNELS)], np.int64)
    cnt[0] = -(10 ** 9)                       # suppress background (ArgMax first-index tie-break)
    return int(cnt.argmax())


def _ref_mostfreq(a):
    return _crop(a, a == _mostfreq_color(a))


def _components_any(a):
    """4-connected components of all non-bg cells (any colour), like the ONNX CA."""
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    out = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == 0:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            cells = [(i, j)]
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            out.append(cells)
    return out


def _ref_largest(a):
    comps = _components_any(a)
    if not comps:
        return None
    maxs = max(len(c) for c in comps)
    mask = np.zeros(a.shape, bool)
    for c in comps:
        if len(c) == maxs:                    # mirror ONNX: select ALL max-size components
            for r, cc in c:
                mask[r, cc] = True
    return _crop(a, mask)


def _diameter_any(a):
    worst = 0
    for cells in _components_any(a):
        cset = set(cells)
        root = max(cells, key=lambda rc: rc[0] * W + rc[1])
        dist = {root: 0}
        q = deque([root])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in cset and nb not in dist:
                    dist[nb] = dist[(r, c)] + 1
                    q.append(nb)
        if dist:
            worst = max(worst, max(dist.values()))
    return worst


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _matches(prs, fn):
    for a, b in prs:
        o = fn(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def _emit(out, seen, name, builder):
    if name in seen:
        return
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    seen.add(name)
    out.append((name, m))


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):       # pure identity -> not our family
        return []
    # crop generally shrinks: require a genuine shrink on at least one pair
    if not any(b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1] for a, b in prs):
        return []

    out, seen = [], set()

    if _matches(prs, _ref_nonbg):
        _emit(out, seen, "dyncrop_nonbg", lambda: build_crop("nonbg"))

    if _matches(prs, _ref_mostfreq):
        _emit(out, seen, "dyncrop_mostfreq", lambda: build_crop("mostfreq"))

    for c in range(1, CHANNELS):
        if _matches(prs, lambda a, c=c: _ref_color(a, c)):
            _emit(out, seen, f"dyncrop_color{c}", lambda c=c: build_crop("color", c))

    # largest 4-connected object (only attempt when the cheaper rules miss)
    if not out and _matches(prs, _ref_largest):
        need = max((_diameter_any(a) for a, _ in prs), default=0)
        T = min(70, max(28, need + 16))
        _emit(out, seen, "dyncrop_largest", lambda: build_crop("largest", T=T))

    return out
