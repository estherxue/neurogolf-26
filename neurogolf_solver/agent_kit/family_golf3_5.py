"""family_golf3_5 -- cheaper exact solvers for a stride slice of golf targets.

Each candidate re-derives the task rule from train+test+arc-gen pairs with a numpy
reference, verifies EXACT equality on every available pair, and only then emits a
minimal opset-10 ONNX graph.  The integrator auto-picks the cheapest correct model,
so these only have to be exact and cheaper than the incumbent.

Targets golfed here (rules verified exact on 100% of provided examples):
  * 289 crk2_upcnt   -> nearest-upscale a 3x3 grid by k = #distinct nonzero colours
  * 199 spray4       -> a lone marker sprays colour-4 checkerboard up its column-parity
  * 31  dyncrop_nonbg-> crop to the bounding box of the non-background content
  * 290 crop_swap2   -> crop to bbox, then swap the two present colours
  * 384 bboxcrop_s2  -> crop to bbox, then 2x nearest upscale
  * 225 t225         -> radiate diagonally-opposite block-corner colours into 4 corners
  * 24  lines_v2_h   -> colour-2 marker = vertical line, others = horizontal line
  * 342 crk2_5_quad  -> 4 quadrant markers paint the central 2x2 block corners

Cost levers used: dynamic crop/upscale via two tiny [30,30]/[30,3] MatMuls whose
final product is written straight into the FREE "output"; bbox extents found with
ArgMax; colours pulled out with Gather (small [1,10,k] intermediates); per-cell
selection via boolean masks fed to Where with the FREE "input" tensor as the base.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def fconst(self, vals, shape):
        nm = self.name("f")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(shape),
                                         [float(v) for v in vals]))
        return nm

    def scalar(self, v):
        return self.fconst([v], [1])

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.iconst(starts), g.iconst(ends), g.iconst(axes)]
    if steps is not None:
        ins.append(g.iconst(steps))
    return g.node("Slice", ins)


def _onehot_channel(g, ch):
    """const [1,10,1,1] one-hot for a single colour channel."""
    vals = [1.0 if i == ch else 0.0 for i in range(CHANNELS)]
    return g.fconst(vals, [1, CHANNELS, 1, 1])


def _chan_mask(g, channels):
    """const [1,10,1,1] that is 1 on the given channels, else 0."""
    vals = [1.0 if i in channels else 0.0 for i in range(CHANNELS)]
    return g.fconst(vals, [1, CHANNELS, 1, 1])


def _ftensor(g, arr):
    arr = np.asarray(arr, np.float32)
    return g.fconst(arr.ravel().tolist(), list(arr.shape))


# common reductions ---------------------------------------------------------- #
def _nz(g):
    """presence of a non-background colour: [1,1,30,30]."""
    ch = _slice(g, "input", [1], [CHANNELS], [1])
    return g.node("ReduceMax", [ch], axes=[1], keepdims=1)


def _realcell(g):
    """1 for real cells (any one-hot channel set), 0 for padding: [1,1,30,30]."""
    return g.node("ReduceSum", ["input"], axes=[1], keepdims=1)


def _argmax_f(g, x, axis):
    """ArgMax -> float [.. with that axis = 1]."""
    a = g.node("ArgMax", [x], axis=axis, keepdims=1)
    return g.node("Cast", [a], to=DATA_TYPE)


def _bbox(g):
    """rmin,rmax,cmin,cmax as float [1,1,1,1] each (bbox of non-bg)."""
    nz = _nz(g)
    rowhas = g.node("ReduceMax", [nz], axes=[3], keepdims=1)   # [1,1,30,1]
    colhas = g.node("ReduceMax", [nz], axes=[2], keepdims=1)   # [1,1,1,30]
    ridx = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])
    cidx = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])
    rmin = _argmax_f(g, rowhas, 2)
    rmax = _argmax_f(g, g.node("Mul", [rowhas, ridx]), 2)
    cmin = _argmax_f(g, colhas, 3)
    cmax = _argmax_f(g, g.node("Mul", [colhas, cidx]), 3)
    return rmin, rmax, cmin, cmax


# --------------------------------------------------------------------------- #
# pairs                                                                        #
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


def _exact(ref, prs):
    if not prs:
        return False
    for a, b in prs:
        try:
            r = ref(a)
        except Exception:
            return False
        if r is None or not isinstance(r, np.ndarray):
            return False
        if r.shape != b.shape or not np.array_equal(r, b):
            return False
    return True


# ===========================================================================
# 289  crk2_upcnt : nearest-upscale a 3x3 grid by k = #distinct nonzero colours
# ===========================================================================
def _ref_289(a):
    if a.shape != (3, 3):
        return None
    k = len(set(a[a > 0].tolist()))
    if k < 1:
        return None
    return np.kron(a, np.ones((k, k), int))


def _build_289():
    g = _G()
    in3 = _slice(g, "input", [0, 0], [3, 3], [2, 3])           # [1,10,3,3]
    pres = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    pres9 = _slice(g, pres, [1], [CHANNELS], [1])             # [1,9,1,1]
    k = g.node("ReduceSum", [pres9], axes=[1], keepdims=1)    # [1,1,1,1]
    kf = g.node("Reshape", [k, g.iconst([1, 1])])            # [1,1]

    # A[i,r] (i 0..29 rows, r 0..2)  = 1 iff floor(i/k)==r  i.e. r*k<=i<=r*k+k-1
    ivec = g.fconst(list(range(HEIGHT)), [HEIGHT, 1])         # [30,1]
    rvec = g.fconst([0.0, 1.0, 2.0], [1, 3])                  # [1,3]
    rk = g.node("Mul", [rvec, kf])                            # [1,3]
    d = g.node("Sub", [ivec, rk])                             # [30,3] = i - r*k
    lo = g.node("Cast", [g.node("Greater", [d, g.scalar(-0.5)])], to=DATA_TYPE)
    km1 = g.node("Sub", [kf, g.scalar(0.5)])                  # k-0.5
    hi = g.node("Cast", [g.node("Less", [d, km1])], to=DATA_TYPE)
    A = g.node("Mul", [lo, hi])                               # [30,3]
    A4 = g.node("Reshape", [A, g.iconst([1, 1, HEIGHT, 3])])  # [1,1,30,3]

    # B[m,j] (m 0..2, j 0..29 cols) = 1 iff floor(j/k)==m
    jvec = g.fconst(list(range(WIDTH)), [1, WIDTH])           # [1,30]
    mvec = g.fconst([0.0, 1.0, 2.0], [3, 1])                  # [3,1]
    mk = g.node("Mul", [mvec, kf])                            # [3,1]
    d2 = g.node("Sub", [jvec, mk])                            # [3,30]
    lo2 = g.node("Cast", [g.node("Greater", [d2, g.scalar(-0.5)])], to=DATA_TYPE)
    hi2 = g.node("Cast", [g.node("Less", [d2, km1])], to=DATA_TYPE)
    B = g.node("Mul", [lo2, hi2])                             # [3,30]
    B4 = g.node("Reshape", [B, g.iconst([1, 1, 3, WIDTH])])   # [1,1,3,30]

    t1 = g.node("MatMul", [A4, in3])                          # [1,10,30,3]
    g.node("MatMul", [t1, B4], "output")                     # [1,10,30,30]
    return _model(g.nodes, g.inits)


# ===========================================================================
# 199  spray4 : lone marker sprays colour-4 in same-parity columns up to its row
# ===========================================================================
def _ref_199(a):
    nz = np.argwhere(a > 0)
    if len(nz) != 1:
        return None
    r, c = nz[0]
    col = a[r, c]
    H, W = a.shape
    o = np.zeros_like(a)
    for i in range(0, r + 1):
        for j in range(W):
            if (j % 2) == (c % 2):
                o[i, j] = 4
    if r + 1 < H:
        o[r + 1, c] = col
    return o


def _build_199():
    g = _G()
    nz = _nz(g)
    real = _realcell(g)
    rowhas = g.node("ReduceMax", [nz], axes=[3], keepdims=1)   # [1,1,30,1]
    colhas = g.node("ReduceMax", [nz], axes=[2], keepdims=1)   # [1,1,1,30]
    r = _argmax_f(g, rowhas, 2)                                # [1,1,1,1]
    c = _argmax_f(g, colhas, 3)                                # [1,1,1,1]
    ridx = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])
    cidx = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])

    # region4: rows<=r, same column parity, real cell
    rowmask = g.node("Cast", [g.node("Less", [ridx, g.node("Add", [r, g.scalar(0.5)])])],
                     to=DATA_TYPE)                              # i<=r  -> [1,1,30,1]
    par = g.node("Mod", [g.node("Add", [cidx, c]), g.scalar(2.0)], fmod=1)  # [1,1,1,30]
    colmask = g.node("Cast", [g.node("Less", [par, g.scalar(0.5)])], to=DATA_TYPE)
    region = g.node("Mul", [g.node("Mul", [rowmask, colmask]), real])  # [1,1,30,30]
    regionb = g.node("Greater", [region, g.scalar(0.5)])

    # marker position (r+1, c)
    posr = g.node("Cast", [g.node("Less",
                  [g.node("Abs", [g.node("Sub", [ridx, g.node("Add", [r, g.scalar(1.0)])])]),
                   g.scalar(0.5)])], to=DATA_TYPE)             # [1,1,30,1]
    posc = g.node("Cast", [g.node("Less",
                  [g.node("Abs", [g.node("Sub", [cidx, c])]), g.scalar(0.5)])], to=DATA_TYPE)
    pos = g.node("Mul", [g.node("Mul", [posr, posc]), real])
    posb = g.node("Greater", [pos, g.scalar(0.5)])

    # marker colour one-hot [1,10,1,1]
    pres = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    mcol = g.node("Mul", [pres, _chan_mask(g, set(range(1, CHANNELS)))])

    inner = g.node("Where", [posb, mcol, "input"])            # marker placed
    g.node("Where", [regionb, _onehot_channel(g, 4), inner], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# crop helpers : build A,B that translate bbox top-left to origin (+ optional zoom)
# ===========================================================================
def _crop_matrices(g, rmin, cmin, h, w, zoom=1):
    """Return A4 [1,1,30,30], B4 [1,1,30,30] such that A@X@B = the cropped
    (and zoom-upscaled) content anchored at the origin, padded with zeros."""
    II = g.fconst(list(range(HEIGHT)), [HEIGHT, 1])           # output row idx
    RR = g.fconst(list(range(HEIGHT)), [1, HEIGHT])           # input row idx
    JJ = g.fconst(list(range(WIDTH)), [WIDTH, 1])             # input col idx
    KK = g.fconst(list(range(WIDTH)), [1, WIDTH])             # output col idx
    if zoom == 1:
        srcr = g.node("Add", [II, rmin])                     # [30,1] want RR==i+rmin
        srcc = g.node("Add", [KK, cmin])                     # [1,30] want JJ==k+cmin
        lim_r = h            # valid output rows
        lim_c = w
    else:
        half_i = g.fconst([i // zoom for i in range(HEIGHT)], [HEIGHT, 1])
        half_k = g.fconst([k // zoom for k in range(WIDTH)], [1, WIDTH])
        srcr = g.node("Add", [half_i, rmin])
        srcc = g.node("Add", [half_k, cmin])
        lim_r = None
        lim_c = None
    A = g.node("Cast", [g.node("Less",
              [g.node("Abs", [g.node("Sub", [RR, srcr])]), g.scalar(0.5)])], to=DATA_TYPE)
    B = g.node("Cast", [g.node("Less",
              [g.node("Abs", [g.node("Sub", [JJ, srcc])]), g.scalar(0.5)])], to=DATA_TYPE)
    # restrict to the valid output extent so padding stays all-zero
    if zoom == 1:
        rowlim = g.node("Cast", [g.node("Less",
                  [II, g.node("Sub", [h, g.scalar(0.5)])])], to=DATA_TYPE)   # i<h
        collim = g.node("Cast", [g.node("Less",
                  [KK, g.node("Sub", [w, g.scalar(0.5)])])], to=DATA_TYPE)   # k<w
    else:
        twoh = g.node("Mul", [h, g.scalar(float(zoom))])
        twow = g.node("Mul", [w, g.scalar(float(zoom))])
        rowlim = g.node("Cast", [g.node("Less",
                  [II, g.node("Sub", [twoh, g.scalar(0.5)])])], to=DATA_TYPE)
        collim = g.node("Cast", [g.node("Less",
                  [KK, g.node("Sub", [twow, g.scalar(0.5)])])], to=DATA_TYPE)
    A = g.node("Mul", [A, rowlim])                            # zero rows beyond extent
    B = g.node("Mul", [B, collim])
    A4 = g.node("Reshape", [A, g.iconst([1, 1, HEIGHT, HEIGHT])])
    B4 = g.node("Reshape", [B, g.iconst([1, 1, WIDTH, WIDTH])])
    return A4, B4


def _hw(g, rmin, rmax, cmin, cmax):
    h = g.node("Add", [g.node("Sub", [rmax, rmin]), g.scalar(1.0)])
    w = g.node("Add", [g.node("Sub", [cmax, cmin]), g.scalar(1.0)])
    return h, w


def _bbox_int(g):
    """rminf,cminf as float [1]; h,w as float [1,1,1,1]."""
    nz = _nz(g)
    rowhas = g.node("ReduceMax", [nz], axes=[3], keepdims=1)
    colhas = g.node("ReduceMax", [nz], axes=[2], keepdims=1)
    ridx = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])
    cidx = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])
    rmaxf = _argmax_f(g, g.node("Mul", [rowhas, ridx]), 2)
    cmaxf = _argmax_f(g, g.node("Mul", [colhas, cidx]), 3)
    rminf = _argmax_f(g, rowhas, 2)
    cminf = _argmax_f(g, colhas, 3)
    h = g.node("Add", [g.node("Sub", [rmaxf, rminf]), g.scalar(1.0)])
    w = g.node("Add", [g.node("Sub", [cmaxf, cminf]), g.scalar(1.0)])
    rmin1 = g.node("Reshape", [rminf, g.iconst([1])])
    cmin1 = g.node("Reshape", [cminf, g.iconst([1])])
    return rmin1, cmin1, h, w


def _crop_gather(g, src, rmin1, cmin1, h, w, zoom=1):
    """Crop bbox (top-left -> origin) with optional integer zoom, via two Gathers
    and a final extent mask written straight to a fresh tensor (returned).
    rmin1,cmin1 are float [1]."""
    az = g.fconst([i // zoom for i in range(HEIGHT)], [HEIGHT])  # [30] float
    aw = g.fconst([j // zoom for j in range(WIDTH)], [WIDTH])
    rf = g.node("Min", [g.node("Max", [g.node("Add", [az, rmin1]), g.scalar(0.0)]),
                        g.scalar(float(HEIGHT - 1))])
    cf = g.node("Min", [g.node("Max", [g.node("Add", [aw, cmin1]), g.scalar(0.0)]),
                        g.scalar(float(WIDTH - 1))])
    ridx = g.node("Cast", [rf], to=INT64)                    # [30]
    cidx = g.node("Cast", [cf], to=INT64)
    rows = g.node("Gather", [src, ridx], axis=2)             # [1,C,30,30]
    grid = g.node("Gather", [rows, cidx], axis=3)            # [1,C,30,30]
    # extent mask: i < zoom*h and j < zoom*w
    II = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])
    KK = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])
    zh = g.node("Mul", [h, g.scalar(float(zoom))])
    zw = g.node("Mul", [w, g.scalar(float(zoom))])
    rl = g.node("Cast", [g.node("Less", [II, g.node("Sub", [zh, g.scalar(0.5)])])], to=DATA_TYPE)
    cl = g.node("Cast", [g.node("Less", [KK, g.node("Sub", [zw, g.scalar(0.5)])])], to=DATA_TYPE)
    mhw = g.node("Mul", [rl, cl])                            # [1,1,30,30]
    return grid, mhw


# ===========================================================================
# 31  dyncrop_nonbg : crop to bbox of non-background
# ===========================================================================
def _ref_31(a):
    nz = np.argwhere(a > 0)
    if len(nz) == 0:
        return None
    r0, c0 = nz.min(0)
    r1, c1 = nz.max(0)
    return a[r0:r1 + 1, c0:c1 + 1]


def _build_31():
    g = _G()
    rmin1, cmin1, h, w = _bbox_int(g)
    grid, mhw = _crop_gather(g, "input", rmin1, cmin1, h, w, zoom=1)
    g.node("Mul", [grid, mhw], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 384  bboxcrop_bg0_s2 : crop bbox then 2x nearest-upscale
# ===========================================================================
def _ref_384(a):
    nz = np.argwhere(a > 0)
    if len(nz) == 0:
        return None
    r0, c0 = nz.min(0)
    r1, c1 = nz.max(0)
    return np.kron(a[r0:r1 + 1, c0:c1 + 1], np.ones((2, 2), int))


def _build_384():
    g = _G()
    rmin1, cmin1, h, w = _bbox_int(g)
    grid, mhw = _crop_gather(g, "input", rmin1, cmin1, h, w, zoom=2)
    g.node("Mul", [grid, mhw], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 290  crop_swap2 : crop bbox then swap the two present colours
# ===========================================================================
def _ref_290(a):
    nz = np.argwhere(a > 0)
    if len(nz) == 0:
        return None
    r0, c0 = nz.min(0)
    r1, c1 = nz.max(0)
    sub = a[r0:r1 + 1, c0:c1 + 1]
    cols = sorted(set(sub[sub > 0].tolist()))
    if len(cols) != 2:
        return None
    x, y = cols
    out = sub.copy()
    out[sub == x] = y
    out[sub == y] = x
    return out


def _build_290():
    g = _G()
    rmin1, cmin1, h, w = _bbox_int(g)
    grid, mhw = _crop_gather(g, "input", rmin1, cmin1, h, w, zoom=1)
    cropped = g.node("Mul", [grid, mhw])                      # [1,10,30,30]

    # build swap permutation P = I - 2 diag(p) + p p^T  over present colours
    pres = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    p = g.node("Mul", [pres, _chan_mask(g, set(range(1, CHANNELS)))])  # [1,10,1,1]
    pcol = g.node("Reshape", [p, g.iconst([CHANNELS, 1])])   # [10,1]
    prow = g.node("Reshape", [p, g.iconst([1, CHANNELS])])   # [1,10]
    ppt = g.node("MatMul", [pcol, prow])                     # [10,10]
    eye = _ftensor(g, np.eye(CHANNELS))
    diagp = g.node("Mul", [ppt, eye])                        # diag(p) (= p on diagonal)
    P = g.node("Sub", [g.node("Add", [eye, ppt]),
                       g.node("Mul", [diagp, g.scalar(2.0)])])   # [10,10]
    Pw = g.node("Reshape", [P, g.iconst([CHANNELS, CHANNELS, 1, 1])])
    g.node("Conv", [cropped, Pw], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 225  t225 : 2x2 block radiates diagonally-opposite corner colours outward
# ===========================================================================
def _ref_225(a):
    nz = np.argwhere(a > 0)
    if len(nz) == 0:
        return None
    r0, c0 = nz.min(0)
    r1, c1 = nz.max(0)
    if (r1 - r0, c1 - c0) != (1, 1):
        return None
    H, W = a.shape
    o = a.copy()
    ctl, ctr, cbl, cbr = a[r0, c0], a[r0, c1], a[r1, c0], a[r1, c1]

    def fill(rs, re, cs, ce, col):
        for i in range(rs, re + 1):
            for j in range(cs, ce + 1):
                if 0 <= i < H and 0 <= j < W:
                    o[i, j] = col
    fill(r0 - 2, r0 - 1, c0 - 2, c0 - 1, cbr)
    fill(r0 - 2, r0 - 1, c1 + 1, c1 + 2, cbl)
    fill(r1 + 1, r1 + 2, c0 - 2, c0 - 1, ctr)
    fill(r1 + 1, r1 + 2, c1 + 1, c1 + 2, ctl)
    return o


def _build_225():
    g = _G()
    rmin, _, cmin, _ = _bbox(g)
    real = _realcell(g)
    ridx = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])
    cidx = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])

    # corner colours via Gather of the 2x2 block
    rmin_i = g.node("Cast", [g.node("Reshape", [rmin, g.iconst([1])])], to=INT64)
    cmin_i = g.node("Cast", [g.node("Reshape", [cmin, g.iconst([1])])], to=INT64)
    ridx2 = g.node("Add", [rmin_i, g.iconst([0, 1])])        # [2]
    cidx2 = g.node("Add", [cmin_i, g.iconst([0, 1])])        # [2]
    rows = g.node("Gather", ["input", ridx2], axis=2)        # [1,10,2,30]
    blk = g.node("Gather", [rows, cidx2], axis=3)            # [1,10,2,2]
    ctl = _slice(g, blk, [0, 0], [1, 1], [2, 3])             # [1,10,1,1]
    ctr = _slice(g, blk, [0, 1], [1, 2], [2, 3])
    cbl = _slice(g, blk, [1, 0], [2, 1], [2, 3])
    cbr = _slice(g, blk, [1, 1], [2, 2], [2, 3])

    dr = g.node("Sub", [ridx, rmin])                         # [1,1,30,1]
    dc = g.node("Sub", [cidx, cmin])                         # [1,1,1,30]

    def band(x, lo, hi):
        a1 = g.node("Greater", [x, g.scalar(lo - 0.5)])
        a2 = g.node("Less", [x, g.scalar(hi + 0.5)])
        return g.node("Mul", [g.node("Cast", [a1], to=DATA_TYPE),
                              g.node("Cast", [a2], to=DATA_TYPE)])
    drT = band(dr, -2, -1)
    drB = band(dr, 2, 3)
    dcL = band(dc, -2, -1)
    dcR = band(dc, 2, 3)

    def quad(drm, dcm):
        m = g.node("Mul", [g.node("Mul", [drm, dcm]), real])
        return g.node("Greater", [m, g.scalar(0.5)])
    tl = quad(drT, dcL)
    tr = quad(drT, dcR)
    bl = quad(drB, dcL)
    br = quad(drB, dcR)

    cur = g.node("Where", [tl, cbr, "input"])
    cur = g.node("Where", [tr, cbl, cur])
    cur = g.node("Where", [bl, ctr, cur])
    g.node("Where", [br, ctl, cur], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 24  lines_v2_h : colour-2 -> vertical line in its column, others -> horizontal
# ===========================================================================
def _ref_24(a):
    H, W = a.shape
    o = np.zeros_like(a)
    for (r, c) in np.argwhere(a == 2):
        o[:, c] = 2
    for (r, c) in np.argwhere((a > 0) & (a != 2)):
        o[r, :] = a[r, c]
    return o


def _build_24():
    g = _G()
    real = _realcell(g)
    ch2 = _slice(g, "input", [2], [3], [1])                  # [1,1,30,30]
    vcolmask = g.node("ReduceMax", [ch2], axes=[2], keepdims=1)  # [1,1,1,30]
    vcolfull = g.node("Mul", [vcolmask, real])               # [1,1,30,30]
    vcolb = g.node("Greater", [vcolfull, g.scalar(0.5)])

    rowcolors = g.node("ReduceMax", ["input"], axes=[3], keepdims=1)  # [1,10,30,1]
    hmask = _chan_mask(g, set(range(1, CHANNELS)) - {2})
    hrow = g.node("Mul", [rowcolors, hmask])                 # [1,10,30,1]
    hrowhas = g.node("ReduceMax", [hrow], axes=[1], keepdims=1)  # [1,1,30,1]
    hrowfull = g.node("Mul", [hrowhas, real])                # [1,1,30,30]
    hrowb = g.node("Greater", [hrowfull, g.scalar(0.5)])

    inner = g.node("Where", [vcolb, _onehot_channel(g, 2), "input"])
    g.node("Where", [hrowb, hrow, inner], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 342  crk2_5_quad : 4 quadrant markers -> central 2x2 colour-8 block corners
# ===========================================================================
def _ref_342(a):
    blk = np.argwhere(a == 8)
    if len(blk) == 0:
        return None
    r0, c0 = blk.min(0)
    r1, c1 = blk.max(0)
    if (r1 - r0, c1 - c0) != (1, 1):
        return None
    o = np.zeros_like(a)
    mk = np.argwhere((a > 0) & (a != 8))

    def find(cond):
        for (r, c) in mk:
            if cond(r, c):
                return a[r, c]
        return 0
    o[r0, c0] = find(lambda r, c: r < r0 and c < c0)
    o[r0, c1] = find(lambda r, c: r < r0 and c > c1)
    o[r1, c0] = find(lambda r, c: r > r1 and c < c0)
    o[r1, c1] = find(lambda r, c: r > r1 and c > c1)
    return o


def _build_342():
    g = _G()
    real = _realcell(g)
    ch8 = _slice(g, "input", [8], [9], [1])                  # [1,1,30,30]
    rowhas = g.node("ReduceMax", [ch8], axes=[3], keepdims=1)
    colhas = g.node("ReduceMax", [ch8], axes=[2], keepdims=1)
    ridx = g.fconst(list(range(HEIGHT)), [1, 1, HEIGHT, 1])
    cidx = g.fconst(list(range(WIDTH)), [1, 1, 1, WIDTH])
    r0 = _argmax_f(g, rowhas, 2)
    c0 = _argmax_f(g, colhas, 3)
    r1 = g.node("Add", [r0, g.scalar(1.0)])
    c1 = g.node("Add", [c0, g.scalar(1.0)])

    # markers (non-bg, non-8) one-hot
    markers = g.node("Mul", ["input", _chan_mask(g, set(range(1, CHANNELS)) - {8})])

    # half-plane selectors
    rsel_top = g.node("Cast", [g.node("Less", [ridx, r0])], to=DATA_TYPE)        # i<r0 [1,1,30,1]
    rsel_bot = g.node("Cast", [g.node("Greater", [ridx, r1])], to=DATA_TYPE)     # i>r1
    csel_lft = g.node("Cast", [g.node("Less", [cidx, c0])], to=DATA_TYPE)        # j<c0 [1,1,1,30]
    csel_rgt = g.node("Cast", [g.node("Greater", [cidx, c1])], to=DATA_TYPE)     # j>c1

    # column-reduced markers for left / right half-planes -> [1,10,30,1]
    cl = g.node("Reshape", [csel_lft, g.iconst([1, 1, WIDTH, 1])])  # [1,1,30,1]
    cr = g.node("Reshape", [csel_rgt, g.iconst([1, 1, WIDTH, 1])])
    Mleft = g.node("MatMul", [markers, cl])                 # [1,10,30,1]
    Mright = g.node("MatMul", [markers, cr])                # [1,10,30,1]
    rt = g.node("Reshape", [rsel_top, g.iconst([1, 1, 1, HEIGHT])])  # [1,1,1,30]
    rb = g.node("Reshape", [rsel_bot, g.iconst([1, 1, 1, HEIGHT])])

    def col_at(rowsel, Mside):
        # rowsel [1,1,1,30] @ Mside [1,10,30,1] -> [1,10,1,1]
        return g.node("MatMul", [rowsel, Mside])
    cTL = col_at(rt, Mleft)
    cTR = col_at(rt, Mright)
    cBL = col_at(rb, Mleft)
    cBR = col_at(rb, Mright)

    # point masks at the four block corners
    def pmask(rv, cv):
        pr = g.node("Cast", [g.node("Less",
              [g.node("Abs", [g.node("Sub", [ridx, rv])]), g.scalar(0.5)])], to=DATA_TYPE)
        pc = g.node("Cast", [g.node("Less",
              [g.node("Abs", [g.node("Sub", [cidx, cv])]), g.scalar(0.5)])], to=DATA_TYPE)
        return g.node("Greater", [g.node("Mul", [pr, pc]), g.scalar(0.5)])
    pTL = pmask(r0, c0)
    pTR = pmask(r0, c1)
    pBL = pmask(r1, c0)
    pBR = pmask(r1, c1)

    bgbase = g.node("Mul", [real, _onehot_channel(g, 0)])    # [1,10,30,30] real bg
    cur = g.node("Where", [pTL, cTL, bgbase])
    cur = g.node("Where", [pTR, cTR, cur])
    cur = g.node("Where", [pBL, cBL, cur])
    g.node("Where", [pBR, cBR, cur], "output")
    return _model(g.nodes, g.inits)


# ===========================================================================
# 56  swatchmap_1x1_ncomp : 3x3 -> 1x1 colour = LUT(#4-conn foreground comps)
# ===========================================================================
_LUT56 = {1: 6, 2: 3, 3: 1, 5: 2}


def _n4(f):
    nfg = f.sum()
    eh = (f[:, :-1] * f[:, 1:]).sum()
    ev = (f[:-1, :] * f[1:, :]).sum()
    blk = (f[:-1, :-1] * f[:-1, 1:] * f[1:, :-1] * f[1:, 1:]).sum()
    return int(nfg - eh - ev + blk)


def _ref_56(a):
    if a.shape != (3, 3):
        return None
    col = _LUT56.get(_n4((a > 0).astype(int)))
    if col is None:
        return None
    return np.array([[col]])


def _build_56():
    g = _G()
    ch3 = _slice(g, "input", [1, 0, 0], [CHANNELS, 3, 3], [1, 2, 3])  # [1,9,3,3]
    f3 = g.node("ReduceMax", [ch3], axes=[1], keepdims=1)    # [1,1,3,3]
    nfg = g.node("ReduceSum", [f3], axes=[2, 3], keepdims=1)  # [1,1,1,1]
    c0 = _slice(g, f3, [0], [2], [3])
    c1 = _slice(g, f3, [1], [3], [3])
    eh = g.node("ReduceSum", [g.node("Mul", [c0, c1])], axes=[2, 3], keepdims=1)
    r0 = _slice(g, f3, [0], [2], [2])
    r1 = _slice(g, f3, [1], [3], [2])
    ev = g.node("ReduceSum", [g.node("Mul", [r0, r1])], axes=[2, 3], keepdims=1)
    a00 = _slice(g, f3, [0, 0], [2, 2], [2, 3])
    a01 = _slice(g, f3, [0, 1], [2, 3], [2, 3])
    a10 = _slice(g, f3, [1, 0], [3, 2], [2, 3])
    a11 = _slice(g, f3, [1, 1], [3, 3], [2, 3])
    blk = g.node("ReduceSum", [g.node("Mul", [g.node("Mul", [a00, a01]),
                                              g.node("Mul", [a10, a11])])],
                 axes=[2, 3], keepdims=1)
    n4 = g.node("Add", [g.node("Sub", [g.node("Sub", [nfg, eh]), ev]), blk])  # [1,1,1,1]

    def sel(val):
        return g.node("Cast", [g.node("Less",
              [g.node("Abs", [g.node("Sub", [n4, g.scalar(float(val))])]),
               g.scalar(0.5)])], to=DATA_TYPE)
    colorvec = None
    for cnt, col in _LUT56.items():
        term = g.node("Mul", [sel(cnt), _onehot_channel(g, col)])  # [1,10,1,1]
        colorvec = term if colorvec is None else g.node("Add", [colorvec, term])
    g.node("Pad", [colorvec], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - 1, WIDTH - 1])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
_FAMILY = [
    ("ncomp", _ref_56, _build_56),
    ("upcnt", _ref_289, _build_289),
    ("spray4", _ref_199, _build_199),
    ("dyncrop", _ref_31, _build_31),
    ("cropswap", _ref_290, _build_290),
    ("bboxs2", _ref_384, _build_384),
    ("radiate", _ref_225, _build_225),
    ("linesv2h", _ref_24, _build_24),
    ("quadblk", _ref_342, _build_342),
]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    for name, ref, build in _FAMILY:
        if _exact(ref, prs):
            try:
                out.append((name, build()))
            except Exception:
                pass
    return out
