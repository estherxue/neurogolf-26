"""family_crk2_3 — a slice of unsolved NeuroGolf 2026 ARC tasks.

Each rule is detected STRUCTURALLY (a numpy mirror that reproduces every provided
train/test/arc-gen pair exactly) before the matching opset-10 ONNX graph is
emitted.  The ONNX semantics mirror the numpy reference one-for-one.

Rules implemented
-----------------
  crop_swap2   crop to the bounding box of all non-bg cells, then SWAP the two
               distinct non-bg colours present (out = (A+B) - in, per cell).
  square_hw    keep only the first  W/3  columns (the top-left HxH block of an
               Hx(3H) grid); rows untouched.  output[:, j] = in[:, j] for j<W/3.
  midcol       keep only the centre column  floor(size/2)  of a square grid.
  rowswapck    2-row grids: at odd columns swap the two rows (column-parity
               checkerboard of the two row colours).
  linefill     each row holding two endpoints (col 0 & col W-1) is filled: left
               half = left colour, centre col = 5, right half = right colour.
  cross        a vertical segment (colour V) and a horizontal segment (colour H)
               are extended to a full cross; their crossing cell becomes 4.
  rectfill     two solid rectangles: recolour the interior (eroded) of the larger
               one to 2 and of the smaller one to 1.
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


def _nbg_mask(g, src="input"):
    """[1,1,30,30] = 1 on non-background cells, 0 on bg + padding."""
    realmask = g.nd("ReduceSum", [src], axes=[1], keepdims=1)
    ch0 = g.nd("Slice", [src, g.i64([0]), g.i64([1]), g.i64([1])])
    return g.nd("Sub", [realmask, ch0])


# ========================================================================== #
# Task 290 : crop bbox of non-bg  +  swap the two non-bg colours             #
# ========================================================================== #
def build_crop_swap2():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    chanidx = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))

    # --- per-channel swap: out[:,o] = in[:, src(o)], src present->S-o else o ---
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    present = g.nd("Cast", [g.nd("Greater", [counts, half])], to=F)
    presentNB = g.nd("Mul", [present, nbg])
    S = g.nd("ReduceSum", [g.nd("Mul", [presentNB, chanidx])], axes=[1], keepdims=1)  # A+B
    twoc = g.nd("Add", [chanidx, chanidx])
    term = g.nd("Mul", [presentNB, g.nd("Sub", [S, twoc])])
    src = g.nd("Add", [chanidx, term])                                      # [1,10,1,1]
    src2d = g.nd("Reshape", [src, g.i64([CHANNELS, 1])])                    # [10,1]
    iidx = g.f([1, CHANNELS], list(range(CHANNELS)))                        # [1,10]
    M = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [src2d, iidx])]), half])], to=F)  # [10,10]
    flat = g.nd("Reshape", ["input", g.i64([CHANNELS, H * W])])             # [10,900]
    swflat = g.nd("MatMul", [M, flat])                                      # [10,900]
    swapped = g.nd("Reshape", [swflat, g.i64([1, CHANNELS, H, W])])         # [1,10,30,30]

    # --- crop to bbox of non-bg ---
    content = _nbg_mask(g, "input")                                         # [1,1,30,30]
    rowhas = g.nd("ReduceMax", [content], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [content], axes=[2], keepdims=1)
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    bbox_h = g.nd("Add", [g.nd("Sub", [maxrow, minrow]), one])
    bbox_w = g.nd("Add", [g.nd("Sub", [maxcol, mincol]), one])

    diff_c = g.nd("Sub", [g.nd("Add", [colidx, mincol]), rowidx])
    match_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_c]), half])], to=F)
    trunc_c = g.nd("Cast", [g.nd("Less", [colidx, bbox_w])], to=F)
    Scol = g.nd("Mul", [match_c, trunc_c])
    diff_r = g.nd("Sub", [colidx, g.nd("Add", [rowidx, minrow])])
    match_r = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_r]), half])], to=F)
    trunc_r = g.nd("Cast", [g.nd("Less", [rowidx, bbox_h])], to=F)
    Srow = g.nd("Mul", [match_r, trunc_r])

    shift1 = g.nd("MatMul", [swapped, Scol])
    g.nd("MatMul", [Srow, shift1], "output")
    return _model(g)


# ========================================================================== #
# Task 67 : keep first W/3 columns (top-left HxH of an Hx3H grid)            #
# ========================================================================== #
def build_square_hw():
    g = _G()
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    third = g.f([1, 1, 1, 1], [1.0 / 3.0])
    # use the FULL real region (bg cells count) so grid width is robust
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)            # [1,1,30,30]
    colhas = g.nd("ReduceMax", [realmask], axes=[2], keepdims=1)             # [1,1,1,30]
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    Wd = g.nd("Add", [maxcol, one])
    p = g.nd("Cast", [g.nd("Cast", [g.nd("Add", [g.nd("Mul", [Wd, third]), half])], to=INT64)], to=F)
    colmask = g.nd("Cast", [g.nd("Less", [colidx, p])], to=F)                # [1,1,1,30]
    g.nd("Mul", ["input", colmask], "output")
    return _model(g)


# ========================================================================== #
# Task 329 : keep the centre column floor(size/2) of a square grid          #
# ========================================================================== #
def build_midcol():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)            # [1,1,30,30]
    rowhas = g.nd("ReduceMax", [realmask], axes=[3], keepdims=1)             # [1,1,30,1]
    colhas = g.nd("ReduceMax", [realmask], axes=[2], keepdims=1)             # [1,1,1,30]
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    size = g.nd("Add", [g.nd("Max", [maxrow, maxcol]), one])                 # [1,1,1,1]
    mid = g.nd("Cast", [g.nd("Cast", [g.nd("Mul", [size, half])], to=INT64)], to=F)
    colmask = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, mid])]), half])], to=F)
    kept = g.nd("Mul", ["input", colmask])                                  # [1,10,30,30] mid column
    other_real = g.nd("Mul", [realmask, g.nd("Sub", [one, colmask])])       # [1,1,30,30]
    bg = g.nd("Mul", [oh0, other_real])                                     # [1,10,30,30] channel0
    g.nd("Add", [kept, bg], "output")
    return _model(g)


# ========================================================================== #
# Task 373 : 2-row column-parity row swap                                    #
# ========================================================================== #
def build_rowswapck():
    g = _G()
    idx = list(range(H))
    idx[0], idx[1] = 1, 0
    swapped = g.nd("Gather", ["input", g.i64(idx)], axis=2)                  # swap rows 0,1
    evenvals = [1.0 if (j % 2 == 0) else 0.0 for j in range(W)]
    oddvals = [0.0 if (j % 2 == 0) else 1.0 for j in range(W)]
    even = g.f([1, 1, 1, W], evenvals)
    odd = g.f([1, 1, 1, W], oddvals)
    t1 = g.nd("Mul", ["input", even])
    t2 = g.nd("Mul", [swapped, odd])
    g.nd("Add", [t1, t2], "output")
    return _model(g)


# ========================================================================== #
# Task 60 : per-row line fill (endpoints at col 0 and col W-1, centre = 5)   #
# size-constant grid; ncols given                                            #
# ========================================================================== #
def build_linefill(ncols):
    g = _G()
    mid = (0 + (ncols - 1)) // 2
    col0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([3])])     # [1,10,30,1]
    colR = g.nd("Slice", ["input", g.i64([ncols - 1]), g.i64([ncols]), g.i64([3])])
    # broadcasts across width
    left_full = g.nd("Tile", [col0, g.i64([1, 1, 1, W])])                   # [1,10,30,30]
    right_full = g.nd("Tile", [colR, g.i64([1, 1, 1, W])])
    mask_left = g.f([1, 1, 1, W], [1.0 if j < mid else 0.0 for j in range(W)])
    mask_right = g.f([1, 1, 1, W], [1.0 if (mid < j < ncols) else 0.0 for j in range(W)])
    mask_mid = g.f([1, 1, 1, W], [1.0 if j == mid else 0.0 for j in range(W)])
    leftpart = g.nd("Mul", [left_full, mask_left])
    rightpart = g.nd("Mul", [right_full, mask_right])
    # middle column one-hot per row
    realrow = g.nd("ReduceSum", [col0], axes=[1], keepdims=1)               # [1,1,30,1]
    ch0col = g.nd("Slice", [col0, g.i64([0]), g.i64([1]), g.i64([1])])      # [1,1,30,1]
    contentrow = g.nd("Sub", [realrow, ch0col])                            # [1,1,30,1]
    oh5 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 5 else 0.0 for c in range(CHANNELS)])
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])
    midcol_oh = g.nd("Add", [g.nd("Mul", [contentrow, oh5]),
                             g.nd("Mul", [ch0col, oh0])])                   # [1,10,30,1]
    mid_full = g.nd("Tile", [midcol_oh, g.i64([1, 1, 1, W])])
    midpart = g.nd("Mul", [mid_full, mask_mid])
    g.nd("Add", [g.nd("Add", [leftpart, rightpart]), midpart], "output")
    return _model(g)


# ========================================================================== #
# Task 299 : extend vertical + horizontal segments to a cross (size const)   #
# ========================================================================== #
def build_cross(nrows, ncols):
    g = _G()
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    onehalf = g.f([1, 1, 1, 1], [1.5])
    nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    realrows = g.f([1, 1, H, 1], [1.0 if r < nrows else 0.0 for r in range(H)])
    realcols = g.f([1, 1, 1, W], [1.0 if c < ncols else 0.0 for c in range(W)])

    content = _nbg_mask(g, "input")                                         # [1,1,30,30]
    colcount = g.nd("ReduceSum", [content], axes=[2], keepdims=1)           # [1,1,1,30]
    rowcount = g.nd("ReduceSum", [content], axes=[3], keepdims=1)           # [1,1,30,1]
    vcol = g.nd("Cast", [g.nd("Greater", [colcount, onehalf])], to=F)       # [1,1,1,30] vertical col
    hrow = g.nd("Cast", [g.nd("Greater", [rowcount, onehalf])], to=F)       # [1,1,30,1] horiz row

    # vertical colour (one-hot over channel)
    vcells = g.nd("Mul", ["input", vcol])                                   # [1,10,30,30]
    Vchan = g.nd("Mul", [g.nd("ReduceMax", [vcells], axes=[2, 3], keepdims=1), nbg])  # [1,10,1,1]
    hcells = g.nd("Mul", ["input", hrow])
    Hchan = g.nd("Mul", [g.nd("ReduceMax", [hcells], axes=[2, 3], keepdims=1), nbg])

    oh4 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 4 else 0.0 for c in range(CHANNELS)])
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])

    vlinemask = g.nd("Mul", [vcol, realrows])                               # [1,1,30,30]
    hlinemask = g.nd("Mul", [hrow, realcols])                               # [1,1,30,30]
    intersect = g.nd("Mul", [vcol, hrow])                                   # [1,1,30,30] (rh,cv)
    noni = g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), intersect])

    vfill = g.nd("Mul", [g.nd("Mul", [Vchan, vlinemask]), noni])            # [1,10,30,30]
    hfill = g.nd("Mul", [g.nd("Mul", [Hchan, hlinemask]), noni])
    ifill = g.nd("Mul", [oh4, intersect])
    # background on real non-cross cells
    realmask = g.nd("Mul", [realrows, realcols])                            # [1,1,30,30]
    crossmask = g.nd("Max", [vlinemask, hlinemask])
    bgmask = g.nd("Sub", [realmask, crossmask])
    bgfill = g.nd("Mul", [oh0, bgmask])
    g.nd("Add", [g.nd("Add", [vfill, hfill]), g.nd("Add", [ifill, bgfill])], "output")
    return _model(g)


# ========================================================================== #
# Task 156 : recolour interiors of two solid rects (larger->2, smaller->1)    #
# size-constant grid nr x nc                                                  #
# ========================================================================== #
def build_rectfill(nr, nc):
    g = _G()
    N = nr * nc
    T = nr + nc + 2
    x = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([nr, nc]), g.i64([2, 3])])  # [1,10,nr,nc]
    realmask = g.nd("ReduceSum", [x], axes=[1], keepdims=1)                  # [1,1,nr,nc]
    ch0 = g.nd("Slice", [x, g.i64([0]), g.i64([1]), g.i64([1])])            # [1,1,nr,nc]
    Mnb = g.nd("Sub", [realmask, ch0])                                      # [1,1,nr,nc] non-bg
    # interior via 3x3 all-nonbg count
    ones3 = g.f([1, 1, 3, 3], [1.0] * 9)
    cnt = g.nd("Conv", [Mnb, ones3], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    interior = g.nd("Cast", [g.nd("Greater", [cnt, g.f([1, 1, 1, 1], [8.5])])], to=F)

    def shift(t, dr, dc):
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        p = g.nd("Pad", [t], mode="constant", value=0.0,
                 pads=[0, 0, pt, pl, 0, 0, pb, pr])
        st = g.i64([max(-dr, 0), max(-dc, 0)])
        en = g.i64([max(-dr, 0) + nr, max(-dc, 0) + nc])
        return g.nd("Slice", [p, st, en, g.i64([2, 3])])

    P = g.f([1, 1, nr, nc], np.arange(1, N + 1))
    L = g.nd("Mul", [Mnb, P])
    for _ in range(T):
        nbrs = [shift(L, dr, dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        L = g.nd("Mul", [g.nd("Max", [L] + nbrs), Mnb])
    Lcol = g.nd("Reshape", [L, g.i64([N, 1])])
    Lrow = g.nd("Reshape", [L, g.i64([1, N])])
    E = g.nd("Equal", [g.nd("Cast", [Lcol], to=INT32), g.nd("Cast", [Lrow], to=INT32)])
    Ef = g.nd("Cast", [E], to=F)                                            # [N,N]
    size_col = g.nd("ReduceSum", [Ef], axes=[1], keepdims=1)                # [N,1]
    size2d = g.nd("Reshape", [size_col, g.i64([1, 1, nr, nc])])
    size2d = g.nd("Mul", [size2d, Mnb])
    maxsize = g.nd("ReduceMax", [size2d], axes=[2, 3], keepdims=1)
    isbig = g.nd("Cast", [g.nd("Greater", [size2d, g.nd("Sub", [maxsize, g.f([1, 1, 1, 1], [0.5])])])], to=F)
    mask2 = g.nd("Mul", [interior, isbig])
    mask1 = g.nd("Mul", [interior, g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), isbig])])
    oh1 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 1 else 0.0 for c in range(CHANNELS)])
    oh2 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 2 else 0.0 for c in range(CHANNELS)])
    keep = g.nd("Mul", [x, g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), interior])])
    cropped = g.nd("Add", [g.nd("Add", [keep, g.nd("Mul", [oh2, mask2])]),
                           g.nd("Mul", [oh1, mask1])])                      # [1,10,nr,nc]
    g.nd("Pad", [cropped], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, H - nr, W - nc])
    return _model(g)


# ========================================================================== #
# Task 20 : C4 rotational symmetrisation about the content centre             #
# ========================================================================== #
def build_symrot():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])

    content = _nbg_mask(g, "input")
    rowhas = g.nd("ReduceMax", [content], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [content], axes=[2], keepdims=1)
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    R = g.nd("Add", [minrow, maxrow])
    C = g.nd("Add", [mincol, maxcol])
    hs = g.nd("Mul", [g.nd("Add", [R, C]), half])
    hd = g.nd("Mul", [g.nd("Sub", [R, C]), half])

    rc = g.nd("Add", [rowidx, colidx])                                      # [1,1,30,30]
    rmc = g.nd("Sub", [rowidx, colidx])
    AR = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rc, R])]), half])], to=F)
    AC = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rc, C])]), half])], to=F)
    P = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rc, hs])]), half])], to=F)
    Q = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rmc, hd])]), half])], to=F)

    T = g.nd("Transpose", ["input"], perm=[0, 1, 3, 2])
    rot180 = g.nd("MatMul", [AR, g.nd("MatMul", ["input", AC])])
    rot90 = g.nd("MatMul", [P, g.nd("MatMul", [T, Q])])
    rot270 = g.nd("MatMul", [Q, g.nd("MatMul", [T, P])])
    allmax = g.nd("Max", ["input", rot90, rot180, rot270])
    contentmax = g.nd("Mul", [allmax, nbg])
    hascontent = g.nd("ReduceSum", [contentmax], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    bgmask = g.nd("Mul", [realmask, g.nd("Sub", [one, hascontent])])
    g.nd("Add", [contentmax, g.nd("Mul", [oh0, bgmask])], "output")
    return _model(g)


# ========================================================================== #
# Task 273 : fill bg cells lying strictly inside a marker rectangle with 2    #
# (cell qualifies iff every strict diagonal quadrant contains a marker)       #
# ========================================================================== #
def build_cornerfill():
    g = _G()
    L = np.array([[1.0 if k < i else 0.0 for k in range(H)] for i in range(H)], np.float32)
    U = L.T
    Lc = g.f([H, H], L)
    Uc = g.f([H, H], U)
    half = g.f([1, 1, 1, 1], [0.5])
    diff20 = g.f([1, CHANNELS, 1, 1],
                 [(-1.0 if c == 0 else (1.0 if c == 2 else 0.0)) for c in range(CHANNELS)])
    M = _nbg_mask(g, "input")                                               # [1,1,30,30]
    above = g.nd("MatMul", [Lc, M])
    below = g.nd("MatMul", [Uc, M])
    NW = g.nd("MatMul", [above, Uc])
    NE = g.nd("MatMul", [above, Lc])
    SW = g.nd("MatMul", [below, Uc])
    SE = g.nd("MatMul", [below, Lc])

    def pos(t):
        return g.nd("Cast", [g.nd("Greater", [t, half])], to=F)
    quad = g.nd("Mul", [g.nd("Mul", [pos(NW), pos(NE)]), g.nd("Mul", [pos(SW), pos(SE)])])
    bg = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])       # [1,1,30,30]
    filled = g.nd("Mul", [quad, bg])                                        # [1,1,30,30]
    g.nd("Add", ["input", g.nd("Mul", [diff20, filled])], "output")
    return _model(g)


# ========================================================================== #
# Task 237 : each marker -> ray right to the last column, then fill down       #
# the last column (forward-fill) with the marker colour                        #
# ========================================================================== #
def build_staircase():
    g = _G()
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])
    TT = np.array([[1.0 if k <= c else 0.0 for c in range(W)] for k in range(W)], np.float32)
    TTc = g.f([W, W], TT)

    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)            # [1,1,30,30]
    colhas = g.nd("ReduceMax", [realmask], axes=[2], keepdims=1)             # [1,1,1,30]
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    lastcol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, maxcol])]), half])], to=F)
    notlast = g.nd("Cast", [g.nd("Less", [colidx, maxcol])], to=F)

    inNB = g.nd("Mul", ["input", nbg])                                      # channels 1-9
    cumright = g.nd("MatMul", [inNB, TTc])                                  # [1,10,30,30]
    HF = g.nd("Cast", [g.nd("Greater", [cumright, half])], to=F)            # fill-right mask
    nonlast = g.nd("Mul", [HF, notlast])
    lastseed = g.nd("Mul", [HF, lastcol])
    seed1d = g.nd("ReduceSum", [lastseed], axes=[3], keepdims=1)            # [1,10,30,1]

    filled = seed1d
    for _ in range(H):
        p = g.nd("Pad", [filled], mode="constant", value=0.0, pads=[0, 0, 1, 0, 0, 0, 0, 0])
        down = g.nd("Slice", [p, g.i64([0]), g.i64([H]), g.i64([2])])       # shift down 1
        has = g.nd("ReduceSum", [filled], axes=[1], keepdims=1)             # [1,1,30,1]
        empty = g.nd("Sub", [one, g.nd("Clip", [has], min=0.0, max=1.0)])
        filled = g.nd("Add", [filled, g.nd("Mul", [down, empty])])
    filled_full = g.nd("Mul", [g.nd("Tile", [filled, g.i64([1, 1, 1, W])]), lastcol])
    content = g.nd("Add", [nonlast, filled_full])                          # channels 1-9
    content = g.nd("Mul", [content, realmask])                             # clip to real region
    hascontent = g.nd("ReduceSum", [content], axes=[1], keepdims=1)
    bg = g.nd("Mul", [realmask, g.nd("Sub", [one, hascontent])])
    g.nd("Add", [content, g.nd("Mul", [oh0, bg])], "output")
    return _model(g)


# ========================================================================== #
# Task 254 : tallest non-bg column -> colour 1, shortest -> colour 2          #
# (bottom-aligned vertical bars; all other bars removed)                      #
# ========================================================================== #
def build_colheight():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    big = g.f([1, 1, 1, 1], [_CBIG])
    one = g.f([1, 1, 1, 1], [1.0])
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])
    oh1 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 1 else 0.0 for c in range(CHANNELS)])
    oh2 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 2 else 0.0 for c in range(CHANNELS)])
    M = _nbg_mask(g, "input")                                              # [1,1,30,30]
    counts = g.nd("ReduceSum", [M], axes=[2], keepdims=1)                   # [1,1,1,30]
    present = g.nd("Cast", [g.nd("Greater", [counts, half])], to=F)
    maxc = g.nd("ReduceMax", [counts], axes=[3], keepdims=1)
    pushed = g.nd("Add", [counts, g.nd("Mul", [g.nd("Sub", [one, present]), big])])
    minc = g.nd("ReduceMin", [pushed], axes=[3], keepdims=1)
    tallcol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [counts, maxc])]), half])], to=F)
    shortcol = g.nd("Mul", [present,
                  g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [counts, minc])]), half])], to=F)])
    ch1 = g.nd("Mul", [M, tallcol])                                        # [1,1,30,30]
    ch2 = g.nd("Mul", [M, shortcol])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    bg = g.nd("Sub", [g.nd("Sub", [realmask, ch1]), ch2])
    g.nd("Add", [g.nd("Add", [g.nd("Mul", [oh1, ch1]), g.nd("Mul", [oh2, ch2])]),
                 g.nd("Mul", [oh0, bg])], "output")
    return _model(g)


# ========================================================================== #
# Task 75 : stamp the top-left 3x3 template centred on every colour-1 marker  #
# ========================================================================== #
def _shift(g, x, dr, dc):
    pads = [0, 0, max(dr, 0), max(dc, 0), 0, 0, max(-dr, 0), max(-dc, 0)]
    p = g.nd("Pad", [x], mode="constant", value=0.0, pads=pads)
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
    return g.nd("Slice", [p, st, en, g.i64([2, 3])])


def build_markerstamp():
    g = _G()
    keepmask = g.f([1, CHANNELS, 1, 1], [0.0 if c == 1 else 1.0 for c in range(CHANNELS)])
    base = g.nd("Mul", ["input", keepmask])                                 # markers removed
    marker = g.nd("Slice", ["input", g.i64([1]), g.i64([2]), g.i64([1])])   # [1,1,30,30] color-1
    stamps = None
    for tr in range(3):
        for tc in range(3):
            tcell = g.nd("Slice", ["input", g.i64([tr, tc]), g.i64([tr + 1, tc + 1]), g.i64([2, 3])])
            sh = _shift(g, marker, tr - 1, tc - 1)                          # [1,1,30,30]
            contrib = g.nd("Mul", [sh, tcell])                             # [1,10,30,30]
            stamps = contrib if stamps is None else g.nd("Add", [stamps, contrib])
    stampmask = g.nd("ReduceSum", [stamps], axes=[1], keepdims=1)           # [1,1,30,30]
    cleaned = g.nd("Mul", [base, g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), stampmask])])
    g.nd("Add", [cleaned, stamps], "output")
    return _model(g)


# ========================================================================== #
# Task 281 : stretch a rectangle (outline O / interior I) toward an isolated   #
# marker; redraw the rectangle over the extended bounding box                  #
# ========================================================================== #
def build_stretchrect():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])
    plus = g.f([1, 1, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1, 0])

    M = _nbg_mask(g, "input")                                              # [1,1,30,30]
    nbrcnt = g.nd("Conv", [M, plus], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    isolated = g.nd("Cast", [g.nd("Less", [nbrcnt, half])], to=F)
    marker = g.nd("Mul", [M, isolated])                                    # [1,1,30,30] single cell
    rect = g.nd("Sub", [M, marker])                                        # [1,1,30,30]

    mr = g.nd("ReduceMax", [g.nd("Mul", [marker, rowidx])], axes=[2, 3], keepdims=1)
    mc = g.nd("ReduceMax", [g.nd("Mul", [marker, colidx])], axes=[2, 3], keepdims=1)

    rowhas = g.nd("ReduceMax", [rect], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [rect], axes=[2], keepdims=1)
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    r0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    c1 = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    c0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    # corner colour O (at r0,c0) and interior colour I
    cmask = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, r0])]), half])], to=F),
                         g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, c0])]), half])], to=F)])
    O_oh = g.nd("ReduceSum", [g.nd("Mul", ["input", cmask])], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    rectcols = g.nd("Mul", [g.nd("ReduceMax", [g.nd("Mul", ["input", rect])], axes=[2, 3], keepdims=1), nbg])
    I_oh = g.nd("Sub", [rectcols, O_oh])                                   # [1,10,1,1]

    nr0 = g.nd("Min", [r0, mr])
    nr1 = g.nd("Max", [r1, mr])
    nc0 = g.nd("Min", [c0, mc])
    nc1 = g.nd("Max", [c1, mc])

    inrows = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [rowidx, g.nd("Sub", [nr0, half])])], to=F),
                          g.nd("Cast", [g.nd("Less", [rowidx, g.nd("Add", [nr1, half])])], to=F)])
    incols = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [colidx, g.nd("Sub", [nc0, half])])], to=F),
                          g.nd("Cast", [g.nd("Less", [colidx, g.nd("Add", [nc1, half])])], to=F)])
    inside = g.nd("Mul", [inrows, incols])                                 # [1,1,30,30]
    onr = g.nd("Add", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, nr0])]), half])], to=F),
                       g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, nr1])]), half])], to=F)])
    onc = g.nd("Add", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, nc0])]), half])], to=F),
                       g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, nc1])]), half])], to=F)])
    onborder = g.nd("Clip", [g.nd("Add", [onr, onc])], min=0.0, max=1.0)   # [1,1,30,30]
    border = g.nd("Mul", [inside, onborder])
    interior = g.nd("Mul", [inside, g.nd("Sub", [one, onborder])])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    bg = g.nd("Sub", [realmask, inside])
    g.nd("Add", [g.nd("Add", [g.nd("Mul", [O_oh, border]), g.nd("Mul", [I_oh, interior])]),
                 g.nd("Mul", [oh0, bg])], "output")
    return _model(g)


# ========================================================================== #
# Task 85 : stripe the middle row of every 3-tall solid rectangle             #
# (keep cells at even offset from the rect's left edge, others -> bg)          #
# ========================================================================== #
def build_stripemid():
    g = _G()
    colidx = g.f([1, 1, 1, W], list(range(W)))
    colpar = g.f([1, 1, 1, W], [float(c % 2) for c in range(W)])
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    big = g.f([1, 1, 1, 1], [_CBIG])
    nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])

    sd1 = _shift(g, "input", 1, 0)
    su1 = _shift(g, "input", -1, 0)
    sd2 = _shift(g, "input", 2, 0)
    su2 = _shift(g, "input", -2, 0)
    midch = g.nd("Mul", [g.nd("Mul", ["input", sd1]),
                         g.nd("Mul", [su1, g.nd("Mul", [g.nd("Sub", [one, sd2]),
                                                        g.nd("Sub", [one, su2])])])])
    midmask = g.nd("ReduceSum", [g.nd("Mul", [midch, nbg])], axes=[1], keepdims=1)  # [1,1,30,30]

    pushed = g.nd("Add", [colidx, g.nd("Mul", [g.nd("Sub", [one, midmask]), big])])
    leftcol = g.nd("ReduceMin", [pushed], axes=[3], keepdims=1)              # [1,1,30,1]
    fl = g.nd("Cast", [g.nd("Cast", [g.nd("Mul", [leftcol, half])], to=INT64)], to=F)
    lpar = g.nd("Sub", [leftcol, g.nd("Mul", [two, fl])])                    # [1,1,30,1]
    pardiff = g.nd("Cast", [g.nd("Greater", [g.nd("Abs", [g.nd("Sub", [colpar, lpar])]), half])], to=F)
    remove = g.nd("Mul", [midmask, pardiff])                                 # [1,1,30,30]
    keep = g.nd("Mul", ["input", g.nd("Sub", [one, remove])])
    g.nd("Add", [keep, g.nd("Mul", [oh0, remove])], "output")
    return _model(g)


# ========================================================================== #
# Task 35 : markers project perpendicular onto the nearest edge of the rect    #
# (rect = most-frequent colour, solid rectangle)                              #
# ========================================================================== #
def build_project():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))

    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    bgneg = g.f([1, CHANNELS, 1, 1], [-_CBIG * 1000] + [0.0] * (CHANNELS - 1))
    amax = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    gate = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)                 # [1,10,1,1]
    rectmask = g.nd("ReduceSum", [g.nd("Mul", ["input", gate])], axes=[1], keepdims=1)  # [1,1,30,30]
    Mnb = _nbg_mask(g, "input")
    markermask = g.nd("Sub", [Mnb, rectmask])
    marker_oh = g.nd("Mul", ["input", markermask])                         # [1,10,30,30]

    rowhas = g.nd("ReduceMax", [rectmask], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [rectmask], axes=[2], keepdims=1)
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    r0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    c1 = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    c0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    def C(t):
        return g.nd("Cast", [t], to=F)
    lt_r0 = C(g.nd("Less", [rowidx, r0]))
    gt_r1 = C(g.nd("Greater", [rowidx, r1]))
    lt_c0 = C(g.nd("Less", [colidx, c0]))
    gt_c1 = C(g.nd("Greater", [colidx, c1]))
    eq_r0 = C(g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, r0])]), half]))
    eq_r1 = C(g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, r1])]), half]))
    eq_c0 = C(g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, c0])]), half]))
    eq_c1 = C(g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, c1])]), half]))
    in_cols = g.nd("Mul", [C(g.nd("Greater", [colidx, g.nd("Sub", [c0, half])])),
                           C(g.nd("Less", [colidx, g.nd("Add", [c1, half])]))])  # [1,1,1,30]
    in_rows = g.nd("Mul", [C(g.nd("Greater", [rowidx, g.nd("Sub", [r0, half])])),
                           C(g.nd("Less", [rowidx, g.nd("Add", [r1, half])]))])  # [1,1,30,1]

    colmk_top = g.nd("ReduceSum", [g.nd("Mul", [marker_oh, lt_r0])], axes=[2], keepdims=1)  # [1,10,1,30]
    colmk_bot = g.nd("ReduceSum", [g.nd("Mul", [marker_oh, gt_r1])], axes=[2], keepdims=1)
    rowmk_lft = g.nd("ReduceSum", [g.nd("Mul", [marker_oh, lt_c0])], axes=[3], keepdims=1)  # [1,10,30,1]
    rowmk_rgt = g.nd("ReduceSum", [g.nd("Mul", [marker_oh, gt_c1])], axes=[3], keepdims=1)

    top = g.nd("Mul", [g.nd("Mul", [colmk_top, eq_r0]), in_cols])
    bot = g.nd("Mul", [g.nd("Mul", [colmk_bot, eq_r1]), in_cols])
    lft = g.nd("Mul", [g.nd("Mul", [rowmk_lft, eq_c0]), in_rows])
    rgt = g.nd("Mul", [g.nd("Mul", [rowmk_rgt, eq_c1]), in_rows])
    P = g.nd("Add", [g.nd("Add", [top, bot]), g.nd("Add", [lft, rgt])])     # [1,10,30,30]
    projmask = g.nd("ReduceSum", [P], axes=[1], keepdims=1)
    g.nd("Add", [g.nd("Mul", ["input", g.nd("Sub", [one, projmask])]), P], "output")
    return _model(g)


# ========================================================================== #
# Task 228 : move interior corner markers to the opposite EXTERIOR corners     #
# (rot180 of the four corner colours), clearing the interior                   #
# ========================================================================== #
def build_cornerswap():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])

    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    bgneg = g.f([1, CHANNELS, 1, 1], [-_CBIG * 1000] + [0.0] * (CHANNELS - 1))
    amax = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    gate = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)
    rectmask = g.nd("ReduceSum", [g.nd("Mul", ["input", gate])], axes=[1], keepdims=1)
    rectoh = g.nd("Mul", ["input", rectmask])                              # outline kept

    rowhas = g.nd("ReduceMax", [rectmask], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [rectmask], axes=[2], keepdims=1)
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    r0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    c1 = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    c0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    def point(rexpr, cexpr):
        mr = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, rexpr])]), half])], to=F)
        mc = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, cexpr])]), half])], to=F)
        return g.nd("Mul", [mr, mc])

    def colorat(rexpr, cexpr):
        return g.nd("ReduceSum", [g.nd("Mul", ["input", point(rexpr, cexpr)])], axes=[2, 3], keepdims=1)

    r0p = g.nd("Add", [r0, one]); r1m = g.nd("Sub", [r1, one])
    c0p = g.nd("Add", [c0, one]); c1m = g.nd("Sub", [c1, one])
    TL = colorat(r0p, c0p); TR = colorat(r0p, c1m); BL = colorat(r1m, c0p); BR = colorat(r1m, c1m)
    r0m = g.nd("Sub", [r0, one]); r1p = g.nd("Add", [r1, one])
    c0m = g.nd("Sub", [c0, one]); c1p = g.nd("Add", [c1, one])
    extTL = g.nd("Mul", [BR, point(r0m, c0m)])
    extTR = g.nd("Mul", [BL, point(r0m, c1p)])
    extBL = g.nd("Mul", [TR, point(r1p, c0m)])
    extBR = g.nd("Mul", [TL, point(r1p, c1p)])
    content = g.nd("Add", [g.nd("Add", [rectoh, g.nd("Add", [extTL, extTR])]),
                           g.nd("Add", [extBL, extBR])])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    hascontent = g.nd("ReduceSum", [content], axes=[1], keepdims=1)
    bg = g.nd("Mul", [realmask, g.nd("Sub", [one, hascontent])])
    g.nd("Add", [content, g.nd("Mul", [oh0, bg])], "output")
    return _model(g)


# ========================================================================== #
# Task 114 : replicate-pad the grid by 1 on every side (corners -> bg)         #
# ========================================================================== #
def build_edgepad():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    twohalf = g.f([1, 1, 1, 1], [2.5])
    zero = g.f([1, 1, 1, 1], [0.0])
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])

    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    rowhas = g.nd("ReduceMax", [realmask], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [realmask], axes=[2], keepdims=1)
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)  # H-1
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)  # W-1

    # RowDup[k,m] = (m == clamp(k-1,0,maxrow)) & (k <= maxrow+2)
    clr = g.nd("Max", [zero, g.nd("Min", [g.nd("Sub", [rowidx, one]), maxrow])])  # [1,1,30,1]
    rowdup = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, clr])]), half])], to=F),
                          g.nd("Cast", [g.nd("Less", [rowidx, g.nd("Add", [maxrow, twohalf])])], to=F)])
    rows = g.nd("MatMul", [rowdup, "input"])                               # [1,10,30,30]

    clc = g.nd("Max", [zero, g.nd("Min", [g.nd("Sub", [colidx, one]), maxcol])])  # [1,1,1,30]
    coldup = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, clc])]), half])], to=F),
                          g.nd("Cast", [g.nd("Less", [colidx, g.nd("Add", [maxcol, twohalf])])], to=F)])
    res = g.nd("MatMul", [rows, coldup])                                   # [1,10,30,30]

    # corners (0 / maxrow+2) x (0 / maxcol+2) -> background
    redge = g.nd("Add", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [rowidx]), half])], to=F),
                         g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, g.nd("Add", [maxrow, two])])]), half])], to=F)])
    cedge = g.nd("Add", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [colidx]), half])], to=F),
                         g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, g.nd("Add", [maxcol, two])])]), half])], to=F)])
    corner = g.nd("Mul", [redge, cedge])                                   # [1,1,30,30]
    cleaned = g.nd("Mul", [res, g.nd("Sub", [one, corner])])
    g.nd("Add", [cleaned, g.nd("Mul", [oh0, corner])], "output")
    return _model(g)


# ========================================================================== #
# Task 93 : markers gravitate and stack against a bar (most-freq colour),      #
# filling N adjacent cells (N = #markers on that side of that row/col)         #
# ========================================================================== #
def build_barstack():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    oh0 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 0 else 0.0 for c in range(CHANNELS)])

    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    bgneg = g.f([1, CHANNELS, 1, 1], [-_CBIG * 1000] + [0.0] * (CHANNELS - 1))
    amax = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    gate = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)                 # [1,10,1,1]
    barmask = g.nd("ReduceSum", [g.nd("Mul", ["input", gate])], axes=[1], keepdims=1)
    Mnb = _nbg_mask(g, "input")
    marker = g.nd("Sub", [Mnb, barmask])                                    # [1,1,30,30]

    rowhas = g.nd("ReduceMax", [barmask], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [barmask], axes=[2], keepdims=1)
    rb = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    rt = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    cb = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    ct = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    def C(t):
        return g.nd("Cast", [t], to=F)
    gt = lambda a, b: C(g.nd("Greater", [a, g.nd("Add", [b, half])]))
    lt = lambda a, b: C(g.nd("Less", [a, g.nd("Sub", [b, half])]))
    inrows = g.nd("Mul", [C(g.nd("Greater", [rowidx, g.nd("Sub", [rt, half])])),
                          C(g.nd("Less", [rowidx, g.nd("Add", [rb, half])]))])  # [1,1,30,1]
    incols = g.nd("Mul", [C(g.nd("Greater", [colidx, g.nd("Sub", [ct, half])])),
                          C(g.nd("Less", [colidx, g.nd("Add", [cb, half])]))])  # [1,1,1,30]
    c_gt_cb = gt(colidx, cb); c_lt_ct = lt(colidx, ct)
    r_gt_rb = gt(rowidx, rb); r_lt_rt = lt(rowidx, rt)

    rightcount = g.nd("ReduceSum", [g.nd("Mul", [marker, c_gt_cb])], axes=[3], keepdims=1)  # [1,1,30,1]
    leftcount = g.nd("ReduceSum", [g.nd("Mul", [marker, c_lt_ct])], axes=[3], keepdims=1)
    abovecount = g.nd("ReduceSum", [g.nd("Mul", [marker, r_lt_rt])], axes=[2], keepdims=1)  # [1,1,1,30]
    belowcount = g.nd("ReduceSum", [g.nd("Mul", [marker, r_gt_rb])], axes=[2], keepdims=1)

    rightfill = g.nd("Mul", [g.nd("Mul", [c_gt_cb, inrows]),
                  C(g.nd("Less", [g.nd("Sub", [colidx, cb]), g.nd("Add", [rightcount, half])]))])
    leftfill = g.nd("Mul", [g.nd("Mul", [c_lt_ct, inrows]),
                  C(g.nd("Less", [g.nd("Sub", [ct, colidx]), g.nd("Add", [leftcount, half])]))])
    abovefill = g.nd("Mul", [g.nd("Mul", [r_lt_rt, incols]),
                  C(g.nd("Less", [g.nd("Sub", [rt, rowidx]), g.nd("Add", [abovecount, half])]))])
    belowfill = g.nd("Mul", [g.nd("Mul", [r_gt_rb, incols]),
                  C(g.nd("Less", [g.nd("Sub", [rowidx, rb]), g.nd("Add", [belowcount, half])]))])
    fill = g.nd("Max", [g.nd("Max", [rightfill, leftfill]), g.nd("Max", [abovefill, belowfill])])
    Bcells = g.nd("Clip", [g.nd("Add", [barmask, fill])], min=0.0, max=1.0)  # [1,1,30,30]
    content = g.nd("Mul", [gate, Bcells])                                    # [1,10,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    bg = g.nd("Mul", [oh0, g.nd("Sub", [realmask, Bcells])])
    g.nd("Add", [content, bg], "output")
    return _model(g)


# ========================================================================== #
# numpy references                                                           #
# ========================================================================== #
def _bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def ref_crop_swap2(a):
    bb = _bbox(a != 0)
    if bb is None:
        return None
    r0, r1, c0, c1 = bb
    crop = a[r0:r1 + 1, c0:c1 + 1].copy()
    cols = [c for c in range(1, CHANNELS) if (crop == c).any()]
    if len(cols) != 2:
        return None
    s = cols[0] + cols[1]
    out = crop.copy()
    nb = crop != 0
    out[nb] = s - crop[nb]
    return out


def ref_square_hw(a):
    h, w = a.shape
    if w % 3 != 0 or w // 3 != h:
        return None
    cols = [c for c in range(w) if (a[:, c] != 0).any()]
    maxc = max(cols) if cols else -1
    p = int((maxc + 1) * (1.0 / 3.0) + 0.5)
    out = a.copy()
    out[:, p:] = 0
    return out[:, :h]


def ref_midcol(a):
    h, w = a.shape
    rows = [r for r in range(h) if (a[r] != 0).any()]
    cols = [c for c in range(w) if (a[:, c] != 0).any()]
    mr = max(rows) if rows else -1
    mc = max(cols) if cols else -1
    size = max(mr, mc) + 1
    mid = size // 2
    out = np.zeros_like(a)
    if 0 <= mid < w:
        out[:, mid] = a[:, mid]
    return out


def ref_rowswapck(a):
    h, w = a.shape
    if h != 2:
        return None
    out = a.copy()
    for r in range(h):
        for c in range(w):
            if c % 2 == 1:
                out[r, c] = a[1 - r, c]
    return out


def ref_linefill(a):
    h, w = a.shape
    out = np.zeros_like(a)
    for r in range(h):
        cs = [c for c in range(w) if a[r, c] != 0]
        if not cs:
            continue
        if len(cs) != 2 or cs[0] != 0 or cs[1] != w - 1:
            return None
        cl, cr = cs
        colL, colR = a[r, cl], a[r, cr]
        mid = (cl + cr) // 2
        for c in range(cl, cr + 1):
            out[r, c] = colL if c < mid else (5 if c == mid else colR)
    return out


def ref_cross(a):
    h, w = a.shape
    colors = {}
    for c in range(1, CHANNELS):
        ys, xs = np.where(a == c)
        if len(ys):
            colors[c] = (ys, xs)
    vert = horz = None
    for c, (ys, xs) in colors.items():
        if len(set(xs.tolist())) == 1 and len(ys) >= 2:
            vert = (c, int(xs[0]))
        if len(set(ys.tolist())) == 1 and len(xs) >= 2:
            horz = (c, int(ys[0]))
    if vert is None or horz is None:
        return None
    V, cv = vert
    Hc, rh = horz
    out = np.zeros_like(a)
    out[:, cv] = V
    out[rh, :] = Hc
    out[rh, cv] = 4
    return out


def ref_barstack(a):
    h, w = a.shape
    cnt = np.array([int((a == c).sum()) for c in range(CHANNELS)], np.int64)
    cnt[0] = -(10 ** 12)
    B = int(cnt.argmax())
    ys, xs = np.where(a == B)
    if len(ys) == 0:
        return None
    rt, rb, ct, cb = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    mk = (a != 0) & (a != B)
    out = np.zeros_like(a)
    out[a == B] = B
    for r in range(h):
        rc = int(mk[r, cb + 1:].sum())
        for k in range(1, rc + 1):
            if cb + k < w and rt <= r <= rb:
                out[r, cb + k] = B
        lc = int(mk[r, :ct].sum())
        for k in range(1, lc + 1):
            if ct - k >= 0 and rt <= r <= rb:
                out[r, ct - k] = B
    for c in range(w):
        ac = int(mk[:rt, c].sum())
        for k in range(1, ac + 1):
            if rt - k >= 0 and ct <= c <= cb:
                out[rt - k, c] = B
        bc = int(mk[rb + 1:, c].sum())
        for k in range(1, bc + 1):
            if rb + k < h and ct <= c <= cb:
                out[rb + k, c] = B
    return out


def ref_edgepad(a):
    h, w = a.shape
    if h + 2 > 30 or w + 2 > 30:
        return None
    out = np.zeros((h + 2, w + 2), int)
    for r in range(h + 2):
        for c in range(w + 2):
            out[r, c] = a[min(max(r - 1, 0), h - 1), min(max(c - 1, 0), w - 1)]
    out[0, 0] = out[0, w + 1] = out[h + 1, 0] = out[h + 1, w + 1] = 0
    return out


def ref_cornerswap(a):
    h, w = a.shape
    cnt = np.array([int((a == c).sum()) for c in range(CHANNELS)], np.int64)
    cnt[0] = -(10 ** 12)
    rc = int(cnt.argmax())
    ys, xs = np.where(a == rc)
    if len(ys) == 0:
        return None
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    TL, TR = a[r0 + 1, c0 + 1], a[r0 + 1, c1 - 1]
    BL, BR = a[r1 - 1, c0 + 1], a[r1 - 1, c1 - 1]
    out = np.zeros_like(a)
    out[a == rc] = rc
    for (r, c), v in {(r0 - 1, c0 - 1): BR, (r0 - 1, c1 + 1): BL,
                      (r1 + 1, c0 - 1): TR, (r1 + 1, c1 + 1): TL}.items():
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = v
    return out


def ref_project(a):
    h, w = a.shape
    cnt = np.array([int((a == c).sum()) for c in range(CHANNELS)], np.int64)
    cnt[0] = -(10 ** 12)
    rc = int(cnt.argmax())
    if (a == rc).sum() == 0:
        return None
    ys, xs = np.where(a == rc)
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if (a == rc).sum() != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    out = a.copy()
    cover = {}
    for r in range(h):
        for c in range(w):
            v = a[r, c]
            if v == 0 or v == rc:
                continue
            tgt = None
            if c0 <= c <= c1:
                if r < r0:
                    tgt = (r0, c)
                elif r > r1:
                    tgt = (r1, c)
            if r0 <= r <= r1:
                if c < c0:
                    tgt = (r, c0)
                elif c > c1:
                    tgt = (r, c1)
            if tgt is not None:
                if tgt in cover and cover[tgt] != v:
                    return None
                cover[tgt] = v
                out[tgt] = v
    return out


def ref_stripemid(a):
    h, w = a.shape

    def g(r, c, k):
        return 1 if (0 <= r < h and 0 <= c < w and a[r, c] == k) else 0

    mid = np.zeros((h, w), int)
    for r in range(h):
        for c in range(w):
            k = a[r, c]
            if k != 0 and g(r - 1, c, k) and g(r + 1, c, k) and not g(r - 2, c, k) and not g(r + 2, c, k):
                mid[r, c] = 1
    if not mid.any():
        return None
    out = a.copy()
    for r in range(h):
        cols = [c for c in range(w) if mid[r, c]]
        if not cols:
            continue
        lpar = min(cols) % 2
        for c in cols:
            if c % 2 != lpar:
                out[r, c] = 0
    return out


def ref_stretchrect(a):
    h, w = a.shape
    M = (a != 0)
    iso = []
    for r in range(h):
        for c in range(w):
            if not M[r, c]:
                continue
            n = sum(1 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= r + dr < h and 0 <= c + dc < w and M[r + dr, c + dc])
            if n == 0:
                iso.append((r, c))
    if len(iso) != 1:
        return None
    mr, mc = iso[0]
    rect = [(r, c) for r in range(h) for c in range(w) if M[r, c] and (r, c) != (mr, mc)]
    if not rect:
        return None
    ys = [r for r, c in rect]
    xs = [c for r, c in rect]
    r0, r1, c0, c1 = min(ys), max(ys), min(xs), max(xs)
    O = a[r0, c0]
    cols = set(a[r, c] for r, c in rect)
    Io = [c for c in cols if c != O]
    if len(Io) != 1:
        return None
    I = Io[0]
    nr0, nr1 = min(r0, mr), max(r1, mr)
    nc0, nc1 = min(c0, mc), max(c1, mc)
    out = np.zeros_like(a)
    for r in range(nr0, nr1 + 1):
        for c in range(nc0, nc1 + 1):
            out[r, c] = O if (r in (nr0, nr1) or c in (nc0, nc1)) else I
    return out


def ref_markerstamp(a):
    h, w = a.shape
    if h < 3 or w < 3:
        return None
    tmpl = a[0:3, 0:3]
    if (tmpl == 1).any():
        return None
    markers = [(r, c) for r in range(h) for c in range(w) if a[r, c] == 1]
    if not markers:
        return None
    # reject overlapping stamps (ONNX is additive -> would differ)
    cover = {}
    for (mr, mc) in markers:
        for tr in range(3):
            for tc in range(3):
                rr, cc = mr + tr - 1, mc + tc - 1
                if 0 <= rr < h and 0 <= cc < w:
                    if (rr, cc) in cover:
                        return None
                    cover[(rr, cc)] = tmpl[tr, tc]
    out = a.copy()
    for (r, c) in markers:
        out[r, c] = 0
    for (rr, cc), v in cover.items():
        out[rr, cc] = v
    return out


def ref_colheight(a):
    h, w = a.shape
    out = np.zeros_like(a)
    counts = {c: int((a[:, c] != 0).sum()) for c in range(w)}
    nz = [c for c in range(w) if counts[c] > 0]
    if not nz:
        return None
    mx = max(counts[c] for c in nz)
    mn = min(counts[c] for c in nz)
    tall = [c for c in nz if counts[c] == mx]
    short = [c for c in nz if counts[c] == mn]
    if len(tall) != 1 or len(short) != 1:
        return None
    tc, sc = tall[0], short[0]
    for r in range(h):
        if a[r, tc] != 0:
            out[r, tc] = 1
        if a[r, sc] != 0:
            out[r, sc] = 2
    return out


def ref_staircase(a):
    h, w = a.shape
    out = a.copy()
    mk = [(r, c, a[r, c]) for r in range(h) for c in range(w) if a[r, c] != 0]
    if not mk:
        return None
    # at most one marker per row
    rows = [r for r, _, _ in mk]
    if len(set(rows)) != len(rows):
        return None
    for (r, c, v) in mk:
        out[r, c:w] = v
    cur = 0
    for r in range(h):
        if out[r, w - 1] != 0:
            cur = out[r, w - 1]
        elif cur != 0:
            out[r, w - 1] = cur
    return out


def ref_cornerfill(a):
    h, w = a.shape
    M = (a != 0)
    out = a.copy()
    for r in range(h):
        for c in range(w):
            if a[r, c] != 0:
                continue
            if (M[:r, :c].any() and M[:r, c + 1:].any()
                    and M[r + 1:, :c].any() and M[r + 1:, c + 1:].any()):
                out[r, c] = 2
    return out


def ref_symrot(a):
    h, w = a.shape
    ys, xs = np.where(a != 0)
    if len(ys) == 0:
        return None
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if (r1 - r0) != (c1 - c0):
        return None
    R, C = r0 + r1, c0 + c1
    if (R + C) % 2 or (R - C) % 2:
        return None
    hs, hd = (R + C) // 2, (R - C) // 2
    out = a.copy()
    pts = [(r, c) for r in range(h) for c in range(w) if a[r, c] != 0]

    def place(dst):
        for (r, c) in pts:
            rr, cc = dst(r, c)
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == 0:
                out[rr, cc] = a[r, c]

    place(lambda r, c: (R - r, C - c))
    place(lambda r, c: (hs - c, r - hd))
    place(lambda r, c: (c + hd, hs - r))
    return out


def _comps_samecolor(a):
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    res = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == 0:
                continue
            col = a[i, j]
            q = deque([(i, j)])
            seen[i, j] = True
            cells = [(i, j)]
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            res.append((col, cells))
    return res


def ref_rectfill(a):
    h, w = a.shape
    cs = _comps_samecolor(a)
    if len(cs) != 2:
        return None
    sizes = [len(c) for _, c in cs]
    if sizes[0] == sizes[1]:
        return None
    big = 0 if sizes[0] > sizes[1] else 1
    out = a.copy()
    for idx, (col, cells) in enumerate(cs):
        cset = set(cells)
        newcol = 2 if idx == big else 1
        for (r, c) in cells:
            if all((r + dr, c + dc) in cset for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                out[r, c] = newcol
    return out


# ========================================================================== #
# entry point                                                                #
# ========================================================================== #
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
        try:
            o = fn(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def _const_size(prs):
    shp = set(a.shape for a, _ in prs)
    return next(iter(shp)) if len(shp) == 1 else None


def _emit(out, name, builder):
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    out.append((name, m))


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    out = []

    if _matches(prs, ref_crop_swap2):
        _emit(out, "crop_swap2", build_crop_swap2)

    if _matches(prs, ref_square_hw):
        _emit(out, "square_hw", build_square_hw)

    if _matches(prs, ref_midcol):
        _emit(out, "midcol", build_midcol)

    if _matches(prs, ref_rowswapck):
        _emit(out, "rowswapck", build_rowswapck)

    if _matches(prs, ref_symrot):
        _emit(out, "symrot", build_symrot)

    if _matches(prs, ref_cornerfill):
        _emit(out, "cornerfill", build_cornerfill)

    if _matches(prs, ref_staircase):
        _emit(out, "staircase", build_staircase)

    if _matches(prs, ref_colheight):
        _emit(out, "colheight", build_colheight)

    if _matches(prs, ref_markerstamp):
        _emit(out, "markerstamp", build_markerstamp)

    if _matches(prs, ref_stretchrect):
        _emit(out, "stretchrect", build_stretchrect)

    if _matches(prs, ref_stripemid):
        _emit(out, "stripemid", build_stripemid)

    if _matches(prs, ref_project):
        _emit(out, "project", build_project)

    if _matches(prs, ref_cornerswap):
        _emit(out, "cornerswap", build_cornerswap)

    if _matches(prs, ref_edgepad):
        _emit(out, "edgepad", build_edgepad)

    if _matches(prs, ref_barstack):
        _emit(out, "barstack", build_barstack)

    sz = _const_size(prs)

    if sz is not None and _matches(prs, ref_linefill):
        ncols = sz[1]
        _emit(out, "linefill", lambda nc=ncols: build_linefill(nc))

    if sz is not None and _matches(prs, ref_cross):
        nr, nc = sz
        _emit(out, "cross", lambda nr=nr, nc=nc: build_cross(nr, nc))

    if sz is not None and _matches(prs, ref_rectfill):
        nr, nc = sz
        _emit(out, "rectfill", lambda nr=nr, nc=nc: build_rectfill(nr, nc))

    return out
