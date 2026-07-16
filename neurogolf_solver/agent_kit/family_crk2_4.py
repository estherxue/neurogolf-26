"""family_crk2_4 -- a grab-bag of exact ARC->ONNX solvers (opset-10, static shapes).

Each rule is detected structurally with a numpy reference that is validated EXACTLY
on every provided train/test/arc-gen pair before its ONNX model is emitted.  The ONNX
graph mirrors the numpy semantics 1:1 (verified offline with onnxruntime).

Tasks covered (NeuroGolf-2026 numbering):
  21   bands  -> solid background grid of size (#h-bands x #v-bands)
  29   crop the INTERIOR of the hollow-rectangle marker colour
  39   top-left quadrant (half size) of the non-background bounding box
  68   stamp a 3x3 ring of colour-2 around the unique (count==1) colour cell
  167  3x3: distinct-colour-count -> top-row / main-diag / anti-diag of colour 5
  195  fractal: input is a 3x3 shape upscaled x3; output = Kron(shape,shape)
  246  L-shaped colour-8 connector between the colour-2 and colour-3 cells
  291  -> 1x1 of the colour that forms the unique hollow rectangle
  341  bridge (colour 8) filling the gap between two aligned rectangles
  359  denoise: replace every row OR column by its majority colour
  375  draw both diagonals as background (0) on a solid colour square
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
H, W = HEIGHT, WIDTH
CBIG = 1000.0
BIG = 1.0e6


# --------------------------------------------------------------------------- #
# graph accumulator                                                            #
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


def _consts(g):
    g.rowidx = g.f([1, 1, H, 1], list(range(H)))
    g.colidx = g.f([1, 1, 1, W], list(range(W)))
    g.half = g.f([1, 1, 1, 1], [0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])
    g.two = g.f([1, 1, 1, 1], [2.0])
    g.cbig = g.f([1, 1, 1, 1], [CBIG])


def _onehot(g, c):
    v = [0.0] * CHANNELS
    v[c] = 1.0
    return g.f([1, CHANNELS, 1, 1], v)


# --------------------------------------------------------------------------- #
# shared pieces                                                                #
# --------------------------------------------------------------------------- #
def _realmask(g):
    return g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]


def _nonbg_mask(g):
    rm = _realmask(g)
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    return g.nd("Sub", [rm, ch0])                                      # [1,1,30,30]


def _bbox(g, content):
    """content [1,1,30,30] -> (minr,maxr,minc,maxc) each [1,1,1,1]."""
    rowhas = g.nd("ReduceMax", [content], axes=[3], keepdims=1)        # [1,1,30,1]
    colhas = g.nd("ReduceMax", [content], axes=[2], keepdims=1)        # [1,1,1,30]
    maxr = g.nd("ReduceMax", [g.nd("Mul", [rowhas, g.rowidx])], axes=[2], keepdims=1)
    minr = g.nd("Sub", [g.cbig, g.nd("ReduceMax",
                [g.nd("Mul", [rowhas, g.nd("Sub", [g.cbig, g.rowidx])])], axes=[2], keepdims=1)])
    maxc = g.nd("ReduceMax", [g.nd("Mul", [colhas, g.colidx])], axes=[3], keepdims=1)
    minc = g.nd("Sub", [g.cbig, g.nd("ReduceMax",
                [g.nd("Mul", [colhas, g.nd("Sub", [g.cbig, g.colidx])])], axes=[3], keepdims=1)])
    return minr, maxr, minc, maxc


def _crop(g, off_r, off_c, out_h, out_w, out_name="output"):
    """output = Srow @ input @ Scol  (re-anchor crop at origin)."""
    # Scol[k,j] = (k == j+off_c) & (j < out_w)
    diff_c = g.nd("Sub", [g.nd("Add", [g.colidx, off_c]), g.rowidx])
    match_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_c]), g.half])], to=F)
    trunc_c = g.nd("Cast", [g.nd("Less", [g.colidx, out_w])], to=F)
    Scol = g.nd("Mul", [match_c, trunc_c])
    # Srow[r,k] = (k == r+off_r) & (r < out_h)
    diff_r = g.nd("Sub", [g.colidx, g.nd("Add", [g.rowidx, off_r])])
    match_r = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_r]), g.half])], to=F)
    trunc_r = g.nd("Cast", [g.nd("Less", [g.rowidx, out_h])], to=F)
    Srow = g.nd("Mul", [match_r, trunc_r])
    shift1 = g.nd("MatMul", ["input", Scol])
    g.nd("MatMul", [Srow, shift1], out_name)


def _shift(g, x, dr, dc):
    """y[r,c] = x[r-dr, c-dc] (translate down dr, right dc), zero fill."""
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0, pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
    return g.nd("Slice", [p, st, en, g.i64([2, 3])])


def _argmax_gate(g, score):
    """score [1,10,1,1] -> one-hot gate [1,10,1,1] (argmax over channel)."""
    amax = g.nd("ArgMax", [score], axis=1, keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    return g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)


def _per_channel_bbox(g):
    """return rmin,rmax,cmin,cmax,count,present each [1,10,1,1]."""
    rowhas = g.nd("ReduceMax", ["input"], axes=[3], keepdims=1)        # [1,10,30,1]
    colhas = g.nd("ReduceMax", ["input"], axes=[2], keepdims=1)        # [1,10,1,30]
    rmax = g.nd("ReduceMax", [g.nd("Mul", [rowhas, g.rowidx])], axes=[2], keepdims=1)
    rmin = g.nd("Sub", [g.cbig, g.nd("ReduceMax",
                [g.nd("Mul", [rowhas, g.nd("Sub", [g.cbig, g.rowidx])])], axes=[2], keepdims=1)])
    cmax = g.nd("ReduceMax", [g.nd("Mul", [colhas, g.colidx])], axes=[3], keepdims=1)
    cmin = g.nd("Sub", [g.cbig, g.nd("ReduceMax",
                [g.nd("Mul", [colhas, g.nd("Sub", [g.cbig, g.colidx])])], axes=[3], keepdims=1)])
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    present = g.nd("Cast", [g.nd("Greater", [count, g.half])], to=F)
    return rmin, rmax, cmin, cmax, count, present


# --------------------------------------------------------------------------- #
# builders                                                                      #
# --------------------------------------------------------------------------- #
def build_21(g):
    _consts(g)
    c0 = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([1, 1]), g.i64([2, 3])])  # [1,10,1,1]
    rm = _realmask(g)                                                # [1,1,30,30]
    bgmask = g.nd("ReduceSum", [g.nd("Mul", ["input", c0])], axes=[1], keepdims=1)
    bg_row = g.nd("ReduceSum", [bgmask], axes=[3], keepdims=1)       # [1,1,30,1]
    cont_row = g.nd("ReduceSum", [rm], axes=[3], keepdims=1)
    seprow = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [bg_row, g.half])], to=F),
                          g.nd("Cast", [g.nd("Greater", [cont_row, g.half])], to=F)])
    hlines = g.nd("ReduceSum", [seprow], axes=[2], keepdims=1)
    R = g.nd("Add", [hlines, g.one])
    bg_col = g.nd("ReduceSum", [bgmask], axes=[2], keepdims=1)       # [1,1,1,30]
    cont_col = g.nd("ReduceSum", [rm], axes=[2], keepdims=1)
    sepcol = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [bg_col, g.half])], to=F),
                          g.nd("Cast", [g.nd("Greater", [cont_col, g.half])], to=F)])
    vlines = g.nd("ReduceSum", [sepcol], axes=[3], keepdims=1)
    C = g.nd("Add", [vlines, g.one])
    solid = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.rowidx, R])], to=F),
                         g.nd("Cast", [g.nd("Less", [g.colidx, C])], to=F)])
    g.nd("Mul", [solid, c0], "output")
    return _model(g)


def build_39(g):
    _consts(g)
    content = _nonbg_mask(g)
    minr, maxr, minc, maxc = _bbox(g, content)
    bh = g.nd("Add", [g.nd("Sub", [maxr, minr]), g.one])
    bw = g.nd("Add", [g.nd("Sub", [maxc, minc]), g.one])
    hh = g.nd("Mul", [bh, g.half])
    hw = g.nd("Mul", [bw, g.half])
    _crop(g, minr, minc, hh, hw)
    return _model(g)


def build_29(g):
    _consts(g)
    rmin, rmax, cmin, cmax, count, present = _per_channel_bbox(g)
    bh = g.nd("Add", [g.nd("Sub", [rmax, rmin]), g.one])
    bw = g.nd("Add", [g.nd("Sub", [cmax, cmin]), g.one])
    perim = g.nd("Sub", [g.nd("Add", [g.nd("Mul", [bh, g.two]), g.nd("Mul", [bw, g.two])]),
                         g.f([1, 1, 1, 1], [4.0])])
    eqp = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [count, perim])]), g.half])], to=F)
    big_bh = g.nd("Cast", [g.nd("Greater", [bh, g.f([1, 1, 1, 1], [2.5])])], to=F)
    big_bw = g.nd("Cast", [g.nd("Greater", [bw, g.f([1, 1, 1, 1], [2.5])])], to=F)
    # interior cells of each colour inside its own bbox must be empty (true hollow border)
    in_r = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [g.rowidx, rmin])], to=F),
                        g.nd("Cast", [g.nd("Less", [g.rowidx, rmax])], to=F)])   # [1,10,30,1]
    in_c = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [g.colidx, cmin])], to=F),
                        g.nd("Cast", [g.nd("Less", [g.colidx, cmax])], to=F)])   # [1,10,1,30]
    interior_mask = g.nd("Mul", [in_r, in_c])                                    # [1,10,30,30]
    interior_cnt = g.nd("ReduceSum", [g.nd("Mul", ["input", interior_mask])], axes=[2, 3], keepdims=1)
    hollow = g.nd("Cast", [g.nd("Less", [interior_cnt, g.half])], to=F)          # [1,10,1,1]
    match = g.nd("Mul", [g.nd("Mul", [eqp, big_bh]), g.nd("Mul", [big_bw, hollow])])  # [1,10,1,1]
    content = g.nd("ReduceSum", [g.nd("Mul", ["input", match])], axes=[1], keepdims=1)
    mnr, mxr, mnc, mxc = _bbox(g, content)
    off_r = g.nd("Add", [mnr, g.one])
    off_c = g.nd("Add", [mnc, g.one])
    out_h = g.nd("Sub", [g.nd("Sub", [mxr, mnr]), g.one])
    out_w = g.nd("Sub", [g.nd("Sub", [mxc, mnc]), g.one])
    _crop(g, off_r, off_c, out_h, out_w)
    return _model(g)


def build_68(g):
    _consts(g)
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)     # [1,10,1,1]
    isone = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [count, g.one])]), g.half])], to=F)
    # suppress channel0 (never the unique colour we want)
    nobg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    gate = g.nd("Mul", [isone, nobg])                                # [1,10,1,1]
    uniq = g.nd("ReduceSum", [g.nd("Mul", ["input", gate])], axes=[1], keepdims=1)  # [1,1,30,30]
    # 3x3 box via conv with ones kernel
    kern = g.f([1, 1, 3, 3], [1.0] * 9)
    box = g.nd("Conv", [uniq, kern], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    box = g.nd("Cast", [g.nd("Greater", [box, g.half])], to=F)       # [1,1,30,30]
    ring = g.nd("Sub", [box, uniq])                                  # 8 neighbours
    markerplane = g.nd("Mul", ["input", gate])                       # [1,10,30,30] marker colour at its cell
    out2 = g.nd("Mul", [ring, _onehot(g, 2)])                        # [1,10,30,30]
    # remaining grid cells stay background (colour 0 -> channel0)
    rm = _realmask(g)
    rowany = g.nd("ReduceMax", [rm], axes=[3], keepdims=1)
    colany = g.nd("ReduceMax", [rm], axes=[2], keepdims=1)
    grid = g.nd("Mul", [rowany, colany])
    gbg = g.nd("Mul", [grid, g.nd("Sub", [g.one, box])])
    obg = g.nd("Mul", [gbg, _onehot(g, 0)])
    g.nd("Add", [g.nd("Add", [markerplane, out2]), obg], "output")
    return _model(g)


def build_167(g):
    _consts(g)
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    present = g.nd("Cast", [g.nd("Greater", [count, g.half])], to=F)
    k = g.nd("ReduceSum", [present], axes=[1], keepdims=1)            # [1,1,1,1]
    top = np.zeros((1, 1, H, W), np.float32); top[0, 0, 0, 0:3] = 1
    mn = np.zeros((1, 1, H, W), np.float32)
    an = np.zeros((1, 1, H, W), np.float32)
    for i in range(3):
        mn[0, 0, i, i] = 1
        an[0, 0, i, 2 - i] = 1
    ctop = g.f([1, 1, H, W], top); cmn = g.f([1, 1, H, W], mn); can = g.f([1, 1, H, W], an)
    g1 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [k, g.one])]), g.half])], to=F)
    g2 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [k, g.two])]), g.half])], to=F)
    g3 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [k, g.f([1, 1, 1, 1], [3.0])])]), g.half])], to=F)
    mask = g.nd("Add", [g.nd("Add", [g.nd("Mul", [g1, ctop]), g.nd("Mul", [g2, cmn])]),
                        g.nd("Mul", [g3, can])])                     # [1,1,30,30]
    # region 3x3, off cells -> colour 0
    region = np.zeros((1, 1, H, W), np.float32); region[0, 0, 0:3, 0:3] = 1
    creg = g.f([1, 1, H, W], region)
    off = g.nd("Mul", [creg, g.nd("Sub", [g.one, mask])])
    o5 = g.nd("Mul", [mask, _onehot(g, 5)])
    o0 = g.nd("Mul", [off, _onehot(g, 0)])
    g.nd("Add", [o5, o0], "output")
    return _model(g)


def build_195(g):
    _consts(g)
    content = _nonbg_mask(g)
    minr, maxr, minc, maxc = _bbox(g, content)
    # sample S[i,j] = content at (minr+1+3i, minc+1+3j)
    igrid = g.f([1, 1, 3, 1], [0, 1, 2])
    jgrid = g.f([1, 1, 1, 3], [0, 1, 2])
    three = g.f([1, 1, 1, 1], [3.0])
    tr = g.nd("Add", [g.nd("Add", [minr, g.one]), g.nd("Mul", [igrid, three])])  # [1,1,3,1] target row per i
    tc = g.nd("Add", [g.nd("Add", [minc, g.one]), g.nd("Mul", [jgrid, three])])  # [1,1,1,3]
    # Rsel[i,k] = |k - tr(i)| < .5  ; k = colidx [1,1,1,30]  -> [1,1,3,30]
    Rsel = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.colidx, tr])]), g.half])], to=F)
    # Csel[k,j] = |k - tc(j)| < .5  ; k = rowidx [1,1,30,1] -> [1,1,30,3]
    Csel = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, tc])]), g.half])], to=F)
    S = g.nd("MatMul", [g.nd("MatMul", [Rsel, content]), Csel])      # [1,1,3,3]
    # constant block/tile matrices
    Krow = np.zeros((9, 3), np.float32); Kcol = np.zeros((3, 9), np.float32)
    Trow = np.zeros((9, 3), np.float32); Tcol = np.zeros((3, 9), np.float32)
    for r in range(9):
        Krow[r, r // 3] = 1; Trow[r, r % 3] = 1
    for c in range(9):
        Kcol[c // 3, c] = 1; Tcol[c % 3, c] = 1
    cKrow = g.f([1, 1, 9, 3], Krow); cKcol = g.f([1, 1, 3, 9], Kcol)
    cTrow = g.f([1, 1, 9, 3], Trow); cTcol = g.f([1, 1, 3, 9], Tcol)
    A = g.nd("MatMul", [g.nd("MatMul", [cKrow, S]), cKcol])          # [1,1,9,9]
    B = g.nd("MatMul", [g.nd("MatMul", [cTrow, S]), cTcol])
    out01 = g.nd("Mul", [A, B])                                      # [1,1,9,9]
    out_full = g.nd("Pad", [out01], mode="constant", value=0.0,
                    pads=[0, 0, 0, 0, 0, 0, H - 9, W - 9])           # [1,1,30,30]
    # colour
    bgneg = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    cnt = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    colorvec = _argmax_gate(g, g.nd("Add", [cnt, bgneg]))            # [1,10,1,1]
    region = np.zeros((1, 1, H, W), np.float32); region[0, 0, 0:9, 0:9] = 1
    creg = g.f([1, 1, H, W], region)
    off = g.nd("Mul", [creg, g.nd("Sub", [g.one, out_full])])
    on = g.nd("Mul", [out_full, colorvec])
    o0 = g.nd("Mul", [off, _onehot(g, 0)])
    g.nd("Add", [on, o0], "output")
    return _model(g)


def build_246(g):
    _consts(g)
    m2 = g.nd("Slice", ["input", g.i64([2]), g.i64([3]), g.i64([1])])  # [1,1,30,30]
    m3 = g.nd("Slice", ["input", g.i64([3]), g.i64([4]), g.i64([1])])
    rA = g.nd("ReduceSum", [g.nd("Mul", [m2, g.rowidx])], axes=[2, 3], keepdims=1)
    cA = g.nd("ReduceSum", [g.nd("Mul", [m2, g.colidx])], axes=[2, 3], keepdims=1)
    rB = g.nd("ReduceSum", [g.nd("Mul", [m3, g.rowidx])], axes=[2, 3], keepdims=1)
    cB = g.nd("ReduceSum", [g.nd("Mul", [m3, g.colidx])], axes=[2, 3], keepdims=1)
    cmin = g.nd("Min", [cA, cB]); cmax = g.nd("Max", [cA, cB])
    rmin = g.nd("Min", [rA, rB]); rmax = g.nd("Max", [rA, rB])
    # Hm
    onrowA = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, rA])]), g.half])], to=F)
    cge = g.nd("Cast", [g.nd("Greater", [g.colidx, g.nd("Sub", [cmin, g.half])])], to=F)
    cle = g.nd("Cast", [g.nd("Less", [g.colidx, g.nd("Add", [cmax, g.half])])], to=F)
    cnotA = g.nd("Cast", [g.nd("Greater", [g.nd("Abs", [g.nd("Sub", [g.colidx, cA])]), g.half])], to=F)
    Hm = g.nd("Mul", [g.nd("Mul", [onrowA, cge]), g.nd("Mul", [cle, cnotA])])
    # Vm
    oncolB = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.colidx, cB])]), g.half])], to=F)
    rge = g.nd("Cast", [g.nd("Greater", [g.rowidx, g.nd("Sub", [rmin, g.half])])], to=F)
    rle = g.nd("Cast", [g.nd("Less", [g.rowidx, g.nd("Add", [rmax, g.half])])], to=F)
    rnotB = g.nd("Cast", [g.nd("Greater", [g.nd("Abs", [g.nd("Sub", [g.rowidx, rB])]), g.half])], to=F)
    Vm = g.nd("Mul", [g.nd("Mul", [oncolB, rge]), g.nd("Mul", [rle, rnotB])])
    path = g.nd("Max", [Hm, Vm])                                     # [1,1,30,30]
    keep = g.nd("Mul", ["input", g.nd("Sub", [g.one, path])])
    add8 = g.nd("Mul", [path, _onehot(g, 8)])
    g.nd("Add", [keep, add8], "output")
    return _model(g)


def build_291(g):
    _consts(g)
    rmin, rmax, cmin, cmax, count, present = _per_channel_bbox(g)
    area = g.nd("Mul", [g.nd("Add", [g.nd("Sub", [rmax, rmin]), g.one]),
                        g.nd("Add", [g.nd("Sub", [cmax, cmin]), g.one])])
    deficit = g.nd("Sub", [area, count])
    cbigt = g.f([1, 1, 1, 1], [BIG])
    # score = present? deficit : -BIG ; also suppress channel0
    score = g.nd("Sub", [g.nd("Mul", [deficit, present]),
                         g.nd("Mul", [g.nd("Sub", [g.one, present]), cbigt])])
    bgpen = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    score = g.nd("Add", [score, bgpen])
    gate = _argmax_gate(g, score)                                    # [1,10,1,1]
    cell00 = np.zeros((1, 1, H, W), np.float32); cell00[0, 0, 0, 0] = 1
    c00 = g.f([1, 1, H, W], cell00)
    g.nd("Mul", [c00, gate], "output")
    return _model(g)


def build_341(g):
    _consts(g)
    rmin, rmax, cmin, cmax, count, present = _per_channel_bbox(g)
    cbigt = g.f([1, 1, 1, 1], [BIG])
    np1 = g.nd("Sub", [g.one, present])                              # 1 when empty
    # max over channels of (present? x : -BIG) ; min over channels of (present? x : +BIG)
    def red_max(x):
        return g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [x, present]), g.nd("Mul", [np1, cbigt])])],
                    axes=[1], keepdims=1)
    def red_min(x):
        return g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [x, present]), g.nd("Mul", [np1, cbigt])])],
                    axes=[1], keepdims=1)
    maxcmin = red_max(cmin); mincmax = red_min(cmax)
    maxrmin = red_max(rmin); minrmax = red_min(rmax)
    # bridge_h : c in (mincmax, maxcmin) , r in (maxrmin, minrmax)
    def gt(x, v):
        return g.nd("Cast", [g.nd("Greater", [x, g.nd("Add", [v, g.half])])], to=F)
    def lt(x, v):
        return g.nd("Cast", [g.nd("Less", [x, g.nd("Sub", [v, g.half])])], to=F)
    bh = g.nd("Mul", [g.nd("Mul", [gt(g.colidx, mincmax), lt(g.colidx, maxcmin)]),
                      g.nd("Mul", [gt(g.rowidx, maxrmin), lt(g.rowidx, minrmax)])])
    bv = g.nd("Mul", [g.nd("Mul", [gt(g.rowidx, minrmax), lt(g.rowidx, maxrmin)]),
                      g.nd("Mul", [gt(g.colidx, maxcmin), lt(g.colidx, mincmax)])])
    bridge = g.nd("Max", [bh, bv])
    keep = g.nd("Mul", ["input", g.nd("Sub", [g.one, bridge])])
    add8 = g.nd("Mul", [bridge, _onehot(g, 8)])
    g.nd("Add", [keep, add8], "output")
    return _model(g)


def build_359(g):
    _consts(g)
    rm = _realmask(g)
    rowany = g.nd("ReduceMax", [rm], axes=[3], keepdims=1)           # [1,1,30,1]
    colany = g.nd("ReduceMax", [rm], axes=[2], keepdims=1)           # [1,1,1,30]
    grid = g.nd("Mul", [rowany, colany])                            # [1,1,30,30]
    bgneg = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    crow = g.nd("Add", [g.nd("ReduceSum", ["input"], axes=[3], keepdims=1), bgneg])  # [1,10,30,1]
    rarg = g.nd("ArgMax", [crow], axis=1, keepdims=1)               # [1,1,30,1]
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    roh = g.nd("Cast", [g.nd("Equal", [rarg, idx])], to=F)          # [1,10,30,1]
    ccol = g.nd("Add", [g.nd("ReduceSum", ["input"], axes=[2], keepdims=1), bgneg])  # [1,10,1,30]
    carg = g.nd("ArgMax", [ccol], axis=1, keepdims=1)
    coh = g.nd("Cast", [g.nd("Equal", [carg, idx])], to=F)          # [1,10,1,30]
    rowout = g.nd("Mul", [roh, grid])                              # [1,10,30,30]
    colout = g.nd("Mul", [coh, grid])
    agr = g.nd("ReduceSum", [g.nd("Mul", ["input", roh])], axes=[1, 2, 3], keepdims=1)
    agc = g.nd("ReduceSum", [g.nd("Mul", ["input", coh])], axes=[1, 2, 3], keepdims=1)
    orow = g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [agr, agc]), g.f([1, 1, 1, 1], [-0.5])])], to=F)
    ocol = g.nd("Sub", [g.one, orow])
    g.nd("Add", [g.nd("Mul", [rowout, orow]), g.nd("Mul", [colout, ocol])], "output")
    return _model(g)


def build_375(g):
    _consts(g)
    rm = _realmask(g)
    colany = g.nd("ReduceMax", [rm], axes=[2], keepdims=1)           # [1,1,1,30]
    maxc = g.nd("ReduceMax", [g.nd("Mul", [colany, g.colidx])], axes=[3], keepdims=1)
    N = g.nd("Add", [maxc, g.one])
    grid = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.rowidx, N])], to=F),
                        g.nd("Cast", [g.nd("Less", [g.colidx, N])], to=F)])
    main = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, g.colidx])]), g.half])], to=F)
    nm1 = g.nd("Sub", [N, g.one])
    anti = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.nd("Add", [g.rowidx, g.colidx]), nm1])]), g.half])], to=F)
    X = g.nd("Max", [main, anti])
    bgneg = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    cnt = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    colorvec = _argmax_gate(g, g.nd("Add", [cnt, bgneg]))
    fill = g.nd("Mul", [grid, g.nd("Sub", [g.one, X])])
    xgrid = g.nd("Mul", [grid, X])
    o_c = g.nd("Mul", [fill, colorvec])
    o_0 = g.nd("Mul", [xgrid, _onehot(g, 0)])
    g.nd("Add", [o_c, o_0], "output")
    return _model(g)


def build_399(g):
    _consts(g)
    B = g.nd("Slice", ["input", g.i64([2]), g.i64([3]), g.i64([1])])  # colour-2 mask [1,1,30,30]
    erode = g.nd("Mul", [g.nd("Mul", [B, _shift(g, B, 0, -1)]),
                         g.nd("Mul", [_shift(g, B, -1, 0), _shift(g, B, -1, -1)])])
    N = g.nd("ReduceSum", [erode], axes=[2, 3], keepdims=1)            # [1,1,1,1]
    rank = np.full((1, 1, H, W), 999.0, np.float32)
    for (rr, cc), k in {(0, 0): 0, (0, 2): 1, (1, 1): 2, (2, 0): 3, (2, 2): 4}.items():
        rank[0, 0, rr, cc] = k
    crank = g.f([1, 1, H, W], rank)
    on = g.nd("Cast", [g.nd("Less", [crank, N])], to=F)               # [1,1,30,30]
    region = np.zeros((1, 1, H, W), np.float32); region[0, 0, 0:3, 0:3] = 1
    creg = g.f([1, 1, H, W], region)
    o1 = g.nd("Mul", [on, _onehot(g, 1)])
    o0 = g.nd("Mul", [g.nd("Mul", [creg, g.nd("Sub", [g.one, on])]), _onehot(g, 0)])
    g.nd("Add", [o1, o0], "output")
    return _model(g)


def _band_project(g, ch0, nbg, axis):
    """axis='h': horizontal bands -> project holes vertically within each colour band.
    axis='v': vertical bands.  Returns newzero mask [1,1,30,30]."""
    nobg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))    # suppress channel0
    if axis == "h":
        rowhas_k = g.nd("Mul", [g.nd("ReduceMax", ["input"], axes=[3], keepdims=1), nobg])  # [1,10,30,1]
        rowhasnbg = g.nd("ReduceMax", [nbg], axes=[3], keepdims=1)      # [1,1,30,1]
        hole = g.nd("Mul", [ch0, rowhasnbg])                          # [1,1,30,30]
        colhole_k = g.nd("ReduceMax", [g.nd("Mul", [hole, rowhas_k])], axes=[2], keepdims=1)  # [1,10,1,30]
        nz = g.nd("ReduceMax", [g.nd("Mul", [rowhas_k, colhole_k])], axes=[1], keepdims=1)    # [1,1,30,30]
    else:
        colhas_k = g.nd("Mul", [g.nd("ReduceMax", ["input"], axes=[2], keepdims=1), nobg])  # [1,10,1,30]
        colhasnbg = g.nd("ReduceMax", [nbg], axes=[2], keepdims=1)      # [1,1,1,30]
        hole = g.nd("Mul", [ch0, colhasnbg])
        rowhole_k = g.nd("ReduceMax", [g.nd("Mul", [hole, colhas_k])], axes=[3], keepdims=1)  # [1,10,30,1]
        nz = g.nd("ReduceMax", [g.nd("Mul", [colhas_k, rowhole_k])], axes=[1], keepdims=1)
    return nz


def _row_color_oh(g, axis):
    """one-hot dominant non-bg colour per row (axis='h') or per col (axis='v'),
    gated to grid rows/cols only."""
    bgneg = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    rm = _realmask(g)
    if axis == "h":
        counts = g.nd("ReduceSum", ["input"], axes=[3], keepdims=1)     # [1,10,30,1]
        amax = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
        idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
        oh = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)          # [1,10,30,1]
        valid = g.nd("ReduceMax", [rm], axes=[3], keepdims=1)          # [1,1,30,1]
    else:
        counts = g.nd("ReduceSum", ["input"], axes=[2], keepdims=1)     # [1,10,1,30]
        amax = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
        idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
        oh = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)          # [1,10,1,30]
        valid = g.nd("ReduceMax", [rm], axes=[2], keepdims=1)          # [1,1,1,30]
    return g.nd("Mul", [oh, valid]), valid


def build_178(g):
    _consts(g)
    Lt = np.zeros((H, W), np.float32)
    for p in range(H):
        for r in range(W):
            if r <= p: Lt[p, r] = 1
    Ltri = g.f([1, 1, H, W], Lt)            # lower-tri (incl diag)
    Utri = g.f([1, 1, H, W], Lt.T)          # upper-tri
    # orientation
    nbg = _nonbg_mask(g)

    def cost(dr, dc):
        sh = _shift(g, "input", dr, dc)
        same = g.nd("ReduceSum", [g.nd("Mul", ["input", sh])], axes=[1], keepdims=1)
        nbg_sh = _shift(g, nbg, dr, dc)
        ch = g.nd("Mul", [g.nd("Mul", [g.nd("Sub", [g.one, same]), nbg]), nbg_sh])
        return g.nd("ReduceSum", [ch], axes=[2, 3], keepdims=1)
    ru = g.nd("Cast", [g.nd("Less", [cost(0, 1), cost(1, 0)])], to=F)  # rows uniform -> column output

    # ---- column branch (rows uniform) ----
    roh, rvalid = _row_color_oh(g, "h")                                # [1,10,30,1],[1,1,30,1]
    same_r = g.nd("ReduceSum", [g.nd("Mul", [roh, _shift(g, roh, 1, 0)])], axes=[1], keepdims=1)  # [1,1,30,1]
    start_r = g.nd("Mul", [rvalid, g.nd("Sub", [g.one, same_r])])      # [1,1,30,1]
    rank_r = g.nd("Sub", [g.nd("MatMul", [Ltri, start_r]), g.one])     # [1,1,30,1]
    rankT = g.nd("Transpose", [rank_r], perm=[0, 1, 3, 2])             # [1,1,1,30]
    startT = g.nd("Transpose", [start_r], perm=[0, 1, 3, 2])           # [1,1,1,30]
    eq = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rankT, g.rowidx])]), g.half])], to=F)  # [1,1,30,30]
    Sel = g.nd("Mul", [eq, startT])
    out_col = g.nd("MatMul", [Sel, roh])                               # [1,10,30,1]
    colpad = g.nd("Pad", [out_col], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, W - 1])

    # ---- row branch (cols uniform) ----
    coh, cvalid = _row_color_oh(g, "v")                                # [1,10,1,30],[1,1,1,30]
    same_c = g.nd("ReduceSum", [g.nd("Mul", [coh, _shift(g, coh, 0, 1)])], axes=[1], keepdims=1)  # [1,1,1,30]
    start_c = g.nd("Mul", [cvalid, g.nd("Sub", [g.one, same_c])])      # [1,1,1,30]
    rank_c = g.nd("Sub", [g.nd("MatMul", [start_c, Utri]), g.one])     # [1,1,1,30]
    rank_cT = g.nd("Transpose", [rank_c], perm=[0, 1, 3, 2])           # [1,1,30,1]
    start_cT = g.nd("Transpose", [start_c], perm=[0, 1, 3, 2])         # [1,1,30,1]
    eq2 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rank_cT, g.colidx])]), g.half])], to=F)  # [1,1,30,30]
    Sel2 = g.nd("Mul", [eq2, start_cT])
    out_row = g.nd("MatMul", [coh, Sel2])                              # [1,10,1,30]
    rowpad = g.nd("Pad", [out_row], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, H - 1, 0])

    g.nd("Add", [g.nd("Mul", [colpad, ru]), g.nd("Mul", [rowpad, g.nd("Sub", [g.one, ru])])], "output")
    return _model(g)


def build_62(g):
    _consts(g)
    nobg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    bgneg = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    pres = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [count, g.half])], to=F), nobg])
    # shape = majority nonbg, marker = minority nonbg
    shapevec = _argmax_gate(g, g.nd("Add", [count, bgneg]))           # most frequent nonbg
    bigpos = g.f([1, CHANNELS, 1, 1], [BIG] + [0.0] * (CHANNELS - 1))
    markerscore = g.nd("Add", [g.nd("Sub", [g.nd("Mul", [g.nd("Sub", [g.f([1, 1, 1, 1], [BIG]), count]), pres]),
                                            g.nd("Mul", [g.nd("Sub", [g.one, pres]), g.f([1, 1, 1, 1], [BIG])])]), bigpos])
    markervec = _argmax_gate(g, markerscore)                          # least frequent present nonbg
    sh = g.nd("ReduceSum", [g.nd("Mul", ["input", shapevec])], axes=[1], keepdims=1)  # [1,1,30,30]
    mk = g.nd("ReduceSum", [g.nd("Mul", ["input", markervec])], axes=[1], keepdims=1)

    def stat(m):
        sx = g.nd("ReduceSum", [g.nd("Mul", [m, g.colidx])], axes=[2, 3], keepdims=1)
        sy = g.nd("ReduceSum", [g.nd("Mul", [m, g.rowidx])], axes=[2, 3], keepdims=1)
        cn = g.nd("ReduceSum", [m], axes=[2, 3], keepdims=1)
        rowhas = g.nd("ReduceMax", [m], axes=[3], keepdims=1)
        colhas = g.nd("ReduceMax", [m], axes=[2], keepdims=1)
        maxc = g.nd("ReduceMax", [g.nd("Mul", [colhas, g.colidx])], axes=[3], keepdims=1)
        minc = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [g.nd("Mul", [colhas, g.nd("Sub", [g.cbig, g.colidx])])], axes=[3], keepdims=1)])
        maxr = g.nd("ReduceMax", [g.nd("Mul", [rowhas, g.rowidx])], axes=[2], keepdims=1)
        minr = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [g.nd("Mul", [rowhas, g.nd("Sub", [g.cbig, g.rowidx])])], axes=[2], keepdims=1)])
        return sx, sy, cn, maxc, minc, maxr, minr
    sxS, syS, cS, smaxc, sminc, smaxr, sminr = stat(sh)
    sxM, syM, cM, mmaxc, mminc, mmaxr, mminr = stat(mk)
    DX = g.nd("Sub", [g.nd("Mul", [sxM, cS]), g.nd("Mul", [sxS, cM])])
    DY = g.nd("Sub", [g.nd("Mul", [syM, cS]), g.nd("Mul", [syS, cM])])
    horiz = g.nd("Sub", [g.one, g.nd("Cast", [g.nd("Less", [g.nd("Abs", [DX]), g.nd("Abs", [DY])])], to=F)])
    mright = g.nd("Cast", [g.nd("Greater", [DX, g.f([1, 1, 1, 1], [0.0])])], to=F)
    mbelow = g.nd("Cast", [g.nd("Greater", [DY, g.f([1, 1, 1, 1], [0.0])])], to=F)
    axis_h = g.nd("Add", [g.nd("Mul", [mright, g.nd("Add", [smaxc, mminc])]),
                          g.nd("Mul", [g.nd("Sub", [g.one, mright]), g.nd("Add", [mmaxc, sminc])])])
    axis_v = g.nd("Add", [g.nd("Mul", [mbelow, g.nd("Add", [smaxr, mminr])]),
                          g.nd("Mul", [g.nd("Sub", [g.one, mbelow]), g.nd("Add", [mmaxr, sminr])])])
    RefH = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.nd("Add", [g.rowidx, g.colidx]), axis_h])]), g.half])], to=F)
    RefV = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.nd("Add", [g.rowidx, g.colidx]), axis_v])]), g.half])], to=F)
    mir_h = g.nd("MatMul", [sh, RefH])
    mir_v = g.nd("MatMul", [RefV, sh])
    sel_mir = g.nd("Add", [g.nd("Mul", [mir_h, horiz]), g.nd("Mul", [mir_v, g.nd("Sub", [g.one, horiz])])])
    shapeout = g.nd("Max", [sh, sel_mir])                            # [1,1,30,30]
    rm = _realmask(g)
    Hc = g.nd("Add", [g.nd("ReduceMax", [g.nd("Mul", [g.nd("ReduceMax", [rm], axes=[3], keepdims=1), g.rowidx])], axes=[2], keepdims=1), g.one])
    Wc = g.nd("Add", [g.nd("ReduceMax", [g.nd("Mul", [g.nd("ReduceMax", [rm], axes=[2], keepdims=1), g.colidx])], axes=[3], keepdims=1), g.one])
    grid = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.rowidx, Hc])], to=F),
                        g.nd("Cast", [g.nd("Less", [g.colidx, Wc])], to=F)])
    o_shape = g.nd("Mul", [shapeout, shapevec])
    o_bg = g.nd("Mul", [g.nd("Mul", [grid, g.nd("Sub", [g.one, shapeout])]), _onehot(g, 3)])
    g.nd("Add", [o_shape, o_bg], "output")
    return _model(g)


def build_301(g):
    _consts(g)
    nobg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    bgneg = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    rmin, rmax, cmin, cmax, count, present = _per_channel_bbox(g)
    anchorvec = _argmax_gate(g, g.nd("Add", [rmax, bgneg]))           # colour reaching bottom
    present_nb = g.nd("Mul", [present, nobg])
    gate_bar = g.nd("Mul", [present_nb, g.nd("Sub", [g.one, anchorvec])])  # [1,10,1,1]
    nb = g.nd("ReduceSum", [gate_bar], axes=[1], keepdims=1)          # [1,1,1,1]
    # grid extent
    rm = _realmask(g)
    Hc = g.nd("Add", [g.nd("ReduceMax", [g.nd("Mul", [g.nd("ReduceMax", [rm], axes=[3], keepdims=1), g.rowidx])], axes=[2], keepdims=1), g.one])
    Wc = g.nd("Add", [g.nd("ReduceMax", [g.nd("Mul", [g.nd("ReduceMax", [rm], axes=[2], keepdims=1), g.colidx])], axes=[3], keepdims=1), g.one])
    # rank by length (count), among bars only
    li = g.nd("Reshape", [count, g.i64([1, 1, CHANNELS, 1])])
    lj = g.nd("Reshape", [count, g.i64([1, 1, 1, CHANNELS])])
    gbj = g.nd("Reshape", [gate_bar, g.i64([1, 1, 1, CHANNELS])])
    less = g.nd("Cast", [g.nd("Less", [lj, li])], to=F)               # [1,1,10,10]
    rank = g.nd("ReduceSum", [g.nd("Mul", [less, gbj])], axes=[3], keepdims=1)  # [1,1,10,1]
    rank = g.nd("Reshape", [rank, g.i64([1, CHANNELS, 1, 1])])        # [1,10,1,1]
    row_c = g.nd("Sub", [g.nd("Add", [g.nd("Sub", [Hc, g.one]), rank]), nb])  # H-1-nb+rank [1,10,1,1]
    rowmatch = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, row_c])]), g.half])], to=F)  # [1,10,30,1]
    startcol = g.nd("Sub", [Wc, count])                              # W-length [1,10,1,1]
    cge = g.nd("Cast", [g.nd("Greater", [g.colidx, g.nd("Sub", [startcol, g.half])])], to=F)  # [1,10,1,30]
    clt = g.nd("Cast", [g.nd("Less", [g.colidx, Wc])], to=F)         # [1,1,1,30]
    colmatch = g.nd("Mul", [cge, clt])                              # [1,10,1,30]
    barcells = g.nd("Mul", [g.nd("Mul", [rowmatch, colmatch]), gate_bar])  # [1,10,30,30]
    # anchor row (bottom) full width
    anchorrow = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, g.nd("Sub", [Hc, g.one])])]), g.half])], to=F),
                             g.nd("Cast", [g.nd("Less", [g.colidx, Wc])], to=F)])  # [1,1,30,30]
    anchorout = g.nd("Mul", [anchorrow, anchorvec])                  # [1,10,30,30]
    # background fill
    occ = g.nd("ReduceSum", [g.nd("Add", [barcells, anchorout])], axes=[1], keepdims=1)
    grid = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.rowidx, Hc])], to=F),
                        g.nd("Cast", [g.nd("Less", [g.colidx, Wc])], to=F)])
    bgcells = g.nd("Mul", [grid, g.nd("Sub", [g.one, occ])])
    bgout = g.nd("Mul", [bgcells, _onehot(g, 0)])
    g.nd("Add", [g.nd("Add", [barcells, anchorout]), bgout], "output")
    return _model(g)


def build_51(g):
    _consts(g)
    nobg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)      # [1,10,1,1]
    isone = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [count, g.one])]), g.half])], to=F)
    gate_B = g.nd("Mul", [isone, nobg])                               # marker colour [1,10,1,1]
    pres = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [count, g.half])], to=F), nobg])
    gate_A = g.nd("Mul", [pres, g.nd("Sub", [g.one, gate_B])])        # shape colour
    mk = g.nd("ReduceSum", [g.nd("Mul", ["input", gate_B])], axes=[1], keepdims=1)   # [1,1,30,30]
    sh = g.nd("ReduceSum", [g.nd("Mul", ["input", gate_A])], axes=[1], keepdims=1)
    mr = g.nd("ReduceSum", [g.nd("Mul", [mk, g.rowidx])], axes=[2, 3], keepdims=1)
    mc = g.nd("ReduceSum", [g.nd("Mul", [mk, g.colidx])], axes=[2, 3], keepdims=1)
    sy = g.nd("ReduceSum", [g.nd("Mul", [sh, g.rowidx])], axes=[2, 3], keepdims=1)
    sx = g.nd("ReduceSum", [g.nd("Mul", [sh, g.colidx])], axes=[2, 3], keepdims=1)
    cA = g.nd("ReduceSum", [sh], axes=[2, 3], keepdims=1)
    DX = g.nd("Sub", [sx, g.nd("Mul", [mc, cA])])
    DY = g.nd("Sub", [sy, g.nd("Mul", [mr, cA])])
    aDX = g.nd("Abs", [DX]); aDY = g.nd("Abs", [DY])
    horiz = g.nd("Sub", [g.one, g.nd("Cast", [g.nd("Less", [aDX, aDY])], to=F)])  # |DX|>=|DY|
    nh = g.nd("Sub", [g.one, horiz])
    dxpos = g.nd("Cast", [g.nd("Greater", [DX, g.half])], to=F)
    dypos = g.nd("Cast", [g.nd("Greater", [DY, g.half])], to=F)
    d_right = g.nd("Mul", [horiz, dxpos])
    d_left = g.nd("Mul", [horiz, g.nd("Sub", [g.one, dxpos])])
    d_down = g.nd("Mul", [nh, dypos])
    d_up = g.nd("Mul", [nh, g.nd("Sub", [g.one, dypos])])
    # shape bbox edges
    rowhas = g.nd("ReduceMax", [sh], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [sh], axes=[2], keepdims=1)
    smaxc = g.nd("ReduceMax", [g.nd("Mul", [colhas, g.colidx])], axes=[3], keepdims=1)
    sminc = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [g.nd("Mul", [colhas, g.nd("Sub", [g.cbig, g.colidx])])], axes=[3], keepdims=1)])
    smaxr = g.nd("ReduceMax", [g.nd("Mul", [rowhas, g.rowidx])], axes=[2], keepdims=1)
    sminr = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [g.nd("Mul", [rowhas, g.nd("Sub", [g.cbig, g.rowidx])])], axes=[2], keepdims=1)])
    # grid extent
    rm = _realmask(g)
    Wc = g.nd("Add", [g.nd("ReduceMax", [g.nd("Mul", [g.nd("ReduceMax", [rm], axes=[2], keepdims=1), g.colidx])], axes=[3], keepdims=1), g.one])
    Hc = g.nd("Add", [g.nd("ReduceMax", [g.nd("Mul", [g.nd("ReduceMax", [rm], axes=[3], keepdims=1), g.rowidx])], axes=[2], keepdims=1), g.one])
    onrow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, mr])]), g.half])], to=F)  # [1,1,30,1]
    oncol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.colidx, mc])]), g.half])], to=F)  # [1,1,1,30]
    cltW = g.nd("Cast", [g.nd("Less", [g.colidx, Wc])], to=F)
    rltH = g.nd("Cast", [g.nd("Less", [g.rowidx, Hc])], to=F)
    rr = g.nd("Mul", [g.nd("Mul", [onrow, g.nd("Cast", [g.nd("Greater", [g.colidx, g.nd("Add", [smaxc, g.half])])], to=F)]), cltW])
    rl = g.nd("Mul", [onrow, g.nd("Cast", [g.nd("Less", [g.colidx, g.nd("Sub", [sminc, g.half])])], to=F)])
    rd = g.nd("Mul", [g.nd("Mul", [oncol, g.nd("Cast", [g.nd("Greater", [g.rowidx, g.nd("Add", [smaxr, g.half])])], to=F)]), rltH])
    ru = g.nd("Mul", [oncol, g.nd("Cast", [g.nd("Less", [g.rowidx, g.nd("Sub", [sminr, g.half])])], to=F)])
    ray = g.nd("Add", [g.nd("Add", [g.nd("Mul", [rr, d_right]), g.nd("Mul", [rl, d_left])]),
                       g.nd("Add", [g.nd("Mul", [rd, d_down]), g.nd("Mul", [ru, d_up])])])  # [1,1,30,30]
    keep = g.nd("Mul", ["input", g.nd("Sub", [g.one, ray])])
    addB = g.nd("Mul", [ray, gate_B])
    g.nd("Add", [keep, addB], "output")
    return _model(g)


def build_146(g):
    _consts(g)
    outs = []
    for k in range(3):
        bl = g.nd("Slice", ["input", g.i64([3 * k, 0]), g.i64([3 * k + 3, 3]), g.i64([2, 3])])  # [1,10,3,3]
        blT = g.nd("Transpose", [bl], perm=[0, 1, 3, 2])
        same = g.nd("ReduceSum", [g.nd("Mul", [bl, blT])], axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
        asym = g.nd("Sub", [g.f([1, 1, 1, 1], [9.0]), same])
        gate = g.nd("Cast", [g.nd("Greater", [asym, g.half])], to=F)
        pad = g.nd("Pad", [bl], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, H - 3, W - 3])
        outs.append(g.nd("Mul", [pad, gate]))
    g.nd("Add", [g.nd("Add", [outs[0], outs[1]]), outs[2]], "output")
    return _model(g)


def build_221(g):
    _consts(g)
    nbg = _nonbg_mask(g)                                              # [1,1,30,30] P at top-left
    K = g.nd("ReduceSum", [nbg], axes=[2, 3], keepdims=1)             # [1,1,1,1]
    N = g.nd("Sub", [g.f([1, 1, 1, 1], [9.0]), K])
    threeN = g.nd("Mul", [N, g.f([1, 1, 1, 1], [3.0])])
    Pmask = g.nd("Slice", [nbg, g.i64([0, 0]), g.i64([3, 3]), g.i64([2, 3])])  # [1,1,3,3]
    Trow = np.zeros((H, 3), np.float32); Tcol = np.zeros((3, W), np.float32)
    for R in range(H): Trow[R, R % 3] = 1
    for C in range(W): Tcol[C % 3, C] = 1
    cTrow = g.f([1, 1, H, 3], Trow); cTcol = g.f([1, 1, 3, W], Tcol)
    Ptiled = g.nd("MatMul", [g.nd("MatMul", [cTrow, Pmask]), cTcol])  # [1,1,30,30]
    block_bi = g.f([1, 1, H, 1], [R // 3 for R in range(H)])
    block_bj = g.f([1, 1, 1, W], [C // 3 for C in range(W)])
    blocklin = g.nd("Add", [g.nd("Mul", [block_bi, N]), block_bj])    # [1,1,30,30]
    active = g.nd("Cast", [g.nd("Less", [blocklin, K])], to=F)
    gridmask = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.rowidx, threeN])], to=F),
                            g.nd("Cast", [g.nd("Less", [g.colidx, threeN])], to=F)])
    on = g.nd("Mul", [g.nd("Mul", [active, Ptiled]), gridmask])       # [1,1,30,30]
    bgneg = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    cnt = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    colorvec = _argmax_gate(g, g.nd("Add", [cnt, bgneg]))
    zero = g.nd("Mul", [gridmask, g.nd("Sub", [g.one, on])])
    g.nd("Add", [g.nd("Mul", [on, colorvec]), g.nd("Mul", [zero, _onehot(g, 0)])], "output")
    return _model(g)


def build_115(g):
    _consts(g)
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)      # [1,10,1,1]
    present = g.nd("Cast", [g.nd("Greater", [count, g.half])], to=F)
    sumcol = g.nd("ReduceSum", [g.nd("Mul", ["input", g.colidx])], axes=[2, 3], keepdims=1)
    sumrow = g.nd("ReduceSum", [g.nd("Mul", ["input", g.rowidx])], axes=[2, 3], keepdims=1)
    countsafe = g.nd("Add", [count, g.nd("Sub", [g.one, present])])
    meancol = g.nd("Div", [sumcol, countsafe])                        # [1,10,1,1]
    meanrow = g.nd("Div", [sumrow, countsafe])
    cbigt = g.f([1, 1, 1, 1], [BIG])
    npres = g.nd("Sub", [g.one, present])

    def spread(m):
        mx = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [m, present]), g.nd("Mul", [npres, cbigt])])], axes=[1], keepdims=1)
        mn = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [m, present]), g.nd("Mul", [npres, cbigt])])], axes=[1], keepdims=1)
        return g.nd("Sub", [mx, mn])
    cspread = spread(meancol); rspread = spread(meanrow)
    orient_h = g.nd("Sub", [g.one, g.nd("Cast", [g.nd("Less", [cspread, rspread])], to=F)])  # [1,1,1,1]
    present_j = g.nd("Reshape", [present, g.i64([1, 1, 1, CHANNELS])])  # [1,1,1,10]

    def rank(m):
        mi = g.nd("Reshape", [m, g.i64([1, 1, CHANNELS, 1])])          # [1,1,10,1]
        mj = g.nd("Reshape", [m, g.i64([1, 1, 1, CHANNELS])])          # [1,1,1,10]
        less = g.nd("Cast", [g.nd("Less", [mj, mi])], to=F)            # [1,1,10,10] (j<i)
        rk = g.nd("ReduceSum", [g.nd("Mul", [less, present_j])], axes=[3], keepdims=1)  # [1,1,10,1]
        return g.nd("Reshape", [rk, g.i64([1, CHANNELS, 1, 1])])       # [1,10,1,1]
    rankcol = rank(meancol); rankrow = rank(meanrow)
    out_h = g.nd("Mul", [present, g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.colidx, rankcol])]), g.half])], to=F)])  # [1,10,1,30]
    out_v = g.nd("Mul", [present, g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, rankrow])]), g.half])], to=F)])  # [1,10,30,1]
    hpad = g.nd("Pad", [out_h], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, H - 1, 0])
    vpad = g.nd("Pad", [out_v], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, W - 1])
    g.nd("Add", [g.nd("Mul", [hpad, orient_h]), g.nd("Mul", [vpad, g.nd("Sub", [g.one, orient_h])])], "output")
    return _model(g)


def build_202(g):
    _consts(g)
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])  # [1,1,30,30]
    rm = _realmask(g)
    nbg = g.nd("Sub", [rm, ch0])                                       # [1,1,30,30] (1 on non-bg cells)
    # orientation cost: colour-changes between adjacent non-bg cells
    def cost(dr, dc):
        sh = _shift(g, "input", dr, dc)
        same = g.nd("ReduceSum", [g.nd("Mul", ["input", sh])], axes=[1], keepdims=1)  # [1,1,30,30]
        nbg_sh = _shift(g, nbg, dr, dc)
        change = g.nd("Mul", [g.nd("Mul", [g.nd("Sub", [g.one, same]), nbg]), nbg_sh])
        return g.nd("ReduceSum", [change], axes=[2, 3], keepdims=1)
    hcost = cost(0, 1)                                                 # horizontal neighbour changes
    vcost = cost(1, 0)
    orient_h = g.nd("Cast", [g.nd("Less", [hcost, vcost])], to=F)      # [1,1,1,1]
    nz_h = _band_project(g, ch0, nbg, "h")
    nz_v = _band_project(g, ch0, nbg, "v")
    out_h = g.nd("Add", [g.nd("Mul", ["input", g.nd("Sub", [g.one, nz_h])]),
                         g.nd("Mul", [nz_h, _onehot(g, 0)])])
    out_v = g.nd("Add", [g.nd("Mul", ["input", g.nd("Sub", [g.one, nz_v])]),
                         g.nd("Mul", [nz_v, _onehot(g, 0)])])
    g.nd("Add", [g.nd("Mul", [out_h, orient_h]),
                 g.nd("Mul", [out_v, g.nd("Sub", [g.one, orient_h])])], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics 1:1)                              #
# --------------------------------------------------------------------------- #
def r21(a):
    h, w = a.shape; bg = a[0, 0]
    hl = sum(1 for r in range(h) if not (a[r] == bg).any() and (a[r] != bg).any())
    vl = sum(1 for c in range(w) if not (a[:, c] == bg).any() and (a[:, c] != bg).any())
    return np.full((hl + 1, vl + 1), bg, int)


def r39(a):
    ys, xs = np.where(a != 0)
    if ys.size == 0: return None
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    Hh, Ww = r1 - r0 + 1, c1 - c0 + 1
    if Hh % 2 or Ww % 2: return None
    return a[r0:r0 + Hh // 2, c0:c0 + Ww // 2].copy()


def r29(a):
    content = np.zeros(a.shape, bool)
    for c in range(1, 10):
        ys, xs = np.where(a == c)
        if ys.size == 0: continue
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh < 3 or bw < 3: continue
        perim = 2 * bh + 2 * bw - 4
        interior = int(((ys > r0) & (ys < r1) & (xs > c0) & (xs < c1)).sum())
        if ys.size == perim and interior == 0:
            content[ys, xs] = True
    yy, xx = np.where(content)
    if yy.size == 0: return None
    r0, r1, c0, c1 = yy.min(), yy.max(), xx.min(), xx.max()
    if r1 - r0 < 2 or c1 - c0 < 2: return None
    return a[r0 + 1:r1, c0 + 1:c1].copy()


def r68(a):
    cnt = {c: int((a == c).sum()) for c in range(1, 10)}
    uniq = [c for c in cnt if cnt[c] == 1]
    if len(uniq) != 1: return None
    c = uniq[0]; ys, xs = np.where(a == c); r, cc = ys[0], xs[0]
    o = np.zeros_like(a)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nn = r + dr, cc + dc
            if 0 <= nr < a.shape[0] and 0 <= nn < a.shape[1]: o[nr, nn] = 2
    o[r, cc] = c
    return o


def r167(a):
    if a.shape != (3, 3): return None
    k = len(set(a.flatten().tolist()))
    o = np.zeros((3, 3), int)
    if k == 1: o[0, :] = 5
    elif k == 2:
        for i in range(3): o[i, i] = 5
    elif k == 3:
        for i in range(3): o[i, 2 - i] = 5
    else: return None
    return o


def r195(a):
    ys, xs = np.where(a != 0)
    if ys.size == 0: return None
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    if r1 - r0 + 1 != 9 or c1 - c0 + 1 != 9: return None
    sub = a[r0:r1 + 1, c0:c1 + 1]
    S = sub[1::3, 1::3]
    if S.shape != (3, 3): return None
    cols = set(S[S != 0].tolist())
    if len(cols) != 1: return None
    col = cols.pop(); S01 = (S != 0).astype(int)
    o = np.zeros((9, 9), int)
    for i in range(3):
        for j in range(3):
            if S01[i, j]:
                for x in range(3):
                    for y in range(3):
                        if S01[x, y]: o[i * 3 + x, j * 3 + y] = col
    return o


def r246(a):
    p2 = np.where(a == 2); p3 = np.where(a == 3)
    if len(p2[0]) != 1 or len(p3[0]) != 1: return None
    rA, cA = p2[0][0], p2[1][0]; rB, cB = p3[0][0], p3[1][0]
    o = a.copy()
    lo, hi = sorted((cA, cB))
    for c in range(lo, hi + 1):
        if c != cA: o[rA, c] = 8
    lo, hi = sorted((rA, rB))
    for r in range(lo, hi + 1):
        if r != rB: o[r, cB] = 8
    return o


def r291(a):
    best = None; bd = 0
    for c in range(1, 10):
        ys, xs = np.where(a == c)
        if ys.size == 0: continue
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        d = (r1 - r0 + 1) * (c1 - c0 + 1) - ys.size
        if d > bd: bd = d; best = c
    if best is None: return None
    return np.array([[best]], int)


def r341(a):
    cols = [c for c in range(1, 10) if (a == c).any()]
    if len(cols) != 2: return None
    bb = {}
    for c in cols:
        ys, xs = np.where(a == c); bb[c] = (ys.min(), ys.max(), xs.min(), xs.max())
    (r1a, r1b, c1a, c1b) = bb[cols[0]]; (r2a, r2b, c2a, c2b) = bb[cols[1]]
    o = a.copy()
    gcl, gch = min(c1b, c2b) + 1, max(c1a, c2a) - 1
    orl, orh = max(r1a, r2a) + 1, min(r1b, r2b) - 1
    for r in range(orl, orh + 1):
        for c in range(gcl, gch + 1): o[r, c] = 8
    grl, grh = min(r1b, r2b) + 1, max(r1a, r2a) - 1
    ocl, och = max(c1a, c2a) + 1, min(c1b, c2b) - 1
    for r in range(grl, grh + 1):
        for c in range(ocl, och + 1): o[r, c] = 8
    return o


def r359(a):
    h, w = a.shape
    ro = np.zeros_like(a); co = np.zeros_like(a)
    for r in range(h):
        v, c = np.unique(a[r], return_counts=True); ro[r, :] = v[np.argmax(c)]
    for c in range(w):
        v, ct = np.unique(a[:, c], return_counts=True); co[:, c] = v[np.argmax(ct)]
    return ro if (ro == a).sum() >= (co == a).sum() else co


def r399(a):
    h, w = a.shape
    N = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if a[r, c] == 2 and a[r, c + 1] == 2 and a[r + 1, c] == 2 and a[r + 1, c + 1] == 2:
                N += 1
    if N == 0: return None
    rank = {(0, 0): 0, (0, 2): 1, (1, 1): 2, (2, 0): 3, (2, 2): 4}
    o = np.zeros((3, 3), int)
    for (rr, cc), k in rank.items():
        if k < N: o[rr, cc] = 1
    return o


def _cost(a, dr, dc):
    h, w = a.shape; n = 0
    for r in range(h):
        for c in range(w):
            pr, pc = r - dr, c - dc
            if 0 <= pr < h and 0 <= pc < w and a[r, c] != 0 and a[pr, pc] != 0 and a[r, c] != a[pr, pc]:
                n += 1
    return n


def r178(a):
    h, w = a.shape

    def domcolor(vec):
        cnt = np.bincount(vec, minlength=10).astype(float); cnt[0] -= 1e9
        return int(cnt.argmax())
    if _cost(a, 0, 1) < _cost(a, 1, 0):           # rows uniform -> column
        seq = [domcolor(a[r]) for r in range(h)]
        ded = [seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
        return np.array([[x] for x in ded], int)
    seq = [domcolor(a[:, c]) for c in range(w)]
    ded = [seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
    return np.array([ded], int)


def r62(a):
    h, w = a.shape
    cols = [c for c in range(1, 10) if (a == c).any()]
    if len(cols) != 2: return None
    cnt = {c: int((a == c).sum()) for c in cols}
    marker = min(cols, key=lambda c: cnt[c]); shape = [c for c in cols if c != marker][0]
    sy, sx = np.where(a == shape); my, mx = np.where(a == marker)
    out = np.full((h, w), 3, int)
    if abs(mx.mean() - sx.mean()) >= abs(my.mean() - sy.mean()):
        axis2 = (sx.max() + mx.min()) if mx.mean() > sx.mean() else (mx.max() + sx.min())
        for (r, c) in zip(sy, sx):
            out[r, c] = shape
            cc = axis2 - c
            if 0 <= cc < w: out[r, cc] = shape
    else:
        axis2 = (sy.max() + my.min()) if my.mean() > sy.mean() else (my.max() + sy.min())
        for (r, c) in zip(sy, sx):
            out[r, c] = shape
            rr = axis2 - r
            if 0 <= rr < h: out[rr, c] = shape
    return out


def r301(a):
    h, w = a.shape
    anchor = a[h - 1, 0]
    if anchor == 0 or not (a[h - 1] == anchor).all(): return None
    bars = [c for c in range(1, 10) if c != anchor and (a == c).any()]
    if not bars: return None
    length = {c: int((a == c).sum()) for c in bars}
    if len(set(length.values())) != len(bars): return None
    nb = len(bars)
    out = np.zeros((h, w), int); out[h - 1, :] = anchor
    for c in bars:
        rank = sum(1 for d in bars if length[d] < length[c])
        out[h - 1 - nb + rank, w - length[c]:w] = c
    return out


def r51(a):
    h, w = a.shape
    cnt = {c: int((a == c).sum()) for c in range(1, 10)}
    pres = [c for c in cnt if cnt[c] > 0]
    marks = [c for c in pres if cnt[c] == 1]
    if len(marks) != 1: return None
    B = marks[0]; others = [c for c in pres if c != B]
    if len(others) != 1: return None
    A = others[0]
    mr, mc = [int(x[0]) for x in np.where(a == B)]
    ys, xs = np.where(a == A)
    dr, dc = ys.mean() - mr, xs.mean() - mc
    out = a.copy()
    if abs(dc) >= abs(dr):
        if dc > 0:
            for c in range(xs.max() + 1, w): out[mr, c] = B
        else:
            for c in range(0, xs.min()): out[mr, c] = B
    else:
        if dr > 0:
            for r in range(ys.max() + 1, h): out[r, mc] = B
        else:
            for r in range(0, ys.min()): out[r, mc] = B
    return out


def r146(a):
    if a.shape != (9, 3): return None
    blocks = [a[3 * k:3 * k + 3, :] for k in range(3)]
    asym = [k for k in range(3) if not np.array_equal(blocks[k], blocks[k].T)]
    if len(asym) != 1: return None
    return blocks[asym[0]].copy()


def r221(a):
    if a.shape != (3, 3): return None
    nz = [(r, c) for r in range(3) for c in range(3) if a[r, c] != 0]
    K = len(nz)
    if K == 0: return None
    cset = set(a[a != 0].tolist())
    if len(cset) != 1: return None
    N = 9 - K
    if N <= 0 or K > N * N: return None
    out = np.zeros((3 * N, 3 * N), int)
    for idx in range(K):
        bi, bj = idx // N, idx % N
        out[3 * bi:3 * bi + 3, 3 * bj:3 * bj + 3] = a
    return out


def r115(a):
    colors = [c for c in range(1, 10) if (a == c).any()]
    if not colors: return None
    mr = {}; mc = {}
    for c in colors:
        ys, xs = np.where(a == c); mr[c] = ys.mean(); mc[c] = xs.mean()
    cspread = max(mc.values()) - min(mc.values())
    rspread = max(mr.values()) - min(mr.values())
    horiz = not (cspread < rspread)

    def rank(c, m): return sum(1 for d in colors if m[d] < m[c])
    if horiz:
        o = np.zeros((1, len(colors)), int)
        for c in colors: o[0, rank(c, mc)] = c
        return o
    o = np.zeros((len(colors), 1), int)
    for c in colors: o[rank(c, mr), 0] = c
    return o


def r202(a):
    h, w = a.shape

    def cost(dr, dc):
        n = 0
        for r in range(h):
            for c in range(w):
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and a[r, c] != 0 and a[pr, pc] != 0 and a[r, c] != a[pr, pc]:
                    n += 1
        return n

    horiz = cost(0, 1) < cost(1, 0)
    out = a.copy()
    if horiz:
        for k in range(1, 10):
            rows = [r for r in range(h) if (a[r] == k).any()]
            if not rows: continue
            for c in range(w):
                if any(a[r, c] == 0 for r in rows):
                    for r in rows: out[r, c] = 0
    else:
        for k in range(1, 10):
            cols = [c for c in range(w) if (a[:, c] == k).any()]
            if not cols: continue
            for r in range(h):
                if any(a[r, c] == 0 for c in cols):
                    for c in cols: out[r, c] = 0
    return out


def r375(a):
    h, w = a.shape
    if h != w: return None
    v, c = np.unique(a, return_counts=True)
    nz = [(x, y) for x, y in zip(v, c) if x != 0]
    if not nz: return None
    col = max(nz, key=lambda t: t[1])[0]
    o = np.full((h, w), col, int)
    for i in range(h): o[i, i] = 0; o[i, h - 1 - i] = 0
    return o


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
_RULES = [
    ("crk2_4_t21", r21, build_21),
    ("crk2_4_t29", r29, build_29),
    ("crk2_4_t39", r39, build_39),
    ("crk2_4_t68", r68, build_68),
    ("crk2_4_t167", r167, build_167),
    ("crk2_4_t195", r195, build_195),
    ("crk2_4_t246", r246, build_246),
    ("crk2_4_t291", r291, build_291),
    ("crk2_4_t341", r341, build_341),
    ("crk2_4_t359", r359, build_359),
    ("crk2_4_t375", r375, build_375),
    ("crk2_4_t399", r399, build_399),
    ("crk2_4_t202", r202, build_202),
    ("crk2_4_t178", r178, build_178),
    ("crk2_4_t115", r115, build_115),
    ("crk2_4_t146", r146, build_146),
    ("crk2_4_t221", r221, build_221),
    ("crk2_4_t51", r51, build_51),
    ("crk2_4_t301", r301, build_301),
    ("crk2_4_t62", r62, build_62),
]


def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0: continue
            if max(a.shape) > 30 or max(b.shape) > 30: continue
            out.append((a, b))
    return out


def _matches(prs, fn):
    for a, b in prs:
        try:
            o = fn(a)
        except Exception:
            return False
        if o is None: return False
        o = np.array(o)
        if o.shape != b.shape or not np.array_equal(o, b): return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs: return []
    if all(np.array_equal(a, b) for a, b in prs): return []
    out = []
    for name, ref, builder in _RULES:
        if _matches(prs, ref):
            try:
                g = _G(); m = builder(g)
                onnx.checker.check_model(m, full_check=True)
            except Exception:
                continue
            out.append((name, m))
    return out
