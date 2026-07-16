"""family_crk2_2 -- a grab-bag of exact ARC->ONNX solvers (opset-10, static shapes).

Each rule is detected STRUCTURALLY and a numpy mirror of the ONNX semantics is
validated EXACTLY on every provided pair (train+test+arc-gen) before a candidate
is emitted, so wrong hypotheses are dropped before scoring.

Rules implemented (all use the "computed selection matrix + MatMul" data-dependent
positioning trick where positions vary):

  q4odd   (T207)  5x5 grid = 2x2 quadrants separated by a zero cross; output the
                  quadrant whose 2x2 pattern differs from the other three.
  mk8     (T121)  a single colour-8 marker sits at the centre of a 3x3 object of
                  one colour C; output that 3x3 region with 8 recoloured to C.
  upcnt   (T289)  upscale the whole grid by k = number of distinct non-bg colours
                  (data-dependent integer scale via a computed expansion matrix).
  blkodd  (T263)  the grid tiles into 3x3 blocks (a strip / grid of glyphs); output
                  the one block whose binary footprint differs from the consensus.
  compress(T218)  remove all-background border + collapse adjacent duplicate rows
                  and columns (block-grid -> block-colour grid) via cumsum-select.
  plusmid (T371)  exactly two colour-1 markers; stamp a colour-3 plus at their
                  integer midpoint, keeping the markers.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
H, W, C = HEIGHT, WIDTH, CHANNELS
_BIG = 1.0e9


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


# --------------------------------------------------------------------------- #
# shared constants                                                            #
# --------------------------------------------------------------------------- #
def _consts(g):
    g.rowidx = g.f([1, 1, H, 1], list(range(H)))     # axis-2 index  [1,1,30,1]
    g.colidx = g.f([1, 1, 1, W], list(range(W)))     # axis-3 index  [1,1,1,30]
    g.half = g.f([1, 1, 1, 1], [0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])


def _slice_ch(g, x, c0, c1):
    return g.nd("Slice", [x, g.i64([c0]), g.i64([c1]), g.i64([1])])


def _nonbg(g):
    """[1,1,30,30] mask = 1 at non-background real cells."""
    s = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    ch0 = _slice_ch(g, "input", 0, 1)
    return g.nd("Sub", [s, ch0])


def _col_sel(g, off, wbox):
    """Scol[in,out]=1 iff in==out+off and out<wbox  (in=axis2, out=axis3).
    Used as MatMul(src, Scol) to shift/select columns."""
    diff = g.nd("Sub", [g.nd("Add", [g.colidx, off]), g.rowidx])   # (out+off)-in
    match = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), g.half])], to=F)
    trunc = g.nd("Cast", [g.nd("Less", [g.colidx, wbox])], to=F)    # out<wbox  [1,1,1,30]
    return g.nd("Mul", [match, trunc])


def _row_sel(g, off, hbox):
    """Srow[out,in]=1 iff in==out+off and out<hbox  (out=axis2, in=axis3).
    Used as MatMul(Srow, src) to shift/select rows."""
    diff = g.nd("Sub", [g.colidx, g.nd("Add", [g.rowidx, off])])   # in-(out+off)
    match = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), g.half])], to=F)
    trunc = g.nd("Cast", [g.nd("Less", [g.rowidx, hbox])], to=F)    # out<hbox  [1,1,30,1]
    return g.nd("Mul", [match, trunc])


# ===========================================================================
# T263 / T121 style: extract a fixed-size region anchored at (minrow,mincol)
# ===========================================================================
def _extract_region(g, src, minrow, mincol, hbox, wbox):
    """output[outr,outc] = src[outr+minrow, outc+mincol], outr<hbox, outc<wbox."""
    Scol = _col_sel(g, mincol, wbox)                          # [1,1,30(in),30(out)]
    shift1 = g.nd("MatMul", [src, Scol])                      # [1,10,30,30]
    Srow = _row_sel(g, minrow, hbox)                          # [1,1,30(out),30(in)]
    return g.nd("MatMul", [Srow, shift1])


# --------------------------------------------------------------------------- #
# T207 -- odd-one-out quadrant of a 5x5 grid                                  #
# --------------------------------------------------------------------------- #
def build_q4odd():
    g = _G()
    quads = []
    for (r0, c0) in [(0, 0), (0, 3), (3, 0), (3, 3)]:
        q = g.nd("Slice", ["input", g.i64([r0, c0]), g.i64([r0 + 2, c0 + 2]),
                           g.i64([2, 3])])                     # [1,10,2,2]
        quads.append(q)
    # pairwise L1 distances
    p = {}
    for i in range(4):
        for j in range(i + 1, 4):
            d = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [quads[i], quads[j]])])],
                     axes=[1, 2, 3], keepdims=1)               # [1,1,1,1]
            p[(i, j)] = d

    def tot(i):
        terms = [p[tuple(sorted((i, j)))] for j in range(4) if j != i]
        s = terms[0]
        for t in terms[1:]:
            s = g.nd("Add", [s, t])
        return s

    # distance with reading-order tie-break (lower index wins), unique max guaranteed
    scaled = []
    for i in range(4):
        s = g.nd("Mul", [tot(i), g.f([1, 1, 1, 1], [10.0])])
        s = g.nd("Add", [s, g.f([1, 1, 1, 1], [float(3 - i)])])
        scaled.append(s)
    mx = scaled[0]
    for s in scaled[1:]:
        mx = g.nd("Max", [mx, s])
    thr = g.nd("Sub", [mx, g.f([1, 1, 1, 1], [0.5])])
    sel = None
    for i in range(4):
        gate = g.nd("Cast", [g.nd("Greater", [scaled[i], thr])], to=F)   # [1,1,1,1]
        part = g.nd("Mul", [quads[i], gate])                            # [1,10,2,2]
        sel = part if sel is None else g.nd("Add", [sel, part])
    g.nd("Pad", [sel], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, H - 2, W - 2])
    return _model(g)


# --------------------------------------------------------------------------- #
# T121 -- 3x3 region around the single colour-8 marker, recolour 8 -> C        #
# --------------------------------------------------------------------------- #
def build_mk8():
    g = _G()
    _consts(g)
    ch8 = _slice_ch(g, "input", 8, 9)                # [1,1,30,30]
    r0 = g.nd("ReduceSum", [g.nd("Mul", [ch8, g.rowidx])], axes=[2, 3], keepdims=1)
    c0 = g.nd("ReduceSum", [g.nd("Mul", [ch8, g.colidx])], axes=[2, 3], keepdims=1)
    minrow = g.nd("Sub", [r0, g.one])
    mincol = g.nd("Sub", [c0, g.one])
    region = _extract_region(g, "input", minrow, mincol, g.f([1, 1, 1, 1], [3.0]),
                             g.f([1, 1, 1, 1], [3.0]))         # [1,10,30,30]
    # colour C = argmax channel count excluding 0 and 8
    counts = g.nd("ReduceSum", [region], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    bias = g.f([1, C, 1, 1], [-_BIG] + [0.0] * 6 + [0.0, -_BIG, 0.0])  # ch0,ch8 = -big
    sel = g.nd("Add", [counts, bias])
    cidx = g.nd("ArgMax", [sel], axis=1, keepdims=1)               # int64 [1,1,1,1]
    idxgrid = g.i64(list(range(C)), dims=[1, C, 1, 1])
    conehot = g.nd("Cast", [g.nd("Equal", [cidx, idxgrid])], to=F)  # [1,10,1,1]
    # merge channel-8 content into channel C, zero channel 8
    not8 = g.f([1, C, 1, 1], [1.0] * 8 + [0.0, 1.0])
    region_no8 = g.nd("Mul", [region, not8])
    reg_ch8 = _slice_ch(g, region, 8, 9)                          # [1,1,30,30]
    add_c = g.nd("Mul", [reg_ch8, conehot])                       # [1,10,30,30]
    g.nd("Add", [region_no8, add_c], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# T289 -- upscale by k = number of distinct non-bg colours                     #
# --------------------------------------------------------------------------- #
def build_upcnt():
    g = _G()
    _consts(g)
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    present = g.nd("Cast", [g.nd("Greater", [counts, g.half])], to=F)
    notbg = g.f([1, C, 1, 1], [0.0] + [1.0] * 9)
    k = g.nd("ReduceSum", [g.nd("Mul", [present, notbg])], axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    km = g.nd("Sub", [k, g.half])      # k - 0.5
    neghalf = g.f([1, 1, 1, 1], [-0.5])

    rowidx_T = g.f([1, 1, 1, W], list(range(W)))   # axis-3 copy of row values

    def expand(outg, ing):
        # E[out,in]=1 iff 0 <= out - in*k < k  (i.e. in == out//k)
        term = g.nd("Sub", [outg, g.nd("Mul", [ing, k])])   # out - in*k
        ge = g.nd("Cast", [g.nd("Greater", [term, neghalf])], to=F)
        lt = g.nd("Cast", [g.nd("Less", [term, km])], to=F)
        return g.nd("Mul", [ge, lt])

    # Ucol[inc,outc]=1 iff inc==outc//k :  out=outc(axis3), in=inc(axis2)
    Ucol = expand(g.colidx, g.rowidx)                  # [1,1,30(inc),30(outc)]
    cols = g.nd("MatMul", ["input", Ucol])             # upscale columns
    # Urow[outr,inr]=1 iff inr==outr//k :  out=outr(axis2), in=inr(axis3)
    Urow = expand(g.rowidx, rowidx_T)                  # [1,1,30(outr),30(inr)]
    g.nd("MatMul", [Urow, cols], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# T263 -- odd-one-out 3x3 block (binary footprint) among the strip of blocks   #
# --------------------------------------------------------------------------- #
def build_blkodd():
    g = _G()
    _consts(g)
    M = _nonbg(g)                                              # [1,1,30,30]
    ones3 = g.f([1, 1, 3, 3], [1.0] * 9)
    blockcount = g.nd("Conv", [M, ones3], kernel_shape=[3, 3], strides=[3, 3],
                      pads=[0, 0, 0, 0])                       # [1,1,10,10]
    valid = g.nd("Cast", [g.nd("Greater", [blockcount, g.half])], to=F)
    nreal = g.nd("ReduceSum", [valid], axes=[2, 3], keepdims=1)   # [1,1,1,1]
    halfn = g.nd("Mul", [nreal, g.half])
    # consensus per (u,v): blocksum over blocks > N/2
    M6 = g.nd("Reshape", [M, g.i64([1, 1, 10, 3, 10, 3])])
    blocksum = g.nd("ReduceSum", [M6], axes=[2, 4], keepdims=1)   # [1,1,1,3,1,3]
    halfn6 = g.nd("Reshape", [halfn, g.i64([1, 1, 1, 1, 1, 1])])
    consensus = g.nd("Cast", [g.nd("Greater", [blocksum, halfn6])], to=F)
    cons_t = g.nd("Tile", [consensus, g.i64([1, 1, 10, 1, 10, 1])])  # [1,1,10,3,10,3]
    cons2d = g.nd("Reshape", [cons_t, g.i64([1, 1, H, W])])          # [1,1,30,30]
    diff = g.nd("Abs", [g.nd("Sub", [M, cons2d])])
    blockdist = g.nd("Conv", [diff, ones3], kernel_shape=[3, 3], strides=[3, 3],
                     pads=[0, 0, 0, 0])                              # [1,1,10,10]
    masked = g.nd("Mul", [blockdist, valid])
    mx = g.nd("ReduceMax", [masked], axes=[2, 3], keepdims=1)
    gate = g.nd("Cast", [g.nd("Greater", [masked, g.nd("Sub", [mx, g.half])])], to=F)
    # block index grids (block coords *3 = pixel offset)
    brow = g.f([1, 1, 10, 10], [3 * (i // 10) for i in range(100)])
    bcol = g.f([1, 1, 10, 10], [3 * (i % 10) for i in range(100)])
    minrow = g.nd("ReduceSum", [g.nd("Mul", [gate, brow])], axes=[2, 3], keepdims=1)
    mincol = g.nd("ReduceSum", [g.nd("Mul", [gate, bcol])], axes=[2, 3], keepdims=1)
    region = _extract_region(g, "input", minrow, mincol, g.f([1, 1, 1, 1], [3.0]),
                             g.f([1, 1, 1, 1], [3.0]))
    g.nd("Identity", [region], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# T218 -- crop bg border + collapse adjacent duplicate rows & cols             #
# --------------------------------------------------------------------------- #
def build_compress():
    g = _G()
    _consts(g)
    # lower/upper triangular helpers
    Llow = g.f([1, 1, H, H], [[1.0 if b <= a else 0.0 for b in range(H)] for a in range(H)])
    Lup = g.f([1, 1, W, W], [[1.0 if a <= b else 0.0 for b in range(W)] for a in range(W)])

    def keepvec_rows(T):
        rmask = g.nd("ReduceSum", [T], axes=[1], keepdims=1)          # [1,1,30,30]
        ch0 = _slice_ch(g, T, 0, 1)
        nb = g.nd("Sub", [rmask, ch0])                               # nonbg
        active = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceMax", [nb], axes=[3], keepdims=1), g.half])], to=F)  # [1,1,30,1]
        # shift down by 1 row
        pad = g.nd("Pad", [T], mode="constant", value=0.0, pads=[0, 0, 1, 0, 0, 0, 0, 0])
        sd = g.nd("Slice", [pad, g.i64([0]), g.i64([H]), g.i64([2])])
        d = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [T, sd])])], axes=[1, 3], keepdims=1)
        diff = g.nd("Cast", [g.nd("Greater", [d, g.half])], to=F)    # [1,1,30,1]
        return g.nd("Mul", [active, diff])                           # keepRow [1,1,30,1]

    def keepvec_cols(T):
        rmask = g.nd("ReduceSum", [T], axes=[1], keepdims=1)
        ch0 = _slice_ch(g, T, 0, 1)
        nb = g.nd("Sub", [rmask, ch0])
        active = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceMax", [nb], axes=[2], keepdims=1), g.half])], to=F)  # [1,1,1,30]
        pad = g.nd("Pad", [T], mode="constant", value=0.0, pads=[0, 0, 0, 1, 0, 0, 0, 0])
        sr = g.nd("Slice", [pad, g.i64([0]), g.i64([W]), g.i64([3])])
        d = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [T, sr])])], axes=[1, 2], keepdims=1)
        diff = g.nd("Cast", [g.nd("Greater", [d, g.half])], to=F)    # [1,1,1,30]
        return g.nd("Mul", [active, diff])                           # keepCol [1,1,1,30]

    # rows
    keepRow = keepvec_rows("input")                                  # [1,1,30,1]
    cumR = g.nd("MatMul", [Llow, keepRow])                           # [1,1,30,1]
    rankR = g.nd("Sub", [cumR, g.one])                               # [1,1,30,1]
    rankRT = g.nd("Transpose", [rankR], perm=[0, 1, 3, 2])           # [1,1,1,30]
    keepRowT = g.nd("Transpose", [keepRow], perm=[0, 1, 3, 2])       # [1,1,1,30]
    matchR = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [g.rowidx, rankRT])]), g.half])], to=F)  # [1,1,30,30]
    Srow = g.nd("Mul", [matchR, keepRowT])                          # [1,1,30(out),30(in)]
    step = g.nd("MatMul", [Srow, "input"])                          # rows compacted
    # cols (on row-compacted)
    keepCol = keepvec_cols(step)                                     # [1,1,1,30]
    cumC = g.nd("MatMul", [keepCol, Lup])                            # [1,1,1,30]
    rankC = g.nd("Sub", [cumC, g.one])                              # [1,1,1,30]
    rankCT = g.nd("Transpose", [rankC], perm=[0, 1, 3, 2])          # [1,1,30,1]
    keepColT = g.nd("Transpose", [keepCol], perm=[0, 1, 3, 2])      # [1,1,30,1]
    matchC = g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [rankCT, g.colidx])]), g.half])], to=F)  # [1,1,30,30]
    Scol = g.nd("Mul", [matchC, keepColT])                          # [1,1,30(in),30(out)]
    g.nd("MatMul", [step, Scol], "output")                          # cols compacted
    return _model(g)


# --------------------------------------------------------------------------- #
# T371 -- stamp a colour-3 plus at the midpoint of two colour-1 markers        #
# --------------------------------------------------------------------------- #
def build_plusmid():
    g = _G()
    _consts(g)
    ch1 = _slice_ch(g, "input", 1, 2)                  # [1,1,30,30]
    sr = g.nd("ReduceSum", [g.nd("Mul", [ch1, g.rowidx])], axes=[2, 3], keepdims=1)
    sc = g.nd("ReduceSum", [g.nd("Mul", [ch1, g.colidx])], axes=[2, 3], keepdims=1)
    mr = g.nd("Mul", [sr, g.half])                     # midpoint row [1,1,1,1]
    mc = g.nd("Mul", [sc, g.half])                     # midpoint col
    # distance of each cell from centre
    dr = g.nd("Abs", [g.nd("Sub", [g.rowidx, mr])])    # [1,1,30,1]
    dc = g.nd("Abs", [g.nd("Sub", [g.colidx, mc])])    # [1,1,1,30]
    onehalf = g.f([1, 1, 1, 1], [1.5])
    # horizontal arm: dr<0.5 & dc<1.5 ; vertical arm: dc<0.5 & dr<1.5
    drs = g.nd("Cast", [g.nd("Less", [dr, g.half])], to=F)   # [1,1,30,1]
    drm = g.nd("Cast", [g.nd("Less", [dr, onehalf])], to=F)  # [1,1,30,1]
    dcs = g.nd("Cast", [g.nd("Less", [dc, g.half])], to=F)   # [1,1,1,30]
    dcm = g.nd("Cast", [g.nd("Less", [dc, onehalf])], to=F)  # [1,1,1,30]
    horiz = g.nd("Mul", [drs, dcm])                    # [1,1,30,30]
    vert = g.nd("Mul", [dcs, drm])                     # [1,1,30,30]
    plus = g.nd("Cast", [g.nd("Greater",
            [g.nd("Add", [horiz, vert]), g.half])], to=F)    # [1,1,30,30] in {0,1}
    # output = input with channel3 |= plus, channel0 cleared where plus
    e3 = g.f([1, C, 1, 1], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e0 = g.f([1, C, 1, 1], [1.0] + [0.0] * 9)
    add3 = g.nd("Mul", [plus, e3])                     # [1,10,30,30] channel3=plus
    # clear bg (channel0) where plus is on, so thresholding stays exact
    clear0 = g.nd("Mul", [plus, e0])                   # channel0=plus
    base = g.nd("Sub", ["input", clear0])              # remove bg under plus
    g.nd("Add", [base, add3], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# T84 -- vertical bar (col0, colour C) -> + anti-diagonal(2) + bottom-row(4)   #
# --------------------------------------------------------------------------- #
def build_diagbar():
    g = _G()
    _consts(g)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    ch0 = _slice_ch(g, "input", 0, 1)
    nb = g.nd("Sub", [realmask, ch0])                              # bar mask [1,1,30,30]
    N = g.nd("ReduceSum", [nb], axes=[2, 3], keepdims=1)          # grid size [1,1,1,1]
    Nm1 = g.nd("Sub", [N, g.one])
    Nmhalf = g.nd("Sub", [N, g.half])                            # N-0.5
    # colour C of the bar
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    bias = g.f([1, C, 1, 1], [-_BIG] + [0.0] * 9)
    cidx = g.nd("ArgMax", [g.nd("Add", [counts, bias])], axis=1, keepdims=1)
    idxgrid = g.i64(list(range(C)), dims=[1, C, 1, 1])
    conehot = g.nd("Cast", [g.nd("Equal", [cidx, idxgrid])], to=F)   # [1,10,1,1]
    rc = g.nd("Add", [g.rowidx, g.colidx])                       # r+c  [1,1,30,30]
    colpos = g.nd("Cast", [g.nd("Greater", [g.colidx, g.half])], to=F)   # col>=1 [1,1,1,30]
    # anti-diagonal: r+c==N-1 and col>=1
    Adiag = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rc, Nm1])]), g.half])], to=F)
    A2d = g.nd("Mul", [Adiag, colpos])                          # [1,1,30,30]
    # bottom row: row==N-1 and 1<=col<N
    rowbot = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, Nm1])]), g.half])], to=F)
    colin = g.nd("Cast", [g.nd("Less", [g.colidx, Nmhalf])], to=F)
    B2d = g.nd("Mul", [g.nd("Mul", [rowbot, colpos]), colin])    # [1,1,30,30]
    # background = realmask region minus bar/A/B
    occ = g.nd("Add", [g.nd("Add", [nb, A2d]), B2d])
    bg2d = g.nd("Sub", [realmask, occ])
    e2 = g.f([1, C, 1, 1], [1.0 if i == 2 else 0.0 for i in range(C)])
    e4 = g.f([1, C, 1, 1], [1.0 if i == 4 else 0.0 for i in range(C)])
    e0 = g.f([1, C, 1, 1], [1.0 if i == 0 else 0.0 for i in range(C)])
    o = g.nd("Mul", [nb, conehot])
    o = g.nd("Add", [o, g.nd("Mul", [A2d, e2])])
    o = g.nd("Add", [o, g.nd("Mul", [B2d, e4])])
    g.nd("Add", [o, g.nd("Mul", [bg2d, e0])], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# T253 -- four L-trominoes placed into a 4x4 grid by their elbow direction      #
# --------------------------------------------------------------------------- #
def _shift(g, x, dr, dc):
    """result[r,c] = x[r-dr, c-dc] (zero fill)."""
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0, pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = [max(-dr, 0), max(-dc, 0)]
    return g.nd("Slice", [p, g.i64(st), g.i64([st[0] + H, st[1] + W]), g.i64([2, 3])])


def _col_sel_r(g, off, lo, hi):
    """Scol[in,out]=1 iff in==out+off and lo<=out<hi (in=axis2, out=axis3)."""
    diff = g.nd("Sub", [g.nd("Add", [g.colidx, off]), g.rowidx])
    match = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), g.half])], to=F)
    ge = g.nd("Cast", [g.nd("Greater", [g.colidx, g.f([1, 1, 1, 1], [lo - 0.5])])], to=F)
    lt = g.nd("Cast", [g.nd("Less", [g.colidx, g.f([1, 1, 1, 1], [hi - 0.5])])], to=F)
    return g.nd("Mul", [g.nd("Mul", [match, ge]), lt])


def _row_sel_r(g, off, lo, hi):
    """Srow[out,in]=1 iff in==out+off and lo<=out<hi (out=axis2, in=axis3)."""
    diff = g.nd("Sub", [g.colidx, g.nd("Add", [g.rowidx, off])])
    match = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), g.half])], to=F)
    ge = g.nd("Cast", [g.nd("Greater", [g.rowidx, g.f([1, 1, 1, 1], [lo - 0.5])])], to=F)
    lt = g.nd("Cast", [g.nd("Less", [g.rowidx, g.f([1, 1, 1, 1], [hi - 0.5])])], to=F)
    return g.nd("Mul", [g.nd("Mul", [match, ge]), lt])


def build_lplace():
    g = _G()
    _consts(g)
    M = _nonbg(g)                                                   # [1,1,30,30]
    right = _shift(g, M, 0, -1)
    left = _shift(g, M, 0, 1)
    down = _shift(g, M, -1, 0)
    up = _shift(g, M, 1, 0)
    # (mask, drow_to_blocktop, dcol_to_blockleft, quad_row, quad_col)
    orient = [
        (g.nd("Mul", [g.nd("Mul", [M, right]), down]), 0, 0, 0, 0),
        (g.nd("Mul", [g.nd("Mul", [M, left]), down]), 0, -1, 0, 2),
        (g.nd("Mul", [g.nd("Mul", [M, right]), up]), -1, 0, 2, 0),
        (g.nd("Mul", [g.nd("Mul", [M, left]), up]), -1, -1, 2, 2),
    ]
    parts = []
    for mask, dbr, dbc, qr, qc in orient:
        er = g.nd("ReduceSum", [g.nd("Mul", [mask, g.rowidx])], axes=[2, 3], keepdims=1)
        ec = g.nd("ReduceSum", [g.nd("Mul", [mask, g.colidx])], axes=[2, 3], keepdims=1)
        # rowoff = (er+dbr) - qr ; coloff = (ec+dbc) - qc
        rowoff = g.nd("Sub", [g.nd("Add", [er, g.f([1, 1, 1, 1], [float(dbr)])]),
                              g.f([1, 1, 1, 1], [float(qr)])])
        coloff = g.nd("Sub", [g.nd("Add", [ec, g.f([1, 1, 1, 1], [float(dbc)])]),
                              g.f([1, 1, 1, 1], [float(qc)])])
        Scol = _col_sel_r(g, coloff, qc, qc + 2)
        Srow = _row_sel_r(g, rowoff, qr, qr + 2)
        shift1 = g.nd("MatMul", ["input", Scol])
        parts.append(g.nd("MatMul", [Srow, shift1]))
    o = parts[0]
    for p in parts[1:]:
        o = g.nd("Add", [o, p])
    g.nd("Identity", [o], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# T244 -- block grid (separator lines) -> block-colour grid flipped horizontally
# --------------------------------------------------------------------------- #
def build_blkflip():
    g = _G()
    _consts(g)
    big = g.f([1, 1, 1, 1], [1.0e6])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    ch0 = _slice_ch(g, "input", 0, 1)
    nonbg = g.nd("Sub", [realmask, ch0])
    realcount = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)      # [1,1,30,1]
    nonbgcount = g.nd("ReduceSum", [nonbg], axes=[3], keepdims=1)
    nobg = g.nd("Cast", [g.nd("Less",
            [g.nd("Abs", [g.nd("Sub", [realcount, nonbgcount])]), g.half])], to=F)
    chancount = g.nd("ReduceSum", ["input"], axes=[3], keepdims=1)       # [1,10,30,1]
    maxchan = g.nd("ReduceMax", [chancount], axes=[1], keepdims=1)       # [1,1,30,1]
    uniform = g.nd("Cast", [g.nd("Less",
            [g.nd("Abs", [g.nd("Sub", [maxchan, realcount])]), g.half])], to=F)
    isreal = g.nd("Cast", [g.nd("Greater", [realcount, g.half])], to=F)
    sepf = g.nd("Mul", [g.nd("Mul", [nobg, uniform]), isreal])           # [1,1,30,1]
    cand = g.nd("Add", [g.nd("Mul", [sepf, g.rowidx]),
                        g.nd("Mul", [g.nd("Sub", [g.one, sepf]), big])])
    minsep = g.nd("ReduceMin", [cand], axes=[2], keepdims=1)             # [1,1,1,1]
    p = g.nd("Add", [minsep, g.one])
    k = g.nd("Add", [g.nd("ReduceSum", [sepf], axes=[2], keepdims=1), g.one])
    # Srow[out,in]=1 iff in==out*p & out<k   (out=axis2,in=axis3)
    term = g.nd("Sub", [g.colidx, g.nd("Mul", [g.rowidx, p])])
    mrow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [term]), g.half])], to=F)
    trow = g.nd("Cast", [g.nd("Less", [g.rowidx, k])], to=F)
    Srow = g.nd("Mul", [mrow, trow])
    # Scol[in,out]=1 iff in==(k-1-out)*p & out<k  (in=axis2,out=axis3)
    target = g.nd("Mul", [g.nd("Sub", [g.nd("Sub", [k, g.one]), g.colidx]), p])
    term2 = g.nd("Sub", [g.rowidx, target])
    mcol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [term2]), g.half])], to=F)
    tcol = g.nd("Cast", [g.nd("Less", [g.colidx, k])], to=F)
    Scol = g.nd("Mul", [mcol, tcol])
    colsel = g.nd("MatMul", ["input", Scol])
    g.nd("MatMul", [Srow, colsel], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# T316 -- sort the scattered cells column-major and lay them out in 3x3 snake   #
# --------------------------------------------------------------------------- #
def build_snake():
    g = _G()
    _consts(g)
    M = _nonbg(g)                                                # [1,1,30,30]
    key2d = g.f([1, 1, H, W], [c * 30 + r for r in range(H) for c in range(W)])
    big = g.f([1, 1, 1, 1], [1.0e6])
    keyM = g.nd("Add", [g.nd("Mul", [M, key2d]),
                        g.nd("Mul", [g.nd("Sub", [g.one, M]), big])])   # [1,1,30,30]
    Kcol = g.nd("Reshape", [keyM, g.i64([H * W, 1])])           # [900,1]
    Krow = g.nd("Reshape", [keyM, g.i64([1, H * W])])           # [1,900]
    cmp = g.nd("Cast", [g.nd("Less", [Krow, Kcol])], to=F)      # [900,900] key_j<key_i
    rank = g.nd("ReduceSum", [cmp], axes=[1], keepdims=1)       # [900,1]
    rankRow = g.nd("Reshape", [rank, g.i64([1, H * W])])        # [1,900]
    Mrow = g.nd("Reshape", [M, g.i64([1, H * W])])             # [1,900]
    kgrid = g.f([9, 1], list(range(9)))
    half2 = g.f([1, 1], [0.5])
    eqm = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rankRow, kgrid])]), half2])], to=F)
    Sel = g.nd("Mul", [eqm, Mrow])                             # [9,900]
    colorsT = g.nd("Transpose", ["input"], perm=[0, 2, 3, 1])  # [1,30,30,10]
    colors = g.nd("Reshape", [colorsT, g.i64([H * W, CHANNELS])])   # [900,10]
    slotc = g.nd("MatMul", [Sel, colors])                     # [9,10]
    # snake permutation: slot k -> grid flat pos
    pos_flat = [0, 1, 2, 5, 4, 3, 6, 7, 8]
    perm = [[1.0 if pos_flat[k] == p else 0.0 for k in range(9)] for p in range(9)]
    Perm = g.f([9, 9], perm)
    gridc = g.nd("MatMul", [Perm, slotc])                     # [9,10]
    presence = g.nd("ReduceSum", [gridc], axes=[1], keepdims=1)   # [9,1]
    bg = g.nd("Sub", [g.f([9, 1], [1.0] * 9), presence])
    e0row = g.f([1, CHANNELS], [1.0] + [0.0] * 9)
    bgadd = g.nd("MatMul", [bg, e0row])                       # [9,10]
    gridfull = g.nd("Add", [gridc, bgadd])                   # [9,10]
    g33 = g.nd("Reshape", [gridfull, g.i64([1, 3, 3, CHANNELS])])
    g10 = g.nd("Transpose", [g33], perm=[0, 3, 1, 2])        # [1,10,3,3]
    g.nd("Pad", [g10], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, H - 3, W - 3])
    return _model(g)


# --------------------------------------------------------------------------- #
# T271 -- pick the solid 3x3 glyph with the most colour-1 cells                 #
# --------------------------------------------------------------------------- #
def build_glyphmax():
    g = _G()
    _consts(g)
    ch1 = _slice_ch(g, "input", 1, 2)                            # [1,1,30,30]
    M = _nonbg(g)
    ones3 = g.f([1, 1, 3, 3], [1.0] * 9)
    cnt1 = g.nd("Conv", [ch1, ones3], kernel_shape=[3, 3], strides=[1, 1],
                pads=[0, 0, 0, 0])                               # [1,1,28,28]
    solidc = g.nd("Conv", [M, ones3], kernel_shape=[3, 3], strides=[1, 1],
                  pads=[0, 0, 0, 0])                             # [1,1,28,28]
    solid = g.nd("Cast", [g.nd("Greater", [solidc, g.f([1, 1, 1, 1], [8.5])])], to=F)
    idxg = g.f([1, 1, 28, 28], [r * 28 + c for r in range(28) for c in range(28)])
    sc = g.nd("Sub", [g.nd("Mul", [cnt1, g.f([1, 1, 1, 1], [1000.0])]), idxg])
    score = g.nd("Mul", [sc, solid])                            # [1,1,28,28]
    mx = g.nd("ReduceMax", [score], axes=[2, 3], keepdims=1)
    gate = g.nd("Cast", [g.nd("Greater", [score, g.nd("Sub", [mx, g.half])])], to=F)
    winr = g.f([1, 1, 28, 28], [r for r in range(28) for _ in range(28)])
    winc = g.f([1, 1, 28, 28], [c for _ in range(28) for c in range(28)])
    minrow = g.nd("ReduceSum", [g.nd("Mul", [gate, winr])], axes=[2, 3], keepdims=1)
    mincol = g.nd("ReduceSum", [g.nd("Mul", [gate, winc])], axes=[2, 3], keepdims=1)
    region = _extract_region(g, "input", minrow, mincol, g.f([1, 1, 1, 1], [3.0]),
                             g.f([1, 1, 1, 1], [3.0]))
    g.nd("Identity", [region], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# T112 -- 4-fold reflection about the centroid of the colour-3 marker          #
# --------------------------------------------------------------------------- #
def build_reflect4():
    g = _G()
    _consts(g)
    ch3 = _slice_ch(g, "input", 3, 4)                            # [1,1,30,30]
    cnt = g.nd("ReduceSum", [ch3], axes=[2, 3], keepdims=1)      # [1,1,1,1]
    sumr = g.nd("ReduceSum", [g.nd("Mul", [ch3, g.rowidx])], axes=[2, 3], keepdims=1)
    sumc = g.nd("ReduceSum", [g.nd("Mul", [ch3, g.colidx])], axes=[2, 3], keepdims=1)
    two = g.f([1, 1, 1, 1], [2.0])
    tr2 = g.nd("Mul", [sumr, two])      # 2*sum_r
    tc2 = g.nd("Mul", [sumc, two])      # 2*sum_c
    s = g.nd("Mul", [g.nd("Add", [g.rowidx, g.colidx]), cnt])    # (i+j)*cnt  [1,1,30,30]
    Rr = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [s, tr2])]), g.half])], to=F)
    Rc = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [s, tc2])]), g.half])], to=F)
    not0 = g.f([1, C, 1, 1], [0.0] + [1.0] * 9)
    nb = g.nd("Mul", ["input", not0])                            # drop background
    Hr = g.nd("MatMul", [nb, Rc])
    Vr = g.nd("MatMul", [Rr, nb])
    HV = g.nd("MatMul", [Rr, Hr])
    NB = g.nd("Max", [nb, Hr, Vr, HV])                           # [1,10,30,30]
    presence = g.nd("ReduceSum", [NB], axes=[1], keepdims=1)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    bg = g.nd("Sub", [realmask, presence])
    e0 = g.f([1, C, 1, 1], [1.0] + [0.0] * 9)
    g.nd("Add", [NB, g.nd("Mul", [bg, e0])], "output")
    return _model(g)


# ===========================================================================
# numpy mirrors (used for exact detection)                                    #
# ===========================================================================
def _ref_blkflip(a):
    Hh, Ww = a.shape
    if Hh != Ww:
        return None
    seprows = [r for r in range(Hh) if (a[r] != 0).all() and len(set(a[r].tolist())) == 1]
    if not seprows:
        return None
    p = seprows[0] + 1
    k = len(seprows) + 1
    if k * p - 1 != Hh:
        return None
    if (k - 1) * p >= Ww:
        return None
    out = np.zeros((k, k), int)
    for i in range(k):
        for j in range(k):
            out[i, k - 1 - j] = a[i * p, j * p]
    return out


def _ref_snake(a):
    Hh, Ww = a.shape
    cells = []
    for c in range(Ww):
        for r in range(Hh):
            if a[r, c] != 0:
                cells.append((c * 30 + r, a[r, c]))
    cells.sort()
    vals = [v for _, v in cells]
    if len(vals) > 9:
        return None
    out = np.zeros((3, 3), int)
    pos = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
    for k, v in enumerate(vals):
        out[pos[k]] = v
    return out


def _ref_glyphmax(a):
    Hh, Ww = a.shape
    best, bscore = None, None
    for r in range(Hh - 2):
        for c in range(Ww - 2):
            sub = a[r:r + 3, c:c + 3]
            if (sub != 0).all():
                ones = int((sub == 1).sum())
                score = ones * 100000 - (r * Ww + c)
                if bscore is None or score > bscore:
                    bscore, best = score, sub.copy()
    return best


def _ref_lplace(a):
    Hh, Ww = a.shape
    M = (a != 0).astype(int)

    def sh(dr, dc):
        out = np.zeros_like(M)
        for r in range(Hh):
            for c in range(Ww):
                rr, cc = r - dr, c - dc
                if 0 <= rr < Hh and 0 <= cc < Ww:
                    out[r, c] = M[rr, cc]
        return out

    right, left, down, up = sh(0, -1), sh(0, 1), sh(-1, 0), sh(1, 0)
    orient = [
        (M * right * down, 0, 0, 0, 0),
        (M * left * down, 0, -1, 0, 2),
        (M * right * up, -1, 0, 2, 0),
        (M * left * up, -1, -1, 2, 2),
    ]
    out = np.zeros((4, 4), int)
    for mask, dbr, dbc, qr, qc in orient:
        ys, xs = np.where(mask == 1)
        if len(ys) != 1:
            return None
        br, bc = ys[0] + dbr, xs[0] + dbc
        for u in range(2):
            for v in range(2):
                if 0 <= br + u < Hh and 0 <= bc + v < Ww:
                    out[qr + u, qc + v] = a[br + u, bc + v]
    return out


def _ref_reflect4(a):
    ys, xs = np.where(a == 3)
    if len(ys) == 0:
        return None
    Hh, Ww = a.shape
    tr = 2.0 * ys.sum() / len(ys)
    tc = 2.0 * xs.sum() / len(xs)
    if abs(tr - round(tr)) > 1e-6 or abs(tc - round(tc)) > 1e-6:
        return None
    tr, tc = int(round(tr)), int(round(tc))
    out = np.zeros_like(a)
    for r in range(Hh):
        for c in range(Ww):
            best = 0
            for rr, cc in [(r, c), (r, tc - c), (tr - r, c), (tr - r, tc - c)]:
                if 0 <= rr < Hh and 0 <= cc < Ww and a[rr, cc] != 0:
                    best = a[rr, cc]
            out[r, c] = best
    return out


def _ref_diagbar(a):
    Hh, Ww = a.shape
    if Hh != Ww:
        return None
    nz = set(a.ravel().tolist()) - {0}
    if len(nz) != 1:
        return None
    cc = nz.pop()
    if not (a[:, 0] == cc).all() or (a[:, 1:] != 0).any():
        return None
    n = Hh
    out = np.zeros((Hh, Ww), int)
    out[:, 0] = cc
    for r in range(n - 1):
        out[r, n - 1 - r] = 2
    for c in range(1, n):
        out[n - 1, c] = 4
    return out


def _ref_q4odd(a):
    if a.shape != (5, 5):
        return None
    Q = [a[0:2, 0:2], a[0:2, 3:5], a[3:5, 0:2], a[3:5, 3:5]]
    d = [sum(int(np.abs(Q[i] - Q[j]).sum()) for j in range(4)) for i in range(4)]
    best = max(range(4), key=lambda i: (d[i], -i))
    return Q[best].copy()


def _ref_mk8(a):
    ys, xs = np.where(a == 8)
    if len(ys) != 1:
        return None
    r0, c0 = int(ys[0]), int(xs[0])
    if r0 - 1 < 0 or c0 - 1 < 0 or r0 + 2 > a.shape[0] or c0 + 2 > a.shape[1]:
        return None
    reg = a[r0 - 1:r0 + 2, c0 - 1:c0 + 2].copy()
    cols = set(reg.ravel().tolist()) - {0, 8}
    if len(cols) != 1:
        return None
    cc = cols.pop()
    return np.where(reg == 8, cc, reg)


def _ref_upcnt(a):
    k = len(set(a.ravel().tolist()) - {0})
    if k < 1:
        return None
    if a.shape[0] * k > 30 or a.shape[1] * k > 30:
        return None
    return np.kron(a, np.ones((k, k), int))


def _ref_blkodd(a):
    Hh, Ww = a.shape
    if Hh % 3 or Ww % 3:
        return None
    blocks, shapes = [], []
    for bi in range(Hh // 3):
        for bj in range(Ww // 3):
            blk = a[bi * 3:bi * 3 + 3, bj * 3:bj * 3 + 3]
            if (blk != 0).any():
                blocks.append(blk)
                shapes.append(tuple((blk != 0).ravel().tolist()))
    if len(blocks) < 2:
        return None
    cnt = Counter(shapes)
    maj = max(cnt, key=lambda s: cnt[s])
    odds = [blk for blk, s in zip(blocks, shapes) if s != maj]
    if len(odds) != 1:
        return None
    return odds[0].copy()


def _ref_compress(a):
    Hh, Ww = a.shape
    keepr = []
    for r in range(Hh):
        active = (a[r] != 0).any()
        prev = a[r - 1] if r > 0 else np.zeros(Ww, int)
        if active and not np.array_equal(a[r], prev):
            keepr.append(r)
    if not keepr:
        return None
    sub = a[keepr]
    keepc = []
    for c in range(sub.shape[1]):
        active = (sub[:, c] != 0).any()
        prev = sub[:, c - 1] if c > 0 else np.zeros(sub.shape[0], int)
        if active and not np.array_equal(sub[:, c], prev):
            keepc.append(c)
    if not keepc:
        return None
    return sub[:, keepc].copy()


def _ref_plusmid(a):
    ys, xs = np.where(a == 1)
    if len(ys) != 2:
        return None
    mr, mc = int(ys[0] + ys[1]), int(xs[0] + xs[1])
    if mr % 2 or mc % 2:
        return None
    mr //= 2
    mc //= 2
    out = a.copy()
    Hh, Ww = a.shape
    for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        r, c = mr + dr, mc + dc
        if 0 <= r < Hh and 0 <= c < Ww:
            out[r, c] = 3
    out[ys[0], xs[0]] = 1
    out[ys[1], xs[1]] = 1
    return out


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
    seen = False
    for a, b in prs:
        try:
            o = fn(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
        seen = True
    return seen


def _emit(out, name, builder):
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    out.append((name, m))


_RULES = [
    ("crk2_q4odd", _ref_q4odd, build_q4odd),
    ("crk2_mk8", _ref_mk8, build_mk8),
    ("crk2_upcnt", _ref_upcnt, build_upcnt),
    ("crk2_blkodd", _ref_blkodd, build_blkodd),
    ("crk2_compress", _ref_compress, build_compress),
    ("crk2_plusmid", _ref_plusmid, build_plusmid),
    ("crk2_diagbar", _ref_diagbar, build_diagbar),
    ("crk2_reflect4", _ref_reflect4, build_reflect4),
    ("crk2_lplace", _ref_lplace, build_lplace),
    ("crk2_glyphmax", _ref_glyphmax, build_glyphmax),
    ("crk2_snake", _ref_snake, build_snake),
    ("crk2_blkflip", _ref_blkflip, build_blkflip),
]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    out = []
    for name, ref, builder in _RULES:
        if _matches(prs, ref):
            _emit(out, name, builder)
    return out
