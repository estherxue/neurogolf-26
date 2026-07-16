"""family_golf3_4 -- cheaper EXACT re-derivations of low-scoring golf targets.

Each rule is detected STRUCTURALLY via a numpy mirror validated EXACTLY on every
provided pair (train+test+arc-gen); only then is the (data-driven, static) ONNX
graph emitted.  Graphs are built to minimise cost = params + intermediate memory
(few/small intermediates, write straight to the FREE `output`).

Targets (slice [4::6]):
  stair       (T295)  single colour row of length-c0 prefix -> lower staircase of
                      height W/2; row r has prefix c0+r.
  edgelines   (T161)  the unique colour that appears ONLY on the grid border draws
                      a full line across every matched border pair (top<->bottom =>
                      vertical line, left<->right => horizontal line).
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


# =========================================================================== #
# T295 -- staircase                                                           #
# =========================================================================== #
def build_stair():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])

    # W (= total real cells, since input is one row) and c0 (= non-bg count)
    Wt = g.nd("ReduceSum", ["input"], axes=[1, 2, 3], keepdims=1)          # [1,1,1,1]
    nb = g.nd("Slice", ["input", g.i64([1]), g.i64([C]), g.i64([1])])      # ch 1..9
    c0 = g.nd("ReduceSum", [nb], axes=[1, 2, 3], keepdims=1)               # [1,1,1,1]
    halfW = g.nd("Mul", [Wt, half])

    # scalar colour value of the single present non-bg colour
    total = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    present = g.nd("Cast", [g.nd("Greater", [total, half])], to=F)
    notbg = g.f([1, C, 1, 1], [0.0] + [1.0] * 9)
    idxf = g.f([1, C, 1, 1], list(range(C)))
    colorval = g.nd("ReduceSum", [g.nd("Mul", [g.nd("Mul", [present, notbg]), idxf])],
                    axes=[1], keepdims=1)                                  # [1,1,1,1]

    # masks
    D = g.nd("Sub", [colidx, rowidx])                                      # [1,1,30,30]
    lessD = g.nd("Cast", [g.nd("Less", [D, c0])], to=F)
    Rmask = g.nd("Cast", [g.nd("Less", [rowidx, halfW])], to=F)            # [1,1,30,1]
    Wmask = g.nd("Cast", [g.nd("Less", [colidx, Wt])], to=F)              # [1,1,1,30]
    realmask = g.nd("Mul", [Rmask, Wmask])                                 # [1,1,30,30]
    Cmask = g.nd("Mul", [lessD, realmask])                                 # [1,1,30,30]

    # one-hot output via integer Equal: colour on staircase, 0 on real-bg, -1 on pad
    one = g.f([1, 1, 1, 1], [1.0])
    Gf = g.nd("Sub", [g.nd("Mul", [colorval, Cmask]), g.nd("Sub", [one, realmask])])
    Gint = g.nd("Cast", [Gf], to=INT64)                                    # [1,1,30,30]
    idxvec = g.i64(list(range(C)), dims=[1, C, 1, 1])
    g.nd("Cast", [g.nd("Equal", [Gint, idxvec])], "output", to=F)
    return _model(g)


def _ref_stair(a):
    Hh, Ww = a.shape
    if Hh != 1:
        return None
    c = a[0]
    nz = c[c != 0]
    if len(nz) == 0:
        return None
    color = int(nz[0])
    if (c != 0).tolist() != [True] * int((c != 0).sum()) + [False] * (Ww - int((c != 0).sum())):
        return None  # require contiguous prefix
    c0 = int((c != 0).sum())
    Ho = Ww // 2
    if Ho < 1 or Ho > 30:
        return None
    out = np.zeros((Ho, Ww), int)
    for r in range(Ho):
        out[r, :min(c0 + r, Ww)] = color
    return out


# =========================================================================== #
# T161 -- border-pair lines                                                   #
# =========================================================================== #
def build_edgelines():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])

    # real-cell row / col masks
    realrow = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 3], keepdims=1), half])], to=F)  # [1,1,30,1]
    realcol = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 2], keepdims=1), half])], to=F)  # [1,1,1,30]

    # last-real-row / last-real-col selectors
    pad_r = g.nd("Pad", [realrow], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 1, 0])
    shiftup = g.nd("Slice", [pad_r, g.i64([1]), g.i64([H + 1]), g.i64([2])])              # [1,1,30,1]
    brow = g.nd("Mul", [realrow, g.nd("Sub", [one, shiftup])])                            # [1,1,30,1]
    pad_c = g.nd("Pad", [realcol], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 1])
    shiftleft = g.nd("Slice", [pad_c, g.i64([1]), g.i64([W + 1]), g.i64([3])])            # [1,1,1,30]
    bcol = g.nd("Mul", [realcol, g.nd("Sub", [one, shiftleft])])                          # [1,1,1,30]

    # interior mask (exclude all four borders)
    introw = g.nd("Mul", [g.nd("Mul", [realrow, g.nd("Cast", [g.nd("Greater", [rowidx, half])], to=F)]),
                          g.nd("Sub", [one, brow])])                                      # [1,1,30,1]
    intcol = g.nd("Mul", [g.nd("Mul", [realcol, g.nd("Cast", [g.nd("Greater", [colidx, half])], to=F)]),
                          g.nd("Sub", [one, bcol])])                                      # [1,1,1,30]
    intmask = g.nd("Mul", [introw, intcol])                                               # [1,1,30,30]

    # marker channel = present non-bg colour with zero interior occurrences
    ic = g.nd("ReduceSum", [g.nd("Mul", ["input", intmask])], axes=[2, 3], keepdims=1)    # [1,10,1,1]
    total = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    present = g.nd("Cast", [g.nd("Greater", [total, half])], to=F)
    notbg = g.f([1, C, 1, 1], [0.0] + [1.0] * 9)
    gate = g.nd("Mul", [present, notbg])
    negic = g.nd("Mul", [ic, g.f([1, 1, 1, 1], [-1.0])])
    scoregate = g.nd("Mul", [g.nd("Sub", [gate, one]), g.f([1, 1, 1, 1], [_BIG])])
    score = g.nd("Add", [negic, scoregate])                                               # [1,10,1,1]
    markeridx = g.nd("ArgMax", [score], axis=1, keepdims=1)                               # int64 [1,1,1,1]

    # marker presence grid
    midx1 = g.nd("Reshape", [markeridx, g.i64([1])])                                      # int64 [1]
    mgrid = g.nd("Gather", ["input", midx1], axis=1)                                      # [1,1,30,30]

    # borders of marker presence
    toprow = g.nd("Slice", [mgrid, g.i64([0]), g.i64([1]), g.i64([2])])                   # [1,1,1,30]
    botrow = g.nd("ReduceSum", [g.nd("Mul", [mgrid, brow])], axes=[2], keepdims=1)        # [1,1,1,30]
    leftcol = g.nd("Slice", [mgrid, g.i64([0]), g.i64([1]), g.i64([3])])                  # [1,1,30,1]
    rightcol = g.nd("ReduceSum", [g.nd("Mul", [mgrid, bcol])], axes=[3], keepdims=1)      # [1,1,30,1]

    vmask = g.nd("Mul", [toprow, botrow])                                                 # [1,1,1,30]
    hmask = g.nd("Mul", [leftcol, rightcol])                                              # [1,1,30,1]
    vline = g.nd("Mul", [vmask, realrow])                                                 # [1,1,30,30]
    hline = g.nd("Mul", [hmask, realcol])                                                 # [1,1,30,30]
    linemask = g.nd("Cast", [g.nd("Greater", [g.nd("Add", [vline, hline]), half])], to=F)  # [1,1,30,30]
    realmask = g.nd("Mul", [realrow, realcol])                                            # [1,1,30,30]

    # one-hot output via integer Equal: marker colour on lines, 0 on real-bg, -1 on pad
    markercolor = g.nd("Cast", [markeridx], to=F)                                         # [1,1,1,1]
    Gf = g.nd("Sub", [g.nd("Mul", [markercolor, linemask]), g.nd("Sub", [one, realmask])])
    Gint = g.nd("Cast", [Gf], to=INT64)                                                   # [1,1,30,30]
    idxvec = g.i64(list(range(C)), dims=[1, C, 1, 1])
    g.nd("Cast", [g.nd("Equal", [Gint, idxvec])], "output", to=F)
    return _model(g)


def _ref_edgelines(a):
    Hh, Ww = a.shape
    if Hh < 3 or Ww < 3:
        return None
    interior = a[1:Hh - 1, 1:Ww - 1]
    cands = []
    for col in range(1, 10):
        tot = int((a == col).sum())
        if tot == 0:
            continue
        inter = int((interior == col).sum())
        cands.append((inter, col))
    if not cands:
        return None
    cands.sort()
    inter0, mk = cands[0]
    if inter0 != 0:
        return None
    out = np.zeros_like(a)
    top, bot, left, right = a[0], a[Hh - 1], a[:, 0], a[:, Ww - 1]
    drew = False
    for c in range(Ww):
        if top[c] == mk and bot[c] == mk:
            out[:, c] = mk
            drew = True
    for r in range(Hh):
        if left[r] == mk and right[r] == mk:
            out[r, :] = mk
            drew = True
    if not drew:
        return None
    return out


# =========================================================================== #
# T237 -- staircase rays (one marker per row: fill right + edge-column bands)  #
# =========================================================================== #
def build_rays():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])

    idxf = g.f([1, C, 1, 1], list(range(C)))
    colorgrid = g.nd("ReduceSum", [g.nd("Mul", ["input", idxf])], axes=[1], keepdims=1)   # [1,1,30,30]

    realrow = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 3], keepdims=1), half])], to=F)   # [1,1,30,1]
    realcol = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 2], keepdims=1), half])], to=F)   # [1,1,1,30]
    realmask = g.nd("Mul", [realrow, realcol])
    pad_c = g.nd("Pad", [realcol], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 1])
    shiftleft = g.nd("Slice", [pad_c, g.i64([1]), g.i64([W + 1]), g.i64([3])])
    bcol = g.nd("Mul", [realcol, g.nd("Sub", [one, shiftleft])])                           # [1,1,1,30]

    presence = g.nd("Cast", [g.nd("Greater", [colorgrid, half])], to=F)                    # [1,1,30,30]
    markercolor = g.nd("ReduceSum", [colorgrid], axes=[3], keepdims=1)                     # [1,1,30,1]
    markercol = g.nd("ReduceSum", [g.nd("Mul", [presence, colidx])], axes=[3], keepdims=1)  # [1,1,30,1]

    # horizontal fill: colour from marker col to the right within the grid
    geo = g.nd("Sub", [one, g.nd("Cast", [g.nd("Less", [colidx, markercol])], to=F)])      # colidx>=markercol
    Hc = g.nd("Mul", [g.nd("Mul", [markercolor, geo]), realcol])                           # [1,1,30,30]

    # forward-fill the marker colours down the edge column (Hillis-Steele)
    filled = markercolor
    for step in (1, 2, 4, 8, 16):
        pad = g.nd("Pad", [filled], mode="constant", value=0.0,
                   pads=[0, 0, step, 0, 0, 0, 0, 0])
        shifted = g.nd("Slice", [pad, g.i64([0]), g.i64([H]), g.i64([2])])                 # [1,1,30,1]
        iszero = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [filled]), half])], to=F)
        filled = g.nd("Add", [filled, g.nd("Mul", [iszero, shifted])])
    edgecol = g.nd("Mul", [filled, realrow])                                               # [1,1,30,1]
    edgeC = g.nd("Mul", [edgecol, bcol])                                                   # [1,1,30,30]

    OUT = g.nd("Add", [edgeC, g.nd("Mul", [Hc, g.nd("Sub", [one, bcol])])])                # [1,1,30,30]
    Gf = g.nd("Sub", [g.nd("Mul", [OUT, realmask]), g.nd("Sub", [one, realmask])])
    Gint = g.nd("Cast", [Gf], to=INT64)
    idxvec = g.i64(list(range(C)), dims=[1, C, 1, 1])
    g.nd("Cast", [g.nd("Equal", [Gint, idxvec])], "output", to=F)
    return _model(g)


def _ref_rays(a):
    Hh, Ww = a.shape
    if Hh < 1 or Ww < 1:
        return None
    out = np.zeros_like(a)
    rowcolor = np.zeros(Hh, int)
    rowcol = np.full(Hh, -1)
    nmark = 0
    for r in range(Hh):
        nz = np.where(a[r] != 0)[0]
        if len(nz) > 1:
            return None
        if len(nz) == 1:
            rowcolor[r] = a[r, nz[0]]
            rowcol[r] = nz[0]
            nmark += 1
    if nmark < 1:
        return None
    for r in range(Hh):
        if rowcolor[r]:
            out[r, rowcol[r]:Ww] = rowcolor[r]
    cur = 0
    for r in range(Hh):
        if rowcolor[r]:
            cur = rowcolor[r]
        if cur:
            out[r, Ww - 1] = cur
    return out


# =========================================================================== #
# T215 -- vertical period-3 tiling of a 3-row pattern band                     #
# =========================================================================== #
def build_period3():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    idxf = g.f([1, C, 1, 1], list(range(C)))
    colorgrid = g.nd("ReduceSum", [g.nd("Mul", ["input", idxf])], axes=[1], keepdims=1)   # [1,1,30,30]

    realrow = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 3], keepdims=1), half])], to=F)   # [1,1,30,1]
    realcol = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 2], keepdims=1), half])], to=F)   # [1,1,1,30]
    realmask = g.nd("Mul", [realrow, realcol])

    M = g.f([1, 1, H, H], [[1.0 if (r % 3) == (rr % 3) else 0.0 for rr in range(H)] for r in range(H)])
    OUT = g.nd("MatMul", [M, colorgrid])                                                   # [1,1,30,30]
    Gf = g.nd("Sub", [g.nd("Mul", [OUT, realmask]), g.nd("Sub", [one, realmask])])
    Gint = g.nd("Cast", [Gf], to=INT64)
    idxvec = g.i64(list(range(C)), dims=[1, C, 1, 1])
    g.nd("Cast", [g.nd("Equal", [Gint, idxvec])], "output", to=F)
    return _model(g)


def _ref_period3(a):
    Hh, Ww = a.shape
    out = np.zeros((Hh, Ww), int)
    for r in range(Hh):
        for c in range(Ww):
            s = 0
            for rr in range(r % 3, Hh, 3):
                s += int(a[rr, c])
            out[r, c] = s
    if out.max() > 9 or out.min() < 0:
        return None
    return out


# =========================================================================== #
# T35 -- project outside dots onto the nearest edge of a solid block          #
# =========================================================================== #
def build_project():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    BIGc = g.f([1, 1, 1, 1], [_BIG])
    idxf = g.f([1, C, 1, 1], list(range(C)))

    colorgrid = g.nd("ReduceSum", [g.nd("Mul", ["input", idxf])], axes=[1], keepdims=1)
    realrow = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 3], keepdims=1), half])], to=F)
    realcol = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[1, 2], keepdims=1), half])], to=F)
    realmask = g.nd("Mul", [realrow, realcol])

    # block = majority non-bg colour
    count = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    notbg = g.f([1, C, 1, 1], [0.0] + [1.0] * 9)
    blockidx = g.nd("ArgMax", [g.nd("Mul", [count, notbg])], axis=1, keepdims=1)
    blockcolor = g.nd("Cast", [blockidx], to=F)
    blockgrid = g.nd("Gather", ["input", g.nd("Reshape", [blockidx, g.i64([1])])], axis=1)

    rowhas = g.nd("ReduceMax", [blockgrid], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [blockgrid], axes=[2], keepdims=1)
    rmin = g.nd("ReduceMin", [g.nd("Add", [rowidx, g.nd("Mul", [g.nd("Sub", [one, rowhas]), BIGc])])],
                axes=[2], keepdims=1)
    rmax = g.nd("ReduceMax", [g.nd("Sub", [rowidx, g.nd("Mul", [g.nd("Sub", [one, rowhas]), BIGc])])],
                axes=[2], keepdims=1)
    cmin = g.nd("ReduceMin", [g.nd("Add", [colidx, g.nd("Mul", [g.nd("Sub", [one, colhas]), BIGc])])],
                axes=[3], keepdims=1)
    cmax = g.nd("ReduceMax", [g.nd("Sub", [colidx, g.nd("Mul", [g.nd("Sub", [one, colhas]), BIGc])])],
                axes=[3], keepdims=1)

    dotsgrid = g.nd("Mul", [colorgrid, g.nd("Sub", [one, blockgrid])])
    above = g.nd("Cast", [g.nd("Less", [rowidx, rmin])], to=F)
    below = g.nd("Cast", [g.nd("Greater", [rowidx, rmax])], to=F)
    leftof = g.nd("Cast", [g.nd("Less", [colidx, cmin])], to=F)
    rightof = g.nd("Cast", [g.nd("Greater", [colidx, cmax])], to=F)
    topdot = g.nd("ReduceSum", [g.nd("Mul", [dotsgrid, above])], axes=[2], keepdims=1)
    botdot = g.nd("ReduceSum", [g.nd("Mul", [dotsgrid, below])], axes=[2], keepdims=1)
    leftdot = g.nd("ReduceSum", [g.nd("Mul", [dotsgrid, leftof])], axes=[3], keepdims=1)
    rightdot = g.nd("ReduceSum", [g.nd("Mul", [dotsgrid, rightof])], axes=[3], keepdims=1)

    def eqs(idx, val):
        return g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [idx, val])]), half])], to=F)

    topmap = g.nd("Mul", [eqs(rowidx, rmin), topdot])
    botmap = g.nd("Mul", [eqs(rowidx, rmax), botdot])
    leftmap = g.nd("Mul", [eqs(colidx, cmin), leftdot])
    rightmap = g.nd("Mul", [eqs(colidx, cmax), rightdot])

    def pos(x):
        return g.nd("Cast", [g.nd("Greater", [x, half])], to=F)

    v2 = g.nd("Add", [topmap, g.nd("Mul", [botmap, g.nd("Sub", [one, pos(topmap)])])])
    v3 = g.nd("Add", [v2, g.nd("Mul", [leftmap, g.nd("Sub", [one, pos(v2)])])])
    v4 = g.nd("Add", [v3, g.nd("Mul", [rightmap, g.nd("Sub", [one, pos(v3)])])])
    recolor = g.nd("Mul", [v4, blockgrid])
    blockcontribution = g.nd("Add", [g.nd("Mul", [g.nd("Mul", [blockgrid, blockcolor]),
                                                  g.nd("Sub", [one, pos(recolor)])]), recolor])
    OUT = g.nd("Add", [dotsgrid, blockcontribution])

    Gf = g.nd("Sub", [g.nd("Mul", [OUT, realmask]), g.nd("Sub", [one, realmask])])
    Gint = g.nd("Cast", [Gf], to=INT64)
    idxvec = g.i64(list(range(C)), dims=[1, C, 1, 1])
    g.nd("Cast", [g.nd("Equal", [Gint, idxvec])], "output", to=F)
    return _model(g)


def _ref_project(a):
    vals, cnts = np.unique(a[a != 0], return_counts=True)
    if len(vals) == 0:
        return None
    block = int(vals[cnts.argmax()])
    bg = (a == block)
    if bg.sum() == 0:
        return None
    rows = np.where(bg.any(1))[0]
    cols = np.where(bg.any(0))[0]
    rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
    Hh, Ww = a.shape
    dots = np.where((a != 0) & (a != block), a, 0)
    topdot = np.array([dots[:rmin, c].sum() for c in range(Ww)])
    botdot = np.array([dots[rmax + 1:, c].sum() for c in range(Ww)])
    leftdot = np.array([dots[r, :cmin].sum() for r in range(Hh)])
    rightdot = np.array([dots[r, cmax + 1:].sum() for r in range(Hh)])
    out = a.copy()
    for r in range(Hh):
        for c in range(Ww):
            if not bg[r, c]:
                continue
            v = 0
            if r == rmin and topdot[c] > 0:
                v = topdot[c]
            elif r == rmax and botdot[c] > 0:
                v = botdot[c]
            elif c == cmin and leftdot[r] > 0:
                v = leftdot[r]
            elif c == cmax and rightdot[r] > 0:
                v = rightdot[r]
            if v > 0:
                out[r, c] = v
    if out.max() > 9:
        return None
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
    ("golf_stair", _ref_stair, build_stair),
    ("golf_edgelines", _ref_edgelines, build_edgelines),
    ("golf_rays", _ref_rays, build_rays),
    ("golf_period3", _ref_period3, build_period3),
    ("golf_project", _ref_project, build_project),
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
