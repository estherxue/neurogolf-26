"""family_crk8_3 -- cracks for slice U[3::6].

Solved here
-----------
* task 19   TILE-2x2 + DIAGONAL-HALO (new).  The output is the input tiled 2x2
  (so 2H x 2W, origin-anchored), then every BACKGROUND cell that is diagonally
  adjacent (one of the 4 corner neighbours) to a coloured cell is painted 8.
  Coloured cells keep their colour.

  Fully size-independent.  H,W are read from the real-cell mask; the 2x2 tiling
  is realised as two data-dependent selection matrices  out = S_row @ in @ S_col
  with  S_row[i,j]=1 iff j==(i mod H) and i<2H  (and the W analogue), built from
  static [30,30] coordinate grids via Mod/Less.  The diagonal halo is a fixed
  3x3 Conv (corners=1) over the coloured mask.  Static shapes, no Loop/NonZero.
  Verified EXACT on all train/test/arc-gen (267 pairs).

* task 165  COLUMN RAYS FROM SHAPE THROUGH DOTS (new).  Two colours: one is a
  connected "shape" glyph, the other is scattered single "dots".  In every
  column that contains a shape cell, if a dot lies BELOW the lowest shape cell
  of that column, the whole column from just below the shape down to the grid
  bottom is painted the dot colour.

  Shape vs dot are told apart by isolation: shape = the active colour whose
  cells all have a same-colour 8-neighbour (iso==0), dot = the other.  iso is a
  depthwise 3x3 Conv; per-column "lowest shape row" / "dot below" are ReduceMax
  reductions over coordinate grids.  Static shapes, no Loop/NonZero.  Verified
  EXACT on all train/test/arc-gen (265 pairs).
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
G = HEIGHT  # 30


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                       #
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

    def i64(self, dims, vals):
        n = self.nm("i")
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
# numpy reference                                                              #
# --------------------------------------------------------------------------- #
def _ref(a):
    H, W = a.shape
    T = np.tile(a, (2, 2))
    out = T.copy()
    cc = (T != 0).astype(int)
    p = np.pad(cc, 1)
    Dn = (p[0:2 * H, 0:2 * W] + p[0:2 * H, 2:2 * W + 2] +
          p[2:2 * H + 2, 0:2 * W] + p[2:2 * H + 2, 2:2 * W + 2])
    out[(Dn > 0) & (T == 0)] = 8
    return out


# --------------------------------------------------------------------------- #
# builder                                                                      #
# --------------------------------------------------------------------------- #
def build_tile_halo():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    two = g.f([1, 1, 1, 1], [2.0])

    # integer coordinate grids
    Ii = g.i64([1, 1, G, G], [[i for _ in range(G)] for i in range(G)])
    Ji = g.i64([1, 1, G, G], [[j for j in range(G)] for _ in range(G)])
    If = g.nd("Cast", [Ii], to=F)
    Jf = g.nd("Cast", [Ji], to=F)

    # real-cell mask + grid size H, W (one-hot: channel-sum == 1 on real cells)
    real = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)               # [1,1,30,30]
    rowhas = g.nd("ReduceMax", [real], axes=[3], keepdims=1)                 # [1,1,30,1]
    Hf = g.nd("ReduceSum", [rowhas], axes=[2], keepdims=1)                   # [1,1,1,1]
    colhas = g.nd("ReduceMax", [real], axes=[2], keepdims=1)                 # [1,1,1,30]
    Wf = g.nd("ReduceSum", [colhas], axes=[3], keepdims=1)                   # [1,1,1,1]
    Hi = g.nd("Cast", [Hf], to=INT64)
    Wi = g.nd("Cast", [Wf], to=INT64)

    twoH = g.nd("Mul", [two, Hf])
    twoW = g.nd("Mul", [two, Wf])

    # S_row[i,j] = 1 iff j == (i mod H) and i < 2H
    Imod = g.nd("Cast", [g.nd("Mod", [Ii, Hi])], to=F)
    matchr = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [Jf, Imod])]), half])], to=F)
    validr = g.nd("Cast", [g.nd("Less", [If, g.nd("Sub", [twoH, half])])], to=F)
    Srow = g.nd("Mul", [matchr, validr])                                    # [1,1,30,30]

    # S_col[w,c] = 1 iff w == (c mod W) and c < 2W
    Jmod = g.nd("Cast", [g.nd("Mod", [Ji, Wi])], to=F)
    matchc = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [If, Jmod])]), half])], to=F)
    validc = g.nd("Cast", [g.nd("Less", [Jf, g.nd("Sub", [twoW, half])])], to=F)
    Scol = g.nd("Mul", [matchc, validc])                                    # [1,1,30,30]

    # tile: out = S_row @ input @ S_col
    tiled = g.nd("MatMul", [g.nd("MatMul", [Srow, "input"]), Scol])         # [1,10,30,30]

    # coloured mask (channels 1..9) and background-in-region
    cols19 = g.nd("Slice", [tiled, g.i64([1], [1]), g.i64([1], [CHANNELS]), g.i64([1], [1])])
    colored = g.nd("ReduceSum", [cols19], axes=[1], keepdims=1)             # [1,1,30,30]
    total = g.nd("ReduceSum", [tiled], axes=[1], keepdims=1)                # [1,1,30,30]
    bgreg = g.nd("Sub", [total, colored])                                   # ch0 within region

    # diagonal halo via 3x3 corner conv over coloured mask
    wd = g.f([1, 1, 3, 3], [1, 0, 1, 0, 0, 0, 1, 0, 1])
    Dconv = g.nd("Conv", [colored, wd], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    Dbin = g.nd("Cast", [g.nd("Greater", [Dconv, half])], to=F)
    eight = g.nd("Mul", [Dbin, bgreg])                                      # [1,1,30,30]

    # delta: subtract from ch0, add to ch8
    vec = g.f([1, CHANNELS, 1, 1], [-1.0 if c == 0 else (1.0 if c == 8 else 0.0)
                                    for c in range(CHANNELS)])
    delta = g.nd("Mul", [eight, vec])                                       # [1,10,30,30]
    g.nd("Add", [tiled, delta], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# task 165 -- column rays from shape through dots                              #
# --------------------------------------------------------------------------- #
def _has_self8(m):
    H, W = m.shape
    cnt = np.zeros((H, W), int)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            sh = np.zeros((H, W), int)
            rs0, rs1 = max(0, dr), H + min(0, dr)
            cs0, cs1 = max(0, dc), W + min(0, dc)
            sh[rs0:rs1, cs0:cs1] = m[rs0 - dr:rs1 - dr, cs0 - dc:cs1 - dc]
            cnt += sh
    return cnt


def _ref165(a):
    H, W = a.shape
    out = a.copy()
    colors = [int(c) for c in np.unique(a) if c != 0]
    if len(colors) != 2:
        return None
    iso = {}
    for col in colors:
        m = (a == col).astype(int)
        cnt = _has_self8(m)
        iso[col] = int(((m == 1) & (cnt == 0)).sum())
    shapecol = min(colors, key=lambda c: iso[c])
    dotcol = max(colors, key=lambda c: iso[c])
    if shapecol == dotcol or iso[shapecol] == iso[dotcol]:
        return None
    shape = (a == shapecol)
    dot = (a == dotcol)
    for c in range(W):
        rows = np.where(shape[:, c])[0]
        if len(rows) == 0:
            continue
        lr = rows.max()
        if dot[lr + 1:, c].any():
            out[lr + 1:, c] = dotcol
    return out


def build_lines165():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    big = g.f([1, 1, 1, 1], [10000.0])
    chmask = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))    # exclude bg
    Rg = g.f([1, 1, G, G], [[i for _ in range(G)] for i in range(G)])    # row index
    kdw = g.f([CHANNELS, 1, 3, 3], [[[1, 1, 1], [1, 0, 1], [1, 1, 1]]] * CHANNELS)

    # per-channel isolation count
    nbr = g.nd("Conv", ["input", kdw], group=CHANNELS, kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    isocell = g.nd("Mul", ["input", g.nd("Cast", [g.nd("Less", [nbr, half])], to=F)])
    isocnt = g.nd("ReduceSum", [isocell], axes=[2, 3], keepdims=1)        # [1,10,1,1]

    chsum = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)         # [1,10,1,1]
    active = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [chsum, half])], to=F), chmask])

    # shape channel = active fg channel with minimum isolation
    isoadj = g.nd("Add", [isocnt, g.nd("Mul", [g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), active]), big])])
    minIso = g.nd("ReduceMin", [isoadj], axes=[1], keepdims=1)            # [1,1,1,1]
    shapesel = g.nd("Mul", [active, g.nd("Cast", [g.nd("Less", [isocnt, g.nd("Add", [minIso, half])])], to=F)])
    dotsel = g.nd("Sub", [active, shapesel])                             # [1,10,1,1]

    shape_m = g.nd("ReduceSum", [g.nd("Mul", ["input", shapesel])], axes=[1], keepdims=1)   # [1,1,30,30]
    dot_m = g.nd("ReduceSum", [g.nd("Mul", ["input", dotsel])], axes=[1], keepdims=1)       # [1,1,30,30]

    real = g.nd("Cast", [g.nd("Greater", [g.nd("ReduceSum", ["input"], axes=[1], keepdims=1), half])], to=F)

    lr = g.nd("ReduceMax", [g.nd("Mul", [Rg, shape_m])], axes=[2], keepdims=1)   # [1,1,1,30]
    shapecol = g.nd("ReduceMax", [shape_m], axes=[2], keepdims=1)                # [1,1,1,30]

    below = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [g.nd("Sub", [Rg, lr]), half])], to=F), shapecol])
    dotbelowcol = g.nd("ReduceMax", [g.nd("Mul", [dot_m, below])], axes=[2], keepdims=1)     # [1,1,1,30]
    selected = g.nd("Mul", [shapecol, dotbelowcol])                              # [1,1,1,30]

    fill = g.nd("Mul", [g.nd("Mul", [below, selected]), real])                   # [1,1,30,30]

    kept = g.nd("Mul", ["input", g.nd("Sub", [g.f([1, 1, 1, 1], [1.0]), fill])])
    filldot = g.nd("Mul", [fill, dotsel])
    g.nd("Add", [kept, filldot], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / entry point                                                      #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            out.append((a, b))
    return out


def _tile_candidate(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return []
    for a, b in allp:
        H, W = a.shape
        if 2 * H > G or 2 * W > G:
            return []
        if b.shape != (2 * H, 2 * W):
            return []
        if not np.array_equal(_ref(a), b):
            return []
    try:
        model = build_tile_halo()
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    return [("tile2x2_halo8", model)]


def _lines165_candidate(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return []
    for a, b in allp:
        if a.shape != b.shape or max(a.shape) > G:
            return []
        r = _ref165(a)
        if r is None or not np.array_equal(r, b):
            return []
    try:
        model = build_lines165()
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    return [("col_rays165", model)]


def candidates(ex):
    out = []
    out += _tile_candidate(ex)
    out += _lines165_candidate(ex)
    return out
