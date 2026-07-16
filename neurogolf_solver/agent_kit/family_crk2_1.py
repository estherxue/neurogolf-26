"""family_crk2_1 -- a grab-bag of exact ARC->ONNX solvers (opset 10, static shapes).

Each rule is detected STRUCTURALLY and validated EXACTLY (a numpy mirror of the
ONNX semantics) on every train+test+arc-gen pair before a candidate is emitted,
so wrong hypotheses are dropped before the grader sees them.

Rules implemented
-----------------
  croptile     crop to the non-bg bounding box, then tile it kx*ky times.
  stampctr     stamp the (multi-colour) template centred on a lone marker cell
               (data-dependent translation via computed shift-matrix + MatMul).
  spray        single cell -> checkerboard of colour 4 above it (rows<=r, cols of
               same parity), the cell itself dropped one row.
  sym1x1       3x3 pattern -> single colour: 1 if h&v symmetric else 7.
  symhole      dense symmetric grid with a rectangular hole -> the hole content
               recovered by the grid's mirror symmetry, cropped to the hole.
  headercycle  row0 = W colours, row1 separator; rows>=2 cycle row0 down the grid.
  edgelines    a colour found only on the border, in opposite-edge pairs, drawn as
               full rows/columns.
  quadpick     a full-row + full-col separator splits the grid into 4 quadrants;
               output the odd-one-out quadrant.
  objmarker    extract the 8-connected object adjacent to a unique marker cell.
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
# tiny graph accumulator                                                       #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0
        self._cache = {}

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

    def scalar(self, v):
        key = ("s", float(v))
        if key not in self._cache:
            self._cache[key] = self.f([1, 1, 1, 1], [float(v)])
        return self._cache[key]


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _consts(g):
    g.rowidx = g.f([1, 1, H, 1], list(range(H)))
    g.colidx = g.f([1, 1, 1, W], list(range(W)))
    g.half = g.scalar(0.5)
    g.one = g.scalar(1.0)
    g.cbig = g.scalar(_CBIG)


def _plane(g, ch):
    return g.nd("Slice", ["input", g.i64([ch]), g.i64([ch + 1]), g.i64([1])])


def _nonbg(g):
    rs = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    ch0 = _plane(g, 0)
    return g.nd("Sub", [rs, ch0])


def _bbox_of(g, content):
    """content [1,1,30,30] (1 on cells). Return minrow,maxrow,mincol,maxcol,bh,bw."""
    rowidx, colidx, cbig, one = g.rowidx, g.colidx, g.cbig, g.one
    rowhas = g.nd("ReduceMax", [content], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [content], axes=[2], keepdims=1)
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    bh = g.nd("Add", [g.nd("Sub", [maxrow, minrow]), one])
    bw = g.nd("Add", [g.nd("Sub", [maxcol, mincol]), one])
    return minrow, maxrow, mincol, maxcol, bh, bw


def _col_sel_tiled(g, mincol, bw, kx):
    """Scol[y=axis2,x=axis3]: dest col x (0..kx*bw) <- src col (x mod bw)+mincol."""
    rowidx, colidx, half = g.rowidx, g.colidx, g.half
    acc = None
    for t in range(kx):
        tbw = g.nd("Mul", [bw, g.scalar(t)]) if t else g.scalar(0.0)
        t1bw = g.nd("Mul", [bw, g.scalar(t + 1)])
        diff = g.nd("Sub", [g.nd("Sub", [g.nd("Add", [colidx, mincol]), tbw]), rowidx])
        match = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), half])], to=F)
        lo = g.nd("Cast", [g.nd("Greater", [colidx, g.nd("Sub", [tbw, half])])], to=F)
        hi = g.nd("Cast", [g.nd("Less", [colidx, t1bw])], to=F)
        term = g.nd("Mul", [match, g.nd("Mul", [lo, hi])])
        acc = term if acc is None else g.nd("Add", [acc, term])
    return acc


def _row_sel_tiled(g, minrow, bh, ky):
    """Srow[i=axis2,k=axis3]: dest row i (0..ky*bh) <- src row (i mod bh)+minrow."""
    rowidx, colidx, half = g.rowidx, g.colidx, g.half
    acc = None
    for s in range(ky):
        sbh = g.nd("Mul", [bh, g.scalar(s)]) if s else g.scalar(0.0)
        s1bh = g.nd("Mul", [bh, g.scalar(s + 1)])
        # dest row i (=rowidx) in [s*bh,(s+1)*bh) <- src row k (=colidx) = (i-s*bh)+minrow
        diff = g.nd("Sub", [g.nd("Add", [colidx, sbh]), g.nd("Add", [rowidx, minrow])])
        match = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), half])], to=F)
        lo = g.nd("Cast", [g.nd("Greater", [rowidx, g.nd("Sub", [sbh, half])])], to=F)
        hi = g.nd("Cast", [g.nd("Less", [rowidx, s1bh])], to=F)
        term = g.nd("Mul", [match, g.nd("Mul", [lo, hi])])
        acc = term if acc is None else g.nd("Add", [acc, term])
    return acc


# =========================================================================== #
# RULE: croptile                                                               #
# =========================================================================== #
def build_croptile(kx, ky):
    g = _G()
    _consts(g)
    content = _nonbg(g)
    minrow, _, mincol, _, bh, bw = _bbox_of(g, content)
    Scol = _col_sel_tiled(g, mincol, bw, kx)
    Srow = _row_sel_tiled(g, minrow, bh, ky)
    shift1 = g.nd("MatMul", ["input", Scol])
    g.nd("MatMul", [Srow, shift1], "output")
    return _model(g)


def _shift_any(g, X, dy, dx):
    """Rigidly translate [1,C,30,30] X by scalar (dy,dx) (zero fill) via 2 MatMuls."""
    rowidx, colidx, half = g.rowidx, g.colidx, g.half
    diffr = g.nd("Sub", [g.nd("Sub", [rowidx, colidx]), dy])
    srow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diffr]), half])], to=F)
    rowsh = g.nd("MatMul", [srow, X])
    diffc = g.nd("Sub", [g.nd("Sub", [colidx, rowidx]), dx])
    scol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diffc]), half])], to=F)
    return g.nd("MatMul", [rowsh, scol])


def _realmask(g):
    return g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)


# =========================================================================== #
# RULE: stampctr -- stamp template centred on a lone marker cell               #
# =========================================================================== #
def build_stampctr(mcol):
    g = _G()
    _consts(g)
    half = g.half
    vmask = g.f([1, CHANNELS, 1, 1], [0.0 if c == mcol else 1.0 for c in range(CHANNELS)])
    tmpl = g.nd("Mul", ["input", vmask])            # input without the marker channel
    ch0 = _plane(g, 0)
    content_t = g.nd("Sub", [g.nd("ReduceSum", [tmpl], axes=[1], keepdims=1), ch0])
    minr, maxr, minc, maxc, _, _ = _bbox_of(g, content_t)
    cr = g.nd("Mul", [g.nd("Add", [minr, maxr]), half])
    cc = g.nd("Mul", [g.nd("Add", [minc, maxc]), half])
    MK = _plane(g, mcol)
    mr, _, mc, _, _, _ = _bbox_of(g, MK)
    dy = g.nd("Sub", [mr, cr])
    dx = g.nd("Sub", [mc, cc])
    shifted = _shift_any(g, tmpl, dy, dx)
    nb_shift = g.nd("Sub", [g.nd("ReduceSum", [shifted], axes=[1], keepdims=1),
                            g.nd("Slice", [shifted, g.i64([0]), g.i64([1]), g.i64([1])])])
    present = g.nd("Greater", [nb_shift, half])
    real = g.nd("Greater", [_realmask(g), half])
    cond = g.nd("And", [present, real])
    g.nd("Where", [cond, shifted, "input"], "output")
    return _model(g)


def _bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _isolated_cells(a):
    h, w = a.shape
    res = []
    for r in range(h):
        for c in range(w):
            if a[r, c] == 0:
                continue
            iso = all(not (0 <= r + dr < h and 0 <= c + dc < w and a[r + dr, c + dc] != 0)
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
            if iso:
                res.append((r, c))
    return res


def _ref_stampctr(a, mcol):
    h, w = a.shape
    ys, xs = np.where(a == mcol)
    if len(ys) != 1:
        return None
    mr, mc = int(ys[0]), int(xs[0])
    # marker must be isolated (no non-bg 8-neighbour)
    if any(0 <= mr + dr < h and 0 <= mc + dc < w and a[mr + dr, mc + dc] != 0
           for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
        return None
    tmpl = a.copy()
    tmpl[mr, mc] = 0
    bb = _bbox(tmpl != 0)
    if bb is None:
        return None
    tr0, tr1, tc0, tc1 = bb
    if (tr0 + tr1) % 2 or (tc0 + tc1) % 2:
        return None
    dy = mr - (tr0 + tr1) // 2
    dx = mc - (tc0 + tc1) // 2
    out = a.copy()
    for r in range(tr0, tr1 + 1):
        for c in range(tc0, tc1 + 1):
            v = tmpl[r, c]
            if v:
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = v
    return out


def _ref_croptile(a, kx, ky):
    bb = _bbox(a != 0)
    if bb is None:
        return None
    r0, r1, c0, c1 = bb
    C = a[r0:r1 + 1, c0:c1 + 1]
    return np.tile(C, (ky, kx))


# =========================================================================== #
# RULE: spray -- single cell -> checkerboard of 4 above + cell dropped 1 row   #
# =========================================================================== #
def build_spray():
    g = _G()
    _consts(g)
    half, one = g.half, g.one
    realm = _realmask(g)                                    # 0/1 grid mask
    M = g.nd("Sub", [realm, _plane(g, 0)])                  # non-bg (single cell)
    minr, maxr, minc, maxc, _, _ = _bbox_of(g, M)
    r, c = maxr, maxc
    # colour of the cell -> one-hot vec [1,10,1,1]
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    bgsupp = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    Kvec = g.nd("Mul", [counts, bgsupp])                    # 1 at K, else 0
    # checkerboard mask4
    rowmask = g.nd("Cast", [g.nd("Less", [g.rowidx, g.nd("Add", [r, half])])], to=F)  # rows<=r
    colpar = g.f([1, 1, 1, W], [j % 2 for j in range(W)])
    cpar = g.nd("Cast", [g.nd("Mod", [g.nd("Cast", [c], to=INT64), g.i64([2], dims=[1, 1, 1, 1])],
                                fmod=0)], to=F)
    parmask = g.nd("Cast", [g.nd("Less",
                [g.nd("Abs", [g.nd("Sub", [colpar, cpar])]), half])], to=F)   # [1,1,1,30]
    mask4 = g.nd("Mul", [g.nd("Mul", [rowmask, parmask]), realm])   # [1,1,30,30]
    # dropped cell at (r+1, c)
    rsel = g.nd("Cast", [g.nd("Less",
            [g.nd("Abs", [g.nd("Sub", [g.rowidx, g.nd("Add", [r, one])])]), half])], to=F)
    csel = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.colidx, c])]), half])], to=F)
    dropmask = g.nd("Mul", [g.nd("Mul", [rsel, csel]), realm])
    # assemble one-hot output
    e4 = g.f([1, CHANNELS, 1, 1], [1.0 if ch == 4 else 0.0 for ch in range(CHANNELS)])
    plane4 = g.nd("Mul", [mask4, e4])
    planeK = g.nd("Mul", [dropmask, Kvec])
    covered = g.nd("Add", [mask4, dropmask])
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    bgmask = g.nd("Mul", [realm, g.nd("Sub", [one, covered])])
    bgplane = g.nd("Mul", [bgmask, e0])
    g.nd("Add", [g.nd("Add", [plane4, planeK]), bgplane], "output")
    return _model(g)


def _ref_spray(a):
    h, w = a.shape
    ys, xs = np.where(a != 0)
    if len(ys) != 1:
        return None
    r, c, K = int(ys[0]), int(xs[0]), int(a[ys[0], xs[0]])
    out = np.zeros_like(a)
    for i in range(0, r + 1):
        for j in range(w):
            if j % 2 == c % 2:
                out[i, j] = 4
    if r + 1 < h:
        out[r + 1, c] = K
    return out


# =========================================================================== #
# RULE: headercycle -- row0 colours cycled down the grid below a separator     #
# =========================================================================== #
def build_headercycle():
    g = _G()
    _consts(g)
    half = g.half
    realm = _realmask(g)
    _, _, minc, maxc, _, bw = _bbox_of(g, realm)            # grid width bw, mincol=0
    _, maxr, _, _, _, _ = _bbox_of(g, realm)                # grid bottom row
    H_ = g.nd("Add", [maxr, g.one])                         # grid height
    # COLT[.,.,k,0] = one-hot of row0[k]
    R0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([2])])   # [1,10,1,30]
    COLT = g.nd("Transpose", [R0], perm=[0, 1, 3, 2])       # [1,10,30,1]
    # P[r(axis2),k(axis3)] = 1 iff k==(r-2) mod bw, 2<=r<H, 0<=k<bw  (unrolled bands)
    rowidx, colidx = g.rowidx, g.colidx
    acc = None
    for m in range(15):
        off = g.nd("Add", [g.scalar(2.0), g.nd("Mul", [bw, g.scalar(m)])])  # 2 + m*bw
        diff = g.nd("Sub", [g.nd("Sub", [rowidx, colidx]), off])            # (r-k)-off
        band = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), half])], to=F)
        acc = band if acc is None else g.nd("Add", [acc, band])
    klt = g.nd("Cast", [g.nd("Less", [colidx, bw])], to=F)                  # k<bw
    rlt = g.nd("Cast", [g.nd("Less", [rowidx, H_])], to=F)                  # r<H
    P = g.nd("Mul", [g.nd("Mul", [acc, klt]), rlt])
    sel = g.nd("MatMul", [P, COLT])                          # [1,10,30,1]
    colmask = g.nd("Cast", [g.nd("Less", [colidx, bw])], to=F)   # cols<bw [1,1,1,30]
    body = g.nd("Mul", [sel, colmask])                      # [1,10,30,30]
    # keep rows 0,1 from input; rows>=2 from body
    keep = g.f([1, 1, H, 1], [1.0 if i < 2 else 0.0 for i in range(H)])
    head = g.nd("Mul", ["input", keep])
    g.nd("Add", [head, body], "output")
    return _model(g)


def _ref_headercycle(a):
    h, w = a.shape
    if h < 3:
        return None
    seq = a[0, :].copy()
    out = a.copy()
    for r in range(2, h):
        out[r, :] = seq[(r - 2) % w]
    return out


# =========================================================================== #
# RULE: edgelines -- a border-only colour drawn as full rows/cols              #
# =========================================================================== #
def build_edgelines(colors):
    """For each colour c in `colors`: draw full row r if grid edges (r,0)&(r,W-1)
    are c; full col similarly.  Output = only those lines."""
    g = _G()
    _consts(g)
    half, one = g.half, g.one
    realm = _realmask(g)
    minr, maxr, minc, maxc, _, _ = _bbox_of(g, realm)
    # left/right/top/bottom edge selector planes (within grid)
    left = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.colidx, minc])]), half])], to=F)   # [1,1,1,30]
    right = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.colidx, maxc])]), half])], to=F)
    top = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, minr])]), half])], to=F)    # [1,1,30,1]
    bot = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.rowidx, maxr])]), half])], to=F)
    # interior mask: grid cells strictly inside the border
    rin = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [g.rowidx, g.nd("Add", [minr, half])])], to=F),
                       g.nd("Cast", [g.nd("Less", [g.rowidx, g.nd("Sub", [maxr, half])])], to=F)])
    cin = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [g.colidx, g.nd("Add", [minc, half])])], to=F),
                       g.nd("Cast", [g.nd("Less", [g.colidx, g.nd("Sub", [maxc, half])])], to=F)])
    interior = g.nd("Mul", [g.nd("Mul", [rin, cin]), realm])
    acc_nb = None      # non-bg presence
    acc_oh = None      # one-hot accumulation
    for c in colors:
        pc = _plane(g, c)                                   # [1,1,30,30]
        # gate: only colours that appear ONLY on the border in THIS grid
        has_int = g.nd("ReduceMax", [g.nd("Mul", [pc, interior])], axes=[2, 3], keepdims=1)  # [1,1,1,1]
        allow = g.nd("Sub", [one, g.nd("Cast", [g.nd("Greater", [has_int, half])], to=F)])
        # cells of colour c on the left edge collapsed to a per-row flag
        lflag = g.nd("ReduceMax", [g.nd("Mul", [pc, left])], axes=[3], keepdims=1)   # [1,1,30,1]
        rflag = g.nd("ReduceMax", [g.nd("Mul", [pc, right])], axes=[3], keepdims=1)
        rowflag = g.nd("Mul", [lflag, rflag])              # both ends -> draw row
        tflag = g.nd("ReduceMax", [g.nd("Mul", [pc, top])], axes=[2], keepdims=1)    # [1,1,1,30]
        bflag = g.nd("ReduceMax", [g.nd("Mul", [pc, bot])], axes=[2], keepdims=1)
        colflag = g.nd("Mul", [tflag, bflag])              # both ends -> draw col
        # line presence within the grid (gated to border-only colours)
        lines = g.nd("Mul", [g.nd("Mul", [g.nd("Cast", [g.nd("Greater",
                  [g.nd("Add", [rowflag, colflag]), half])], to=F), realm]), allow])  # [1,1,30,30]
        ec = g.f([1, CHANNELS, 1, 1], [1.0 if ch == c else 0.0 for ch in range(CHANNELS)])
        oh_c = g.nd("Mul", [lines, ec])
        acc_nb = lines if acc_nb is None else g.nd("Add", [acc_nb, lines])
        acc_oh = oh_c if acc_oh is None else g.nd("Add", [acc_oh, oh_c])
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    bgmask = g.nd("Mul", [realm, g.nd("Sub", [one, acc_nb])])
    g.nd("Add", [acc_oh, g.nd("Mul", [bgmask, e0])], "output")
    return _model(g)


def _ref_edgelines(a):
    h, w = a.shape
    cols = []
    for c in range(1, CHANNELS):
        ys, xs = np.where(a == c)
        if len(ys) == 0:
            continue
        if all(y == 0 or y == h - 1 or x == 0 or x == w - 1 for y, x in zip(ys, xs)):
            cols.append(c)
    out = np.zeros_like(a)
    drew = False
    for c in cols:
        for r in range(h):
            if a[r, 0] == c and a[r, w - 1] == c:
                out[r, :] = c
                drew = True
        for col in range(w):
            if a[0, col] == c and a[h - 1, col] == c:
                out[:, col] = c
                drew = True
    if not drew:
        return None
    return out


def _edgeline_colors(prs):
    cols = set()
    for a, b in prs:
        h, w = a.shape
        for c in range(1, CHANNELS):
            ys, xs = np.where(a == c)
            if len(ys) and all(y == 0 or y == h - 1 or x == 0 or x == w - 1
                               for y, x in zip(ys, xs)):
                cols.add(c)
    return sorted(cols)


# =========================================================================== #
# RULE: sym1x1 -- 3x3 pattern -> single colour (1 if h&v symmetric else 7)     #
# =========================================================================== #
def build_sym1x1(c_sym, c_asym):
    g = _G()
    _consts(g)
    half, one = g.half, g.one
    M = g.nd("Sub", [_realmask(g), _plane(g, 0)])          # binary pattern [1,1,30,30]
    col0 = g.nd("Slice", [M, g.i64([0]), g.i64([1]), g.i64([3])])   # [1,1,30,1]
    col2 = g.nd("Slice", [M, g.i64([2]), g.i64([3]), g.i64([3])])
    hmis = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [col0, col2])])], axes=[2, 3], keepdims=1)
    hsym = g.nd("Cast", [g.nd("Less", [hmis, half])], to=F)
    row0 = g.nd("Slice", [M, g.i64([0]), g.i64([1]), g.i64([2])])   # [1,1,1,30]
    row2 = g.nd("Slice", [M, g.i64([2]), g.i64([3]), g.i64([2])])
    vmis = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [row0, row2])])], axes=[2, 3], keepdims=1)
    vsym = g.nd("Cast", [g.nd("Less", [vmis, half])], to=F)
    both = g.nd("Mul", [hsym, vsym])                       # [1,1,1,1]
    e_s = g.f([1, CHANNELS, 1, 1], [1.0 if ch == c_sym else 0.0 for ch in range(CHANNELS)])
    e_a = g.f([1, CHANNELS, 1, 1], [1.0 if ch == c_asym else 0.0 for ch in range(CHANNELS)])
    vec = g.nd("Add", [g.nd("Mul", [both, e_s]), g.nd("Mul", [g.nd("Sub", [one, both]), e_a])])
    omask = g.f([1, 1, H, W], [[1.0 if (i == 0 and j == 0) else 0.0 for j in range(W)] for i in range(H)])
    g.nd("Mul", [vec, omask], "output")
    return _model(g)


def _ref_sym1x1(a, c_sym, c_asym):
    m = (a != 0)
    hs = np.array_equal(m, m[:, ::-1])
    vs = np.array_equal(m, m[::-1, :])
    return np.array([[c_sym if (hs and vs) else c_asym]])


# =========================================================================== #
# RULE: symhole -- symmetric NxN grid with a rectangular hole -> hole content  #
# =========================================================================== #
def build_symhole(N):
    g = _G()
    _consts(g)
    half = g.half
    Pc = np.zeros((W, W), np.float32)   # fliplr within NxN
    Pr = np.zeros((H, H), np.float32)   # flipud within NxN
    for i in range(N):
        Pc[N - 1 - i, i] = 1.0
        Pr[i, N - 1 - i] = 1.0
    Pcn = g.f([W, W], Pc)
    Prn = g.f([H, H], Pr)

    def lr(x):
        return g.nd("MatMul", [x, Pcn])

    def ud(x):
        return g.nd("MatMul", [Prn, x])

    ch0 = _plane(g, 0)                                     # hole cells (grid bg)
    cond0 = g.nd("Greater", [ch0, half])
    rec1 = g.nd("Where", [cond0, lr("input"), "input"])
    cond1 = g.nd("Greater", [g.nd("Slice", [rec1, g.i64([0]), g.i64([1]), g.i64([1])]), half])
    rec2 = g.nd("Where", [cond1, ud("input"), rec1])
    cond2 = g.nd("Greater", [g.nd("Slice", [rec2, g.i64([0]), g.i64([1]), g.i64([1])]), half])
    rec3 = g.nd("Where", [cond2, ud(lr("input")), rec2])
    # crop reconstructed grid to the hole bbox
    minr, _, minc, _, bh, bw = _bbox_of(g, ch0)
    Scol = _col_sel_tiled(g, minc, bw, 1)
    Srow = _row_sel_tiled(g, minr, bh, 1)
    g.nd("MatMul", [Srow, g.nd("MatMul", [rec3, Scol])], "output")
    return _model(g)


def _ref_symhole(a):
    bb = _bbox(a == 0)
    if bb is None:
        return None
    rec = a.copy()
    for f in (a[:, ::-1], a[::-1, :], a[::-1, ::-1]):
        rec = np.where(rec == 0, f, rec)
    r0, r1, c0, c1 = bb
    return rec[r0:r1 + 1, c0:c1 + 1]


# =========================================================================== #
# RULE: quadpick -- full-row+full-col separator; output the odd-one-out quad   #
# =========================================================================== #
def build_quadpick():
    g = _G()
    _consts(g)
    half, one = g.half, g.one
    rowidx, colidx = g.rowidx, g.colidx
    realm = _realmask(g)
    _, maxR, _, maxC, Hf, Wf = _bbox_of(g, realm)
    # unique monochrome row / col
    cnt_row = g.nd("ReduceSum", ["input"], axes=[3], keepdims=1)         # [1,10,30,1]
    rowmono = g.nd("ReduceMax", [cnt_row], axes=[1], keepdims=1)         # [1,1,30,1]
    mflag_r = g.nd("Cast", [g.nd("Greater", [rowmono, g.nd("Sub", [Wf, half])])], to=F)
    sr = g.nd("ReduceMax", [g.nd("Mul", [mflag_r, rowidx])], axes=[2], keepdims=1)  # [1,1,1,1]
    cnt_col = g.nd("ReduceSum", ["input"], axes=[2], keepdims=1)         # [1,10,1,30]
    colmono = g.nd("ReduceMax", [cnt_col], axes=[1], keepdims=1)         # [1,1,1,30]
    mflag_c = g.nd("Cast", [g.nd("Greater", [colmono, g.nd("Sub", [Hf, half])])], to=F)
    sc = g.nd("ReduceMax", [g.nd("Mul", [mflag_c, colidx])], axes=[3], keepdims=1)  # [1,1,1,1]
    # cross / off-cross
    rowsr = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, sr])]), half])], to=F)
    colsc = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, sc])]), half])], to=F)
    cross = g.nd("Cast", [g.nd("Greater", [g.nd("Add", [rowsr, colsc]), half])], to=F)
    off = g.nd("Sub", [one, cross])
    # D = argmax over off-cross colour counts (incl. background)
    offcounts = g.nd("ReduceSum", [g.nd("Mul", ["input", off])], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    D_arg = g.nd("ArgMax", [offcounts], axis=1, keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    D_oh = g.nd("Cast", [g.nd("Equal", [D_arg, idx])], to=F)
    mask_D = g.nd("ReduceSum", [g.nd("Mul", ["input", D_oh])], axes=[1], keepdims=1)
    anomaly = g.nd("Mul", [g.nd("Mul", [realm, off]), g.nd("Sub", [one, mask_D])])
    amin_r, _, amin_c, _, _, _ = _bbox_of(g, anomaly)
    below = g.nd("Cast", [g.nd("Greater", [amin_r, sr])], to=F)
    right = g.nd("Cast", [g.nd("Greater", [amin_c, sc])], to=F)
    top = g.nd("Mul", [below, g.nd("Add", [sr, one])])
    bottom = g.nd("Add", [g.nd("Mul", [below, maxR]),
                          g.nd("Mul", [g.nd("Sub", [one, below]), g.nd("Sub", [sr, one])])])
    left = g.nd("Mul", [right, g.nd("Add", [sc, one])])
    rightb = g.nd("Add", [g.nd("Mul", [right, maxC]),
                          g.nd("Mul", [g.nd("Sub", [one, right]), g.nd("Sub", [sc, one])])])
    bh = g.nd("Add", [g.nd("Sub", [bottom, top]), one])
    bw = g.nd("Add", [g.nd("Sub", [rightb, left]), one])
    Scol = _col_sel_tiled(g, left, bw, 1)
    Srow = _row_sel_tiled(g, top, bh, 1)
    g.nd("MatMul", [Srow, g.nd("MatMul", ["input", Scol])], "output")
    return _model(g)


def _ref_quadpick(a):
    h, w = a.shape
    monor = [r for r in range(h) if len(set(a[r, :])) == 1]
    monoc = [c for c in range(w) if len(set(a[:, c])) == 1]
    if len(monor) != 1 or len(monoc) != 1:
        return None
    sr, sc = monor[0], monoc[0]
    cross = np.zeros((h, w), bool)
    cross[sr, :] = True
    cross[:, sc] = True
    off = ~cross
    cnt = np.array([((a == col) & off).sum() for col in range(CHANNELS)])
    D = int(np.argmax(cnt))
    anom = off & (a != D)
    ys, xs = np.where(anom)
    if len(ys) == 0:
        return None
    amr, amc = int(ys.min()), int(xs.min())
    below, rt = amr > sr, amc > sc
    top = sr + 1 if below else 0
    bottom = (h - 1) if below else (sr - 1)
    left = sc + 1 if rt else 0
    rightb = (w - 1) if rt else (sc - 1)
    if top > bottom or left > rightb:
        return None
    return a[top:bottom + 1, left:rightb + 1]


# =========================================================================== #
# RULE: objmarker -- extract the 8-conn object adjacent to a unique marker     #
# =========================================================================== #
def _shift(g, x, dr, dc):
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0, pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
    return g.nd("Slice", [p, st, en, g.i64([2, 3])])


def _dilate8(g, X):
    nbrs = [X] + [_shift(g, X, dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr, dc) != (0, 0)]
    return g.nd("Max", nbrs)


def _dilate4(g, X):
    nbrs = [X] + [_shift(g, X, dr, dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    return g.nd("Max", nbrs)


def _evec(g, plus, minus=()):
    return g.f([1, CHANNELS, 1, 1],
               [(1.0 if c in plus else 0.0) - (1.0 if c in minus else 0.0)
                for c in range(CHANNELS)])


# =========================================================================== #
# RULE: localadj -- colour `src` next to colour `nbr` -> src->dst, nbr->bg     #
# =========================================================================== #
def build_localadj(src, nbr, dst):
    g = _G()
    _consts(g)
    ps = _plane(g, src)
    pn = _plane(g, nbr)
    becomes = g.nd("Mul", [ps, _dilate4(g, pn)])
    vanish = g.nd("Mul", [pn, _dilate4(g, ps)])
    v1 = _evec(g, {dst}, {src})
    v2 = _evec(g, {0}, {nbr})
    g.nd("Add", ["input", g.nd("Add", [g.nd("Mul", [becomes, v1]),
                                       g.nd("Mul", [vanish, v2])])], "output")
    return _model(g)


def _ref_localadj(a, src, nbr, dst):
    ps = (a == src)
    pn = (a == nbr)

    def dil4(P):
        h, w = P.shape
        Y = P.copy()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sh = np.zeros_like(P)
            sh[max(-dr, 0):h - max(dr, 0), max(-dc, 0):w - max(dc, 0)] = \
                P[max(dr, 0):h - max(-dr, 0), max(dc, 0):w - max(-dc, 0)]
            Y |= sh
        return Y
    out = a.copy()
    out[ps & dil4(pn)] = dst
    out[pn & dil4(ps)] = 0
    return out


# =========================================================================== #
# RULE: ell -- L-path: vertical in cv's column, horizontal in ch's row         #
# =========================================================================== #
def build_ell(cv, ch, cp):
    g = _G()
    _consts(g)
    half, one = g.half, g.one
    rowidx, colidx = g.rowidx, g.colidx
    pv = _plane(g, cv)
    ph = _plane(g, ch)
    rv, _, cv_, _, _, _ = _bbox_of(g, pv)        # marker cv at (rv, cv_)
    rh, _, ch_, _, _, _ = _bbox_of(g, ph)        # marker ch at (rh, ch_)
    rlo = g.nd("Min", [rv, rh]); rhi = g.nd("Max", [rv, rh])
    clo = g.nd("Min", [cv_, ch_]); chi = g.nd("Max", [cv_, ch_])
    csel = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, cv_])]), half])], to=F)  # [1,1,1,30]
    rrange = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [rowidx, g.nd("Sub", [rlo, half])])], to=F),
                          g.nd("Cast", [g.nd("Less", [rowidx, g.nd("Add", [rhi, half])])], to=F)])
    vmask = g.nd("Mul", [csel, rrange])
    rsel = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, rh])]), half])], to=F)  # [1,1,30,1]
    crange = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [colidx, g.nd("Sub", [clo, half])])], to=F),
                          g.nd("Cast", [g.nd("Less", [colidx, g.nd("Add", [chi, half])])], to=F)])
    hmask = g.nd("Mul", [rsel, crange])
    path = g.nd("Cast", [g.nd("Greater", [g.nd("Add", [vmask, hmask]), half])], to=F)
    path4 = g.nd("Mul", [g.nd("Mul", [path, g.nd("Sub", [one, pv])]), g.nd("Sub", [one, ph])])
    ep = _evec(g, {cp}); ev = _evec(g, {cv}); eh = _evec(g, {ch}); e0 = _evec(g, {0})
    covered = g.nd("Add", [path4, g.nd("Add", [pv, ph])])
    bg = g.nd("Mul", [_realmask(g), g.nd("Sub", [one, covered])])
    body = g.nd("Add", [g.nd("Mul", [path4, ep]),
                        g.nd("Add", [g.nd("Mul", [pv, ev]), g.nd("Mul", [ph, eh])])])
    g.nd("Add", [body, g.nd("Mul", [bg, e0])], "output")
    return _model(g)


def _ref_ell(a, cv, ch, cp):
    yv, xv = np.where(a == cv)
    yh, xh = np.where(a == ch)
    if len(yv) != 1 or len(yh) != 1:
        return None
    rv, cvc = int(yv[0]), int(xv[0])
    rh, chc = int(yh[0]), int(xh[0])
    out = np.zeros_like(a)
    for r in range(min(rv, rh), max(rv, rh) + 1):
        out[r, cvc] = cp
    for c in range(min(cvc, chc), max(cvc, chc) + 1):
        out[rh, c] = cp
    out[rv, cvc] = cv
    out[rh, chc] = ch
    return out


# =========================================================================== #
# RULE: countdiag -- count single-colour components -> NxN diagonal            #
# =========================================================================== #
def build_countdiag(T=12):
    g = _G()
    _consts(g)
    half = g.half
    rowidx, colidx = g.rowidx, g.colidx
    M = g.nd("Sub", [_realmask(g), _plane(g, 0)])
    P = g.f([1, 1, H, W], np.arange(1, H * W + 1))
    L = g.nd("Mul", [M, P])
    for _ in range(T):
        L = g.nd("Mul", [M, g.nd("Max", [L, _dilate4(g, L)])])
    rep = g.nd("Mul", [M, g.nd("Cast", [g.nd("Less",
              [g.nd("Abs", [g.nd("Sub", [L, P])]), half])], to=F)])
    N = g.nd("ReduceSum", [rep], axes=[2, 3], keepdims=1)              # [1,1,1,1]
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    bgneg = g.f([1, CHANNELS, 1, 1], [-_CBIG] + [0.0] * (CHANNELS - 1))
    Carg = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    eC = g.nd("Cast", [g.nd("Equal", [Carg, idx])], to=F)
    diag = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, colidx])]), half])], to=F)
    rlt = g.nd("Cast", [g.nd("Less", [rowidx, N])], to=F)
    clt = g.nd("Cast", [g.nd("Less", [colidx, N])], to=F)
    gridN = g.nd("Mul", [rlt, clt])
    dmask = g.nd("Mul", [diag, gridN])
    e0 = _evec(g, {0})
    bg = g.nd("Mul", [gridN, g.nd("Sub", [g.one, dmask])])
    g.nd("Add", [g.nd("Mul", [dmask, eC]), g.nd("Mul", [bg, e0])], "output")
    return _model(g)


def _ref_countdiag(a):
    h, w = a.shape
    cols = set(int(v) for v in np.unique(a) if v != 0)
    if len(cols) != 1:
        return None
    C = cols.pop()
    seen = np.zeros((h, w), bool)
    n = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == 0:
                continue
            n += 1
            q = deque([(i, j)])
            seen[i, j] = True
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    if n == 0 or n > 30:
        return None
    o = np.zeros((n, n), int)
    for i in range(n):
        o[i, i] = C
    return o


def build_objmarker(mcol, T=6):
    g = _G()
    _consts(g)
    half = g.half
    vmask = g.f([1, CHANNELS, 1, 1], [0.0 if c == mcol else 1.0 for c in range(CHANNELS)])
    nomk = g.nd("Mul", ["input", vmask])                       # input minus marker channel
    M = g.nd("Sub", [g.nd("ReduceSum", [nomk], axes=[1], keepdims=1), _plane(g, 0)])  # non-bg, no marker
    MK = _plane(g, mcol)                                        # marker plane
    seed = g.nd("Mul", [M, _dilate8(g, MK)])
    Fc = seed
    for _ in range(T):
        Fc = g.nd("Mul", [M, _dilate8(g, Fc)])
    minr, _, minc, _, bh, bw = _bbox_of(g, Fc)
    Scol = _col_sel_tiled(g, minc, bw, 1)
    Srow = _row_sel_tiled(g, minr, bh, 1)
    g.nd("MatMul", [Srow, g.nd("MatMul", [nomk, Scol])], "output")
    return _model(g)


def _dilate8_np(X):
    h, w = X.shape
    Y = X.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            sh = np.zeros_like(X)
            sh[max(-dr, 0):h - max(dr, 0), max(-dc, 0):w - max(dc, 0)] = \
                X[max(dr, 0):h - max(-dr, 0), max(dc, 0):w - max(-dc, 0)]
            Y |= sh
    return Y


def _ref_objmarker(a, mcol, T=6):
    if (a == mcol).sum() != 1:
        return None
    ys, xs = np.where(a == mcol)
    mr, mc = int(ys[0]), int(xs[0])
    b = a.copy()
    b[mr, mc] = 0
    M = (b != 0)
    mk = np.zeros_like(M)
    mk[mr, mc] = True
    F = M & _dilate8_np(mk)
    for _ in range(T):
        F = M & _dilate8_np(F)
    if F.sum() == 0:
        return None
    rs, cs = np.where(F)
    return b[rs.min():rs.max() + 1, cs.min():cs.max() + 1]


# =========================================================================== #
# RULE: framecrop -- crop largest rectangle of the frame colour, recolour      #
# =========================================================================== #
def build_framecrop(T=30):
    g = _G()
    _consts(g)
    half = g.half
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    bgneg = g.f([1, CHANNELS, 1, 1], [-_CBIG] + [0.0] * (CHANNELS - 1))
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    f_arg = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
    f_oh = g.nd("Cast", [g.nd("Equal", [f_arg, idx])], to=F)          # frame colour [1,10,1,1]
    m_arg = g.nd("ArgMax", [g.nd("Add", [g.nd("Add", [counts, bgneg]),
                  g.nd("Mul", [f_oh, g.scalar(-_CBIG)])])], axis=1, keepdims=1)
    m_oh = g.nd("Cast", [g.nd("Equal", [m_arg, idx])], to=F)          # marker colour
    M = g.nd("ReduceSum", [g.nd("Mul", ["input", f_oh])], axes=[1], keepdims=1)  # frame plane
    # largest 4-connected component of M
    P = g.f([1, 1, H, W], np.arange(1, H * W + 1))
    L = g.nd("Mul", [M, P])
    for _ in range(T):
        L = g.nd("Mul", [M, g.nd("Max", [L, _dilate4(g, L)])])
    Lcol = g.nd("Reshape", [L, g.i64([H * W, 1])])
    Lrow = g.nd("Reshape", [L, g.i64([1, H * W])])
    E = g.nd("Cast", [g.nd("Equal", [g.nd("Cast", [Lcol], to=INT32), g.nd("Cast", [Lrow], to=INT32)])], to=F)
    size = g.nd("Reshape", [g.nd("ReduceSum", [E], axes=[1], keepdims=1), g.i64([1, 1, H, W])])
    size = g.nd("Mul", [size, M])
    maxsize = g.nd("ReduceMax", [size], axes=[2, 3], keepdims=1)
    Lg = g.nd("Cast", [g.nd("Greater", [size, g.nd("Sub", [maxsize, half])])], to=F)
    minr, _, minc, _, bh, bw = _bbox_of(g, Lg)
    Scol = _col_sel_tiled(g, minc, bw, 1)
    Srow = _row_sel_tiled(g, minr, bh, 1)
    cropped = g.nd("MatMul", [Srow, g.nd("MatMul", ["input", Scol])])  # [1,10,30,30]
    fplane = g.nd("ReduceSum", [g.nd("Mul", [cropped, f_oh])], axes=[1], keepdims=1)
    recolor = g.nd("Sub", [m_oh, f_oh])
    g.nd("Add", [cropped, g.nd("Mul", [fplane, recolor])], "output")
    return _model(g)


def _ref_framecrop(a):
    cols = [c for c in range(1, CHANNELS) if (a == c).any()]
    if len(cols) != 2:
        return None
    counts = {c: int((a == c).sum()) for c in cols}
    frame = max(cols, key=lambda c: counts[c])
    marker = [c for c in cols if c != frame][0]
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    best = None
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] != frame:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            cells = [(i, j)]
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] == frame:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            if best is None or len(cells) > len(best):
                best = cells
    if not best:
        return None
    rs = [r for r, c in best]
    cs = [c for r, c in best]
    sub = a[min(rs):max(rs) + 1, min(cs):max(cs) + 1].copy()
    sub[sub == frame] = marker
    return sub


# =========================================================================== #
# RULE: loopfill -- recolour shapes that enclose a hole to `fill`              #
# =========================================================================== #
def build_loopfill(fill, T_out=30, T_sh=16):
    g = _G()
    _consts(g)
    half, one = g.half, g.one
    rowidx, colidx = g.rowidx, g.colidx
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    bg_arg = g.nd("ArgMax", [counts], axis=1, keepdims=1)             # most common = bg
    bg_oh = g.nd("Cast", [g.nd("Equal", [bg_arg, idx])], to=F)
    Mbg = g.nd("ReduceSum", [g.nd("Mul", ["input", bg_oh])], axes=[1], keepdims=1)
    realm = _realmask(g)
    minr, maxr, minc, maxc, _, _ = _bbox_of(g, realm)
    top = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, minr])]), half])], to=F)
    bot = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, maxr])]), half])], to=F)
    lft = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, minc])]), half])], to=F)
    rgt = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, maxc])]), half])], to=F)
    border = g.nd("Cast", [g.nd("Greater", [g.nd("Add", [g.nd("Add", [top, bot]),
                  g.nd("Add", [lft, rgt])]), half])], to=F)
    outside = g.nd("Mul", [Mbg, border])
    for _ in range(T_out):
        outside = g.nd("Mul", [Mbg, _dilate4(g, outside)])
    holes = g.nd("Mul", [Mbg, g.nd("Sub", [one, outside])])
    Msh = g.nd("Sub", [realm, Mbg])
    reached = g.nd("Mul", [Msh, _dilate4(g, holes)])
    for _ in range(T_sh):
        reached = g.nd("Mul", [Msh, _dilate4(g, reached)])
    cond = g.nd("Greater", [reached, half])
    efill = g.f([1, CHANNELS, 1, 1], [1.0 if c == fill else 0.0 for c in range(CHANNELS)])
    fillgrid = g.nd("Mul", [g.f([1, 1, H, W], np.ones((H, W))), efill])  # broadcast e_fill plane
    g.nd("Where", [cond, fillgrid, "input"], "output")
    return _model(g)


def _ref_loopfill(a, fill):
    from collections import deque as _dq
    h, w = a.shape
    vals, cnt = np.unique(a, return_counts=True)
    bg = int(vals[np.argmax(cnt)])
    outside = np.zeros((h, w), bool)
    q = _dq()
    for i in range(h):
        for j in range(w):
            if (i in (0, h - 1) or j in (0, w - 1)) and a[i, j] == bg and not outside[i, j]:
                outside[i, j] = True
                q.append((i, j))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and a[nr, nc] == bg and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    holes = (a == bg) & (~outside)
    seen = np.zeros((h, w), bool)
    out = a.copy()
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == bg:
                continue
            q = _dq([(i, j)])
            seen[i, j] = True
            cells = [(i, j)]
            touch = False
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if a[nr, nc] != bg and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                        if holes[nr, nc]:
                            touch = True
            if touch:
                for r, c in cells:
                    out[r, c] = fill
    return out


# =========================================================================== #
# RULE: anomcrop -- crop the component holding the most cells of colour `anom`  #
# =========================================================================== #
def build_anomcrop(anom, T=30):
    g = _G()
    _consts(g)
    half = g.half
    M = g.nd("Sub", [_realmask(g), _plane(g, 0)])
    P = g.f([1, 1, H, W], np.arange(1, H * W + 1))
    L = g.nd("Mul", [M, P])
    for _ in range(T):
        L = g.nd("Mul", [M, g.nd("Max", [L, _dilate4(g, L)])])
    Lcol = g.nd("Reshape", [L, g.i64([H * W, 1])])
    Lrow = g.nd("Reshape", [L, g.i64([1, H * W])])
    E = g.nd("Cast", [g.nd("Equal", [g.nd("Cast", [Lcol], to=INT32), g.nd("Cast", [Lrow], to=INT32)])], to=F)
    is_anom = g.nd("Reshape", [_plane(g, anom), g.i64([H * W, 1])])      # [900,1]
    cnt = g.nd("Reshape", [g.nd("MatMul", [E, is_anom]), g.i64([1, 1, H, W])])
    cnt = g.nd("Mul", [cnt, M])
    maxc = g.nd("ReduceMax", [cnt], axes=[2, 3], keepdims=1)
    Lg = g.nd("Cast", [g.nd("Greater", [cnt, g.nd("Sub", [maxc, half])])], to=F)
    Lg = g.nd("Mul", [Lg, M])
    minr, _, minc, _, bh, bw = _bbox_of(g, Lg)
    Scol = _col_sel_tiled(g, minc, bw, 1)
    Srow = _row_sel_tiled(g, minr, bh, 1)
    g.nd("MatMul", [Srow, g.nd("MatMul", ["input", Scol])], "output")
    return _model(g)


def _ref_anomcrop(a, anom):
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    best, bestc = None, -1
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
            cnt = sum(1 for r, c in cells if a[r, c] == anom)
            if cnt > bestc:
                bestc = cnt
                best = cells
    if not best or bestc <= 0:
        return None
    rs = [r for r, c in best]
    cs = [c for r, c in best]
    return a[min(rs):max(rs) + 1, min(cs):max(cs) + 1]


# --------------------------------------------------------------------------- #
# entry helpers                                                                #
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
    nontrivial = False
    for a, b in prs:
        try:
            o = fn(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
        if not np.array_equal(a, b):
            nontrivial = True
    return nontrivial


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
    out, seen = [], set()

    # ---- croptile ---------------------------------------------------------- #
    # only when output is an integer multiple of the cropped bbox in both dims
    try:
        ratios = set()
        ok = True
        for a, b in prs:
            bb = _bbox(a != 0)
            if bb is None:
                ok = False
                break
            r0, r1, c0, c1 = bb
            ch, cw = r1 - r0 + 1, c1 - c0 + 1
            if b.shape[0] % ch or b.shape[1] % cw:
                ok = False
                break
            ratios.add((b.shape[0] // ch, b.shape[1] // cw))
        if ok and len(ratios) == 1:
            ky, kx = next(iter(ratios))
            if 1 <= kx <= 5 and 1 <= ky <= 5 and (kx, ky) != (1, 1):
                if _matches(prs, lambda a: _ref_croptile(a, kx, ky)):
                    _emit(out, seen, f"croptile_{kx}x{ky}",
                          lambda: build_croptile(kx, ky))
    except Exception:
        pass

    same_shape = all(a.shape == b.shape for a, b in prs)

    # ---- stampctr ---------------------------------------------------------- #
    if same_shape:
        # marker colour candidates: a colour that is a lone isolated cell
        mcand = set()
        for a, b in prs:
            if np.array_equal(a, b):
                continue
            iso = _isolated_cells(a)
            for (r, c) in iso:
                if (a == a[r, c]).sum() == 1:
                    mcand.add(int(a[r, c]))
        for mcol in sorted(mcand):
            if _matches(prs, lambda a, mcol=mcol: _ref_stampctr(a, mcol)):
                _emit(out, seen, f"stampctr_m{mcol}",
                      lambda mcol=mcol: build_stampctr(mcol))

        # ---- spray --------------------------------------------------------- #
        if _matches(prs, _ref_spray):
            _emit(out, seen, "spray", build_spray)

        # ---- headercycle --------------------------------------------------- #
        if _matches(prs, _ref_headercycle):
            _emit(out, seen, "headercycle", build_headercycle)

        # ---- edgelines ----------------------------------------------------- #
        ecols = _edgeline_colors(prs)
        if ecols and _matches(prs, _ref_edgelines):
            _emit(out, seen, "edgelines", lambda ec=tuple(ecols): build_edgelines(ec))

        # ---- localadj ------------------------------------------------------ #
        dstmap, nbrset = set(), set()
        for a, b in prs:
            for r, c in zip(*np.where(a != b)):
                av, bv = int(a[r, c]), int(b[r, c])
                if av != 0 and bv != 0:
                    dstmap.add((av, bv))
                elif av != 0 and bv == 0:
                    nbrset.add(av)
        if len(dstmap) == 1 and len(nbrset) == 1:
            (src, dst) = next(iter(dstmap))
            nbr = next(iter(nbrset))
            if src != nbr and _matches(prs, lambda a, s=src, n=nbr, d=dst: _ref_localadj(a, s, n, d)):
                _emit(out, seen, f"localadj_{src}_{nbr}_{dst}",
                      lambda s=src, n=nbr, d=dst: build_localadj(s, n, d))

        # ---- loopfill ------------------------------------------------------ #
        fills = set()
        for a, b in prs:
            extra = set(int(v) for v in np.unique(b)) - set(int(v) for v in np.unique(a))
            fills |= extra
        for fill in sorted(fills):
            if _matches(prs, lambda a, fill=fill: _ref_loopfill(a, fill)):
                _emit(out, seen, f"loopfill_{fill}", lambda fill=fill: build_loopfill(fill))
                break

        # ---- ell (L-path between two markers) ------------------------------ #
        ell_ok = True
        m1 = m2 = cp = None
        for a, b in prs:
            inc = sorted(int(v) for v in np.unique(a) if v != 0)
            outc = set(int(v) for v in np.unique(b) if v != 0)
            extra = outc - set(inc)
            if len(inc) != 2 or len(extra) != 1:
                ell_ok = False
                break
            e = extra.pop()
            if m1 is None:
                m1, m2, cp = inc[0], inc[1], e
            elif (inc[0], inc[1], e) != (m1, m2, cp):
                ell_ok = False
                break
        if ell_ok and m1 is not None:
            for cvv, chh in ((m1, m2), (m2, m1)):
                if _matches(prs, lambda a, cvv=cvv, chh=chh, cp=cp: _ref_ell(a, cvv, chh, cp)):
                    _emit(out, seen, f"ell_{cvv}_{chh}_{cp}",
                          lambda cvv=cvv, chh=chh, cp=cp: build_ell(cvv, chh, cp))
                    break

    # ---- sym1x1 ------------------------------------------------------------ #
    if all(a.shape == (3, 3) and b.shape == (1, 1) for a, b in prs):
        sym_cols, asym_cols = set(), set()
        for a, b in prs:
            m = (a != 0)
            sym = np.array_equal(m, m[:, ::-1]) and np.array_equal(m, m[::-1, :])
            (sym_cols if sym else asym_cols).add(int(b[0, 0]))
        if len(sym_cols) == 1 and len(asym_cols) == 1 and sym_cols != asym_cols:
            cs, ca = sym_cols.pop(), asym_cols.pop()
            if _matches(prs, lambda a, cs=cs, ca=ca: _ref_sym1x1(a, cs, ca)):
                _emit(out, seen, f"sym1x1_{cs}_{ca}",
                      lambda cs=cs, ca=ca: build_sym1x1(cs, ca))

    # ---- quadpick ---------------------------------------------------------- #
    if any(b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1] for a, b in prs):
        if _matches(prs, _ref_quadpick):
            _emit(out, seen, "quadpick", build_quadpick)

    # ---- objmarker --------------------------------------------------------- #
    if any(b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1] for a, b in prs):
        mset = None
        for a, _ in prs:
            cs = {int(c) for c in range(1, CHANNELS) if (a == c).sum() == 1}
            mset = cs if mset is None else (mset & cs)
        for mcol in sorted(mset or ()):
            if _matches(prs, lambda a, mcol=mcol: _ref_objmarker(a, mcol)):
                _emit(out, seen, f"objmarker_m{mcol}",
                      lambda mcol=mcol: build_objmarker(mcol))

    # ---- anomcrop ---------------------------------------------------------- #
    if any(b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1] for a, b in prs):
        acols = set()
        for a, _ in prs:
            acols |= set(int(v) for v in np.unique(a) if v != 0)
        for anom in sorted(acols):
            if _matches(prs, lambda a, anom=anom: _ref_anomcrop(a, anom)):
                _emit(out, seen, f"anomcrop_{anom}", lambda anom=anom: build_anomcrop(anom))
                break

    # ---- framecrop --------------------------------------------------------- #
    if any(b.shape != a.shape for a, b in prs) and \
            any(b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1] for a, b in prs):
        if _matches(prs, _ref_framecrop):
            _emit(out, seen, "framecrop", build_framecrop)

    # ---- countdiag --------------------------------------------------------- #
    if all(b.shape[0] == b.shape[1] for a, b in prs) and any(b.shape != a.shape for a, b in prs):
        if _matches(prs, _ref_countdiag):
            _emit(out, seen, "countdiag", build_countdiag)

    # ---- symhole ----------------------------------------------------------- #
    insz = {a.shape for a, _ in prs}
    if (len(insz) == 1 and next(iter(insz))[0] == next(iter(insz))[1]
            and all(b.shape[0] < a.shape[0] for a, b in prs)):
        N = next(iter(insz))[0]
        if N <= 30 and _matches(prs, _ref_symhole):
            _emit(out, seen, f"symhole_{N}", lambda N=N: build_symhole(N))

    return out
