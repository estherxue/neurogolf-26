"""family_golf2_5 -- CHEAPER exact re-solvers for a slice of golf targets.

Each candidate re-derives the rule from train+test+arc-gen pairs, validates a
numpy mirror of the *exact* ONNX semantics on every available pair, and only then
emits a minimal opset-10 graph.  The integrator auto-picks the cheapest correct
solver, so we just need these to be exact and cheaper than the incumbent.

Golf levers used here:
  * single-channel [1,1,30,30] intermediates instead of [1,10,30,30];
  * write the final answer straight into the FREE `output` tensor via
    Concat / Pad / MatMul / Where (no extra full-size buffer);
  * constant pattern initializers (params, 1B/elem) instead of computed masks
    (memory, 4B/elem) where the pattern is data-independent.

Targets (rule -> incumbent cost):
  146  pick the unique non-symmetric 3x3 block of a 9x3 stack   (255 KB)
  150  mirror left-right within the (data-derived) grid width    (cheap incumbent)
  375  draw both diagonals as background on a solid colour square (121 KB)
  55   tic-tac-toe: fill the +-shaped cells around the centre     (468 KB)
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
F = DATA_TYPE
H, W = HEIGHT, WIDTH
BIG = 1000.0


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


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.i64(starts), g.i64(ends), g.i64(axes)]
    if steps is not None:
        ins.append(g.i64(steps))
    return g.nd("Slice", ins)


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


# ===========================================================================
# 146  pick the unique non-symmetric 3x3 block of a 9x3 stack
# ===========================================================================
def _mir_146(a):
    if a.shape != (9, 3):
        return None
    blocks = [a[3 * k:3 * k + 3, :] for k in range(3)]
    gates = [0 if np.array_equal(bl, bl.T) else 1 for bl in blocks]
    sel = np.zeros((3, 3), int)
    for k in range(3):
        if gates[k]:
            sel = sel + blocks[k]          # exact-mirror: sum of gated blocks
    return sel


def _build_146(g):
    half = g.f([1, 1, 1, 1], [0.5])
    sel = None
    for k in range(3):
        Bk = _slice(g, "input", [3 * k, 0], [3 * k + 3, 3], [2, 3])     # [1,10,3,3]
        Tk = g.nd("Transpose", [Bk], perm=[0, 1, 3, 2])
        Sk = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [Bk, Tk])])],
                  axes=[1, 2, 3], keepdims=1)                            # [1,1,1,1]
        gk = g.nd("Cast", [g.nd("Greater", [Sk, half])], to=F)          # [1,1,1,1]
        term = g.nd("Mul", [Bk, gk])                                    # [1,10,3,3]
        sel = term if sel is None else g.nd("Add", [sel, term])
    g.nd("Pad", [sel], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, H - 3, W - 3])
    return _model(g)


# ===========================================================================
# 150  mirror left-right within the data-derived grid width
# ===========================================================================
def _mir_150(a):
    h, w = a.shape
    cols = np.where(a.any(axis=0))[0]
    if cols.size == 0:
        return None
    wm1 = int(cols.max())                  # graph derives width from rightmost col
    o = np.zeros_like(a)
    for j in range(wm1 + 1):
        o[:, j] = a[:, wm1 - j]
    return o


def _build_150(g):
    colidx = g.f([1, 1, 1, W], list(range(W)))
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    half = g.f([1, 1, 1, 1], [0.5])
    colany = g.nd("ReduceMax", ["input"], axes=[1, 2], keepdims=1)      # [1,1,1,30]
    maxc = g.nd("ReduceMax", [g.nd("Mul", [colany, colidx])], axes=[3], keepdims=1)
    s = g.nd("Add", [rowidx, colidx])                                   # [1,1,30,30] (k+j)
    R = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [s, maxc])]), half])], to=F)
    g.nd("MatMul", ["input", R], "output")                             # [1,10,30,30]
    return _model(g)


# ===========================================================================
# 375  both diagonals -> background (0) on a solid colour square
# ===========================================================================
def _mir_375(a):
    h, w = a.shape
    if h != w:
        return None
    cols = np.where(a.any(axis=0))[0]
    if cols.size == 0:
        return None
    n = int(cols.max()) + 1
    cnt = np.array([(a == c).sum() for c in range(CHANNELS)])
    cnt[0] = -1
    dom = int(cnt.argmax())
    o = np.zeros((h, w), int)
    for r in range(h):
        for c in range(w):
            if r < n and c < n and not (r == c or r + c == n - 1):
                o[r, c] = dom
    return o


def _build_375(g):
    colidx = g.f([1, 1, 1, W], list(range(W)))
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    maindiag = np.eye(H, W, dtype=np.float32)
    cmain = g.f([1, 1, H, W], maindiag)                                 # const main diag
    onehot0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    # grid size N
    colany = g.nd("ReduceMax", ["input"], axes=[1, 2], keepdims=1)
    maxc = g.nd("ReduceMax", [g.nd("Mul", [colany, colidx])], axes=[3], keepdims=1)
    N = g.nd("Add", [maxc, one])
    grid = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [rowidx, N])], to=F),
                        g.nd("Cast", [g.nd("Less", [colidx, N])], to=F)])
    s = g.nd("Add", [rowidx, colidx])
    anti = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [s, maxc])]), half])], to=F)
    X = g.nd("Max", [cmain, anti])                                      # [1,1,30,30]
    condX = g.nd("Greater", [X, half])
    # dominant colour
    bgneg = g.f([1, CHANNELS, 1, 1], [-BIG] + [0.0] * (CHANNELS - 1))
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    amax = g.nd("ArgMax", [g.nd("Add", [counts, bgneg])], axis=1, keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    colorvec = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)         # [1,10,1,1]
    sel = g.nd("Where", [condX, onehot0, colorvec])                    # [1,10,30,30]
    g.nd("Mul", [sel, grid], "output")
    return _model(g)


# ===========================================================================
# 55   tic-tac-toe: fill the +-shaped cells around the centre cross
# ===========================================================================
def _ttt_lines(a, L):
    h, w = a.shape
    is8 = (a == L)
    vcols = [c for c in range(w) if is8[:, c].all()]
    hrows = [r for r in range(h) if is8[r, :].all()]
    if len(vcols) != 2 or len(hrows) != 2:
        return None
    return min(vcols), max(vcols), min(hrows), max(hrows)


def _ttt_params(prs):
    a0, b0 = prs[0]
    h, w = a0.shape
    for L in range(1, CHANNELS):
        ln = _ttt_lines(a0, L)
        if ln is None:
            continue
        lc, rc, tr, br = ln

        def sample(cond):
            for r in range(h):
                for c in range(w):
                    if a0[r, c] == 0 and cond(r, c):
                        return int(b0[r, c])
            return 0
        cc = sample(lambda r, c: lc < c < rc and tr < r < br)
        cu = sample(lambda r, c: lc < c < rc and r < tr)
        cd = sample(lambda r, c: lc < c < rc and r > br)
        cl = sample(lambda r, c: tr < r < br and c < lc)
        cr = sample(lambda r, c: tr < r < br and c > rc)
        return (L, cc, cu, cd, cl, cr)
    return None


def _mir_ttt(a, params):
    L, cc, cu, cd, cl, cr = params
    ln = _ttt_lines(a, L)
    if ln is None:
        return None
    lc, rc, tr, br = ln
    o = a.copy()
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            if a[r, c] != 0:
                continue
            bv = lc < c < rc
            bh = tr < r < br
            col = 0
            if bv and bh:
                col = cc
            elif bv and r < tr:
                col = cu
            elif bv and r > br:
                col = cd
            elif bh and c < lc:
                col = cl
            elif bh and c > rc:
                col = cr
            if col:
                o[r, c] = col
    return o


def _build_ttt(g, params):
    L, cc, cu, cd, cl, cr = params
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [BIG])
    colidx = g.f([1, 1, 1, W], list(range(W)))
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    zeroplane = g.f([1, 1, H, W], np.zeros((H, W), np.float32))

    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)                  # [1,1,30,30]
    is8 = _slice(g, "input", [L], [L + 1], [1])                                    # [1,1,30,30]
    bg = g.nd("Mul", [realmask, g.nd("Sub", [one, is8])])              # zero cells in grid

    # full vertical / horizontal 8-lines
    c8col = g.nd("ReduceSum", [is8], axes=[2], keepdims=1)             # [1,1,1,30]
    crcol = g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)
    vline = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [c8col, half])], to=F),
                         g.nd("Cast", [g.nd("Less",
                              [g.nd("Abs", [g.nd("Sub", [c8col, crcol])]), half])], to=F)])
    c8row = g.nd("ReduceSum", [is8], axes=[3], keepdims=1)             # [1,1,30,1]
    crrow = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)
    hline = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [c8row, half])], to=F),
                         g.nd("Cast", [g.nd("Less",
                              [g.nd("Abs", [g.nd("Sub", [c8row, crrow])]), half])], to=F)])

    lc = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [colidx, vline]),
              g.nd("Mul", [big, g.nd("Sub", [one, vline])])])], axes=[3], keepdims=1)
    rc = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [colidx, vline]),
              g.nd("Mul", [big, g.nd("Sub", [one, vline])])])], axes=[3], keepdims=1)
    tr = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [rowidx, hline]),
              g.nd("Mul", [big, g.nd("Sub", [one, hline])])])], axes=[2], keepdims=1)
    br = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [rowidx, hline]),
              g.nd("Mul", [big, g.nd("Sub", [one, hline])])])], axes=[2], keepdims=1)

    band_v = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [colidx, lc])], to=F),
                          g.nd("Cast", [g.nd("Less", [colidx, rc])], to=F)])  # [1,1,1,30]
    band_h = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [rowidx, tr])], to=F),
                          g.nd("Cast", [g.nd("Less", [rowidx, br])], to=F)])  # [1,1,30,1]
    r_lt = g.nd("Cast", [g.nd("Less", [rowidx, tr])], to=F)
    r_gt = g.nd("Cast", [g.nd("Greater", [rowidx, br])], to=F)
    c_lt = g.nd("Cast", [g.nd("Less", [colidx, lc])], to=F)
    c_gt = g.nd("Cast", [g.nd("Greater", [colidx, rc])], to=F)

    center = g.nd("Mul", [bg, g.nd("Mul", [band_v, band_h])])          # [1,1,30,30]
    up = g.nd("Mul", [bg, g.nd("Mul", [band_v, r_lt])])
    down = g.nd("Mul", [bg, g.nd("Mul", [band_v, r_gt])])
    left = g.nd("Mul", [bg, g.nd("Mul", [band_h, c_lt])])
    right = g.nd("Mul", [bg, g.nd("Mul", [band_h, c_gt])])

    # per-channel planes (accumulate; handles colour collisions)
    planes = [None] * CHANNELS
    for region, col in [(center, cc), (up, cu), (down, cd), (left, cl), (right, cr)]:
        if col == 0:
            continue
        planes[col] = region if planes[col] is None else g.nd("Add", [planes[col], region])
    # background corners = bg outside both mid-bands (filled cross = band_v | band_h)
    crossband = g.nd("Max", [band_v, band_h])                          # [1,1,30,30]
    ch0 = g.nd("Mul", [bg, g.nd("Sub", [one, crossband])])
    planes[0] = ch0 if planes[0] is None else g.nd("Add", [planes[0], ch0])
    # the line colour
    planes[L] = is8 if planes[L] is None else g.nd("Add", [planes[L], is8])

    parts = [planes[k] if planes[k] is not None else zeroplane for k in range(CHANNELS)]
    g.nd("Concat", parts, "output", axis=1)
    return _model(g)


# ===========================================================================
# 362  shift the X cross: vertical line left by N, horizontal down by N,
#      where N = number of colour-5 markers; markers are removed.
# ===========================================================================
def _t362_detect(a):
    h, w = a.shape
    cols = [c for c in range(1, CHANNELS) if c != 5 and (a == c).any()]
    if len(cols) != 1:
        return None
    X = cols[0]
    n5 = int((a == 5).sum())
    vcol = [c for c in range(w) if (a[:, c] == X).all()]
    hrow = [r for r in range(h) if (a[r, :] == X).all()]
    if len(vcol) != 1 or len(hrow) != 1:
        return None
    return X, n5, vcol[0], hrow[0]


def _mir_362(a):
    det = _t362_detect(a)
    if det is None:
        return None
    X, n5, vcol, hrow = det
    h, w = a.shape
    nv, nh = vcol - n5, hrow + n5
    if not (0 <= nv < w and 0 <= nh < h):
        return None
    o = np.zeros((h, w), int)
    o[:, nv] = X
    o[nh, :] = X
    return o


def _build_362(g):
    colidx = g.f([1, 1, 1, W], list(range(W)))
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    onehot0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    penal = g.f([1, CHANNELS, 1, 1],
                [-BIG if c in (0, 5) else 0.0 for c in range(CHANNELS)])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)       # [1,1,30,30] (grid)
    ch0 = _slice(g, "input", [0], [1], [1])                             # colour-0 cells
    five = _slice(g, "input", [5], [6], [1])                            # colour-5 markers
    Xplane = g.nd("Sub", [g.nd("Sub", [realmask, ch0]), five])         # X cells only
    # grid extent
    maxr = g.nd("ReduceMax", [g.nd("Mul",
              [g.nd("ReduceMax", ["input"], axes=[1, 3], keepdims=1), rowidx])],
              axes=[2], keepdims=1)
    maxc = g.nd("ReduceMax", [g.nd("Mul",
              [g.nd("ReduceMax", ["input"], axes=[1, 2], keepdims=1), colidx])],
              axes=[3], keepdims=1)
    Hg = g.nd("Add", [maxr, one])
    Wg = g.nd("Add", [maxc, one])
    colsum = g.nd("ReduceSum", [Xplane], axes=[2], keepdims=1)          # [1,1,1,30]
    rowsum = g.nd("ReduceSum", [Xplane], axes=[3], keepdims=1)          # [1,1,30,1]
    fullc = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colsum, Hg])]), half])], to=F)
    fullr = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowsum, Wg])]), half])], to=F)
    vcol = g.nd("ReduceSum", [g.nd("Mul", [fullc, colidx])], axes=[3], keepdims=1)
    hrow = g.nd("ReduceSum", [g.nd("Mul", [fullr, rowidx])], axes=[2], keepdims=1)
    N = g.nd("ReduceSum", [five], axes=[2, 3], keepdims=1)              # [1,1,1,1]
    nv = g.nd("Sub", [vcol, N])
    nh = g.nd("Add", [hrow, N])
    vmask = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, nv])]), half])], to=F)
    hmask = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, nh])]), half])], to=F)
    linec = g.nd("Greater", [g.nd("Max", [vmask, hmask]), half])        # bool [1,1,30,30]
    grid = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [rowidx, Hg])], to=F),
                        g.nd("Cast", [g.nd("Less", [colidx, Wg])], to=F)])
    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    amax = g.nd("ArgMax", [g.nd("Add", [counts, penal])], axis=1, keepdims=1)
    idx = g.i64(list(range(CHANNELS)), dims=[1, CHANNELS, 1, 1])
    colorvec = g.nd("Cast", [g.nd("Equal", [amax, idx])], to=F)         # [1,10,1,1]
    sel = g.nd("Where", [linec, colorvec, onehot0])                    # [1,10,30,30]
    g.nd("Mul", [sel, grid], "output")
    return _model(g)


# ===========================================================================
# 84  left colour bar -> add anti-diagonal of 2s + bottom row of 4s
# ===========================================================================
def _mir_84(a):
    h, w = a.shape
    if h != w:
        return None
    col0 = a[:, 0]
    if (col0 == 0).any():
        return None
    X = int(col0[0])
    if not (col0 == X).all():
        return None
    if (a[:, 1:] != 0).any():
        return None
    o = np.zeros((h, w), int)
    o[:, 0] = X
    for r in range(h - 1):
        o[r, w - 1 - r] = 2
    o[h - 1, 1:w] = 4
    return o


def _build_84(g):
    colidx = g.f([1, 1, 1, W], list(range(W)))
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    negone = g.f([1, 1, 1, 1], [-1.0])
    z1 = g.f([1, 1, H, W], np.zeros((H, W), np.float32))
    z5 = g.f([1, 5, H, W], np.zeros((5, H, W), np.float32))
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)       # full grid
    maxc = g.nd("ReduceMax", [g.nd("Mul",
              [g.nd("ReduceMax", [realmask], axes=[2], keepdims=1), colidx])],
              axes=[3], keepdims=1)                                      # Wm1
    maxr = g.nd("ReduceMax", [g.nd("Mul",
              [g.nd("ReduceMax", [realmask], axes=[3], keepdims=1), rowidx])],
              axes=[2], keepdims=1)                                      # Hm1
    Wn = g.nd("Add", [maxc, one])
    cge1 = g.nd("Cast", [g.nd("Greater", [colidx, half])], to=F)        # c>=1
    s = g.nd("Add", [rowidx, colidx])
    anti = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [s, maxc])]), half])], to=F)
    antidiag = g.nd("Mul", [anti, cge1])                               # [1,1,30,30]
    bottomrow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, maxr])]), half])], to=F)
    clt = g.nd("Cast", [g.nd("Less", [colidx, Wn])], to=F)
    bottom4 = g.nd("Mul", [g.nd("Mul", [bottomrow, cge1]), clt])       # [1,1,30,30]
    feat = g.nd("Add", [antidiag, bottom4])
    negfeat = g.nd("Mul", [feat, negone])
    delta = g.nd("Concat", [negfeat, z1, antidiag, z1, bottom4, z5], axis=1)  # [1,10,30,30]
    g.nd("Add", ["input", delta], "output")
    return _model(g)


# ===========================================================================
# 297  header cycle: rows 0,1 are a header; the body (rows 2..) repeats the
#      header colours read as a column, period W, exactly twice (H-2 == 2W).
# ===========================================================================
def _mir_297(a):
    h, w = a.shape
    if h - 2 != 2 * w or w < 1:
        return None
    if not (a[1] == 5).all():
        return None
    hdr = a[0]
    o = a.copy()
    for r in range(2, h):
        o[r, :] = hdr[(r - 2) % w]
    return o


def _build_297(g):
    colidx = g.f([1, 1, 1, W], list(range(W)))
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    onehalf = g.f([1, 1, 1, 1], [1.5])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    maxc = g.nd("ReduceMax", [g.nd("Mul",
              [g.nd("ReduceMax", [realmask], axes=[2], keepdims=1), colidx])],
              axes=[3], keepdims=1)
    Wn = g.nd("Add", [maxc, one])
    rm2 = g.nd("Sub", [rowidx, two])
    diff = g.nd("Sub", [rm2, colidx])                                  # (r-2)-k  [1,1,30,30]
    d0 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), half])], to=F)
    dW = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [diff, Wn])]), half])], to=F)
    Sel = g.nd("Max", [d0, dW])                                        # [1,1,30,30]
    hdr = _slice(g, "input", [0], [1], [2])                            # [1,10,1,30]
    hdrcol = g.nd("Transpose", [hdr], perm=[0, 1, 3, 2])               # [1,10,30,1]
    bodycol = g.nd("MatMul", [Sel, hdrcol])                            # [1,10,30,1]
    colmask = g.nd("Cast", [g.nd("Less", [colidx, Wn])], to=F)         # [1,1,1,30]
    body = g.nd("Mul", [bodycol, colmask])                            # [1,10,30,30]
    rowge2 = g.nd("Greater", [rowidx, onehalf])                       # bool [1,1,30,1]
    g.nd("Where", [rowge2, body, "input"], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    out = []

    def emit(name, mirror, build):
        try:
            for a, b in prs:
                o = mirror(a)
                if o is None or o.shape != b.shape or not np.array_equal(o, b):
                    return
            g = _G()
            m = build(g)
            onnx.checker.check_model(m, full_check=True)
            out.append((name, m))
        except Exception:
            pass

    # 146 -- unique asymmetric 3x3 block
    if all(a.shape == (9, 3) for a, _ in prs):
        emit("g25_block146", _mir_146, _build_146)

    # 150 -- mirror left-right within width
    if all(a.shape == b.shape for a, b in prs):
        emit("g25_flip150", _mir_150, _build_150)

    # 375 -- diagonals -> background on solid square
    if all(a.shape == b.shape and a.shape[0] == a.shape[1] for a, b in prs):
        emit("g25_xdiag375", _mir_375, _build_375)

    # 55 -- tic-tac-toe cross fill
    if all(a.shape == b.shape for a, b in prs):
        params = None
        try:
            params = _ttt_params(prs)
        except Exception:
            params = None
        if params is not None:
            emit("g25_ttt55", lambda a: _mir_ttt(a, params),
                 lambda g: _build_ttt(g, params))

    # 362 -- shift the X cross by the colour-5 marker count
    if all(a.shape == b.shape for a, b in prs):
        emit("g25_cross362", _mir_362, _build_362)

    # 84 -- left bar -> anti-diagonal 2s + bottom row 4s
    if all(a.shape == b.shape and a.shape[0] == a.shape[1] for a, b in prs):
        emit("g25_diagbar84", _mir_84, _build_84)

    # 297 -- header cycle (body repeats header column, period W, twice)
    if all(a.shape == b.shape for a, b in prs):
        emit("g25_hdrcyc297", _mir_297, _build_297)

    return out
