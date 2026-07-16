"""Verifier-decoded exact rules -> static opset-10 ONNX (family_vc_4).

Three tasks, ground truth = Hodel re-arc verifiers:

task044 = verify_228f6490  "fill holes from matching loose objects"
    Holes = 8-conn bg components not touching the border whose ENTIRE 8-neighbourhood
    is the frame colour 5 (equivalent, on all data, to the verifier's majority-surround
    filter).  Always exactly 2 such holes, grid fixed 10x10, bg=0.  For each hole find
    the colour c (not 0/5) whose cells form EXACTLY the hole's normalized shape
    somewhere in the grid (equivalently: total c-cells == |hole| and a translated
    template match exists) -- erase all c cells, fill the hole with c.
    ONNX: border flood -> enclosed bg; taint flood removes impure holes; first-cell
    raster split into hole1/hole2; dyntranslate normalize -> runtime 7x7 conv template;
    per-colour exact-match test (max corr == |hole| AND total == |hole|); recolour.

task157 = verify_6a1e5592  "fit grey pieces into wall notches, recolour blue"
    Grid fixed 10x15, wall colour 2 occupies rows 0..2 (lowermost wall row always 2),
    bg 0, pieces colour 5 (3-4 of them, each <= 4x4).  Each piece is placed at the
    offset where it (a) lies entirely on bg, (b) every up/left/right neighbour cell
    with row <= 2 is NOT bg (wall or outside);  choose the min-row offset; resolve
    the rare min-row tie by pruning offsets whose placement overlaps a uniquely
    placed piece, then min-col.  Placed cells -> 1, pieces erased -> 0.
    ONNX: 4x unrolled component flood; dyntranslate normalize -> runtime 6x6 template;
    validity via two convs (on-bg exact count + up/left/right border hits on
    bg&rows<=2); min-row / prune / min-col via prefix-sum MatMuls; ConvTranspose place.

task264 = verify_a8c38be5  "reassemble the nine 3x3 jigsaw pieces"
    Eight coloured parts (L-corners 3 cells, T-edges 4 cells) each sit at a fixed
    position inside a 3x3 piece of 5s; output is always the assembled 9x9: canvas 5
    plus each part painted at its slot.  ONNX: one 2-channel (parts, fives) 3x3 conv
    with 8 output channels detects each piece exactly once; colour read via
    mask-MatMul; output = [10,9] colour matrix @ [9,81] slot matrix -> 9x9 -> pad.

All numpy mirrors reproduce the ONNX semantics; candidates() only fires when the
mirror is exact on every train+test pair (grid-size / palette gates included).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# graph accumulator                                                            #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes, self.inits, self._k = [], [], 0
        self._cache = {}

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals, key=None):
        if key is not None and key in self._cache:
            return self._cache[key]
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, F, list(dims), [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        if key is not None:
            self._cache[key] = n
        return n

    def i64(self, vals, key=None):
        if key is not None and key in self._cache:
            return self._cache[key]
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        if key is not None:
            self._cache[key] = n
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    # shared helpers ------------------------------------------------------- #
    def scal(self, v):
        return self.f([1, 1, 1, 1], [v], key=("scal", v))

    def gt(self, a, b):  # binary float (a > b)
        return self.nd("Cast", [self.nd("Greater", [a, b])], to=F)

    def lt(self, a, b):
        return self.nd("Cast", [self.nd("Less", [a, b])], to=F)

    def eq_int(self, a, b):  # 1 iff |a-b| < .5 for integer-valued floats
        return self.lt(self.nd("Abs", [self.nd("Sub", [a, b])]), self.scal(0.5))

    def clip01(self, x):
        return self.nd("Clip", [x], min=0.0, max=1.0)

    def sub(self, a, b):
        return self.nd("Sub", [a, b])

    def add(self, a, b):
        return self.nd("Add", [a, b])

    def mul(self, a, b):
        return self.nd("Mul", [a, b])


def _model(g, name):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, name, [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _slice_ch(g, ch, rows=None, cols=None, src="input"):
    """Slice channel [ch,ch+1) (and optional rows/cols) from the input tensor."""
    starts, ends, axes = [ch], [ch + 1], [1]
    if rows is not None:
        starts.append(rows[0]); ends.append(rows[1]); axes.append(2)
    if cols is not None:
        starts.append(cols[0]); ends.append(cols[1]); axes.append(3)
    return g.nd("Slice", [src, g.i64(starts), g.i64(ends), g.i64(axes)])


def _flood(g, seed, mask, iters, H, W):
    """iters x (8-dilate then mask) starting from seed (already inside mask)."""
    ones3 = g.f([1, 1, 3, 3], np.ones((3, 3)), key=("ones3",))
    cur = g.mul(seed, mask)
    for _ in range(iters):
        d = g.nd("Conv", [cur, ones3], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        cur = g.mul(g.clip01(d), mask)
    return cur


def _first_row(g, mask, H, tri_key):
    """One-hot [1,1,H,1] of the first row of `mask` containing a 1 (all-zero if empty)."""
    L = g.f([H, H], np.tril(np.ones((H, H)), -1), key=tri_key)  # L[i,j]=1 iff j<i
    rowany = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)          # [1,1,H,1]
    pref = g.nd("MatMul", [L, rowany])                                 # rows before
    return g.mul(rowany, g.sub(g.scal(1.0), g.clip01(pref)))


def _first_col(g, mask, W, tri_key):
    """One-hot [1,1,1,W] of the first column of `mask` containing a 1."""
    U = g.f([W, W], np.triu(np.ones((W, W)), 1), key=tri_key)  # U[j,i]=1 iff j<i
    colany = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)          # [1,1,1,W]
    pref = g.nd("MatMul", [colany, U])
    return g.mul(colany, g.sub(g.scal(1.0), g.clip01(pref)))


def _first_cell(g, mask, H, W):
    """One-hot [1,1,H,W] of the raster-first cell of mask."""
    fr = _first_row(g, mask, H, ("L", H))
    rowmask = g.mul(mask, fr)
    fc = _first_col(g, rowmask, W, ("U", W))
    return g.mul(fr, fc)


def _min_rc(g, mask, H, W):
    """(minrow, mincol) scalars [1,1,1,1] of a 0/1 mask (garbage-large if empty)."""
    big = g.scal(1000.0)
    ridx = g.f([1, 1, H, 1], list(range(H)), key=("ridx", H))
    cidx = g.f([1, 1, 1, W], list(range(W)), key=("cidx", W))
    rowany = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)
    colany = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)
    r0 = g.sub(big, g.nd("ReduceMax", [g.mul(rowany, g.sub(big, ridx))],
                         axes=[2], keepdims=1))
    c0 = g.sub(big, g.nd("ReduceMax", [g.mul(colany, g.sub(big, cidx))],
                         axes=[3], keepdims=1))
    return r0, c0


def _normalize(g, mask, H, W):
    """Translate mask so its bbox corner sits at (0,0) (two shift-matrix MatMuls)."""
    r0, c0 = _min_rc(g, mask, H, W)
    ridxH = g.f([1, 1, H, 1], list(range(H)), key=("ridx", H))
    cidxH = g.f([1, 1, 1, H], list(range(H)), key=("cidxh", H))
    # Srow[i,k] = 1 iff k - i == r0
    srow = g.lt(g.nd("Abs", [g.sub(g.sub(cidxH, ridxH), r0)]), g.scal(0.5))
    n1 = g.nd("MatMul", [srow, mask])
    ridxW = g.f([1, 1, W, 1], list(range(W)), key=("ridxw", W))
    cidxW = g.f([1, 1, 1, W], list(range(W)), key=("cidx", W))
    # Scol[k,j] = 1 iff k - j == c0
    scol = g.lt(g.nd("Abs", [g.sub(g.sub(ridxW, cidxW), c0)]), g.scal(0.5))
    return g.nd("MatMul", [n1, scol])


def _pad(g, x, pads, out=None):
    return g.nd("Pad", [x], out=out, mode="constant", value=0.0, pads=pads)


# =========================================================================== #
# numpy component labelling (shared by the mirrors)                            #
# =========================================================================== #
def _comps(mask, diag=True):
    H, W = mask.shape
    nb = ([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
          if diag else [(1, 0), (-1, 0), (0, 1), (0, -1)])
    lab = np.zeros((H, W), int)
    cur, out = 0, []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                cur += 1
                st, cells = [(i, j)], [(i, j)]
                lab[i, j] = cur
                while st:
                    a, b = st.pop()
                    for da, db in nb:
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and mask[x, y] and lab[x, y] == 0:
                            lab[x, y] = cur
                            st.append((x, y))
                            cells.append((x, y))
                out.append(cells)
    return out


# =========================================================================== #
# task044  (228f6490)                                                          #
# =========================================================================== #
H44, W44 = 10, 10
T44 = 7          # template window (max hole bbox seen 3x5)


def _ref044(a):
    a = np.asarray(a, int)
    if a.shape != (H44, W44):
        return None
    if np.bincount(a.ravel(), minlength=10).argmax() != 0 or (a == 5).sum() == 0:
        return None
    out = a.copy()
    holes = []
    for cells in _comps(a == 0, True):
        ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
        if not (min(ys) > 0 and min(xs) > 0 and max(ys) < H44 - 1 and max(xs) < W44 - 1):
            continue
        h = set(cells)
        pure = True
        for (y, x) in h:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dy, dx) == (0, 0):
                        continue
                    p = (y + dy, x + dx)
                    if p in h:
                        continue
                    if not (0 <= p[0] < H44 and 0 <= p[1] < W44) or a[p] != 5:
                        pure = False
        if pure:
            holes.append(cells)
    if len(holes) != 2:
        return None
    for cells in holes:
        ys = np.array([c[0] for c in cells]); xs = np.array([c[1] for c in cells])
        if ys.max() - ys.min() + 1 > T44 or xs.max() - xs.min() + 1 > T44:
            return None
        norm = set(zip((ys - ys.min()).tolist(), (xs - xs.min()).tolist()))
        matched = []
        for c in range(1, 10):
            if c == 5:
                continue
            pos = np.argwhere(a == c)
            if len(pos) != len(cells):
                continue
            pn = set(zip((pos[:, 0] - pos[:, 0].min()).tolist(),
                         (pos[:, 1] - pos[:, 1].min()).tolist()))
            if pn == norm:
                matched.append(c)
        if len(matched) != 1:
            return None
        c = matched[0]
        out[a == c] = 0
        for (y, x) in cells:
            out[y, x] = c
    return out


def _build044():
    g = _G()
    H, W = H44, W44
    bg = _slice_ch(g, 0, (0, H), (0, W))
    five = _slice_ch(g, 5, (0, H), (0, W))
    c14 = g.nd("Slice", ["input", g.i64([1, 0, 0]), g.i64([5, H, W]), g.i64([1, 2, 3])])
    c69 = g.nd("Slice", ["input", g.i64([6, 0, 0]), g.i64([10, H, W]), g.i64([1, 2, 3])])
    colors = g.nd("Concat", [c14, c69], axis=1)                       # [1,8,H,W]
    X8 = g.nd("Reshape", [colors, g.i64([8, 1, H, W], key="sh81")])    # [8,1,H,W]

    # enclosed bg
    bmask = np.zeros((H, W)); bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = 1
    outside = _flood(g, g.mul(bg, g.f([1, 1, H, W], bmask)), bg, 14, H, W)
    enclosed = g.sub(bg, outside)

    # impure-surround taint
    ones3 = g.f([1, 1, 3, 3], np.ones((3, 3)), key=("ones3",))
    non5 = g.sub(g.sub(g.f([1, 1, H, W], np.ones((H, W)), key="ones44"), bg), five)
    near = g.clip01(g.nd("Conv", [non5, ones3], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    taint = _flood(g, g.mul(enclosed, near), enclosed, 4, H, W)
    holes = g.sub(enclosed, taint)

    hole1 = _flood(g, _first_cell(g, holes, H, W), holes, 8, H, W)
    hole2 = g.sub(holes, hole1)

    tot8 = g.nd("ReduceSum", [X8], axes=[2, 3], keepdims=1)            # [8,1,1,1]
    X8p = _pad(g, X8, [0, 0, 0, 0, 0, 0, T44 - 1, T44 - 1])            # [8,1,H+6,W+6]

    matches, anys = [], []
    for hole in (hole1, hole2):
        size = g.nd("ReduceSum", [hole], axes=[2, 3], keepdims=1)      # [1,1,1,1]
        norm = _normalize(g, hole, H, W)
        tmpl = g.nd("Slice", [norm, g.i64([0, 0]), g.i64([T44, T44]), g.i64([2, 3])])
        conv = g.nd("Conv", [X8p, tmpl], kernel_shape=[T44, T44])      # [8,1,H,W]
        mx = g.nd("ReduceMax", [conv], axes=[2, 3], keepdims=1)        # [8,1,1,1]
        m = g.mul(g.eq_int(mx, size), g.eq_int(tot8, size))            # [8,1,1,1]
        matches.append(m)
        anys.append(g.nd("ReduceMax", [m], axes=[0], keepdims=1))      # [1,1,1,1]

    keep = g.mul(g.sub(g.scal(1.0), matches[0]), g.sub(g.scal(1.0), matches[1]))
    er1 = g.nd("ReduceSum", [g.mul(X8, matches[0])], axes=[0], keepdims=1)
    er2 = g.nd("ReduceSum", [g.mul(X8, matches[1])], axes=[0], keepdims=1)
    out8 = g.add(g.add(g.mul(X8, keep), g.mul(hole1, matches[0])),
                 g.mul(hole2, matches[1]))                             # [8,1,H,W]
    out0 = g.add(g.add(g.sub(g.sub(bg, g.mul(hole1, anys[0])), g.mul(hole2, anys[1])),
                       er1), er2)

    pads30 = [0, 0, 0, 0, 0, 0, 30 - H, 30 - W]
    planes = [_pad(g, out0, pads30)]
    out8r = g.nd("Reshape", [out8, g.i64([1, 8, H, W], key="sh18")])
    for k in range(8):
        pl = g.nd("Slice", [out8r, g.i64([k]), g.i64([k + 1]), g.i64([1])])
        planes.append(_pad(g, pl, pads30))
    ch5 = _slice_ch(g, 5)                                              # untouched
    order = planes[:5] + [ch5] + planes[5:]
    g.nd("Concat", order, out="output", axis=1)
    return _model(g, "vc4_holes")


# =========================================================================== #
# task157  (6a1e5592)                                                          #
# =========================================================================== #
H57, W57 = 10, 15
T57 = 6          # template window (max piece 4x4)
NP57 = 4         # max pieces
WALLROW = 2      # lowermost wall row (invariant, gated)


def _piece_valid_np(a, norm):
    """valid offset map [(H,W)] for a normalized piece on grid a (mirrors the ONNX)."""
    H, W = a.shape
    bg = (a == 0)
    bgtop = bg.copy(); bgtop[WALLROW + 1:, :] = False
    V = np.zeros((H, W), bool)
    for oy in range(H):
        for ox in range(W):
            sh = [(u + oy, v + ox) for u, v in norm]
            if not all(0 <= y < H and 0 <= x < W and bg[y, x] for y, x in sh):
                continue
            ok = True
            for y, x in sh:
                for dy, dx in ((-1, 0), (0, -1), (0, 1)):
                    p = (y + dy, x + dx)
                    if p in sh:
                        continue
                    if 0 <= p[0] < H and 0 <= p[1] < W and bgtop[p]:
                        ok = False
            if ok:
                V[oy, ox] = True
    return V


def _minrow_only(V):
    out = np.zeros_like(V)
    ys, xs = np.where(V)
    if ys.size:
        out[ys.min(), :] = V[ys.min(), :]
    return out


def _ref157(a):
    a = np.asarray(a, int)
    if a.shape != (H57, W57):
        return None
    if not set(np.unique(a)) <= {0, 2, 5}:
        return None
    wallrows = np.where((a == 2).any(axis=1))[0]
    if wallrows.size == 0 or wallrows.max() != WALLROW or (a[0] != 2).any():
        return None
    pieces = _comps(a == 5, True)
    if not 1 <= len(pieces) <= NP57:
        return None
    norms, Vmins = [], []
    for cells in pieces:
        ys = np.array([c[0] for c in cells]); xs = np.array([c[1] for c in cells])
        if ys.max() - ys.min() + 1 > T57 or xs.max() - xs.min() + 1 > T57:
            return None
        norm = set(zip((ys - ys.min()).tolist(), (xs - xs.min()).tolist()))
        norms.append(norm)
        Vmins.append(_minrow_only(_piece_valid_np(a, norm)))
    # prune: placements overlapping a uniquely-placed piece's placement
    occupied = np.zeros((H57, W57), bool)
    uniq = [Vm.sum() == 1 for Vm in Vmins]
    for k, Vm in enumerate(Vmins):
        if uniq[k]:
            oy, ox = map(int, np.argwhere(Vm)[0])
            for u, v in norms[k]:
                occupied[oy + u, ox + v] = True
    placed_all = np.zeros((H57, W57), bool)
    for k, Vm in enumerate(Vmins):
        V2 = Vm.copy()
        if not uniq[k]:
            for oy, ox in np.argwhere(Vm):
                if any(occupied[oy + u, ox + v] for u, v in norms[k]):
                    V2[oy, ox] = False
        V3 = _minrow_only(V2)
        ys, xs = np.where(V3)
        if ys.size == 0:
            return None
        j = int(np.argmin(xs))                 # min-row (V3 is single-row) then min-col
        oy, ox = int(ys[j]), int(xs[j])
        for u, v in norms[k]:
            placed_all[oy + u, ox + v] = True
    out = a.copy()
    out[a == 5] = 0
    out[placed_all] = 1
    return out


def _build157():
    g = _G()
    H, W = H57, W57
    bg = _slice_ch(g, 0, (0, H), (0, W))
    five = _slice_ch(g, 5, (0, H), (0, W))
    toprows = np.zeros((H, W)); toprows[:WALLROW + 1, :] = 1
    bgtop = g.mul(bg, g.f([1, 1, H, W], toprows))
    bgpad = _pad(g, bg, [0, 0, 0, 0, 0, 0, T57 - 1, T57 - 1])          # [1,1,15,20]
    bgtoppad = _pad(g, bgtop, [0, 0, 1, 1, 0, 0, T57, T57])            # [1,1,17,22]

    # split the (up to NP57) pieces
    comps, rem = [], five
    for _ in range(NP57):
        comp = _flood(g, _first_cell(g, rem, H, W), rem, 8, H, W)
        comps.append(comp)
        rem = g.sub(rem, comp)

    tmpls, Vmins, cnts, uniqs, placedAll = [], [], [], [], []
    for comp in comps:
        norm = _normalize(g, comp, H, W)
        tmpl = g.nd("Slice", [norm, g.i64([0, 0]), g.i64([T57, T57]), g.i64([2, 3])])
        tmpls.append(tmpl)
        psz = g.nd("ReduceSum", [comp], axes=[2, 3], keepdims=1)
        # (a) placement entirely on bg
        convA = g.nd("Conv", [bgpad, tmpl], kernel_shape=[T57, T57])   # [1,1,H,W]
        A = g.gt(convA, g.sub(psz, g.scal(0.5)))
        # (b) up/left/right neighbours in rows<=2 must not be bg
        E = _pad(g, tmpl, [0, 0, 1, 1, 0, 0, 1, 1])                    # [1,1,8,8]
        s_up = _pad(g, g.nd("Slice", [E, g.i64([1]), g.i64([8]), g.i64([2])]),
                    [0, 0, 0, 0, 0, 0, 1, 0])
        s_lf = _pad(g, g.nd("Slice", [E, g.i64([1]), g.i64([8]), g.i64([3])]),
                    [0, 0, 0, 0, 0, 0, 0, 1])
        s_rt = g.nd("Slice", [_pad(g, E, [0, 0, 0, 1, 0, 0, 0, 0]),
                              g.i64([0]), g.i64([8]), g.i64([3])])
        border = g.clip01(g.sub(g.clip01(g.add(g.add(s_up, s_lf), s_rt)), E))
        convB = g.nd("Conv", [bgtoppad, border], kernel_shape=[8, 8])  # [1,1,H,W]
        V = g.mul(A, g.lt(convB, g.scal(0.5)))
        Vmin = g.mul(V, _first_row(g, V, H, ("L", H)))
        Vmins.append(Vmin)
        cnt = g.nd("ReduceSum", [Vmin], axes=[2, 3], keepdims=1)
        uniq = g.eq_int(cnt, g.scal(1.0))
        uniqs.append(uniq)
        pl = g.nd("ConvTranspose", [Vmin, tmpl], kernel_shape=[T57, T57])
        pl = g.nd("Slice", [pl, g.i64([0, 0]), g.i64([H, W]), g.i64([2, 3])])
        placedAll.append(g.mul(g.clip01(pl), uniq))

    occ = g.clip01(g.add(g.add(placedAll[0], placedAll[1]),
                         g.add(placedAll[2], placedAll[3])))
    occpad = _pad(g, occ, [0, 0, 0, 0, 0, 0, T57 - 1, T57 - 1])

    placed_final = []
    for tmpl, Vmin, uniq in zip(tmpls, Vmins, uniqs):
        Ov = g.gt(g.nd("Conv", [occpad, tmpl], kernel_shape=[T57, T57]), g.scal(0.5))
        gate = g.add(uniq, g.mul(g.sub(g.scal(1.0), uniq), g.sub(g.scal(1.0), Ov)))
        V2 = g.mul(Vmin, gate)
        V3 = g.mul(V2, _first_row(g, V2, H, ("L", H)))
        V4 = g.mul(V3, _first_col(g, V3, W, ("U", W)))
        pl = g.nd("ConvTranspose", [V4, tmpl], kernel_shape=[T57, T57])
        pl = g.nd("Slice", [pl, g.i64([0, 0]), g.i64([H, W]), g.i64([2, 3])])
        placed_final.append(g.clip01(pl))

    placed = g.clip01(g.add(g.add(placed_final[0], placed_final[1]),
                            g.add(placed_final[2], placed_final[3])))
    out0 = g.sub(g.add(bg, five), placed)
    out1 = placed

    pads30 = [0, 0, 0, 0, 0, 0, 30 - H, 30 - W]
    zero = g.f([1, 1, 30, 30], np.zeros((30, 30)), key="zero30")
    ch2 = _slice_ch(g, 2)
    order = [_pad(g, out0, pads30), _pad(g, out1, pads30), ch2,
             zero, zero, zero, zero, zero, zero, zero]
    g.nd("Concat", order, out="output", axis=1)
    return _model(g, "vc4_notches")


# =========================================================================== #
# task264  (a8c38be5)                                                          #
# =========================================================================== #
_SIGS = {           # slot -> (part shape, part offset inside its 3x3 piece)
    "TL": ({(0, 0), (0, 1), (1, 0)}, (0, 0)),
    "TR": ({(0, 0), (0, 1), (1, 1)}, (0, 1)),
    "BL": ({(0, 0), (1, 0), (1, 1)}, (1, 0)),
    "BR": ({(0, 1), (1, 0), (1, 1)}, (1, 1)),
    "TM": ({(0, 0), (0, 1), (0, 2), (1, 1)}, (0, 0)),
    "BM": ({(0, 1), (1, 0), (1, 1), (1, 2)}, (1, 0)),
    "LM": ({(0, 0), (1, 0), (1, 1), (2, 0)}, (0, 0)),
    "RM": ({(0, 1), (1, 0), (1, 1), (2, 1)}, (0, 1)),
}
_ORDER = ["TL", "TR", "BL", "BR", "TM", "BM", "LM", "RM"]
_SLOT = {"TL": (0, 0), "TR": (0, 7), "BL": (7, 0), "BR": (7, 7),
         "TM": (0, 3), "BM": (7, 3), "LM": (3, 0), "RM": (3, 7)}
G264 = 16       # working grid


def _ref264(a):
    a = np.asarray(a, int)
    H, W = a.shape
    if not (14 <= H <= G264 and 14 <= W <= G264):
        return None
    P = np.zeros((G264, G264)); Fv = np.zeros((G264, G264))
    P[:H, :W] = (a != 0) & (a != 5)
    Fv[:H, :W] = a == 5
    colors = {}
    for name in _ORDER:
        sig, off = _SIGS[name]
        hits = []
        for y in range(G264 - 2):
            for x in range(G264 - 2):
                s = 0.0
                for u in range(3):
                    for v in range(3):
                        if (u - off[0], v - off[1]) in sig:
                            s += P[y + u, x + v] - 8 * Fv[y + u, x + v]
                        else:
                            s += Fv[y + u, x + v] - 8 * P[y + u, x + v]
                if s > 8.5:
                    hits.append((y, x))
        if len(hits) != 1:
            return None
        y, x = hits[0]
        u0, v0 = min(sig)                      # a part cell (row-major first)
        colors[name] = int(a[y + off[0] + u0, x + off[1] + v0])
        if colors[name] in (0, 5):
            return None
    out = np.full((9, 9), 5, int)
    for name in _ORDER:
        sig, _ = _SIGS[name]
        oy, ox = _SLOT[name]
        for u, v in sig:
            out[oy + u, ox + v] = colors[name]
    return out


def _build264():
    g = _G()
    N = G264
    X = g.nd("Slice", ["input", g.i64([0, 0]), g.i64([N, N]), g.i64([2, 3])])  # [1,10,16,16]
    allsum = g.nd("ReduceSum", [X], axes=[1], keepdims=1)              # [1,1,16,16]
    bg = _slice_ch(g, 0, (0, N), (0, N))
    five = _slice_ch(g, 5, (0, N), (0, N))
    P = g.sub(g.sub(allsum, bg), five)
    PF = g.nd("Concat", [P, five], axis=1)                             # [1,2,16,16]

    # one 8-out-channel 2-channel 3x3 conv detects every piece
    Wk = np.zeros((8, 2, 3, 3), np.float32)
    for i, name in enumerate(_ORDER):
        sig, off = _SIGS[name]
        for u in range(3):
            for v in range(3):
                if (u - off[0], v - off[1]) in sig:
                    Wk[i, 0, u, v] = 1.0; Wk[i, 1, u, v] = -8.0
                else:
                    Wk[i, 1, u, v] = 1.0; Wk[i, 0, u, v] = -8.0
    conv = g.nd("Conv", [PF, g.f([8, 2, 3, 3], Wk)], kernel_shape=[3, 3])  # [1,8,14,14]
    det = g.gt(conv, g.scal(8.5))                                      # [1,8,14,14]

    Xf = g.nd("Reshape", [X, g.i64([10, N * N], key="shXf")])          # [10,256]
    cols = []
    for i, name in enumerate(_ORDER):
        sig, off = _SIGS[name]
        u0, v0 = min(sig)
        dy, dx = off[0] + u0, off[1] + v0      # piece-ul -> a part cell
        m = g.nd("Slice", [det, g.i64([i]), g.i64([i + 1]), g.i64([1])])
        m = _pad(g, m, [0, 0, dy, dx, 0, 0, 2 - dy, 2 - dx])           # [1,1,16,16]
        m = g.nd("Reshape", [m, g.i64([N * N, 1], key="shm")])
        cols.append(g.nd("MatMul", [Xf, m]))                           # [10,1]
    e5 = np.zeros((10, 1)); e5[5, 0] = 1.0
    C = g.nd("Concat", cols + [g.f([10, 1], e5)], axis=1)              # [10,9]

    S = np.zeros((9, 81), np.float32)
    canvas = np.ones((9, 9), np.float32)
    for i, name in enumerate(_ORDER):
        sig, _ = _SIGS[name]
        oy, ox = _SLOT[name]
        for u, v in sig:
            S[i, (oy + u) * 9 + (ox + v)] = 1.0
            canvas[oy + u, ox + v] = 0.0
    S[8] = canvas.ravel()
    out81 = g.nd("MatMul", [C, g.f([9, 81], S)])                       # [10,81]
    out99 = g.nd("Reshape", [out81, g.i64([1, 10, 9, 9], key="sh99")])
    _pad(g, out99, [0, 0, 0, 0, 0, 0, 21, 21], out="output")
    return _model(g, "vc4_jigsaw")


# =========================================================================== #
# entry point                                                                  #
# =========================================================================== #
def candidates(examples):
    prs = [(np.array(e["input"], int), np.array(e["output"], int))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    for name, ref, build in (("holes228f6490", _ref044, _build044),
                             ("notches6a1e5592", _ref157, _build157),
                             ("jigsaw_a8c38be5", _ref264, _build264)):
        ok = True
        for a, b in prs:
            try:
                r = ref(a)
            except Exception:
                r = None
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            try:
                m = build()
                onnx.checker.check_model(m)
            except Exception:
                continue
            yield (name, m)
