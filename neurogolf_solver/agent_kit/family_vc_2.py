"""Verifier-decoded family for task018 (0e206a2e), task118 (50846271), task219 (90f3ed37).

task118 = verify_50846271 (colors fixed 0/2/5 -> fill 8):
  1) bridge: 4 iterations; a cell q becomes 2 if a 2 is adjacent on one side and another 2 is
     at distance 2..4 on the opposite side (all 4 directions)  [one 4-channel 7x7 conv/iter].
  2) x53 = max over diagonal-connected 2-components of max(bbox h,w). On this task's data
     x53 == max straight run of 2s (validated all 267) -> computed with 8 run-detector convs
     (L=4..7, h/v), monotone flags, x53 = 3 + sum(flags).
  3) cross centers = 2-cells with both a horizontal and a vertical 2-neighbour, PLUS centers of
     isolated straight lines of length exactly x53 (ring-isolation convs gated by [x53==L]).
  4) draw plus-crosses of arm x53//2 at all centers (two gated cross convs), clip to real grid;
     every new cell -> 8, original 2s stay 2.

task219 = verify_90f3ed37 (colors fixed 0/8 -> fill 1):
  8-cells grouped into row-bands (separated by 8-free rows). Widest band = template. Every other
  band gets the template overlaid at the max-overlap offset (candidate offsets = grid-size window
  centered on the band's center; prefer placements whose added cells all lie right of the band's
  rightmost cell; then largest dj, then largest di - reproduces the verifier's frozenset argmax on
  all 265 examples). Added cells -> colour 1 (overwrites, clipped to real grid).
  ONNX: 15 unrolled band slots; per band a Conv correlation with the runtime-normalized template
  as weight ([1,1,59,59] offset scores), scalar-scored argmax, dyntranslate MatMul placement.

task018 = verify_0e206a2e (bg 0, colours vary):
  4-colour 4-connected objects are templates; they are erased; each occurrence of a template's
  3-cell rare-colour marker pattern (searched in 8 orientations SEQUENTIALLY: id, dmirror,
  cmirror, hmirror, vmirror, rot270, rot180, rot90 - later stages see earlier stamps) gets the
  full template stamped (aligned at the rare part's ulcorner).
  ONNX: seeds = nonbg cells with >=2 nonbg 4-neighbours and >=4 distinct colours in their 13x13
  window (provably hits every template component; <=3 seeded components on all 266 examples);
  6 unrolled scan-order flood-extraction slots (14 masked dilations), numcolors==4 gate; runtime
  Conv kernels [S,9,7,7] (marker match) and ConvTranspose kernels [S,9,13,13] (stamp); 8 stages
  with origin-anchored runtime mirror/rotate MatMuls between them.
"""
import json
import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64
CBIG = 1000.0


# =============================================================================
# graph builder helper
# =============================================================================
class _G:
    def __init__(self):
        self.nodes, self.inits = [], []
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

    def i64(self, vals):
        key = ("i64", tuple(int(v) for v in vals))
        if key in self._cache:
            return self._cache[key]
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, I64, [len(vals)], [int(v) for v in vals]))
        self._cache[key] = n
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    def scal(self, v):
        key = ("s", float(v))
        if key not in self._cache:
            self._cache[key] = self.f([1, 1, 1, 1], [float(v)])
        return self._cache[key]

    def rowidx(self):
        if "rowidx" not in self._cache:
            self._cache["rowidx"] = self.f([1, 1, 30, 1], list(range(30)))
        return self._cache["rowidx"]

    def colidx(self):
        if "colidx" not in self._cache:
            self._cache["colidx"] = self.f([1, 1, 1, 30], list(range(30)))
        return self._cache["colidx"]

    # --- common patterns ------------------------------------------------- #
    def ch(self, src, c0, c1):
        """channel slice [c0:c1] of a [1,C,30,30] tensor"""
        return self.nd("Slice", [src, self.i64([c0]), self.i64([c1]), self.i64([1])])

    def gt(self, a, thr):
        t = thr if isinstance(thr, str) else self.scal(thr)
        return self.nd("Cast", [self.nd("Greater", [a, t])], to=F)

    def lt(self, a, thr):
        t = thr if isinstance(thr, str) else self.scal(thr)
        return self.nd("Cast", [self.nd("Less", [a, t])], to=F)

    def mul(self, a, b):
        return self.nd("Mul", [a, b])

    def add(self, a, b):
        return self.nd("Add", [a, b])

    def sub(self, a, b):
        return self.nd("Sub", [a, b])

    def inv(self, a):
        """1 - a"""
        return self.nd("Sub", [self.scal(1.0), a])

    def realmask(self, tensor10):
        s = self.nd("ReduceSum", [tensor10], axes=[1], keepdims=1)
        return self.gt(s, 0.5)

    def static_shift(self, src, dr, dc, out=None):
        """dst[i,j] = src[i-dr, j-dc], zero fill (static Pad+Slice)."""
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        p = self.nd("Pad", [src], mode="constant", value=0.0,
                    pads=[0, 0, pt, pl, 0, 0, pb, pr])
        return self.nd("Slice", [p, self.i64([pb, pr]), self.i64([pb + 30, pr + 30]),
                                 self.i64([2, 3])], out)

    def minrow(self, plane):
        """min row index of nonzero cells of a 0/1 [1,1,30,30] plane (CBIG if empty)."""
        rowhas = self.nd("ReduceMax", [plane], axes=[3], keepdims=1)
        m = self.nd("ReduceMax", [self.mul(rowhas, self.sub(self.scal(CBIG), self.rowidx()))],
                    axes=[2], keepdims=1)
        return self.sub(self.scal(CBIG), m)

    def mincol(self, plane):
        colhas = self.nd("ReduceMax", [plane], axes=[2], keepdims=1)
        m = self.nd("ReduceMax", [self.mul(colhas, self.sub(self.scal(CBIG), self.colidx()))],
                    axes=[3], keepdims=1)
        return self.sub(self.scal(CBIG), m)

    def maxrow(self, plane):
        rowhas = self.nd("ReduceMax", [plane], axes=[3], keepdims=1)
        return self.nd("ReduceMax", [self.mul(rowhas, self.rowidx())], axes=[2], keepdims=1)

    def maxcol(self, plane):
        colhas = self.nd("ReduceMax", [plane], axes=[2], keepdims=1)
        return self.nd("ReduceMax", [self.mul(colhas, self.colidx())], axes=[3], keepdims=1)

    def shift_rows(self, src, dy):
        """out[i,:] = src[i-dy,:] for runtime scalar dy (MatMul with computed matrix)."""
        diff = self.sub(self.sub(self.rowidx(), self.colidx()), dy)   # [1,1,30,30]
        srow = self.lt(self.nd("Abs", [diff]), 0.5)
        return self.nd("MatMul", [srow, src])

    def shift_cols(self, src, dx):
        diff = self.sub(self.sub(self.colidx(), self.rowidx()), dx)
        scol = self.lt(self.nd("Abs", [diff]), 0.5)
        return self.nd("MatMul", [src, scol])

    def shift2(self, src, dy, dx):
        return self.shift_cols(self.shift_rows(src, dy), dx)

    def model(self, name):
        x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
        y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
        graph = oh.make_graph(self.nodes, name, [x], [y], self.inits)
        return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# =============================================================================
# task118  (verify_50846271)
# =============================================================================
def _bridge118_np(P):
    h, w = P.shape
    P = P.copy()
    for _ in range(4):
        add = np.zeros_like(P)
        for k in (2, 3, 4):
            A = np.zeros_like(P); A[:, :-1] = P[:, 1:]
            B = np.zeros_like(P); B[:, k - 1:] = P[:, :w - (k - 1)]
            add |= A & B
            A = np.zeros_like(P); A[:, 1:] = P[:, :-1]
            B = np.zeros_like(P); B[:, :w - (k - 1)] = P[:, k - 1:]
            add |= A & B
            A = np.zeros_like(P); A[:-1, :] = P[1:, :]
            B = np.zeros_like(P); B[k - 1:, :] = P[:h - (k - 1), :]
            add |= A & B
            A = np.zeros_like(P); A[1:, :] = P[:-1, :]
            B = np.zeros_like(P); B[:h - (k - 1), :] = P[k - 1:, :]
            add |= A & B
        P |= add
    return P


def _maxrun_np(P):
    best = 0
    for M in (P, P.T):
        for row in M:
            run = 0
            for v in row:
                run = run + 1 if v else 0
                best = max(best, run)
    return best


def _bbox_x53_np(P):
    h, w = P.shape
    lab = np.full((h, w), -1, int)
    x = 0
    for i in range(h):
        for j in range(w):
            if P[i, j] and lab[i, j] < 0:
                st = [(i, j)]; lab[i, j] = 0; cs = []
                while st:
                    a, b = st.pop(); cs.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            na, nb = a + da, b + db
                            if 0 <= na < h and 0 <= nb < w and P[na, nb] and lab[na, nb] < 0:
                                lab[na, nb] = 0; st.append((na, nb))
                rs = [c[0] for c in cs]; csj = [c[1] for c in cs]
                x = max(x, max(rs) - min(rs) + 1, max(csj) - min(csj) + 1)
    return x


def _solve118(I):
    """numpy mirror of the ONNX graph; None if outside the graph's regime."""
    I = np.asarray(I)
    if I.ndim != 2 or max(I.shape) > 30:
        return None
    if not set(np.unique(I)) <= {0, 2, 5}:
        return None
    if not (I == 2).any():
        return None
    h, w = I.shape
    orig = (I == 2)
    P = _bridge118_np(orig)
    x53 = _maxrun_np(P)
    if not (4 <= x53 <= 7):
        return None
    if x53 != _bbox_x53_np(P):          # ONNX uses maxrun; must equal verifier's bbox value
        return None
    L = np.zeros_like(P); L[:, 1:] = P[:, :-1]
    R = np.zeros_like(P); R[:, :-1] = P[:, 1:]
    U = np.zeros_like(P); U[1:, :] = P[:-1, :]
    D = np.zeros_like(P); D[:-1, :] = P[1:, :]
    centers = P & (L | R) & (U | D)
    centers = centers.copy()
    # isolated straight lines of length exactly x53 (ring-isolation, mirrors the conv)
    Pp = np.zeros((h + 2, w + 2), bool); Pp[1:-1, 1:-1] = P
    for i in range(h):
        for j in range(w):
            # horizontal, anchor = left end
            if j + x53 <= w:
                win = Pp[i:i + 3, j:j + x53 + 2]
                if win[1, 1:x53 + 1].all() and win.sum() == x53:
                    centers[i, j + x53 // 2] = True
            if i + x53 <= h:
                win = Pp[i:i + x53 + 2, j:j + 3]
                if win[1:x53 + 1, 1].all() and win.sum() == x53:
                    centers[i + x53 // 2, j] = True
    r = x53 // 2
    cross = np.zeros_like(P)
    for (ci, cj) in zip(*np.nonzero(centers)):
        for d in range(-r, r + 1):
            if 0 <= ci + d < h:
                cross[ci + d, cj] = True
            if 0 <= cj + d < w:
                cross[ci, cj + d] = True
    total = P | cross
    out = I.copy()
    out[total] = 8
    out[orig] = 2
    return out


def _build118():
    g = _G()
    P0 = g.ch("input", 2, 3)
    B0 = g.ch("input", 0, 1)
    N5 = g.ch("input", 5, 6)
    Rm = g.realmask("input")

    # bridge kernel [4,1,7,7]
    Kb = np.zeros((4, 1, 7, 7), np.float32)
    Kb[0, 0, 3, 4] = 4; Kb[0, 0, 3, 0:3] = 1          # right nb + left 2..4
    Kb[1, 0, 3, 2] = 4; Kb[1, 0, 3, 4:7] = 1
    Kb[2, 0, 4, 3] = 4; Kb[2, 0, 0:3, 3] = 1
    Kb[3, 0, 2, 3] = 4; Kb[3, 0, 4:7, 3] = 1
    kb = g.f([4, 1, 7, 7], Kb)
    P = P0
    for _ in range(4):
        cv = g.nd("Conv", [P, kb], kernel_shape=[7, 7], pads=[3, 3, 3, 3])
        addm = g.nd("ReduceMax", [g.gt(cv, 4.5)], axes=[1], keepdims=1)
        P = g.nd("Max", [P, addm])

    # run-length flags (L=4..7, horizontal ch0-3, vertical ch4-7)
    Kr = np.zeros((8, 1, 7, 7), np.float32)
    for i, L in enumerate((4, 5, 6, 7)):
        Kr[i, 0, 3, 0:L] = 1
        Kr[4 + i, 0, 0:L, 3] = 1
    kr = g.f([8, 1, 7, 7], Kr)
    thr = g.f([1, 8, 1, 1], [3.5, 4.5, 5.5, 6.5, 3.5, 4.5, 5.5, 6.5])
    rsum = g.nd("Conv", [P, kr], kernel_shape=[7, 7], pads=[3, 3, 3, 3])
    hit = g.nd("Cast", [g.nd("Greater", [rsum, thr])], to=F)
    fl = g.nd("ReduceMax", [hit], axes=[2, 3], keepdims=1)          # [1,8,1,1]
    fh = g.ch(fl, 0, 4)
    fv = g.ch(fl, 4, 8)
    fL = g.nd("Max", [fh, fv])                                       # [1,4,1,1] f4..f7
    f = [g.ch(fL, i, i + 1) for i in range(4)]                       # scalars
    eq = [g.sub(f[0], f[1]), g.sub(f[1], f[2]), g.sub(f[2], f[3]), f[3]]

    # x72 centers
    kh = g.f([1, 1, 1, 3], [1, 0, 1])
    kv = g.f([1, 1, 3, 1], [1, 0, 1])
    nh = g.nd("Conv", [P, kh], kernel_shape=[1, 3], pads=[0, 1, 0, 1])
    nv = g.nd("Conv", [P, kv], kernel_shape=[3, 1], pads=[1, 0, 1, 0])
    centers = g.mul(g.mul(P, g.gt(nh, 0.5)), g.gt(nv, 0.5))

    # isolated exact-length line centers, gated per L
    for i, L in enumerate((4, 5, 6, 7)):
        Kh = np.full((3, L + 2), -10.0, np.float32); Kh[1, 1:L + 1] = 1.0
        det = g.nd("Conv", [P, g.f([1, 1, 3, L + 2], Kh)],
                   kernel_shape=[3, L + 2], pads=[1, 1, 1, L])
        m = g.gt(det, L - 0.5)
        ms = g.static_shift(m, 0, L // 2)
        centers = g.add(centers, g.mul(ms, eq[i]))
        Kv = np.full((L + 2, 3), -10.0, np.float32); Kv[1:L + 1, 1] = 1.0
        det = g.nd("Conv", [P, g.f([1, 1, L + 2, 3], Kv)],
                   kernel_shape=[L + 2, 3], pads=[1, 1, L, 1])
        m = g.gt(det, L - 0.5)
        ms = g.static_shift(m, L // 2, 0)
        centers = g.add(centers, g.mul(ms, eq[i]))

    # crosses: arm r = 2 (x53 in 4,5) or 3 (x53 in 6,7); r3 flag = f6
    def plus(r):
        K = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        K[r, :] = 1; K[:, r] = 1
        return K
    c5 = g.nd("Conv", [centers, g.f([1, 1, 5, 5], plus(2))], kernel_shape=[5, 5],
              pads=[2, 2, 2, 2])
    c7 = g.nd("Conv", [centers, g.f([1, 1, 7, 7], plus(3))], kernel_shape=[7, 7],
              pads=[3, 3, 3, 3])
    r3 = f[2]                                                        # f6
    crossraw = g.add(g.mul(c5, g.inv(r3)), g.mul(c7, r3))
    cross = g.mul(g.gt(crossraw, 0.5), Rm)

    total = g.nd("Max", [P, cross])
    t8 = g.mul(total, g.inv(P0))
    ch0o = g.mul(B0, g.inv(t8))
    ch5o = g.mul(N5, g.inv(t8))
    Z = g.sub(P0, P0)
    g.nd("Concat", [ch0o, Z, P0, Z, Z, ch5o, Z, Z, t8, Z], "output", axis=1)
    return g.model("vc2_118")


# =============================================================================
# task219  (verify_90f3ed37)
# =============================================================================
def _solve219(I):
    """numpy mirror of the ONNX graph (colors 0/8, fill 1). None outside regime."""
    I = np.asarray(I)
    if I.ndim != 2 or max(I.shape) > 30:
        return None
    if not set(np.unique(I)) <= {0, 8} or not (I == 8).any():
        return None
    h, w = I.shape
    M = (I == 8)
    rows = np.nonzero(M.any(axis=1))[0]
    bands = []
    cur = [rows[0]]
    for rr in rows[1:]:
        if rr == cur[-1] + 1:
            cur.append(rr)
        else:
            bands.append(cur); cur = [rr]
    bands.append(cur)
    if len(bands) > 15:
        return None
    objs = []
    for b in bands:
        cells = {(int(rr), int(cc)) for rr in b for cc in np.nonzero(M[rr])[0]}
        objs.append(cells)
    widths = [max(c[1] for c in o) - min(c[1] for c in o) + 1 for o in objs]
    # template: widest, topmost on tie (same scoring as the graph)
    ti = max(range(len(objs)), key=lambda k: (widths[k], -k))
    tmpl = objs[ti]
    ur = min(c[0] for c in tmpl); uc = min(c[1] for c in tmpl)
    T = {(a - ur, b - uc) for (a, b) in tmpl}
    adds = set()
    for i, o in enumerate(objs):
        if i == ti:
            continue
        rs = [c[0] for c in o]; cs = [c[1] for c in o]
        oc = (min(rs) + (max(rs) - min(rs) + 1) // 2, min(cs) + (max(cs) - min(cs) + 1) // 2)
        rm = max(cs)
        best = None
        for a in range(h):
            for b in range(w):
                di, dj = a - h // 2 + oc[0], b - w // 2 + oc[1]
                placed = {(x + di, y + dj) for (x, y) in T}
                ov = len(placed & o)
                if ov == 0:
                    continue                         # also drops offsets outside the 59x59 grid
                if not (-29 <= di <= 29 and -29 <= dj <= 29):
                    return None                      # positive overlap requires |d| <= 29
                nlow = sum(1 for (x, y) in T if y + dj <= rm)
                pref = 1 if ov >= nlow else 0
                key = (ov, pref, dj, di)
                if best is None or key > best[0]:
                    best = (key, placed)
        if best is None:
            return None                              # verifier itself would crash here
        placed = {(a, b) for (a, b) in best[1] if 0 <= a < 30 and 0 <= b < 30}  # canvas clip
        adds |= (placed - o)
    out = I.copy()
    for (a, b) in adds:
        if 0 <= a < h and 0 <= b < w:
            out[a, b] = 1
    return out


def _build219():
    g = _G()
    M = g.ch("input", 8, 9)
    B0 = g.ch("input", 0, 1)
    Rm = g.realmask("input")

    hh = g.add(g.maxrow(Rm), g.scal(1.0))
    ww = g.add(g.maxcol(Rm), g.scal(1.0))
    h2 = g.nd("Floor", [g.mul(hh, g.scal(0.5))])
    w2 = g.nd("Floor", [g.mul(ww, g.scal(0.5))])

    a = g.nd("ReduceMax", [M], axes=[3], keepdims=1)                 # [1,1,30,1]
    ash = g.static_shift(a, 1, 0)
    bandstart = g.mul(a, g.inv(ash))
    LT = g.f([1, 1, 30, 30], (np.tril(np.ones((30, 30)))).astype(np.float32))
    cum = g.nd("MatMul", [LT, bandstart])                             # [1,1,30,1]
    bandidx = g.mul(cum, a)
    kvec = g.f([1, 1, 1, 15], list(range(1, 16)))
    BS = g.lt(g.nd("Abs", [g.sub(bandidx, kvec)]), 0.5)               # [1,1,30,15]
    BSt = g.nd("Transpose", [BS], perm=[0, 1, 3, 2])                  # [1,1,15,30]
    OCC = g.gt(g.nd("MatMul", [BSt, M]), 0.5)                         # [1,1,15,30]

    kcol = g.f([1, 1, 15, 1], list(range(1, 16)))
    colidx = g.colidx()
    cmax15 = g.nd("ReduceMax", [g.mul(OCC, colidx)], axes=[3], keepdims=1)      # [1,1,15,1]
    cmin15 = g.sub(g.scal(CBIG),
                   g.nd("ReduceMax", [g.mul(OCC, g.sub(g.scal(CBIG), colidx))],
                        axes=[3], keepdims=1))
    width15 = g.add(g.sub(cmax15, cmin15), g.scal(1.0))
    exists15 = g.nd("ReduceMax", [OCC], axes=[3], keepdims=1)                    # [1,1,15,1]
    tscore = g.add(g.mul(width15, g.scal(16.0)), g.sub(g.scal(15.0), kcol))
    tscore = g.sub(g.mul(tscore, exists15), g.mul(g.inv(exists15), g.scal(1e6)))
    tmaxv = g.nd("ReduceMax", [tscore], axes=[2], keepdims=1)
    tsel = g.nd("Cast", [g.nd("Greater", [tscore, g.sub(tmaxv, g.scal(0.5))])], to=F)

    trow = g.nd("MatMul", [BS, tsel])                                 # [1,1,30,1]
    Tm = g.mul(M, trow)
    r0 = g.minrow(Tm)
    c0 = g.mincol(Tm)
    T0 = g.shift2(Tm, g.nd("Neg", [r0]), g.nd("Neg", [c0]))           # template at origin
    tcnt = g.nd("ReduceSum", [T0], axes=[2], keepdims=1)              # [1,1,1,30]

    dgr = g.f([1, 1, 59, 1], list(range(-29, 30)))                    # di values
    dgc = g.f([1, 1, 1, 59], list(range(-29, 30)))                    # dj values
    ugr = g.f([1, 1, 59, 1], list(range(59)))
    vgc = g.f([1, 1, 1, 59], list(range(59)))
    rowb = g.rowidx()                                                 # b index [1,1,30,1]

    ACC = None
    for k in range(1, 16):
        rowselk = g.nd("Slice", [BS, g.i64([k - 1]), g.i64([k]), g.i64([3])])  # [1,1,30,1]
        Bk = g.mul(M, rowselk)
        exk = g.nd("Slice", [exists15, g.i64([k - 1]), g.i64([k]), g.i64([2])])
        tsk = g.nd("Slice", [tsel, g.i64([k - 1]), g.i64([k]), g.i64([2])])
        gate = g.mul(exk, g.inv(tsk))                                 # [1,1,1,1]

        rmin = g.sub(g.scal(CBIG),
                     g.nd("ReduceMax", [g.mul(rowselk, g.sub(g.scal(CBIG), g.rowidx()))],
                          axes=[2], keepdims=1))
        rmax = g.nd("ReduceMax", [g.mul(rowselk, g.rowidx())], axes=[2], keepdims=1)
        cmin = g.nd("Slice", [cmin15, g.i64([k - 1]), g.i64([k]), g.i64([2])])
        cmax = g.nd("Slice", [cmax15, g.i64([k - 1]), g.i64([k]), g.i64([2])])
        oci = g.add(rmin, g.nd("Floor", [g.mul(g.add(g.sub(rmax, rmin), g.scal(1.0)),
                                               g.scal(0.5))]))
        ocj = g.add(cmin, g.nd("Floor", [g.mul(g.add(g.sub(cmax, cmin), g.scal(1.0)),
                                               g.scal(0.5))]))
        dloi = g.sub(oci, h2)
        dhii = g.add(dloi, g.sub(hh, g.scal(1.0)))
        dloj = g.sub(ocj, w2)
        dhij = g.add(dloj, g.sub(ww, g.scal(1.0)))
        validr = g.mul(g.nd("Cast", [g.nd("Greater", [dgr, g.sub(dloi, g.scal(0.5))])], to=F),
                       g.nd("Cast", [g.nd("Less", [dgr, g.add(dhii, g.scal(0.5))])], to=F))
        validc = g.mul(g.nd("Cast", [g.nd("Greater", [dgc, g.sub(dloj, g.scal(0.5))])], to=F),
                       g.nd("Cast", [g.nd("Less", [dgc, g.add(dhij, g.scal(0.5))])], to=F))

        corr = g.nd("Conv", [Bk, T0], kernel_shape=[30, 30], pads=[29, 29, 29, 29])
        Ap = g.nd("Cast", [g.nd("Less", [g.add(rowb, dgc), g.add(cmax, g.scal(0.5))])], to=F)
        nlow = g.nd("MatMul", [tcnt, Ap])                             # [1,1,1,59]
        pref = g.nd("Cast", [g.nd("Greater", [g.sub(corr, nlow), g.scal(-0.5)])], to=F)
        posit = g.gt(corr, 0.5)
        vp = g.mul(g.mul(validr, validc), posit)
        score = g.add(g.add(g.mul(corr, g.scal(8192.0)), g.mul(pref, g.scal(4096.0))),
                      g.add(g.mul(vgc, g.scal(64.0)), ugr))
        score = g.sub(g.mul(score, vp), g.mul(g.inv(vp), g.scal(1e9)))
        smax = g.nd("ReduceMax", [score], axes=[2, 3], keepdims=1)
        sel = g.nd("Cast", [g.nd("Greater", [score, g.sub(smax, g.scal(0.5))])], to=F)
        cnt = g.nd("ReduceSum", [sel], axes=[2, 3], keepdims=1)
        di = g.nd("Div", [g.nd("ReduceSum", [g.mul(sel, dgr)], axes=[2, 3], keepdims=1), cnt])
        dj = g.nd("Div", [g.nd("ReduceSum", [g.mul(sel, dgc)], axes=[2, 3], keepdims=1), cnt])
        placed = g.shift2(T0, di, dj)
        addk = g.mul(g.mul(g.mul(placed, g.inv(Bk)), Rm), gate)
        ACC = addk if ACC is None else g.add(ACC, addk)

    out1 = g.gt(ACC, 0.5)
    ch8 = g.mul(M, g.inv(out1))
    ch0 = g.mul(B0, g.inv(out1))
    Z = g.sub(M, M)
    g.nd("Concat", [ch0, out1, Z, Z, Z, Z, Z, Z, ch8, Z], "output", axis=1)
    return g.model("vc2_219")


# =============================================================================
# task018  (verify_0e206a2e)
# =============================================================================
_SLOTS18 = 6
_FLOOD18 = 14


def _comps4_np(NB):
    h, w = NB.shape
    lab = np.full((h, w), -1, int)
    out = []
    for i in range(h):
        for j in range(w):
            if NB[i, j] and lab[i, j] < 0:
                st = [(i, j)]; lab[i, j] = len(out); cs = []
                while st:
                    a, b = st.pop(); cs.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and NB[na, nb] and lab[na, nb] < 0:
                            lab[na, nb] = len(out); st.append((na, nb))
                out.append(cs)
    return out


def _tf18(G, kind):
    if kind == "id":
        return G
    if kind == "d":
        return G.T.copy()
    if kind == "c":
        return G[::-1, ::-1].T.copy()
    if kind == "h":
        return G[::-1, :].copy()
    if kind == "v":
        return G[:, ::-1].copy()
    if kind == "r90":
        return np.rot90(G, k=-1).copy()
    if kind == "r180":
        return np.rot90(G, k=2).copy()
    if kind == "r270":
        return np.rot90(G, k=1).copy()
    raise ValueError(kind)


def _solve018(I):
    """numpy mirror of the ONNX graph. None if outside the graph's regime."""
    I = np.asarray(I)
    if I.ndim != 2 or max(I.shape) > 30:
        return None
    vals, counts = np.unique(I, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    h, w = I.shape
    NB = (I != 0)
    deg = np.zeros((h, w), int)
    deg[1:, :] += NB[:-1, :]; deg[:-1, :] += NB[1:, :]
    deg[:, 1:] += NB[:, :-1]; deg[:, :-1] += NB[:, 1:]
    ncols = np.zeros((h, w), int)
    for c in np.unique(I):
        if c == 0:
            continue
        P = np.pad((I == c).astype(np.int64), ((6, 6), (6, 6)))
        S = P.cumsum(0).cumsum(1)
        S = np.pad(S, ((1, 0), (1, 0)))
        cnt = (S[13:13 + h, 13:13 + w] - S[:h, 13:13 + w]
               - S[13:13 + h, :w] + S[:h, :w])
        ncols += cnt > 0
    seed = NB & (deg >= 2) & (ncols >= 4)
    comps = _comps4_np(NB)
    true4 = [frozenset(c) for c in comps
             if len({int(I[a, b]) for a, b in c}) == 4]
    # slot extraction simulation
    Rs = seed.copy()
    got = []
    for _ in range(_SLOTS18):
        if not Rs.any():
            break
        ii, jj = min(zip(*np.nonzero(Rs)))
        F = np.zeros((h, w), bool); F[ii, jj] = True
        for _ in range(_FLOOD18):
            D = F.copy()
            D[1:, :] |= F[:-1, :]; D[:-1, :] |= F[1:, :]
            D[:, 1:] |= F[:, :-1]; D[:, :-1] |= F[:, 1:]
            F = D & NB
        D = F.copy()
        D[1:, :] |= F[:-1, :]; D[:-1, :] |= F[1:, :]
        D[:, 1:] |= F[:, :-1]; D[:, :-1] |= F[:, 1:]
        if ((D & NB) != F).any():
            return None                                  # flood not saturated in 14 steps
        comp = frozenset(zip(*np.nonzero(F)))
        Rs &= ~F
        if len({int(I[a, b]) for a, b in comp}) == 4:
            got.append(comp)
    if Rs.any():
        return None                                      # more seeded comps than slots
    if set(got) != set(true4):
        return None                                      # seed rule missed a template
    sources = []
    for comp in got:
        colc = {}
        for a, b in comp:
            colc[int(I[a, b])] = colc.get(int(I[a, b]), 0) + 1
        mx = max(colc.values())
        Ds = [c for c, n in colc.items() if n == mx]
        if len(Ds) != 1:
            return None                                  # most-colour tie
        D = Ds[0]
        rare = [(a, b) for a, b in comp if I[a, b] != D]
        ar = min(a for a, b in rare); ac = min(b for a, b in rare)
        roff = [(a - ar, b - ac, int(I[a, b])) for a, b in rare]
        coff = [(a - ar, b - ac, int(I[a, b])) for a, b in comp]
        if not all(0 <= dr <= 6 and 0 <= dc <= 6 for dr, dc, _ in roff):
            return None                                  # rare outside 7x7 kernel
        if not all(-6 <= dr <= 6 and -6 <= dc <= 6 for dr, dc, _ in coff):
            return None                                  # object outside 13x13 kernel
        sources.append((roff, coff))
    g = I.copy()
    for comp in got:
        for a, b in comp:
            g[a, b] = 0
    for kind, back in (("id", "id"), ("d", "d"), ("c", "c"), ("h", "h"), ("v", "v"),
                       ("r270", "r90"), ("r180", "r180"), ("r90", "r270")):
        gp = _tf18(g, kind)
        hp, wp = gp.shape
        stamps = {}
        for roff, coff in sources:
            for li in range(hp):
                for lj in range(wp):
                    ok = True
                    for dr, dc, v in roff:
                        a, b = li + dr, lj + dc
                        if not (0 <= a < hp and 0 <= b < wp) or gp[a, b] != v:
                            ok = False
                            break
                    if not ok:
                        continue
                    for dr, dc, v in coff:
                        a, b = li + dr, lj + dc
                        if 0 <= a < hp and 0 <= b < wp:
                            if (a, b) in stamps and stamps[(a, b)] != v:
                                return None              # paint conflict
                            stamps[(a, b)] = v
        for (a, b), v in stamps.items():
            gp[a, b] = v
        g = _tf18(gp, back)
    return g


def _build018():
    g = _G()
    CH = g.ch("input", 1, 10)                                       # [1,9,30,30]
    NB = g.gt(g.nd("ReduceSum", [CH], axes=[1], keepdims=1), 0.5)

    degK = g.f([1, 1, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1, 0])
    deg = g.nd("Conv", [NB, degK], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    deg2 = g.gt(deg, 1.5)
    w13 = g.f([9, 1, 13, 13], np.ones((9, 1, 13, 13)))
    wsum = g.nd("Conv", [CH, w13], kernel_shape=[13, 13], pads=[6, 6, 6, 6], group=9)
    ncol = g.nd("ReduceSum", [g.gt(wsum, 0.5)], axes=[1], keepdims=1)
    seed = g.mul(g.mul(NB, deg2), g.gt(ncol, 3.5))

    plusC = g.f([1, 1, 3, 3], [0, 1, 0, 1, 1, 1, 0, 1, 0])
    Rs = seed
    SRC = None
    matchWs, stampWs, thrs = [], [], []
    for _ in range(_SLOTS18):
        r0 = g.minrow(Rs)
        rowsel = g.lt(g.nd("Abs", [g.sub(g.rowidx(), r0)]), 0.5)
        inrow = g.mul(Rs, rowsel)
        c0 = g.sub(g.scal(CBIG),
                   g.nd("ReduceMax",
                        [g.mul(g.nd("ReduceMax", [inrow], axes=[2], keepdims=1),
                               g.sub(g.scal(CBIG), g.colidx()))], axes=[3], keepdims=1))
        colsel = g.lt(g.nd("Abs", [g.sub(g.colidx(), c0)]), 0.5)
        Fp = g.mul(rowsel, colsel)
        for _ in range(_FLOOD18):
            d = g.nd("Conv", [Fp, plusC], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
            Fp = g.gt(g.mul(d, NB), 0.5)
        comp = Fp
        Rs = g.mul(Rs, g.inv(comp))
        pc = g.nd("ReduceMax", [g.mul(CH, comp)], axes=[2, 3], keepdims=1)   # [1,9,1,1]
        nc = g.nd("ReduceSum", [pc], axes=[1], keepdims=1)
        valid = g.mul(g.gt(nc, 3.5), g.lt(nc, 4.5))
        srcm = g.mul(comp, valid)
        SRC = srcm if SRC is None else g.add(SRC, srcm)
        CP = g.mul(CH, srcm)
        cnt = g.nd("ReduceSum", [CP], axes=[2, 3], keepdims=1)               # [1,9,1,1]
        cmx = g.nd("ReduceMax", [cnt], axes=[1], keepdims=1)
        Dsel = g.nd("Cast", [g.nd("Greater", [cnt, g.sub(cmx, g.scal(0.5))])], to=F)
        Dmask = g.nd("ReduceSum", [g.mul(CP, Dsel)], axes=[1], keepdims=1)
        rare = g.sub(srcm, Dmask)
        ar = g.minrow(rare)
        ac = g.mincol(rare)
        RPn = g.shift2(g.mul(CP, rare), g.nd("Neg", [ar]), g.nd("Neg", [ac]))
        matchWs.append(g.nd("Slice", [RPn, g.i64([0, 0]), g.i64([7, 7]), g.i64([2, 3])]))
        CPn = g.shift2(CP, g.sub(g.scal(6.0), ar), g.sub(g.scal(6.0), ac))
        stampWs.append(g.nd("Slice", [CPn, g.i64([0, 0]), g.i64([13, 13]), g.i64([2, 3])]))
        nrare = g.nd("ReduceSum", [rare], axes=[2, 3], keepdims=1)
        thrs.append(g.add(g.sub(nrare, g.scal(0.5)), g.mul(g.inv(valid), g.scal(1e6))))
    matchW = g.nd("Concat", matchWs, axis=0)                                 # [6,9,7,7]
    stampW = g.nd("Concat", stampWs, axis=0)                                 # [6,9,13,13]
    thrv = g.nd("Concat", thrs, axis=1)                                      # [1,6,1,1]

    ch0cov = g.add(g.ch("input", 0, 1), SRC)
    G = g.nd("Concat", [ch0cov, g.mul(CH, g.inv(SRC))], axis=1)              # covered grid

    def flipmat(dim_scalar):
        s = g.add(g.rowidx(), g.colidx())                                    # i+k
        return g.lt(g.nd("Abs", [g.sub(s, g.sub(dim_scalar, g.scal(1.0)))]), 0.5)

    def transform(G, kind):
        if kind == "id":
            return G
        R = g.realmask(G)
        if kind == "d":
            return g.nd("Transpose", [G], perm=[0, 1, 3, 2])
        rh = g.add(g.maxrow(R), g.scal(1.0))
        rw = g.add(g.maxcol(R), g.scal(1.0))
        if kind == "h":
            return g.nd("MatMul", [flipmat(rh), G])
        if kind == "v":
            return g.nd("MatMul", [G, flipmat(rw)])
        if kind == "r180":
            return g.nd("MatMul", [flipmat(rh), g.nd("MatMul", [G, flipmat(rw)])])
        if kind == "c":
            return g.nd("Transpose",
                        [g.nd("MatMul", [flipmat(rh), g.nd("MatMul", [G, flipmat(rw)])])],
                        perm=[0, 1, 3, 2])
        if kind == "r90":
            return g.nd("Transpose", [g.nd("MatMul", [flipmat(rh), G])], perm=[0, 1, 3, 2])
        if kind == "r270":
            return g.nd("Transpose", [g.nd("MatMul", [G, flipmat(rw)])], perm=[0, 1, 3, 2])
        raise ValueError(kind)

    stages = (("id", "id"), ("d", "d"), ("c", "c"), ("h", "h"), ("v", "v"),
              ("r270", "r90"), ("r180", "r180"), ("r90", "r270"))
    for si, (kind, back) in enumerate(stages):
        Gp = transform(G, kind)
        CHp = g.ch(Gp, 1, 10)
        match = g.nd("Conv", [CHp, matchW], kernel_shape=[7, 7], pads=[0, 0, 6, 6])
        mm = g.nd("Cast", [g.nd("Greater", [match, thrv])], to=F)            # [1,6,30,30]
        ST = g.nd("ConvTranspose", [mm, stampW], kernel_shape=[13, 13])      # [1,9,42,42]
        STc = g.nd("Slice", [ST, g.i64([6, 6]), g.i64([36, 36]), g.i64([2, 3])])
        Rp = g.realmask(Gp)
        stf = g.mul(g.gt(STc, 0.5), Rp)
        anym = g.gt(g.nd("ReduceSum", [stf], axes=[1], keepdims=1), 0.5)
        keep = g.inv(anym)
        ch0n = g.mul(g.ch(Gp, 0, 1), keep)
        chn = g.nd("Max", [stf, g.mul(CHp, keep)])
        Gp2 = g.nd("Concat", [ch0n, chn], axis=1)
        last = (si == len(stages) - 1)
        if back == "id":
            if last:
                g.nd("Add", [Gp2, g.scal(0.0)], "output")
            else:
                G = Gp2
        else:
            out = transform(Gp2, back)
            if last:
                g.nodes[-1].output[0] = "output"
            else:
                G = out
    return g.model("vc2_018")


# =============================================================================
# candidates
# =============================================================================
def _pairs(examples, splits=("train", "test")):
    out = []
    for s in splits:
        for e in examples.get(s, []):
            out.append((np.asarray(e["input"]), np.asarray(e["output"])))
    return out


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return

    # ---- task118 ----
    if all(set(np.unique(a)) <= {0, 2, 5} and set(np.unique(b)) <= {0, 2, 5, 8}
           for a, b in prs):
        ok = True
        for a, b in _pairs(examples, ("train", "test", "arc-gen")):
            r = _solve118(a)
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            yield ("vc2_118_crossrepair", _build118())

    # ---- task219 ----
    if all(set(np.unique(a)) <= {0, 8} and set(np.unique(b)) <= {0, 1, 8}
           for a, b in prs):
        ok = True
        for a, b in _pairs(examples, ("train", "test", "arc-gen")):
            r = _solve219(a)
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            yield ("vc2_219_bandcomplete", _build219())

    # ---- task018 ----
    if all(a.shape == b.shape and len(np.unique(a)) == 5
           and np.bincount(a.ravel(), minlength=10).argmax() == 0 for a, b in prs):
        ok = True
        for a, b in _pairs(examples, ("train", "test", "arc-gen")):
            r = _solve018(a)
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            yield ("vc2_018_markerstamp", _build018())
