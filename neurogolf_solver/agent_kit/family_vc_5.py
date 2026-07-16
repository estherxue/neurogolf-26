"""family_vc_5 — ground-truth-rule translations to static opset-10 ONNX for:

  task046 (verify_234bbc79): 4-connected multicolor pieces carry a "glue" colour
      (the least-frequent colour appearing 1-2x in EVERY piece).  Pieces are
      chained left-to-right: each piece's left glue cell is recoloured to the
      colour of its adjacent cell and placed one step right of the previous
      piece's (also recoloured) right glue cell.  Output = the merged chain
      painted on a bg canvas, cropped to (input height, merged width); a
      surviving trailing glue cell is recoloured to its neighbours' majority.

  task158 (verify_6aa20dc0): the most-multicoloured object is a template.  For
      every scale 1-4 and mirror {id,h,v,d,c}, wherever the template's minority
      cells (scaled+mirrored, normalised) occur exactly AND isolated (their
      4-neighbour halo is bg / off-grid), the full scaled+mirrored template is
      painted with its ulcorner at the matched minority ulcorner.

  task285 (verify_b775ac94): each 8-connected multicolor object = a main shape
      (majority colour) + seed cells.  Each seed lies in one of the 8 regions
      obtained by shifting the main bbox by (di*h, dj*w); the main shape is
      mirrored across that bbox edge into the region and recoloured to the
      seed's colour.

Machinery: 8/4-conn CA component labelling, [900,900] same-label matrix for
per-object aggregates, data-dependent selection-matrix MatMuls for anchoring /
mirroring / per-cell scatter, runtime-weight Conv (pattern match) and
ConvTranspose (stamping).  All arithmetic is exact 0/1-integer float32.
"""
from collections import deque

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64
I32 = onnx.TensorProto.INT32
HW = 900
BIG = 10000.0


# =========================================================================== #
#  numpy references (exact reimplementations of the verifier rules)           #
# =========================================================================== #
def _grid_bg(a):
    return int(np.bincount(a.ravel(), minlength=10).argmax())


def _components(a, bg, diag):
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    dirs = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if diag else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    comps = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == bg:
                continue
            q = deque([(i, j)]); seen[i, j] = True; cells = [(i, j)]
            while q:
                r, c = q.popleft()
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] != bg:
                        seen[nr, nc] = True; q.append((nr, nc)); cells.append((nr, nc))
            comps.append(cells)
    return comps


def _ca_depth(a, bg, diag):
    """CA iterations needed for max-position-label propagation to converge."""
    worst = 0
    dirs = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if diag else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    for cells in _components(a, bg, diag):
        cset = set(cells)
        root = max(cells, key=lambda rc: rc[0] * 30 + rc[1])
        dist = {root: 0}; q = deque([root])
        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                if nb in cset and nb not in dist:
                    dist[nb] = dist[(r, c)] + 1; q.append(nb)
        worst = max(worst, max(dist.values(), default=0))
    return worst


def _solve_b775ac94(a):
    a = np.asarray(a, int)
    h, w = a.shape
    bg = _grid_bg(a)
    out = a.copy()
    paint = {}
    for cells in _components(a, bg, diag=True):
        cnt = np.bincount([a[i, j] for i, j in cells], minlength=10)
        mc = int(cnt.argmax())
        main = [(i, j) for i, j in cells if a[i, j] == mc]
        seeds = [(i, j) for i, j in cells if a[i, j] != mc]
        rs = [i for i, _ in main]; cs = [j for _, j in main]
        minr, maxr, minc, maxc = min(rs), max(rs), min(cs), max(cs)
        h0, w0 = maxr - minr + 1, maxc - minc + 1
        for (si, sj) in seeds:
            found = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di, dj) != (0, 0) and (minr + di * h0 <= si <= maxr + di * h0
                                               and minc + dj * w0 <= sj <= maxc + dj * w0):
                        found = (di, dj)
            if found is None:
                return None
            di, dj = found
            sc = int(a[si, sj])
            for (pi, pj) in main:
                ti = pi if di == 0 else minr + maxr - pi + di * h0
                tj = pj if dj == 0 else minc + maxc - pj + dj * w0
                if 0 <= ti < h and 0 <= tj < w:
                    if paint.get((ti, tj), sc) != sc:
                        return None            # conflicting paint -> not this rule
                    paint[(ti, tj)] = sc
    for (i, j), c in paint.items():
        out[i, j] = c
    return out


def _solve_6aa20dc0(a):
    a = np.asarray(a, int)
    h, w = a.shape
    bg = _grid_bg(a)
    comps = _components(a, bg, diag=True)
    if not comps:
        return None
    ncol = [len({int(a[i, j]) for i, j in cells}) for cells in comps]
    mx = max(ncol)
    merged = [ij for cells, n in zip(comps, ncol) if n == mx for ij in cells]
    rs = [i for i, _ in merged]; cs = [j for _, j in merged]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    tmpl = {(int(a[i, j]), (i - r0, j - c0)) for i in range(r0, r1 + 1)
            for j in range(c0, c1 + 1) if a[i, j] != bg}
    if len({v for v, _ in tmpl}) < 2:
        return None
    dom = int(np.bincount([v for v, _ in tmpl], minlength=10).argmax())
    P = np.full((h + 2, w + 2), bg, int)
    P[1:h + 1, 1:w + 1] = a
    paint = {}
    for s in (1, 2, 3, 4):
        up = {(v, (i * s + x, j * s + y)) for v, (i, j) in tmpl
              for x in range(s) for y in range(s)}
        th = (r1 - r0 + 1) * s; tw = (c1 - c0 + 1) * s
        if th > 30 or tw > 30:                 # mirror the ONNX gate
            continue
        variants = {
            "i": up,
            "h": {(v, (th - 1 - i, j)) for v, (i, j) in up},
            "v": {(v, (i, tw - 1 - j)) for v, (i, j) in up},
            "d": {(v, (j, i)) for v, (i, j) in up},
            "c": {(v, (tw - 1 - j, th - 1 - i)) for v, (i, j) in up},
        }
        for X in variants.values():
            minority = {(v, ij) for v, ij in X if v != dom}
            mr = min(i for _, (i, j) in minority); mc = min(j for _, (i, j) in minority)
            mn = {(v, (i - mr, j - mc)) for v, (i, j) in minority}
            mnidx = {ij for _, ij in mn}
            halo = {(i + di, j + dj) for (i, j) in mnidx
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))} - mnidx
            for qi in range(h):
                for qj in range(w):
                    ok = all(0 <= qi + pi < h and 0 <= qj + pj < w
                             and a[qi + pi, qj + pj] == v for v, (pi, pj) in mn)
                    if not ok:
                        continue
                    if any(P[qi + 1 + pi, qj + 1 + pj] != bg for (pi, pj) in halo):
                        continue
                    for v, (xi, xj) in X:
                        ti, tj = qi + xi, qj + xj
                        if 0 <= ti < h and 0 <= tj < w:
                            if paint.get((ti, tj), v) != v:
                                return None    # conflicting stamp -> reject
                            paint[(ti, tj)] = v
    out = a.copy()
    for (i, j), v in paint.items():
        out[i, j] = v
    return out


def _solve_234bbc79(a):
    a = np.asarray(a, int)
    h, w = a.shape
    bg = _grid_bg(a)
    comps = _components(a, bg, diag=False)
    n = len(comps)
    if n < 2:
        return None
    cand = [c for c in sorted({int(v) for v in a.ravel()})
            if all(1 <= sum(1 for i, j in cells if a[i, j] == c) <= 2 for cells in comps)]
    if not cand:
        return None
    glue = min(cand, key=lambda c: int((a == c).sum()))
    lms = [min(j for _, j in cells) for cells in comps]
    if len(set(lms)) != n:
        return None
    objs = sorted(comps, key=lambda cells: min(j for _, j in cells))
    pieces = [{(i, j): int(a[i, j]) for i, j in cells} for cells in objs]
    g0 = [p for p, v in pieces[0].items() if v == glue]
    if len(g0) != 1:
        return None
    cur = dict(pieces[0]); cur_glue = g0[0]
    for k in range(1, n):
        nxt = dict(pieces[k])
        gl = [p for p, v in nxt.items() if v == glue]
        if not gl or len(gl) > 2:
            return None
        mincol = min(p[1] for p in gl)
        lgs = [p for p in gl if p[1] == mincol]
        if len(lgs) != 1 or (k < n - 1 and len(gl) != 2):
            return None
        lg = lgs[0]
        # recolour current glue: nearest cell of (cur - glue); must be dist-1 unique-colour
        cands = [(p, v) for p, v in cur.items() if p != cur_glue]
        d1 = [(p, v) for p, v in cands
              if abs(p[0] - cur_glue[0]) + abs(p[1] - cur_glue[1]) == 1]
        if not d1 or len({v for _, v in d1}) != 1:
            return None
        cur[cur_glue] = d1[0][1]
        # recolour next's left glue: nearest NON-GLUE cell; dist-1 unique-colour
        nc = [(p, v) for p, v in nxt.items() if p != lg and v != glue]
        d2 = [(p, v) for p, v in nc if abs(p[0] - lg[0]) + abs(p[1] - lg[1]) == 1]
        if not d2 or len({v for _, v in d2}) != 1:
            return None
        nxt[lg] = d2[0][1]
        sh = (cur_glue[0] - lg[0], cur_glue[1] - lg[1] + 1)
        for p, v in nxt.items():
            tp = (p[0] + sh[0], p[1] + sh[1])
            if tp in cur:
                return None                    # overlap -> not this rule
            cur[tp] = v
        rg = [p for p, v in cur.items() if v == glue]
        if k < n - 1:
            if len(rg) != 1:
                return None
            cur_glue = rg[0]
    allc = [p[1] for p in cur]
    if min(p[0] for p in cur) < 0 or min(allc) < 0:
        return None
    wid = max(allc) - min(allc) + 1
    out = np.full((h, wid), bg, int)
    for (i, j), v in cur.items():
        if 0 <= i < h and 0 <= j < wid:
            out[i, j] = v
    gpos = list(zip(*np.where(out == glue)))
    if gpos:
        nbr = {(i + di, j + dj) for (i, j) in gpos
               for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))}
        vals = [int(out[i, j]) for i, j in nbr
                if 0 <= i < h and 0 <= j < wid and out[i, j] != bg]
        if vals:
            cnt = np.bincount(vals, minlength=10)
            if (cnt == cnt.max()).sum() > 1:
                return None
            mcv = int(cnt.argmax())
            for (i, j) in gpos:
                out[i, j] = mcv
    return out


# =========================================================================== #
#  ONNX toolkit                                                               #
# =========================================================================== #
class _G:
    def __init__(self):
        self.nodes, self.inits = [], []
        self._k = 0
        self._cache = {}

    def nm(self, p="t"):
        self._k += 1
        return f"v{self._k}{p}"

    def f(self, arr, dims=None):
        a = np.asarray(arr, np.float32)
        dims = list(a.shape) if dims is None else list(dims)
        key = ("f", tuple(dims), a.tobytes())
        if key in self._cache:
            return self._cache[key]
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, dims, a.ravel().tolist()))
        self._cache[key] = n
        return n

    def i(self, vals, dims=None):
        a = np.asarray(vals, np.int64)
        dims = list(a.shape) if dims is None else list(dims)
        key = ("i", tuple(dims), a.tobytes())
        if key in self._cache:
            return self._cache[key]
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, I64, dims, a.ravel().tolist()))
        self._cache[key] = n
        return n

    def n(self, op, ins, out=None, **attrs):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    # ---- elementwise helpers (all 4-D, exact-integer float32) ----
    def add(self, a, b): return self.n("Add", [a, b])

    def sub(self, a, b): return self.n("Sub", [a, b])

    def mul(self, *xs):
        r = xs[0]
        for x in xs[1:]:
            r = self.n("Mul", [r, x])
        return r

    def castf(self, x): return self.n("Cast", [x], to=F)

    def ltf(self, a, b): return self.castf(self.n("Less", [a, b]))

    def gtf(self, a, b): return self.castf(self.n("Greater", [a, b]))

    def feq(self, a, b):
        return self.ltf(self.n("Abs", [self.sub(a, b)]), self.f([0.5], [1]))

    def notf(self, x): return self.sub(self.f([1.0], [1]), x)

    def rs(self, x, shape): return self.n("Reshape", [x, self.i(shape)])

    def rsum(self, x, axes, keep=1):
        return self.n("ReduceSum", [x], axes=axes, keepdims=keep)

    def rmax(self, x, axes, keep=1):
        return self.n("ReduceMax", [x], axes=axes, keepdims=keep)

    def shift(self, x, dr, dc, hw=30):
        """origin-preserving spatial shift of [1,1,hw,hw]; new cells zero."""
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        p = self.n("Pad", [x], mode="constant", value=0.0,
                   pads=[0, 0, pt, pl, 0, 0, pb, pr])
        st = self.i([max(-dr, 0), max(-dc, 0)])
        en = self.i([max(-dr, 0) + hw, max(-dc, 0) + hw])
        return self.n("Slice", [p, st, en, self.i([2, 3])])

    # ---- shared grid constants ----
    def rowv(self):  return self.f(np.arange(30).reshape(1, 1, 30, 1))

    def colv(self):  return self.f(np.arange(30).reshape(1, 1, 1, 30))

    def rowflat(self):
        return self.f(np.repeat(np.arange(30), 30).reshape(1, 1, HW, 1))

    def colflat(self):
        return self.f(np.tile(np.arange(30), 30).reshape(1, 1, HW, 1))

    def posidx(self):
        return self.f(np.arange(1, HW + 1).reshape(1, 1, 30, 30))

    def rowmat(self):        # [1,1,900,30]: [p,r] = (p_i == r)
        m = np.zeros((HW, 30)); m[np.arange(HW), np.arange(HW) // 30] = 1
        return self.f(m.reshape(1, 1, HW, 30))

    def colmat(self):        # [1,1,900,30]: [p,j] = (p_j == j)
        m = np.zeros((HW, 30)); m[np.arange(HW), np.arange(HW) % 30] = 1
        return self.f(m.reshape(1, 1, HW, 30))

    def rowmatT(self):       # [1,1,30,900]: [r,p] = (p_i == r)
        m = np.zeros((30, HW)); m[np.arange(HW) // 30, np.arange(HW)] = 1
        return self.f(m.reshape(1, 1, 30, HW))

    def colmatT(self):       # [1,1,900,30] transposed use: [p,j] = (p_j == j)
        return self.colmat()

    def ar10f(self):  return self.f(np.arange(10).reshape(1, 10, 1, 1))

    def ar10i(self):  return self.i(np.arange(10).reshape(1, 10, 1, 1))

    def half(self):   return self.f([0.5], [1])

    def one(self):    return self.f([1.0], [1])


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "vc5", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ---- shared subgraphs ------------------------------------------------------ #
def _base(g):
    """R (real mask), bg one-hots, N (non-bg mask), all [1,1,30,30]/[1,10,1,1]."""
    R = g.rsum("input", [1])                                    # [1,1,30,30]
    counts = g.rsum("input", [2, 3])                            # [1,10,1,1]
    bgi = g.n("ArgMax", [counts], axis=1, keepdims=1)           # [1,1,1,1] i64
    bgOH = g.castf(g.n("Equal", [g.ar10i(), bgi]))              # [1,10,1,1]
    bgch = g.rsum(g.mul("input", bgOH), [1])                    # [1,1,30,30]
    N = g.sub(R, bgch)
    return R, N, bgOH, counts


def _labels(g, N, T, diag):
    """CA max-label propagation over mask N -> [1,1,30,30] float labels."""
    dirs = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if diag else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    L = g.mul(N, g.posidx())
    for _ in range(T):
        nb = [g.shift(L, dr, dc) for dr, dc in dirs]
        L = g.mul(g.n("Max", [L] + nb), N)
    return L


def _ematrix(g, L):
    """[1,1,900,900] same-label (component) matrix."""
    Lc = g.n("Cast", [g.rs(L, [1, 1, HW, 1])], to=I32)
    Lr = g.n("Cast", [g.rs(L, [1, 1, 1, HW])], to=I32)
    return g.castf(g.n("Equal", [Lc, Lr]))


def _xt900(g):
    """input one-hot as [1,1,900,10] (rows=cells, cols=colors)."""
    a = g.rs("input", [1, 10, HW, 1])
    b = g.n("Transpose", [a], perm=[0, 2, 1, 3])                # [1,900,10,1]
    return g.rs(b, [1, 1, HW, 10])


def _bbox(g, mask):
    """bbox scalars (minr,maxr,minc,maxc) [1,1,1,1] of a [1,1,30,30] mask."""
    rowhas = g.rmax(mask, [3])                                  # [1,1,30,1]
    colhas = g.rmax(mask, [2])                                  # [1,1,1,30]
    c100 = g.f([100.0], [1])
    maxr = g.rmax(g.mul(rowhas, g.rowv()), [2])
    minr = g.sub(c100, g.rmax(g.mul(rowhas, g.sub(c100, g.rowv())), [2]))
    maxc = g.rmax(g.mul(colhas, g.colv()), [3])
    minc = g.sub(c100, g.rmax(g.mul(colhas, g.sub(c100, g.colv())), [3]))
    return minr, maxr, minc, maxc


def _anchor(g, X, minr, minc):
    """shift content of X so that (minr,minc) moves to the origin."""
    srow = g.feq(g.colv(), g.add(g.rowv(), minr))               # [1,1,30,30]
    scol = g.feq(g.rowv(), g.add(g.colv(), minc))
    return g.n("MatMul", [srow, g.n("MatMul", [X, scol])])


def _minsel(g, binmat):
    """per-row min index of [1,1,900,30] binary presence -> [1,1,900,1]."""
    c100 = g.f([100.0], [1])
    return g.sub(c100, g.rmax(g.mul(binmat, g.sub(c100, g.colv())), [3]))


# =========================================================================== #
#  task285 builder                                                            #
# =========================================================================== #
def _build_285(T):
    g = _G()
    R, N, bgOH, _ = _base(g)
    V = g.rsum(g.mul("input", g.ar10f()), [1])                  # colour values 2d
    L = _labels(g, N, T, diag=True)
    E = _ematrix(g, L)
    CNT = g.n("MatMul", [E, _xt900(g)])                         # [1,1,900,10]
    mainc = g.castf(g.n("ArgMax", [CNT], axis=3, keepdims=1))   # [1,1,900,1]
    Nf = g.rs(N, [1, 1, HW, 1])
    Vf = g.rs(V, [1, 1, HW, 1])
    m = g.mul(Nf, g.feq(Vf, mainc))                             # main-part cells
    # per-object bbox of the main part
    PR = g.n("MatMul", [E, g.mul(g.rowmat(), m)])               # [1,1,900,30]
    binr = g.gtf(PR, g.half())
    maxr = g.rmax(g.mul(binr, g.colv()), [3])
    minr = _minsel(g, binr)
    PC = g.n("MatMul", [E, g.mul(g.colmat(), m)])
    binc = g.gtf(PC, g.half())
    maxc = g.rmax(g.mul(binc, g.colv()), [3])
    minc = _minsel(g, binc)
    one = g.one()
    h0 = g.add(g.sub(maxr, minr), one)
    w0 = g.add(g.sub(maxc, minc), one)
    # seed directions
    sd = g.sub(Nf, m)
    rf, cf = g.rowflat(), g.colflat()
    di = g.sub(g.gtf(rf, maxr), g.ltf(rf, minr))
    dj = g.sub(g.gtf(cf, maxc), g.ltf(cf, minc))
    hf = g.half()
    vi = g.mul(g.ltf(g.sub(g.sub(minr, h0), hf), rf),
               g.ltf(rf, g.add(g.add(maxr, h0), hf)))
    vj = g.mul(g.ltf(g.sub(g.sub(minc, w0), hf), cf),
               g.ltf(cf, g.add(g.add(maxc, w0), hf)))
    nz = g.gtf(g.add(g.n("Abs", [di]), g.n("Abs", [dj])), hf)
    sv = g.mul(sd, vi, vj, nz)
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    sels = [g.mul(sv, g.feq(di, g.f([float(a)], [1])), g.feq(dj, g.f([float(b)], [1])))
            for a, b in dirs]
    S8 = g.n("Concat", sels, axis=3)                            # [1,1,900,8]
    G8 = g.n("MatMul", [E, S8])
    C8 = g.n("MatMul", [E, g.mul(S8, Vf)])
    # reflection row/col source maps
    sumr = g.add(minr, maxr)
    sumc = g.add(minc, maxc)
    rpos = g.rs(g.add(g.sub(sumr, rf), h0), [1, 1, 1, HW])
    rneg = g.rs(g.sub(g.sub(sumr, rf), h0), [1, 1, 1, HW])
    cpos = g.add(g.sub(sumc, cf), w0)                           # [1,1,900,1]
    cneg = g.sub(g.sub(sumc, cf), w0)
    Rpos = g.feq(g.rowv(), rpos)                                # [1,1,30,900]
    Rneg = g.feq(g.rowv(), rneg)
    R0 = g.rowmatT()
    Cpos = g.feq(cpos, g.colv())                                # [1,1,900,30]
    Cneg = g.feq(cneg, g.colv())
    C0 = g.colmat()
    cnts, cols = [], []
    for k, (a, b) in enumerate(dirs):
        gk = g.gtf(g.n("Slice", [G8, g.i([k]), g.i([k + 1]), g.i([3])]), hf)
        ck = g.n("Slice", [C8, g.i([k]), g.i([k + 1]), g.i([3])])
        wk = g.mul(m, gk)
        wT = g.rs(wk, [1, 1, 1, HW])
        vT = g.rs(g.mul(wk, ck), [1, 1, 1, HW])
        Rs = {-1: Rneg, 0: R0, 1: Rpos}[a]
        Cs = {-1: Cneg, 0: C0, 1: Cpos}[b]
        cnts.append(g.n("MatMul", [g.mul(Rs, wT), Cs]))         # [1,1,30,30]
        cols.append(g.n("MatMul", [g.mul(Rs, vT), Cs]))
    cntT = g.n("Sum", cnts)
    colT = g.n("Sum", cols)
    den = g.add(cntT, g.ltf(cntT, hf))
    cmap = g.n("Div", [colT, den])
    pm = g.mul(g.gtf(cntT, hf), R)                              # [1,1,30,30]
    ohp = g.mul(g.feq(cmap, g.ar10f()), pm)                     # [1,10,30,30]
    keepin = g.mul("input", g.notf(pm))
    g.n("Add", [ohp, keepin], out="output")
    return _model(g)


# =========================================================================== #
#  task158 builder                                                            #
# =========================================================================== #
def _build_158(T):
    g = _G()
    R, N, bgOH, _ = _base(g)
    L = _labels(g, N, T, diag=True)
    E = _ematrix(g, L)
    hf, one = g.half(), g.one()
    pres = g.gtf(g.n("MatMul", [E, _xt900(g)]), hf)             # [1,1,900,10]
    nc = g.rsum(pres, [3])                                      # [1,1,900,1]
    Nf = g.rs(N, [1, 1, HW, 1])
    ncm = g.rmax(g.mul(nc, Nf), [2])                            # [1,1,1,1]
    Gm = g.mul(Nf, g.gtf(nc, g.sub(ncm, hf)))
    G2 = g.rs(Gm, [1, 1, 30, 30])
    r0, r1, c0, c1 = _bbox(g, G2)
    inr = g.mul(g.gtf(g.rowv(), g.sub(r0, hf)), g.ltf(g.rowv(), g.add(r1, hf)))
    inc = g.mul(g.gtf(g.colv(), g.sub(c0, hf)), g.ltf(g.colv(), g.add(c1, hf)))
    Tm = g.mul(N, inr, inc)
    TOH = g.mul("input", Tm)
    TN = _anchor(g, TOH, r0, c0)                                # [1,10,30,30]
    th = g.add(g.sub(r1, r0), one)
    tw = g.add(g.sub(c1, c0), one)
    tcnt = g.rsum(TN, [2, 3])                                   # [1,10,1,1]
    dom = g.n("ArgMax", [tcnt], axis=1, keepdims=1)
    domOH = g.castf(g.n("Equal", [g.ar10i(), dom]))             # [1,10,1,1]
    notdom = g.notf(domOH)
    c30 = g.f([30.5], [1])
    stamps = []
    for s in (1, 2, 3, 4):
        if s == 1:
            TS, ths, tws = TN, th, tw
        else:
            u = np.zeros((30, 30)); u[np.arange(30), np.arange(30) // s] = 1
            Us = g.f(u.reshape(1, 1, 30, 30))
            UsT = g.f(u.T.reshape(1, 1, 30, 30))
            TS = g.n("MatMul", [Us, g.n("MatMul", [TN, UsT])])
            sc = g.f([float(s)], [1])
            ths, tws = g.mul(th, sc), g.mul(tw, sc)
        gate = g.mul(g.ltf(ths, c30), g.ltf(tws, c30))          # [1,1,1,1]
        rrev = g.feq(g.colv(), g.sub(g.sub(ths, one), g.rowv()))
        crev = g.feq(g.rowv(), g.sub(g.sub(tws, one), g.colv()))
        tv = g.n("MatMul", [TS, crev])
        variants = [TS,
                    g.n("MatMul", [rrev, TS]),
                    tv,
                    g.n("Transpose", [TS], perm=[0, 1, 3, 2]),
                    g.n("Transpose", [g.n("MatMul", [rrev, tv])], perm=[0, 1, 3, 2])]
        for TU in variants:
            MNt = g.mul(TU, notdom)
            mn2 = g.rmax(MNt, [1])                              # [1,1,30,30]
            mr, _, mc, _ = _bbox(g, mn2)
            MN0 = _anchor(g, MNt, mr, mc)                       # [1,10,30,30]
            mn0 = g.rmax(MN0, [1])
            nsc = g.rsum(MN0, [1, 2, 3])                        # [1,1,1,1]
            corr = g.n("Conv", ["input", MN0], pads=[0, 0, 29, 29])
            mA = g.mul(g.gtf(corr, g.sub(nsc, hf)), g.gtf(nsc, hf))
            mp = g.n("Pad", [mn0], mode="constant", value=0.0,
                     pads=[0, 0, 1, 1, 0, 0, 1, 1])             # [1,1,32,32]
            nb = [g.shift(mp, dr, dc, hw=32)
                  for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
            dil = g.n("Max", nb)
            haloW = g.mul(dil, g.notf(mp))                      # [1,1,32,32]
            viol = g.n("Conv", [N, haloW], pads=[1, 1, 30, 30])
            mB = g.ltf(viol, hf)
            match = g.mul(mA, mB, gate)                         # [1,1,30,30]
            stamps.append(g.n("ConvTranspose", [match, TU], pads=[0, 0, 29, 29]))
    ST = g.n("Sum", stamps)                                     # [1,10,30,30]
    pm = g.mul(g.gtf(g.rmax(ST, [1]), hf), R)
    g.n("Add", [g.mul(ST, pm), g.mul("input", g.notf(pm))], out="output")
    return _model(g)


# =========================================================================== #
#  task046 builder                                                            #
# =========================================================================== #
def _build_046(T):
    g = _G()
    R, N, bgOH, counts = _base(g)
    hf, one = g.half(), g.one()
    V = g.rsum(g.mul("input", g.ar10f()), [1])                  # [1,1,30,30]
    L = _labels(g, N, T, diag=False)
    E = _ematrix(g, L)
    CNT = g.n("MatMul", [E, _xt900(g)])                         # [1,1,900,10]
    Nf = g.rs(N, [1, 1, HW, 1])
    rf, cf = g.rowflat(), g.colflat()
    posf = g.f(np.arange(1, HW + 1).reshape(1, 1, HW, 1))
    rep = g.mul(Nf, g.feq(g.rs(L, [1, 1, HW, 1]), posf))        # object reps
    nobj = g.rsum(rep, [2])                                     # [1,1,1,1]
    ok = g.mul(g.gtf(CNT, hf), g.ltf(CNT, g.f([2.5], [1])))
    totok = g.rsum(g.mul(ok, rep), [2])                         # [1,1,1,10]
    allok = g.feq(totok, nobj)
    c10 = g.rs(counts, [1, 1, 1, 10])
    score = g.add(c10, g.mul(g.notf(allok), g.f([BIG], [1])))
    gam = g.n("ArgMin", [score], axis=3, keepdims=1)            # [1,1,1,1] i64
    gamOHr = g.castf(g.n("Equal", [g.i(np.arange(10).reshape(1, 1, 1, 10)), gam]))
    gamOH = g.rs(gamOHr, [1, 10, 1, 1])
    GL2 = g.rsum(g.mul("input", gamOH), [1])                    # [1,1,30,30]
    GLf = g.rs(GL2, [1, 1, HW, 1])
    # per-object leftmost (all cells) and glue-min-col
    bina = g.gtf(g.n("MatMul", [E, g.mul(g.colmat(), Nf)]), hf)
    lm = _minsel(g, bina)                                       # [1,1,900,1]
    bing = g.gtf(g.n("MatMul", [E, g.mul(g.colmat(), GLf)]), hf)
    gmc = _minsel(g, bing)
    lgm = g.mul(GLf, g.feq(cf, gmc))
    rgm = g.sub(GLf, lgm)
    C1 = g.n("Concat", [g.mul(lgm, rf), g.mul(lgm, cf),
                        g.mul(rgm, rf), g.mul(rgm, cf), GLf], axis=3)
    M1 = g.n("MatMul", [E, C1])                                 # [1,1,900,5]

    def col(x, k):
        return g.n("Slice", [x, g.i([k]), g.i([k + 1]), g.i([3])])

    lgr, lgc, rgsr, rgsc, gcnt = (col(M1, k) for k in range(5))
    has2 = g.gtf(gcnt, g.f([1.5], [1]))
    not2 = g.notf(has2)
    rgr = g.add(rgsr, g.mul(not2, lgr))
    rgc = g.add(rgsc, g.mul(not2, lgc))
    dr = g.sub(rgr, lgr)
    dc = g.sub(rgc, lgc)
    # prefix sums over objects ordered by leftmost (bin trick over lm in 0..29)
    lmT = g.rs(lm, [1, 1, 1, HW])
    IND = g.ltf(lmT, g.rowv())                                  # [1,1,30,900]
    Vst = g.n("Concat", [g.mul(rep, dr), g.mul(rep, dc), rep], axis=3)
    Sby = g.n("MatMul", [IND, Vst])                             # [1,1,30,3]
    lmOH = g.feq(lm, g.colv())                                  # [1,1,900,30]
    Spre = g.n("MatMul", [lmOH, Sby])                           # [1,1,900,3]
    Sr, Sc, rank = (col(Spre, k) for k in range(3))
    # first object's left-glue position
    minlm = g.n("ReduceMin", [g.add(lm, g.mul(g.notf(Nf), g.f([BIG], [1])))],
                axes=[2], keepdims=1)
    fm = g.mul(Nf, g.feq(lm, minlm))
    lgr1 = g.rsum(g.mul(lgm, fm, rf), [2])                      # [1,1,1,1]
    lgc1 = g.rsum(g.mul(lgm, fm, cf), [2])
    shr = g.sub(g.add(lgr1, Sr), lgr)                           # [1,1,900,1]
    shc = g.sub(g.add(g.add(lgc1, Sc), rank), lgc)
    # recolours (nearest = 4-neighbour, colour-unique; validated by the ref)
    candL = g.mul(g.sub(N, GL2), V)
    cL2 = g.n("Max", [g.shift(candL, dr_, dc_)
                      for dr_, dc_ in ((-1, 0), (1, 0), (0, -1), (0, 1))])
    lg2 = g.rs(lgm, [1, 1, 30, 30])
    V1 = g.add(g.mul(V, g.notf(lg2)), g.mul(cL2, lg2))
    candR = g.mul(N, V1)
    cR2 = g.n("Max", [g.shift(candR, dr_, dc_)
                      for dr_, dc_ in ((-1, 0), (1, 0), (0, -1), (0, 1))])
    islast = g.feq(rank, g.sub(nobj, one))
    rgm2 = g.mul(rgm, g.notf(islast))
    lgT = g.rs(lgm, [1, 1, 1, HW])
    rgT = g.rs(rgm2, [1, 1, 1, HW])
    cLT = g.rs(cL2, [1, 1, 1, HW])
    cRT = g.rs(cR2, [1, 1, 1, HW])
    Xc = g.rs("input", [1, 10, 1, HW])
    keep = g.notf(g.mul(gamOH, g.add(lgT, rgT)))                # [1,10,1,900]
    Xmod = g.add(g.add(g.mul(Xc, keep), g.mul(g.feq(g.ar10f(), cLT), lgT)),
                 g.mul(g.feq(g.ar10f(), cRT), rgT))
    Xmv = g.mul(Xmod, g.rs(Nf, [1, 1, 1, HW]))                  # [1,10,1,900]
    tR = g.rs(g.add(rf, shr), [1, 1, 1, HW])
    tC = g.add(cf, shc)                                         # [1,1,900,1]
    Ri = g.feq(g.rowv(), tR)                                    # [1,1,30,900]
    Cj = g.feq(tC, g.colv())                                    # [1,1,900,30]
    moved = g.n("MatMul", [g.mul(Ri, Xmv), Cj])                 # [1,10,30,30]
    any2 = g.rsum(moved, [1])
    _, _, mnc, mxc = _bbox(g, any2)
    wout = g.add(g.sub(mxc, mnc), one)
    rowmask = g.rmax(R, [3])                                    # [1,1,30,1]
    colkeep = g.ltf(g.colv(), g.sub(wout, hf))
    kp = g.mul(rowmask, colkeep)                                # [1,1,30,30]
    content = g.mul(moved, kp)
    occ = g.rsum(content, [1])
    baseo = g.add(content, g.mul(bgOH, g.mul(kp, g.notf(occ))))
    gm = g.rsum(g.mul(baseo, gamOH), [1])                       # [1,1,30,30]
    U = g.n("Max", [g.shift(gm, dr_, dc_)
                    for dr_, dc_ in ((-1, 0), (1, 0), (0, -1), (0, 1))])
    tc = g.rsum(g.mul(baseo, U), [2, 3])                        # [1,10,1,1]
    tsc = g.sub(tc, g.mul(bgOH, g.f([BIG], [1])))
    mcv = g.n("ArgMax", [tsc], axis=1, keepdims=1)
    mcvOH = g.castf(g.n("Equal", [g.ar10i(), mcv]))
    g.n("Add", [g.mul(baseo, g.notf(gm)), g.mul(mcvOH, gm)], out="output")
    return _model(g)


# =========================================================================== #
#  detection / entry point                                                    #
# =========================================================================== #
def _pairs(ex, keys=("train", "test")):
    out = []
    for s in keys:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim == 2 and b.ndim == 2 and a.size and b.size \
                    and max(a.shape + b.shape) <= 30:
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


def _depth_all(ex, diag):
    d = 0
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            if a.ndim == 2 and a.size and max(a.shape) <= 30:
                d = max(d, _ca_depth(a, _grid_bg(a), diag))
    return d


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return
    specs = [("vc5_b775ac94", _solve_b775ac94, _build_285, True),
             ("vc5_6aa20dc0", _solve_6aa20dc0, _build_158, True),
             ("vc5_234bbc79", _solve_234bbc79, _build_046, False)]
    for name, ref, builder, diag in specs:
        if not _matches(prs, ref):
            continue
        T = min(40, _depth_all(ex, diag) + 8)
        try:
            m = builder(T)
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            continue
        yield (name, m)
