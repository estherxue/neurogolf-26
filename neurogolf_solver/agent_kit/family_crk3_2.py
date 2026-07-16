"""family_crk3_2 -- crack module for slice IDX=2 of the unsolved NeuroGolf tasks.

Each detected task gets a structural detector validated EXACTLY against the
provided train/test pairs (numpy mirror) plus a static opset-10 ONNX builder.
All intermediates are static-shape; data-dependent geometry uses computed index
grids + ReduceMin/Max/Mod/Abs/Less and shift-by-Pad/Slice, never dynamic
Resize/Pad-with-runtime-shape.
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

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    # shift content of a [1,C,30,30] tensor by (dr,dc) (down-right positive)
    def shift(self, x, dr, dc):
        pt = max(dr, 0); pb = max(-dr, 0)
        pl = max(dc, 0); pr = max(-dc, 0)
        pads = [0, 0, pt, pl, 0, 0, pb, pr]
        p = self.nd("Pad", [x], mode="constant", pads=pads, value=0.0)
        s = self.i64([0, 0, pb, pr]); e = self.i64([1, 99, pb + 30, pr + 30])
        ax = self.i64([0, 1, 2, 3])
        return self.nd("Slice", [p, s, e, ax])


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _check(m):
    onnx.checker.check_model(m, full_check=True)
    return m


def _pairs(ex, keys=("train", "test")):
    out = []
    for k in keys:
        for p in ex.get(k, []):
            out.append((np.array(p["input"]), np.array(p["output"])))
    return out


def _slc(g, src, lo, hi, axis):
    s = g.i64([lo]); e = g.i64([hi]); a = g.i64([axis])
    return g.nd("Slice", [src, s, e, a])


# =========================================================================== #
# TASK 34 -- thick diagonal beam(s) from a 2x2 block marked with color 2      #
# =========================================================================== #
def _t34_rule(i):
    h, w = i.shape
    ys, xs = np.where(i != 0)
    if len(ys) != 4:
        return None
    r0, c0, r1, c1 = ys.min(), xs.min(), ys.max(), xs.max()
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    block = i[r0:r0 + 2, c0:c0 + 2]
    nz = block[block != 0]
    if len(nz) != 4:
        return None
    others = set(nz.tolist()) - {2}
    if len(others) != 1 or 2 not in set(nz.tolist()):
        return None
    fill = others.pop()
    cor = {(0, 0): (-1, -1), (0, 1): (-1, 1), (1, 0): (1, -1), (1, 1): (1, 1)}
    dirs = [d for (a, b), d in cor.items() if block[a, b] == 2]
    seed = np.zeros((h, w), bool)
    seed[r0:r0 + 2, c0:c0 + 2] = True
    out = seed.copy()
    for dr, dc in dirs:
        cur = seed.copy()
        for _ in range(max(h, w)):
            nxt = np.zeros((h, w), bool)
            yy, xx = np.where(cur)
            ny, nx = yy + dr, xx + dc
            ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            nxt[ny[ok], nx[ok]] = True
            new = nxt & ~out
            out |= nxt
            if not new.any():
                break
            cur = nxt
    res = np.zeros((h, w), dtype=i.dtype)
    res[out] = fill
    return res


def _t34_detect(prs):
    for i, o in prs:
        r = _t34_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
    return True


def _t34_build():
    g = _G()
    inp = "input"
    # grid mask & block & 2-mask
    grid = g.nd("ReduceSum", [inp], axes=[1], keepdims=1)            # [1,1,30,30] 1 on grid
    col = _slc(g, inp, 1, 10, 1)
    B = g.nd("ReduceSum", [col], axes=[1], keepdims=1)               # colored cells (block)
    T = _slc(g, inp, 2, 3, 1)                                        # 2-mask
    # F channel selector
    cnt = g.nd("ReduceSum", [inp], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    half = g.f([1, 1, 1, 1], [0.5])
    pos = g.nd("Cast", [g.nd("Greater", [cnt, half])], to=F)         # [1,10,1,1]
    allow = g.f([1, 10, 1, 1], [0, 1, 0, 1, 1, 1, 1, 1, 1, 1])       # zero ch0 & ch2
    chsel = g.nd("Mul", [pos, allow])                                # one-hot at F over channels
    # neighbour shifts of B (within grid)
    Br = g.shift(B, 0, -1)   # B[r,c+1]
    Bl = g.shift(B, 0, 1)    # B[r,c-1]
    Bd = g.shift(B, -1, 0)   # B[r+1,c]
    Bu = g.shift(B, 1, 0)    # B[r-1,c]

    def enable(a, b):
        p = g.nd("Mul", [T, a]); p = g.nd("Mul", [p, b])
        return g.nd("ReduceMax", [p], axes=[2, 3], keepdims=1)       # [1,1,1,1]
    e_ul = enable(Br, Bd)   # TL corner -> up-left
    e_ur = enable(Bl, Bd)   # TR corner -> up-right
    e_dl = enable(Br, Bu)   # BL corner -> down-left
    e_dr = enable(Bl, Bu)   # BR corner -> down-right

    def prop(seed, dr, dc):
        beam = seed
        for s in (1, 2, 4, 8):
            sh = g.shift(beam, dr * s, dc * s)
            beam = g.nd("Max", [beam, sh])
        return beam
    beams = []
    for (dr, dc), en in (((-1, -1), e_ul), ((-1, 1), e_ur), ((1, -1), e_dl), ((1, 1), e_dr)):
        bm = prop(B, dr, dc)
        beams.append(g.nd("Mul", [bm, en]))
    tot = B
    for bm in beams:
        tot = g.nd("Max", [tot, bm])
    tot = g.nd("Mul", [tot, grid])                                   # crop to grid
    # output: channel F = tot ; channel 0 = grid*(1-tot)
    one = g.f([1, 1, 1, 1], [1.0])
    notbeam = g.nd("Sub", [one, tot])
    bg = g.nd("Mul", [grid, notbeam])
    bgvec = g.f([1, 10, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    bgpart = g.nd("Mul", [bg, bgvec])                               # [1,10,30,30]
    fpart = g.nd("Mul", [tot, chsel])                              # broadcast -> [1,10,30,30]
    g.nd("Add", [bgpart, fpart], "output")
    return _model(g)


# =========================================================================== #
# TASK 104 -- 2x2 block (3 fill + 1 marker), two 4x4 blocks on its diagonal   #
# =========================================================================== #
def _t104_rule(i):
    h, w = i.shape
    ys, xs = np.where(i != 0)
    if len(ys) == 0:
        return None
    r0, c0 = ys.min(), xs.min()
    if ys.max() - r0 != 1 or xs.max() - c0 != 1:
        return None
    block = i[r0:r0 + 2, c0:c0 + 2]
    vals, cnts = np.unique(block[block != 0], return_counts=True)
    if not (len(vals) == 2 and set(cnts.tolist()) == {1, 3}):
        return None
    marker = vals[cnts.argmin()]; fill = vals[cnts.argmax()]
    mr, mc = np.where(block == marker)
    mr, mc = int(mr[0]), int(mc[0])
    main = (mr == mc)
    out = np.zeros((3 * h, 3 * w), dtype=i.dtype)
    if main:
        out[r0:r0 + 4, c0:c0 + 4] = fill
        out[r0 + 4:r0 + 8, c0 + 4:c0 + 8] = fill
    else:
        out[r0:r0 + 4, c0 + 4:c0 + 8] = fill
        out[r0 + 4:r0 + 8, c0:c0 + 4] = fill
    return out


def _t104_detect(prs):
    for i, o in prs:
        if i.shape != (3, 3):
            return False
        r = _t104_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
    return True


def _t104_build():
    g = _G()
    inp = "input"
    Ri = g.f([1, 1, 30, 30], np.repeat(np.arange(30)[:, None], 30, 1))
    Ci = g.f([1, 1, 30, 30], np.repeat(np.arange(30)[None, :], 30, 0))
    BIG = g.f([1, 1, 1, 1], [1000.0])
    one = g.f([1, 1, 1, 1], [1.0])
    col = _slc(g, inp, 1, 10, 1)
    mask = g.nd("ReduceSum", [col], axes=[1], keepdims=1)            # [1,1,30,30] colored
    # dr = min row, dc = min col of colored cells
    rowhas = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)         # [1,1,30,1]
    colhas = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)         # [1,1,1,30]
    Rcol = _slc(g, Ri, 0, 1, 3)                                      # [1,1,30,1] row idx
    Crow = _slc(g, Ci, 0, 1, 2)                                      # [1,1,1,30] col idx
    dr = g.nd("ReduceMin", [g.nd("Add", [Rcol, g.nd("Mul", [g.nd("Sub", [one, rowhas]), BIG])])],
              axes=[2], keepdims=1)                                  # [1,1,1,1]
    dc = g.nd("ReduceMin", [g.nd("Add", [Crow, g.nd("Mul", [g.nd("Sub", [one, colhas]), BIG])])],
              axes=[3], keepdims=1)
    # fill / marker channel selectors
    cnt = g.nd("ReduceSum", [inp], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    notch0 = g.f([1, 10, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    g15 = g.f([1, 1, 1, 1], [1.5]); g05 = g.f([1, 1, 1, 1], [0.5])
    fillsel = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [cnt, g15])], to=F), notch0])  # F
    markersel = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [cnt, g05])], to=F),
                             g.nd("Sub", [notch0, fillsel])])        # marker (cnt==1, ch!=0)
    markmap = g.nd("ReduceSum", [g.nd("Mul", [inp, markersel])], axes=[1], keepdims=1)   # [1,1,30,30]
    mr = g.nd("ReduceSum", [g.nd("Mul", [markmap, Ri])], axes=[2, 3], keepdims=1)
    mc = g.nd("ReduceSum", [g.nd("Mul", [markmap, Ci])], axes=[2, 3], keepdims=1)
    # anti = | (mr-dr) - (mc-dc) |
    off = g.nd("Sub", [g.nd("Sub", [mr, dr]), g.nd("Sub", [mc, dc])])
    anti = g.nd("Abs", [off])                                        # 0 main, 1 anti  [1,1,1,1]
    # a = R-dr , b = C-dc
    a = g.nd("Sub", [Ri, dr]); b = g.nd("Sub", [Ci, dc])
    eight = g.f([1, 1, 1, 1], [8.0]); g35 = g.f([1, 1, 1, 1], [3.5]); zero = g.f([1, 1, 1, 1], [0.0])
    va = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [a, g.f([1, 1, 1, 1], [-0.5])])], to=F),
                      g.nd("Cast", [g.nd("Less", [a, eight])], to=F)])
    vb = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [b, g.f([1, 1, 1, 1], [-0.5])])], to=F),
                      g.nd("Cast", [g.nd("Less", [b, eight])], to=F)])
    qa = g.nd("Cast", [g.nd("Greater", [a, g35])], to=F)
    qb = g.nd("Cast", [g.nd("Greater", [b, g35])], to=F)
    d = g.nd("Abs", [g.nd("Sub", [qa, qb])])                         # |qa-qb|
    # chosen = (1-anti)*(1-d) + anti*d
    chosen = g.nd("Add", [g.nd("Mul", [g.nd("Sub", [one, anti]), g.nd("Sub", [one, d])]),
                          g.nd("Mul", [anti, d])])
    fillmask = g.nd("Mul", [g.nd("Mul", [va, vb]), chosen])          # [1,1,30,30]
    # restrict to 9x9 output region
    reg = np.zeros((30, 30), np.float32); reg[:9, :9] = 1
    regc = g.f([1, 1, 30, 30], reg)
    fillmask = g.nd("Mul", [fillmask, regc])
    bg = g.nd("Mul", [regc, g.nd("Sub", [one, fillmask])])
    bgvec = g.f([1, 10, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    out = g.nd("Add", [g.nd("Mul", [bg, bgvec]), g.nd("Mul", [fillmask, fillsel])])
    g.nd("Identity", [out], "output")
    return _model(g)


# =========================================================================== #
# TASK 123 -- nested-L colours, output 10x10 = S[max(R,C) mod m]              #
# =========================================================================== #
def _t123_rule(i):
    if i.shape != (5, 5):
        return None
    S = np.array([i[k, k] for k in range(5)])
    # verify nested-L
    for r in range(5):
        for c in range(5):
            if i[r, c] != S[max(r, c)]:
                return None
    m = int((S != 0).sum())
    if m == 0 or not (S[:m] != 0).all():
        return None
    out = np.zeros((10, 10), dtype=i.dtype)
    for R in range(10):
        for C in range(10):
            out[R, C] = S[max(R, C) % m]
    return out


def _t123_detect(prs):
    for i, o in prs:
        r = _t123_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
    return True


def _t123_build():
    g = _G()
    inp = "input"
    Xs = g.nd("Slice", [inp, g.i64([0, 0]), g.i64([5, 5]), g.i64([2, 3])])   # [1,10,5,5]
    eye = g.f([1, 1, 5, 5], np.eye(5))
    diag = g.nd("Mul", [Xs, eye])
    D = g.nd("ReduceSum", [diag], axes=[2], keepdims=0)              # [1,10,5]
    D0 = g.nd("Slice", [D, g.i64([0]), g.i64([1]), g.i64([1])])     # [1,1,5]
    sumD0 = g.nd("ReduceSum", [D0], axes=[2], keepdims=1)           # [1,1,1]
    five = g.f([1, 1, 1], [5.0])
    m = g.nd("Sub", [five, sumD0])                                  # [1,1,1]
    m4 = g.nd("Reshape", [m, g.i64([1, 1, 1, 1])])
    M = g.f([1, 1, 10, 10], np.maximum.outer(np.arange(10), np.arange(10)))
    idx = g.nd("Mod", [M, m4], fmod=1)                             # [1,1,10,10]
    idx5 = g.nd("Reshape", [idx, g.i64([1, 1, 10, 10, 1])])
    kvec = g.f([1, 1, 1, 1, 5], [0, 1, 2, 3, 4])
    diff = g.nd("Sub", [idx5, kvec])
    half = g.f([1, 1, 1, 1, 1], [0.5])
    oh5 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff]), half])], to=F)   # [1,1,10,10,5]
    oh2d = g.nd("Reshape", [oh5, g.i64([100, 5])])
    D2 = g.nd("Reshape", [D, g.i64([10, 5])])
    Dt = g.nd("Transpose", [D2], perm=[1, 0])                       # [5,10]
    prod = g.nd("MatMul", [oh2d, Dt])                              # [100,10]
    pr3 = g.nd("Reshape", [prod, g.i64([10, 10, 10])])             # R,C,ch
    prt = g.nd("Transpose", [pr3], perm=[2, 0, 1])                 # ch,R,C
    pr4 = g.nd("Reshape", [prt, g.i64([1, 10, 10, 10])])
    g.nd("Pad", [pr4], "output", mode="constant", pads=[0, 0, 0, 0, 0, 0, 20, 20], value=0.0)
    return _model(g)


# =========================================================================== #
# TASK 9 -- connect same-colour markers along cell-grid rows/cols (span fill) #
# =========================================================================== #
def _t9_wall(i):
    h, w = i.shape
    for r in range(h):
        if (i[r] == i[r, 0]).all() and i[r, 0] != 0:
            return int(i[r, 0])
    for c in range(w):
        if (i[:, c] == i[0, c]).all() and i[0, c] != 0:
            return int(i[0, c])
    return None


def _t9_rule(i):
    h, w = i.shape
    W = _t9_wall(i)
    if W is None:
        return None
    out = i.copy()
    wall = (i == W)
    markers = sorted(set(np.unique(i).tolist()) - {0, W})
    if not markers:
        return None
    for k in markers:
        pos = (i == k)
        for r in range(h):
            cols = np.where(pos[r])[0]
            if len(cols) >= 2:
                for c in range(cols.min(), cols.max() + 1):
                    if not wall[r, c]:
                        out[r, c] = k
        for c in range(w):
            rows = np.where(pos[:, c])[0]
            if len(rows) >= 2:
                for r in range(rows.min(), rows.max() + 1):
                    if not wall[r, c]:
                        out[r, c] = k
    return out


def _t9_detect(prs):
    for i, o in prs:
        r = _t9_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
    return True


def _t9_build():
    g = _G()
    inp = "input"
    Ci = g.f([1, 1, 30, 30], np.repeat(np.arange(30)[None, :], 30, 0))
    Ri = g.f([1, 1, 30, 30], np.repeat(np.arange(30)[:, None], 30, 1))
    BIG = g.f([1, 1, 1, 1], [1000.0]); half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    notch0 = g.f([1, 10, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    bgvec = g.f([1, 10, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    grid = g.nd("ReduceSum", [inp], axes=[1], keepdims=1)            # [1,1,30,30]
    cnt = g.nd("ReduceSum", [inp], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    cntC = g.nd("Mul", [cnt, notch0])
    maxc = g.nd("ReduceMax", [cntC], axes=[1], keepdims=1)           # [1,1,1,1]
    Wsel = g.nd("Cast", [g.nd("Greater", [cntC, g.nd("Sub", [maxc, half])])], to=F)  # [1,10,1,1]
    wallmask = g.nd("ReduceSum", [g.nd("Mul", [inp, Wsel])], axes=[1], keepdims=1)   # [1,1,30,30]
    omX = g.nd("Sub", [one, inp])
    XC = g.nd("Mul", [inp, Ci]); XR = g.nd("Mul", [inp, Ri])
    omB = g.nd("Mul", [omX, BIG])
    Rr = g.nd("ReduceMax", [g.nd("Sub", [XC, omB])], axes=[3], keepdims=1)   # [1,10,30,1]
    Lr = g.nd("ReduceMin", [g.nd("Add", [XC, omB])], axes=[3], keepdims=1)
    Hm = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [Ci, g.nd("Sub", [Lr, half])])], to=F),
                      g.nd("Cast", [g.nd("Less", [Ci, g.nd("Add", [Rr, half])])], to=F)])
    Br = g.nd("ReduceMax", [g.nd("Sub", [XR, omB])], axes=[2], keepdims=1)   # [1,10,1,30]
    Tr = g.nd("ReduceMin", [g.nd("Add", [XR, omB])], axes=[2], keepdims=1)
    Vm = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [Ri, g.nd("Sub", [Tr, half])])], to=F),
                      g.nd("Cast", [g.nd("Less", [Ri, g.nd("Add", [Br, half])])], to=F)])
    fill = g.nd("Max", [Hm, Vm])                                    # [1,10,30,30]
    markersel = g.nd("Sub", [notch0, Wsel])
    notwall = g.nd("Sub", [one, wallmask])
    mf = g.nd("Mul", [g.nd("Mul", [g.nd("Mul", [fill, notwall]), grid]), markersel])  # [1,10,30,30]
    anyf = g.nd("ReduceSum", [mf], axes=[1], keepdims=1)
    wall_out = g.nd("Mul", [wallmask, Wsel])                        # [1,10,30,30]
    bg = g.nd("Mul", [g.nd("Mul", [grid, notwall]), g.nd("Sub", [one, anyf])])
    bg_all = g.nd("Mul", [bg, bgvec])
    g.nd("Add", [g.nd("Add", [mf, wall_out]), bg_all], "output")
    return _model(g)


# =========================================================================== #
# TASK 148 -- twin walls (col 2); marker beams + mirrored full beams          #
# =========================================================================== #
def _t148_rule(i):
    h, w = i.shape
    o = i.copy()
    wallcols = [c for c in range(w) if (i[:, c] == 2).any()]
    if len(wallcols) != 2:
        return None
    walls = {}
    for c in wallcols:
        rows = np.where(i[:, c] == 2)[0]
        if rows.max() - rows.min() + 1 != len(rows):
            return None
        walls[c] = (rows.min(), rows.max())
    mys, mxs = np.where(i == 8)
    if len(mys) == 0:
        return None
    # one marker per row
    if len(set(mys.tolist())) != len(mys):
        return None
    src = None
    for c, (t, b) in walls.items():
        if all(t <= r <= b for r in mys):
            src = c
    if src is None:
        return None
    tgt = [c for c in wallcols if c != src][0]
    st, sb = walls[src]; tt, tb = walls[tgt]
    if sb - st != tb - tt:
        return None
    for r, c in zip(mys, mxs):
        if src == 0:
            for cc in range(1, c):
                o[r, cc] = 8
        else:
            for cc in range(c + 1, src):
                o[r, cc] = 8
        o[r, c] = 4
    for off in sorted(set((mys - st).tolist())):
        tr = tt + off
        if tgt == 0:
            for cc in range(1, w):
                o[tr, cc] = 8
        else:
            for cc in range(0, tgt):
                o[tr, cc] = 8
    return o


def _t148_detect(prs):
    for i, o in prs:
        # only colors 0,2,8 allowed in input
        if set(np.unique(i).tolist()) - {0, 2, 8}:
            return False
        r = _t148_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
    return True


def _t148_build():
    g = _G()
    inp = "input"
    Ri = g.f([1, 1, 30, 30], np.repeat(np.arange(30)[:, None], 30, 1))
    Ci = g.f([1, 1, 30, 30], np.repeat(np.arange(30)[None, :], 30, 0))
    Cidx1 = g.f([1, 1, 1, 30], np.arange(30))
    BIG = g.f([1, 1, 1, 1], [1000.0]); half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = _slc(g, inp, 2, 3, 1)
    eight = _slc(g, inp, 8, 9, 1)
    grid = g.nd("ReduceSum", [inp], axes=[1], keepdims=1)
    omtwo = g.nd("Sub", [one, two]); ome = g.nd("Sub", [one, eight])
    Rtwo = g.nd("Mul", [Ri, two]); Reight = g.nd("Mul", [Ri, eight])
    wcol = g.nd("ReduceMax", [two], axes=[2], keepdims=1)            # [1,1,1,30]
    top2 = g.nd("ReduceMin", [g.nd("Add", [Rtwo, g.nd("Mul", [omtwo, BIG])])], axes=[2], keepdims=1)
    bot2 = g.nd("ReduceMax", [g.nd("Sub", [Rtwo, g.nd("Mul", [omtwo, BIG])])], axes=[2], keepdims=1)
    mmin = g.nd("ReduceMin", [g.nd("Add", [Reight, g.nd("Mul", [ome, BIG])])], axes=[2, 3], keepdims=1)
    mmax = g.nd("ReduceMax", [g.nd("Sub", [Reight, g.nd("Mul", [ome, BIG])])], axes=[2, 3], keepdims=1)
    srcsel = g.nd("Mul", [g.nd("Mul", [wcol, g.nd("Cast", [g.nd("Less", [top2, g.nd("Add", [mmin, half])])], to=F)]),
                          g.nd("Cast", [g.nd("Greater", [bot2, g.nd("Sub", [mmax, half])])], to=F)])  # [1,1,1,30]
    tgtsel = g.nd("Sub", [wcol, srcsel])
    src_col = g.nd("ReduceSum", [g.nd("Mul", [srcsel, Cidx1])], axes=[3], keepdims=1)
    src_top = g.nd("ReduceSum", [g.nd("Mul", [srcsel, top2])], axes=[3], keepdims=1)
    tgt_col = g.nd("ReduceSum", [g.nd("Mul", [tgtsel, Cidx1])], axes=[3], keepdims=1)
    tgt_top = g.nd("ReduceSum", [g.nd("Mul", [tgtsel, top2])], axes=[3], keepdims=1)
    delta = g.nd("Sub", [tgt_top, src_top])                         # [1,1,1,1]
    mc_row = g.nd("ReduceSum", [g.nd("Mul", [eight, Ci])], axes=[3], keepdims=1)   # [1,1,30,1]
    has_row = g.nd("ReduceMax", [eight], axes=[3], keepdims=1)      # [1,1,30,1]
    lo = g.nd("Min", [src_col, mc_row]); hi = g.nd("Max", [src_col, mc_row])
    SB = g.nd("Mul", [g.nd("Mul", [has_row, g.nd("Cast", [g.nd("Greater", [Ci, g.nd("Add", [lo, half])])], to=F)]),
                      g.nd("Cast", [g.nd("Less", [Ci, g.nd("Sub", [hi, half])])], to=F)])
    # target rows via shift matrix S[i,j]=1 iff i-j==delta
    Ig = g.f([30, 30], np.repeat(np.arange(30)[:, None], 30, 1))
    Jg = g.f([30, 30], np.repeat(np.arange(30)[None, :], 30, 0))
    d2 = g.nd("Reshape", [delta, g.i64([1, 1])])
    Smat = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [g.nd("Sub", [Ig, Jg]), d2])]),
                                       g.f([1, 1], [0.5])])], to=F)
    hr2 = g.nd("Reshape", [has_row, g.i64([30, 1])])
    tgtrow = g.nd("Reshape", [g.nd("MatMul", [Smat, hr2]), g.i64([1, 1, 30, 1])])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [Ci, grid])], axes=[2, 3], keepdims=1)
    gridcol = g.nd("Cast", [g.nd("Less", [Ci, g.nd("Add", [maxcol, half])])], to=F)
    notgt = g.nd("Cast", [g.nd("Greater", [g.nd("Abs", [g.nd("Sub", [Ci, tgt_col])]), half])], to=F)
    TB = g.nd("Mul", [g.nd("Mul", [tgtrow, gridcol]), notgt])
    ch8 = g.nd("Add", [SB, TB]); ch4 = eight; ch2 = two
    s3 = g.nd("Add", [g.nd("Add", [ch8, ch4]), ch2])
    ch0 = g.nd("Mul", [grid, g.nd("Sub", [one, s3])])
    z = g.f([1, 1, 30, 30], np.zeros((30, 30)))
    g.nd("Concat", [ch0, z, ch2, z, ch4, z, z, z, ch8, z], "output", axis=1)
    return _model(g)


# =========================================================================== #
# TASK 333 -- connect each aligned marker to the 2x2 block with a colour line #
# =========================================================================== #
def _t333_block(i):
    for k in set(np.unique(i).tolist()) - {0}:
        ys, xs = np.where(i == k)
        if len(ys) == 4 and ys.max() - ys.min() == 1 and xs.max() - xs.min() == 1:
            if (i[ys.min():ys.min() + 2, xs.min():xs.min() + 2] == k).all():
                return int(k), int(ys.min()), int(xs.min())
    return None


def _t333_rule(i):
    fb = _t333_block(i)
    if fb is None:
        return None
    B, br0, bc0 = fb
    h, w = i.shape
    o = i.copy()
    brs = {br0, br0 + 1}; bcs = {bc0, bc0 + 1}
    blockcells = {(a, b) for a in (br0, br0 + 1) for b in (bc0, bc0 + 1)}
    for r in range(h):
        for c in range(w):
            k = i[r, c]
            if k == 0 or k == B or (r, c) in blockcells:
                continue
            if r in brs:
                if c < bc0:
                    for cc in range(c + 1, bc0):
                        o[r, cc] = k
                elif c > bc0 + 1:
                    for cc in range(bc0 + 2, c):
                        o[r, cc] = k
            if c in bcs:
                if r < br0:
                    for rr in range(r + 1, br0):
                        o[rr, c] = k
                elif r > br0 + 1:
                    for rr in range(br0 + 2, r):
                        o[rr, c] = k
    return o


def _t333_detect(prs):
    for i, o in prs:
        r = _t333_rule(i)
        if r is None or r.shape != o.shape or not np.array_equal(r, o):
            return False
    return True


def _t333_build():
    g = _G()
    inp = "input"
    Ri = g.f([1, 1, 30, 30], np.repeat(np.arange(30)[:, None], 30, 1))
    Ci = g.f([1, 1, 30, 30], np.repeat(np.arange(30)[None, :], 30, 0))
    BIG = g.f([1, 1, 1, 1], [1000.0]); half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    notch0 = g.f([1, 10, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    bgvec = g.f([1, 10, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # detect 2x2 block channel
    sL = g.shift(inp, 0, -1); sU = g.shift(inp, -1, 0); sUL = g.shift(inp, -1, -1)
    prod = g.nd("Mul", [g.nd("Mul", [g.nd("Mul", [inp, sL]), sU]), sUL])
    Bsel = g.nd("Mul", [g.nd("ReduceMax", [prod], axes=[2, 3], keepdims=1), notch0])  # [1,10,1,1]
    Xb = g.nd("ReduceSum", [g.nd("Mul", [inp, Bsel])], axes=[1], keepdims=1)   # [1,1,30,30]
    markersel = g.nd("Sub", [notch0, Bsel])
    omb = g.nd("Sub", [one, Xb])
    bc0 = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Ci, Xb]), g.nd("Mul", [omb, BIG])])], axes=[2, 3], keepdims=1)
    bc1 = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [Ci, Xb]), g.nd("Mul", [omb, BIG])])], axes=[2, 3], keepdims=1)
    br0 = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Ri, Xb]), g.nd("Mul", [omb, BIG])])], axes=[2, 3], keepdims=1)
    br1 = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [Ri, Xb]), g.nd("Mul", [omb, BIG])])], axes=[2, 3], keepdims=1)
    blockrow = g.nd("ReduceMax", [Xb], axes=[3], keepdims=1)         # [1,1,30,1]
    blockcol = g.nd("ReduceMax", [Xb], axes=[2], keepdims=1)         # [1,1,1,30]
    ltBc0 = g.nd("Cast", [g.nd("Less", [Ci, bc0])], to=F)
    gtBc1 = g.nd("Cast", [g.nd("Greater", [Ci, bc1])], to=F)
    ltBr0 = g.nd("Cast", [g.nd("Less", [Ri, br0])], to=F)
    gtBr1 = g.nd("Cast", [g.nd("Greater", [Ri, br1])], to=F)
    # horizontal
    leftmask = g.nd("Mul", [inp, ltBc0])
    minLeft = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Ci, leftmask]),
                                              g.nd("Mul", [g.nd("Sub", [one, leftmask]), BIG])])], axes=[3], keepdims=1)
    leftfill = g.nd("Mul", [g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [Ci, g.nd("Sub", [minLeft, half])])], to=F), ltBc0]), blockrow])
    rightmask = g.nd("Mul", [inp, gtBc1])
    maxRight = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [Ci, rightmask]),
                                              g.nd("Mul", [g.nd("Sub", [one, rightmask]), BIG])])], axes=[3], keepdims=1)
    rightfill = g.nd("Mul", [g.nd("Mul", [g.nd("Cast", [g.nd("Less", [Ci, g.nd("Add", [maxRight, half])])], to=F), gtBc1]), blockrow])
    Hfill = g.nd("Add", [leftfill, rightfill])
    # vertical
    upmask = g.nd("Mul", [inp, ltBr0])
    minUp = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [Ri, upmask]),
                                            g.nd("Mul", [g.nd("Sub", [one, upmask]), BIG])])], axes=[2], keepdims=1)
    upfill = g.nd("Mul", [g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [Ri, g.nd("Sub", [minUp, half])])], to=F), ltBr0]), blockcol])
    downmask = g.nd("Mul", [inp, gtBr1])
    maxDown = g.nd("ReduceMax", [g.nd("Sub", [g.nd("Mul", [Ri, downmask]),
                                              g.nd("Mul", [g.nd("Sub", [one, downmask]), BIG])])], axes=[2], keepdims=1)
    downfill = g.nd("Mul", [g.nd("Mul", [g.nd("Cast", [g.nd("Less", [Ri, g.nd("Add", [maxDown, half])])], to=F), gtBr1]), blockcol])
    Vfill = g.nd("Add", [upfill, downfill])
    fill = g.nd("Add", [Hfill, Vfill])
    mf = g.nd("Mul", [fill, markersel])
    anyf = g.nd("ReduceSum", [mf], axes=[1], keepdims=1)
    base = g.nd("Max", [inp, mf])
    out = g.nd("Sub", [base, g.nd("Mul", [anyf, bgvec])])
    g.nd("Identity", [out], "output")
    return _model(g)


# =========================================================================== #
# dispatch                                                                    #
# =========================================================================== #
_SOLVERS = [
    ("t9", _t9_detect, _t9_build),
    ("t148", _t148_detect, _t148_build),
    ("t333", _t333_detect, _t333_build),
    ("t34", _t34_detect, _t34_build),
    ("t104", _t104_detect, _t104_build),
    ("t123", _t123_detect, _t123_build),
]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    for name, detect, build in _SOLVERS:
        try:
            if detect(prs):
                m = _check(build())
                out.append((name, m))
        except Exception:
            pass
    return out
