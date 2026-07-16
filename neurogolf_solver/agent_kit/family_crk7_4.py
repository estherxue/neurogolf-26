"""crk7_4 -- crackers for the U[4::6] unsolved slice.

Sub-solvers
-----------
  beamdefl (task 345): vertical light-beams rise from every colour-2 cell in the
      bottom row; whenever a beam's column contains a colour-5 obstacle the beam
      steps one column to the RIGHT at (and above) the obstacle row, leaving a
      diagonal connector cell one row below.  Realised as a monotone cellular
      automaton on a single-channel 0/1 mask (Pad/Slice shifts + Max/Mul), which
      reaches its fixpoint in <= 10 sweeps for the data and is unrolled 14 times.

Every sub-solver mirrors the ONNX numerics in numpy and only emits a candidate
after reproducing EVERY available train+test+arc-gen pair, so wrong hypotheses
are dropped before scoring.
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


def _pairs(ex, splits=("train", "test", "arc-gen")):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


# =========================================================================== #
# sub-solver: beamdefl (task 345)                                              #
# =========================================================================== #
_ITERS = 14


def _beamdefl_ref(a):
    """Numpy mirror of the ONNX beam-deflection automaton."""
    seed = (a == 2)
    obst = (a == 5)
    lit = seed.copy()
    for _ in range(_ITERS):
        new = lit.copy()
        up = np.zeros_like(lit); up[:-1, :] = lit[1:, :]
        new |= up & ~obst
        a1 = np.zeros_like(lit); a1[:-1, 1:] = lit[1:, :-1]
        o1 = np.zeros_like(obst); o1[:, 1:] = obst[:, :-1]
        new |= a1 & o1
        b1 = np.zeros_like(lit); b1[:, 1:] = lit[:, :-1]
        o2 = np.zeros_like(obst); o2[1:, 1:] = obst[:-1, :-1]
        new |= b1 & o2
        lit = new
    out = a.copy()
    out[lit & ~obst] = 2
    return out


def _shift(g, x, dr, dc):
    """result[r,c] = x[r-dr, c-dc] (zero fill, 30x30 window)."""
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0,
             pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
    ax = g.i64([2, 3])
    return g.nd("Slice", [p, st, en, ax])


def _build_beamdefl():
    g = _G()
    one = g.f([1, 1, 1, 1], [1.0])
    seed = g.nd("Slice", ["input", g.i64([2]), g.i64([3]), g.i64([1])])   # ch 2
    obst = g.nd("Slice", ["input", g.i64([5]), g.i64([6]), g.i64([1])])   # ch 5
    notobst = g.nd("Sub", [one, obst])                                    # 1-obst
    o1 = _shift(g, obst, 0, 1)                                            # obst[r,c-1]
    o2 = _shift(g, obst, 1, 1)                                            # obst[r-1,c-1]

    lit = seed
    for _ in range(_ITERS):
        up = _shift(g, lit, -1, 0)        # lit[r+1,c]
        a1 = _shift(g, lit, -1, 1)        # lit[r+1,c-1]
        b1 = _shift(g, lit, 0, 1)         # lit[r,c-1]
        tb = g.nd("Mul", [up, notobst])
        tc = g.nd("Mul", [a1, o1])
        td = g.nd("Mul", [b1, o2])
        lit = g.nd("Max", [lit, tb, tc, td])

    beam2 = g.nd("Mul", [lit, notobst])                                  # colour-2 cells
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)         # [1,1,30,30]
    zero = g.nd("Sub", [realmask, realmask])                              # [1,1,30,30] zeros
    ch0 = g.nd("Sub", [g.nd("Sub", [realmask, beam2]), obst])            # bg cells

    chans = [ch0, zero, beam2, zero, zero, obst, zero, zero, zero, zero]
    g.nd("Concat", chans, "output", axis=1)
    return _model(g)


def _beamdefl(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    # only colours {0,2,5} involved
    for a, b in allp:
        if set(np.unique(a)) - {0, 2, 5} or set(np.unique(b)) - {0, 2, 5}:
            return []
    # require at least one obstacle deflection somewhere (else it is trivial fill)
    if not any((a == 5).any() for a, b in det):
        return []

    def ok(plist):
        for a, b in plist:
            o = _beamdefl_ref(a)
            if o.shape != b.shape or not np.array_equal(o, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    try:
        m = _build_beamdefl()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("crk7_4_beamdefl", m)]


# =========================================================================== #
# sub-solver: beams (task 280)                                                 #
# =========================================================================== #
# Each colour-3 rectangle carries a single colour-2 marker on one edge.  A beam
# fires OUTWARD (perpendicular to that edge, the side whose neighbour is
# background); the beam runs to the real-grid edge, its centre line coloured 2
# and (thickness-1) flank lines on each side coloured 3, where `thickness` is the
# rectangle's extent along the beam axis.

_MAXHW = 5
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _beams_ref(a):
    H, W = a.shape
    o = a.copy()

    def g(r, c):
        return a[r, c] if (0 <= r < H and 0 <= c < W) else -1

    for (r, c) in zip(*np.where(a == 2)):
        for (dr, dc) in _DIRS:
            if not (g(r + dr, c + dc) == 0 and g(r - dr, c - dc) == 3):
                continue
            t = 1; rr, cc = r - dr, c - dc
            while g(rr, cc) == 3:
                t += 1; rr -= dr; cc -= dc
            hw = t - 1
            pr, pc = (0, 1) if dr != 0 else (1, 0)
            rr, cc = r + dr, c + dc
            while 0 <= rr < H and 0 <= cc < W:
                o[rr, cc] = 2
                for k in range(1, hw + 1):
                    for sgn in (-1, 1):
                        fr, fc = rr + pr * k * sgn, cc + pc * k * sgn
                        if 0 <= fr < H and 0 <= fc < W:
                            o[fr, fc] = 3
                rr += dr; cc += dc
    return o


def _build_beams():
    g = _G()
    rm = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)               # realmask
    B0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    M2 = g.nd("Slice", ["input", g.i64([2]), g.i64([3]), g.i64([1])])
    M3 = g.nd("Slice", ["input", g.i64([3]), g.i64([4]), g.i64([1])])
    zero = g.nd("Sub", [rm, rm])

    def fill(stack, dr, dc):
        acc = g.nd("Mul", [rm, _shift(g, stack, dr, dc)])
        for s in (1, 2, 4, 8, 16):
            acc = g.nd("Mul", [rm, g.nd("Max", [acc, _shift(g, acc, s * dr, s * dc)])])
        return acc

    centerAll = None
    flank3 = None
    for (dr, dc) in _DIRS:
        pr, pc = (0, 1) if dr != 0 else (1, 0)
        nbr_out_bg = _shift(g, B0, -dr, -dc)       # B0(r+dr,c+dc)
        nbr_in_3 = _shift(g, M3, dr, dc)           # M3(r-dr,c-dc)
        seed = g.nd("Mul", [g.nd("Mul", [M2, nbr_out_bg]), nbr_in_3])
        seeds = [seed]
        cur = seed
        for k in range(1, _MAXHW + 1):
            in_k = _shift(g, M3, k * dr, k * dc)   # M3(r-k*dr,c-k*dc)
            cur = g.nd("Mul", [cur, in_k])
            seeds.append(cur)
        stack = g.nd("Concat", seeds, axis=1)      # [1,6,30,30]
        filled = fill(stack, dr, dc)
        cen = g.nd("Slice", [filled, g.i64([0]), g.i64([1]), g.i64([1])])
        centerAll = cen if centerAll is None else g.nd("Max", [centerAll, cen])
        for k in range(1, _MAXHW + 1):
            fk = g.nd("Slice", [filled, g.i64([k]), g.i64([k + 1]), g.i64([1])])
            fp = g.nd("Mul", [rm, _shift(g, fk, k * pr, k * pc)])
            fm = g.nd("Mul", [rm, _shift(g, fk, -k * pr, -k * pc)])
            term = g.nd("Max", [fp, fm])
            flank3 = term if flank3 is None else g.nd("Max", [flank3, term])

    ch2 = g.nd("Max", [M2, centerAll])
    ch3 = g.nd("Max", [M3, flank3])
    ch0 = g.nd("Sub", [g.nd("Sub", [rm, ch2]), ch3])
    chans = [ch0, zero, ch2, ch3, zero, zero, zero, zero, zero, zero]
    g.nd("Concat", chans, "output", axis=1)
    return _model(g)


def _beams(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    for a, b in allp:
        if set(np.unique(a)) - {0, 2, 3} or set(np.unique(b)) - {0, 2, 3}:
            return []
    if not any((a == 2).any() and (a == 3).any() for a, b in det):
        return []

    def ok(plist):
        for a, b in plist:
            o = _beams_ref(a)
            if o.shape != b.shape or not np.array_equal(o, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    try:
        m = _build_beams()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("crk7_4_beams", m)]


# =========================================================================== #
# dispatch                                                                     #
# =========================================================================== #
def candidates(ex):
    out = []
    for fn in (_beamdefl, _beams):
        try:
            out += fn(ex)
        except Exception:
            pass
    return out
