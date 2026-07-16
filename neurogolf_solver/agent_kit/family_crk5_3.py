"""family_crk5_3 -- two exact, generalizing solvers for hard ARC->ONNX tasks.

Covers (verified EXACT on train+test+arc-gen):

1. STAMP  (e.g. task 323): a single "marker" colour `mc` occurs (one or many
   times) and a FIXED little template of cells around every marker is painted a
   fixed colour.  The template is translation-equivariant so ONE Conv (kernel ==
   the flipped template, symmetric pads) paints them all at once, clipped to the
   real grid.  Radius up to 13 (markerstamp's family caps at 3, so it misses the
   long zig-zag stamps this one catches).

2. GRIDGAP  (e.g. task 198): the grid is ruled by solid horizontal/vertical
   separator lines of one colour `L`.  Some line cells are "gaps" (background
   instead of `L`).  Every background cell that is reachable (4-conn, through
   background only) from a gap is recoloured `fill`; every other background cell
   (sitting inside a perfectly-sealed rectangular pocket) is recoloured `other`.
   Line cells keep `L`.  Detection: a row/col is a "line" when its `L`-count
   exceeds frac*width/height; gaps = line-cell that is background; then an
   unrolled cellular-automaton flood (banned Loop -> fixed conv chain) computes
   reachability, exactly as the reference numpy below.

Both rules are inferred from the pairs and a candidate is emitted ONLY when it
reproduces EVERY available pair (train+test+arc-gen), so wrong guesses never
reach the grader.
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


# ===========================================================================
# 1. STAMP
# ===========================================================================
def _stamp_ref(a, mc, T):
    """Mirror the ONNX stamp semantics exactly (multi-marker, clipped, one-hot)."""
    h, w = a.shape
    stamp = np.zeros((CHANNELS, h, w), np.int64)
    for mr, ml in np.argwhere(a == mc):
        for (dy, dx), col in T.items():
            r, c = mr + dy, ml + dx
            if 0 <= r < h and 0 <= c < w:
                stamp[col, r, c] += 1
    cond = stamp.sum(0) > 0
    nch = (stamp > 0).sum(0)
    if (nch[cond] != 1).any():
        return None
    out = a.copy()
    if cond.any():
        ch = np.argmax(stamp, axis=0)
        out[cond] = ch[cond]
    return out


def _infer_T(prs, mc, R):
    obs = {}
    for a, b in prs:
        h, w = a.shape
        for mr, ml in np.argwhere(a == mc):
            for r in range(max(0, mr - R), min(h, mr + R + 1)):
                for c in range(max(0, ml - R), min(w, ml + R + 1)):
                    if b[r, c] != a[r, c]:
                        key = (int(r - mr), int(c - ml))
                        col = int(b[r, c])
                        if key in obs and obs[key] != col:
                            return None
                        obs[key] = col
    return obs or None


def _build_stamp_single(mc, T, col, R):
    """All template offsets share one colour `col` -> cheap single-channel Conv."""
    g = _G()
    K = 2 * R + 1
    M = g.nd("Slice", ["input", g.i64([mc]), g.i64([mc + 1]), g.i64([1])])   # [1,1,30,30]
    Wk = np.zeros((1, 1, K, K), np.float32)
    for (dy, dx) in T:
        Wk[0, 0, R - dy, R - dx] = 1.0
    wt = g.f([1, 1, K, K], Wk)
    stamp = g.nd("Conv", [M, wt], kernel_shape=[K, K], pads=[R, R, R, R])     # [1,1,30,30]
    half = g.f([1, 1, 1, 1], [0.5])
    present = g.nd("Greater", [stamp, half])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)             # [1,1,30,30]
    real = g.nd("Greater", [realmask, half])
    cond = g.nd("And", [present, real])
    oneh = np.zeros((1, CHANNELS, 1, 1), np.float32)
    if 0 <= col < CHANNELS:
        oneh[0, col, 0, 0] = 1.0
    ohc = g.f([1, CHANNELS, 1, 1], oneh)
    g.nd("Where", [cond, ohc, "input"], "output")
    return _model(g)


def _build_stamp_multi(mc, T, R):
    """Multi-colour template -> full 10-channel Conv + argmax-style Where."""
    g = _G()
    K = 2 * R + 1
    M = g.nd("Slice", ["input", g.i64([mc]), g.i64([mc + 1]), g.i64([1])])
    Wk = np.zeros((CHANNELS, 1, K, K), np.float32)
    for (dy, dx), col in T.items():
        Wk[col, 0, R - dy, R - dx] = 1.0
    wt = g.f([CHANNELS, 1, K, K], Wk)
    stamp = g.nd("Conv", [M, wt], kernel_shape=[K, K], pads=[R, R, R, R])     # [1,10,30,30]
    csum = g.nd("ReduceSum", [stamp], axes=[1], keepdims=1)
    half = g.f([1, 1, 1, 1], [0.5])
    present = g.nd("Greater", [csum, half])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    real = g.nd("Greater", [realmask, half])
    cond = g.nd("And", [present, real])
    g.nd("Where", [cond, stamp, "input"], "output")
    return _model(g)


def _stamp_candidates(prs):
    if any(a.shape != b.shape for a, b in prs):
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    incolors = set()
    for a, _ in prs:
        incolors |= set(int(v) for v in np.unique(a).tolist())
    incolors.discard(0)

    out = []
    for mc in sorted(incolors):
        # marker must occur in every example, and be sparse
        if not all((a == mc).any() for a, _ in prs):
            continue
        if any((a == mc).sum() * 2 > a.size for a, _ in prs):
            continue
        # cheap stamp gate: every changed cell must sit within radius 13 of SOME
        # marker, and changes must be modest relative to the marker count
        bad = False
        for a, b in prs:
            ch = np.argwhere(a != b)
            if ch.size == 0:
                continue
            mk = np.argwhere(a == mc)
            if mk.size == 0 or len(ch) > 60 * len(mk) + 60:
                bad = True
                break
        if bad:
            continue
        for R in range(1, 14):
            T = _infer_T(prs, mc, R)
            if not T:
                continue
            ok = True
            for a, b in prs:
                o = _stamp_ref(a, mc, T)
                if o is None or o.shape != b.shape or not np.array_equal(o, b):
                    ok = False
                    break
            if not ok:
                continue
            cols = set(T.values())
            try:
                if len(cols) == 1:
                    m = _build_stamp_single(mc, T, cols.pop(), R)
                else:
                    m = _build_stamp_multi(mc, T, R)
                onnx.checker.check_model(m, full_check=True)
            except Exception:
                break
            out.append((f"stamp_mc{mc}_r{R}", m))
            break  # smallest validating radius for this marker
    return out


# ===========================================================================
# 2. GRIDGAP
# ===========================================================================
def _gridgap_ref(a, L, fill, other, frac):
    """Reference the ONNX mirrors exactly."""
    h, w = a.shape
    rowc = (a == L).sum(1)
    colc = (a == L).sum(0)
    lr = rowc > frac * w
    lc = colc > frac * h
    linemask = lr[:, None] | lc[None, :]
    is0 = (a == 0)
    seed = linemask & is0
    # 4-conn flood through background to convergence
    reach = seed.copy()
    while True:
        nb = reach.copy()
        nb[1:, :] |= reach[:-1, :]
        nb[:-1, :] |= reach[1:, :]
        nb[:, 1:] |= reach[:, :-1]
        nb[:, :-1] |= reach[:, 1:]
        nb &= is0
        if np.array_equal(nb, reach):
            break
        reach = nb
    out = a.copy()
    out[is0 & reach] = fill
    out[is0 & ~reach] = other
    return out


def _gridgap_steps(a, L, frac):
    h, w = a.shape
    rowc = (a == L).sum(1)
    colc = (a == L).sum(0)
    linemask = (rowc > frac * w)[:, None] | (colc > frac * h)[None, :]
    is0 = (a == 0)
    reach = linemask & is0
    steps = 0
    while True:
        nb = reach.copy()
        nb[1:, :] |= reach[:-1, :]
        nb[:-1, :] |= reach[1:, :]
        nb[:, 1:] |= reach[:, :-1]
        nb[:, :-1] |= reach[:, 1:]
        nb &= is0
        steps += 1
        if np.array_equal(nb, reach):
            steps -= 1
            break
        reach = nb
    return steps


def _build_gridgap(fill, other, frac, n_steps):
    g = _G()
    bg = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])   # background mask
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    Lchan = g.nd("Sub", [realmask, bg])                        # non-bg (== line colour) mask

    # grid height / width (content top-left, padding zero)
    rowreal = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)   # [1,1,30,1]
    Wt = g.nd("ReduceMax", [rowreal], axes=[2], keepdims=1)         # [1,1,1,1] == width
    colreal = g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)   # [1,1,1,30]
    Ht = g.nd("ReduceMax", [colreal], axes=[3], keepdims=1)         # [1,1,1,1] == height

    fr = g.f([1, 1, 1, 1], [frac])
    rowsum = g.nd("ReduceSum", [Lchan], axes=[3], keepdims=1)       # [1,1,30,1]
    colsum = g.nd("ReduceSum", [Lchan], axes=[2], keepdims=1)       # [1,1,1,30]
    thr_r = g.nd("Mul", [Wt, fr])
    thr_c = g.nd("Mul", [Ht, fr])
    lr = g.nd("Cast", [g.nd("Greater", [rowsum, thr_r])], to=F)     # [1,1,30,1] float 0/1
    lc = g.nd("Cast", [g.nd("Greater", [colsum, thr_c])], to=F)     # [1,1,1,30]
    linef = g.nd("Max", [lr, lc])                                   # [1,1,30,30] OR
    seed = g.nd("Mul", [linef, bg])                                 # gaps (real bg on a line)

    # unrolled 4-neighbour+self flood, confined to background
    kp = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float32).reshape(1, 1, 3, 3)
    kprop = g.f([1, 1, 3, 3], kp)
    prev = seed
    for _ in range(n_steps):
        nb = g.nd("Conv", [prev, kprop], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        prev = g.nd("Min", [nb, bg])
    reach = prev                                                    # {0,1} reached bg
    three = g.nd("Sub", [bg, reach])                               # unreached bg

    # output = (input with channel0 removed) + fill*reach + other*three
    keep = np.ones((1, CHANNELS, 1, 1), np.float32)
    keep[0, 0, 0, 0] = 0.0
    keepc = g.f([1, CHANNELS, 1, 1], keep)
    base = g.nd("Mul", ["input", keepc])
    oh_fill = np.zeros((1, CHANNELS, 1, 1), np.float32); oh_fill[0, fill, 0, 0] = 1.0
    oh_oth = np.zeros((1, CHANNELS, 1, 1), np.float32); oh_oth[0, other, 0, 0] = 1.0
    add4 = g.nd("Mul", [reach, g.f([1, CHANNELS, 1, 1], oh_fill)])
    add3 = g.nd("Mul", [three, g.f([1, CHANNELS, 1, 1], oh_oth)])
    s1 = g.nd("Add", [base, add4])
    g.nd("Add", [s1, add3], "output")
    return _model(g)


def _gridgap_candidates(prs):
    if any(a.shape != b.shape for a, b in prs):
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    # every input must be exactly {0, L} for a single (per-example) line colour L
    Ls = []
    for a, _ in prs:
        nz = sorted(set(np.unique(a).tolist()) - {0})
        if len(nz) != 1 or nz[0] == 0:
            return []
        Ls.append(nz[0])

    # candidate output colours: the new colours appearing in outputs (not 0, not L)
    newcols = set()
    for (a, b), L in zip(prs, Ls):
        newcols |= set(int(v) for v in np.unique(b).tolist())
        newcols -= {0, L}
    newcols = sorted(newcols)
    if not (1 < len(newcols) <= 4):
        return []

    out = []
    for frac in (0.4, 0.34, 0.5):
        found = None
        cand_pairs = [(f, o) for f in newcols for o in newcols if f != o]
        for fill, other in cand_pairs:
            if all(np.array_equal(_gridgap_ref(a, L, fill, other, frac), b)
                   for (a, b), L in zip(prs, Ls)):
                found = (fill, other)
                break
        if found is None:
            continue
        fill, other = found
        n = max(_gridgap_steps(a, L, frac) for (a, _), L in zip(prs, Ls))
        n_steps = max(14, n + 5)
        try:
            m = _build_gridgap(fill, other, frac, n_steps)
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            continue
        out.append((f"gridgap_f{fill}_o{other}", m))
        break
    return out


# ===========================================================================
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    try:
        out += _stamp_candidates(prs)
    except Exception:
        pass
    try:
        out += _gridgap_candidates(prs)
    except Exception:
        pass
    return out
