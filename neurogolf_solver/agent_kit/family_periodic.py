"""Periodic pattern-completion family (fixed input size only).

The task: the input grid carries a spatially periodic pattern, but some cells are
masked / replaced by a "hole" color. The output is the same-size grid made FULLY
periodic -- every hole filled with the value its period demands.

Realizability (the padding gotcha): a periodic tiling repeats at period (p, q) ==
the *content* period, but the one-hot tensor is zero-padded to 30x30. A plain Tile
over the full 30x30 tensor would repeat at period 30, not p/q. So we only emit when
the grid size is CONSTANT across every available split (train+test+arc-gen) -- a
static graph cannot match a data-dependent size/period -- and we assemble the result
ORIGIN-ANCHORED: slice the [0:Hp,0:Wp] region, residue-fold it to a clean (p,q)
period block, optionally repair the block via a diagonal-symmetry roll (for diagonal
stripe patterns whose block is itself diagonal), tile the block back up, crop to the
grid and pad to 30x30 -- everything kept top-left.

Period-block reconstruction (origin-safe, hole-robust):
  * Slice input[:, :, 0:Hp, 0:Wp]  (Hp,Wp = grid rounded up to a multiple of p,q;
    the extra rows/cols are padding zeros, harmless under Max).
  * Reshape to [1,10, Hp/p, p, Wp/q, q] and ReduceMax over the two block axes ->
    base[1,10,p,q]: base[r,c] = OR of every grid cell with (i%p,j%q)==(r,c). All such
    cells share the same period value, so this recovers it wherever ANY copy is clean.
  * Optionally Mul by a per-channel mask to zero the hole color (so holes contribute
    nothing to the Max). Only used when the hole color never appears in any output.
  * Optional diagonal roll-Max (modes 'anti'/'main', requires p==q): some periodic
    patterns are diagonal stripes -- base[r,c] depends only on (r+c)%n (or (r-c)%n).
    A residue that is a hole everywhere is then filled by rotating the block along its
    (anti)diagonal: full = Max_s R^s(base), R the wrap-around (anti)diagonal shift.
  * Tile -> [1,10, ceil(H/p)*p, ceil(W/q)*q], Slice to [0:H,0:W], Pad to 30x30.

Detection is fully verified in numpy against EVERY train+test+arc-gen pair before a
model is emitted, so wrong guesses are never proposed (the harness is the real grader
and re-checks anyway).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# numpy reference reconstruction (used for detection / verification)
# --------------------------------------------------------------------------- #
def _onehot(a):
    H, W = a.shape
    o = np.zeros((CHANNELS, H, W), dtype=np.float32)
    for c in range(CHANNELS):
        o[c] = (a == c)
    return o


def _np_reconstruct(a, p, q, hc, mode, H, W):
    """Return the reconstructed raw grid, or None if the result is not a clean
    one-hot (i.e. this (p,q,hc,mode) does not explain the pair)."""
    o = _onehot(a)
    Hp = ((H + p - 1) // p) * p
    Wp = ((W + q - 1) // q) * q
    op = np.zeros((CHANNELS, Hp, Wp), dtype=np.float32)
    op[:, :H, :W] = o
    base = op.reshape(CHANNELS, Hp // p, p, Wp // q, q).max(axis=(1, 3))  # [10,p,q]
    if 0 <= hc < CHANNELS:
        base = base.copy()
        base[hc] = 0.0

    def roll_anti(b):
        b2 = np.concatenate([b[:, p - 1:p, :], b[:, 0:p - 1, :]], axis=1)
        return np.concatenate([b2[:, :, 1:q], b2[:, :, 0:1]], axis=2)

    def roll_main(b):
        b2 = np.concatenate([b[:, p - 1:p, :], b[:, 0:p - 1, :]], axis=1)
        return np.concatenate([b2[:, :, q - 1:q], b2[:, :, 0:q - 1]], axis=2)

    if mode == "plain":
        full = base
    elif mode in ("anti", "main") and p == q:
        roll = roll_anti if mode == "anti" else roll_main
        full = base.copy()
        cur = base.copy()
        for _ in range(p - 1):
            cur = roll(cur)
            full = np.maximum(full, cur)
    else:
        return None

    rh = (H + p - 1) // p
    rw = (W + q - 1) // q
    tiled = np.tile(full, (1, rh, rw))[:, :H, :W]
    if (tiled.sum(0) != 1).any():       # must be exactly one hot channel per cell
        return None
    return tiled.argmax(0)


# --------------------------------------------------------------------------- #
# detection helpers
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            out.append((np.array(e["input"]), np.array(e["output"])))
    return out


def _min_period(b):
    H, W = b.shape
    p = 1
    while p < H and not all(np.array_equal(b[i], b[i % p]) for i in range(H)):
        p += 1
    q = 1
    while q < W and not all(np.array_equal(b[:, j], b[:, j % q]) for j in range(W)):
        q += 1
    return p, q


def _candidate_periods(prs, H, W):
    cps = set()
    pers = set(_min_period(b) for _, b in prs)
    if len(pers) == 1:
        pp = pers.pop()
        if pp != (H, W):
            cps.add(pp)
    for pp in range(1, 7):
        for qq in range(1, 7):
            if pp <= H and qq <= W and (pp, qq) != (H, W):
                cps.add((pp, qq))
    return cps


# --------------------------------------------------------------------------- #
# ONNX graph construction
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init(self, dtype, dims, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, dtype, list(dims), list(vals)))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _slice(g, src, starts, ends, axes):
    s = g.init(INT64, [len(axes)], starts)
    e = g.init(INT64, [len(axes)], ends)
    a = g.init(INT64, [len(axes)], axes)
    return g.node("Slice", [src, s, e, a])


def _build(p, q, hc, mode, H, W):
    Hp = ((H + p - 1) // p) * p
    Wp = ((W + q - 1) // q) * q
    g = _G()

    # 1) origin-anchored region [0:Hp, 0:Wp]
    region = _slice(g, "input", [0, 0], [Hp, Wp], [2, 3])

    # 2) residue fold -> period block [1,10,p,q]
    shp = g.init(INT64, [6], [1, CHANNELS, Hp // p, p, Wp // q, q])
    resh = g.node("Reshape", [region, shp])
    base = g.node("ReduceMax", [resh], axes=[2, 4], keepdims=0)

    # 3) zero the hole channel (cheap 10-param broadcast multiply)
    if 0 <= hc < CHANNELS:
        mvals = [1.0] * CHANNELS
        mvals[hc] = 0.0
        mask = g.init(DATA_TYPE, [1, CHANNELS, 1, 1], mvals)
        base = g.node("Mul", [base, mask])

    # 4) optional diagonal roll-Max repair
    def roll(src, main):
        # rows: shift down by 1 with wrap  (concat last row + first p-1 rows)
        r_last = _slice(g, src, [p - 1], [p], [2])
        r_head = _slice(g, src, [0], [p - 1], [2])
        rowsh = g.node("Concat", [r_last, r_head], axis=2)
        if not main:   # cols: shift left by 1 with wrap
            c_tail = _slice(g, rowsh, [1], [q], [3])
            c_head = _slice(g, rowsh, [0], [1], [3])
            return g.node("Concat", [c_tail, c_head], axis=3)
        # main diagonal: cols shift right by 1 with wrap
        c_last = _slice(g, rowsh, [q - 1], [q], [3])
        c_head = _slice(g, rowsh, [0], [q - 1], [3])
        return g.node("Concat", [c_last, c_head], axis=3)

    if mode in ("anti", "main") and p == q and p >= 2:
        full = base
        cur = base
        for _ in range(p - 1):
            cur = roll(cur, mode == "main")
            full = g.node("Max", [full, cur])
    else:
        full = base

    # 5) tile, crop to grid, pad to 30x30 (all anchored top-left)
    rh = (H + p - 1) // p
    rw = (W + q - 1) // q
    reps = g.init(INT64, [4], [1, 1, rh, rw])
    tiled = g.node("Tile", [full, reps])
    cropped = _slice(g, tiled, [0, 0], [H, W], [2, 3])
    g.node("Pad", [cropped], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])

    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def candidates(ex):
    det = _pairs(ex, ("train", "test"))
    if not det:
        return []
    # constant grid size across every split (static-graph requirement)
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not all(a.shape == b.shape for a, b in allp):
        return []
    shapes = set(a.shape for a, _ in allp) | set(b.shape for _, b in allp)
    if len(shapes) != 1:
        return []
    H, W = shapes.pop()
    if H == 0 or W == 0 or H > HEIGHT or W > WIDTH:
        return []

    # hole-color candidates: colors that never appear in any output (safe to drop),
    # plus 0 (common background hole) and -1 (no mask).
    outcolors = set()
    for _, b in det:
        outcolors |= set(int(x) for x in np.unique(b))
    hcs = [-1, 0] + [c for c in range(CHANNELS) if c not in outcolors and c != 0]

    out = []
    for (p, q) in sorted(_candidate_periods(det, H, W)):
        for hc in hcs:
            for mode in ("plain", "anti", "main"):
                if mode in ("anti", "main") and p != q:
                    continue
                # fast check on train+test
                if not all(
                    (rec := _np_reconstruct(a, p, q, hc, mode, H, W)) is not None
                    and np.array_equal(rec, b)
                    for a, b in det
                ):
                    continue
                # full verification incl. arc-gen
                if not all(
                    (rec := _np_reconstruct(a, p, q, hc, mode, H, W)) is not None
                    and np.array_equal(rec, b)
                    for a, b in allp
                ):
                    continue
                try:
                    model = _build(p, q, hc, mode, H, W)
                except Exception:
                    continue
                tag = f"periodic_{p}x{q}_{mode}_h{hc}"
                out.append((tag, model))
                return out          # one exact solution suffices
    return out
