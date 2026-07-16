"""SELF-OVERLAY WITH SHIFT / RECOLOR (origin-anchored, opset 10).

The output is the input OR-ed with one or more *translated* (and optionally
*recoloured*) copies of itself -- a "shadow" / "duplicate" offset by constant
displacements:

    output = combine( input, recolor(shift(input, dy_1, dx_1)), ... )

i.e. a SUPERPOSITION of a small fixed set of shift "taps".  ``combine`` is a
conflict-free OR: a real cell is background only if neither the original nor any
shifted copy claims a colour there; if several copies claim the SAME colour they
agree (still one-hot); a genuine colour CLASH is rejected by detection (it could
never be exact under the grader's per-channel ``output>0`` test).

Realisation (all opset-10, origin-anchored):
  * each ``shift`` is ``Pad`` (attribute pads) + ``Slice`` -> any constant
    offset, content stays anchored at the top-left for grids of ANY size;
  * an optional shared recolour is a single 1x1 ``Conv``;
  * the copies are summed (``Add``); the non-background channels are MASKED by the
    real-cell mask ``R = ReduceSum(input, channel)`` so a copy pushed into the
    zero-padded region is dropped (mirrors the variable-size numpy semantics);
  * the background channel is rebuilt as ``R - clip(sum_nonbg, 0, 1)`` so every
    real, unclaimed cell is exactly channel-0 and padding stays all-zero.

GENERALISATION GUARD (anti-overfit).  A fixed-offset duplication is size
independent, but a propagation/tiling lattice needs ever more taps as the grid
grows.  So a multi-tap rule is emitted only when EITHER the input size is
CONSTANT across every split (size-dependent then provably generalises -- the
held-out grids are the same size) OR the tap set is small (<= 4, a true geometric
shadow/duplicate).  Detection mirrors the ONNX semantics exactly and validates
EVERY available train+test+arc-gen pair before emitting, so wrong hypotheses are
dropped before the grader sees them.
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

_BRUTE = 6            # always-tried brute shift radius
_MAXOFF = 14          # max |offset| for data-derived taps
_ANCHORS = 4          # source anchors for data-derived taps
_MAXCANDS = 240       # cap on candidate taps
_MAXTAPS = 8          # cap on selected taps
_MAXTAPS_VARY = 4     # tap cap for VARYING-size tasks (anti-overfit)
_SAMPLE = 24          # arc-gen pairs used during tap search


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

    def cf(self, dims, vals):
        n = self.nm("cf")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("ci")
        vals = list(vals)
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def node(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def _shift(g, x, dy, dx):
    """translate content by (dy, dx) with zero fill (out[r,c] = x[r-dy, c-dx])."""
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    padded = g.node("Pad", [x], mode="constant", value=0.0,
                    pads=[0, 0, h0, w0, 0, 0, h1, w1])
    return g.node("Slice", [padded, g.i64([h1, w1]),
                            g.i64([h1 + HEIGHT, w1 + WIDTH]), g.i64([2, 3])])


def build_super(taps, recolor):
    g = _G()
    # sum the shifted copies first (Conv is linear, so a single recolour of the
    # sum == recolouring each copy then summing -> one cheap conv for any #taps)
    shifts = [_shift(g, "input", dy, dx) for dy, dx in taps]
    accs = shifts[0]
    for s in shifts[1:]:
        accs = g.node("Add", [accs, s])

    if all(recolor.get(i, i) == i for i in range(CHANNELS)):
        overlay = accs
    else:
        W = np.zeros((CHANNELS, CHANNELS, 1, 1), np.float32)
        for i in range(CHANNELS):
            W[recolor.get(i, i), i, 0, 0] = 1.0
        overlay = g.node("Conv", [accs, g.cf([CHANNELS, CHANNELS, 1, 1], W)],
                         kernel_shape=[1, 1], pads=[0, 0, 0, 0])

    acc = g.node("Add", ["input", overlay])        # base (original colours) + overlay
    input_real = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    nbg = g.node("Slice", [acc, g.i64([1]), g.i64([CHANNELS]), g.i64([1])])  # [1,9,30,30]
    nbg_m = g.node("Mul", [nbg, input_real])                            # mask padding
    sum_nbg = g.node("ReduceSum", [nbg_m], axes=[1], keepdims=1)        # [1,1,30,30]
    clipped = g.node("Clip", [sum_nbg], min=0.0, max=1.0)
    ch0 = g.node("Sub", [input_real, clipped])                         # bg channel
    g.node("Concat", [ch0, nbg_m], "output", axis=1)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX semantics for detection)                  #
# --------------------------------------------------------------------------- #
def _shifted_grid(a, dy, dx):
    h, w = a.shape
    s = np.zeros_like(a)
    r0, r1 = max(0, dy), min(h, h + dy)
    c0, c1 = max(0, dx), min(w, w + dx)
    if r0 < r1 and c0 < c1:
        s[r0:r1, c0:c1] = a[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
    return s


def _rec(s, recolor):
    out = np.zeros_like(s)
    for src, t in recolor.items():
        out[s == src] = t
    return out


def _valid_tap(prs, dy, dx, recolor):
    """The recoloured shifted copy must be a colour-consistent subset of b."""
    has = False
    for a, b in prs:
        s = _shifted_grid(a, dy, dx)
        rec = _rec(s, recolor)
        m = (s != 0) & (rec != 0)
        if m.any():
            has = True
            if not np.array_equal(b[m], rec[m]):
                return False
    return has


def _reconstruct(a, taps, recolor):
    """OR base with the recoloured shifted copies; None on a colour clash."""
    r = np.where(a != 0, a, -1)
    for dy, dx in taps:
        s = _shifted_grid(a, dy, dx)
        rec = _rec(s, recolor)
        m = (s != 0) & (rec != 0)
        clash = (r != -1) & m & (r != rec)
        if clash.any():
            return None
        fill = (r == -1) & m
        r[fill] = rec[fill]
    return np.where(r == -1, 0, r)


def _match_all(prs, taps, recolor):
    for a, b in prs:
        r = _reconstruct(a, taps, recolor)
        if r is None or not np.array_equal(r, b):
            return False
    return True


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
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _cand_taps(prs):
    maxd = max(max(a.shape) for a, _ in prs)
    R = min(maxd - 1, _BRUTE)
    cands = {(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1) if dy or dx}
    a0, b0 = prs[0]
    nz = np.argwhere(a0 != 0)
    chg = np.argwhere(a0 != b0)
    if nz.size and chg.size:
        for (pr, pc) in nz[:_ANCHORS]:
            for (qr, qc) in chg:
                dy, dx = int(qr - pr), int(qc - pc)
                if (dy or dx) and abs(dy) <= _MAXOFF and abs(dx) <= _MAXOFF:
                    cands.add((dy, dx))
                    if len(cands) > _MAXCANDS:
                        break
            if len(cands) > _MAXCANDS:
                break
    return sorted(cands, key=lambda s: (abs(s[0]) + abs(s[1]), abs(s[0]), abs(s[1]), s[0], s[1]))


def _recolors(prs):
    added, src = set(), set()
    for a, b in prs:
        added |= set(np.unique(b[(a == 0) & (b != 0)]).tolist())
        src |= set(np.unique(a[a != 0]).tolist())
    added.discard(0)
    recs = [{s: s for s in src}]                        # identity (keep colours)
    for R_ in sorted(added)[:3]:                        # single shadow colour
        m = {s: R_ for s in src}
        if m != recs[0]:
            recs.append(m)
    return recs


def _select_taps(sample, cands, recolor):
    """Greedy minimal tap set reproducing every sample pair (else None)."""
    res = [np.where(a != 0, a, -1) for a, _ in sample]
    if all(np.array_equal(np.where(r == -1, 0, r), b) for r, (_, b) in zip(res, sample)):
        return None                                     # base alone (no change)
    taps = []
    for dy, dx in cands:
        if len(taps) >= _MAXTAPS:
            break
        if not _valid_tap(sample, dy, dx, recolor):
            continue
        newres, adds, clash = [], False, False
        for r, (a, b) in zip(res, sample):
            s = _shifted_grid(a, dy, dx)
            rec = _rec(s, recolor)
            m = (s != 0) & (rec != 0)
            if ((r != -1) & m & (r != rec)).any():
                clash = True
                break
            fill = (r == -1) & m
            if fill.any():
                adds = True
            nr = r.copy()
            nr[fill] = rec[fill]
            newres.append(nr)
        if clash or not adds:
            continue
        res, taps = newres, taps + [(dy, dx)]
        if all(np.array_equal(np.where(r == -1, 0, r), b) for r, (_, b) in zip(res, sample)):
            return taps
    return None


def candidates(ex):
    tt = _pairs(ex, ("train", "test"))
    full = _pairs(ex, ("train", "test", "arc-gen"))
    if not full:
        return []
    if any(a.shape != b.shape for a, b in full):        # shape-preserving family
        return []
    if all(np.array_equal(a, b) for a, b in full):      # identity -> not ours
        return []
    if not all(np.array_equal(b[a != 0], a[a != 0]) for a, b in full):  # base kept
        return []

    sample = tt + full[len(tt):][:_SAMPLE]
    const_size = len({a.shape for a, _ in full}) == 1
    cands = _cand_taps(full)

    for recolor in _recolors(full):
        taps = _select_taps(sample, cands, recolor)
        if not taps:
            continue
        if not _match_all(full, taps, recolor):
            continue
        if not const_size and len(taps) > _MAXTAPS_VARY:
            continue                                    # anti-overfit gate
        try:
            m = build_super(taps, recolor)
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            continue
        ident = all(recolor.get(i, i) == i for i in range(CHANNELS))
        tg = "id" if ident else "R" + str(sorted(set(recolor.values()))[0])
        tapstr = "_".join(f"{dy}.{dx}" for dy, dx in taps)
        return [(f"ovl_{tg}_{tapstr}", m)]
    return []
