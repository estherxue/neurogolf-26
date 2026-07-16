"""Program-search engine: auto-solve compositional ARC tasks by searching
depth<=3 compositions of ORIGIN-SAFE primitives and validating each candidate
exactly (in numpy) before emitting the composed ONNX graph.

Primitives
----------
Size-independent (work on the full 30x30 zero-padded tensor, content stays anchored
top-left for ANY grid size, so they generalise across the variable-size splits):
    identity, transpose, upscale k (k=2..5), downscale k (k=2..5)
Size-dependent (correct position depends on the data-dependent grid size, so a STATIC
graph can only express them when the input size is CONSTANT across every split):
    fliplr, flipud, rot180, tile(n, m)
and a trailing per-colour `recolor(map)` (inferred from data; Gather if bijective,
else 1x1 Conv).

Why "spatial composition  +  one trailing recolor" is the whole space
--------------------------------------------------------------------
Every spatial primitive here is COLOUR-BLIND (it only moves / copies / drops whole
pixels by position).  A recolor only relabels colours by colour.  They therefore
commute -- T(R(x)) == R(T(x)) -- and two recolors fuse into one.  So ANY depth<=3
sequence drawn from {spatial ops, recolors} equals  (composed spatial op) then
(a single recolor).  The engine thus enumerates spatial compositions only and, for
each, infers a single consistent colour map from the data; this covers recolors at
any position within the budget.

Algorithm (per task)
--------------------
1. Collect the scored pairs (grader skips grids whose H or W exceeds 30).
2. BFS over spatial sequences (depth 0..3) on a small probe set, deduplicating by
   the exact predicted-grid signature and pruning any step whose output would leave
   the 30x30 canvas (such intermediates are not faithfully representable).  Record a
   sequence when, on the probe, every predicted shape matches the target shape AND a
   consistent colour map exists.
3. Full-validate each recorded sequence on ALL scored pairs (train+test+arc-gen),
   re-inferring the colour map over all of them; keep the exact matches.
4. Order cheapest-first (shortest / simplest), build the composed ONNX graph by
   chaining the per-primitive fragments with unique tensor names, and yield up to a
   few candidates.  The harness re-validates exactness on the full + private split.
"""
from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)  # "to the beginning" sentinel for reverse Slice

# search budgets (keep per-task search well under ~1s)
_PROBE = 4          # pairs used for the cheap BFS / dedup pass
_MAX_STATES = 2500  # cap on distinct visited spatial-effect states
_MAX_YIELD = 6      # candidates emitted per task


# --------------------------------------------------------------------------- #
# node / initializer accumulator with auto-unique tensor names
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


def _is_bijection(cmap):
    return sorted(cmap) == list(range(CHANNELS))


def _identity_map(cmap):
    return cmap == list(range(CHANNELS))


# --------------------------------------------------------------------------- #
# ONNX fragment builders.  Each: build(g, src, h, w) -> out_tensor_name.
# `h, w` are the content size flowing INTO the fragment (ints in constant-size
# mode, None otherwise); size-independent fragments ignore them.
# --------------------------------------------------------------------------- #
def _b_transpose(g, src, h, w):
    return g.node("Transpose", [src], perm=[0, 1, 3, 2])


def _b_upscale(k):
    def build(g, src, h, w):
        nh = math.ceil(HEIGHT / k)
        nw = math.ceil(WIDTH / k)
        s0 = g.init(INT64, [2], [0, 0])
        se = g.init(INT64, [2], [nh, nw])
        sa = g.init(INT64, [2], [2, 3])
        sm = g.node("Slice", [src, s0, se, sa])                  # top-left feeder
        sc = g.init(DATA_TYPE, [4], [1.0, 1.0, float(k), float(k)])
        up = g.node("Resize", [sm, sc], mode="nearest")          # nh*k x nw*k
        if nh * k == HEIGHT and nw * k == WIDTH:
            return up
        c0 = g.init(INT64, [2], [0, 0])
        ce = g.init(INT64, [2], [HEIGHT, WIDTH])
        ca = g.init(INT64, [2], [2, 3])
        return g.node("Slice", [up, c0, ce, ca])
    return build


def _b_downscale(k):
    def build(g, src, h, w):
        sz_h = len(range(0, HEIGHT, k))
        sz_w = len(range(0, WIDTH, k))
        s0 = g.init(INT64, [2], [0, 0])
        se = g.init(INT64, [2], [HEIGHT, WIDTH])
        sa = g.init(INT64, [2], [2, 3])
        st = g.init(INT64, [2], [k, k])
        sm = g.node("Slice", [src, s0, se, sa, st])              # sz_h x sz_w
        return g.node("Pad", [sm], mode="constant", value=0.0,
                      pads=[0, 0, 0, 0, 0, 0, HEIGHT - sz_h, WIDTH - sz_w])
    return build


def _slice_content(g, src, h, w):
    if h == HEIGHT and w == WIDTH:
        return src
    cs = g.init(INT64, [2], [0, 0])
    ce = g.init(INT64, [2], [h, w])
    ca = g.init(INT64, [2], [2, 3])
    return g.node("Slice", [src, cs, ce, ca])


def _pad_full(g, src, h, w):
    if h == HEIGHT and w == WIDTH:
        return src
    return g.node("Pad", [src], mode="constant", value=0.0,
                  pads=[0, 0, 0, 0, 0, 0, HEIGHT - h, WIDTH - w])


def _reverse(g, src, axes, dims):
    n = len(axes)
    starts = g.init(INT64, [n], [dims[a] - 1 for a in axes])
    ends = g.init(INT64, [n], [_NEG] * n)
    ax = g.init(INT64, [n], list(axes))
    steps = g.init(INT64, [n], [-1] * n)
    return g.node("Slice", [src, starts, ends, ax, steps])


def _b_flip(axes):
    def build(g, src, h, w):
        c = _slice_content(g, src, h, w)
        r = _reverse(g, c, axes, {2: h, 3: w})
        return _pad_full(g, r, h, w)
    return build


def _b_tile(n, m):
    def build(g, src, h, w):
        c = _slice_content(g, src, h, w)
        rep = g.init(INT64, [4], [1, 1, n, m])
        t = g.node("Tile", [c, rep])
        return _pad_full(g, t, n * h, m * w)
    return build


def _b_recolor(cmap):
    def build(g, src, h, w):
        if _is_bijection(cmap):
            inv = [0] * CHANNELS              # output[:, j] = input[:, inv[j]]
            for i, o in enumerate(cmap):
                inv[o] = i
            idx = g.init(INT64, [CHANNELS], inv)
            return g.node("Gather", [src, idx], axis=1)
        weights = [0.0] * (CHANNELS * CHANNELS)   # [O, I, 1, 1]
        for i, o in enumerate(cmap):
            weights[o * CHANNELS + i] = 1.0
        wt = g.init(DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], weights)
        return g.node("Conv", [src, wt], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return build


# --------------------------------------------------------------------------- #
# numpy simulators (faithful to the fragments; surviving branches never leave
# the 30x30 canvas so no cropping is required -- branches that would are pruned)
# --------------------------------------------------------------------------- #
def _sim_transpose(a):
    return a.T


def _sim_upscale(k):
    return lambda a: np.kron(a, np.ones((k, k), dtype=a.dtype))


def _sim_downscale(k):
    return lambda a: a[::k, ::k]


def _sim_fliplr(a):
    return a[:, ::-1]


def _sim_flipud(a):
    return a[::-1, :]


def _sim_rot180(a):
    return a[::-1, ::-1]


def _sim_tile(n, m):
    return lambda a: np.tile(a, (n, m))


# size transform on (h, w) -- mirrors the simulator's effect on shape
def _st_transpose(hw):
    return (hw[1], hw[0])


def _st_upscale(k):
    return lambda hw: (k * hw[0], k * hw[1])


def _st_downscale(k):
    return lambda hw: (math.ceil(hw[0] / k), math.ceil(hw[1] / k))


def _st_id(hw):
    return hw


def _st_tile(n, m):
    return lambda hw: (n * hw[0], m * hw[1])


# --------------------------------------------------------------------------- #
# primitive table.  Each entry: (key, sim, build, size_transform, weight)
# weight = rough cost rank for cheapest-first ordering.
# --------------------------------------------------------------------------- #
def _primitives(const_size, h0, w0, tile_nm):
    prims = [("T", _sim_transpose, _b_transpose, _st_transpose, 1)]
    for k in range(2, 6):
        prims.append((f"up{k}", _sim_upscale(k), _b_upscale(k), _st_upscale(k), 4))
    for k in range(2, 6):
        prims.append((f"dn{k}", _sim_downscale(k), _b_downscale(k), _st_downscale(k), 2))
    if const_size:
        prims.append(("fliplr", _sim_fliplr, _b_flip([3]), _st_id, 2))
        prims.append(("flipud", _sim_flipud, _b_flip([2]), _st_id, 2))
        prims.append(("rot180", _sim_rot180, _b_flip([2, 3]), _st_id, 2))
        for (n, m) in sorted(tile_nm):
            if n * h0 > HEIGHT or m * w0 > WIDTH:
                continue
            prims.append((f"tile{n}x{m}", _sim_tile(n, m), _b_tile(n, m),
                          _st_tile(n, m), 3))
    return prims


def _tile_candidates(allp):
    """Plausible (n, m) tile factors: integer output/input shape ratios seen in the
    data, in both orientations (so transpose-then-tile / tile-then-transpose are
    reachable), plus their transposes."""
    cands = set()

    def add(n, m):
        if 1 <= n <= 5 and 1 <= m <= 5 and not (n == 1 and m == 1):
            cands.add((n, m))
            cands.add((m, n))

    for a, b in allp:
        h, w = a.shape
        H, W = b.shape
        if h and w:
            if H % h == 0 and W % w == 0:
                add(H // h, W // w)
            if H % w == 0 and W % h == 0:   # transposed-input orientation
                add(H // w, W // h)
    return cands


# --------------------------------------------------------------------------- #
# colour-map inference (single consistent cell-wise map, or None)
# --------------------------------------------------------------------------- #
def _consistent_map(pairs):
    m = {}
    for X, Y in pairs:
        if X.shape != Y.shape:
            return None
        for xv, yv in zip(X.ravel().tolist(), Y.ravel().tolist()):
            xv, yv = int(xv), int(yv)
            if not (0 <= xv < CHANNELS) or not (0 <= yv < CHANNELS):
                return None
            if xv in m:
                if m[xv] != yv:
                    return None
            else:
                m[xv] = yv
    return [m.get(i, i) for i in range(CHANNELS)]


# --------------------------------------------------------------------------- #
# task pairs
# --------------------------------------------------------------------------- #
def _all_pairs(ex):
    """Every (input, output) the grader will actually score (both <= 30x30)."""
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"])
            b = np.array(e["output"])
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > HEIGHT or max(b.shape) > HEIGHT:
                continue
            out.append((a, b))
    return out


def _probe_pairs(ex, allp):
    """Small representative probe (train+test first), with distinct shapes favoured."""
    pref = []
    for s in ("train", "test"):
        for e in ex.get(s, []):
            a = np.array(e["input"])
            b = np.array(e["output"])
            if a.ndim == 2 and b.ndim == 2 and a.size and b.size \
               and max(a.shape) <= HEIGHT and max(b.shape) <= HEIGHT:
                pref.append((a, b))
    pool = pref if pref else allp
    chosen, seen_shapes = [], set()
    for a, b in pool:                       # distinct input shapes first
        key = (a.shape, b.shape)
        if key not in seen_shapes:
            seen_shapes.add(key)
            chosen.append((a, b))
        if len(chosen) >= _PROBE:
            break
    for a, b in pool:                       # top up if still room
        if len(chosen) >= _PROBE:
            break
        chosen.append((a, b))
    return chosen or pool[:_PROBE]


# --------------------------------------------------------------------------- #
# BFS over spatial sequences (probe-based, deduped, pruned)
# --------------------------------------------------------------------------- #
def _sig(preds):
    return tuple((p.shape, p.astype(np.int16).tobytes()) for p in preds)


def _apply(sim, preds):
    out = []
    for p in preds:
        q = sim(p)
        if q.ndim != 2 or q.size == 0 or q.shape[0] > HEIGHT or q.shape[1] > WIDTH:
            return None
        out.append(q)
    return out


def _search(prims, probe):
    inputs = [a for a, _ in probe]
    targets = [b for _, b in probe]
    records = []           # list of spatial sequences (each: list of prim tuples)
    seen = {_sig(inputs)}

    def recordable(preds):
        if any(p.shape != t.shape for p, t in zip(preds, targets)):
            return False
        return _consistent_map(list(zip(preds, targets))) is not None

    if recordable(inputs):
        records.append([])                       # empty seq -> pure recolor/identity

    frontier = [([], inputs)]
    for _ in range(3):                           # depth 1..3
        nxt = []
        for seq, preds in frontier:
            for prim in prims:
                newp = _apply(prim[1], preds)
                if newp is None:
                    continue
                s = _sig(newp)
                if s in seen:
                    continue
                seen.add(s)
                newseq = seq + [prim]
                if recordable(newp):
                    records.append(newseq)
                nxt.append((newseq, newp))
                if len(seen) > _MAX_STATES:
                    return records
        frontier = nxt
        if not frontier:
            break
    return records


# --------------------------------------------------------------------------- #
# full validation + build
# --------------------------------------------------------------------------- #
def _full_map(seq, allp):
    """Apply spatial seq to every scored input; return a colour map valid on ALL
    pairs (cmap(seq(a)) == b for all), or None."""
    mapped = []
    for a, b in allp:
        p = a
        for prim in seq:
            p = prim[1](p)
            if p.ndim != 2 or p.size == 0 or p.shape[0] > HEIGHT or p.shape[1] > WIDTH:
                return None
        if p.shape != b.shape:
            return None
        mapped.append((p, b))
    return _consistent_map(mapped)


def _build(seq, cmap, const_size, h0, w0):
    g = _G()
    cur = "input"
    h, w = (h0, w0) if const_size else (None, None)
    for prim in seq:
        cur = prim[2](g, cur, h, w)
        if const_size:
            h, w = prim[3]((h, w))
    if not _identity_map(cmap):
        cur = _b_recolor(cmap)(g, cur, h, w)
    if not g.nodes:                              # pure identity
        return _model([oh.make_node("Identity", ["input"], ["output"])])
    g.nodes[-1].output[0] = "output"
    return _model(g.nodes, g.inits)


def _cost_rank(seq, cmap):
    w = sum(p[4] for p in seq)
    if not _identity_map(cmap):
        w += 1 if _is_bijection(cmap) else 3
    return (len(seq), w)


def _name(seq, cmap):
    parts = [p[0] for p in seq]
    if not _identity_map(cmap):
        parts.append("recolorB" if _is_bijection(cmap) else "recolorC")
    return "search_" + ("_".join(parts) if parts else "identity")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def candidates(ex):
    allp = _all_pairs(ex)
    if not allp:
        return []

    shapes = {a.shape for a, _ in allp}
    const_size = len(shapes) == 1
    h0, w0 = next(iter(shapes)) if const_size else (0, 0)

    tile_nm = _tile_candidates(allp) if const_size else set()
    prims = _primitives(const_size, h0, w0, tile_nm)
    probe = _probe_pairs(ex, allp)

    records = _search(prims, probe)

    # full-validate (all scored pairs) + collect (rank, name, seq, cmap)
    solutions, seen_names = [], set()
    for seq in records:
        cmap = _full_map(seq, allp)
        if cmap is None:
            continue
        nm = _name(seq, cmap)
        if nm in seen_names:
            continue
        seen_names.add(nm)
        solutions.append((_cost_rank(seq, cmap), nm, seq, cmap))

    if not solutions:
        return []

    solutions.sort(key=lambda s: s[0])

    out = []
    for _, nm, seq, cmap in solutions[:_MAX_YIELD]:
        try:
            model = _build(seq, cmap, const_size, h0, w0)
        except Exception:
            continue
        out.append((nm, model))
    return out
