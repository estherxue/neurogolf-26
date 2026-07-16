"""Deep geometric/color composition search (depth 1..3), origin-anchored.

This is a *deeper* sibling of ``family_search``: it enumerates depth<=3 spatial
compositions and pairs each with a single inferred recolor, validating every
candidate EXACTLY in numpy before emitting the chained ONNX graph.

Primitive set
-------------
Size-independent (operate on the full 30x30 zero-padded tensor; content stays
anchored at the top-left for ANY grid size, so they generalise across the
variable-size splits):
    identity, transpose, upscale k (k=2..4), downscale k (k=2..4),
    top-left crop(h, w)            (output is a fixed h x w; pins the size)
Size-dependent (the correct position depends on the data-dependent grid size, so
a STATIC graph can only express them when the content size flowing in is CONSTANT
across every split -- which holds when the input size is constant, OR after a crop
has pinned the size to a known h x w):
    fliplr, flipud, rot90, rot180, rot270, tile(n, m)
and a single trailing ``recolor(map)`` (Gather if bijective else 1x1 Conv).

Why "spatial composition  +  one trailing recolor" is the whole space
--------------------------------------------------------------------
Every spatial primitive here is COLOUR-BLIND (it only moves / copies / drops whole
pixels by position).  A recolor only relabels colours by colour.  They therefore
commute -- T(R(x)) == R(T(x)) -- and two recolors fuse into one.  So ANY depth<=3
sequence drawn from {spatial ops, recolors} equals (composed spatial op) then (one
recolor).  The engine enumerates spatial compositions only and, for each, infers a
single consistent colour map from the data; this covers recolors at any position.

What this module adds over ``family_search``
--------------------------------------------
* ATOMIC rot90 / rot270 (family_search only reaches them as a 2-step transpose+flip,
  which leaves no budget for a third op).
* A top-left ``crop(h, w)`` primitive.  Crop is origin-anchored and -- crucially --
  pins the downstream content size to a constant ``(h, w)`` regardless of the input
  size, which RE-ENABLES the size-dependent ops (tile / flips / rotations) for the
  variable-size splits.  This unlocks crop+tile / crop+flip / upscale+crop combos.

Anti-overfit
------------
The integrator runs this on 70% of arc-gen and grades on 100%, so detection is
purely STRUCTURAL: size-dependent ops are only emitted when the size is provably
constant (constant input size, or pinned by a crop); every candidate is
full-validated for EXACT equality on ALL visible pairs (train+test+arc-gen) and the
single colour map is re-inferred over all of them.  The harness re-checks exactness
on the held-out split, so coincidental matches are rejected.
"""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)  # "to the beginning" sentinel for reverse Slice

# search budgets (keep per-task search well under ~2s)
_PROBE = 4           # pairs used for the cheap BFS / dedup pass
_MAX_STATES = 3500   # cap on distinct visited spatial-effect states
_MAX_RECORDS = 600   # cap on recorded sequences to full-validate
_MAX_YIELD = 6       # candidates emitted per task
_MAX_CROPS = 5       # crop primitive candidates
_MAX_TILES = 6       # tile primitive candidates

Prim = namedtuple("Prim", "key sim build st size_dep weight")


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
# low-level ONNX helpers
# --------------------------------------------------------------------------- #
def _slice_content(g, src, h, w):
    """input[:, :, 0:h, 0:w] (no-op when it is already the full canvas)."""
    if h == HEIGHT and w == WIDTH:
        return src
    cs = g.init(INT64, [2], [0, 0])
    ce = g.init(INT64, [2], [h, w])
    ca = g.init(INT64, [2], [2, 3])
    return g.node("Slice", [src, cs, ce, ca])


def _pad_full(g, src, h, w):
    """Pad a top-left h x w content block back up to 30x30 (no-op when full)."""
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


# --------------------------------------------------------------------------- #
# ONNX fragment builders.  Each: build(g, src, h, w) -> out_tensor_name.
# `h, w` are the content size flowing INTO the fragment; size-independent
# fragments ignore them (they may be None).
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


def _b_crop(ch, cw):
    def build(g, src, h, w):
        c = _slice_content(g, src, ch, cw)
        return _pad_full(g, c, ch, cw)
    return build


def _b_flip(axes):
    def build(g, src, h, w):
        c = _slice_content(g, src, h, w)
        r = _reverse(g, c, axes, {2: h, 3: w})
        return _pad_full(g, r, h, w)
    return build


def _b_rot90(g, src, h, w):       # a.T[::-1, :] -> shape (w, h)
    c = _slice_content(g, src, h, w)
    t = g.node("Transpose", [c], perm=[0, 1, 3, 2])              # (w, h)
    r = _reverse(g, t, [2], {2: w, 3: h})
    return _pad_full(g, r, w, h)


def _b_rot270(g, src, h, w):      # a.T[:, ::-1] -> shape (w, h)
    c = _slice_content(g, src, h, w)
    t = g.node("Transpose", [c], perm=[0, 1, 3, 2])              # (w, h)
    r = _reverse(g, t, [3], {2: w, 3: h})
    return _pad_full(g, r, w, h)


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
# numpy simulators (faithful to the fragments, INCLUDING the zero-padding
# semantics; surviving branches never leave the 30x30 canvas)
# --------------------------------------------------------------------------- #
def _sim_transpose(a):
    return a.T


def _sim_upscale(k):
    return lambda a: np.kron(a, np.ones((k, k), dtype=a.dtype))


def _sim_downscale(k):
    return lambda a: a[::k, ::k]


def _sim_crop(ch, cw):
    def f(a):
        out = np.zeros((ch, cw), dtype=a.dtype)
        hh = min(ch, a.shape[0])
        ww = min(cw, a.shape[1])
        out[:hh, :ww] = a[:hh, :ww]
        return out
    return f


def _sim_fliplr(a):
    return a[:, ::-1]


def _sim_flipud(a):
    return a[::-1, :]


def _sim_rot90(a):
    return a.T[::-1, :]


def _sim_rot180(a):
    return a[::-1, ::-1]


def _sim_rot270(a):
    return a.T[:, ::-1]


def _sim_tile(n, m):
    return lambda a: np.tile(a, (n, m))


# --------------------------------------------------------------------------- #
# size transforms on a (h, w) state (None = "varies / unknown")
# --------------------------------------------------------------------------- #
def _st_swap(s):
    return (s[1], s[0]) if s else None


def _st_id(s):
    return s


def _st_upscale(k):
    return lambda s: (k * s[0], k * s[1]) if s else None


def _st_downscale(k):
    return lambda s: (math.ceil(s[0] / k), math.ceil(s[1] / k)) if s else None


def _st_crop(ch, cw):
    return lambda s: (ch, cw)          # crop PINS the size for everyone


def _st_tile(n, m):
    return lambda s: (n * s[0], m * s[1]) if s else None


# --------------------------------------------------------------------------- #
# candidate factor inference
# --------------------------------------------------------------------------- #
def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _crop_candidates(allp, const_size, h0, w0):
    """Priority-ordered top-left crop sizes worth trying, capped."""
    out_shapes = {b.shape for _, b in allp}
    const_out = next(iter(out_shapes)) if len(out_shapes) == 1 else None

    ordered = []

    def push(c):
        if c is None:
            return
        h, w = c
        if 1 <= h <= HEIGHT and 1 <= w <= WIDTH and (h, w) != (HEIGHT, WIDTH):
            if c not in ordered:
                ordered.append(c)

    # 1) crop straight to the constant output shape (upscale+crop, tile+crop, ...)
    if const_out is not None:
        push(const_out)
        # 2) periods of a constant output (crop-then-tile), smallest factors first
        H, W = const_out
        for s in range(2, 11):
            for n in range(1, s):
                m = s - n
                if 1 <= n <= 5 and 1 <= m <= 5 and (n, m) != (1, 1):
                    if H % n == 0 and W % m == 0:
                        push((H // n, W // m))

    # 3) top-left divisor blocks of a constant input (crop a fixed window)
    if const_size:
        for ph in _divisors(h0):
            for pw in _divisors(w0):
                if (ph, pw) != (h0, w0):
                    push((ph, pw))

    return ordered[:_MAX_CROPS]


def _tile_candidates(allp, crop_cands):
    """Plausible (n, m) tile factors as integer output/cell ratios (cells = raw
    input shapes and crop-candidate sizes), both orientations, capped."""
    cands = set()

    def add(n, m):
        for a, b in ((n, m), (m, n)):
            if 1 <= a <= 5 and 1 <= b <= 5 and not (a == 1 and b == 1):
                cands.add((a, b))

    cells = set(crop_cands)
    for a, _ in allp:
        cells.add(a.shape)

    for _, b in allp:
        H, W = b.shape
        for ph, pw in cells:
            if ph and pw:
                if H % ph == 0 and W % pw == 0:
                    add(H // ph, W // pw)
                if H % pw == 0 and W % ph == 0:      # transposed-cell orientation
                    add(H // pw, W // ph)

    return sorted(cands, key=lambda nm: (nm[0] * nm[1], nm))[:_MAX_TILES]


# --------------------------------------------------------------------------- #
# primitive table
# --------------------------------------------------------------------------- #
def _primitives(const_size, crop_cands, tile_cands):
    prims = [Prim("T", _sim_transpose, _b_transpose, _st_swap, False, 1)]
    for k in (2, 3, 4):
        prims.append(Prim(f"up{k}", _sim_upscale(k), _b_upscale(k),
                          _st_upscale(k), False, 4))
    for k in (2, 3, 4):
        prims.append(Prim(f"dn{k}", _sim_downscale(k), _b_downscale(k),
                          _st_downscale(k), False, 2))
    for (ch, cw) in crop_cands:
        prims.append(Prim(f"crop{ch}x{cw}", _sim_crop(ch, cw), _b_crop(ch, cw),
                          _st_crop(ch, cw), False, 2))
    # size-dependent ops need a KNOWN size: available from the start when the
    # input size is constant, otherwise only reachable after a crop pins it.
    if const_size or crop_cands:
        prims.append(Prim("fliplr", _sim_fliplr, _b_flip([3]), _st_id, True, 2))
        prims.append(Prim("flipud", _sim_flipud, _b_flip([2]), _st_id, True, 2))
        prims.append(Prim("rot90", _sim_rot90, _b_rot90, _st_swap, True, 3))
        prims.append(Prim("rot180", _sim_rot180, _b_flip([2, 3]), _st_id, True, 2))
        prims.append(Prim("rot270", _sim_rot270, _b_rot270, _st_swap, True, 3))
        for (n, m) in tile_cands:
            prims.append(Prim(f"tile{n}x{m}", _sim_tile(n, m), _b_tile(n, m),
                              _st_tile(n, m), True, 3))
    return prims


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
    """Small representative probe (train+test first), distinct shapes favoured."""
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
    for a, b in pool:                       # distinct (in, out) shapes first
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
# BFS over spatial sequences (probe-based, deduped, pruned, size-aware)
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


def _search(prims, probe, init_size):
    inputs = [a for a, _ in probe]
    targets = [b for _, b in probe]
    records = []                                  # (seq, init_size) tuples
    seen = {(_sig(inputs), init_size)}

    def recordable(preds):
        if any(p.shape != t.shape for p, t in zip(preds, targets)):
            return False
        return _consistent_map(list(zip(preds, targets))) is not None

    if recordable(inputs):
        records.append([])                        # empty seq -> pure recolor/id

    frontier = [([], inputs, init_size)]
    for _ in range(3):                            # depth 1..3
        nxt = []
        for seq, preds, size in frontier:
            for prim in prims:
                if prim.size_dep and size is None:
                    continue
                newp = _apply(prim.sim, preds)
                if newp is None:
                    continue
                nsize = prim.st(size)
                key = (_sig(newp), nsize)
                if key in seen:
                    continue
                seen.add(key)
                newseq = seq + [prim]
                if recordable(newp):
                    records.append(newseq)
                    if len(records) >= _MAX_RECORDS:
                        return records
                nxt.append((newseq, newp, nsize))
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
    pairs (cmap(seq(a)) == b), or None."""
    mapped = []
    for a, b in allp:
        p = a
        for prim in seq:
            p = prim.sim(p)
            if p.ndim != 2 or p.size == 0 or p.shape[0] > HEIGHT or p.shape[1] > WIDTH:
                return None
        if p.shape != b.shape:
            return None
        mapped.append((p, b))
    return _consistent_map(mapped)


def _build(seq, cmap, init_size):
    g = _G()
    cur = "input"
    size = init_size
    for prim in seq:
        h, w = (size if size else (None, None))
        cur = prim.build(g, cur, h, w)
        size = prim.st(size)
    if not _identity_map(cmap):
        h, w = (size if size else (None, None))
        cur = _b_recolor(cmap)(g, cur, h, w)
    if not g.nodes:                               # pure identity
        return _model([oh.make_node("Identity", ["input"], ["output"])])
    g.nodes[-1].output[0] = "output"
    return _model(g.nodes, g.inits)


def _cost_rank(seq, cmap):
    w = sum(p.weight for p in seq)
    if not _identity_map(cmap):
        w += 1 if _is_bijection(cmap) else 3
    return (len(seq), w)


def _name(seq, cmap):
    parts = [p.key for p in seq]
    if not _identity_map(cmap):
        parts.append("recolorB" if _is_bijection(cmap) else "recolorC")
    return "mega_" + ("_".join(parts) if parts else "identity")


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
    init_size = (h0, w0) if const_size else None

    crop_cands = _crop_candidates(allp, const_size, h0, w0)
    tile_cands = _tile_candidates(allp, crop_cands)
    prims = _primitives(const_size, crop_cands, tile_cands)
    probe = _probe_pairs(ex, allp)

    records = _search(prims, probe, init_size)

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
            model = _build(seq, cmap, init_size)
        except Exception:
            continue
        out.append((nm, model))
    return out
