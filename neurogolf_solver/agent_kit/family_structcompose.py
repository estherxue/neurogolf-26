"""Structural compositions: ONE structural op, optionally followed by a geometric
op and/or a global recolor, chained end-to-end into a single opset-10 graph.

Why a separate "structural composition" family
----------------------------------------------
The single-op families (``family_flood`` enclosed-hole / seed flood,
``family_symmetry`` / ``family_symfixed`` symmetric overlays) each express ONE
structural primitive.  Many ARC tasks are that primitive PLUS a colour relabel or
a transpose -- e.g. "fill the enclosed holes, then recolour blue->red", or
"diagonally symmetrise, then recolour".  None of the single-op families can emit
those.  This module searches such two/three-step compositions, simulating every
candidate in numpy and keeping only the ones that reproduce ALL train+test+arc-gen
pairs EXACTLY, then builds the chained ONNX (unique tensor names) cheapest first.

The composition shape
---------------------
    output = recolor( geom( STRUCT( input ) ) )

with ``geom`` in {identity, transpose} (transpose is origin-safe) and ``recolor`` a
single global per-colour map (inferred from data; Gather if bijective else 1x1
Conv).  ``recolor`` is always placed LAST: the structural op's parameters are
inferred structurally first (so the relabel only ever has to explain the colours,
never the geometry/topology), which keeps inference tractable and avoids the
intractable "recolor-first" search.

Structural primitives (reused logic)
------------------------------------
  * enclosed-hole flood-fill         (family_flood ``_build``):   fill enclosed
        background(0) cells with one colour; origin-safe (padding floods like the
        outside).  conn4 / conn8.
  * seed flood-fill                  (family_flood ``_build_seed``): spread a seed
        colour through background until it hits a wall.  conn4 / conn8.
  * symmetric overlay via Max        (family_symmetry / family_symfixed): one-hot
        OR of the grid with transformed copies of itself.  Diagonal (id|T) is
        origin-safe for any square grid; for CONSTANT-size grids the windowed
        flips/rotations (lr/ud/rot180/diag/antidiag/D4) are origin-safe too.

Origin safety (CONTEXT padding gotcha)
--------------------------------------
Every emitted primitive keeps content anchored at the top-left for the variable,
zero-padded 30x30 canvas: enclosed/seed flood treat padding like the outside;
``Transpose`` keeps the origin; windowed flips operate inside the known fixed
window and zero-pad back.  Recolour is a no-bias linear channel op so the all-zero
padding stays all-zero.  Anything size-dependent is emitted ONLY when the grid size
is constant across every split (then the exact window is known).

Anti-overfit: parameters (fill colour, connectivity, overlay axes, recolour map)
are inferred from train+test first and the candidate is re-validated for EXACT
equality on every scored pair (the grader's gate), so coincidences are rejected.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)
_MAX_YIELD = 6


# --------------------------------------------------------------------------- #
# node / initializer accumulator with auto-unique tensor names                #
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
# colour-map inference (single consistent cell-wise map over equal-shape pairs)#
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
# ONNX fragments (each: take graph `g` + src tensor name -> new tensor name)   #
# --------------------------------------------------------------------------- #
def _frag_recolor(g, src, cmap):
    if _is_bijection(cmap):
        inv = [0] * CHANNELS                       # out[:, j] = in[:, inv[j]]
        for i, o in enumerate(cmap):
            inv[o] = i
        idx = g.init(INT64, [CHANNELS], inv)
        return g.node("Gather", [src, idx], axis=1)
    weights = [0.0] * (CHANNELS * CHANNELS)        # [O, I, 1, 1]
    for i, o in enumerate(cmap):
        weights[o * CHANNELS + i] = 1.0
    wt = g.init(DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], weights)
    return g.node("Conv", [src, wt], kernel_shape=[1, 1], pads=[0, 0, 0, 0])


def _border_mask():
    m = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    m[:, :, 0, :] = 1.0
    m[:, :, HEIGHT - 1, :] = 1.0
    m[:, :, :, 0] = 1.0
    m[:, :, :, WIDTH - 1] = 1.0
    return m.ravel().tolist()


def _prop_kernel(conn8):
    if conn8:
        k = np.ones((1, 1, 3, 3), np.float32)
    else:
        k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float32).reshape(1, 1, 3, 3)
    return k.ravel().tolist()


def _frag_fill_enc(g, src, dst, n_steps, conn8):
    """Recolour enclosed background(0) holes to colour `dst` (family_flood._build)."""
    w_open = [0.0] + [-1.0] * (CHANNELS - 1)       # [1,10,1,1]: open = 1 - sum(ch1..9)
    wo = g.init(DATA_TYPE, [1, CHANNELS, 1, 1], w_open)
    bo = g.init(DATA_TYPE, [1], [1.0])
    open_ = g.node("Conv", [src, wo, bo], kernel_shape=[1, 1], pads=[0, 0, 0, 0])

    bmask = g.init(DATA_TYPE, [1, 1, HEIGHT, WIDTH], _border_mask())
    prev = g.node("Mul", [open_, bmask])

    kprop = g.init(DATA_TYPE, [1, 1, 3, 3], _prop_kernel(conn8))
    for _ in range(n_steps):
        nb = g.node("Conv", [prev, kprop], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        prev = g.node("Min", [nb, open_])

    enc = g.node("Sub", [open_, prev])
    w_add = [0.0] * CHANNELS                        # [10,1,1,1]: -enc on ch0, +enc on dst
    w_add[0] = -1.0
    w_add[dst] = 1.0
    wa = g.init(DATA_TYPE, [CHANNELS, 1, 1, 1], w_add)
    addmap = g.node("Conv", [enc, wa], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return g.node("Add", [src, addmap])


def _frag_fill_seed(g, src, seed, n_steps, conn8):
    """Spread colour `seed` through background(0) until it hits a wall
    (family_flood._build_seed)."""
    w_ch0 = [0.0] * CHANNELS
    w_ch0[0] = 1.0
    wc0 = g.init(DATA_TYPE, [1, CHANNELS, 1, 1], w_ch0)
    ch0 = g.node("Conv", [src, wc0], kernel_shape=[1, 1], pads=[0, 0, 0, 0])

    w_chs = [0.0] * CHANNELS
    w_chs[seed] = 1.0
    wcs = g.init(DATA_TYPE, [1, CHANNELS, 1, 1], w_chs)
    r0 = g.node("Conv", [src, wcs], kernel_shape=[1, 1], pads=[0, 0, 0, 0])

    allowed = g.node("Add", [ch0, r0])
    kprop = g.init(DATA_TYPE, [1, 1, 3, 3], _prop_kernel(conn8))
    prev = r0
    for _ in range(n_steps):
        nb = g.node("Conv", [prev, kprop], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        prev = g.node("Min", [nb, allowed])

    fill = g.node("Min", [prev, ch0])
    w_add = [0.0] * CHANNELS
    w_add[0] = -1.0
    w_add[seed] = 1.0
    wa = g.init(DATA_TYPE, [CHANNELS, 1, 1, 1], w_add)
    addmap = g.node("Conv", [fill, wa], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return g.node("Add", [src, addmap])


def _slice_ch(g, src, a, b):
    s = g.init(INT64, [1], [a])
    e = g.init(INT64, [1], [b])
    ax = g.init(INT64, [1], [1])
    return g.node("Slice", [src, s, e, ax])


def _combine_overlay(g, copies):
    """One-hot OR of equal-shape copies: Max over colour channels 1..9, product
    (AND) over channel 0, concat -> valid one-hot."""
    cols = [_slice_ch(g, c, 1, CHANNELS) for c in copies]
    colored = cols[0] if len(cols) == 1 else g.node("Max", cols)
    bgs = [_slice_ch(g, c, 0, 1) for c in copies]
    bg = bgs[0]
    for j in range(1, len(bgs)):
        bg = g.node("Mul", [bg, bgs[j]])
    return g.node("Concat", [bg, colored], axis=1)


def _frag_overlay_full(g, src, keys):
    """Origin-safe overlay on the full 30x30 tensor (keys subset of {id, T})."""
    copies = []
    for k in keys:
        if k == "id":
            copies.append(src)
        elif k == "T":
            copies.append(g.node("Transpose", [src], perm=[0, 1, 3, 2]))
        else:
            raise ValueError(k)
    return _combine_overlay(g, copies)


def _window(g, src, H, W):
    if H == HEIGHT and W == WIDTH:
        return src
    s = g.init(INT64, [2], [0, 0])
    e = g.init(INT64, [2], [H, W])
    ax = g.init(INT64, [2], [2, 3])
    return g.node("Slice", [src, s, e, ax])


def _unwindow(g, src, H, W):
    if H == HEIGHT and W == WIDTH:
        return g.node("Identity", [src])
    return g.node("Pad", [src], mode="constant", value=0.0,
                  pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])


def _copy_win(g, win, key):
    if key == "id":
        return win
    if key == "T":
        return g.node("Transpose", [win], perm=[0, 1, 3, 2])
    if key == "antiT":
        r = _copy_win(g, win, "rot180")
        return g.node("Transpose", [r], perm=[0, 1, 3, 2])
    axes = {"fliplr": [3], "flipud": [2], "rot180": [2, 3]}[key]
    n = len(axes)
    s = g.init(INT64, [n], [-1] * n)
    e = g.init(INT64, [n], [_NEG] * n)
    ax = g.init(INT64, [n], axes)
    st = g.init(INT64, [n], [-1] * n)
    return g.node("Slice", [win, s, e, ax, st])


def _frag_overlay_win(g, src, keys, H, W):
    """Origin-safe windowed overlay for CONSTANT-size grids (supports flips)."""
    win = _window(g, src, H, W)
    copies = [_copy_win(g, win, k) for k in keys]
    merged = _combine_overlay(g, copies)
    return _unwindow(g, merged, H, W)


# --------------------------------------------------------------------------- #
# numpy simulators (faithful to the fragments above)                          #
# --------------------------------------------------------------------------- #
def _pad_open(grid):
    g = np.asarray(grid, int)
    h, w = g.shape
    op = np.ones((HEIGHT, WIDTH), np.int64)
    op[:h, :w] = (g == 0)
    return op


def _offs(conn8):
    if conn8:
        return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    return [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]


def _flood_outside(op, conn8, n_steps=None):
    reach = np.zeros((HEIGHT, WIDTH), np.int64)
    reach[0, :] = op[0, :]
    reach[-1, :] = op[-1, :]
    reach[:, 0] = op[:, 0]
    reach[:, -1] = op[:, -1]
    offs = _offs(conn8)
    steps = 0
    while True:
        nb = np.zeros_like(reach)
        for dy, dx in offs:
            ys0, ys1 = max(0, dy), HEIGHT + min(0, dy)
            xs0, xs1 = max(0, dx), WIDTH + min(0, dx)
            yd0, yd1 = max(0, -dy), HEIGHT + min(0, -dy)
            xd0, xd1 = max(0, -dx), WIDTH + min(0, -dx)
            nb[yd0:yd1, xd0:xd1] += reach[ys0:ys1, xs0:xs1]
        new = np.minimum(nb, op)
        steps += 1
        if np.array_equal(new, reach):
            steps -= 1
            break
        reach = new
        if n_steps is not None and steps >= n_steps:
            break
    return reach, steps


def _enclosed(grid, conn8):
    op = _pad_open(grid)
    reach, _ = _flood_outside(op, conn8)
    h, w = np.asarray(grid).shape
    return ((op == 1) & (reach == 0))[:h, :w]


def _sim_fill_enc(grid, dst, conn8):
    g = np.asarray(grid, int).copy()
    g[_enclosed(g, conn8)] = dst
    return g


def _flood_seed(grid, seed, conn8, n_steps=None):
    g = np.asarray(grid, int)
    h, w = g.shape
    medium = (g == 0)
    reach = (g == seed)
    offs = _offs(conn8)
    allowed = medium | reach
    steps = 0
    while True:
        nb = np.zeros((h, w), np.int64)
        ri = reach.astype(np.int64)
        for dy, dx in offs:
            ys0, ys1 = max(0, dy), h + min(0, dy)
            xs0, xs1 = max(0, dx), w + min(0, dx)
            yd0, yd1 = max(0, -dy), h + min(0, -dy)
            xd0, xd1 = max(0, -dx), w + min(0, -dx)
            nb[yd0:yd1, xd0:xd1] += ri[ys0:ys1, xs0:xs1]
        new = (nb > 0) & allowed
        steps += 1
        if np.array_equal(new, reach):
            steps -= 1
            break
        reach = new
        if n_steps is not None and steps >= n_steps:
            break
    return reach, steps


def _sim_fill_seed(grid, seed, conn8):
    g0 = np.asarray(grid, int)
    g = g0.copy()
    reach, _ = _flood_seed(g0, seed, conn8)
    g[reach & (g0 == 0)] = seed
    return g


def _np_transform(a, key):
    if key == "id":
        return a
    if key == "T":
        return a.T
    if key == "fliplr":
        return a[:, ::-1]
    if key == "flipud":
        return a[::-1, :]
    if key == "rot180":
        return a[::-1, ::-1]
    if key == "antiT":
        return a[::-1, ::-1].T
    raise ValueError(key)


def _sim_overlay(a, keys):
    copies = [np.asarray(_np_transform(a, k)) for k in keys]
    if any(c.shape != a.shape for c in copies):
        return None, False
    arr = np.stack(copies, axis=0)
    anynz = (arr != 0).any(axis=0)
    mxcol = arr.max(axis=0)
    masked = np.where(arr != 0, arr, 99)
    mncol = masked.min(axis=0)
    valid = bool(((~anynz) | (mxcol == mncol)).all())
    return np.where(anynz, mxcol, 0), valid


# --------------------------------------------------------------------------- #
# task pairs                                                                   #
# --------------------------------------------------------------------------- #
def _filter(lst):
    out = []
    for e in lst:
        a = np.array(e["input"], int)
        b = np.array(e["output"], int)
        if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
            continue
        if max(a.shape) > HEIGHT or max(b.shape) > WIDTH:
            continue
        out.append((a, b))
    return out


def _split_pairs(ex):
    """(probe, allp): probe = train+test (cheap detection); allp = everything the
    grader scores (full exact validation)."""
    probe = _filter(ex.get("train", []) + ex.get("test", []))
    allp = probe + _filter(ex.get("arc-gen", []))
    return probe, allp


# --------------------------------------------------------------------------- #
# build a model from a validated spec                                          #
# --------------------------------------------------------------------------- #
def _build(spec):
    kind, params, geom, cmap = spec
    g = _G()
    cur = "input"
    if kind == "fillenc":
        dst, nsteps, conn8 = params
        cur = _frag_fill_enc(g, cur, dst, nsteps, conn8)
    elif kind == "fillseed":
        seed, nsteps, conn8 = params
        cur = _frag_fill_seed(g, cur, seed, nsteps, conn8)
    elif kind == "overlayF":
        cur = _frag_overlay_full(g, cur, params)
    elif kind == "overlayW":
        keys, H, W = params
        cur = _frag_overlay_win(g, cur, keys, H, W)
    else:
        raise ValueError(kind)

    if geom == "T":
        cur = g.node("Transpose", [cur], perm=[0, 1, 3, 2])

    if cmap is not None and not _identity_map(cmap):
        cur = _frag_recolor(g, cur, cmap)

    g.nodes[-1].output[0] = "output"
    return _model(g.nodes, g.inits)


def _rank(spec):
    kind, params, geom, cmap = spec
    nsteps = params[1] if kind in ("fillenc", "fillseed") else 0
    nops = 1 + (1 if geom == "T" else 0) + (0 if cmap is None or _identity_map(cmap) else 1)
    rec = 0 if cmap is None or _identity_map(cmap) else (1 if _is_bijection(cmap) else 3)
    return (nops, nsteps, rec)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def _geom_out(s, geom):
    return s if geom == "id" else s.T


def candidates(ex):
    probe, allp = _split_pairs(ex)
    if not allp:
        return []
    # never a pure-identity task
    if all(a.shape == b.shape and np.array_equal(a, b) for a, b in allp):
        return []

    in_shapes = {a.shape for a, _ in allp}
    same_io = all(a.shape == b.shape for a, b in allp)
    all_square = all(a.shape[0] == a.shape[1] for a, _ in allp)
    const_size = len(in_shapes) == 1
    H0, W0 = next(iter(in_shapes)) if const_size else (0, 0)
    pcolors = sorted({int(v) for a, _ in probe for v in np.unique(a) if v != 0})

    specs = []        # validated (kind, params, geom, cmap)
    seen = set()
    _nstep_cache = {}

    def _full_cmap(sim_fn, geom):
        """sim_fn(a)->structural grid; apply geom, infer one map over ALL pairs."""
        outs = []
        for a, b in allp:
            so = _geom_out(sim_fn(a), geom)
            if so.shape != b.shape:
                return None
            outs.append(so)
        return _consistent_map(list(zip(outs, [b for _, b in allp])))

    def add_fill(kind, params, geom, sim_fn):
        cmap = _full_cmap(sim_fn, geom)
        if cmap is None:
            return
        key = (kind, params, geom, tuple(cmap))
        if key in seen:
            return
        seen.add(key)
        specs.append((kind, params, geom, cmap))

    # ------- enclosed-hole flood-fill (+ transpose) (+ recolor) ------------- #
    for conn8 in (False, True):
        enc_probe = [_enclosed(a, conn8) for a, _ in probe]
        if not any(m.any() for m in enc_probe):
            continue
        for geom in ("id", "T"):
            # detect (on probe) the single colour holes become in the output
            cols, ok = set(), True
            for (a, b), enc in zip(probe, enc_probe):
                if geom == "id":
                    if b.shape != a.shape:
                        ok = False
                        break
                    bv = b
                else:
                    if b.shape != (a.shape[1], a.shape[0]):
                        ok = False
                        break
                    bv = b.T
                if enc.any():
                    cols.update(int(v) for v in np.unique(bv[enc]))
            if not ok or len(cols) != 1:
                continue
            dst = cols.pop()
            if dst == 0:
                continue
            # cheap probe prune
            pc = _consistent_map([(_geom_out(_sim_fill_enc(a, dst, conn8), geom), b)
                                  for a, b in probe])
            if pc is None:
                continue
            if all(np.array_equal(_sim_fill_enc(a, dst, conn8), a) for a, _ in probe):
                continue
            if conn8 not in _nstep_cache:
                st = 0
                for a, _ in allp:
                    _, s = _flood_outside(_pad_open(a), conn8)
                    st = max(st, s)
                # extra flood steps are idempotent (reach is a monotone fixed point),
                # so a generous buffer only helps held-out grids that need more steps.
                _nstep_cache[conn8] = min(60, max(1, st) + 5)
            n_steps = _nstep_cache[conn8]
            add_fill("fillenc", (dst, n_steps, conn8), geom,
                     lambda a, d=dst, c=conn8: _sim_fill_enc(a, d, c))

    # ------- seed flood-fill (+ transpose) (+ recolor) --------------------- #
    for conn8 in (False, True):
        for seed in pcolors:
            sims_p = [_sim_fill_seed(a, seed, conn8) for a, _ in probe]
            if all(np.array_equal(s, a) for s, (a, _) in zip(sims_p, probe)):
                continue                            # nothing flooded on probe
            for geom in ("id", "T"):
                if geom == "id" and not same_io:
                    continue
                if geom == "T" and not all(
                        b.shape == (a.shape[1], a.shape[0]) for a, b in allp):
                    continue
                pc = _consistent_map([(_geom_out(s, geom), b)
                                      for s, (_, b) in zip(sims_p, probe)])
                if pc is None:
                    continue
                add_fill("fillseed", (seed, conn8), geom,
                         lambda a, s=seed, c=conn8: _sim_fill_seed(a, s, c))

    # ------- symmetric overlay (+ recolor) --------------------------------- #
    def add_overlay(kind, params, keys):
        outs, ok, changed = [], True, False
        for a, b in allp:
            o, v = _sim_overlay(a, keys)
            if not v or o.shape != b.shape:
                ok = False
                break
            if not np.array_equal(o, a):
                changed = True
            outs.append(o)
        if not ok or not changed:
            return
        cmap = _consistent_map(list(zip(outs, [b for _, b in allp])))
        if cmap is None:
            return
        key = (kind, params, "id", tuple(cmap))
        if key in seen:
            return
        seen.add(key)
        specs.append((kind, params, "id", cmap))

    if same_io:
        if const_size:
            keysets = [["id", "fliplr"], ["id", "flipud"], ["id", "rot180"],
                       ["id", "fliplr", "flipud", "rot180"]]
            if H0 == W0:
                keysets += [["id", "T"], ["id", "antiT"],
                            ["id", "fliplr", "flipud", "rot180", "T", "antiT"]]
            for keys in keysets:
                # cheap probe prune first
                good = True
                for a, _ in probe:
                    o, v = _sim_overlay(a, keys)
                    if not v:
                        good = False
                        break
                if good:
                    add_overlay("overlayW", (tuple(keys), H0, W0), keys)
        elif all_square:
            add_overlay("overlayF", ("id", "T"), ["id", "T"])

    if not specs:
        return []

    specs.sort(key=_rank)
    out, names = [], set()
    for spec in specs[:_MAX_YIELD]:
        kind, params, geom, cmap = spec
        rc = "" if cmap is None or _identity_map(cmap) else \
            ("_recB" if _is_bijection(cmap) else "_recC")
        gm = "_T" if geom == "T" else ""
        if kind == "fillenc":
            tag = f"fillenc{'8' if params[2] else '4'}_c{params[0]}"
        elif kind == "fillseed":
            tag = f"fillseed{'8' if params[1] else '4'}_c{params[0]}"
        elif kind == "overlayW":
            tag = "overlayW_" + "_".join(params[0])
        else:
            tag = "overlayF_" + "_".join(params)
        nm = f"sc_{tag}{gm}{rc}"
        if nm in names:
            continue
        names.add(nm)
        try:
            out.append((nm, _build(spec)))
        except Exception:
            continue
    return out
