"""Per-color pipeline family.

Some tasks are realised by treating each colour CHANNEL independently with an
origin-safe transform and recombining.  The realizable, generalizing subset on
top-left/zero-padded variable-size grids is:

  * GLOBAL per-colour map (pointwise recolor): output colour = map[input colour],
    a single map consistent over every pixel and every example.  This covers
        - relabel / colour permutation        (bijection  -> cheap Gather)
        - swap two colours                     (bijection)
        - remove colour X  (X -> background 0) (non-bijection -> 1x1 Conv)
        - keep only colour X (others -> 0)     (non-bijection -> 1x1 Conv)
    A no-bias linear channel op leaves zero-padding all-zero, so it is padding
    safe, and being pointwise it is origin-anchored for any grid size.

  * GLOBAL transpose, optionally composed with a per-colour map.  Transpose
    (perm [0,1,3,2]) keeps content at the top-left origin, so it generalizes for
    variable grid sizes (unlike flip / rot180 / translate, which send content to
    the far edge for grids < 30x30 and are therefore NOT emitted here).  This is
    the degenerate "every channel gets the same spatial transform" pipeline.

  * GENUINE per-channel split: on SQUARE grids, some colour channels kept as-is
    and others transposed (the rest dropped), recombined with channel masks.
    This is a true different-transform-per-colour pipeline; it is only proposed
    when it does NOT collapse to a single global op.

Detection infers parameters from all available train+test+arc-gen pairs; the
harness re-validates EXACT equality on the full grader set, so over-proposing is
safe (wrong guesses are rejected).  Data-dependent / relational recolorings
(minority colour, frequency rank, cyclic permutation of the *present* colours,
predicate-gated swaps whose trigger varies per example) are NOT a fixed static
map and are intentionally skipped.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64


# --------------------------------------------------------------------------- #
# tiny node / initializer accumulator with auto-unique tensor names
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


def _recolor(g, src, cmap, out=None):
    """Append a per-colour recolor on `src`: Gather (10 params) if the map is a
    bijection, otherwise a 1x1 Conv (100 params) which can merge channels."""
    if _is_bijection(cmap):
        inv = [0] * CHANNELS              # output[:, j] = input[:, inv[j]]
        for i, o in enumerate(cmap):
            inv[o] = i
        idx = g.init(INT64, [CHANNELS], inv)
        return g.node("Gather", [src, idx], out, axis=1)
    weights = [0.0] * (CHANNELS * CHANNELS)   # [O, I, 1, 1]
    for i, o in enumerate(cmap):
        weights[o * CHANNELS + i] = 1.0
    w = g.init(DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], weights)
    return g.node("Conv", [src, w], out, kernel_shape=[1, 1], pads=[0, 0, 0, 0])


# --------------------------------------------------------------------------- #
# graph builders
# --------------------------------------------------------------------------- #
def _build_recolor(cmap):
    g = _G()
    _recolor(g, "input", cmap, "output")
    return _model(g.nodes, g.inits)


def _build_transpose(cmap):
    g = _G()
    if _identity_map(cmap):
        g.node("Transpose", ["input"], "output", perm=[0, 1, 3, 2])
    else:
        t = g.node("Transpose", ["input"], perm=[0, 1, 3, 2])
        _recolor(g, t, cmap, "output")
    return _model(g.nodes, g.inits)


def _build_split(keep, tch):
    """Genuine per-channel pipeline: channels in `keep` taken from input as-is,
    channels in `tch` taken from the transpose, all others dropped (zero).
    output = input * keepmask + transpose(input) * tmask  (channel masks)."""
    g = _G()
    km = [1.0 if c in keep else 0.0 for c in range(CHANNELS)]
    keepmask = g.init(DATA_TYPE, [1, CHANNELS, 1, 1], km)
    a = g.node("Mul", ["input", keepmask])
    t = g.node("Transpose", ["input"], perm=[0, 1, 3, 2])
    tm = [1.0 if c in tch else 0.0 for c in range(CHANNELS)]
    tmask = g.init(DATA_TYPE, [1, CHANNELS, 1, 1], tm)
    b = g.node("Mul", [t, tmask])
    g.node("Add", [a, b], "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# detection helpers
# --------------------------------------------------------------------------- #
def _pairs(ex):
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


def _consistent_map(grids):
    """Single per-colour map over (X, Y) pairs of equal shape, or None."""
    m = {}
    for X, Y in grids:
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


def _onehot(g):
    H, W = g.shape
    oh3 = np.zeros((CHANNELS, H, W), dtype=bool)
    for c in range(CHANNELS):
        oh3[c] = (g == c)
    return oh3


def _detect_split(prs):
    """On square grids, find a per-channel transform in {id, T, drop} that is a
    GENUINE mix (cannot be a single global op).  Returns (keep, tch) or None."""
    # transpose only makes sense recombined with identity when every grid is square
    if not all(a.shape == b.shape and a.shape[0] == a.shape[1] for a, b in prs):
        return None
    keep, tch, drop = set(), set(), set()
    for c in range(CHANNELS):
        masks = [(_onehot(a)[c], _onehot(b)[c]) for a, b in prs]
        is_id = all(np.array_equal(im, om) for im, om in masks)
        is_t = all(np.array_equal(im.T, om) for im, om in masks)
        is_zero = all(not om.any() for _, om in masks)
        # informative channels (non-empty input or output) drive the choice
        nonempty = any(im.any() or om.any() for im, om in masks)
        if is_id and is_t:
            # symmetric / empty -> non-informative; default to identity
            keep.add(c)
        elif is_id:
            keep.add(c)
        elif is_t:
            tch.add(c)
        elif is_zero:
            drop.add(c)
        else:
            return None
        if not nonempty and c in keep:
            keep.discard(c)  # truly empty channel: leave out of masks (still 0)
    # require a genuine mix: at least one transposed channel AND at least one
    # identity channel that is actually informative, else a global op suffices.
    informative_keep = False
    for c in keep:
        masks = [(_onehot(a)[c], _onehot(b)[c]) for a, b in prs]
        if any(im.any() for im, _ in masks) and not all(np.array_equal(im, im.T) for im, _ in masks):
            informative_keep = True
            break
    if not tch or not informative_keep:
        return None
    return keep, tch


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    # pure identity is not this family
    if all(a.shape == b.shape and np.array_equal(a, b) for a, b in prs):
        return []

    out = []

    # ---- global transpose (+ optional per-colour recolor) -------------------
    try:
        if all(b.shape == (a.shape[1], a.shape[0]) for a, b in prs) \
                and not all(np.array_equal(a, b) for a, b in prs):
            cmap = _consistent_map([(a.T, b) for a, b in prs])
            if cmap is not None:
                nm = "transpose" if _identity_map(cmap) else "transpose_recolor"
                out.append((nm, _build_transpose(cmap)))
    except Exception:
        pass

    # ---- global per-colour map (same shape): recolor / swap / remove / keep --
    try:
        if all(a.shape == b.shape for a, b in prs):
            cmap = _consistent_map(prs)
            if cmap is not None and not _identity_map(cmap):
                out.append(("colormap", _build_recolor(cmap)))
    except Exception:
        pass

    # ---- genuine per-channel split (keep some colours, transpose others) -----
    try:
        split = _detect_split(prs)
        if split is not None:
            keep, tch = split
            out.append(("split_transpose", _build_split(keep, tch)))
    except Exception:
        pass

    # de-dup by name (keep first)
    seen, uniq = set(), []
    for name, model in out:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, model))
    return uniq
