"""Fixed-size symmetry completion / overlay family.

This family is the *origin-correct* counterpart of ``family_symmetry`` for grids
whose size is CONSTANT across every split.  Naive 30x30 reverse-slice flips push
content of a <30 grid to the far edge (the padding gotcha), so ``family_symmetry``
can only emit them as attempts that the harness rejects for sub-30 grids.  Here we
KNOW the exact (H, W) because it never varies, so we build *windowed* flips that
operate on the [0:H, 0:W] sub-grid and zero-pad back to 30x30 -> the result stays
anchored at the top-left and is exact for the constant size.

Two rules, both detected from the train/test/arc-gen pairs and validated for EXACT
equality before being emitted (the grader's gate):

  * occluder / hole fill   output[i,j] = mirror[i,j]  if input[i,j] == C
                           output[i,j] = input[i,j]   otherwise
    where ``C`` is an occluder colour (often background 0) and ``mirror`` is one
    windowed transform (fliplr / flipud / rot180 / transpose / anti-transpose).
    Realised as  Where( input[:,C], mirror, input )  -- one mirror intermediate.

  * symmetric overlay      output = one-hot OR of input with windowed transformed
    copies of itself (4-fold mirror / full D4).  Colour channels 1..9 are combined
    with ``Max`` (a cell lights up if any copy carries that colour); channel 0 is
    the product (background only where every copy is background) so the output
    stays a valid one-hot.

Why origin-safe: every windowed transform maps the HxW window bijectively onto
itself and leaves the padding all-zero, and ``Conv``-free pointwise / Slice / Pad /
Transpose ops preserve the origin.  We emit ONLY for constant-size tasks.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
_NEG = -(1 << 31)


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


# --------------------------------------------------------------------------- #
# numpy reference of the windowed transforms (== applied to the raw HxW grid)  #
# --------------------------------------------------------------------------- #
def _np_transform(a, key):
    if key == "id":
        return a
    if key == "fliplr":
        return a[:, ::-1]
    if key == "flipud":
        return a[::-1, :]
    if key == "rot180":
        return a[::-1, ::-1]
    if key == "T":
        return a.T
    if key == "antiT":
        return a[::-1, ::-1].T
    raise ValueError(key)


# --------------------------------------------------------------------------- #
# window helpers: slice the [0:H,0:W] sub-grid, transform inside it, pad back.  #
# Working in the window keeps every intermediate HxW (not 30x30) -> far cheaper #
# for sub-30 grids while staying origin-anchored (Pad only adds trailing zeros).#
# --------------------------------------------------------------------------- #
def _window(g, src, H, W):
    """src [1,10,30,30] -> the real [1,10,H,W] sub-grid."""
    if H == HEIGHT and W == WIDTH:
        return src
    s = g.iconst([0, 0])
    e = g.iconst([H, W])
    ax = g.iconst([2, 3])
    return g.node("Slice", [src, s, e, ax])


def _unwindow(g, src, H, W, out):
    """[1,10,H,W] -> [1,10,30,30] (trailing zero-pad), bound to output name `out`."""
    if H == HEIGHT and W == WIDTH:
        return g.node("Identity", [src], out)
    pads = [0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]
    return g.node("Pad", [src], out, mode="constant", value=0.0, pads=pads)


def _copy_win(g, win, key):
    """Transform a [1,10,H,W] window in place (square required for T / antiT)."""
    if key == "id":
        return win
    if key == "T":
        return g.node("Transpose", [win], perm=[0, 1, 3, 2])
    if key == "antiT":
        r = _copy_win(g, win, "rot180")
        return g.node("Transpose", [r], perm=[0, 1, 3, 2])
    axes = {"fliplr": [3], "flipud": [2], "rot180": [2, 3]}[key]
    n = len(axes)
    s = g.iconst([-1] * n)                       # start at last index of each axis
    e = g.iconst([_NEG] * n)                     # ...down to (and incl.) index 0
    ax = g.iconst(axes)
    st = g.iconst([-1] * n)
    return g.node("Slice", [win, s, e, ax, st])


# --------------------------------------------------------------------------- #
# Rule 1: occluder / hole fill   Where(window[:,C], mirror, window)            #
# --------------------------------------------------------------------------- #
def build_fill(C, key, H, W):
    g = _G()
    win = _window(g, "input", H, W)
    mirror = _copy_win(g, win, key)
    s = g.iconst([C])
    e = g.iconst([C + 1])
    ax = g.iconst([1])
    chanC = g.node("Slice", [win, s, e, ax])         # [1,1,H,W] float 0/1
    cond = g.node("Cast", [chanC], to=BOOL)
    filled = g.node("Where", [cond, mirror, win])    # [1,10,H,W]
    _unwindow(g, filled, H, W, "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# Rule 2: symmetric overlay (one-hot OR of windowed copies)                    #
# --------------------------------------------------------------------------- #
def build_overlay(keys, H, W):
    g = _G()
    win = _window(g, "input", H, W)
    copies = [_copy_win(g, win, k) for k in keys]

    # colour channels 1..9 of each copy -> elementwise Max (logical OR)
    cs = g.iconst([1]); ce = g.iconst([CHANNELS]); cax = g.iconst([1])
    cols = [g.node("Slice", [c, cs, ce, cax]) for c in copies]
    colored = cols[0] if len(cols) == 1 else g.node("Max", cols)

    # channel 0 of each copy -> product (background only where ALL are background)
    bs = g.iconst([0]); be = g.iconst([1]); bax = g.iconst([1])
    bgs = [g.node("Slice", [c, bs, be, bax]) for c in copies]
    if len(bgs) == 1:
        bg = bgs[0]
    else:
        bg = bgs[0]
        for j in range(1, len(bgs)):
            bg = g.node("Mul", [bg, bgs[j]])

    merged = g.node("Concat", [bg, colored], axis=1)  # [1,10,H,W]
    _unwindow(g, merged, H, W, "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# numpy reference for the overlay (mirrors the ONNX Max/product exactly)        #
# --------------------------------------------------------------------------- #
def _overlay_ref(a, keys):
    copies = [np.asarray(_np_transform(a, k)) for k in keys]
    arr = np.stack(copies, axis=0)
    anynz = (arr != 0).any(axis=0)
    mxcol = arr.max(axis=0)
    masked = np.where(arr != 0, arr, 99)
    mncol = masked.min(axis=0)
    valid = bool(((~anynz) | (mxcol == mncol)).all())  # no colour conflicts
    out = np.where(anynz, mxcol, 0)
    return out, valid


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    # constant size across EVERY split (the family's precondition) -------------
    shapes = {a.shape for a, _ in prs}
    if len(shapes) != 1:
        return []
    if any(a.shape != b.shape for a, b in prs):
        return []
    H, W = next(iter(shapes))
    if not any(not np.array_equal(a, b) for a, b in prs):   # must change
        return []
    square = (H == W)

    keys_single = ["fliplr", "flipud", "rot180"] + (["T", "antiT"] if square else [])
    out, seen = [], set()

    def add(name, model):
        if name not in seen:
            seen.add(name)
            out.append((name, model))

    # --- Rule 1: occluder / hole fill (single windowed transform) ---------- #
    colors = sorted({int(v) for a, _ in prs for v in np.unique(a)})
    for C in colors:
        for key in keys_single:
            ok = True
            for a, b in prs:
                m = _np_transform(a, key)
                if m.shape != a.shape:
                    ok = False
                    break
                exp = np.where(a == C, m, a)
                if not np.array_equal(exp, b):
                    ok = False
                    break
            if ok:
                try:
                    add(f"fill_C{C}_{key}", build_fill(C, key, H, W))
                except Exception:
                    pass

    # --- Rule 2: symmetric overlay (multi-copy completion) ----------------- #
    keysets = [
        ("lr", ["id", "fliplr"]),
        ("ud", ["id", "flipud"]),
        ("rot180", ["id", "rot180"]),
        ("quad", ["id", "fliplr", "flipud", "rot180"]),
    ]
    if square:
        keysets += [
            ("diag", ["id", "T"]),
            ("antidiag", ["id", "antiT"]),
            ("d4", ["id", "fliplr", "flipud", "rot180", "T", "antiT"]),
        ]
    for tag, keys in keysets:
        ok = True
        for a, b in prs:
            o, valid = _overlay_ref(a, keys)
            if not valid or not np.array_equal(o, b):
                ok = False
                break
        if ok:
            try:
                add(f"overlay_{tag}", build_overlay(keys, H, W))
            except Exception:
                pass

    return out
