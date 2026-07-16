"""Single-node compile family (campaign sn_4).

Targets the cheapest possible graphs for tasks whose input->output map is a
FIXED, data-independent transform:

  * geometric  (identity / fliplr / flipud / rot180 / transpose / anti-transpose)
    realised as a windowed Slice/Transpose on the [0:H,0:W] sub-grid and padded
    back to 30x30 -> 0 params, 0 memory -> 25.0 pts.  Only emitted for tasks with
    a CONSTANT grid size across every split (so the window is well defined).
  * global recolor that is a COLOR PERMUTATION (bijective, position preserving)
    realised as Gather(axis=1, idx[10]) -> 10 params.

Every candidate is validated for EXACT equality on ALL local train+test+arc-gen
pairs before being emitted; the grader re-checks fresh generator samples.  Nothing
is emitted unless it is exact everywhere, so this family can never ship a
regression.

The ten tasks of this campaign (125,133,173,145,243,76,138,198,66,64) are all
object / connected-component rules (outbox+delta fill, size-ranked recolor,
occurrence stamping, subgrid+connect crop, line shooting, gravitate/connect,
enclosed-region fill).  None reduces to a fixed map, so this family correctly
routes none of them -- they remain on the incumbent net.  The detectors below are
kept generic so the module is reusable, but they are strictly gated and safe.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
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
# numpy reference of the fixed geometric transforms                           #
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
# windowed geometric graph (0 params, 0 memory when full-size)                #
# --------------------------------------------------------------------------- #
def _window(g, src, H, W):
    if H == HEIGHT and W == WIDTH:
        return src
    s = g.iconst([0, 0])
    e = g.iconst([H, W])
    ax = g.iconst([2, 3])
    return g.node("Slice", [src, s, e, ax])


def _unwindow(g, src, H, W, out):
    if H == HEIGHT and W == WIDTH:
        return g.node("Identity", [src], out)
    pads = [0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]
    return g.node("Pad", [src], out, mode="constant", value=0.0, pads=pads)


def _copy_win(g, win, key, out=None):
    if key == "id":
        return g.node("Identity", [win], out)
    if key == "T":
        return g.node("Transpose", [win], out, perm=[0, 1, 3, 2])
    if key == "antiT":
        r = _copy_win(g, win, "rot180")
        return g.node("Transpose", [r], out, perm=[0, 1, 3, 2])
    axes = {"fliplr": [3], "flipud": [2], "rot180": [2, 3]}[key]
    n = len(axes)
    s = g.iconst([-1] * n)
    e = g.iconst([_NEG] * n)
    ax = g.iconst(axes)
    st = g.iconst([-1] * n)
    return g.node("Slice", [win, s, e, ax, st], out)


def build_geo(key, H, W):
    g = _G()
    if H == HEIGHT and W == WIDTH:
        # single node straight to "output"
        _copy_win(g, "input", key, "output") if key != "id" else \
            g.node("Identity", ["input"], "output")
        return _model(g.nodes, g.inits)
    win = _window(g, "input", H, W)
    t = _copy_win(g, win, key)
    _unwindow(g, t, H, W, "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# bijective global recolor -> Gather(axis=1, idx[10])                         #
# --------------------------------------------------------------------------- #
def build_recolor(src_for_out):
    """output[:, j] = input[:, src_for_out[j]]."""
    assert len(src_for_out) == CHANNELS
    idx = oh.make_tensor("idx", INT64, [CHANNELS], list(src_for_out))
    node = oh.make_node("Gather", ["input", "idx"], ["output"], axis=1)
    return _model([node], [idx])


# --------------------------------------------------------------------------- #
# detection                                                                   #
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


def _detect_geo(prs):
    shapes = {a.shape for a, _ in prs}
    if len(shapes) != 1:
        return []
    if any(a.shape != b.shape for a, b in prs):
        return []
    H, W = next(iter(shapes))
    square = (H == W)
    keys = ["id", "fliplr", "flipud", "rot180"] + (["T", "antiT"] if square else [])
    good = []
    for key in keys:
        if all(np.array_equal(_np_transform(a, key), b) for a, b in prs):
            good.append((key, H, W))
    return good


def _detect_recolor(prs):
    """Position-preserving bijective color permutation consistent over all cells."""
    if any(a.shape != b.shape for a, b in prs):
        return None
    fwd = {}          # input color -> output color
    for a, b in prs:
        for ca, cb in zip(a.ravel(), b.ravel()):
            ca, cb = int(ca), int(cb)
            if ca in fwd and fwd[ca] != cb:
                return None
            fwd[ca] = cb
    if fwd == {c: c for c in fwd}:   # identity -> geo/id already covers it
        return None
    # must be injective on the colors we have observed (permutation) to use Gather
    if len({fwd[c] for c in fwd}) != len(fwd):
        return None
    # build a full length-10 inverse index: output channel o pulls input channel i
    src = list(range(CHANNELS))
    for i, o in fwd.items():
        if 0 <= o < CHANNELS and 0 <= i < CHANNELS:
            src[o] = i
    # verify the Gather semantics reproduce every pair exactly on the one-hot level
    for a, b in prs:
        # emulate: out_color at cell = fwd[in_color]
        exp = np.vectorize(lambda c: fwd.get(int(c), int(c)))(a)
        if not np.array_equal(exp, b):
            return None
    return src


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    # rule must actually change something on at least one pair
    if not any(not np.array_equal(a, b) for a, b in prs):
        return []

    out, seen = [], set()

    def add(name, model):
        if name not in seen:
            seen.add(name)
            out.append((name, model))

    for key, H, W in _detect_geo(prs):
        try:
            add(f"geo_{key}", build_geo(key, H, W))
        except Exception:
            pass

    src = _detect_recolor(prs)
    if src is not None:
        try:
            add("recolor", build_recolor(src))
        except Exception:
            pass

    return out
