"""family_crk10_6 -- hardest remaining unsolved tasks, slice U[6::8].

Assigned tasks: [54, 91, 143, 173, 219, 363].

Every rule below was reverse-engineered EXACTLY from all train pairs and then
validated (numpy mirror) on train+test+arc-gen before any ONNX graph is emitted.
The grader compares the FULL [1,10,30,30] one-hot: cells outside the true HxW grid
are all-zero in every channel, so "variable output size" == place content at the
top-left origin and zero everything outside the HxW box.

--------------------------------------------------------------------------------
TASK 91  (9xN -> small, SOLVED)  "crop the stud-frame to its bounding box"
  The grid holds one rectangle "frame": four 8 corners, and colour-5 studs down
  the two vertical edges (top/bottom edges are empty apart from the corners),
  plus scattered noise 8s outside.  Output = crop of that frame.  The colour-5
  studs give the exact left/right columns and the interior rows, but miss the two
  corner rows (top & bottom).  So: content = colour-5 mask dilated by 1 vertically;
  crop input to bbox(content), re-anchored top-left (dyncrop MatMul trick).
  Verified EXACT on all 266 pairs.

TASK 54  (30x30, "cross-flood + glyph stamp")  NOT EMITTED
  Rectangular fill-panels (colour A) on a background, each holding one seed cell.
  Output stamps a 3x3 checker core at each seed (corners+centre=seedcol, edges=
  crosscol) and floods a full horizontal+vertical cross (crosscol) through the seed
  across the panel, clipped by the panel border.  The role colours (fill / seed /
  cross / bg) differ per example (train0: 1/3/2/8; train1: 2/4/3/1), so which colour
  is "cross" vs "seed" must be inferred from the data -> data-dependent colour roles
  plus per-panel directional flood.  Too many data-dependent branches for an exact
  static graph.

TASK 143 (10x10, "recolour the shape-twin of the bracketed key")  NOT EMITTED
  A fixed colour-5 L-bracket frames a "key" object in the top-left; the grid holds
  several objects, exactly one of which is CONGRUENT (same cell-shape) to the key.
  That twin is recoloured to 5.  Size alone does not discriminate (many equal-size
  objects), so it needs a shape cross-correlation against a DATA-dependent key mask
  (== a data-dependent conv kernel / im2col) -> banned in opset-10.

TASK 173 (var, "complete the partial glyphs")  NOT EMITTED
  Legend glyphs = a centre-colour cell + surrounding arm cells (plus / X / triple).
  Partial copies appear as either a lone centre (arms missing) or the arms (centre
  missing); each is completed to the full glyph.  Glyph shape+colours are read from
  the legend and DIFFER per example (a colour is a centre in one pair, an arm in
  another), and there are multiple glyph types -> data-dependent stamp kernel(s).

TASK 219 (15x10, "extend truncated bars/patterns to the right wall")  NOT EMITTED
  A template block (reaches the right wall) and truncated copies; each copy is
  extended rightward with colour 1.  Verified numpy mirror reproduces 245/265 pairs
  but NOT all: the vertical alignment of the template onto each object is not a
  single top/bottom rule -- the diagonal-ramp objects (train) and the periodic
  objects (arc-gen) align differently, and the ramp objects are not translates of
  the template at all.  No exact unified rule found -> would fail arc-gen, score 0.

TASK 363 (10x10, "replicate the 2-marker on every matching 5-texture motif")
  NOT EMITTED.  A field of colour-5 texture; a 2-marker (diamond / bar / cluster)
  annotates one motif; the same annotation is stamped on every other location whose
  surrounding 5-texture matches (self-correlation of the texture with a data-
  dependent template) -> banned data-dependent conv.
--------------------------------------------------------------------------------
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
_CBIG = 1000.0


# --------------------------------------------------------------------------- #
# tiny graph accumulator (same style as family_dyncrop)                        #
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
        self.inits.append(oh.make_tensor(n, INT64, list(dims), [int(v) for v in np.asarray(vals).ravel()]))
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


def _consts(g):
    g.rowidx = g.f([1, 1, H, 1], list(range(H)))
    g.colidx = g.f([1, 1, 1, W], list(range(W)))
    g.half = g.f([1, 1, 1, 1], [0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])
    g.cbig = g.f([1, 1, 1, 1], [_CBIG])


def _shift(g, x, dr, dc):
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0,
             pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H, max(-dc, 0) + W])
    ax = g.i64([2, 3])
    return g.nd("Slice", [p, st, en, ax])


def _finish_crop(g, content):
    """Crop input to bbox(content), re-anchored at top-left (MatMul selection)."""
    rowidx, colidx = g.rowidx, g.colidx
    half, one, cbig = g.half, g.one, g.cbig

    rowhas = g.nd("ReduceMax", [content], axes=[3], keepdims=1)
    colhas = g.nd("ReduceMax", [content], axes=[2], keepdims=1)

    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    bbox_h = g.nd("Add", [g.nd("Sub", [maxrow, minrow]), one])
    bbox_w = g.nd("Add", [g.nd("Sub", [maxcol, mincol]), one])

    diff_c = g.nd("Sub", [g.nd("Add", [colidx, mincol]), rowidx])
    match_c = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_c]), half])], to=F)
    trunc_c = g.nd("Cast", [g.nd("Less", [colidx, bbox_w])], to=F)
    Scol = g.nd("Mul", [match_c, trunc_c])

    diff_r = g.nd("Sub", [colidx, g.nd("Add", [rowidx, minrow])])
    match_r = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diff_r]), half])], to=F)
    trunc_r = g.nd("Cast", [g.nd("Less", [rowidx, bbox_h])], to=F)
    Srow = g.nd("Mul", [match_r, trunc_r])

    shift1 = g.nd("MatMul", ["input", Scol])
    g.nd("MatMul", [Srow, shift1], "output")


def _content_color_vdilate(g, c):
    """Colour-c mask, dilated by 1 vertically (Max with up/down shifts)."""
    m = g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])  # [1,1,30,30]
    up = _shift(g, m, -1, 0)
    dn = _shift(g, m, 1, 0)
    return g.nd("Max", [m, up, dn])


def build_crop91():
    g = _G()
    _consts(g)
    content = _content_color_vdilate(g, 5)
    _finish_crop(g, content)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX numerics for detection / anti-overfit)     #
# --------------------------------------------------------------------------- #
def _ref91(a):
    ys, xs = np.where(a == 5)
    if ys.size == 0:
        return None
    r0, r1 = ys.min() - 1, ys.max() + 1
    c0, c1 = xs.min(), xs.max()
    if r0 < 0 or r1 >= a.shape[0]:
        return None
    return a[r0:r1 + 1, c0:c1 + 1].copy()


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


def _matches(prs, fn):
    if not prs:
        return False
    for a, b in prs:
        o = fn(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def _emit(out, name, builder):
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    out.append((name, m))


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    # TASK 91: crop stud-frame (colour-5 vertical studs) to its bbox.
    if _matches(prs, _ref91):
        _emit(out, "crk10_6_crop91", build_crop91)

    return out
