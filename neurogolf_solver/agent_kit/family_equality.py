"""Per-cell EQUALITY / two-halves comparison family (logic gates & merges).

Many ARC tasks present the input as TWO equal sub-grids (placed side-by-side or
stacked, optionally separated by a one/two-cell divider line) and ask for a
single half-sized output whose every cell is a boolean/equality function of the
two corresponding sub-grid cells:

    output[i,j] = C            if  f( left[i,j]!=bg , right[i,j]!=bg )      (a "gate")
                  background    otherwise

with f one of and / or / xor / nand / nor / xnor / left-and-not-right /
right-and-not-left.  A second sub-rule is the colour-preserving MERGE
(overlay one half on top of the other, keeping the original colours):

    output[i,j] = left[i,j]    if left[i,j] != bg  else  right[i,j]   (and the
    symmetric right-over-left), used when the two halves never disagree.

Origin safety (CONTEXT.md padding gotcha)
-----------------------------------------
The split position depends on the grid size, so the rule is only well defined
when the geometry is CONSTANT across every split (it is, for these template
tasks).  We therefore emit ONLY for tasks whose input shape AND output shape are
identical across all train+test+arc-gen pairs.  Knowing the exact (H,W) we slice
the two fixed sub-windows, combine them, and zero-pad the half-sized result back
to 30x30 -> the content stays anchored at the top-left and every padding cell is
all-zero, so the one-hot contract holds for grids of any (fixed) size.

Realisation (opset 10, origin-safe, cheap)
------------------------------------------
Per side we take ONE combined Slice over (channel 1..9, rows, cols) and a
ReduceSum over the colour axis to get a {0,1} "active" plane [1,1,h,w] -- the
full [1,10,h,w] half is never materialised for gates.  The gate is one/two
pointwise ops (Mul / Max / Abs / Sub), the one-hot is rebuilt by broadcasting the
{0,1} result against two tiny [1,10,1,1] colour vectors, and a single Pad anchors
it.  Intermediates are only h*w-sized, so cost is small -> high score.

Detection mirrors the ONNX semantics exactly and validates EVERY available pair
(the grader's gate); wrong hypotheses are dropped before scoring.
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


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0
        self._one = None

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def fconst(self, dims, vals):
        nm = self.name("f")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         [float(v) for v in vals]))
        return nm

    def one(self):
        """Single shared scalar 1.0 [1,1,1,1] initializer."""
        if self._one is None:
            self._one = self.fconst([1, 1, 1, 1], [1.0])
        return self._one

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# windows                                                                     #
# --------------------------------------------------------------------------- #
def _active(g, r0, r1, c0, c1):
    """{0,1} non-background plane [1,1,h,w] of the sub-window.

    Within a (fully real) sub-window every cell is one-hot, so background channel
    0 is 1 on background cells and 0 on coloured cells; ``1 - ch0`` is therefore
    the non-background indicator.  Slicing only channel 0 keeps the intermediate
    [1,1,h,w] (vs [1,9,h,w]) -> cheaper."""
    s = g.iconst([0, r0, c0])
    e = g.iconst([1, r1, c1])
    ax = g.iconst([1, 2, 3])
    ch0 = g.node("Slice", ["input", s, e, ax])          # [1,1,h,w] 1 where bg
    return g.node("Sub", [g.one(), ch0])                # [1,1,h,w] in {0,1}


def _half(g, r0, r1, c0, c1):
    """Full [1,10,h,w] sub-window (all channels)."""
    s = g.iconst([r0, c0])
    e = g.iconst([r1, c1])
    ax = g.iconst([2, 3])
    return g.node("Slice", ["input", s, e, ax])


def _gate(g, la, ra, fname):
    one = g.one()
    if fname == "and":
        return g.node("Mul", [la, ra])
    if fname == "or":
        return g.node("Max", [la, ra])
    if fname == "xor":
        return g.node("Abs", [g.node("Sub", [la, ra])])
    if fname == "nand":
        return g.node("Sub", [one, g.node("Mul", [la, ra])])
    if fname == "nor":
        return g.node("Sub", [one, g.node("Max", [la, ra])])
    if fname == "xnor":
        return g.node("Sub", [one, g.node("Abs", [g.node("Sub", [la, ra])])])
    if fname == "dlr":                       # left and not right
        return g.node("Sub", [la, g.node("Mul", [la, ra])])
    if fname == "drl":                       # right and not left
        return g.node("Sub", [ra, g.node("Mul", [la, ra])])
    raise ValueError(fname)


def _pad_out(g, win, oh_, ow):
    """[1,10,oh_,ow] -> output [1,10,30,30] (trailing zero pad, origin anchored)."""
    if oh_ == HEIGHT and ow == WIDTH:
        g.node("Identity", [win], "output")
        return
    pads = [0, 0, 0, 0, 0, 0, HEIGHT - oh_, WIDTH - ow]
    g.node("Pad", [win], "output", mode="constant", value=0.0, pads=pads)


# --------------------------------------------------------------------------- #
# builders                                                                     #
# --------------------------------------------------------------------------- #
def build_gate(L, R, oh_, ow, fname, C):
    """Fixed-colour gate. L,R = (r0,r1,c0,c1) sub-window rectangles."""
    g = _G()
    la = _active(g, *L)
    ra = _active(g, *R)
    res = _gate(g, la, ra, fname)                       # [1,1,oh_,ow] in {0,1}
    notres = g.node("Sub", [g.one(), res])
    vecC = g.fconst([1, CHANNELS, 1, 1], [1.0 if k == C else 0.0 for k in range(CHANNELS)])
    vec0 = g.fconst([1, CHANNELS, 1, 1], [1.0 if k == 0 else 0.0 for k in range(CHANNELS)])
    win = g.node("Add", [g.node("Mul", [res, vecC]), g.node("Mul", [notres, vec0])])
    _pad_out(g, win, oh_, ow)
    return _model(g.nodes, g.inits)


def build_merge(L, R, oh_, ow, top):
    """Colour-preserving overlay.  top='L' -> left over right, else right over left."""
    g = _G()
    Lh = _half(g, *L)
    Rh = _half(g, *R)
    front, back = (Lh, Rh) if top == "L" else (Rh, Lh)
    fr0, fr1, fc0, fc1 = L if top == "L" else R
    # background channel (0) of the FRONT half -> True where front is background
    s = g.iconst([0, fr0, fc0]); e = g.iconst([1, fr1, fc1]); ax = g.iconst([1, 2, 3])
    ch0 = g.node("Slice", ["input", s, e, ax])          # [1,1,h,w] 1 where front==bg
    condbg = g.node("Cast", [ch0], to=BOOL)
    win = g.node("Where", [condbg, back, front])        # front where active, else back
    _pad_out(g, win, oh_, ow)
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics)                                #
# --------------------------------------------------------------------------- #
_BOOLF = {
    "and": lambda l, r: l & r,
    "or": lambda l, r: l | r,
    "xor": lambda l, r: l ^ r,
    "nand": lambda l, r: ~(l & r),
    "nor": lambda l, r: ~(l | r),
    "xnor": lambda l, r: ~(l ^ r),
    "dlr": lambda l, r: l & ~r,
    "drl": lambda l, r: r & ~l,
}


def _rects(orient, dw, oh_, ow):
    """Return ((Lrect),(Rrect)) as (r0,r1,c0,c1) for the two halves."""
    if orient == "vcol":                # side by side; oh_==H
        L = (0, oh_, 0, ow)
        R = (0, oh_, ow + dw, ow + dw + ow)
    else:                               # stacked; ow==W
        L = (0, oh_, 0, ow)
        R = (oh_ + dw, oh_ + dw + oh_, 0, ow)
    return L, R


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
    ish = {a.shape for a, _ in prs}
    osh = {b.shape for _, b in prs}
    if len(ish) != 1 or len(osh) != 1:      # geometry must be constant -> origin-safe
        return []
    H, W = next(iter(ish))
    oh_, ow = next(iter(osh))
    if (oh_, ow) == (H, W):                  # output not a half -> not this family
        return []

    out, seen = [], set()

    def add(name, model):
        if name in seen:
            return
        seen.add(name)
        out.append((name, model))

    # enumerate split orientations / divider widths consistent with the shapes
    configs = []
    for dw in (0, 1, 2):
        if oh_ == H and W == 2 * ow + dw:
            configs.append(("vcol", dw))
        if ow == W and H == 2 * oh_ + dw:
            configs.append(("hrow", dw))

    for orient, dw in configs:
        L, R = _rects(orient, dw, oh_, ow)
        # extract numpy halves for validation
        def half(a, rect):
            r0, r1, c0, c1 = rect
            return a[r0:r1, c0:c1]
        # bail if any rect goes out of bounds / mismatched
        good_geom = True
        Ls, Rs, Bs = [], [], []
        for a, b in prs:
            lh, rh = half(a, L), half(a, R)
            if lh.shape != (oh_, ow) or rh.shape != (oh_, ow) or b.shape != (oh_, ow):
                good_geom = False
                break
            Ls.append(lh); Rs.append(rh); Bs.append(b)
        if not good_geom:
            continue

        # --- fixed-colour gates -------------------------------------------- #
        for fname, fn in _BOOLF.items():
            for C in range(1, CHANNELS):
                ok = True
                for lh, rh, b in zip(Ls, Rs, Bs):
                    res = fn(lh != 0, rh != 0)
                    if not np.array_equal(np.where(res, C, 0), b):
                        ok = False
                        break
                if ok:
                    try:
                        add(f"gate_{orient}{dw}_{fname}_C{C}",
                            build_gate(L, R, oh_, ow, fname, C))
                    except Exception:
                        pass

        # --- colour-preserving merges -------------------------------------- #
        for top, mf in (("L", lambda l, r: np.where(l != 0, l, r)),
                        ("R", lambda l, r: np.where(r != 0, r, l))):
            ok = True
            for lh, rh, b in zip(Ls, Rs, Bs):
                if not np.array_equal(mf(lh, rh), b):
                    ok = False
                    break
            if ok:
                try:
                    add(f"merge_{orient}{dw}_{top}", build_merge(L, R, oh_, ow, top))
                except Exception:
                    pass

    return out
