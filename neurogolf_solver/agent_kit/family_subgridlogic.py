"""TWO-PANEL LOGIC: combine two equal sub-grids cell-by-cell (opset-10, origin-safe).

Many ARC tasks present the input as TWO equal panels laid side-by-side or stacked
(optionally separated by a one/two-cell divider line, or simply the grid being 2x
one panel) and ask for a single PANEL-SIZED output whose every cell is a logical /
comparison function of the two corresponding panel cells.  This family realises a
RICH, structured set of those combiners and recolours the result to one inferred
output colour:

  mask gates       output[i,j] = C  iff  f( left!=bg , right!=bg )  else background,
                   with f in {and, or, xor, nand, nor, xnor,
                              diff-left (l & ~r), diff-right (r & ~l)}.
                   "difference" and "cells where they differ" are the dlr/drl and
                   xor members.

  colour gates     compare the two panels by COLOUR VALUE (not just bg/non-bg):
                     differ      C where left_colour != right_colour
                     agree_nbg   C where left_colour == right_colour and non-bg
                     agree_all   C where left_colour == right_colour
                   Realised via the channel dot-product  agree = sum_c L[c]*R[c]
                   (==1 iff the one-hot colours match), so it needs no float Equal.

  merges           colour-preserving overlay: output = left  where left!=bg else
                   right  (and the symmetric right-over-left), used when the two
                   panels never disagree.

Origin safety (CONTEXT.md padding gotcha)
-----------------------------------------
The split column/row depends on the grid size, so the rule is only well defined
when the panel geometry is CONSTANT across every pair.  We therefore emit ONLY for
tasks whose input shape AND output shape are identical across all
train+test+arc-gen pairs.  Knowing the exact (H,W) we slice the two fixed
sub-windows, combine them, and zero-pad the half-sized result back to 30x30 -> the
content stays anchored top-left and every padding cell is all-zero, so the one-hot
contract holds for grids of any (fixed) size.

Realisation is opset-10 only (Slice/ReduceSum/Mul/Max/Abs/Sub/Where/Pad) and
materialises only panel-sized [1,*,h,w] intermediates, so the cost stays low.
Detection mirrors the ONNX semantics exactly and validates EVERY available pair
(the grader's gate); every rule is STRUCTURAL (a fixed combiner + a single inferred
colour), so wrong hypotheses are dropped and matches generalise to unseen grids.
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
# sub-window extraction                                                        #
# --------------------------------------------------------------------------- #
def _active(g, rect):
    """{0,1} non-background plane [1,1,h,w] of a sub-window (1 - background ch0)."""
    r0, r1, c0, c1 = rect
    s = g.iconst([0, r0, c0]); e = g.iconst([1, r1, c1]); ax = g.iconst([1, 2, 3])
    ch0 = g.node("Slice", ["input", s, e, ax])          # [1,1,h,w] 1 where bg
    return g.node("Sub", [g.one(), ch0])                # [1,1,h,w] in {0,1}


def _half(g, rect):
    """Full [1,10,h,w] sub-window (all colour channels)."""
    r0, r1, c0, c1 = rect
    s = g.iconst([r0, c0]); e = g.iconst([r1, c1]); ax = g.iconst([2, 3])
    return g.node("Slice", ["input", s, e, ax])


def _agree(g, L, R):
    """[1,1,h,w] == 1 iff the two one-hot colours match (incl. bg==bg)."""
    Lh = _half(g, L)
    Rh = _half(g, R)
    prod = g.node("Mul", [Lh, Rh])                      # [1,10,h,w]
    return g.node("ReduceSum", [prod], axes=[1], keepdims=1)   # [1,1,h,w]


def _mask_gate(g, la, ra, fname):
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
    if fname == "dlr":                                   # left and not right
        return g.node("Sub", [la, g.node("Mul", [la, ra])])
    if fname == "drl":                                   # right and not left
        return g.node("Sub", [ra, g.node("Mul", [la, ra])])
    raise ValueError(fname)


def _pad_out(g, win, oh_, ow):
    """[1,10,oh_,ow] -> output [1,10,30,30] (trailing zero pad, origin anchored)."""
    if oh_ == HEIGHT and ow == WIDTH:
        g.node("Identity", [win], "output")
        return
    pads = [0, 0, 0, 0, 0, 0, HEIGHT - oh_, WIDTH - ow]
    g.node("Pad", [win], "output", mode="constant", value=0.0, pads=pads)


def _emit_res(g, res, oh_, ow, C):
    """res in {0,1} [1,1,oh_,ow] -> one-hot output: colour C where res, bg else."""
    notres = g.node("Sub", [g.one(), res])
    vecC = g.fconst([1, CHANNELS, 1, 1], [1.0 if k == C else 0.0 for k in range(CHANNELS)])
    vec0 = g.fconst([1, CHANNELS, 1, 1], [1.0 if k == 0 else 0.0 for k in range(CHANNELS)])
    win = g.node("Add", [g.node("Mul", [res, vecC]), g.node("Mul", [notres, vec0])])
    _pad_out(g, win, oh_, ow)


# --------------------------------------------------------------------------- #
# ONNX builders                                                                #
# --------------------------------------------------------------------------- #
def build_mask_gate(L, R, oh_, ow, fname, C):
    g = _G()
    res = _mask_gate(g, _active(g, L), _active(g, R), fname)
    _emit_res(g, res, oh_, ow, C)
    return _model(g.nodes, g.inits)


def build_color_gate(L, R, oh_, ow, kind, C):
    """kind in {'differ','agree_all','agree_nbg'} by colour-value comparison."""
    g = _G()
    agree = _agree(g, L, R)                              # 1 iff same colour
    if kind == "differ":
        res = g.node("Sub", [g.one(), agree])
    elif kind == "agree_all":
        res = agree
    else:                                                # agree_nbg
        res = g.node("Mul", [agree, _active(g, L)])
    _emit_res(g, res, oh_, ow, C)
    return _model(g.nodes, g.inits)


def build_merge(L, R, oh_, ow, top):
    """Colour-preserving overlay.  top='L' -> left over right, else right over left."""
    g = _G()
    Lh = _half(g, L)
    Rh = _half(g, R)
    front, back = (Lh, Rh) if top == "L" else (Rh, Lh)
    frect = L if top == "L" else R
    r0, r1, c0, c1 = frect
    s = g.iconst([0, r0, c0]); e = g.iconst([1, r1, c1]); ax = g.iconst([1, 2, 3])
    ch0 = g.node("Slice", ["input", s, e, ax])          # [1,1,h,w] 1 where front==bg
    condbg = g.node("Cast", [ch0], to=BOOL)
    win = g.node("Where", [condbg, back, front])        # front where active, else back
    _pad_out(g, win, oh_, ow)
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics for detection)                  #
# --------------------------------------------------------------------------- #
_MASKF = {
    "and": lambda l, r: l & r,
    "or": lambda l, r: l | r,
    "xor": lambda l, r: l ^ r,
    "nand": lambda l, r: ~(l & r),
    "nor": lambda l, r: ~(l | r),
    "xnor": lambda l, r: ~(l ^ r),
    "dlr": lambda l, r: l & ~r,
    "drl": lambda l, r: r & ~l,
}


def _color_res(kind, lh, rh):
    if kind == "differ":
        return lh != rh
    if kind == "agree_all":
        return lh == rh
    return (lh == rh) & (lh != 0)                        # agree_nbg


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


def _rects(orient, dw, oh_, ow):
    """((Lrect),(Rrect)) as (r0,r1,c0,c1) for the two equal panels."""
    if orient == "vcol":                # side by side; oh_ == H
        return (0, oh_, 0, ow), (0, oh_, ow + dw, ow + dw + ow)
    return (0, oh_, 0, ow), (oh_ + dw, oh_ + dw + oh_, 0, ow)   # stacked; ow == W


def _infer_color(masks, outs):
    """Single non-bg colour C such that out == C where mask, bg elsewhere; else None."""
    cols = set()
    for m, b in zip(masks, outs):
        if (b[~m] != 0).any():
            return None
        cols |= set(b[m].tolist())
    cols.discard(0)
    if len(cols) != 1:
        return None
    C = next(iter(cols))
    # require the combiner to actually fire somewhere (avoid degenerate all-bg match)
    if not any(m.any() for m in masks):
        return None
    return C


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
    if (oh_, ow) == (H, W):                  # output not a half-panel -> not this family
        return []

    # split orientations / divider widths consistent with the fixed shapes
    configs = []
    for dw in (0, 1, 2):
        if oh_ == H and W == 2 * ow + dw:
            configs.append(("vcol", dw))
        if ow == W and H == 2 * oh_ + dw:
            configs.append(("hrow", dw))
    if not configs:
        return []

    out, seen = [], set()

    def add(name, builder):
        if name in seen:
            return
        try:
            model = builder()
        except Exception:
            return
        seen.add(name)
        out.append((name, model))

    for orient, dw in configs:
        L, R = _rects(orient, dw, oh_, ow)

        def half(a, rect):
            r0, r1, c0, c1 = rect
            return a[r0:r1, c0:c1]

        Ls, Rs, Bs, good = [], [], [], True
        for a, b in prs:
            lh, rh = half(a, L), half(a, R)
            if lh.shape != (oh_, ow) or rh.shape != (oh_, ow) or b.shape != (oh_, ow):
                good = False
                break
            Ls.append(lh); Rs.append(rh); Bs.append(b)
        if not good:
            continue

        # ---- mask boolean gates ------------------------------------------- #
        for fname, fn in _MASKF.items():
            masks = [fn(lh != 0, rh != 0) for lh, rh in zip(Ls, Rs)]
            C = _infer_color(masks, Bs)
            if C is not None:
                add(f"maskgate_{orient}{dw}_{fname}_C{C}",
                    lambda L=L, R=R, fn=fname, C=C: build_mask_gate(L, R, oh_, ow, fn, C))

        # ---- colour-value gates ------------------------------------------- #
        for kind in ("differ", "agree_nbg", "agree_all"):
            masks = [_color_res(kind, lh, rh) for lh, rh in zip(Ls, Rs)]
            C = _infer_color(masks, Bs)
            if C is not None:
                add(f"colorgate_{orient}{dw}_{kind}_C{C}",
                    lambda L=L, R=R, k=kind, C=C: build_color_gate(L, R, oh_, ow, k, C))

        # ---- colour-preserving merges ------------------------------------- #
        for top, mf in (("L", lambda l, r: np.where(l != 0, l, r)),
                        ("R", lambda l, r: np.where(r != 0, r, l))):
            if all(np.array_equal(mf(lh, rh), b) for lh, rh, b in zip(Ls, Rs, Bs)):
                # skip the degenerate case where one panel is empty everywhere
                if any((lh != 0).any() for lh in Ls) and any((rh != 0).any() for rh in Rs):
                    add(f"merge_{orient}{dw}_{top}",
                        lambda L=L, R=R, top=top: build_merge(L, R, oh_, ow, top))

    return out
