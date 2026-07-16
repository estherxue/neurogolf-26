"""RECTANGLE / BOUNDING-BOX (frame) operations (origin-anchored, opset 10).

Every rule here computes an axis-aligned bounding box from a colour mask and then
paints a rectangular region (the whole box / its perimeter / its interior) with a
colour.  Bounding boxes are obtained WITHOUT any Loop/NonZero: the min / max row &
column of a 0/1 presence mask are read off with position-weighted ``ReduceMax``
reductions, and the box / edge masks are reconstructed from those four scalars with
broadcast ``Greater``/``Less`` comparisons against absolute row / column index
vectors.  Because the one-hot tensor is zero-padded to 30x30 with the grid anchored
at (0,0), every presence mask is exactly 0 over the padding, so the boxes always lie
inside the real region and the rules generalise to grids of ANY size.

Scopes
------
  global    one bounding box over ALL non-background cells.
  percolor  one bounding box per colour channel (vectorised over the 10 channels;
            channel 0 / background is excluded), painted independently.

Regions
-------
  fill      the whole box.
  perim     the 4 edges of the box (the classic "draw a rectangle around the cells").
  interior  the box minus its perimeter (e.g. hollow a filled rectangle).

Colours / policy
----------------
  fixed B   paint with a single inferred colour B (B may be 0 -> erase to background).
  self      (percolor only) paint each box with ITS OWN colour.
  overwrite paint every selected cell;   onbg paint only currently-background cells
            (keeps existing non-background cells, e.g. fill a frame's inside).

Realisation:
  minrow = Cbig - ReduceMax(rowhas * (Cbig - rowidx));  maxrow = ReduceMax(rowhas * rowidx)
  in_rows = (rowidx >= minrow) & (rowidx <= maxrow);  inbox = in_rows * in_cols
  perim   = inbox * max(rowedge, coledge);  interior = inbox - perim
then a single Where (overwrite) or a small additive routing (onbg) produces the
output.  Detection reproduces the numpy semantics exactly and only emits a
candidate when it matches EVERY available train+test+arc-gen pair (the grader's
gate), so wrong hypotheses are dropped before scoring.
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
F = DATA_TYPE
_CBIG = 1000.0


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0
        self._half = None
        self._cbig = None

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    def half(self):
        if self._half is None:
            self._half = self.f([1, 1, 1, 1], [0.5])
        return self._half

    def cbig(self):
        if self._cbig is None:
            self._cbig = self.f([1, 1, 1, 1], [_CBIG])
        return self._cbig


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _onehot(k):
    return [1.0 if c == k else 0.0 for c in range(CHANNELS)]


def _pos_consts(g):
    """rowidx [1,1,30,1] = 0..29 ; colidx [1,1,1,30] = 0..29 (float)."""
    rowidx = g.f([1, 1, HEIGHT, 1], list(range(HEIGHT)))
    colidx = g.f([1, 1, 1, WIDTH], list(range(WIDTH)))
    return rowidx, colidx


# --------------------------------------------------------------------------- #
# bounding-box region from a presence mask                                    #
# --------------------------------------------------------------------------- #
def _box_region(g, rh, ch, rowidx, colidx, region):
    """rh: row-presence [1,K,30,1] (>=1 where a row contains the mask colour),
    ch: col-presence [1,K,1,30].  Returns sel [1,K,30,30] for the chosen region."""
    half = g.half()
    cbig = g.cbig()

    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rh, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rh, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [ch, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [ch, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    ge_r = g.nd("Cast", [g.nd("Greater", [rowidx, g.nd("Sub", [minrow, half])])], to=F)
    le_r = g.nd("Cast", [g.nd("Less", [rowidx, g.nd("Add", [maxrow, half])])], to=F)
    in_rows = g.nd("Mul", [ge_r, le_r])                          # [1,K,30,1]
    ge_c = g.nd("Cast", [g.nd("Greater", [colidx, g.nd("Sub", [mincol, half])])], to=F)
    le_c = g.nd("Cast", [g.nd("Less", [colidx, g.nd("Add", [maxcol, half])])], to=F)
    in_cols = g.nd("Mul", [ge_c, le_c])                          # [1,K,1,30]
    inbox = g.nd("Mul", [in_rows, in_cols])                      # [1,K,30,30]

    if region == "fill":
        return inbox

    # perimeter / interior need the edge lines
    def eqline(idx, val):
        gt = g.nd("Cast", [g.nd("Greater", [idx, g.nd("Sub", [val, half])])], to=F)
        lt = g.nd("Cast", [g.nd("Less", [idx, g.nd("Add", [val, half])])], to=F)
        return g.nd("Mul", [gt, lt])

    rowedge = g.nd("Cast", [g.nd("Greater",
                  [g.nd("Add", [eqline(rowidx, minrow), eqline(rowidx, maxrow)]), half])], to=F)
    coledge = g.nd("Cast", [g.nd("Greater",
                  [g.nd("Add", [eqline(colidx, mincol), eqline(colidx, maxcol)]), half])], to=F)
    edge = g.nd("Max", [rowedge, coledge])                       # [1,K,30,30]
    perim = g.nd("Mul", [inbox, edge])
    if region == "perim":
        return perim
    return g.nd("Sub", [inbox, perim])                           # interior


def _ch0(g):
    return g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])   # [1,1,30,30]


# --------------------------------------------------------------------------- #
# ONNX builders                                                                #
# --------------------------------------------------------------------------- #
def build_global(region, B, policy):
    g = _G()
    rowidx, colidx = _pos_consts(g)
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    ch0 = _ch0(g)
    nbg = g.nd("Sub", [realmask, ch0])                             # [1,1,30,30]
    rh = g.nd("ReduceMax", [nbg], axes=[3], keepdims=1)            # [1,1,30,1]
    cw = g.nd("ReduceMax", [nbg], axes=[2], keepdims=1)            # [1,1,1,30]
    sel = _box_region(g, rh, cw, rowidx, colidx, region)           # [1,1,30,30]

    eB = g.f([1, CHANNELS, 1, 1], _onehot(B))
    if policy == "overwrite":
        cond = g.nd("Cast", [g.nd("Greater", [sel, g.half()])], to=BOOL)
        g.nd("Where", [cond, eB, "input"], "output")
    else:  # onbg
        paint = g.nd("Mul", [sel, ch0])                            # bg cells in region
        bminus0 = g.f([1, CHANNELS, 1, 1],
                      [(1.0 if c == B else 0.0) - (1.0 if c == 0 else 0.0)
                       for c in range(CHANNELS)])
        g.nd("Add", ["input", g.nd("Mul", [paint, bminus0])], "output")
    return _model(g)


def build_percolor(region, colmode, B, policy):
    g = _G()
    rowidx, colidx = _pos_consts(g)
    nbgmask = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    rh = g.nd("Mul", [g.nd("ReduceMax", ["input"], axes=[3], keepdims=1), nbgmask])  # [1,10,30,1]
    cw = g.nd("Mul", [g.nd("ReduceMax", ["input"], axes=[2], keepdims=1), nbgmask])  # [1,10,1,30]
    sel = _box_region(g, rh, cw, rowidx, colidx, region)          # [1,10,30,30] per channel
    ch0 = _ch0(g)

    if colmode == "self":
        if policy == "overwrite":
            cov = g.nd("ReduceSum", [sel], axes=[1], keepdims=1)  # [1,1,30,30]
            cond = g.nd("Cast", [g.nd("Greater", [cov, g.half()])], to=BOOL)
            g.nd("Where", [cond, sel, "input"], "output")
        else:  # onbg
            cb = g.nd("Mul", [sel, ch0])                          # [1,10,30,30] bg cells
            cbsum = g.nd("ReduceSum", [cb], axes=[1], keepdims=1)
            e0 = g.f([1, CHANNELS, 1, 1], _onehot(0))
            g.nd("Sub", [g.nd("Add", ["input", cb]), g.nd("Mul", [cbsum, e0])], "output")
        return _model(g)

    # fixed colour B (applied to every selected cell, any colour box)
    total = g.nd("ReduceSum", [sel], axes=[1], keepdims=1)        # [1,1,30,30]
    eB = g.f([1, CHANNELS, 1, 1], _onehot(B))
    if policy == "overwrite":
        cond = g.nd("Cast", [g.nd("Greater", [total, g.half()])], to=BOOL)
        g.nd("Where", [cond, eB, "input"], "output")
    else:  # onbg
        cov = g.nd("Cast", [g.nd("Greater", [total, g.half()])], to=F)
        paint = g.nd("Mul", [cov, ch0])
        bminus0 = g.f([1, CHANNELS, 1, 1],
                      [(1.0 if c == B else 0.0) - (1.0 if c == 0 else 0.0)
                       for c in range(CHANNELS)])
        g.nd("Add", ["input", g.nd("Mul", [paint, bminus0])], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics exactly for detection)          #
# --------------------------------------------------------------------------- #
def _bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _paint(out, bb, col, region, onbg, a):
    r0, r1, c0, c1 = bb
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            per = i in (r0, r1) or j in (c0, c1)
            if region == "fill":
                sel = True
            elif region == "perim":
                sel = per
            else:  # interior
                sel = not per
            if sel and (not onbg or a[i, j] == 0):
                out[i, j] = col


def _apply_global(a, region, B, onbg):
    bb = _bbox(a != 0)
    if bb is None:
        return None
    out = a.copy()
    _paint(out, bb, B, region, onbg, a)
    return out


def _apply_percolor(a, region, colmode, B, onbg):
    out = a.copy()
    seen = False
    for c in range(1, CHANNELS):
        bb = _bbox(a == c)
        if bb is None:
            continue
        seen = True
        col = c if colmode == "self" else B
        _paint(out, bb, col, region, onbg, a)
    return out if seen else None


# --------------------------------------------------------------------------- #
# entry point                                                                  #
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


def _new_colors(prs):
    s = set()
    for a, b in prs:
        d = a != b
        if d.any():
            s |= set(int(v) for v in b[d].tolist())
    return s


def _matches(prs, fn):
    for a, b in prs:
        o = fn(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def _emit(out, seen, name, builder):
    if name in seen:
        return
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    seen.add(name)
    out.append((name, m))


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):       # all framebox ops preserve shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []

    newc = _new_colors(prs)
    fixedB = next(iter(newc)) if len(newc) == 1 else None

    out, seen = [], set()
    for region in ("fill", "perim", "interior"):
        for policy in ("overwrite", "onbg"):
            # ---- global bounding box (all non-bg), fixed colour ----------- #
            if fixedB is not None and not (policy == "onbg" and fixedB == 0):
                if _matches(prs, lambda a, r=region, b=fixedB, p=policy:
                            _apply_global(a, r, b, p == "onbg")):
                    _emit(out, seen, f"fb_glob_{region}_{policy}_B{fixedB}",
                          lambda r=region, b=fixedB, p=policy: build_global(r, b, p))

            # ---- per-colour bounding boxes, own colour ------------------- #
            if _matches(prs, lambda a, r=region, p=policy:
                        _apply_percolor(a, r, "self", 0, p == "onbg")):
                _emit(out, seen, f"fb_pc_{region}_{policy}_self",
                      lambda r=region, p=policy: build_percolor(r, "self", 0, p))

            # ---- per-colour bounding boxes, fixed colour ----------------- #
            if fixedB is not None and not (policy == "onbg" and fixedB == 0):
                if _matches(prs, lambda a, r=region, b=fixedB, p=policy:
                            _apply_percolor(a, r, "fixed", b, p == "onbg")):
                    _emit(out, seen, f"fb_pc_{region}_{policy}_B{fixedB}",
                          lambda r=region, b=fixedB, p=policy: build_percolor(r, "fixed", b, p))

    return out
