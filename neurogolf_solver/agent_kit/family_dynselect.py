"""DATA-DEPENDENT SELECT + CROP (origin-anchored output, opset 10).

Every rule here SELECTS a colour by a GLOBAL per-colour statistic, takes the
axis-aligned bounding box of that colour, and outputs the CROP of that box moved
to the top-left (0,0) of the grid -- a genuine data-dependent positioning op.

The crop is realised with STATIC shapes via a COMPUTED selection / shift matrix +
MatMul (the breakthrough technique):

  * the selected colour's presence mask  M = Conv(input, gate)        [1,1,30,30]
    where ``gate`` [1,10,1,1] is a runtime one-hot over the chosen channel;
  * its bounding box (minrow,maxrow,mincol,maxcol) is read off with
    position-weighted ReduceMax reductions (no Loop/NonZero), exactly as in
    family_framebox -- all four are data-dependent scalars [1,1,1,1];
  * a box mask zeroes everything outside the rectangle, so only the crop survives;
  * two [1,1,30,30] shift matrices Srow / Scol with  Srow[i,k]=1 iff k==i+minrow
    and  Scol[k,j]=1 iff k==j+mincol  are built from constant index grids with
    Sub/Abs/Less/Cast, and  output = MatMul(Srow, box*input) @ Scol  translates the
    cropped rectangle to (0,0) for grids of ANY size.

Because the one-hot tensor is zero-padded to 30x30 with the grid at (0,0), every
presence mask is exactly 0 over the padding, so the box always lies inside the
real region and the rule generalises to grids of any size; the output is the crop
anchored top-left, zero-padded -- matching the grader's top-left target placement.

Colour-selection rules (the exact one is inferred from the train/test/arc-gen
pairs and only emitted when it reproduces EVERY available pair):

  rare      the unique least-frequent non-background colour (min cell count).
  freq      the unique most-frequent non-background colour (max cell count).
  minarea   the unique colour with the smallest bounding-box area.

Detection mirrors the ONNX semantics exactly (strict unique arg-extremum, integer
counts/areas representable in float32) and rejects any task where the winner is
not unique on some pair, so wrong hypotheses are dropped before scoring.
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
_CBIG = 1000.0          # > 30, for min via max-of-complement
_BIG = 1.0e6            # push absent / background channels out of arg-extremum
H, W = HEIGHT, WIDTH


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0
        self._cache = {}

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                          [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    # cached scalar / grid constants ---------------------------------------- #
    def scalar(self, v):
        key = ("s", v)
        if key not in self._cache:
            self._cache[key] = self.f([1, 1, 1, 1], [v])
        return self._cache[key]

    def rowidx(self):
        if "rowidx" not in self._cache:
            self._cache["rowidx"] = self.f([1, 1, H, 1], list(range(H)))
        return self._cache["rowidx"]

    def colidx(self):
        if "colidx" not in self._cache:
            self._cache["colidx"] = self.f([1, 1, 1, W], list(range(W)))
        return self._cache["colidx"]

    def chan(self, vals):
        return self.f([1, CHANNELS, 1, 1], vals)


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _nbg():
    return [0.0] + [1.0] * (CHANNELS - 1)


# --------------------------------------------------------------------------- #
# colour-selection gate  ->  [1,10,1,1] one-hot over the chosen channel        #
# --------------------------------------------------------------------------- #
def _counts(g):
    return g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)        # [1,10,1,1]


def _gate(g, rule):
    nbg = g.chan(_nbg())
    half = g.scalar(0.5)
    counts = _counts(g)
    if rule == "freq":
        offneg = g.chan([-_BIG] + [0.0] * (CHANNELS - 1))
        masked = g.nd("Add", [counts, offneg])
        mx = g.nd("ReduceMax", [masked], axes=[1], keepdims=1)
        sel = g.nd("Greater", [masked, g.nd("Sub", [mx, half])])
        return g.nd("Mul", [g.nd("Cast", [sel], to=F), nbg])

    if rule == "rare":
        stat = counts
    else:  # minarea
        stat = _bbox_area(g)                                            # [1,10,1,1]
    ones = g.chan([1.0] * CHANNELS)
    present = g.nd("Mul", [g.nd("Clip", [counts], min=0.0, max=1.0), nbg])
    push = g.nd("Mul", [g.nd("Sub", [ones, present]), g.scalar(_BIG)])
    masked = g.nd("Add", [g.nd("Mul", [stat, present]), push])         # absent -> BIG
    mn = g.nd("ReduceMin", [masked], axes=[1], keepdims=1)
    sel = g.nd("Less", [masked, g.nd("Add", [mn, half])])
    return g.nd("Mul", [g.nd("Cast", [sel], to=F), nbg])


def _bbox_area(g):
    """Per-channel bounding-box area -> [1,10,1,1] (garbage for absent channels,
    but those are pushed to +BIG by the caller before the arg-min)."""
    cbig = g.scalar(_CBIG)
    one = g.scalar(1.0)
    rowidx, colidx = g.rowidx(), g.colidx()
    rowhas = g.nd("ReduceMax", ["input"], axes=[3], keepdims=1)         # [1,10,30,1]
    colhas = g.nd("ReduceMax", ["input"], axes=[2], keepdims=1)         # [1,10,1,30]
    maxr = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minr = g.nd("Sub", [cbig, g.nd("ReduceMax",
                 [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxc = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    minc = g.nd("Sub", [cbig, g.nd("ReduceMax",
                 [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    height = g.nd("Add", [g.nd("Sub", [maxr, minr]), one])
    width = g.nd("Add", [g.nd("Sub", [maxc, minc]), one])
    return g.nd("Mul", [height, width])                                 # [1,10,1,1]


# --------------------------------------------------------------------------- #
# crop the selected colour's bounding box and move it to (0,0)                 #
# --------------------------------------------------------------------------- #
def build_crop(rule):
    g = _G()
    gate = _gate(g, rule)                                               # [1,10,1,1]
    selmask = g.nd("Conv", ["input", gate], kernel_shape=[1, 1], pads=[0, 0, 0, 0])  # [1,1,30,30]

    cbig = g.scalar(_CBIG)
    half = g.scalar(0.5)
    rowidx, colidx = g.rowidx(), g.colidx()

    rowhas = g.nd("ReduceMax", [selmask], axes=[3], keepdims=1)         # [1,1,30,1]
    colhas = g.nd("ReduceMax", [selmask], axes=[2], keepdims=1)         # [1,1,1,30]
    maxr = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowidx])], axes=[2], keepdims=1)
    minr = g.nd("Sub", [cbig, g.nd("ReduceMax",
                 [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxc = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
    minc = g.nd("Sub", [cbig, g.nd("ReduceMax",
                 [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])

    # box mask: keep only the rectangle [minr..maxr] x [minc..maxc]
    ge_r = g.nd("Cast", [g.nd("Greater", [rowidx, g.nd("Sub", [minr, half])])], to=F)
    le_r = g.nd("Cast", [g.nd("Less", [rowidx, g.nd("Add", [maxr, half])])], to=F)
    in_rows = g.nd("Mul", [ge_r, le_r])                                # [1,1,30,1]
    ge_c = g.nd("Cast", [g.nd("Greater", [colidx, g.nd("Sub", [minc, half])])], to=F)
    le_c = g.nd("Cast", [g.nd("Less", [colidx, g.nd("Add", [maxc, half])])], to=F)
    in_cols = g.nd("Mul", [ge_c, le_c])                                # [1,1,1,30]
    boxmask = g.nd("Mul", [in_rows, in_cols])                          # [1,1,30,30]
    masked = g.nd("Mul", ["input", boxmask])                          # [1,10,30,30]

    # shift matrices (data-dependent): Srow[i,k]=1 iff k==i+minr ; Scol[k,j]=1 iff k==j+minc
    ipmin = g.nd("Add", [rowidx, minr])                                # [1,1,30,1]
    diffr = g.nd("Sub", [colidx, ipmin])                              # [1,1,30,30]
    srow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diffr]), half])], to=F)
    jpmin = g.nd("Add", [colidx, minc])                                # [1,1,1,30]
    diffc = g.nd("Sub", [rowidx, jpmin])                              # [1,1,30,30]
    scol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [diffc]), half])], to=F)

    shifted = g.nd("MatMul", [srow, masked])                          # shift up by minr
    g.nd("MatMul", [shifted, scol], "output")                         # shift left by minc
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics for detection)                  #
# --------------------------------------------------------------------------- #
def _bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _select(a, rule):
    cols = [c for c in range(1, CHANNELS) if (a == c).any()]
    if not cols:
        return None
    if rule == "rare":
        stat = {c: int((a == c).sum()) for c in cols}
        m = min(stat.values())
        win = [c for c in cols if stat[c] == m]
    elif rule == "freq":
        stat = {c: int((a == c).sum()) for c in cols}
        m = max(stat.values())
        win = [c for c in cols if stat[c] == m]
    else:  # minarea
        stat = {}
        for c in cols:
            r0, r1, c0, c1 = _bbox(a == c)
            stat[c] = (r1 - r0 + 1) * (c1 - c0 + 1)
        m = min(stat.values())
        win = [c for c in cols if stat[c] == m]
    return win[0] if len(win) == 1 else None


def _apply(a, rule):
    c = _select(a, rule)
    if c is None:
        return None
    bb = _bbox(a == c)
    if bb is None:
        return None
    r0, r1, c0, c1 = bb
    return a[r0:r1 + 1, c0:c1 + 1]


# --------------------------------------------------------------------------- #
# entry point                                                                 #
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


def _matches(prs, rule):
    for a, b in prs:
        o = _apply(a, rule)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []
    # this family CROPS: output strictly inside input
    if not all(b.shape[0] <= a.shape[0] and b.shape[1] <= a.shape[1] and b.size < a.size
               for a, b in prs):
        return []

    out = []
    for rule in ("rare", "freq", "minarea"):
        if not _matches(prs, rule):
            continue
        try:
            m = build_crop(rule)
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            continue
        out.append((f"cropsel_{rule}", m))
    return out
