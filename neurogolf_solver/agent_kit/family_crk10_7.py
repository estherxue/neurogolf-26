"""family_crk10_7 -- hardest-slice cracks  U[7::8] = [66,96,145,175,233,366].

SOLVED here: task 145.

--- task 145 (RULE, verified EXACT on 4 train + 262 arc-gen + 1 test) ---
Input grids contain only colors {0 (empty), 2 (walls)}.  The 2-walls partition
the empty cells into axis-aligned RECTANGULAR rooms (verified: every one of the
2054 rooms across all pairs is a perfect rectangle).  Output:
    * the room(s) with the LARGEST area  -> filled with color 1
    * the room(s) with the SMALLEST area -> filled with color 8
    * every other empty cell stays 0, walls stay 2, padding stays blank.
    * tie-break when a room is both max and min (single room / all equal): min
      wins -> filled 8 (matches the reference numpy solver, which paints max
      then overwrites with min).

How it is expressed with static opset-10 ops (no data-dependent shapes):
    empty = input[:,0], wall = input[:,2]           (padding is all-zero)
    For a rectangle, area == (horizontal run length) * (vertical run length),
    and that product is CONSTANT over the whole rectangle.  Each run length is
    computed with a segmented "count consecutive empties" doubling scan
    (offsets 1,2,4,8,16 handle runs up to 32 -> covers 30x30):
        left  = consecutive empties ending at cell going left   (shift +d)
        right = consecutive empties ending at cell going right  (shift -d)
        hrun  = left + right - empty     (full width, same for every cell)
      similarly vrun over rows; area = hrun * vrun.
    Amax = ReduceMax(area); Amin = ReduceMin(area masked to empties).
    is_max = area==Amax ; is_min = area==Amin (min has priority).
    output one-hot channels: 0=empty&~max&~min, 1=is_max_only, 2=wall,
    8=is_min, all others 0.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT
H = W = 30
DOUBLE = [1, 2, 4, 8, 16]


# ------------------------------------------------------------------ numpy mirror
def _sh(x, ax, d, sign):
    out = np.zeros_like(x)
    n = x.shape[ax]
    if d >= n:
        return out
    dst = [slice(None)] * x.ndim
    src = [slice(None)] * x.ndim
    if sign > 0:                       # result[k] = x[k-d]
        dst[ax] = slice(d, n); src[ax] = slice(0, n - d)
    else:                              # result[k] = x[k+d]
        dst[ax] = slice(0, n - d); src[ax] = slice(d, n)
    out[tuple(dst)] = x[tuple(src)]
    return out


def _runcount(a, ax):
    L = a.copy(); R = a.copy()
    for d in DOUBLE:
        L = L + (L >= d - 0.5) * _sh(L, ax, d, +1)
    for d in DOUBLE:
        R = R + (R >= d - 0.5) * _sh(R, ax, d, -1)
    return L + R - a


def _solve(grid):
    Hn, Wn = grid.shape
    empty = np.zeros((30, 30)); wall = np.zeros((30, 30))
    empty[:Hn, :Wn] = (grid == 0)
    wall[:Hn, :Wn] = (grid == 2)
    hrun = _runcount(empty, 1); vrun = _runcount(empty, 0)
    area = hrun * vrun
    Amax = area.max()
    Amin = (area + (1 - empty) * 1e6).min()
    is_min = empty * (np.abs(area - Amin) < 0.5)
    is_max = empty * (np.abs(area - Amax) < 0.5) * (1 - is_min)
    g = np.full((30, 30), -1, int)
    g[wall > 0.5] = 2
    g[(empty - is_max - is_min) > 0.5] = 0
    g[is_max > 0.5] = 1
    g[is_min > 0.5] = 8
    return g[:Hn, :Wn]


# ------------------------------------------------------------------ onnx builder
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def name(self, p="t"):
        self.n += 1
        return f"{p}{self.n}"

    def const(self, val, itype=FLOAT):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, itype, [1], [val]))
        return nm

    def islice(self, starts, ends, axes):
        s = self.name("s"); e = self.name("e"); a = self.name("a")
        self.inits.append(oh.make_tensor(s, INT64, [len(starts)], starts))
        self.inits.append(oh.make_tensor(e, INT64, [len(ends)], ends))
        self.inits.append(oh.make_tensor(a, INT64, [len(axes)], axes))
        return s, e, a

    def node(self, op, ins, **attr):
        o = self.name(op.lower())
        self.nodes.append(oh.make_node(op, ins, [o], **attr))
        return o

    def shift(self, x, axis, d, sign):
        # pad then crop 30 along `axis`; sign>0 -> result[k]=x[k-d]
        pads = [0, 0, 0, 0, 0, 0, 0, 0]
        if sign > 0:
            pads[axis] = d                       # begin pad
            s, e, a = self.islice([0], [30], [axis])
        else:
            pads[4 + axis] = d                   # end pad
            s, e, a = self.islice([d], [d + 30], [axis])
        p = self.name("pad")
        self.nodes.append(oh.make_node("Pad", [x], [p], mode="constant",
                                       value=0.0, pads=pads))
        return self.node("Slice", [p, s, e, a])

    def runcount(self, empty, axis):
        # L = consecutive empties ending here going toward smaller index
        L = empty
        for d in DOUBLE:
            sh = self.shift(L, axis, d, +1)
            added = self.node("Add", [L, sh])
            cond = self.node("Greater", [L, self.const(d - 0.5)])
            L = self.node("Where", [cond, added, L])
        R = empty
        for d in DOUBLE:
            sh = self.shift(R, axis, d, -1)
            added = self.node("Add", [R, sh])
            cond = self.node("Greater", [R, self.const(d - 0.5)])
            R = self.node("Where", [cond, added, R])
        full = self.node("Add", [L, R])
        return self.node("Sub", [full, empty])


def _build():
    g = _G()
    # extract channels (axis=1). padding cells are all-zero so empty=color0 real only.
    es, ee, ea = g.islice([0], [1], [1])
    empty = g.node("Slice", ["input", es, ee, ea])
    ws, we, wa = g.islice([2], [3], [1])
    wall = g.node("Slice", ["input", ws, we, wa])

    hrun = g.runcount(empty, 3)       # width axis
    vrun = g.runcount(empty, 2)       # height axis
    area = g.node("Mul", [hrun, vrun])

    half = g.const(0.5); one = g.const(1.0); big = g.const(1e6)

    Amax = g.node("ReduceMax", [area], keepdims=1)
    AmaxM = g.node("Sub", [Amax, half])
    ismax = g.node("Cast", [g.node("Greater", [area, AmaxM])], to=FLOAT)

    notempty = g.node("Sub", [one, empty])
    bigmask = g.node("Mul", [notempty, big])
    aminIn = g.node("Add", [area, bigmask])
    Amin = g.node("ReduceMin", [aminIn], keepdims=1)
    AminP = g.node("Add", [Amin, half])
    isminb = g.node("Cast", [g.node("Less", [area, AminP])], to=FLOAT)
    ismin = g.node("Mul", [empty, isminb])

    notmin = g.node("Sub", [one, ismin])
    ismaxonly = g.node("Mul", [ismax, notmin])

    ch0 = g.node("Sub", [g.node("Sub", [empty, ismaxonly]), ismin])
    zeros = g.node("Sub", [empty, empty])

    chans = [ch0, ismaxonly, wall, zeros, zeros, zeros, zeros, zeros, ismin, zeros]
    g.nodes.append(oh.make_node("Concat", chans, ["output"], axis=1))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g145", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _match(examples):
    pairs = []
    for e in examples.get("train", []) + examples.get("test", []):
        pairs.append((np.array(e["input"]), np.array(e["output"])))
    if not pairs:
        return False
    for a, b in pairs:
        # only fires when input is walls/empty and output uses {0,1,2,8}
        if not set(np.unique(a).tolist()) <= {0, 2}:
            return False
        pred = _solve(a)
        if pred.shape != b.shape or not (pred == b).all():
            return False
    return True


def candidates(examples):
    out = []
    if _match(examples):
        out.append(("crk10_7_rooms_minmax", _build()))
    return out
