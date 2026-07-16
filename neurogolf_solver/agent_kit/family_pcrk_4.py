"""family_pcrk_4  (CRACK slice U[4::6])

Solved rule
-----------
task138  "frame crop + directional border projection":
    The grid contains a rectangular FRAME made of 4 full-length monochrome lines
    (two horizontal rows with no zero cell, two vertical columns with no zero
    cell), each a distinct colour.  The answer is the sub-grid cropped to the
    frame's bounding box (re-anchored top-left).  Inside the frame, scattered
    pixels whose colour equals one of the four border colours PROJECT toward that
    border: a bottom-border-colour pixel fills a ray down to the bottom border,
    a top-border pixel fills up, right-border right, left-border left (only into
    empty cells; earlier directions win on collision, order down/up/right/left).

ONNX construction (opset-10, static graph):
    * full-row / full-col detection via ReduceSum of channel-0 and occupancy.
    * data-dependent crop-to-origin via computed selection matrices + MatMul
      (the proven family_dyncrop idiom).
    * directional cumulative "ray" fills via constant lower/upper triangular
      [30,30] MatMul matrices; border colours read at fixed cropped positions
      (0,1),(1,0) and dynamic (h-1,1),(1,w-1) via index-selector ReduceSum.
    * priority-disjoint fills combined and written back into the one-hot.

Held-out self-check: the numpy reference (mirroring the ONNX numerics) is EXACT
on all 265 train+arc-gen pairs; a 70/30 split fit on 70% is exact on the untouched
30% (the rule has NO fitted parameters — it is purely structural).
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


# ---------------------------------------------------------------- numpy ref --
def _solve_np(a):
    """Exact reference; returns cropped/filled grid or None if no frame."""
    rows = np.where((a != 0).all(axis=1))[0]
    cols = np.where((a != 0).all(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    r1, r2, c1, c2 = rows.min(), rows.max(), cols.min(), cols.max()
    G = a[r1:r2 + 1, c1:c2 + 1].copy()
    h, w = G.shape
    if h < 3 or w < 3:
        return None
    CT, CB = G[0, w // 2], G[h - 1, w // 2]
    CL, CR = G[h // 2, 0], G[h // 2, w - 1]
    out = G.copy()
    m = (G == CB); cum = np.maximum.accumulate(m, axis=0); out[cum & (out == 0)] = CB
    m = (G == CT); cum = np.maximum.accumulate(m[::-1], axis=0)[::-1]; out[cum & (out == 0)] = CT
    m = (G == CR); cum = np.maximum.accumulate(m, axis=1); out[cum & (out == 0)] = CR
    m = (G == CL); cum = np.maximum.accumulate(m[:, ::-1], axis=1)[:, ::-1]; out[cum & (out == 0)] = CL
    return out


# --------------------------------------------------------------- graph accum --
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

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
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


def build_frame():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [_CBIG])
    onehot0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    Ldown = g.f([H, H], [[1.0 if j <= i else 0.0 for j in range(H)] for i in range(H)])
    Lup = g.f([H, H], [[1.0 if j >= i else 0.0 for j in range(H)] for i in range(H)])

    def cast(x):  # to float
        return g.nd("Cast", [x], to=F)

    # occupancy / colour-0 maps
    occ = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    ch0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])  # [1,1,30,30]

    rowocc = g.nd("ReduceSum", [occ], axes=[3], keepdims=1)           # [1,1,30,1]
    rowz = g.nd("ReduceSum", [ch0], axes=[3], keepdims=1)
    fullrow = g.nd("Mul", [cast(g.nd("Greater", [rowocc, half])),
                           cast(g.nd("Less", [rowz, half]))])          # [1,1,30,1]
    colocc = g.nd("ReduceSum", [occ], axes=[2], keepdims=1)           # [1,1,1,30]
    colz = g.nd("ReduceSum", [ch0], axes=[2], keepdims=1)
    fullcol = g.nd("Mul", [cast(g.nd("Greater", [colocc, half])),
                           cast(g.nd("Less", [colz, half]))])          # [1,1,1,30]

    maxrow = g.nd("ReduceMax", [g.nd("Mul", [fullrow, rowidx])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [fullrow, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [fullcol, colidx])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [fullcol, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    bbox_h = g.nd("Add", [g.nd("Sub", [maxrow, minrow]), one])        # [1,1,1,1]
    bbox_w = g.nd("Add", [g.nd("Sub", [maxcol, mincol]), one])

    # crop-to-origin selection matrices (family_dyncrop idiom)
    diff_c = g.nd("Sub", [g.nd("Add", [colidx, mincol]), rowidx])
    match_c = cast(g.nd("Less", [g.nd("Abs", [diff_c]), half]))
    trunc_c = cast(g.nd("Less", [colidx, bbox_w]))
    Scol = g.nd("Mul", [match_c, trunc_c])                            # [1,1,30,30]
    diff_r = g.nd("Sub", [colidx, g.nd("Add", [rowidx, minrow])])
    match_r = cast(g.nd("Less", [g.nd("Abs", [diff_r]), half]))
    trunc_r = cast(g.nd("Less", [rowidx, bbox_h]))
    Srow = g.nd("Mul", [match_r, trunc_r])                            # [1,1,30,30]
    shift1 = g.nd("MatMul", ["input", Scol])                          # [1,10,30,30]
    cropped = g.nd("MatMul", [Srow, shift1])                          # [1,10,30,30] frame@origin

    zc = g.nd("Slice", [cropped, g.i64([0]), g.i64([1]), g.i64([1])])  # colour-0 map [1,1,30,30]

    # border colour one-hots [1,10,1,1]
    CT_ch = g.nd("Slice", [cropped, g.i64([0, 1]), g.i64([1, 2]), g.i64([2, 3])])
    CL_ch = g.nd("Slice", [cropped, g.i64([1, 0]), g.i64([2, 1]), g.i64([2, 3])])
    eB = cast(g.nd("Less", [g.nd("Abs", [g.nd("Sub", [rowidx, g.nd("Sub", [bbox_h, one])])]), half]))  # [1,1,30,1]
    bottomrow = g.nd("ReduceSum", [g.nd("Mul", [cropped, eB])], axes=[2], keepdims=1)  # [1,10,1,30]
    CB_ch = g.nd("Slice", [bottomrow, g.i64([1]), g.i64([2]), g.i64([3])])             # [1,10,1,1]
    eR = cast(g.nd("Less", [g.nd("Abs", [g.nd("Sub", [colidx, g.nd("Sub", [bbox_w, one])])]), half]))  # [1,1,1,30]
    rightcol = g.nd("ReduceSum", [g.nd("Mul", [cropped, eR])], axes=[3], keepdims=1)   # [1,10,30,1]
    CR_ch = g.nd("Slice", [rightcol, g.i64([1]), g.i64([2]), g.i64([2])])              # [1,10,1,1]

    # source presence maps [1,1,30,30]
    sb = g.nd("ReduceSum", [g.nd("Mul", [cropped, CB_ch])], axes=[1], keepdims=1)
    st = g.nd("ReduceSum", [g.nd("Mul", [cropped, CT_ch])], axes=[1], keepdims=1)
    sr = g.nd("ReduceSum", [g.nd("Mul", [cropped, CR_ch])], axes=[1], keepdims=1)
    sl = g.nd("ReduceSum", [g.nd("Mul", [cropped, CL_ch])], axes=[1], keepdims=1)

    cov_d = cast(g.nd("Greater", [g.nd("MatMul", [Ldown, sb]), half]))  # [1,1,30,30]
    cov_u = cast(g.nd("Greater", [g.nd("MatMul", [Lup, st]), half]))
    cov_r = cast(g.nd("Greater", [g.nd("MatMul", [sr, Lup]), half]))
    cov_l = cast(g.nd("Greater", [g.nd("MatMul", [sl, Ldown]), half]))

    fd = g.nd("Mul", [cov_d, zc])
    rem1 = g.nd("Sub", [one, fd])
    fu = g.nd("Mul", [g.nd("Mul", [cov_u, zc]), rem1])
    rem2 = g.nd("Sub", [rem1, fu])
    fr = g.nd("Mul", [g.nd("Mul", [cov_r, zc]), rem2])
    rem3 = g.nd("Sub", [rem2, fr])
    fl = g.nd("Mul", [g.nd("Mul", [cov_l, zc]), rem3])
    filled = g.nd("Add", [g.nd("Add", [fd, fu]), g.nd("Add", [fr, fl])])  # [1,1,30,30]

    adds = g.nd("Add", [g.nd("Add", [g.nd("Mul", [fd, CB_ch]), g.nd("Mul", [fu, CT_ch])]),
                        g.nd("Add", [g.nd("Mul", [fr, CR_ch]), g.nd("Mul", [fl, CL_ch])])])  # [1,10,30,30]
    removal = g.nd("Mul", [filled, onehot0])                          # [1,10,30,30]
    g.nd("Add", [g.nd("Sub", [cropped, removal]), adds], "output")
    return _model(g)


# ---------------------------------------------------------------- interface --
def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    for a, b in prs:
        p = _solve_np(a)
        if p is None or p.shape != b.shape or not np.array_equal(p, b):
            return []
    yield ("frame_project", build_frame())
