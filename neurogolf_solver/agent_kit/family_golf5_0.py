"""family_golf5_0 -- CHEAPER exact re-solvers for a slice of golf targets.

Each candidate re-derives the rule from train+test+arc-gen pairs, validates a
numpy mirror of the *exact* ONNX semantics on every available pair, and only then
emits a minimal opset-10 graph.  The integrator auto-picks the cheapest correct
solver, so we just need these to be exact and cheaper than the incumbent.

Golf levers used here:
  * data-independent index/identity matrices baked as INITIALIZERS (params, free
    of intermediate-memory), not Constant nodes;
  * small sub-grid intermediates ([1,10,k,30]) wherever possible -- never grow a
    full [1,10,30,30] tensor when a narrow slice will do;
  * write the final answer straight into the FREE `output` tensor via Pad/Concat.

Targets (rule -> incumbent points):
  3   vtile_h9: 6x3 -> 9x3.  recolor 1->2 then continue the vertical period for
      three more rows (out[6:9] = in[6-p:9-p] for a valid period p)      (10.99)
  295 golf_stair: 1xW -> (W/2)xW.  staircase fill of one colour c: cell (i,j)
      gets c iff j-i <= K-1, where K = #colour cells in the input row      (13.77)
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


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                       #
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
        self.inits.append(oh.make_tensor(n, INT64, list(dims),
                          [int(v) for v in np.asarray(vals).ravel()]))
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


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.i64(starts), g.i64(ends), g.i64(axes)]
    if steps is not None:
        ins.append(g.i64(steps))
    return g.nd("Slice", ins)


def _eq(g, a, b, half):
    """scalar [1,1,1,1] == 1.0 iff one-hot blocks a,b are identical."""
    d = g.nd("Abs", [g.nd("Sub", [a, b])])
    s = g.nd("ReduceSum", [d], axes=[1, 2, 3], keepdims=1)
    return g.nd("Cast", [g.nd("Less", [s, half])], to=F)


# --------------------------------------------------------------------------- #
# pairs                                                                        #
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


# ===========================================================================
# 3  vtile_h9: 6x3 -> 9x3, recolor 1->2 then continue vertical period 3 rows
# ===========================================================================
def _mir_003(a):
    if a.shape != (6, 3):
        return None
    if not set(np.unique(a)).issubset({0, 1}):
        return None
    rec = np.where(a == 1, 2, a)
    m3 = bool((a[0:3] == a[3:6]).all())
    m4 = bool((a[0:2] == a[4:6]).all())
    m5 = bool((a[0:1] == a[5:6]).all())
    if m3:
        s = 3
    elif m4:
        s = 2
    elif m5:
        s = 1
    else:
        return None
    window = rec[s:s + 3]
    return np.vstack([rec, window])


def _build_003(g):
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])

    top06 = _slice(g, "input", [0], [6], [2])      # [1,10,6,30]
    r03 = _slice(g, "input", [0], [3], [2])         # rows 0..2
    s3 = _slice(g, "input", [3], [6], [2])          # rows 3..5  (window for p=3)
    r02 = _slice(g, "input", [0], [2], [2])         # rows 0..1
    r46 = _slice(g, "input", [4], [6], [2])         # rows 4..5
    r05 = _slice(g, "input", [0], [1], [2])         # row 0
    r56 = _slice(g, "input", [5], [6], [2])         # row 5
    s2 = _slice(g, "input", [2], [5], [2])          # rows 2..4  (window for p=4)
    s1 = _slice(g, "input", [1], [4], [2])          # rows 1..3  (window for p=5)

    m3 = _eq(g, r03, s3, half)
    m4 = _eq(g, r02, r46, half)
    m5 = _eq(g, r05, r56, half)

    nm3 = g.nd("Sub", [one, m3])
    nm4 = g.nd("Sub", [one, m4])
    c3 = m3
    c2 = g.nd("Mul", [nm3, m4])
    c1 = g.nd("Mul", [g.nd("Mul", [nm3, nm4]), m5])

    w3 = g.nd("Mul", [c3, s3])
    w2 = g.nd("Mul", [c2, s2])
    w1 = g.nd("Mul", [c1, s1])
    window = g.nd("Add", [g.nd("Add", [w3, w2]), w1])      # [1,10,3,30]

    cat9 = g.nd("Concat", [top06, window], axis=2)         # [1,10,9,30]
    # recolor 1->2 : out ch j <- in ch idx[j]; in only has ch 0,1 populated,
    # so route empty outputs through an always-empty channel (9).
    idx = g.i64([0, 9, 1, 9, 9, 9, 9, 9, 9, 9])
    rec9 = g.nd("Gather", [cat9, idx], axis=1)             # [1,10,9,30]
    g.nd("Pad", [rec9], "output", mode="constant", value=0.0,
         pads=[0, 0, 0, 0, 0, 0, 21, 0])
    return _model(g)


# ===========================================================================
# 295  golf_stair: 1xW -> (W/2)xW staircase fill of a single colour
# ===========================================================================
def _mir_295(a):
    if a.shape[0] != 1:
        return None
    row = a[0]
    Wn = len(row)
    nz = row[row != 0]
    if len(nz) == 0 or len(set(nz.tolist())) != 1:
        return None
    c = int(nz[0])
    K = len(nz)
    if not (row[:K] == c).all() or (row[K:] != 0).any():
        return None
    Hn = Wn // 2
    out = np.zeros((Hn, Wn), int)
    for i in range(Hn):
        for j in range(Wn):
            if j - i <= K - 1:
                out[i][j] = c
    return out


def _build_295(g):
    half = g.f([1, 1, 1, 1], [0.5])
    Dmat = g.f([1, 1, H, W],
               (np.arange(W)[None, :] - np.arange(H)[:, None]).astype(np.float32))
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))

    Wn = g.nd("ReduceSum", ["input"], axes=[1, 2, 3], keepdims=1)     # grid width
    ch0 = _slice(g, "input", [0], [1], [1])                          # [1,1,30,30]
    Kbg = g.nd("ReduceSum", [ch0], axes=[1, 2, 3], keepdims=1)
    Kn = g.nd("Sub", [Wn, Kbg])                                      # #colour cells

    tri = g.nd("Cast", [g.nd("Less", [Dmat, Kn])], to=F)             # j-i <= K-1
    whalf = g.nd("Mul", [Wn, half])
    rowok = g.nd("Cast", [g.nd("Less", [rowidx, whalf])], to=F)      # i < W/2
    colok = g.nd("Cast", [g.nd("Less", [colidx, Wn])], to=F)         # j < W
    grid = g.nd("Mul", [rowok, colok])                              # [1,1,30,30]
    trim = g.nd("Mul", [tri, grid])                                 # colour cells
    bg = g.nd("Sub", [grid, trim])                                  # background cells

    csel = g.nd("ReduceMax", ["input"], axes=[2, 3], keepdims=1)     # [1,10,1,1]
    csel19 = _slice(g, csel, [1], [10], [1])                        # [1,9,1,1]
    chrest = g.nd("Mul", [trim, csel19])                            # [1,9,30,30]
    g.nd("Concat", [bg, chrest], "output", axis=1)                  # [1,10,30,30]
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []
    out = []

    def emit(name, mirror, build):
        try:
            for a, b in prs:
                o = mirror(a)
                if o is None or o.shape != b.shape or not np.array_equal(o, b):
                    return
            g = _G()
            m = build(g)
            onnx.checker.check_model(m, full_check=True)
            out.append((name, m))
        except Exception:
            pass

    emit("g5_vtile_h9", _mir_003, _build_003)
    emit("g5_golf_stair", _mir_295, _build_295)

    return out
