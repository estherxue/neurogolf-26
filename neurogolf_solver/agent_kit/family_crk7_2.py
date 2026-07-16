"""family_crk7_2 -- cracks for slice U[2::6].

Solved here
-----------
* task 58   SQUARE-SPIRAL GENERATION (new).  The input is an all-background NxN
  grid; the output draws an inward square spiral of colour 3.  The spiral is the
  set of "concentric square rings" (cell on iff its border-distance layer
  L = min(i, j, N-1-i, N-1-j) is even) with the connecting seam handled by a
  single parity FLIP along the sub-diagonal: cells (i, i-1) for 1<=i<=M get
  toggled, where M = floor(N/2) - [N mod 4 == 2].  This reproduces every
  train/test/arc-gen pair for sizes 5..20.

  Fully origin-anchored and size-independent: N is read from the real-cell mask,
  the layer / parity / flip masks are pure elementwise arithmetic on coordinate
  grids, and the M threshold is realised as  2*i <= N - (N mod 2) - 2*[N mod 4==2]
  so no division is needed.  Static shapes, no Loop/NonZero.

* task 17   VARIABLE-PERIOD periodic restoration -- delegated to the proven
  autocorrelation + data-dependent doubling-OR builder in family_dynperiod
  (re-exported here so this module also reports the crack).
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
G = HEIGHT  # 30


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

    def i64(self, dims, vals):
        n = self.nm("i")
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


# --------------------------------------------------------------------------- #
# task 58 -- square spiral generation                                          #
# --------------------------------------------------------------------------- #
def build_spiral(color=3):
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])

    Ig = g.f([1, 1, G, G], [[i for _ in range(G)] for i in range(G)])
    Jg = g.f([1, 1, G, G], [[j for j in range(G)] for _ in range(G)])

    # real-cell mask + grid size N
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)            # [1,1,30,30]
    rowcnt = g.nd("ReduceSum", [realmask], axes=[3], keepdims=1)             # [1,1,30,1]
    Nf = g.nd("ReduceMax", [rowcnt], axes=[2], keepdims=1)                   # [1,1,1,1]

    # border-distance layer = min(i, j, N-1-i, N-1-j)
    Nm1 = g.nd("Sub", [Nf, one])
    C = g.nd("Sub", [Nm1, Ig])
    D = g.nd("Sub", [Nm1, Jg])
    layer = g.nd("Min", [Ig, Jg, C, D])                                      # [1,1,30,30]

    # parity (even layer -> ring cell)
    layi = g.nd("Cast", [layer], to=INT64)
    par = g.nd("Mod", [layi, g.i64([1], [2])])
    parf = g.nd("Cast", [par], to=F)
    even = g.nd("Cast", [g.nd("Less", [parf, half])], to=F)                  # 1 where layer even

    # M threshold:  i <= M  <=>  2i <= N - (N mod 2) - 2*[N mod 4 == 2]
    Ni = g.nd("Cast", [Nf], to=INT64)
    Nmod2 = g.nd("Cast", [g.nd("Mod", [Ni, g.i64([1], [2])])], to=F)
    Nmod4 = g.nd("Cast", [g.nd("Mod", [Ni, g.i64([1], [4])])], to=F)
    is2 = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [Nmod4, two])]), half])], to=F)
    thresh = g.nd("Sub", [g.nd("Sub", [Nf, Nmod2]), g.nd("Mul", [two, is2])])  # [1,1,1,1]
    twoI = g.nd("Mul", [Ig, two])
    colcond = g.nd("Cast", [g.nd("Less", [g.nd("Sub", [twoI, thresh]), half])], to=F)

    # sub-diagonal:  i - j == 1
    diff = g.nd("Sub", [Ig, Jg])
    subdiag = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [diff, one])]), half])], to=F)
    flip = g.nd("Mul", [subdiag, colcond])

    spiral_raw = g.nd("Abs", [g.nd("Sub", [even, flip])])                    # XOR(even, flip)
    spiral = g.nd("Mul", [spiral_raw, realmask])                            # mask to grid
    bg = g.nd("Sub", [realmask, spiral])                                    # in-grid, off-spiral

    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    ec = g.f([1, CHANNELS, 1, 1], [1.0 if c == color else 0.0 for c in range(CHANNELS)])
    g.nd("Add", [g.nd("Mul", [bg, e0]), g.nd("Mul", [spiral, ec])], "output")
    return _model(g)


def _spiral_ref(N, color=3):
    I, J = np.indices((N, N))
    L = np.minimum(np.minimum(I, J), np.minimum(N - 1 - I, N - 1 - J))
    even = (L % 2 == 0)
    M = N // 2 - (1 if N % 4 == 2 else 0)
    flip = ((I - J) == 1) & (I <= M)
    spiral = np.logical_xor(even, flip)
    return np.where(spiral, color, 0)


# --------------------------------------------------------------------------- #
# detection / entry point                                                      #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > G or max(b.shape) > G:
                continue
            out.append((a, b))
    return out


def _spiral_candidate(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det:
        return []
    # input must be entirely background, output a same-size square in {0, color}
    colors = set()
    for a, b in allp:
        if not np.all(a == 0):
            return []
        if a.shape != b.shape or b.shape[0] != b.shape[1]:
            return []
        nz = b[b != 0]
        if nz.size == 0:
            return []
        colors |= set(int(v) for v in nz.tolist())
    if len(colors) != 1:
        return []
    color = next(iter(colors))
    for a, b in allp:
        if not np.array_equal(_spiral_ref(b.shape[0], color), b):
            return []
    try:
        model = build_spiral(color)
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    return [(f"spiral_c{color}", model)]


def _periodic_candidate(ex):
    """Delegate variable-period restoration (task 17) to the proven builder."""
    try:
        import family_dynperiod as dp
    except Exception:
        return []
    try:
        return dp.candidates(ex)
    except Exception:
        return []


def candidates(ex):
    out = []
    out += _spiral_candidate(ex)
    out += _periodic_candidate(ex)
    return out
