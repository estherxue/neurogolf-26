"""family_pcrk_1 -- CRACK slice U[1::6] = tasks [18,76,107,158,209,285].

Only task 107 admits an exact, grid-agnostic static-graph rule; the other five in
this slice are per-object / template-stamping transforms whose correct output
positions are data-dependent in ways not expressible as a single static ONNX graph
(they are documented in the report, not emitted here).

=========================== TASK 107 -- fractal frame ===========================
Input is ALWAYS 5x5.  Its last row + last column form a coloured BORDER; the top-left
4x4 is the INTERIOR, which contains exactly one solid colour block plus background.

Rule (derived, verified EXACT on all 266 train+test+arc-gen pairs, and on a 70/30
held-out split):

  k = 1 + (# distinct non-zero colours on the border)          # scale factor 2..6
  OUT is (5k)x(5k):
    * OUT = kron-upscale(input, k)          # each input cell -> a k x k block,
                                            #   so the border becomes a thick frame
                                            #   and the interior block a 2k x 2k block
    * then, from each CORNER of the upscaled interior block, shoot a diagonal ray of
      colour 2 OUTWARD (up-left / up-right / down-left / down-right) across background
      cells, staying inside the 4k x 4k interior region.

Everything is expressed on the fixed [1,10,30,30] one-hot canvas with DATA-DEPENDENT
scalars (k and the block bbox) computed from the input by reductions, so a single
static graph handles every k and every block position.  Upscale is R @ X @ C with
[30,30] selection matrices R[i,s]=[s==floor(i/k)], C[s,j]=[s==floor(j/k)].  Padding
is automatically all-zero because floor(i/k)>=5 indexes the (empty) rows of the 5x5.

Ray membership of an interior background cell (i,j), block bbox [R0..R1]x[C0..C1]:
    up-left  : i-j == R0-C0  and i<R0
    down-right: i-j == R1-C1 and i>R1
    up-right : i+j == R0+C1  and i<R0
    down-left: i+j == R1+C0  and i>R1
(the perpendicular bound follows algebraically from the diagonal + side condition).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT
BOOL = onnx.TensorProto.BOOL


# --------------------------------------------------------------------------- #
# numpy reference (ground truth used for detection / self-check)              #
# --------------------------------------------------------------------------- #
def _rule107(a):
    a = np.asarray(a, int)
    if a.shape != (5, 5):
        return None
    border = np.concatenate([a[4, :], a[:, 4]])
    k = 1 + len({int(x) for x in border if x != 0})
    up = np.kron(a, np.ones((k, k), int))
    out = up.copy()
    IN = 4 * k
    interior = a[0:4, 0:4]
    ys, xs = np.nonzero(interior)
    if len(ys):
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        R0, R1 = r0 * k, (r1 + 1) * k - 1
        C0, C1 = c0 * k, (c1 + 1) * k - 1
        for i in range(IN):
            for j in range(IN):
                if up[i, j] != 0:
                    continue
                if ((i < R0 and (i - j) == (R0 - C0)) or
                        (i > R1 and (i - j) == (R1 - C1)) or
                        (i < R0 and (i + j) == (R0 + C1)) or
                        (i > R1 and (i + j) == (R1 + C0))):
                    out[i, j] = 2
    return out


# --------------------------------------------------------------------------- #
# graph accumulator                                                           #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def const_f(self, arr, name=None):
        arr = np.asarray(arr, np.float32)
        nm = name or self.nm("c")
        self.inits.append(oh.make_tensor(nm, FLOAT, list(arr.shape), arr.flatten().tolist()))
        return nm

    def const_i(self, arr, name=None):
        arr = np.asarray(arr, np.int64)
        nm = name or self.nm("ci")
        self.inits.append(oh.make_tensor(nm, INT64, list(arr.shape), arr.flatten().tolist()))
        return nm

    def n(self, op, ins, out=None, **attrs):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _build107():
    g = _G()
    X = "input"

    # ---- constants ----------------------------------------------------------
    half = g.const_f(0.5)
    one = g.const_f(1.0)
    fourc = g.const_f(4.0)
    big = g.const_f(99.0)
    nbig = g.const_f(-99.0)

    ar = np.arange(30, dtype=np.float32)
    I_col = g.const_f(ar.reshape(30, 1))   # [30,1] row index i
    J_row = g.const_f(ar.reshape(1, 30))   # [1,30] col index j
    idx30 = g.const_f(ar.reshape(30))      # [30]

    Igrid = g.const_f(np.tile(ar.reshape(30, 1), (1, 30)))   # [30,30] i
    Jgrid = g.const_f(np.tile(ar.reshape(1, 30), (30, 1)))   # [30,30] j

    # border mask [1,1,30,30]: row 4 (cols0-4) and col 4 (rows0-4)
    bm = np.zeros((1, 1, 30, 30), np.float32)
    bm[0, 0, 4, 0:5] = 1.0
    bm[0, 0, 0:5, 4] = 1.0
    borderMask = g.const_f(bm)
    # interior mask [1,1,30,30]: i<4 and j<4
    im = np.zeros((1, 1, 30, 30), np.float32)
    im[0, 0, 0:4, 0:4] = 1.0
    interiorMask = g.const_f(im)
    # e2 one-hot colour 2, [1,10,1,1]
    e2v = np.zeros((1, 10, 1, 1), np.float32)
    e2v[0, 2, 0, 0] = 1.0
    e2 = g.const_f(e2v)

    # ---- k = 1 + #distinct nonzero border colours --------------------------
    Xb = g.n("Mul", [X, borderMask])                       # [1,10,30,30]
    Bc = g.n("ReduceSum", [Xb], axes=[0, 2, 3], keepdims=0)  # [10]
    Bc19 = g.n("Slice", [Bc, g.const_i([1]), g.const_i([10]), g.const_i([0])])  # [9]
    present = g.n("Cast", [g.n("Greater", [Bc19, half])], to=FLOAT)  # [9]
    distinct = g.n("ReduceSum", [present], keepdims=0)      # scalar
    k = g.n("Add", [distinct, one])                         # scalar float

    # ---- interior colour mask M [1,1,30,30] --------------------------------
    Xc = g.n("Slice", [X, g.const_i([1]), g.const_i([10]), g.const_i([1])])  # [1,9,30,30]
    nonbg = g.n("ReduceSum", [Xc], axes=[1], keepdims=1)    # [1,1,30,30]
    M = g.n("Mul", [nonbg, interiorMask])                   # [1,1,30,30]

    rowhas = g.n("ReduceMax", [M], axes=[0, 1, 3], keepdims=0)  # [30]
    colhas = g.n("ReduceMax", [M], axes=[0, 1, 2], keepdims=0)  # [30]
    hasBlock = g.n("Greater", [g.n("ReduceMax", [rowhas], keepdims=0), half])  # scalar bool

    def bounds(has):
        cond = g.n("Greater", [has, half])                 # [30] bool
        lo = g.n("ReduceMin", [g.n("Where", [cond, idx30, big])], keepdims=0)   # scalar
        hi = g.n("ReduceMax", [g.n("Where", [cond, idx30, nbig])], keepdims=0)  # scalar
        return lo, hi

    r0, r1 = bounds(rowhas)
    c0, c1 = bounds(colhas)

    # upscaled block bbox (scalars)
    R0 = g.n("Mul", [r0, k])
    C0 = g.n("Mul", [c0, k])
    R1 = g.n("Sub", [g.n("Mul", [g.n("Add", [r1, one]), k]), one])
    C1 = g.n("Sub", [g.n("Mul", [g.n("Add", [c1, one]), k]), one])
    IN = g.n("Mul", [fourc, k])                            # 4k scalar

    # ---- upscale U = R_mat @ X @ C_mat -------------------------------------
    floI = g.n("Floor", [g.n("Div", [I_col, k])])          # [30,1]
    R_mat = g.n("Cast", [g.n("Less", [g.n("Abs", [g.n("Sub", [floI, J_row])]), half])], to=FLOAT)  # [30,30]
    floJ = g.n("Floor", [g.n("Div", [J_row, k])])          # [1,30]
    C_mat = g.n("Cast", [g.n("Less", [g.n("Abs", [g.n("Sub", [I_col, floJ])]), half])], to=FLOAT)  # [30,30]
    Xr = g.n("MatMul", [R_mat, X])                         # [1,10,30,30]
    U = g.n("MatMul", [Xr, C_mat])                         # [1,10,30,30]

    # ---- ray mask m [30,30] -------------------------------------------------
    diffIJ = g.n("Sub", [Igrid, Jgrid])                    # [30,30]
    sumIJ = g.n("Add", [Igrid, Jgrid])                     # [30,30]

    def eqline(expr, val):
        return g.n("Less", [g.n("Abs", [g.n("Sub", [expr, val])]), half])  # [30,30] bool

    lt_R0 = g.n("Less", [Igrid, R0])
    gt_R1 = g.n("Greater", [Igrid, R1])

    ray1 = g.n("And", [eqline(diffIJ, g.n("Sub", [R0, C0])), lt_R0])
    ray4 = g.n("And", [eqline(diffIJ, g.n("Sub", [R1, C1])), gt_R1])
    ray2 = g.n("And", [eqline(sumIJ, g.n("Add", [R0, C1])), lt_R0])
    ray3 = g.n("And", [eqline(sumIJ, g.n("Add", [R1, C0])), gt_R1])

    m0 = g.n("Or", [g.n("Or", [ray1, ray2]), g.n("Or", [ray3, ray4])])
    inbound = g.n("And", [g.n("Less", [Igrid, IN]), g.n("Less", [Jgrid, IN])])
    m = g.n("And", [g.n("And", [m0, inbound]), hasBlock])  # [30,30] bool (hasBlock scalar broadcasts)

    mCond = g.n("Reshape", [g.n("Cast", [m], to=FLOAT), g.const_i([1, 1, 30, 30])])
    mBool = g.n("Cast", [mCond], to=BOOL)                  # [1,1,30,30]

    g.n("Where", [mBool, e2, U], "output")                 # [1,10,30,30]
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# candidate generation                                                        #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2:
                continue
            prs.append((a, b))
    if not prs:
        return []
    # detect task-107 signature: input always 5x5, output (5k)x(5k), rule exact
    if any(a.shape != (5, 5) for a, _ in prs):
        return []
    for a, b in prs:
        r = _rule107(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            return []
    try:
        return [("fractal_frame", _build107())]
    except Exception:
        return []
