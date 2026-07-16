"""CRK6_5 family: hard residual tasks (slice U[5::6]).

Each task gets a dedicated numpy reference + ONNX builder.  A candidate is emitted
only when the reference reproduces EVERY available train/test/arc-gen pair, so a
builder fires only for the task(s) whose semantics it actually matches.

Opset-10, FLOAT[1,10,30,30] one-hot in/out, cost = params + intermediate_memory.
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


# ========================================================================== #
# TASK 63 -- "wall-to-wall sight" fill.                                       #
#                                                                            #
# The grid is a closed container whose border / interior structure is drawn  #
# in non-background colours.  A background (0) cell is repainted with colour  #
# K iff it lies on a row whose interior is entirely empty (clear horizontal  #
# sight from wall to wall) OR a column whose interior is entirely empty.      #
#                                                                            #
# Per-cell formulation (works for 1-thick walls):                            #
#   fg(r,c)        = cell is non-background                                   #
#   leftHas/right  = a non-bg cell exists strictly to the left / right        #
#   enclH          = leftHas AND rightHas                                     #
#   encNZ_h        = fg AND enclH    (an "interior wall")                     #
#   rowFree(r)     = no interior wall on row r                                #
#   fillH(r,c)     = bg AND enclH AND rowFree(r)                              #
#   (vertical analogue with up/down -> enclV, colFree -> fillV)              #
#   fill = fillH OR fillV ; output paints those cells colour K.              #
# ========================================================================== #
def ref_fill(a, K):
    H, W = a.shape
    nz = (a != 0).astype(int)
    leftH = np.zeros_like(nz)
    rightH = np.zeros_like(nz)
    upH = np.zeros_like(nz)
    downH = np.zeros_like(nz)
    for r in range(H):
        for c in range(W):
            leftH[r, c] = 1 if nz[r, :c].any() else 0
            rightH[r, c] = 1 if nz[r, c + 1:].any() else 0
            upH[r, c] = 1 if nz[:r, c].any() else 0
            downH[r, c] = 1 if nz[r + 1:, c].any() else 0
    enclH = leftH & rightH
    enclV = upH & downH
    rowFree = ((nz & enclH).sum(axis=1) == 0)
    colFree = ((nz & enclV).sum(axis=0) == 0)
    out = a.copy()
    for r in range(H):
        for c in range(W):
            if a[r, c] == 0:
                fh = enclH[r, c] and rowFree[r]
                fv = enclV[r, c] and colFree[c]
                if fh or fv:
                    out[r, c] = K
    return out


def build_fill(K):
    g = _G()
    # strict-upper (A[w,c]=1 iff w<c) and strict-lower (B[w,c]=1 iff w>c)
    idx = np.arange(30)
    A = (idx[:, None] < idx[None, :]).astype(np.float32)   # [30,30]
    B = (idx[:, None] > idx[None, :]).astype(np.float32)   # [30,30]
    An = g.f([30, 30], A)
    Bn = g.f([30, 30], B)
    zero = g.f([1], [0.0])
    half = g.f([1], [0.5])
    # delta selector: at a painted cell move the one-hot from channel 0 to channel K
    d_sel = np.zeros((1, CHANNELS, 1, 1), np.float32)
    d_sel[0, 0, 0, 0] = -1.0
    d_sel[0, K, 0, 0] += 1.0
    dK = g.f([1, CHANNELS, 1, 1], d_sel)

    # In the grader encoding an in-grid background cell is one-hot at channel 0
    # while padding cells are all-zero.  So:
    #   x0    = channel 0           -> in-grid background mask (never paint padding)
    #   total = sum over channels   -> 1 in-grid, 0 padding
    #   fg    = total - x0          -> foreground mask (0 for background AND padding)
    x0 = g.nd("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])   # [1,1,30,30]
    total = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # [1,1,30,30]
    fg = g.nd("Sub", [total, x0])                                       # [1,1,30,30]

    # horizontal sight
    leftSum = g.nd("MatMul", [fg, An])
    rightSum = g.nd("MatMul", [fg, Bn])
    leftHas = g.nd("Cast", [g.nd("Greater", [leftSum, zero])], to=F)
    rightHas = g.nd("Cast", [g.nd("Greater", [rightSum, zero])], to=F)
    enclH = g.nd("Mul", [leftHas, rightHas])

    # vertical sight (transpose H<->W, reuse A/B, transpose back)
    fgT = g.nd("Transpose", [fg], perm=[0, 1, 3, 2])
    upSumT = g.nd("MatMul", [fgT, An])
    downSumT = g.nd("MatMul", [fgT, Bn])
    upSum = g.nd("Transpose", [upSumT], perm=[0, 1, 3, 2])
    downSum = g.nd("Transpose", [downSumT], perm=[0, 1, 3, 2])
    upHas = g.nd("Cast", [g.nd("Greater", [upSum, zero])], to=F)
    downHas = g.nd("Cast", [g.nd("Greater", [downSum, zero])], to=F)
    enclV = g.nd("Mul", [upHas, downHas])

    # free-row / free-col gates
    encNZh = g.nd("Mul", [fg, enclH])
    rowSum = g.nd("ReduceSum", [encNZh], axes=[3], keepdims=1)           # [1,1,30,1]
    rowFree = g.nd("Cast", [g.nd("Less", [rowSum, half])], to=F)
    encNZv = g.nd("Mul", [fg, enclV])
    colSum = g.nd("ReduceSum", [encNZv], axes=[2], keepdims=1)           # [1,1,1,30]
    colFree = g.nd("Cast", [g.nd("Less", [colSum, half])], to=F)

    # fill = (bg & enclH & rowFree) | (bg & enclV & colFree)
    fillH = g.nd("Mul", [g.nd("Mul", [x0, enclH]), rowFree])
    fillV = g.nd("Mul", [g.nd("Mul", [x0, enclV]), colFree])
    fill = g.nd("Max", [fillH, fillV])                                   # [1,1,30,30]

    # output = input + fill * (eK - e0): moves the painted cells' one-hot 0 -> K
    dfill = g.nd("Mul", [fill, dK])                                      # [1,10,30,30]
    g.nd("Add", ["input", dfill], out="output")
    return _model(g)


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


def _detect_fillcolor(prs):
    """Return K if every change is (0 -> K) for a single K, else None."""
    K = None
    changed = False
    for a, b in prs:
        if a.shape != b.shape:
            return None
        d = a != b
        if not d.any():
            continue
        changed = True
        frm = set(a[d].tolist())
        to = set(b[d].tolist())
        if frm != {0} or len(to) != 1:
            return None
        k = to.pop()
        if K is None:
            K = k
        elif K != k:
            return None
    return K if changed else None


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    # ---- task 63: wall-to-wall sight fill --------------------------------
    K = _detect_fillcolor(prs)
    if K is not None and 0 <= K < CHANNELS:
        ok = True
        for a, b in prs:
            try:
                o = ref_fill(a, K)
            except Exception:
                ok = False
                break
            if o.shape != b.shape or not np.array_equal(o, b):
                ok = False
                break
        if ok:
            try:
                m = build_fill(K)
                onnx.checker.check_model(m, full_check=True)
                out.append((f"sightfill_{K}", m))
            except Exception:
                pass

    return out
