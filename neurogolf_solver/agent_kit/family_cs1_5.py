"""family_cs1_5 — COMPLETE-SWEEP minimal recompiles.

Only tasks whose incumbent (out_blend6) can be strictly beaten are emitted here;
the rest of the requested batch sit at their algorithmic floor and are SKIPped
(see the module-level notes / final report).

task246 (a2fd1cf0, "hpwl" L-connector): the incumbent graph fails to LOAD under
    local ORT 1.23.2 (Min(13) on an int tensor is NOT_IMPLEMENTED) and therefore
    scores 0.  The true rule is a fixed geometric construction: a red dot (colour 2)
    and a green dot (colour 3) are joined by a cyan (colour 8) Manhattan "L" — a
    horizontal run along the RED dot's row between the two columns, then a vertical
    run down the GREEN dot's column between the two rows (the corner cell lands on
    the red row / green column and is cyan; the two endpoints keep their colours).
    We rebuild it from scratch in float space and it loads + is exact.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh
from onnx import TensorProto as TP

from ng_utils_shim import IR_VERSION

F = TP.FLOAT
BOOL = TP.BOOL
G = 30


class _B:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def c(self, arr, dt=F):
        arr = np.asarray(arr)
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, dt, list(arr.shape), arr.ravel().tolist()))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


# --------------------------------------------------------------------------- #
# task246 — a2fd1cf0                                                           #
# --------------------------------------------------------------------------- #
def _mirror_246(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or max(a.shape) > 30:
        return None
    if list(np.unique(a)) not in ([0, 2, 3], [0, 2], [2, 3], [0], [2], [3]):
        # need exactly one red and one green
        pass
    reds = np.argwhere(a == 2)
    grns = np.argwhere(a == 3)
    if reds.shape[0] != 1 or grns.shape[0] != 1:
        return None
    # any other nonzero colours -> not this task
    other = ((a != 0) & (a != 2) & (a != 3)).sum()
    if other:
        return None
    r0, c0 = reds[0]
    r1, c1 = grns[0]
    if r0 == r1 or c0 == c1:
        return None
    out = a.copy()
    lo, hi = sorted([c0, c1])
    out[r0, lo + 1:hi] = 8
    rlo, rhi = sorted([r0, r1])
    out[rlo:rhi + 1, c1] = 8
    out[r0, c0] = 2
    out[r1, c1] = 3
    return out


def _build_246():
    b = _B()
    R = b.c(np.arange(G, dtype=np.float32).reshape(1, 1, G, 1))      # [1,1,30,1]
    C = b.c(np.arange(G, dtype=np.float32).reshape(1, 1, 1, G))      # [1,1,1,30]
    cidx = b.c(np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))  # [1,10,1,1]

    def sc(v):  # scalar float const
        return b.c(np.array(v, np.float32).reshape(1, 1, 1, 1))

    def slc(x, ax, s, e):
        return b.nd("Slice", [x, b.c(np.array([s], np.int64), TP.INT64),
                              b.c(np.array([e], np.int64), TP.INT64),
                              b.c(np.array([ax], np.int64), TP.INT64)])

    rmax = b.nd("ReduceMax", ["input"], axes=[3], keepdims=1)        # [1,10,30,1]
    cmax = b.nd("ReduceMax", ["input"], axes=[2], keepdims=1)        # [1,10,1,30]

    rr = slc(rmax, 1, 2, 3)     # red row indicator  [1,1,30,1]
    gr = slc(rmax, 1, 3, 4)     # green row indicator
    rc = slc(cmax, 1, 2, 3)     # red col indicator   [1,1,1,30]
    gc = slc(cmax, 1, 3, 4)     # green col indicator

    r0 = b.nd("ReduceSum", [b.nd("Mul", [rr, R])], axes=[2], keepdims=1)  # [1,1,1,1]
    r1 = b.nd("ReduceSum", [b.nd("Mul", [gr, R])], axes=[2], keepdims=1)
    c0 = b.nd("ReduceSum", [b.nd("Mul", [rc, C])], axes=[3], keepdims=1)
    c1 = b.nd("ReduceSum", [b.nd("Mul", [gc, C])], axes=[3], keepdims=1)

    # in-grid validity from the (channel-max) presence profiles
    rowany = b.nd("ReduceMax", [rmax], axes=[1], keepdims=1)         # [1,1,30,1]
    colany = b.nd("ReduceMax", [cmax], axes=[1], keepdims=1)         # [1,1,1,30]
    half = sc(0.5)
    validB = b.nd("And", [b.nd("Greater", [rowany, half]),
                          b.nd("Greater", [colany, half])])          # [1,1,30,30] bool

    rowRed = b.nd("Equal", [R, r0])                                  # [1,1,30,1] bool
    colGrn = b.nd("Equal", [C, c1])                                  # [1,1,1,30] bool

    cmn = b.nd("Min", [c0, c1]); cmx = b.nd("Max", [c0, c1])
    rmn = b.nd("Min", [r0, r1]); rmx = b.nd("Max", [r0, r1])
    betC = b.nd("And", [b.nd("Greater", [C, cmn]), b.nd("Less", [C, cmx])])   # [1,1,1,30]
    betR = b.nd("And", [b.nd("Greater", [R, b.nd("Sub", [rmn, half])]),
                        b.nd("Less", [R, b.nd("Add", [rmx, half])])])          # [1,1,30,1]

    cyan = b.nd("Or", [b.nd("And", [rowRed, betC]),
                       b.nd("And", [colGrn, betR])])                 # [1,1,30,30]
    redCell = b.nd("And", [rowRed, b.nd("Equal", [C, c0])])
    grnCell = b.nd("And", [b.nd("Equal", [R, r1]), colGrn])

    # label grid (fp16): precedence green > red > cyan > background(0); outside -> -1
    H = TP.FLOAT16

    def h(v):
        return b.c(np.array(v, np.float16).reshape(1, 1, 1, 1), H)

    Lc = b.nd("Where", [cyan, h(8.0), h(0.0)])                      # [1,1,30,30] fp16
    Lc = b.nd("Where", [redCell, h(2.0), Lc])
    Lc = b.nd("Where", [grnCell, h(3.0), Lc])
    L = b.nd("Where", [validB, Lc, h(-1.0)])
    cidx16 = b.c(np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), H)
    b.nd("Equal", [L, cidx16], "output")                           # [1,10,30,30] bool = output

    x = oh.make_tensor_value_info("input", F, [1, 10, G, G])
    y = oh.make_tensor_value_info("output", BOOL, [1, 10, G, G])
    used = {i for n in b.nodes for i in n.input}
    inits = [t for t in b.inits if t.name in used]
    graph = oh.make_graph(b.nodes, "cs1_5_246", [x], [y], inits)
    m = oh.make_model(graph, ir_version=IR_VERSION,
                      opset_imports=[oh.make_operatorsetid("", 11)])
    onnx.checker.check_model(m, full_check=True)
    return m


# --------------------------------------------------------------------------- #
# routing                                                                      #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            o = np.array(e["output"], int)
            if a.ndim != 2 or o.ndim != 2 or max(a.shape + o.shape) > 30:
                return []
            out.append((a, o))
    return out


def _matches(prs, fn):
    ok = False
    for a, o in prs:
        m = fn(a)
        if m is None or m.shape != o.shape or not np.array_equal(m, o):
            return False
        ok = True
    return ok


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return
    if _matches(prs, _mirror_246):
        try:
            yield ("cs1_5_246", _build_246())
        except Exception:
            pass
