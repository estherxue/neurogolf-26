"""CRK5_5 family: hard residual tasks (slice U[5::6]).

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
# tiny graph accumulator                                                      #
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


def _slice_ch(g, src, c0, c1):
    """Slice channels [c0,c1) on axis 1."""
    return g.nd("Slice", [src, g.i64([c0]), g.i64([c1]), g.i64([1])])


# ========================================================================== #
# TASK 398 -- diagonal smear of a 1x5 row into an NxN square                  #
# N = 5 * (number of non-zero cells).  Input value at column p is smeared     #
# along the anti-diagonal r+c == (N-1)+p inside the NxN top-left square.      #
# ========================================================================== #
def ref_398(a):
    if a.ndim != 2 or a.shape[0] != 1:
        return None
    row = a[0]
    nz = [(p, int(v)) for p, v in enumerate(row) if v != 0]
    k = len(nz)
    N = 5 * k
    if N <= 0 or N > 30:
        return None
    out = np.zeros((N, N), int)
    for p, v in nz:
        s = (N - 1) + p
        for r in range(N):
            c = s - r
            if 0 <= c < N:
                out[r, c] = v
    return out


def build_398():
    g = _G()
    S = 59     # anti-diagonal index range 0..58 (r+c, r,c in 0..29)
    W5 = 5     # the 1xW input row only ever uses columns 0..4
    # ---- input row 0, colours 1..9, cols 0..4  -> [1,9,5] -----------------
    sl = g.nd("Slice", ["input", g.i64([1, 0, 0]), g.i64([CHANNELS, 1, W5]),
                        g.i64([1, 2, 3])])                              # [1,9,1,5]
    inrow = g.nd("Reshape", [sl, g.i64([1, CHANNELS - 1, W5])])          # [1,9,5]
    # ---- N = 5 * (#non-zero foreground cells) ----------------------------
    ksum0 = g.nd("ReduceSum", [inrow], axes=[0, 1, 2], keepdims=1)        # [1,1,1]
    ksum = g.nd("Reshape", [ksum0, g.i64([1, 1])])                       # [1,1]
    c5 = g.f([1, 1], [5.0])
    c1 = g.f([1, 1], [1.0])
    Nmat = g.nd("Mul", [ksum, c5])                                       # [1,1] = N
    N1mat = g.nd("Sub", [Nmat, c1])                                      # [1,1] = N-1
    # ---- data-dependent shift matrix Sh[p,s] = 1 iff s-p == N-1 ----------
    diff = (np.arange(S)[None, :] - np.arange(W5)[:, None]).astype(np.float32)  # [5,59]
    diffn = g.f([W5, S], diff)
    half = g.f([1, 1], [0.5])
    ad = g.nd("Abs", [g.nd("Sub", [diffn, N1mat])])                      # [5,59]
    Sh = g.nd("Cast", [g.nd("Less", [ad, half])], to=F)                  # [5,59]
    # ---- G[v,s] = inrow[v, s-(N-1)] --------------------------------------
    G = g.nd("MatMul", [inrow, Sh])                                      # [1,9,59]
    # ---- Hankel anti-diagonal broadcast: smear[v,r,c] = G[v, r+c] --------
    idx = (np.add.outer(np.arange(HEIGHT), np.arange(WIDTH))).astype(np.int64)
    idxn = g.nm("idx")
    g.inits.append(oh.make_tensor(idxn, INT64, [HEIGHT, WIDTH], idx.ravel().tolist()))
    smear = g.nd("Gather", [G, idxn], axis=2)                           # [1,9,30,30]
    # ---- NxN square mask, crop -------------------------------------------
    ridx = g.f([HEIGHT, 1], np.arange(HEIGHT))
    cidx = g.f([1, WIDTH], np.arange(WIDTH))
    rok = g.nd("Cast", [g.nd("Less", [ridx, Nmat])], to=F)              # [30,1]
    cok = g.nd("Cast", [g.nd("Less", [cidx, Nmat])], to=F)              # [1,30]
    sq = g.nd("Mul", [rok, cok])                                        # [30,30]
    colored = g.nd("Mul", [smear, sq])                                  # [1,9,30,30]
    anyfg = g.nd("ReduceSum", [colored], axes=[1], keepdims=1)          # [1,1,30,30]
    bg = g.nd("Sub", [sq, anyfg])                                       # [1,1,30,30]
    g.nd("Concat", [bg, colored], "output", axis=1)                    # [1,10,30,30]
    return _model(g)


# ========================================================================== #
# TASK 381 -- horizontal "between two 2s" gap fill with colour 9             #
# A background (0) cell becomes 9 iff there is a 2 strictly to its left AND   #
# a 2 strictly to its right on the same row.                                  #
# ========================================================================== #
def ref_381(a):
    h, w = a.shape
    out = a.copy()
    is2 = (a == 2)
    left = np.zeros((h, w), bool)
    right = np.zeros((h, w), bool)
    for r in range(h):
        seen = False
        for c in range(w):
            left[r, c] = seen
            if is2[r, c]:
                seen = True
        seen = False
        for c in range(w - 1, -1, -1):
            right[r, c] = seen
            if is2[r, c]:
                seen = True
    fill = (a == 0) & left & right
    out[fill] = 9
    return out


def build_381():
    g = _G()
    # strict-upper triangular U[k,c] = 1 iff k < c  (so matmul -> sum to the left)
    U = np.triu(np.ones((WIDTH, WIDTH), np.float32), k=1)
    Un = g.f([WIDTH, WIDTH], U)
    Vn = g.nd("Transpose", [Un], perm=[1, 0])          # strict-lower -> sum to the right

    in2 = _slice_ch(g, "input", 2, 3)                  # [1,1,30,30]
    bg = _slice_ch(g, "input", 0, 1)                   # [1,1,30,30]
    pre = g.nd("MatMul", [in2, Un])                    # count of 2s to the left
    suf = g.nd("MatMul", [in2, Vn])                    # count of 2s to the right
    t = g.nd("Mul", [pre, suf])                        # >0 iff both sides have a 2
    t2 = g.nd("Mul", [t, bg])                          # require background cell
    half = g.f([1, 1, 1, 1], [0.5])
    gb = g.nd("Greater", [t2, half])                   # bool
    between = g.nd("Cast", [gb], to=F)                 # [1,1,30,30] float 0/1

    sel = np.zeros((1, CHANNELS, 1, 1), np.float32)
    sel[0, 0, 0, 0] = -1.0                             # remove from background channel
    sel[0, 9, 0, 0] = +1.0                             # add to colour-9 channel
    seln = g.f([1, CHANNELS, 1, 1], sel)
    delta = g.nd("Mul", [between, seln])               # [1,10,30,30]
    g.nd("Add", ["input", delta], "output")
    return _model(g)


# ========================================================================== #
# TASK 231 -- horizontal period extension to double width                     #
# output[r][c] = input[r][c mod d]  for c in [0, 2W), where d is the smallest #
# horizontal period of the pattern (autocorrelation).                         #
# ========================================================================== #
QMAX_231 = 3


def _per_row(a, qmax):
    """Smallest d in 1..qmax with zero pattern mismatch over the real region
    (mirrors the ONNX autocorrelation exactly)."""
    H, W = a.shape
    for d in range(1, qmax + 1):
        if W - d <= 0:
            score = 0
        else:
            score = int(np.sum(a[:, :W - d] != a[:, d:]))
        if score == 0:
            return d
    return 0


def ref_231(a):
    H, W = a.shape
    d = _per_row(a, QMAX_231)
    if d == 0 or 2 * W > 30:
        return None
    out = np.zeros((H, 2 * W), int)
    for r in range(H):
        for c in range(2 * W):
            out[r, c] = a[r, c % d]
    return out


def _shift_left(g, t, d, axis):
    """out[...,j] = t[...,j+d] along axis (2 rows / 3 cols), zero filled."""
    if axis == 3:
        sl = g.nd("Slice", [t, g.i64([d]), g.i64([HEIGHT]), g.i64([3])])
        return g.nd("Pad", [sl], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, d])
    sl = g.nd("Slice", [t, g.i64([d]), g.i64([HEIGHT]), g.i64([2])])
    return g.nd("Pad", [sl], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, d, 0])


def _period_scalar(g, known, color, axis, half2, halfc, qmax):
    """Smallest static d in 1..QMAX whose autocorrelation is 0, as a [1,1] float."""
    qstar = None
    remaining = None
    one = g.f([1, 1], [1.0])
    for d in range(1, qmax + 1):
        shK = _shift_left(g, known, d, axis)
        shC = _shift_left(g, color, d, axis)
        both = g.nd("Mul", [known, shK])
        diff = g.nd("Abs", [g.nd("Sub", [color, shC])])
        mism = g.nd("Cast", [g.nd("Greater", [diff, halfc])], to=F)
        score = g.nd("ReduceSum", [g.nd("Mul", [both, mism])], axes=[2, 3], keepdims=0)  # [1,1]
        zero = g.nd("Cast", [g.nd("Less", [score, half2])], to=F)        # 1 iff period
        if remaining is None:
            gate = zero
            remaining = g.nd("Sub", [one, zero])
        else:
            gate = g.nd("Mul", [zero, remaining])
            remaining = g.nd("Mul", [remaining, g.nd("Sub", [one, zero])])
        contrib = g.nd("Mul", [gate, g.f([1, 1], [float(d)])])
        qstar = contrib if qstar is None else g.nd("Add", [qstar, contrib])
    return qstar


def build_231():
    g = _G()
    half2 = g.f([1, 1], [0.5])
    halfc = g.f([1, 1, 1, 1], [0.5])
    colvec = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    color = g.nd("ReduceSum", [g.nd("Mul", ["input", colvec])], axes=[1], keepdims=1)  # [1,1,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)                       # [1,1,30,30]
    d_h = _period_scalar(g, realmask, color, 3, half2, halfc, QMAX_231)                           # [1,1]

    # P[k,c] = 1 iff k == c mod d  (data-dependent column gather)
    cidx = g.f([1, HEIGHT], list(range(HEIGHT)))                                        # [1,30]
    cmod = g.nd("Mod", [cidx, d_h], fmod=1)                                             # [1,30]
    src = g.f([HEIGHT, 1], list(range(HEIGHT)))                                         # [30,1]
    P = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [src, cmod])]), half2])], to=F)  # [30,30]
    content = g.nd("MatMul", ["input", P])                                              # [1,10,30,30]

    # width mask: keep cols < 2W, where W = #in-grid columns
    colsum = g.nd("ReduceSum", [realmask], axes=[2], keepdims=1)                        # [1,1,1,30]
    colpres = g.nd("Cast", [g.nd("Greater", [colsum, halfc])], to=F)
    W = g.nd("ReduceSum", [colpres], axes=[3], keepdims=1)                              # [1,1,1,1]
    twoW = g.nd("Mul", [W, g.f([1, 1, 1, 1], [2.0])])
    cidx4 = g.f([1, 1, 1, HEIGHT], list(range(HEIGHT)))
    wmask = g.nd("Cast", [g.nd("Less", [cidx4, twoW])], to=F)                           # [1,1,1,30]
    g.nd("Mul", [content, wmask], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# pair extraction / matching                                                  #
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


def _all_match(prs, ref):
    for a, b in prs:
        try:
            o = ref(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


# registry of (name, ref, builder, needs_same_dims)
_TASKS = [
    ("diag_smear_398", ref_398, build_398, False),
    ("hperiod_x2_231", ref_231, build_231, False),
]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    for name, ref, build, same in _TASKS:
        if same and any(a.shape != b.shape for a, b in prs):
            continue
        if all(np.array_equal(a, b) for a, b in prs):
            continue
        if not _all_match(prs, ref):
            continue
        try:
            m = build()
            onnx.checker.check_model(m, full_check=True)
        except Exception:
            continue
        out.append((name, m))
    return out
