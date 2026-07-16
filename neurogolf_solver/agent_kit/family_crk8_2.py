"""family_crk8_2 — diagonal-ray solvers (slice U[2::6]).

Task 324: two background colors tile the grid; each background region contains a
single 'seed' cell of a rare color.  An X (both diagonals) is drawn through every
seed across the whole grid; a cell lying on such a diagonal is recoloured to the
seed colour that lives in *its own* background region (background->ray mapping is
global).  Everything is computed at runtime from the one-hot input, so a single
static graph generalises to every example.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64
FLOAT = DATA_TYPE


# ----------------------------------------------------------------------------
# numpy reference (mirrors the ONNX graph exactly) — used for detection
# ----------------------------------------------------------------------------
def _onehot(a):
    H, W = a.shape
    g = np.zeros((10, 30, 30), np.float32)
    for k in range(10):
        g[k, :H, :W] = (a == k)
    return g


def _shift(x, dr, dc):
    out = np.zeros_like(x)
    H, W = x.shape[-2:]
    sr0, sr1 = max(-dr, 0), H - max(dr, 0)
    dr0, dr1 = max(dr, 0), H - max(-dr, 0)
    sc0, sc1 = max(-dc, 0), W - max(dc, 0)
    dc0, dc1 = max(dc, 0), W - max(-dc, 0)
    out[..., dr0:dr1, dc0:dc1] = x[..., sr0:sr1, sc0:sc1]
    return out


def _prop(mask):
    def fill(m, sgn):
        cur = m.copy()
        for k in (1, 2, 4, 8, 16):
            cur = np.maximum(cur, _shift(cur, k, k * sgn))
        for k in (1, 2, 4, 8, 16):
            cur = np.maximum(cur, _shift(cur, -k, -k * sgn))
        return cur
    return np.maximum(fill(mask, 1), fill(mask, -1))


def _solve_ref(a):
    g = _onehot(a)
    flat = g.reshape(10, -1)
    NBc = np.zeros((10, 30, 30), np.float32)
    for k in range(10):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            NBc[k] += _shift(g[k], dr, dc)
    adj = flat @ NBc.reshape(10, -1).T            # adj[i,j]
    M = np.zeros((10, 10), np.float32)            # M[b,k]
    isseed = np.zeros(10, np.float32)
    for j in range(1, 10):
        if adj[j, j] < adj[j].max():              # seed: own-colour adjacency not dominant
            isseed[j] = 1.0
            M[int(np.argmax(adj[j])), j] = 1.0
    seedmask = np.zeros((30, 30), np.float32)
    for k in range(10):
        if isseed[k]:
            seedmask = np.maximum(seedmask, g[k])
    D = _prop(seedmask)
    T = np.zeros((10, 10), np.float32)
    for b in range(10):
        if M[b].sum() > 0:
            T[b] = M[b]
        else:
            T[b, b] = 1.0
    Tmix = np.einsum('bp,bk->kp', flat, T).reshape(10, 30, 30)
    out = g * (1 - D) + Tmix * D
    return out.argmax(0)


def _matches(examples):
    pairs = [(np.array(e["input"], int), np.array(e["output"], int))
             for e in examples.get("train", []) + examples.get("test", [])]
    if not pairs:
        return False
    for a, b in pairs:
        if a.shape != b.shape:
            return False
        pred = _solve_ref(a)[:b.shape[0], :b.shape[1]]
        if not (pred == b).all():
            return False
    # require the structure we expect (>=1 seed, exactly 2 backgrounds-ish)
    a0 = pairs[0][0]
    g = _onehot(a0)
    if g.reshape(10, -1).sum(1)[1:].astype(bool).sum() < 3:
        return False
    return True


# ----------------------------------------------------------------------------
# ONNX builders
# ----------------------------------------------------------------------------
def _const(name, arr, dtype=FLOAT):
    arr = np.asarray(arr)
    return oh.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist())


class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def uid(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def node(self, op, ins, out=None, **attr):
        out = out or self.uid(op.lower())
        self.nodes.append(oh.make_node(op, ins, [out], **attr))
        return out

    def const(self, arr, dtype=FLOAT, name=None):
        name = name or self.uid("c")
        self.inits.append(_const(name, arr, dtype))
        return name

    def shift(self, x, dr, dc):
        """out[r,c] = x[r-dr, c-dc] with zero fill (content moves by dr,dc)."""
        h0, w0 = max(dr, 0), max(dc, 0)
        h1, w1 = max(-dr, 0), max(-dc, 0)
        padded = self.node("Pad", [x], mode="constant", value=0.0,
                            pads=[0, 0, h0, w0, 0, 0, h1, w1])
        s = self.const([h1, w1], INT64)
        e = self.const([h1 + 30, w1 + 30], INT64)
        ax = self.const([2, 3], INT64)
        return self.node("Slice", [padded, s, e, ax])

    def maxshift(self, cur, dr, dc):
        s = self.shift(cur, dr, dc)
        return self.node("Max", [cur, s])


def _build():
    g = _G()
    X = "input"
    sh_10_900 = g.const([10, 900], INT64)
    sh_grid = g.const([1, 1, 30, 30], INT64)
    sh_full = g.const([1, 10, 30, 30], INT64)
    Xflat = g.node("Reshape", [X, sh_10_900])               # [10,900]

    # neighbour counts NBc (sum of 4 orthogonal shifts of the full tensor)
    su = g.shift(X, -1, 0)
    sd = g.shift(X, 1, 0)
    sl = g.shift(X, 0, -1)
    sr = g.shift(X, 0, 1)
    NBc = g.node("Add", [su, sd])
    NBc = g.node("Add", [NBc, sl])
    NBc = g.node("Add", [NBc, sr])                          # [1,10,30,30]
    NBcf = g.node("Reshape", [NBc, sh_10_900])              # [10,900]
    NBcfT = g.node("Transpose", [NBcf], perm=[1, 0])        # [900,10]
    adj = g.node("MatMul", [Xflat, NBcfT])                  # [10,10] adj[i,j]

    I10 = g.const(np.eye(10, dtype=np.float32))
    adj_diag = g.node("Mul", [adj, I10])
    adj_diag = g.node("ReduceSum", [adj_diag], axes=[1], keepdims=1)   # [10,1]
    adj_max = g.node("ReduceMax", [adj], axes=[1], keepdims=1)         # [10,1]
    isseed_lt = g.node("Less", [adj_diag, adj_max])
    isseed = g.node("Cast", [isseed_lt], to=FLOAT)          # [10,1]

    dom = g.node("ArgMax", [adj], axis=1, keepdims=1)       # [10,1] int64
    domf = g.node("Cast", [dom], to=FLOAT)                  # [10,1]
    brange = g.const(np.arange(10, dtype=np.float32).reshape(1, 10))  # [1,10]
    diff = g.node("Sub", [domf, brange])                   # [10,10]
    diff = g.node("Abs", [diff])
    half = g.const(np.float32(0.5).reshape(1))
    domOH = g.node("Less", [diff, half])
    domOH = g.node("Cast", [domOH], to=FLOAT)              # [10,10] domOH[k,b]
    MdomT = g.node("Mul", [domOH, isseed])                # [k,b]  (isseed[k] broadcast)
    M = g.node("Transpose", [MdomT], perm=[1, 0])         # [b,k]

    rowsum = g.node("ReduceSum", [M], axes=[1], keepdims=1)  # [10,1]
    half2 = g.const(np.float32(0.5).reshape(1))
    hasray = g.node("Greater", [rowsum, half2])
    hasray = g.node("Cast", [hasray], to=FLOAT)            # [10,1]
    one = g.const(np.float32(1.0).reshape(1))
    nothas = g.node("Sub", [one, hasray])
    Tray = g.node("Mul", [hasray, M])
    Tid = g.node("Mul", [nothas, I10])
    T = g.node("Add", [Tray, Tid])                         # [b,k]
    Ttrans = g.node("Transpose", [T], perm=[1, 0])        # [k,b]
    Tmix = g.node("MatMul", [Ttrans, Xflat])              # [10,900]
    Tmixr = g.node("Reshape", [Tmix, sh_full])            # [1,10,30,30]

    # seed mask + diagonal propagation
    isseedR = g.node("Transpose", [isseed], perm=[1, 0])  # [1,10]
    seedflat = g.node("MatMul", [isseedR, Xflat])         # [1,900]
    seed = g.node("Reshape", [seedflat, sh_grid])         # [1,1,30,30]

    def diag(sgn):
        cur = seed
        for k in (1, 2, 4, 8, 16):
            cur = g.maxshift(cur, k, k * sgn)
        for k in (1, 2, 4, 8, 16):
            cur = g.maxshift(cur, -k, -k * sgn)
        return cur
    D = g.node("Max", [diag(1), diag(-1)])                # [1,1,30,30]

    nD = g.node("Sub", [one, D])
    keep = g.node("Mul", [X, nD])
    rec = g.node("Mul", [Tmixr, D])
    g.node("Add", [keep, rec], out="output")

    x = oh.make_tensor_value_info("input", FLOAT, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", FLOAT, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "crk8_2_324", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ----------------------------------------------------------------------------
# Task 110 — variable-period 2D pattern repair
# ----------------------------------------------------------------------------
_PLIST = list(range(1, 16)) + [30]


def _sh_rows(x, p):
    out = np.zeros_like(x)
    out[..., :30 - p, :] = x[..., p:, :]
    return out


def _sh_cols(x, p):
    out = np.zeros_like(x)
    out[..., :, :30 - p] = x[..., :, p:]
    return out


def _pick_period(X, axis):
    present = X[1:].sum(0)
    counts = []
    for p in _PLIST:
        if axis == 0:
            Xp = _sh_rows(X, p); pp = _sh_rows(present, p)
        else:
            Xp = _sh_cols(X, p); pp = _sh_cols(present, p)
        agree = (X[1:] * Xp[1:]).sum(0)
        both = present * pp
        counts.append(float((both - agree).sum()))
    counts = np.array(counts)
    prio = (counts < 0.5).astype(np.float32) * (100 - np.array(_PLIST))
    return _PLIST[int(np.argmax(prio))]


def _tmat(p):
    I = np.arange(30)[:, None]; J = np.arange(30)[None, :]
    return ((I - J) % p == 0).astype(np.float32)


def _solve110(a):
    X = _onehot(a)
    py = _pick_period(X, 0); px = _pick_period(X, 1)
    Tr = _tmat(py); Tc = _tmat(px)
    outsum = np.einsum('ij,kjw,wc->kic', Tr, X, Tc)
    grid = X.sum(0)
    res = np.zeros((30, 30), int)
    for k in range(1, 10):
        res[(outsum[k] > 0) & (grid > 0)] = k
    return res


def _matches110(examples):
    pairs = [(np.array(e["input"], int), np.array(e["output"], int))
             for e in examples.get("train", []) + examples.get("test", [])]
    if not pairs:
        return False
    for a, b in pairs:
        if a.shape != b.shape:
            return False
        if (a == 0).sum() == 0:          # need a hole region
            return False
        pred = _solve110(a)[:b.shape[0], :b.shape[1]]
        if not (pred == b).all():
            return False
    return True


def _build110():
    g = _G()
    X = "input"
    chan = np.ones((1, 10, 1, 1), np.float32); chan[0, 0, 0, 0] = 0.0
    chanmask = g.const(chan)
    present = g.node("Mul", [X, chanmask])
    present = g.node("ReduceSum", [present], axes=[1], keepdims=1)   # [1,1,30,30]
    grid = g.node("ReduceSum", [X], axes=[1], keepdims=1)           # [1,1,30,30]
    sh1 = g.const([1], INT64)

    def mism(dr, dc):
        Xp = g.shift(X, dr, dc)
        prod = g.node("Mul", [X, Xp])
        prod = g.node("Mul", [prod, chanmask])
        agree = g.node("ReduceSum", [prod], axes=[1], keepdims=1)
        pp = g.shift(present, dr, dc)
        both = g.node("Mul", [present, pp])
        d = g.node("Sub", [both, agree])
        c = g.node("ReduceSum", [d], axes=[0, 1, 2, 3], keepdims=1)  # [1,1,1,1]
        return g.node("Reshape", [c, sh1])                           # [1]

    plist_arr = np.array(_PLIST, np.float32)

    def select(axis):
        cnts = [mism(-p, 0) if axis == 0 else mism(0, -p) for p in _PLIST]
        counts = g.node("Concat", cnts, axis=0)                      # [16]
        half = g.const(np.float32([0.5]))
        valid = g.node("Less", [counts, half])
        valid = g.node("Cast", [valid], to=FLOAT)
        wpri = g.const(100.0 - plist_arr)
        prio = g.node("Mul", [valid, wpri])
        idx = g.node("ArgMax", [prio], axis=0, keepdims=1)           # [1]
        return idx

    stack = np.stack([_tmat(p) for p in _PLIST]).astype(np.float32)  # [16,30,30]
    Tstack = g.const(stack)
    pyidx = select(0)
    pxidx = select(1)
    Tr = g.node("Gather", [Tstack, pyidx], axis=0)                   # [1,30,30]
    Tr = g.node("Squeeze", [Tr], axes=[0])                           # [30,30]
    Tc = g.node("Gather", [Tstack, pxidx], axis=0)
    Tc = g.node("Squeeze", [Tc], axes=[0])

    mid = g.node("MatMul", [Tr, X])                                  # [1,10,30,30]
    outsum = g.node("MatMul", [mid, Tc])
    masked = g.node("Mul", [outsum, grid])
    g.node("Mul", [masked, chanmask], out="output")

    x = oh.make_tensor_value_info("input", FLOAT, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", FLOAT, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "crk8_2_110", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(examples):
    out = []
    try:
        if _matches(examples):
            out.append(("diagray", _build()))
    except Exception:
        pass
    try:
        if _matches110(examples):
            out.append(("period2d", _build110()))
    except Exception:
        pass
    return out
