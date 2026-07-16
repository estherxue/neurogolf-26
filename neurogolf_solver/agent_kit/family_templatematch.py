"""TEMPLATE MATCHING via Conv correlation, then stamp/recolor the occurrences.

Idea (origin-anchored, opset-10)
--------------------------------
A fixed small sub-pattern (the TEMPLATE ``T``, ``th x tw`` over colours, possibly
including background) is detected EVERYWHERE it occurs in the grid, and every
occurrence is overwritten in-place by a fixed output sub-pattern (the STAMP
``Q``, same shape); cells outside every occurrence keep their original colour.
This covers "find the X and highlight it" and "recolor every occurrence of
pattern P" (``Q`` solid -> recolor-whole-window; ``Q`` decorated -> stamp a motif).

Exact match as a single correlation Conv
----------------------------------------
The input is a one-hot ``[1,10,30,30]`` tensor (grid at the top-left, zero
padded).  Build a Conv kernel ``Wt[0, T[r,c], r, c] = 1`` (one tap per template
cell, on the channel of that cell's colour).  Then

    response[i,j] = Conv(input, Wt)[i,j]
                  = #{(r,c) : input has colour T[r,c] at (i+r, j+c)}

Because every real cell is one-hot, each tap contributes at most 1, so
``response <= th*tw`` and ``response == th*tw`` IFF the whole window equals ``T``.
Threshold ``response > th*tw - 0.5`` -> a {0,1} hit map at the window's top-left
corner.  Padding can never complete a match (padding cells are all-zero, so they
match neither a coloured nor a background template cell), hence matches occur only
fully inside the real grid -> ORIGIN SAFE for grids of any size.

Stamping the output template with a second Conv
-----------------------------------------------
"Spread" each corner hit over its ``th x tw`` window and paint the stamp colours
with one more Conv.  With the hit map zero-padded back to ``[1,1,30,30]`` and a
kernel built from the 180-rotated stamp,

    Wstamp[o, 0, r, c] = 1  iff  Q[th-1-r, tw-1-c] == o,

a Conv with ``pads=[th-1, tw-1, 0, 0]`` (cross-correlation dilation) yields
``cover[1,10,30,30]`` where ``cover[o,y,x]`` counts occurrences that paint colour
``o`` onto cell ``(y,x)``.  ``ReduceSum`` over channels -> a "covered" mask, and a
single ``Where(covered, cover, input)`` keeps the rest of the grid.  The grader
thresholds ``output > 0``, so the integer counts in ``cover`` decode to the right
colours directly (no extra threshold needed).  Only ONE full ``[1,10,30,30]``
intermediate is materialised, so the cost stays low.

Detection derives ``(T, Q)`` from the train pairs (every window position that
changes), then keeps only the smallest template that reproduces EVERY available
train+test+arc-gen pair EXACTLY (the grader's gate); wrong hypotheses are dropped
before scoring, and the minimal template is the most structural / generalising
one.
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

    def cf(self, dims, vals):
        n = self.nm("cf")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
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
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def build_stamp(T, Q):
    """T, Q : int arrays of identical shape (th, tw).  Detect every exact
    occurrence of T and overwrite it in place with Q, keeping the rest."""
    T = np.asarray(T, int)
    Q = np.asarray(Q, int)
    th, tw = T.shape
    ncells = th * tw
    g = _G()

    # template correlation kernel [1,10,th,tw]
    Wt = np.zeros((1, CHANNELS, th, tw), np.float32)
    for r in range(th):
        for c in range(tw):
            Wt[0, int(T[r, c]), r, c] = 1.0
    wt = g.cf([1, CHANNELS, th, tw], Wt)
    resp = g.nd("Conv", ["input", wt], kernel_shape=[th, tw], pads=[0, 0, 0, 0])

    thr = g.cf([1, 1, 1, 1], [ncells - 0.5])
    hits = g.nd("Cast", [g.nd("Greater", [resp, thr])], to=F)        # [1,1,Hh,Wh]
    hits30 = g.nd("Pad", [hits], mode="constant", value=0.0,
                  pads=[0, 0, 0, 0, 0, 0, th - 1, tw - 1])           # [1,1,30,30]

    # stamp kernel from the 180-rotated Q -> [10,1,th,tw]
    Qf = Q[::-1, ::-1]
    Ws = np.zeros((CHANNELS, 1, th, tw), np.float32)
    for r in range(th):
        for c in range(tw):
            Ws[int(Qf[r, c]), 0, r, c] = 1.0
    ws = g.cf([CHANNELS, 1, th, tw], Ws)
    cover = g.nd("Conv", [hits30, ws], kernel_shape=[th, tw],
                 pads=[th - 1, tw - 1, 0, 0])                        # [1,10,30,30]

    usum = g.nd("ReduceSum", [cover], axes=[1], keepdims=1)          # [1,1,30,30]
    half = g.cf([1, 1, 1, 1], [0.5])
    cond = g.nd("Cast", [g.nd("Greater", [usum, half])], to=BOOL)    # [1,1,30,30]
    g.nd("Where", [cond, cover, "input"], "output")                  # [1,10,30,30]
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX semantics exactly)                        #
# --------------------------------------------------------------------------- #
def _apply_stamp(a, T, Q):
    th, tw = T.shape
    H, W = a.shape
    if th > H or tw > W:
        return None
    stamped = np.full((H, W), -1, int)
    for i in range(H - th + 1):
        for j in range(W - tw + 1):
            if np.array_equal(a[i:i + th, j:j + tw], T):
                for r in range(th):
                    for c in range(tw):
                        y, x = i + r, j + c
                        col = int(Q[r, c])
                        if stamped[y, x] == -1:
                            stamped[y, x] = col
                        elif stamped[y, x] != col:
                            return None         # conflicting overlap -> invalid
    out = a.copy()
    m = stamped != -1
    out[m] = stamped[m]
    return out


def _derive(a, b, maxh=5, maxw=5, maxcells=25):
    """All (T, Q) templates that reproduce a->b for this single pair."""
    if a.shape != b.shape or not (a != b).any():
        return {}
    H, W = a.shape
    found = {}
    for th in range(1, maxh + 1):
        for tw in range(1, maxw + 1):
            if th * tw < 2 or th * tw > maxcells or th > H or tw > W:
                continue
            groups = {}
            for i in range(H - th + 1):
                for j in range(W - tw + 1):
                    groups.setdefault(a[i:i + th, j:j + tw].tobytes(), []).append((i, j))
            for lst in groups.values():
                i0, j0 = lst[0]
                T = a[i0:i0 + th, j0:j0 + tw]
                if not (T != 0).any():            # all-background template -> skip
                    continue
                Q = b[i0:i0 + th, j0:j0 + tw]
                if np.array_equal(Q, T):          # nothing changes here -> skip
                    continue
                # Q must be identical at every occurrence of this window
                if any(not np.array_equal(b[i:i + th, j:j + tw], Q) for i, j in lst):
                    continue
                key = (th, tw, T.tobytes(), Q.tobytes())
                if key in found:
                    continue
                r = _apply_stamp(a, T.copy(), Q.copy())
                if r is not None and np.array_equal(r, b):
                    found[key] = (T.copy(), Q.copy())
    return found


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def _pairs(ex, which=("train", "test", "arc-gen")):
    out = []
    for s in which:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):       # in-place stamp -> same shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []

    # derive candidate templates from the CHANGED train pairs, intersect
    train = _pairs(ex, ("train",))
    seed = [(a, b) for a, b in train if (a != b).any()]
    if not seed:                                       # fall back to any changed pair
        seed = [(a, b) for a, b in prs if (a != b).any()][:3]
    if not seed:
        return []

    keep = None
    store = {}
    for a, b in seed:
        d = _derive(a, b)
        store.update(d)
        keep = set(d) if keep is None else (keep & set(d))
        if not keep:
            return []

    # validate survivors against EVERY available pair (the grader's gate)
    valid = []
    for k in keep:
        T, Q = store[k]
        ok = True
        for a, b in prs:
            r = _apply_stamp(a, T, Q)
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            valid.append((T, Q))
    if not valid:
        return []

    # smallest template = most parsimonious / structural (best points); emit the
    # few smallest as cheap insurance for the held-out gate (the grader keeps the
    # highest-scoring candidate that still reproduces the private pairs).
    valid.sort(key=lambda tq: (tq[0].size, tq[0].shape[0], tq[0].shape[1]))
    out = []
    seen = set()
    for T, Q in valid:
        key = (T.tobytes(), Q.tobytes())
        if key in seen:
            continue
        seen.add(key)
        try:
            model = build_stamp(T, Q)
            onnx.checker.check_model(model, full_check=True)
            out.append((f"tmatch_{T.shape[0]}x{T.shape[1]}_{len(out)}", model))
        except Exception:
            continue
        if len(out) >= 3:
            break
    return out
