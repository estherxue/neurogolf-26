"""family_crk4_5 -- hard-tail crackers (slice IDX=5).

Each task is detected structurally from the train/test/arc-gen pairs and, when the
hypothesis reproduces EVERY pair exactly, an opset-10 ONNX graph (static shapes,
origin-anchored, one-hot FLOAT[1,10,30,30]) is emitted.

Tasks cracked here:
  * task192 -- "denoise to clean rectangles": exactly two nonzero colours, a majority
    colour C (the rectangles) and a minority noise colour N.  Output is monochrome:
    keep every C cell, turn an N cell into C iff it is enclosed -- has a C immediate
    neighbour on (left OR right) AND on (up OR down) -- and drop everything else.
    C/N are picked in-graph by per-channel pixel counts (argmax / min-positive).
  * task397 -- "3-tails below 2x2 blocks": every coloured object is a 2x2 block; a
    colour-3 tail is drawn straight below it (same two columns) whose length equals
    the number of distinct colours in the block.  The tail is grown with a 5-step
    "fuel" cellular automaton (fuel = max(seed, shift_down(fuel) - 1)).
  * task234 -- "slide block to anchor": two nonzero colours; the "mover" is a solid
    block with a 1-wide stem aimed at the solid "anchor" block.  A 2x2 morphological
    opening removes the stem; the resulting block is translated (data-dependent shift
    via MatMul permutation matrices, distance/direction from in-graph bounding boxes)
    until it touches the anchor, the stem vanishing.
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


def _shift(g, t, dy, dx):
    """Shift content by (dy,dx): out[r,c]=t[r-dy,c-dx], zero filled, kept 30x30."""
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    pad = g.nd("Pad", [t], mode="constant", value=0.0,
               pads=[0, 0, h0, w0, 0, 0, h1, w1])
    return g.nd("Slice", [pad, g.i64([h1, w1]), g.i64([h1 + G, w1 + G]), g.i64([2, 3])])


# ----------------------------------------------------------------------------- #
# task192                                                                        #
# ----------------------------------------------------------------------------- #
def _build_192():
    g = _G()
    ar = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    e_nz = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    BIG = g.f([1, 1, 1, 1], [1e6])

    cnt = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)        # [1,10,1,1]
    cnt_nz = g.nd("Mul", [cnt, e_nz])

    Cidx = g.nd("ArgMax", [cnt_nz], axis=1, keepdims=1)                # [1,1,1,1] i64
    Cidxf = g.nd("Cast", [Cidx], to=F)
    wC = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ar, Cidxf])]), half])], to=F)

    present = g.nd("Cast", [g.nd("Greater", [cnt_nz, half])], to=F)
    score = g.nd("Mul", [present, g.nd("Sub", [BIG, cnt])])
    Nidx = g.nd("ArgMax", [score], axis=1, keepdims=1)
    Nidxf = g.nd("Cast", [Nidx], to=F)
    wN = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ar, Nidxf])]), half])], to=F)

    Cmask = g.nd("ReduceSum", [g.nd("Mul", ["input", wC])], axes=[1], keepdims=1)  # [1,1,30,30]
    Nmask = g.nd("ReduceSum", [g.nd("Mul", ["input", wN])], axes=[1], keepdims=1)

    # neighbour lookups of Cmask
    Lcut = g.nd("Slice", [Cmask, g.i64([0]), g.i64([G - 1]), g.i64([3])])
    L = g.nd("Pad", [Lcut], mode="constant", value=0.0, pads=[0, 0, 0, 1, 0, 0, 0, 0])
    Rcut = g.nd("Slice", [Cmask, g.i64([1]), g.i64([G]), g.i64([3])])
    R = g.nd("Pad", [Rcut], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 0, 1])
    Ucut = g.nd("Slice", [Cmask, g.i64([0]), g.i64([G - 1]), g.i64([2])])
    U = g.nd("Pad", [Ucut], mode="constant", value=0.0, pads=[0, 0, 1, 0, 0, 0, 0, 0])
    Dcut = g.nd("Slice", [Cmask, g.i64([1]), g.i64([G]), g.i64([2])])
    Dn = g.nd("Pad", [Dcut], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 1, 0])

    LR = g.nd("Max", [L, R])
    UD = g.nd("Max", [U, Dn])
    encl = g.nd("Mul", [LR, UD])
    enclN = g.nd("Mul", [Nmask, encl])
    M = g.nd("Max", [Cmask, enclN])                                    # [1,1,30,30]

    colored = g.nd("Mul", [M, wC])                                     # [1,10,30,30]
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    bgplane = g.nd("Mul", [realmask, g.nd("Sub", [one, M])])
    bg = g.nd("Mul", [bgplane, e0])
    g.nd("Add", [colored, bg], "output")
    return _model(g)


def _solve_192(a):
    """numpy mirror of the task192 graph; returns predicted grid or None."""
    cnt = np.array([(a == ch).sum() for ch in range(CHANNELS)], float)
    cnt_nz = cnt.copy(); cnt_nz[0] = 0.0
    if int((cnt_nz > 0).sum()) != 2:
        return None
    C = int(np.argmax(cnt_nz))
    score = np.where(cnt_nz > 0, 1e6 - cnt_nz, 0.0); score[0] = 0.0
    N = int(np.argmax(score))
    Cm = (a == C).astype(np.float32); Nm = (a == N)

    def sh(m, dr, dc):
        return _npshift(m, -dr, -dc)   # neighbour at (r+dr,c+dc): out[r,c]=m[r+dr,c+dc]
    encl = ((sh(Cm, 0, -1) + sh(Cm, 0, 1)) > 0.5) & ((sh(Cm, -1, 0) + sh(Cm, 1, 0)) > 0.5)
    M = (Cm > 0.5) | (Nm & encl)
    return np.where(M, C, 0)


# ----------------------------------------------------------------------------- #
# task397 -- 3-coloured tails below 2x2 colour blocks                            #
# ----------------------------------------------------------------------------- #
# Every coloured object is a 2x2 block.  Directly below it (same two columns) a
# tail of colour 3 is drawn whose length == the number of distinct colours in the
# block (2, 3 or 4).  The tail is grown by a 5-step "fuel" cellular automaton.
def _build_397():
    g = _G()
    e_nz = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    e3 = g.f([1, CHANNELS, 1, 1], [1.0 if c == 3 else 0.0 for c in range(CHANNELS)])
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])

    # distinct-colour count of the 2x2 window anchored at each top-left
    appears = g.nd("Max", ["input", _shift(g, "input", 0, -1),
                           _shift(g, "input", -1, 0), _shift(g, "input", -1, -1)])
    k = g.nd("ReduceSum", [g.nd("Mul", [appears, e_nz])], axes=[1], keepdims=1)  # [1,1,30,30]

    colored = g.nd("ReduceSum", [g.nd("Mul", ["input", e_nz])], axes=[1], keepdims=1)
    top = _shift(g, colored, 1, 0)
    left = _shift(g, colored, 0, 1)
    TL = g.nd("Mul", [g.nd("Mul", [colored, g.nd("Sub", [one, top])]),
                      g.nd("Sub", [one, left])])
    kplane = g.nd("Mul", [TL, k])
    seed = g.nd("Add", [_shift(g, kplane, 2, 0), _shift(g, kplane, 2, 1)])

    fuel = seed
    for _ in range(5):
        fuel = g.nd("Max", [seed, g.nd("Sub", [_shift(g, fuel, 1, 0), one])])
    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    tail = g.nd("Mul", [g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [fuel, half])], to=F),
                                     realmask]),
                        g.nd("Sub", [one, colored])])                    # [1,1,30,30]

    out = g.nd("Add", [g.nd("Mul", ["input", g.nd("Sub", [one, tail])]),
                       g.nd("Mul", [tail, e3])])
    g.nd("Identity", [out], "output")
    return _model(g)


def _onehot(a):
    H, W = a.shape
    X = np.zeros((CHANNELS, G, G), np.float32)
    for c in range(CHANNELS):
        X[c, :H, :W] = (a == c)
    return X


def _npshift(t, dy, dx):
    out = np.zeros_like(t)
    H, W = t.shape[-2], t.shape[-1]
    r0, r1 = max(dy, 0), min(H, H + dy)
    c0, c1 = max(dx, 0), min(W, W + dx)
    out[..., r0:r1, c0:c1] = t[..., r0 - dy:r1 - dy, c0 - dx:c1 - dx]
    return out


def _solve_397(a):
    X = _onehot(a)
    realmask = X.sum(0)
    colored = X[1:].sum(0)
    k = np.zeros((G, G), np.float32)
    for c in range(1, CHANNELS):
        ch = X[c]
        k = k + np.maximum.reduce([ch, _npshift(ch, 0, -1),
                                   _npshift(ch, -1, 0), _npshift(ch, -1, -1)])
    top = _npshift(colored, 1, 0)
    left = _npshift(colored, 0, 1)
    TL = colored * (1 - (top > 0.5)) * (1 - (left > 0.5))
    kplane = TL * k
    seed = _npshift(kplane, 2, 0) + _npshift(kplane, 2, 1)
    fuel = seed.copy()
    for _ in range(5):
        fuel = np.maximum(seed, _npshift(fuel, 1, 0) - 1)
    tail = (fuel > 0.5).astype(np.float32) * realmask * (1 - colored)
    H, W = a.shape
    out = a.copy()
    out = np.where(tail[:H, :W] > 0.5, 3, out)
    return out


# ----------------------------------------------------------------------------- #
# task234 -- slide a stemmed block until it touches its anchor block             #
# ----------------------------------------------------------------------------- #
# Exactly two nonzero colours.  One ("mover") is a solid block with a 1-wide stem
# pointing at the other solid "anchor" block.  A 2x2 morphological opening drops
# the stem, leaving the block; the block is translated (data-dependent shift via
# MatMul permutation matrices) until adjacent to the anchor, the stem vanishing.
def _build_234():
    g = _G()
    e_nz = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    BIG = g.f([1, 1, 1, 1], [1e6])
    half2 = g.f([1, 1], [0.5])
    p1 = g.f([1, 1, 1, 1], [1.0])

    X = "input"
    # 2x2 opening per channel
    erode = g.nd("Mul", [g.nd("Mul", [X, _shift(g, X, 0, -1)]),
                         g.nd("Mul", [_shift(g, X, -1, 0), _shift(g, X, -1, -1)])])
    opened = g.nd("Max", [erode, g.nd("Max", [_shift(g, erode, 0, 1),
                          g.nd("Max", [_shift(g, erode, 1, 0), _shift(g, erode, 1, 1)])])])

    cnt_m = g.nd("ReduceSum", [X], axes=[2, 3], keepdims=1)            # [1,10,1,1]
    cnt_o = g.nd("ReduceSum", [opened], axes=[2, 3], keepdims=1)
    diff = g.nd("Mul", [g.nd("Sub", [cnt_m, cnt_o]), e_nz])
    Midx = g.nd("ArgMax", [diff], axis=1, keepdims=1)
    Midxf = g.nd("Cast", [Midx], to=F)
    ar = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    wM = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [ar, Midxf])]), half])], to=F)
    present = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [cnt_m, half])], to=F), e_nz])
    wA = g.nd("Mul", [present, g.nd("Sub", [one, wM])])

    block = g.nd("ReduceSum", [g.nd("Mul", [opened, wM])], axes=[1], keepdims=1)  # [1,1,30,30]
    anchor = g.nd("ReduceSum", [g.nd("Mul", [X, wA])], axes=[1], keepdims=1)      # [1,1,30,30]

    ridx_r = g.f([1, 1, G, 1], list(range(G)))      # row index along axis2
    cidx_c = g.f([1, 1, 1, G], list(range(G)))      # col index along axis3

    def bbox(plane):
        hr = g.nd("ReduceMax", [plane], axes=[3], keepdims=1)        # [1,1,30,1]
        hc = g.nd("ReduceMax", [plane], axes=[2], keepdims=1)        # [1,1,1,30]
        nhr = g.nd("Mul", [g.nd("Sub", [one, hr]), BIG])
        nhc = g.nd("Mul", [g.nd("Sub", [one, hc]), BIG])
        r0 = g.nd("ReduceMin", [g.nd("Add", [ridx_r, nhr])], axes=[2], keepdims=1)
        r1 = g.nd("ReduceMax", [g.nd("Sub", [ridx_r, nhr])], axes=[2], keepdims=1)
        c0 = g.nd("ReduceMin", [g.nd("Add", [cidx_c, nhc])], axes=[3], keepdims=1)
        c1 = g.nd("ReduceMax", [g.nd("Sub", [cidx_c, nhc])], axes=[3], keepdims=1)
        return r0, r1, c0, c1                                        # each [1,1,1,1]
    br0, br1, bc0, bc1 = bbox(block)
    ar0, ar1, ac0, ac1 = bbox(anchor)

    above = g.nd("Cast", [g.nd("Less", [ar1, br0])], to=F)
    below = g.nd("Cast", [g.nd("Less", [br1, ar0])], to=F)
    left = g.nd("Cast", [g.nd("Less", [ac1, bc0])], to=F)
    right = g.nd("Cast", [g.nd("Less", [bc1, ac0])], to=F)
    dy = g.nd("Add", [g.nd("Mul", [above, g.nd("Add", [g.nd("Sub", [ar1, br0]), p1])]),
                      g.nd("Mul", [below, g.nd("Sub", [g.nd("Sub", [ar0, br1]), p1])])])
    dx = g.nd("Add", [g.nd("Mul", [left, g.nd("Add", [g.nd("Sub", [ac1, bc0]), p1])]),
                      g.nd("Mul", [right, g.nd("Sub", [g.nd("Sub", [ac0, bc1]), p1])])])
    dy2 = g.nd("Squeeze", [dy], axes=[0, 1])        # [1,1]
    dx2 = g.nd("Squeeze", [dx], axes=[0, 1])        # [1,1]

    rIr = g.f([G, 1], list(range(G)))
    rIc = g.f([1, G], list(range(G)))
    trow = g.nd("Sub", [rIr, rIc])                  # [30,30] element[r,r']=r-r'
    Srow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [trow, dy2])]), half2])], to=F)
    tcol = g.nd("Sub", [rIc, rIr])                  # [30,30] element[c',c]=c-c'
    Scol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [tcol, dx2])]), half2])], to=F)

    nb = g.nd("MatMul", [g.nd("MatMul", [Srow, block]), Scol])       # [1,1,30,30]

    realmask = g.nd("ReduceSum", [X], axes=[1], keepdims=1)
    occ = g.nd("Add", [anchor, nb])
    bg = g.nd("Mul", [g.nd("Mul", [realmask, g.nd("Sub", [one, occ])]), e0])
    colored = g.nd("Add", [g.nd("Mul", [anchor, wA]), g.nd("Mul", [nb, wM])])
    g.nd("Add", [colored, bg], "output")
    return _model(g)


def _solve_234(a):
    X = _onehot(a)
    erode = X * _npshift(X, 0, -1) * _npshift(X, -1, 0) * _npshift(X, -1, -1)
    opened = np.maximum.reduce([erode, _npshift(erode, 0, 1),
                                _npshift(erode, 1, 0), _npshift(erode, 1, 1)])
    cnt_m = X.sum((1, 2)); cnt_o = opened.sum((1, 2))
    diff = (cnt_m - cnt_o).copy(); diff[0] = 0.0
    if diff.max() <= 0.5:
        return None
    M = int(np.argmax(diff))
    present = (cnt_m > 0.5).astype(float); present[0] = 0.0
    wA = present.copy(); wA[M] = 0.0
    if int(wA.sum()) != 1:
        return None
    A = int(np.argmax(wA))
    block = opened[M]; anchor = X[A]
    BIG = 1e6

    def bbox(m):
        hr = m.max(1); hc = m.max(0)
        idx = np.arange(G)
        return ((idx + (1 - hr) * BIG).min(), (idx - (1 - hr) * BIG).max(),
                (idx + (1 - hc) * BIG).min(), (idx - (1 - hc) * BIG).max())
    br0, br1, bc0, bc1 = bbox(block)
    ar0, ar1, ac0, ac1 = bbox(anchor)
    above = float(ar1 < br0); below = float(br1 < ar0)
    left = float(ac1 < bc0); right = float(bc1 < ac0)
    dy = int(round(above * (ar1 - br0 + 1) + below * (ar0 - br1 - 1)))
    dx = int(round(left * (ac1 - bc0 + 1) + right * (ac0 - bc1 - 1)))
    nb = _npshift(block, dy, dx)
    out = np.zeros((G, G), int)
    out[anchor > 0.5] = A; out[nb > 0.5] = M
    H, W = a.shape
    return out[:H, :W]


# ----------------------------------------------------------------------------- #
# entry point                                                                    #
# ----------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > G or max(b.shape) > G:
                continue
            out.append((a, b))
    return out


def candidates(ex):
    out = []
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if det and allp:
        # task192: denoise-to-rectangles
        def ok192(plist):
            for a, b in plist:
                if a.shape != b.shape:
                    return False
                p = _solve_192(a)
                if p is None or not np.array_equal(p, b):
                    return False
            return True
        if ok192(det) and ok192(allp):
            try:
                m = _build_192()
                onnx.checker.check_model(m, full_check=True)
                out.append(("crk4_192_denoiserect", m))
            except Exception:
                pass

        # task397: 3-tails below 2x2 blocks
        def ok397(plist):
            for a, b in plist:
                if a.shape != b.shape:
                    return False
                if not np.array_equal(_solve_397(a), b):
                    return False
            return True
        if not any(np.array_equal(a, b) for a, b in det) and ok397(det) and ok397(allp):
            try:
                m = _build_397()
                onnx.checker.check_model(m, full_check=True)
                out.append(("crk4_397_tails", m))
            except Exception:
                pass

        # task234: slide stemmed block to anchor
        def ok234(plist):
            for a, b in plist:
                if a.shape != b.shape:
                    return False
                p = _solve_234(a)
                if p is None or not np.array_equal(p, b):
                    return False
            return True
        if not any(np.array_equal(a, b) for a, b in det) and ok234(det) and ok234(allp):
            try:
                m = _build_234()
                onnx.checker.check_model(m, full_check=True)
                out.append(("crk4_234_slideblock", m))
            except Exception:
                pass
    return out
