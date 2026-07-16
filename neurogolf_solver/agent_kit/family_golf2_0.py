"""family_golf2_0 -- cheaper exact solvers for a slice of golf targets.

Each candidate re-derives the rule from train+test+arc-gen pairs, verifies EXACT
equality against a numpy reference on every available pair, and only then emits a
minimal opset-10 ONNX graph.  The integrator auto-picks the cheapest correct
solver, so we just need exact + cheaper than the incumbent.

Targets golfed here (all verified exact on 100% of provided examples):
  * 28   dotframe   -> two markers (rows 2 & 7) recolor a fixed 10x10 frame template
  * 399  countdice  -> count 2x2 blocks of colour 2, render as a 3x3 "dice" of 1s
  * 222  keeprect   -> keep the solid monochrome rectangle (>=2 stacked 2x2 blocks)

Cost levers: tiny [1,9,10,10] / [1,1,3,3] intermediates, grouped Conv/ConvTranspose
for 2x2 block detect+dilate, Pad to write the final output (free 'output' tensor),
broadcasting instead of Tile, never materialising spare [1,10,30,30] tensors.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def fconst(self, vals, shape):
        nm = self.name("f")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(shape),
                                         [float(v) for v in np.ravel(vals)]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.iconst(starts), g.iconst(ends), g.iconst(axes)]
    if steps is not None:
        ins.append(g.iconst(steps))
    return g.node("Slice", ins)


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
# 28  dotframe: top marker (row 2) colours the T-cells, bottom marker (row 7)
#               colours the B-cells of a fixed 10x10 frame template.
# ===========================================================================
_T28 = np.zeros((10, 10), int)
_T28[0, :] = 1
_T28[1, [0, 9]] = 1
_T28[2, :] = 1
_T28[3, [0, 9]] = 1
_T28[4, [0, 9]] = 1
_B28 = np.zeros((10, 10), int)
_B28[5, [0, 9]] = 1
_B28[6, [0, 9]] = 1
_B28[7, :] = 1
_B28[8, [0, 9]] = 1
_B28[9, :] = 1


def _ref_28(a):
    if a.shape != (10, 10):
        return None
    nz = [(r, c, int(a[r, c])) for r in range(10) for c in range(10) if a[r, c]]
    if len(nz) != 2:
        return None
    nz.sort()
    (r1, _, col1), (r2, _, col2) = nz
    if r1 != 2 or r2 != 7:
        return None
    o = np.zeros((10, 10), int)
    o[_T28 == 1] = col1
    o[_B28 == 1] = col2
    return o


def _build_28():
    g = _G()
    G = _slice(g, "input", [1, 0, 0], [10, 10, 10], [1, 2, 3])      # [1,9,10,10]
    topG = _slice(g, G, [0], [5], [2])                              # [1,9,5,10]
    botG = _slice(g, G, [5], [10], [2])                             # [1,9,5,10]
    topP = g.node("ReduceMax", [topG], axes=[2, 3], keepdims=1)     # [1,9,1,1]
    botP = g.node("ReduceMax", [botG], axes=[2, 3], keepdims=1)     # [1,9,1,1]
    Tm = g.fconst(_T28, [1, 1, 10, 10])
    Bm = g.fconst(_B28, [1, 1, 10, 10])
    cT = g.node("Mul", [Tm, topP])                                  # [1,9,10,10]
    cB = g.node("Mul", [Bm, botP])                                  # [1,9,10,10]
    colored = g.node("Add", [cT, cB])                               # [1,9,10,10]
    summ = g.node("ReduceSum", [colored], axes=[1], keepdims=1)     # [1,1,10,10]
    one = g.fconst([1.0], [1])
    ch0 = g.node("Sub", [one, summ])                                # [1,1,10,10]
    full = g.node("Concat", [ch0, colored], axis=1)                 # [1,10,10,10]
    g.node("Pad", [full], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - 10, WIDTH - 10])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 399  countdice: n = (#colour-2 cells)/4 (each block is a 2x2 square); render
#                 the count as 1s on a fixed 3x3 "dice" order.
# ===========================================================================
_DICE = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]


def _ref_399(a):
    if (a != 0).sum() != (a == 2).sum():
        return None
    cells = int((a == 2).sum())
    if cells % 4 != 0:
        return None
    n = cells // 4
    if n > 5:
        return None
    o = np.zeros((3, 3), int)
    for i in range(n):
        o[_DICE[i]] = 1
    return o


def _build_399():
    g = _G()
    ch2 = _slice(g, "input", [2], [3], [1])                         # [1,1,30,30]
    s = g.node("ReduceSum", [ch2], axes=[2, 3], keepdims=1)         # [1,1,1,1]
    contribs = []
    for p, (i, j) in enumerate(_DICE):
        thr = g.fconst([4.0 * (p + 1) - 0.5], [1])
        gp = g.node("Greater", [s, thr])
        gpf = g.node("Cast", [gp], to=DATA_TYPE)                    # [1,1,1,1]
        pm = np.zeros((3, 3), float)
        pm[i, j] = 1.0
        pmc = g.fconst(pm, [1, 1, 3, 3])
        contribs.append(g.node("Mul", [gpf, pmc]))                  # [1,1,3,3]
    ch1 = contribs[0]
    for c in contribs[1:]:
        ch1 = g.node("Add", [ch1, c])                              # [1,1,3,3]
    one = g.fconst([1.0], [1])
    ch0 = g.node("Sub", [one, ch1])                                # [1,1,3,3]
    ch01 = g.node("Concat", [ch0, ch1], axis=1)                    # [1,2,3,3]
    g.node("Pad", [ch01], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, CHANNELS - 2, HEIGHT - 3, WIDTH - 3])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 222  keeprect: keep cells of the unique solid monochrome rectangle. A colour
#                qualifies iff it has >=2 stacked 2x2 monochrome blocks (noise
#                produces at most one by chance).
# ===========================================================================
def _ref_222(a):
    H, W = a.shape
    o = np.zeros_like(a)
    for c in range(1, 10):
        m = (a == c).astype(int)
        if m.sum() < 4:
            continue
        B = (m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:])
        if B.sum() < 2:
            continue
        full = np.zeros((H, W), bool)
        for di in (0, 1):
            for dj in (0, 1):
                full[di:di + B.shape[0], dj:dj + B.shape[1]] |= B.astype(bool)
        o[full] = c
    return o


def _build_222():
    g = _G()
    mc = _slice(g, "input", [1, 0, 0], [10, 30, 30], [1, 2, 3])     # [1,9,30,30]
    W2 = oh.make_tensor(g.name("w"), DATA_TYPE, [9, 1, 2, 2], [1.0] * 36)
    g.inits.append(W2)
    conv = g.node("Conv", [mc, W2.name], group=9, kernel_shape=[2, 2],
                  pads=[0, 0, 0, 0])                                # [1,9,29,29]
    thr = g.fconst([3.5], [1])
    B = g.node("Cast", [g.node("Greater", [conv, thr])], to=DATA_TYPE)  # [1,9,29,29]
    cnt = g.node("ReduceSum", [B], axes=[2, 3], keepdims=1)         # [1,9,1,1]
    g2 = g.fconst([1.5], [1])
    gate = g.node("Cast", [g.node("Greater", [cnt, g2])], to=DATA_TYPE)  # [1,9,1,1]
    Wt = oh.make_tensor(g.name("wt"), DATA_TYPE, [9, 1, 2, 2], [1.0] * 36)
    g.inits.append(Wt)
    dil = g.node("ConvTranspose", [B, Wt.name], group=9,
                 kernel_shape=[2, 2], pads=[0, 0, 0, 0])            # [1,9,30,30]
    half = g.fconst([0.5], [1])
    full = g.node("Cast", [g.node("Greater", [dil, half])], to=DATA_TYPE)
    gated = g.node("Mul", [full, gate])                            # [1,9,30,30]
    inmask = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    sumg = g.node("ReduceSum", [gated], axes=[1], keepdims=1)       # [1,1,30,30]
    ch0 = g.node("Sub", [inmask, sumg])                            # [1,1,30,30]
    g.node("Concat", [ch0, gated], "output", axis=1)               # [1,10,30,30]
    return _model(g.nodes, g.inits)


# ===========================================================================
# 279  loopfill: colour-1 components that enclose background (colour 9) become
#                colour 8.  Two floods: (a) border-flood the bg to find enclosed
#                pockets, (b) propagate the enclosed-adjacency across each loop.
# ===========================================================================
_BD_279 = 22   # border-flood steps (observed max depth 13; margin for unseen)
_CD_279 = 12   # component-flood steps (observed max depth 3)


def _dil(x):
    n = x.copy()
    n[1:, :] |= x[:-1, :]
    n[:-1, :] |= x[1:, :]
    n[:, 1:] |= x[:, :-1]
    n[:, :-1] |= x[:, 1:]
    return n


def _ref_279(a):
    if set(np.unique(a).tolist()) - {1, 9}:
        return None
    free = (a == 9)
    H, W = a.shape
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    reach = free & border
    for _ in range(_BD_279):
        reach = reach | (_dil(reach) & free)
    encl = free & ~reach
    region = (a == 1)
    loop = region & _dil(encl)
    for _ in range(_CD_279):
        loop = loop | (_dil(loop) & region)
    o = a.copy()
    o[loop] = 8
    return o


# row-0 / col-0 seed mask (top & left grid edges; bottom & right come from the
# padding region via the dilation of `outside`).
_RC0 = np.zeros((HEIGHT, WIDTH), float)
_RC0[0, :] = 1.0
_RC0[:, 0] = 1.0

# 4-connected plus kernel (centre + N/E/S/W) used for every dilation/flood step.
_PLUS = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]


def _build_279():
    g = _G()
    plus = oh.make_tensor(g.name("w"), DATA_TYPE, [1, 1, 3, 3], _PLUS)
    g.inits.append(plus)

    def conv(x):
        return g.node("Conv", [x, plus.name], kernel_shape=[3, 3],
                      pads=[1, 1, 1, 1])

    free = _slice(g, "input", [9], [10], [1])                      # [1,1,30,30]
    inmask = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    one = g.fconst([1.0], [1])
    outside = g.node("Sub", [one, inmask])                          # [1,1,30,30]
    rc0 = g.fconst(_RC0, [1, 1, HEIGHT, WIDTH])
    seedraw = g.node("Add", [rc0, conv(outside)])                  # >0 on grid edges
    reach = g.node("Mul", [free, seedraw])                         # seed (float)
    for _ in range(_BD_279):
        reach = g.node("Mul", [free, conv(reach)])
    rsupp = g.node("Cast", [g.node("Greater", [reach, g.fconst([0.5], [1])])],
                   to=DATA_TYPE)                                    # [1,1,30,30]
    encl = g.node("Mul", [free, g.node("Sub", [one, rsupp])])      # enclosed bg
    region = _slice(g, "input", [1], [2], [1])                     # [1,1,30,30]
    loop = g.node("Mul", [region, conv(encl)])                     # seed
    for _ in range(_CD_279):
        loop = g.node("Mul", [region, conv(loop)])
    lsupp = g.node("Cast", [g.node("Greater", [loop, g.fconst([0.5], [1])])],
                   to=DATA_TYPE)                                    # [1,1,30,30]
    # delta: move loop cells from channel 1 (-1) to channel 8 (+1).
    dv = [0.0] * CHANNELS
    dv[1] = -1.0
    dv[8] = 1.0
    delta = g.fconst(dv, [1, CHANNELS, 1, 1])
    filled = g.node("Mul", [lsupp, delta])                        # [1,10,30,30]
    g.node("Add", ["input", filled], "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    def emit(name, fn):
        try:
            out.append((name, fn()))
        except Exception:
            pass

    def exact(ref):
        for a, b in prs:
            r = ref(a)
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                return False
        return True

    # ---- 28 dotframe --------------------------------------------------------
    if all(a.shape == (10, 10) and b.shape == (10, 10) for a, b in prs) and exact(_ref_28):
        emit("dotframe", _build_28)

    # ---- 399 countdice ------------------------------------------------------
    if all(b.shape == (3, 3) for a, b in prs) and exact(_ref_399):
        emit("countdice", _build_399)

    # ---- 222 keeprect -------------------------------------------------------
    if all(a.shape == b.shape for a, b in prs) and exact(_ref_222):
        emit("keeprect", _build_222)

    # ---- 279 loopfill -------------------------------------------------------
    if all(a.shape == b.shape for a, b in prs) and \
       all(not (set(np.unique(a).tolist()) - {1, 9}) for a, b in prs) and \
       exact(_ref_279):
        emit("loopfill", _build_279)

    return out
