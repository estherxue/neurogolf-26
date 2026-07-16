"""family_golf_2 -- cheaper exact solvers for a slice of golf targets.

Each candidate re-derives the task's rule from train+test+arc-gen pairs, verifies
EXACT equality with a numpy reference on every available pair, and only then emits
a minimal opset-10 ONNX graph.  The integrator auto-picks the cheapest correct
solver, so we just need these to be exact and cheaper than the incumbent.

Targets golfed here (all rules verified exact on 100% of provided examples):
  * 373 rowswapck  -> column-parity row swap on a height-2 grid (Where + const mask)
  * 45  connect_same_RL -> fill a row when its left/right edge share a colour
  * 113 sym_ud     -> up/down symmetric overlay within a constant-height window
  * 299 cross      -> extend two perpendicular dominoes into a full cross (mark 4)

Cost levers used: single-channel [1,1,*,*] reductions, [1,9,*,*] colour-only
windows (skip the background channel), broadcasting instead of Tile, and never
materialising a full [1,10,30,30] intermediate.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
_NEG = -(1 << 31)


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
                                         [float(v) for v in vals]))
        return nm

    def bconst(self, vals, shape):
        nm = self.name("b")
        self.inits.append(oh.make_tensor(nm, BOOL, list(shape),
                                         [bool(v) for v in vals]))
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
# 373  rowswapck:  H==2,  out[r,c] = in[r,c] if c even else in[1-r,c]
# ===========================================================================
def _ref_rowswap(a):
    H, W = a.shape
    o = a.copy()
    for r in range(H):
        for c in range(W):
            o[r, c] = a[r, c] if c % 2 == 0 else a[1 - r, c]
    return o


def _build_rowswap(W):
    g = _G()
    win = _slice(g, "input", [0, 0], [2, W], [2, 3])     # [1,10,2,W]
    rev = _slice(g, win, [1], [_NEG], [2], [-1])         # rows [1,0]
    cond = g.bconst([c % 2 == 0 for c in range(W)], [1, 1, 1, W])
    merged = g.node("Where", [cond, win, rev])           # [1,10,2,W]
    g.node("Pad", [merged], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - 2, WIDTH - W])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 45  connect_same_RL:  W const; fill row r with c if in[r,0]==in[r,W-1]==c!=0
# ===========================================================================
def _ref_connectRL(a, W):
    o = a.copy()
    H = a.shape[0]
    for r in range(H):
        if a[r, 0] != 0 and a[r, 0] == a[r, W - 1]:
            o[r, :] = a[r, 0]
    return o


def _build_connectRL(H, W):
    g = _G()
    L = _slice(g, "input", [0, 0], [H, 1], [2, 3])           # [1,10,H,1]
    R = _slice(g, "input", [0, W - 1], [H, W], [2, 3])       # [1,10,H,1]
    M = g.node("Mul", [L, R])                                # one-hot matched colour
    Mc = _slice(g, M, [1], [CHANNELS], [1])                 # drop bg channel
    rowhas = g.node("ReduceSum", [Mc], axes=[1], keepdims=1)  # [1,1,H,1]
    cond = g.node("Cast", [rowhas], to=BOOL)
    content = _slice(g, "input", [0, 0], [H, W], [2, 3])     # [1,10,H,W]
    filled = g.node("Where", [cond, M, content])            # broadcast [1,10,H,W]
    g.node("Pad", [filled], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 113  sym_ud:  H const; output = one-hot OR of input and its row-flip (flipud)
# ===========================================================================
def _ref_overlay_ud(a):
    f = a[::-1, :]
    H, W = a.shape
    o = np.zeros_like(a)
    ok = True
    for r in range(H):
        for c in range(W):
            vals = {x for x in (a[r, c], f[r, c]) if x != 0}
            if len(vals) > 1:
                ok = False
            o[r, c] = vals.pop() if vals else 0
    return o, ok


def _build_overlay_ud(H):
    g = _G()
    in0 = _slice(g, "input", [0, 0], [1, H], [1, 2])     # bg chan, rows [1,1,H,30]
    rev0 = _slice(g, in0, [H - 1], [_NEG], [2], [-1])    # flipud bg
    bg = g.node("Mul", [in0, rev0])                      # bg only where both bg
    inC = _slice(g, "input", [1, 0], [CHANNELS, H], [1, 2])   # colours [1,9,H,30]
    revC = _slice(g, inC, [H - 1], [_NEG], [2], [-1])    # flipud colours
    colored = g.node("Max", [inC, revC])                 # [1,9,H,30]
    merged = g.node("Concat", [bg, colored], axis=1)     # [1,10,H,30]
    g.node("Pad", [merged], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, 0])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 299  cross:  const HxW; vertical domino -> fill column, horizontal -> fill row,
#              intersection -> colour 4.
# ===========================================================================
def _ref_cross(a):
    H, W = a.shape
    o = np.zeros((H, W), int)
    vcol = vcc = hrow = hrc = None
    for k in range(1, 10):
        cells = list(zip(*np.where(a == k)))
        if len(cells) < 2:
            continue
        rs = {r for r, _ in cells}
        cs = {c for _, c in cells}
        if len(cs) == 1:
            vcol, vcc = next(iter(cs)), k
        elif len(rs) == 1:
            hrow, hrc = next(iter(rs)), k
    if vcol is not None:
        o[:, vcol] = vcc
    if hrow is not None:
        o[hrow, :] = hrc
    if vcol is not None and hrow is not None:
        o[hrow, vcol] = 4
    return o


def _build_cross(H, W):
    g = _G()
    win = _slice(g, "input", [1, 0, 0], [CHANNELS, H, W], [1, 2, 3])  # [1,9,H,W]
    half = g.fconst([1.5], [1])
    # vertical: colour with >=2 cells stacked in a column
    rowcount = g.node("ReduceSum", [win], axes=[2], keepdims=1)       # [1,9,1,W]
    vb = g.node("Greater", [rowcount, half])
    vf = g.node("Cast", [vb], to=DATA_TYPE)                           # [1,9,1,W]
    # horizontal: colour with >=2 cells in a row
    colcount = g.node("ReduceSum", [win], axes=[3], keepdims=1)       # [1,9,H,1]
    hb = g.node("Greater", [colcount, half])
    hf = g.node("Cast", [hb], to=DATA_TYPE)                           # [1,9,H,1]
    merged = g.node("Max", [vf, hf])                                  # [1,9,H,W]
    # intersection mask (both a vertical column and a horizontal row present)
    vany = g.node("ReduceSum", [vf], axes=[1], keepdims=1)            # [1,1,1,W]
    hany = g.node("ReduceSum", [hf], axes=[1], keepdims=1)            # [1,1,H,1]
    im = g.node("Mul", [vany, hany])                                  # [1,1,H,W]
    imb = g.node("Cast", [im], to=BOOL)
    e4 = g.fconst([1.0 if i == 3 else 0.0 for i in range(CHANNELS - 1)],
                  [1, CHANNELS - 1, 1, 1])                            # colour 4
    four = g.node("Mul", [im, e4])                                    # [1,9,H,W]
    crossed = g.node("Where", [imb, four, merged])                    # [1,9,H,W]
    s = g.node("ReduceSum", [crossed], axes=[1], keepdims=1)          # [1,1,H,W]
    one = g.fconst([1.0], [1])
    bg = g.node("Sub", [one, s])                                      # [1,1,H,W]
    full = g.node("Concat", [bg, crossed], axis=1)                    # [1,10,H,W]
    if H == HEIGHT and W == WIDTH:
        g.node("Identity", [full], "output")
    else:
        g.node("Pad", [full], "output", mode="constant", value=0.0,
               pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])
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

    shapes = {a.shape for a, _ in prs}
    Hs = {a.shape[0] for a, _ in prs}
    Ws = {a.shape[1] for a, _ in prs}
    same_size = all(a.shape == b.shape for a, b in prs)
    changes = any(not np.array_equal(a, b) for a, b in prs)

    # ---- 373 rowswapck -------------------------------------------------------
    if changes and same_size and Hs == {2} and len(Ws) == 1:
        W = next(iter(Ws))
        if 1 < W <= WIDTH and all(np.array_equal(_ref_rowswap(a), b)
                                  for a, b in prs):
            emit("rowswapck", lambda: _build_rowswap(W))

    # ---- 45 connect_same_RL --------------------------------------------------
    if changes and same_size and len(shapes) == 1:
        H, W = next(iter(shapes))
        if 1 < W <= WIDTH and all(np.array_equal(_ref_connectRL(a, W), b)
                                  for a, b in prs):
            emit("connect_RL", lambda: _build_connectRL(H, W))

    # ---- 113 sym_ud ----------------------------------------------------------
    if changes and same_size and len(Hs) == 1:
        H = next(iter(Hs))
        if 1 < H <= HEIGHT:
            ok = True
            for a, b in prs:
                o, valid = _ref_overlay_ud(a)
                if not valid or not np.array_equal(o, b):
                    ok = False
                    break
            if ok:
                emit("sym_ud", lambda: _build_overlay_ud(H))

    # ---- 299 cross -----------------------------------------------------------
    if changes and same_size and len(shapes) == 1:
        H, W = next(iter(shapes))
        if all(np.array_equal(_ref_cross(a), b) for a, b in prs):
            emit("cross", lambda: _build_cross(H, W))

    return out
