"""family_sgolf3_2 -- cheap EXACT solvers for a few origin-anchored propagation
rules, expressed as short clip-free Conv chains over single-channel intermediates.

Each rule below has a numpy reference that mirrors the ONNX arithmetic bit-for-bit;
we only emit a model when that reference reproduces every train+test+arc-gen pair
exactly (the grader's gate).  All propagation uses Hillis-Steele DOUBLING offsets
(1,2,4,8,16) so a straight, unobstructed line covers any distance <=31 inside a
30x30 grid.  Because the propagation shifts content only ALONG the line, off-line
cells stay identically 0, so we can drop the usual Clip after every step (the grader
only tests output>0) -- halving the intermediate count.

Implemented rules
-----------------
* xdiag  : a solid single-colour field with one background hole -> paint both full
           diagonals through the hole with the background colour (an 'X').  Value is
           colour-agnostic (the field colour may differ per example).
* altfill5: every coloured seed cell shoots a horizontal ray to the RIGHT grid edge,
           painting its own colour at even offsets and colour 5 at odd offsets
           (K,5,K,5,...).  Per-colour rightward doubling fill; 5 = the coloured cells
           shifted one to the right.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

_OFFS = [1, 2, 4, 8, 16]  # doubling: covers distance 31 >= 29 (max inside 30)
_BIG = 1.0e6


# --------------------------------------------------------------------------- #
# graph helper                                                                #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def _n(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init_f(self, dims, vals):
        nm = self._n("w")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         np.asarray(vals, np.float32).ravel().tolist()))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self._n()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    def conv(self, src, W, B=None, out=None):
        """W: (O,I,KH,KW) ndarray.  SAME padding for odd kernels."""
        O, I, KH, KW = W.shape
        wt = self.init_f([O, I, KH, KW], W)
        ins = [src, wt]
        if B is not None:
            ins.append(self.init_f([O], B))
        return self.node("Conv", ins, out,
                         kernel_shape=[KH, KW], pads=[KH // 2, KW // 2, KH // 2, KW // 2])


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# one-hot / decode helpers                                                    #
# --------------------------------------------------------------------------- #
def _onehot(grid):
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = grid.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (grid == c)
    return t


def _decode(o, h, w):
    """o: (CHANNELS,30,30) -> (h,w) int grid, or None if not exact one-hot."""
    b = o > 0
    if b[:, h:, :].any() or b[:, :, w:].any():
        return None
    out = np.zeros((h, w), int)
    for i in range(h):
        for j in range(w):
            ch = np.where(b[:, i, j])[0]
            if len(ch) != 1:
                return None
            out[i, j] = ch[0]
    return out


# --------------------------------------------------------------------------- #
# xdiag                                                                        #
# --------------------------------------------------------------------------- #
def _diag_shift_kernel(d, anti):
    """(2d+1)x(2d+1) kernel: out[i,j] = in[i,j] + in[i-d,j-d] + in[i+d,j+d]
    (main diagonal) or the anti-diagonal variant."""
    K = 2 * d + 1
    w = np.zeros((1, 1, K, K), np.float32)
    w[0, 0, d, d] = 1.0                       # centre
    if not anti:
        w[0, 0, 0, 0] = 1.0                   # in[i-d,j-d]
        w[0, 0, 2 * d, 2 * d] = 1.0           # in[i+d,j+d]
    else:
        w[0, 0, 0, 2 * d] = 1.0               # in[i-d,j+d]
        w[0, 0, 2 * d, 0] = 1.0               # in[i+d,j-d]
    return w


def _diag_line_np(H):
    """numpy mirror of the unclipped doubling diagonal propagation (both diags)."""
    def prop(anti):
        cur = H.astype(np.float32).copy()
        for d in _OFFS:
            w = _diag_shift_kernel(d, anti)[0, 0]
            pad = np.zeros((HEIGHT + 2 * d, WIDTH + 2 * d), np.float32)
            pad[d:d + HEIGHT, d:d + WIDTH] = cur
            nxt = np.zeros((HEIGHT, WIDTH), np.float32)
            for u in range(2 * d + 1):
                for v in range(2 * d + 1):
                    if w[u, v]:
                        nxt += w[u, v] * pad[u:u + HEIGHT, v:v + WIDTH]
            cur = nxt
        return cur
    return prop(False) + prop(True)


def _xdiag_np(t):
    """t: (CHANNELS,30,30) one-hot -> predicted (CHANNELS,30,30) via the exact
    ONNX arithmetic (no clips)."""
    H = t[0].copy()                                   # hole indicator (bg in grid)
    G = t.sum(0)                                       # in-grid mask (0/1)
    P = _diag_line_np(H)                               # >0 on the X, incl padding leak
    out = t.astype(np.float32).copy()
    out[0] = t[0] + P + _BIG * G - _BIG                # channel 0 = background on X
    for c in range(1, CHANNELS):
        out[c] = t[c] - P
    return out


def _build_xdiag():
    g = _G()
    # hole H (channel 0)
    Wh = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Wh[0, 0, 0, 0] = 1.0
    H = g.conv("input", Wh)
    # grid G (sum of channels)
    Wg = np.ones((1, CHANNELS, 1, 1), np.float32)
    G = g.conv("input", Wg)
    # diagonal propagation (both diagonals), clip-free
    def prop(anti):
        cur = H
        for d in _OFFS:
            cur = g.conv(cur, _diag_shift_kernel(d, anti))
        return cur
    m1 = prop(False)
    m2 = prop(True)
    P = g.node("Add", [m1, m2])
    PG = g.node("Concat", [P, G], axis=1)              # [1,2,30,30]
    # V: channel0 = P + BIG*G - BIG ; channel c(1..9) = -P
    Wv = np.zeros((CHANNELS, 2, 1, 1), np.float32)
    Bv = np.zeros((CHANNELS,), np.float32)
    Wv[0, 0, 0, 0] = 1.0
    Wv[0, 1, 0, 0] = _BIG
    Bv[0] = -_BIG
    for c in range(1, CHANNELS):
        Wv[c, 0, 0, 0] = -1.0
    V = g.conv(PG, Wv, Bv)                             # [1,10,30,30]
    g.node("Add", ["input", V], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# altfill5                                                                     #
# --------------------------------------------------------------------------- #
_FILL_OFFS = [2, 4, 8, 16]  # even offsets 0..30 via doubling


def _rshift(a, d):
    """right shift along last axis by d, zero fill (content moves right)."""
    out = np.zeros_like(a)
    if d < a.shape[-1]:
        out[..., d:] = a[..., :a.shape[-1] - d]
    return out


def _altfill_np(t):
    """numpy mirror of the ONNX altfill5 arithmetic."""
    cur = t.astype(np.float32).copy()
    for d in _FILL_OFFS:
        cur = cur + _rshift(cur, d)
    Ec = cur                                   # >0 at even offsets, per channel
    Ecol = Ec[1:].sum(0)                        # any coloured even cell (exclude bg)
    five = _rshift(Ecol, 1)                     # colour 5 at odd offsets
    G = t.sum(0)
    out = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    for c in range(1, CHANNELS):
        out[c] = Ec[c] + _BIG * G - _BIG
    out[5] = five + _BIG * G - _BIG
    out[0] = G - Ecol - five
    return out


def _build_altfill():
    g = _G()
    # per-channel rightward doubling fill (grouped conv, group=CHANNELS)
    cur = "input"
    for d in _FILL_OFFS:
        W = np.zeros((CHANNELS, 1, 1, d + 1), np.float32)
        W[:, 0, 0, 0] = 1.0                    # in[i,j-d]
        W[:, 0, 0, d] = 1.0                    # in[i,j]
        wt = g.init_f([CHANNELS, 1, 1, d + 1], W)
        cur = g.node("Conv", [cur, wt], kernel_shape=[1, d + 1],
                     pads=[0, d, 0, 0], group=CHANNELS)
    Ec = cur
    # Ecol = sum of coloured channels (1..9)
    Wcol = np.ones((1, CHANNELS, 1, 1), np.float32)
    Wcol[0, 0, 0, 0] = 0.0
    Ecol = g.conv(Ec, Wcol)
    # five = right shift of Ecol by 1
    Wf = np.zeros((1, 1, 1, 2), np.float32)
    Wf[0, 0, 0, 0] = 1.0
    wtf = g.init_f([1, 1, 1, 2], Wf)
    five = g.node("Conv", [Ecol, wtf], kernel_shape=[1, 2], pads=[0, 1, 0, 0])
    # G = in-grid mask
    G = g.conv("input", np.ones((1, CHANNELS, 1, 1), np.float32))
    # assemble output
    cat = g.node("Concat", [Ec, five, Ecol, G], axis=1)   # 10+1+1+1 = 13 ch
    W = np.zeros((CHANNELS, CHANNELS + 3, 1, 1), np.float32)
    B = np.zeros((CHANNELS,), np.float32)
    iF, iEcol, iG = CHANNELS, CHANNELS + 1, CHANNELS + 2
    for c in range(1, CHANNELS):
        if c == 5:
            continue
        W[c, c, 0, 0] = 1.0
        W[c, iG, 0, 0] = _BIG
        B[c] = -_BIG
    W[5, iF, 0, 0] = 1.0
    W[5, iG, 0, 0] = _BIG
    B[5] = -_BIG
    W[0, iG, 0, 0] = 1.0
    W[0, iEcol, 0, 0] = -1.0
    W[0, iF, 0, 0] = -1.0
    g.conv(cat, W, B, out="output")
    return _model(g)


# --------------------------------------------------------------------------- #
# recolor5 (build389): two colours {5, X}; 5-cells -> X, everything else -> bg  #
# --------------------------------------------------------------------------- #
def _recolor5_np(t):
    five = t[5]
    G = t.sum(0)
    present = t.reshape(CHANNELS, -1).max(1)
    out = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    for c in range(1, CHANNELS):
        if c == 5:
            out[c] = five - _BIG
        else:
            out[c] = five + _BIG * present[c] - _BIG
    out[0] = G - five
    return out


def _build_recolor5():
    g = _G()
    # five (channel 5 mask), G (grid mask)
    Wf = np.zeros((1, CHANNELS, 1, 1), np.float32); Wf[0, 5, 0, 0] = 1.0
    five = g.conv("input", Wf)
    G = g.conv("input", np.ones((1, CHANNELS, 1, 1), np.float32))
    # presence per channel via GlobalMaxPool -> [1,10,1,1]
    pres = g.node("GlobalMaxPool", ["input"])
    # presT_c: c in 1..9,c!=5 -> BIG*pres-BIG ; c==5 -> -BIG ; c==0 -> 0
    Wp = np.zeros((CHANNELS, CHANNELS, 1, 1), np.float32)
    Bp = np.zeros((CHANNELS,), np.float32)
    for c in range(1, CHANNELS):
        if c == 5:
            Bp[c] = -_BIG
        else:
            Wp[c, c, 0, 0] = _BIG
            Bp[c] = -_BIG
    presT = g.conv(pres, Wp, Bp)                        # [1,10,1,1]
    S = g.node("Add", [five, presT])                    # broadcast -> [1,10,30,30]
    # channel-0 correction T_0 = G - 2*five
    cat = g.node("Concat", [G, five], axis=1)           # [1,2,30,30]
    Wt = np.zeros((CHANNELS, 2, 1, 1), np.float32)
    Wt[0, 0, 0, 0] = 1.0
    Wt[0, 1, 0, 0] = -2.0
    T = g.conv(cat, Wt)
    g.node("Add", [S, T], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
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


def _match(prs, fn):
    for a, b in prs:
        if a.shape != b.shape:
            return False
        pred = _decode(fn(_onehot(a)), *b.shape)
        if pred is None or not np.array_equal(pred, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    # xdiag: same-shape, and the reference reproduces every pair exactly.
    if all(a.shape == b.shape for a, b in prs) and any((a != b).any() for a, b in prs):
        try:
            if _match(prs, _xdiag_np):
                out.append(("xdiag", _build_xdiag()))
        except Exception:
            pass
        try:
            if _match(prs, _altfill_np):
                out.append(("altfill5", _build_altfill()))
        except Exception:
            pass
        try:
            if _match(prs, _recolor5_np):
                out.append(("recolor5", _build_recolor5()))
        except Exception:
            pass

    return out
