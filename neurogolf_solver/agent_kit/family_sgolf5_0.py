"""family_sgolf5_0 -- CHEAP, GRID-AGNOSTIC, NO-CROP line/connect solvers built on
SINGLE-CHANNEL Hillis-Steele doubling (offsets 1,2,4,8,16 cover a full 30-cell
grid in 5 steps).  Everything runs on the FULL 30x30 canvas end to end; nothing
is cropped, no dtype is changed, and every intermediate is a single-channel
[1,1,30,30] mask (10x cheaper than the [1,10,30,30] tensors the line baselines
carry).  The final answer is written into the FREE `output` tensor via Add.

Two origin-anchored rules, both exact under the top-left zero-pad contract:

  * SPAN  -- on each row/column that holds >=2 non-background cells, paint every
             background cell strictly BETWEEN the outermost non-background cells
             with a single fixed colour `f`.  (marker colours are preserved.)
             Implemented with prefix-sum doubling: a cell is "between" iff a
             non-bg cell exists strictly to its left AND right (H) / up AND down
             (V).  Only a mask flows -> one channel.

  * HCONNECT -- connect two SAME-colour cells that face each other along a row
             (only background between them) by a straight segment of that same
             colour.  Implemented on a SCALAR colour field with a "carry the
             nearest non-bg colour" doubling scan (Where + Greater), so the
             per-colour rule needs a single channel, not ten.

Detection fits the rule from train+test+arc-gen with an exact numpy mirror and
only emits when every pair reproduces exactly; the harness is the final judge.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

NSTEPS = 5
_OFFS = [1 << k for k in range(NSTEPS)]  # 1,2,4,8,16


# --------------------------------------------------------------------------- #
# numpy reference mirrors (used only for detection)                           #
# --------------------------------------------------------------------------- #
def _span_np(a, bg, f, do_h, do_v):
    H, W = a.shape
    nb = (a != bg)
    out = a.copy()
    fill = np.zeros((H, W), bool)
    if do_h:
        for r in range(H):
            cols = np.where(nb[r])[0]
            if len(cols) >= 2:
                for c in range(cols.min() + 1, cols.max()):
                    if a[r, c] == bg:
                        fill[r, c] = True
    if do_v:
        for c in range(W):
            rows = np.where(nb[:, c])[0]
            if len(rows) >= 2:
                for r in range(rows.min() + 1, rows.max()):
                    if a[r, c] == bg:
                        fill[r, c] = True
    out[fill] = f
    return out


def _hconnect_np(a, bg):
    """Connect two equal-colour cells on a row (only bg between) with that colour."""
    H, W = a.shape
    out = a.copy()
    for r in range(H):
        cs = np.where(a[r] != bg)[0]
        for i in range(len(cs) - 1):
            c0, c1 = cs[i], cs[i + 1]
            if a[r, c0] == a[r, c1] and c1 > c0 + 1:
                out[r, c0 + 1:c1] = a[r, c0]
    return out


# --------------------------------------------------------------------------- #
# ONNX graph builder                                                          #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init_f(self, dims, vals):
        n = self.nm("w")
        self.inits.append(oh.make_tensor(
            n, DATA_TYPE, list(dims), np.asarray(vals, np.float32).ravel().tolist()))
        return n

    def node(self, op, ins, out=None, **attrs):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    # 1x1 channel remap: input[1,Cin,30,30] -> [1,Cout,30,30]
    def conv1x1(self, src, W, out=None):
        Cout, Cin = W.shape[0], W.shape[1]
        wt = self.init_f([Cout, Cin, 1, 1], W)
        return self.node("Conv", [src, wt], out, kernel_shape=[1, 1], pads=[0, 0, 0, 0])

    # single-channel weighted neighbourhood op; taps: dict (kr,kc)->weight
    def stencil(self, src, kh, kw, pads, taps, out=None):
        W = np.zeros((1, 1, kh, kw), np.float32)
        for (kr, kc), v in taps.items():
            W[0, 0, kr, kc] = v
        wt = self.init_f([1, 1, kh, kw], W)
        return self.node("Conv", [src, wt], out, kernel_shape=[kh, kw], pads=pads)

    def clip01(self, src, out=None):
        return self.node("Clip", [src], out, min=0.0, max=1.0)


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# axis/sign -> prefix-sum doubling that accumulates the marker mask so a cell is
# >0 iff a marker lies at or "behind" it along the given direction.
def _prefix(g, src, axis, back):
    """axis 'col'/'row'; back=+1 reads the cell `d` before (smaller index)."""
    cur = src
    for d in _OFFS:
        if axis == 'col':
            kh, kw = 1, d + 1
            cc_cur = d if back > 0 else 0
            cc_beh = 0 if back > 0 else d
            pads = [0, d, 0, 0] if back > 0 else [0, 0, 0, d]
            taps = {(0, cc_cur): 1.0, (0, cc_beh): 1.0}
        else:
            kh, kw = d + 1, 1
            kr_cur = d if back > 0 else 0
            kr_beh = 0 if back > 0 else d
            pads = [d, 0, 0, 0] if back > 0 else [0, 0, d, 0]
            taps = {(kr_cur, 0): 1.0, (kr_beh, 0): 1.0}
        cur = g.stencil(cur, kh, kw, pads, taps)
    return cur


def _shift1(g, src, axis, back):
    """Return the value one cell behind (strictly), zero-filled at the border."""
    if axis == 'col':
        if back > 0:
            return g.stencil(src, 1, 2, [0, 1, 0, 0], {(0, 0): 1.0})
        return g.stencil(src, 1, 2, [0, 0, 0, 1], {(0, 1): 1.0})
    if back > 0:
        return g.stencil(src, 2, 1, [1, 0, 0, 0], {(0, 0): 1.0})
    return g.stencil(src, 2, 1, [0, 0, 1, 0], {(1, 0): 1.0})


def build_span(bg, f, do_h, do_v):
    g = _G()
    Wnb = np.zeros((1, CHANNELS, 1, 1), np.float32)
    for c in range(CHANNELS):
        if c != bg:
            Wnb[0, c, 0, 0] = 1.0
    xnb = g.conv1x1("input", Wnb)
    negone = g.init_f([1], [-1.0])

    betweens = []
    if do_h:
        pl = _prefix(g, xnb, 'col', +1)
        pr = _prefix(g, xnb, 'col', -1)
        ls = g.clip01(_shift1(g, pl, 'col', +1))
        rs = g.clip01(_shift1(g, pr, 'col', -1))
        bh = g.clip01(g.node("Add", [g.node("Sum", [ls, rs]), negone]))
        betweens.append(bh)
    if do_v:
        pu = _prefix(g, xnb, 'row', +1)
        pd = _prefix(g, xnb, 'row', -1)
        us = g.clip01(_shift1(g, pu, 'row', +1))
        ds = g.clip01(_shift1(g, pd, 'row', -1))
        bv = g.clip01(g.node("Add", [g.node("Sum", [us, ds]), negone]))
        betweens.append(bv)

    orf = g.clip01(g.node("Sum", betweens)) if len(betweens) > 1 else betweens[0]
    # gate to real background cells
    Wbg = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Wbg[0, bg, 0, 0] = 1.0
    xbg = g.conv1x1("input", Wbg)
    fillmask = g.clip01(g.node("Add", [g.node("Sum", [orf, xbg]), negone]))
    # scatter into channels f(+1) and bg(-1), add to input -> output (free tensor)
    Wd = np.zeros((CHANNELS, 1, 1, 1), np.float32)
    Wd[f, 0, 0, 0] = 1.0
    Wd[bg, 0, 0, 0] = -1.0
    D = g.conv1x1(fillmask, Wd)
    g.node("Add", ["input", D], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# HCONNECT: scalar "carry nearest non-bg colour" doubling scan                 #
# --------------------------------------------------------------------------- #
def _carry(g, cval, zero, back):
    """Fill-forward the nearest non-zero colour along +col (back=+1 -> from left)."""
    cur = cval
    for d in _OFFS:
        g_pos = g.node("Greater", [cur, zero])          # bool mask cur>0
        if back > 0:
            sh = g.stencil(cur, 1, d + 1, [0, d, 0, 0], {(0, 0): 1.0})  # cur[c-d]
        else:
            sh = g.stencil(cur, 1, d + 1, [0, 0, 0, d], {(0, d): 1.0})  # cur[c+d]
        cur = g.node("Where", [g_pos, cur, sh])
        # NB: bg==0 required for Greater test; enforced by detection
    return cur


def _eq_int(g, x, k, half):
    """bool mask (x == k) for integer-valued float field x (opset-10 safe)."""
    kc = g.init_f([1], [float(k)])
    diff = g.node("Sub", [x, kc])
    return g.node("Not", [g.node("Greater", [g.node("Abs", [diff]), half])])


def build_hconnect(bg):
    """bg must be 0 (colour field uses >0 as the non-bg test)."""
    g = _G()
    ramp = np.zeros((1, CHANNELS, 1, 1), np.float32)
    for c in range(CHANNELS):
        ramp[0, c, 0, 0] = float(c)
    cval = g.conv1x1("input", ramp)                     # scalar colour field
    zero = g.init_f([1], [0.0])
    half = g.init_f([1], [0.5])
    nl = _carry(g, cval, zero, +1)                      # nearest colour to the left
    nr = _carry(g, cval, zero, -1)                      # nearest colour to the right
    # same colour on both sides (integer equality via |nl-nr| < 0.5)
    eqcol = g.node("Not", [g.node("Greater",
                    [g.node("Abs", [g.node("Sub", [nl, nr])]), half])])
    bgpos = g.node("Not", [g.node("Greater", [cval, zero])])  # cell is background
    fillhere = g.node("And", [eqcol, bgpos])
    A = g.node("Where", [fillhere, nl, cval])           # cval==0 at bg -> A=0 when no join
    # one-hot delta: +1 on colour channel k (k>=1), -1 on the bg channel.
    pos = []
    for k in range(1, CHANNELS):
        pos.append(g.node("Cast", [_eq_int(g, A, k, half)], to=int(DATA_TYPE)))
    fillany = g.node("Sum", pos)                        # 1 where a colour was placed
    negbg = g.node("Neg", [fillany])
    delta = g.node("Concat", [negbg] + pos, axis=1)     # [1,10,30,30]
    g.node("Add", ["input", delta], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# detection / entry point                                                     #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _bg(prs):
    from collections import Counter
    cnt = Counter()
    for a, _ in prs:
        for v in a.ravel():
            cnt[int(v)] += 1
    return cnt.most_common(1)[0][0]


def candidates(ex):
    tt = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if len(tt) < 1 or not all(a.shape == b.shape for a, b in allp):
        return []
    if not any((a != b).any() for a, b in allp):
        return []
    bg = _bg(allp)
    if not (0 <= bg < CHANNELS):
        return []
    # every non-background input cell must be preserved (additive rules)
    for a, b in allp:
        if not np.array_equal(b[a != bg], a[a != bg]):
            return []

    out = []

    # ---- SPAN: single distinct fill colour on background cells ----
    fset = set()
    for a, b in allp:
        fset |= set(b[(a == bg) & (b != bg)].tolist())
    fset.discard(bg)
    if len(fset) == 1:
        f = fset.pop()
        for do_h, do_v in ((1, 1), (1, 0), (0, 1)):
            if all(np.array_equal(_span_np(a, bg, f, do_h, do_v), b) for a, b in tt) and \
               all(np.array_equal(_span_np(a, bg, f, do_h, do_v), b) for a, b in allp):
                tag = ("H" if do_h else "") + ("V" if do_v else "")
                try:
                    out.append((f"span{tag}_f{f}_bg{bg}", build_span(bg, f, do_h, do_v)))
                except Exception:
                    pass
                break

    # ---- HCONNECT: same-colour horizontal join, own colour (needs bg==0) ----
    if bg == 0:
        if all(np.array_equal(_hconnect_np(a, bg), b) for a, b in tt) and \
           all(np.array_equal(_hconnect_np(a, bg), b) for a, b in allp):
            try:
                out.append((f"hconnect_bg{bg}", build_hconnect(bg)))
            except Exception:
                pass

    return out
