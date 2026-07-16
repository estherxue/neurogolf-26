"""Object / color-frequency selection family.

Global-but-expressible color selection rules applied to the one-hot grid:

  * keep ONLY the most-frequent (or least-frequent) non-background color and
    recolor every other object color to a fill color B (origin-anchored, same
    shape), or
  * remove the most/least-frequent color (keep the rest), or
  * keep colors whose cell-count satisfies a fixed threshold (count >/>=/... T),
  * (small-output variants) emit a 1x1 grid whose single cell is the
    frequency-selected color.

Why it is expressible under the padding gotcha
----------------------------------------------
Per-channel cell counts are a `ReduceSum` over the spatial axes; padding cells
are all-zero in the one-hot tensor so they never contribute.  The winning
count is found with `ReduceMax`/`ReduceMin` over the channel axis, turned into a
per-channel 0/1 gate with `Greater`/`Less` + `Cast`, and applied with a
broadcast `Mul`.  Everything is either pointwise or a global reduction, so the
top-left origin is preserved for grids of any size -> generalizes.

Same-shape recolor output construction
--------------------------------------
Let s[c] in {0,1} be the per-channel "keep its own color" gate (excluded
background / fill colors are always kept).  With B the fill color:

    gated   = input * s                       (removed colors -> all-zero cell)
    removed = sum_c input[c] - sum_c gated[c] (1 exactly at recolored real cells)
    output  = gated + onehot_B * removed

This sends every removed real cell to channel B and leaves padding untouched
(its column is all-zero throughout).  Only `gated` and the broadcast
`onehot_B*removed` are full [1,10,30,30] intermediates (the output tensor is
free); the rest are tiny [1,10,1,1] / [1,1,30,30] tensors.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
_BIG = 1.0e6


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

    def const(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         [float(v) for v in vals]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _chan_const(g, perchan):
    """[1,10,1,1] constant from a length-10 list."""
    return g.const([1, CHANNELS, 1, 1], list(perchan))


# --------------------------------------------------------------------------- #
# gate builder: produce a [1,10,1,1] float gate `selgate` = 1 for the SELECTED #
# candidate channels (argmax / argmin / threshold over candidate counts).      #
# --------------------------------------------------------------------------- #
def _selgate(g, count, excl, rule, T=0):
    candmask = [0.0 if c in excl else 1.0 for c in range(CHANNELS)]
    cm = _chan_const(g, candmask)
    half = g.const([1, 1, 1, 1], [0.5])

    if rule == "max":
        offneg = _chan_const(g, [(-_BIG if c in excl else 0.0) for c in range(CHANNELS)])
        masked = g.node("Add", [g.node("Mul", [count, cm]), offneg])
        red = g.node("ReduceMax", [masked], axes=[1], keepdims=1)
        thr = g.node("Sub", [red, half])
        sb = g.node("Greater", [masked, thr])
    elif rule == "min":
        # exclude non-candidates AND absent candidates (count 0) by pushing to +BIG
        p = g.node("Clip", [count], min=0.0, max=1.0)        # present indicator
        pc = g.node("Mul", [p, cm])
        ones = _chan_const(g, [1.0] * CHANNELS)
        bigc = g.const([1, 1, 1, 1], [_BIG])
        bigterm = g.node("Mul", [g.node("Sub", [ones, pc]), bigc])
        masked = g.node("Add", [count, bigterm])
        red = g.node("ReduceMin", [masked], axes=[1], keepdims=1)
        thr = g.node("Add", [red, half])
        sb = g.node("Less", [masked, thr])
    else:  # fixed threshold on candidate counts: compare count vs T
        offneg = _chan_const(g, [(-_BIG if c in excl else 0.0) for c in range(CHANNELS)])
        masked = g.node("Add", [g.node("Mul", [count, cm]), offneg])
        tc = g.const([1, 1, 1, 1], [float(T)])
        if rule == "gt":
            sb = g.node("Greater", [masked, tc])
        elif rule == "ge":
            sb = g.node("Greater", [masked, g.const([1, 1, 1, 1], [T - 0.5])])
        elif rule == "lt":
            # candidates only (excluded are -BIG -> would pass Less); guard with present
            tc2 = g.const([1, 1, 1, 1], [float(T)])
            ltb = g.node("Less", [masked, tc2])
            geb = g.node("Greater", [masked, g.const([1, 1, 1, 1], [0.5])])  # present
            sb = g.node("And", [ltb, geb])
        else:  # le
            tc2 = g.const([1, 1, 1, 1], [T + 0.5])
            ltb = g.node("Less", [masked, tc2])
            geb = g.node("Greater", [masked, g.const([1, 1, 1, 1], [0.5])])
            sb = g.node("And", [ltb, geb])
    return g.node("Cast", [sb], to=DATA_TYPE)


# --------------------------------------------------------------------------- #
# same-shape recolor model                                                    #
# --------------------------------------------------------------------------- #
def build_recolor(B, excl, rule, mode, T=0):
    g = _G()
    count = g.node("ReduceSum", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    selgate = _selgate(g, count, excl, rule, T)

    if mode == "keep":
        exclmask = _chan_const(g, [1.0 if c in excl else 0.0 for c in range(CHANNELS)])
        s = g.node("Add", [selgate, exclmask])
    else:  # remove selected, keep the rest
        ones = _chan_const(g, [1.0] * CHANNELS)
        s = g.node("Sub", [ones, selgate])

    gated = g.node("Mul", ["input", s])                              # BIG [1,10,30,30]
    ti = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    tg = g.node("ReduceSum", [gated], axes=[1], keepdims=1)          # [1,1,30,30]
    removed = g.node("Sub", [ti, tg])                                # [1,1,30,30]
    maskB = _chan_const(g, [1.0 if c == B else 0.0 for c in range(CHANNELS)])
    removed_b = g.node("Mul", [removed, maskB])                      # BIG [1,10,30,30]
    g.node("Add", [gated, removed_b], "output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# same-shape recolor via a single 1x1 Conv with a DYNAMICALLY computed weight  #
# matrix W[o,i] = (o==i)*s[i] + (o==B)*(1-s[i]).  No [1,10,30,30] intermediate  #
# is materialized (only the free output), so cost is just params + tiny gate   #
# tensors -> far cheaper than the broadcast-add construction.                  #
# --------------------------------------------------------------------------- #
def build_recolor_conv(B, excl, rule, mode, T=0):
    g = _G()
    count = g.node("ReduceSum", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    selgate = _selgate(g, count, excl, rule, T)

    if mode == "keep":
        exclmask = _chan_const(g, [1.0 if c in excl else 0.0 for c in range(CHANNELS)])
        s = g.node("Add", [selgate, exclmask])                       # [1,10,1,1]
    else:
        ones = _chan_const(g, [1.0] * CHANNELS)
        s = g.node("Sub", [ones, selgate])

    # identity [O,I,1,1]; e_B as [O,1,1,1]
    ident = [0.0] * (CHANNELS * CHANNELS)
    for c in range(CHANNELS):
        ident[c * CHANNELS + c] = 1.0
    identc = g.const([CHANNELS, CHANNELS, 1, 1], ident)
    eBc = g.const([CHANNELS, 1, 1, 1], [1.0 if c == B else 0.0 for c in range(CHANNELS)])
    onesI = _chan_const(g, [1.0] * CHANNELS)
    term1 = g.node("Mul", [identc, s])                               # [10,10,1,1]
    oneminus = g.node("Sub", [onesI, s])                             # [1,10,1,1]
    term2 = g.node("Mul", [eBc, oneminus])                           # [10,10,1,1]
    W = g.node("Add", [term1, term2])                                # [10,10,1,1]
    g.node("Conv", ["input", W], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# 1x1 small-output model: single cell at (0,0) = selected color               #
# --------------------------------------------------------------------------- #
def build_pick1(excl, rule, T=0):
    g = _G()
    count = g.node("ReduceSum", ["input"], axes=[2, 3], keepdims=1)
    selgate = _selgate(g, count, excl, rule, T)
    pos = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    pos[0, 0, 0, 0] = 1.0
    posc = g.const([1, 1, HEIGHT, WIDTH], pos.ravel().tolist())
    g.node("Mul", [selgate, posc], "output")                        # [1,10,30,30]
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX semantics) for detection                  #
# --------------------------------------------------------------------------- #
def _counts(a, excl):
    return {c: int((a == c).sum()) for c in np.unique(a).tolist() if c not in excl}


def _selected(cnt, rule, T):
    if not cnt:
        return set()
    if rule == "max":
        m = max(cnt.values())
        return {c for c, v in cnt.items() if v == m}
    if rule == "min":
        m = min(cnt.values())
        return {c for c, v in cnt.items() if v == m}
    cmp = {"gt": lambda v: v > T, "ge": lambda v: v >= T,
           "lt": lambda v: v < T, "le": lambda v: v <= T}[rule]
    return {c for c, v in cnt.items() if cmp(v)}


def _apply_recolor(a, B, excl, rule, mode, T):
    cnt = _counts(a, excl)
    sel = _selected(cnt, rule, T)
    keep = sel if mode == "keep" else (set(cnt) - sel)
    out = a.copy()
    for c in cnt:
        if c not in keep:
            out[a == c] = B
    return out


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


_RULES = ["max", "min", "gt", "ge", "lt", "le"]
_THRS = range(0, 6)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    same_shape = all(a.shape == b.shape for a, b in prs)
    changes = any(not np.array_equal(a, b) for a, b in prs)

    # ---- same-shape color-frequency recolor ------------------------------- #
    if same_shape and changes:
        becomes = set()
        for a, b in prs:
            d = a != b
            if d.any():
                becomes |= set(np.unique(b[d]).tolist())
        Bcands = list(becomes) if len(becomes) == 1 else []

        # enumerate (rule, T) hypotheses: argmax/argmin and fixed thresholds
        hyps = [("max", 0), ("min", 0)]
        for rl in ("gt", "ge", "lt", "le"):
            for T in _THRS:
                hyps.append((rl, T))

        matched = None
        for B in Bcands:
            for excl in ({0, B}, {B}, {0}):
                for mode in ("keep", "remove"):
                    for rule, T in hyps:
                        if all(np.array_equal(
                                _apply_recolor(a, B, excl, rule, mode, T), b)
                                for a, b in prs):
                            matched = (B, excl, rule, mode, T)
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                break

        if matched:
            B, excl, rule, mode, T = matched
            tag = rule + (str(T) if rule not in ("max", "min") else "")
            # cheap single-Conv first, broadcast-add fallback second
            for label, fn in (("conv", build_recolor_conv), ("bcast", build_recolor)):
                try:
                    m = fn(B, excl, rule, mode, T)
                except Exception:
                    m = None
                if m is not None:
                    out.append((f"recolor_{label}_{tag}_{mode}_B{B}", m))

    # ---- 1x1 small output: pick the frequency-selected color -------------- #
    if all(b.shape == (1, 1) for a, b in prs):
        for excl in ({0},):
            for rule in ("max", "min"):
                ok = True
                for a, b in prs:
                    cnt = _counts(a, excl)
                    sel = _selected(cnt, rule, 0)
                    if len(sel) != 1 or int(b[0, 0]) != next(iter(sel)):
                        ok = False
                        break
                if ok:
                    try:
                        m = build_pick1(excl, rule)
                    except Exception:
                        m = None
                    if m is not None:
                        out.append((f"pick1_{rule}", m))
                    break

    return out
