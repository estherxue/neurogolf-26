"""Global color / frequency-selection family (origin-anchored, same shape).

All rules in this family pick colors by a GLOBAL per-color statistic (cell count)
and apply a single, position-independent color remap.  Because the one-hot tensor
is zero-padded to 30x30 with the grid at (0,0), per-color counts are a plain
`ReduceSum` over the spatial axes (padding is all-zero, contributes nothing) and
every remap is a 1x1 `Conv`, so the top-left origin is preserved for grids of any
size -> the rules generalise.

Implemented rules (the exact one is inferred from the train/test/arc-gen pairs):

  keep-max / keep-min   keep ONLY the most- (least-) frequent non-bg color,
                        recolor every other non-bg color to a fill color B
                        (B is inferred; B = background 0 is the common case).
  remove-max/remove-min recolor ONLY the most- (least-) frequent non-bg color to
                        B, keep the rest (X is the unique color satisfying the
                        global predicate "most / least common").
  swap-2-present        the grid has exactly two non-bg colors -> swap them
                        (every cell of one becomes the other and vice-versa).
  swap-max-min          swap the most-frequent and least-frequent non-bg colors.
  recolor-all-max/min   recolor EVERY non-bg cell to the most- (least-) frequent
                        non-bg color (a single inferred color).
  recolor-all-fixed     recolor every non-bg cell to one fixed inferred color.

Realisation (opset 10, cheap)
-----------------------------
`count = ReduceSum(input, axes=[2,3])`  -> [1,10,1,1] per-color counts.
A per-channel 0/1 gate for argmax / argmin (over present non-bg channels) is built
with ReduceMax/ReduceMin + Greater/Less + Cast.  The gate (a tiny [1,10,1,1]
tensor) is folded into a DYNAMIC 1x1 conv weight  W[o,i] in {0,1}  and applied
with a single bias-free `Conv`.  No [1,10,30,30] intermediate is ever
materialised (only the free output), so the cost is just a handful of params plus
tiny gate tensors -> high score.

Detection mirrors the ONNX semantics exactly (including tie behaviour) and only
emits a candidate when it reproduces EVERY available pair, so wrong hypotheses are
dropped before they reach the grader.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model, recolor_conv
from ng_utils_shim import DATA_TYPE, CHANNELS

INT64 = onnx.TensorProto.INT64
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


def _chan(g, perchan):
    """[1,10,1,1] float constant from a length-10 list."""
    return g.const([1, CHANNELS, 1, 1], list(perchan))


def _nbg_mask():
    """1 for non-background colors 1..9, 0 for background 0."""
    return [0.0] + [1.0] * (CHANNELS - 1)


def _ident_init(g):
    """[10,10,1,1] identity initializer (W[o,i]=o==i)."""
    vals = [0.0] * (CHANNELS * CHANNELS)
    for c in range(CHANNELS):
        vals[c * CHANNELS + c] = 1.0
    nm = g.name("ident")
    g.inits.append(oh.make_tensor(nm, DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], vals))
    return nm


def _count(g):
    """Per-channel cell count -> [1,10,1,1]."""
    return g.node("ReduceSum", ["input"], axes=[2, 3], keepdims=1)


# --------------------------------------------------------------------------- #
# per-channel argmax / argmin gate over PRESENT non-bg channels -> [1,10,1,1] #
# (1.0 at every winning channel, 0.0 elsewhere; mirrors the numpy reference).  #
# --------------------------------------------------------------------------- #
def _gate(g, count, rule):
    cm = _chan(g, _nbg_mask())
    half = g.const([1, 1, 1, 1], [0.5])
    if rule == "max":
        offneg = _chan(g, [(-_BIG if c == 0 else 0.0) for c in range(CHANNELS)])
        masked = g.node("Add", [g.node("Mul", [count, cm]), offneg])
        red = g.node("ReduceMax", [masked], axes=[1], keepdims=1)
        sb = g.node("Greater", [masked, g.node("Sub", [red, half])])
    else:  # min over present non-bg
        present = g.node("Mul", [g.node("Clip", [count], min=0.0, max=1.0), cm])
        ones = _chan(g, [1.0] * CHANNELS)
        bigc = g.const([1, 1, 1, 1], [_BIG])
        push = g.node("Mul", [g.node("Sub", [ones, present]), bigc])
        masked = g.node("Add", [count, push])
        red = g.node("ReduceMin", [masked], axes=[1], keepdims=1)
        sb = g.node("Less", [masked, g.node("Add", [red, half])])
    gate = g.node("Cast", [sb], to=DATA_TYPE)
    return g.node("Mul", [gate, cm])               # zero out background channel


def _T(g, x):
    """Transpose a [1,10,1,1] gate to [10,1,1,1] (index the conv OUTPUT dim)."""
    return g.node("Transpose", [x], perm=[1, 0, 2, 3])


# --------------------------------------------------------------------------- #
# ONNX builders                                                               #
# --------------------------------------------------------------------------- #
def build_keepremove(rule, mode, B):
    """keep/remove the argmax/argmin non-bg color; removed cells -> color B.
    s[i] = 1 keep own color, 0 -> B.  W = ident*s + eB*(1-s)."""
    g = _G()
    sel = _gate(g, _count(g), rule)                # [1,10,1,1] winners
    ones = _chan(g, [1.0] * CHANNELS)
    if mode == "keep":
        bg = _chan(g, [1.0] + [0.0] * (CHANNELS - 1))
        s = g.node("Add", [bg, sel])               # keep bg + winners
    else:                                          # remove winners, keep the rest
        s = g.node("Sub", [ones, sel])
    ident = _ident_init(g)
    eB = g.const([CHANNELS, 1, 1, 1], [1.0 if c == B else 0.0 for c in range(CHANNELS)])
    term1 = g.node("Mul", [ident, s])
    term2 = g.node("Mul", [eB, g.node("Sub", [ones, s])])
    W = g.node("Add", [term1, term2])
    g.node("Conv", ["input", W], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


def build_recolor_all(rule):
    """recolor EVERY non-bg cell to the argmax/argmin non-bg color.
    W = ident*(1-nbg) + gt_o*nbg  (all non-bg inputs funnel to the target)."""
    g = _G()
    gt = _gate(g, _count(g), rule)                 # [1,10,1,1] single target
    gt_o = _T(g, gt)                               # [10,1,1,1]
    nbg = _chan(g, _nbg_mask())
    ones = _chan(g, [1.0] * CHANNELS)
    ident = _ident_init(g)
    term1 = g.node("Mul", [ident, g.node("Sub", [ones, nbg])])
    term2 = g.node("Mul", [gt_o, nbg])
    W = g.node("Add", [term1, term2])
    g.node("Conv", ["input", W], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


def build_swap_maxmin():
    """swap the argmax and argmin non-bg colors.
    W = ident*(1-gmax-gmin) + gmax_o*gmin_i + gmin_o*gmax_i."""
    g = _G()
    count = _count(g)
    gmax = _gate(g, count, "max")
    gmin = _gate(g, count, "min")
    gmax_o, gmin_o = _T(g, gmax), _T(g, gmin)
    ones = _chan(g, [1.0] * CHANNELS)
    ident = _ident_init(g)
    diag = g.node("Mul", [ident, g.node("Sub", [ones, g.node("Add", [gmax, gmin])])])
    cross1 = g.node("Mul", [gmax_o, gmin])
    cross2 = g.node("Mul", [gmin_o, gmax])
    W = g.node("Add", [g.node("Add", [diag, cross1]), cross2])
    g.node("Conv", ["input", W], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


def build_swap_2present():
    """swap the (exactly two) present non-bg colors regardless of count.
    P[i]=present non-bg indicator.  W = ident*(1-P) + (P_i*P_o)*(1-ident)."""
    g = _G()
    count = _count(g)
    cm = _chan(g, _nbg_mask())
    P = g.node("Mul", [g.node("Clip", [count], min=0.0, max=1.0), cm])  # [1,10,1,1]
    P_o = _T(g, P)
    ones = _chan(g, [1.0] * CHANNELS)
    ident = _ident_init(g)
    notident = g.name("notident")
    nivals = [1.0] * (CHANNELS * CHANNELS)
    for c in range(CHANNELS):
        nivals[c * CHANNELS + c] = 0.0
    g.inits.append(oh.make_tensor(notident, DATA_TYPE, [CHANNELS, CHANNELS, 1, 1], nivals))
    term1 = g.node("Mul", [ident, g.node("Sub", [ones, P])])
    term2 = g.node("Mul", [g.node("Mul", [P, P_o]), notident])
    W = g.node("Add", [term1, term2])
    g.node("Conv", ["input", W], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics for detection)                  #
# --------------------------------------------------------------------------- #
def _counts(a):
    return {c: int((a == c).sum()) for c in range(1, CHANNELS) if (a == c).any()}


def _winners(a, rule):
    cnt = _counts(a)
    if not cnt:
        return set()
    m = (max if rule == "max" else min)(cnt.values())
    return {c for c, v in cnt.items() if v == m}


def _apply_keepremove(a, rule, mode, B):
    win = _winners(a, rule)
    out = a.copy()
    for c in _counts(a):
        if (mode == "keep" and c not in win) or (mode == "remove" and c in win):
            out[a == c] = B
    return out


def _apply_recolor_all(a, rule):
    win = _winners(a, rule)
    if len(win) != 1:
        return None
    out = a.copy()
    out[a != 0] = next(iter(win))
    return out


def _apply_swap_maxmin(a):
    mx, mn = _winners(a, "max"), _winners(a, "min")
    if len(mx) != 1 or len(mn) != 1:
        return None
    p, q = next(iter(mx)), next(iter(mn))
    if p == q:
        return None
    out = a.copy()
    out[a == p] = q
    out[a == q] = p
    return out


def _apply_swap_2present(a):
    pres = sorted(_counts(a))
    if len(pres) != 2:
        return None
    p, q = pres
    out = a.copy()
    out[a == p] = q
    out[a == q] = p
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


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):     # every rule preserves shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):   # no change -> not our family
        return []

    out = []

    # ---- swaps (bijective; not expressible by family_objects' single-B map) -- #
    if all(_apply_swap_2present(a) is not None and
           np.array_equal(_apply_swap_2present(a), b) for a, b in prs):
        out.append(("swap2present", build_swap_2present()))
    elif all(_apply_swap_maxmin(a) is not None and
             np.array_equal(_apply_swap_maxmin(a), b) for a, b in prs):
        out.append(("swapmaxmin", build_swap_maxmin()))

    # ---- recolor every non-bg cell to one inferred color ------------------- #
    for rule in ("max", "min"):
        if all(_apply_recolor_all(a, rule) is not None and
               np.array_equal(_apply_recolor_all(a, rule), b) for a, b in prs):
            out.append((f"recolorall_{rule}", build_recolor_all(rule)))
            break
    else:
        # fixed single target color (recolor all non-bg -> constant C)
        tgt = set()
        for a, b in prs:
            d = a != b
            if d.any():
                tgt |= set(np.unique(b[d]).tolist())
        if len(tgt) == 1:
            C = next(iter(tgt))
            if all(np.array_equal(np.where(a != 0, C, a), b) for a, b in prs):
                cmap = [0] + [C] * (CHANNELS - 1)
                out.append((f"recolorall_fix{C}", recolor_conv(cmap)))

    # ---- keep / remove the most- or least-frequent non-bg color ------------ #
    becomes = set()
    for a, b in prs:
        d = a != b
        if d.any():
            becomes |= set(np.unique(b[d]).tolist())
    Bcands = (set(becomes) | {0}) if len(becomes) <= 1 else set()
    done = False
    for mode in ("keep", "remove"):
        for rule in ("max", "min"):
            for B in sorted(Bcands):
                if all(np.array_equal(_apply_keepremove(a, rule, mode, B), b)
                       for a, b in prs):
                    out.append((f"{mode}_{rule}_B{B}", build_keepremove(rule, mode, B)))
                    done = True
                    break
            if done:
                break
        if done:
            break

    return out
