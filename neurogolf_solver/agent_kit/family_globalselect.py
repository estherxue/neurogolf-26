"""Global color SELECTION / RECOLOR by per-color reductions (opset-10 rich ops).

Every rule here picks colors by a GLOBAL per-color statistic (the cell count) and
applies a single, position-independent color transform.  Because the one-hot
tensor is zero-padded to 30x30 with the grid anchored at (0,0):

  * per-color counts are a plain ``ReduceSum`` over the spatial axes (padding is
    all-zero, contributes nothing) -- including the background channel 0, whose
    count = number of *real* background cells (padding cells are all-zero, so they
    do NOT inflate channel 0);
  * "is this a real cell?" is ``ReduceSum`` over the channel axis (1 at every real
    cell, 0 at padding);
  * every recolor is a (possibly data-dependent) 1x1 ``Conv`` or a broadcast
    ``Mul``, so the top-left origin is preserved for grids of ANY size -> the
    rules generalise structurally.

Rules (the exact one is inferred from the train/test/arc-gen pairs)
------------------------------------------------------------------
  fillall_max     fill the WHOLE grid (every real cell, background included) with
                  the single most-frequent cell value counting the background too
                  -- ``ReduceMax`` over all 10 channel counts, then broadcast the
                  winner one-hot over the real-cell mask.
  rankrev         swap colors by FREQUENCY RANK: the i-th most frequent non-bg
                  color becomes the i-th least frequent (full rank reversal; the
                  generalisation of "swap two colors by frequency rank").  Built by
                  pairwise count comparisons -> per-color ranks -> a permutation
                  conv weight, all with Greater/Less/ReduceSum (no Equal, which is
                  int-only at opset 10).
  max2min_rm /    recolor the argmax-count color to the argmin color and erase the
  min2max_rm      argmin (-> background), or the mirror.
  swap2 /         swap the two present non-bg colors, or the argmax/argmin colors.
  swapmaxmin
  recolorall_max/ recolor every NON-bg cell to the argmax/argmin non-bg color
  recolorall_min  (background kept).
  keep_* / remove_* keep / remove the argmax / argmin / threshold-selected non-bg
                  colors, recoloring the others to an inferred fill color B.

Realisation is cheap: gates are tiny [1,10,1,1] tensors folded into a DYNAMIC 1x1
conv weight ``W[o,i]`` (no [1,10,30,30] intermediate is materialised except the
free output), so cost is a handful of params + tiny gate tensors -> high score.

Detection mirrors the ONNX semantics exactly (including tie behaviour) and only
emits a candidate when it reproduces EVERY available pair, so wrong hypotheses are
dropped before they reach the grader.  Extends family_objects/objects2 with the
ArgMax/Where (fill-all, rank-reversal, max<->min-erase) variants.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
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


def _e0_o(g):
    """[10,1,1,1] output-channel one-hot for background color 0."""
    return g.const([CHANNELS, 1, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))


def _count(g):
    """Per-channel cell count -> [1,10,1,1] (channel 0 = real background cells)."""
    return g.node("ReduceSum", ["input"], axes=[2, 3], keepdims=1)


def _T(g, x):
    """Transpose a [1,10,1,1] tensor to [10,1,1,1] (index the conv OUTPUT dim)."""
    return g.node("Transpose", [x], perm=[1, 0, 2, 3])


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


# --------------------------------------------------------------------------- #
# ONNX builders                                                               #
# --------------------------------------------------------------------------- #
def build_fill_all():
    """Fill every real cell (background included) with the most-frequent cell
    value, counting the background.  output[:,j,y,x] = gate[j] * realmask[y,x]."""
    g = _G()
    count = _count(g)                              # [1,10,1,1] incl. background
    mx = g.node("ReduceMax", [count], axes=[1], keepdims=1)   # [1,1,1,1]
    half = g.const([1, 1, 1, 1], [0.5])
    gate = g.node("Cast", [g.node("Greater", [count, g.node("Sub", [mx, half])])],
                  to=DATA_TYPE)                    # [1,10,1,1] winner one-hot
    realmask = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)  # [1,1,30,30]
    g.node("Mul", [gate, realmask], "output")      # broadcast -> [1,10,30,30]
    return _model(g.nodes, g.inits)


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
    """recolor EVERY non-bg cell to the argmax/argmin non-bg color (bg kept).
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
    """swap the argmax and argmin non-bg colors."""
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
    """swap the (exactly two) present non-bg colors regardless of count."""
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


def build_max2min_remove(direction):
    """direction 'max2min': argmax color -> argmin color, argmin color -> bg.
    direction 'min2max': argmin color -> argmax color, argmax color -> bg.
    Everything else identity."""
    g = _G()
    count = _count(g)
    gmax = _gate(g, count, "max")
    gmin = _gate(g, count, "min")
    gmax_o, gmin_o = _T(g, gmax), _T(g, gmin)
    ones = _chan(g, [1.0] * CHANNELS)
    ident = _ident_init(g)
    e0 = _e0_o(g)
    diag = g.node("Mul", [ident, g.node("Sub", [ones, g.node("Add", [gmax, gmin])])])
    if direction == "max2min":
        cross_recolor = g.node("Mul", [gmin_o, gmax])   # input argmax -> output argmin color
        cross_erase = g.node("Mul", [e0, gmin])         # input argmin -> bg
    else:                                               # min2max
        cross_recolor = g.node("Mul", [gmax_o, gmin])   # input argmin -> output argmax color
        cross_erase = g.node("Mul", [e0, gmax])         # input argmax -> bg
    W = g.node("Add", [g.node("Add", [diag, cross_recolor]), cross_erase])
    g.node("Conv", ["input", W], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


def build_rank_reverse():
    """Swap non-bg colors by frequency rank (i-th most frequent <-> i-th least).
    rank_asc(i)  = #{present non-bg k : count_k < count_i}
    rank_desc(i) = #{present non-bg k : count_k > count_i}
    output color j of input color i iff rank_asc(j) == rank_desc(i) (both present);
    background is kept.  Requires distinct counts (enforced by detection)."""
    g = _G()
    count = _count(g)                              # [1,10,1,1] (axis1 = color i)
    nbg = _chan(g, _nbg_mask())
    present = g.node("Mul", [g.node("Clip", [count], min=0.0, max=1.0), nbg])  # p_i [1,10,1,1]
    countT = _T(g, count)                          # [10,1,1,1] (axis0 = color k)
    presentT = _T(g, present)                      # p_k [10,1,1,1]
    half = g.const([1, 1, 1, 1], [0.5])

    # G[k,i] = (count_i > count_k); rank_asc(i) = sum_k p_k * G[k,i]
    gmat = g.node("Cast", [g.node("Greater", [count, countT])], to=DATA_TYPE)   # [10,10,1,1]
    ra = g.node("ReduceSum", [g.node("Mul", [gmat, presentT])], axes=[0], keepdims=1)  # [1,10,1,1]
    # L[k,i] = (count_i < count_k); rank_desc(i) = sum_k p_k * L[k,i]
    lmat = g.node("Cast", [g.node("Less", [count, countT])], to=DATA_TYPE)      # [10,10,1,1]
    rd = g.node("ReduceSum", [g.node("Mul", [lmat, presentT])], axes=[0], keepdims=1)  # [1,10,1,1]

    ra_j = _T(g, ra)                               # [10,1,1,1] (axis0 = output color j)
    diff = g.node("Sub", [ra_j, rd])               # [10,10,1,1] = ra_j - rd_i
    match = g.node("Cast", [g.node("Less", [g.node("Abs", [diff]), half])], to=DATA_TYPE)
    W = g.node("Mul", [g.node("Mul", [match, presentT]), present])  # mask present j and i
    # keep background identity (W[0,0] = 1)
    e00 = g.const([CHANNELS, CHANNELS, 1, 1],
                  [1.0 if (o == 0 and i == 0) else 0.0
                   for o in range(CHANNELS) for i in range(CHANNELS)])
    W = g.node("Add", [W, e00])
    g.node("Conv", ["input", W], "output", kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics for detection)                  #
# --------------------------------------------------------------------------- #
def _counts(a):
    """non-bg color -> count (only present colors)."""
    return {c: int((a == c).sum()) for c in range(1, CHANNELS) if (a == c).any()}


def _fullcounts(a):
    """all colors 0..9 -> count over real cells (background included)."""
    return {c: int((a == c).sum()) for c in range(CHANNELS)}


def _uniq(cnt, fn):
    if not cnt:
        return None
    m = fn(cnt.values())
    w = [c for c, v in cnt.items() if v == m]
    return w[0] if len(w) == 1 else None


def _apply_fillall(a):
    fc = _fullcounts(a)
    m = max(fc.values())
    w = [c for c, v in fc.items() if v == m]
    if len(w) != 1:
        return None
    return np.full_like(a, w[0])


def _apply_keepremove(a, rule, mode, B):
    cnt = _counts(a)
    if not cnt:
        return a.copy()
    m = (max if rule == "max" else min)(cnt.values())
    win = {c for c, v in cnt.items() if v == m}
    out = a.copy()
    for c in cnt:
        if (mode == "keep" and c not in win) or (mode == "remove" and c in win):
            out[a == c] = B
    return out


def _apply_recolor_all(a, rule):
    w = _uniq(_counts(a), max if rule == "max" else min)
    if w is None:
        return None
    out = a.copy()
    out[a != 0] = w
    return out


def _apply_swap_maxmin(a):
    cnt = _counts(a)
    p = _uniq(cnt, max)
    q = _uniq(cnt, min)
    if p is None or q is None or p == q:
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


def _apply_max2min(a, direction):
    cnt = _counts(a)
    p = _uniq(cnt, max)      # argmax color
    q = _uniq(cnt, min)      # argmin color
    if p is None or q is None or p == q:
        return None
    out = a.copy()
    if direction == "max2min":
        out[a == p] = q
        out[a == q] = 0
    else:
        out[a == q] = p
        out[a == p] = 0
    return out


def _apply_rankrev(a):
    cnt = _counts(a)
    if not cnt:
        return None
    cols = sorted(cnt, key=lambda c: cnt[c])      # ascending count
    vals = [cnt[c] for c in cols]
    if len(set(vals)) != len(vals):               # distinct counts required
        return None
    rev = cols[::-1]
    mp = {cols[i]: rev[i] for i in range(len(cols))}
    out = a.copy()
    for c, d in mp.items():
        out[a == c] = d
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


def _matches(prs, fn):
    for a, b in prs:
        o = fn(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):     # every rule preserves shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):   # no change -> not our family
        return []

    out = []

    # ---- fill the whole grid with the dominant cell value (bg counted) ------ #
    if _matches(prs, _apply_fillall):
        out.append(("fillall_max", build_fill_all()))

    # ---- frequency-rank reversal (generalised swap-by-rank) ----------------- #
    if _matches(prs, _apply_rankrev):
        out.append(("rankrev", build_rank_reverse()))

    # ---- recolor argmax->argmin (or mirror) and erase the partner ----------- #
    for d in ("max2min", "min2max"):
        if _matches(prs, lambda a, dd=d: _apply_max2min(a, dd)):
            out.append((f"{d}_rm", build_max2min_remove(d)))
            break

    # ---- bijective swaps ---------------------------------------------------- #
    if _matches(prs, _apply_swap_2present):
        out.append(("swap2", build_swap_2present()))
    elif _matches(prs, _apply_swap_maxmin):
        out.append(("swapmaxmin", build_swap_maxmin()))

    # ---- recolor every non-bg cell to one frequency-selected color ---------- #
    for rule in ("max", "min"):
        if _matches(prs, lambda a, r=rule: _apply_recolor_all(a, r)):
            out.append((f"recolorall_{rule}", build_recolor_all(rule)))
            break

    # ---- keep / remove the most- or least-frequent non-bg color ------------- #
    becomes = set()
    for a, b in prs:
        d = a != b
        if d.any():
            becomes |= set(np.unique(b[d]).tolist())
    Bcands = sorted(set(becomes) | {0}) if len(becomes) <= 1 else []
    done = False
    for mode in ("keep", "remove"):
        for rule in ("max", "min"):
            for B in Bcands:
                if _matches(prs, lambda a, r=rule, mm=mode, bb=B:
                            _apply_keepremove(a, r, mm, bb)):
                    out.append((f"{mode}_{rule}_B{B}", build_keepremove(rule, mode, B)))
                    done = True
                    break
            if done:
                break
        if done:
            break

    return out
