"""family_crack2 — structural ARC->ONNX solvers for a slice of NeuroGolf 2026.

Each detector infers its rule from train+test (+arc-gen) numpy pairs and, when the
rule matches EXACTLY on every pair, emits a static opset-10 graph.  All graphs keep
content origin-anchored (top-left) so they survive the zero-padding contract.

Solved tasks (rules):
  * t43  : column-markers (top row, color 5) x row-markers (last col, color 5) ->
           fill the outer-product background cells with color 2.  (const 10x10)
  * t130 : 9x9 grid of 3x3 monochrome blocks; downscale to 3x3 taking each block's
           fill color (3x3-block max-pool), with the noise color 5 suppressed.
  * t271 : 9x9 grid of 3x3 blocks of two colors; downscale to 3x3 taking each
           block's MAJORITY color.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(ex):
    out = []
    for e in ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", []):
        a = np.array(e["input"], int)
        b = np.array(e["output"], int)
        if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
            return None
        if max(a.shape) > 30 or max(b.shape) > 30:
            continue
        out.append((a, b))
    return out


# --------------------------------------------------------------------------- #
# t43: outer-product projection                                               #
# --------------------------------------------------------------------------- #
def _detect_43(prs):
    """All pairs 10x10 same shape; marker color M in top row (cols) and last col
    (rows); fill color F placed at background cells (r,c) where row r and col c are
    both marked.  Returns (M, F) or None."""
    M, F = 5, 2
    for a, b in prs:
        if a.shape != (10, 10) or b.shape != (10, 10):
            return None
        colmask = a[0, :] == M
        rowmask = a[:, 9] == M
        exp = a.copy()
        for r in range(10):
            for c in range(10):
                if rowmask[r] and colmask[c] and a[r, c] == 0:
                    exp[r, c] = F
        if not np.array_equal(exp, b):
            return None
    return (M, F)


def _build_43(M, F):
    H = W = 30
    nodes, inits = [], []

    def kt(name, shape, vals, dt=INT64):
        t = oh.make_tensor(name, dt, shape, vals)
        inits.append(t)
        return name

    # channel M (marker) -> [1,1,30,30]
    kt("m_s", [3], [M, 0, 0]); kt("m_e", [3], [M + 1, H, W]); kt("m_a", [3], [1, 2, 3])
    nodes.append(oh.make_node("Slice", ["input", "m_s", "m_e", "m_a"], ["chM"]))
    # channel 0 (background) -> [1,1,30,30]
    kt("z_s", [3], [0, 0, 0]); kt("z_e", [3], [1, H, W]); kt("z_a", [3], [1, 2, 3])
    nodes.append(oh.make_node("Slice", ["input", "z_s", "z_e", "z_a"], ["ch0"]))
    # colmask = chM[:, :, 0:1, :]  -> [1,1,1,30]
    kt("c_s", [2], [0, 0]); kt("c_e", [2], [1, W]); kt("c_a", [2], [2, 3])
    nodes.append(oh.make_node("Slice", ["chM", "c_s", "c_e", "c_a"], ["colmask"]))
    # rowmask = chM[:, :, :, 9:10] -> [1,1,30,1]
    kt("r_s", [2], [0, 9]); kt("r_e", [2], [H, 10]); kt("r_a", [2], [2, 3])
    nodes.append(oh.make_node("Slice", ["chM", "r_s", "r_e", "r_a"], ["rowmask"]))
    # outer = colmask * rowmask -> [1,1,30,30]; then *ch0 (only background cells)
    nodes.append(oh.make_node("Mul", ["colmask", "rowmask"], ["outer"]))
    nodes.append(oh.make_node("Mul", ["outer", "ch0"], ["outer_bg"]))
    # delta: channel F += outer_bg, channel 0 -= outer_bg
    coef = [0.0] * CHANNELS
    coef[0] = -1.0
    coef[F] = 1.0
    kt("coef", [1, CHANNELS, 1, 1], coef, dt=DATA_TYPE)
    nodes.append(oh.make_node("Mul", ["outer_bg", "coef"], ["delta"]))
    nodes.append(oh.make_node("Add", ["input", "delta"], ["output"]))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# t130 / t271: 3x3 block downscale of a 9x9 grid                              #
# --------------------------------------------------------------------------- #
def _blocks_9x9(a):
    """Yield (bi,bj, 3x3 block) for a 9x9 grid."""
    for bi in range(3):
        for bj in range(3):
            yield bi, bj, a[bi * 3:bi * 3 + 3, bj * 3:bj * 3 + 3]


def _detect_130(prs):
    """9x9 -> 3x3.  Each 3x3 block is monochrome fill C (>0, !=5) with optional 5
    speck noise, OR fully background.  Output[bi,bj] = fill color (0 if empty).
    Blocks must never mix a color with background (so block max-pool is exact).
    Returns True if matches."""
    for a, b in prs:
        if a.shape != (9, 9) or b.shape != (3, 3):
            return False
        for bi, bj, blk in _blocks_9x9(a):
            vals = blk.flatten()
            colors = set(vals.tolist()) - {0, 5}
            if len(colors) > 1:
                return False
            if colors:
                c = colors.pop()
                # block must be fully c (besides 5 specks) — no background mix
                if np.any((vals != c) & (vals != 5)):
                    return False
                if b[bi, bj] != c:
                    return False
            else:
                if b[bi, bj] != 0:
                    return False
    return True


def _detect_271(prs):
    """9x9 -> 3x3.  Each 3x3 block holds <=2 nonzero colors; output = strict
    majority color (count tie never occurs).  Returns the (lo,hi) color pair used
    across the task if it is a clean 2-color majority, else None."""
    palette = set()
    for a, b in prs:
        if a.shape != (9, 9) or b.shape != (3, 3):
            return None
        for bi, bj, blk in _blocks_9x9(a):
            vals = blk.flatten()
            colors = [v for v in vals.tolist() if v != 0]
            cset = set(colors)
            palette |= cset
            if not colors:
                if b[bi, bj] != 0:
                    return None
                continue
            # majority
            counts = {c: colors.count(c) for c in cset}
            mx = max(counts.values())
            winners = [c for c, n in counts.items() if n == mx]
            if len(winners) != 1 or winners[0] != b[bi, bj]:
                return None
    if len(palette) != 2:
        return None
    return tuple(sorted(palette))


def _build_blockmax(zero_color=5):
    """9x9 -> 3x3 block max-pool, anchored top-left, with `zero_color` suppressed."""
    nodes, inits = [], []

    def kt(name, shape, vals, dt=INT64):
        inits.append(oh.make_tensor(name, dt, shape, vals)); return name

    kt("s9_s", [2], [0, 0]); kt("s9_e", [2], [9, 9]); kt("s9_a", [2], [2, 3])
    nodes.append(oh.make_node("Slice", ["input", "s9_s", "s9_e", "s9_a"], ["g9"]))
    kt("rs6", [6], [1, CHANNELS, 3, 3, 3, 3])
    nodes.append(oh.make_node("Reshape", ["g9", "rs6"], ["g6"]))
    nodes.append(oh.make_node("ReduceMax", ["g6"], ["pooled"], axes=[3, 5], keepdims=0))
    mask = [1.0] * CHANNELS
    if zero_color is not None:
        mask[zero_color] = 0.0
    kt("zmask", [1, CHANNELS, 1, 1], mask, dt=DATA_TYPE)
    nodes.append(oh.make_node("Mul", ["pooled", "zmask"], ["clean"]))
    nodes.append(oh.make_node("Pad", ["clean"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - 3, WIDTH - 3]))
    return _model(nodes, inits)


def _build_blockmajority(lo, hi):
    """9x9 -> 3x3, output the majority of the two colors lo/hi per block.

    Per block: s_lo = sum(channel lo), s_hi = sum(channel hi), tot = s_lo+s_hi.
    out channel hi = (s_hi >= s_lo) & (tot>0); out channel lo = (s_lo > s_hi).
    Implemented with block-sum (reshape+ReduceSum) then Greater/Where -> one-hot.
    """
    nodes, inits = [], []

    def kt(name, shape, vals, dt=INT64):
        inits.append(oh.make_tensor(name, dt, shape, vals)); return name

    kt("s9_s", [2], [0, 0]); kt("s9_e", [2], [9, 9]); kt("s9_a", [2], [2, 3])
    nodes.append(oh.make_node("Slice", ["input", "s9_s", "s9_e", "s9_a"], ["g9"]))
    kt("rs6", [6], [1, CHANNELS, 3, 3, 3, 3])
    nodes.append(oh.make_node("Reshape", ["g9", "rs6"], ["g6"]))
    nodes.append(oh.make_node("ReduceSum", ["g6"], ["bsum"], axes=[3, 5], keepdims=0))
    # bsum: [1,10,3,3]; slice the two color channels
    kt("lo_s", [1], [lo]); kt("lo_e", [1], [lo + 1]); kt("lo_a", [1], [1])
    nodes.append(oh.make_node("Slice", ["bsum", "lo_s", "lo_e", "lo_a"], ["slo"]))
    kt("hi_s", [1], [hi]); kt("hi_e", [1], [hi + 1]); kt("hi_a", [1], [1])
    nodes.append(oh.make_node("Slice", ["bsum", "hi_s", "hi_e", "hi_a"], ["shi"]))
    # hi wins if shi >= slo and (slo+shi)>0  -> hi_win
    nodes.append(oh.make_node("Greater", ["slo", "shi"], ["lo_strict"]))  # slo>shi
    nodes.append(oh.make_node("Add", ["slo", "shi"], ["tot"]))
    kt("zero1", [1, 1, 1, 1], [0.0], dt=DATA_TYPE)
    nodes.append(oh.make_node("Greater", ["tot", "zero1"], ["nonempty"]))
    # lo_win = lo_strict & nonempty ; hi_win = (~lo_strict) & nonempty
    nodes.append(oh.make_node("Cast", ["lo_strict"], ["lo_strict_f"], to=DATA_TYPE))
    nodes.append(oh.make_node("Cast", ["nonempty"], ["nonempty_f"], to=DATA_TYPE))
    nodes.append(oh.make_node("Mul", ["lo_strict_f", "nonempty_f"], ["lo_win"]))
    nodes.append(oh.make_node("Sub", ["nonempty_f", "lo_win"], ["hi_win"]))  # nonempty & ~lo
    # background channel0 = 1 - nonempty
    kt("one1", [1, 1, 1, 1], [1.0], dt=DATA_TYPE)
    nodes.append(oh.make_node("Sub", ["one1", "nonempty_f"], ["bg"]))
    # assemble 10-channel one-hot [1,10,3,3] by scaling each win into its channel.
    # delta channels: ch0 += bg, ch_lo += lo_win, ch_hi += hi_win
    coef_lo = [0.0] * CHANNELS; coef_lo[lo] = 1.0
    coef_hi = [0.0] * CHANNELS; coef_hi[hi] = 1.0
    coef_bg = [0.0] * CHANNELS; coef_bg[0] = 1.0
    kt("clo", [1, CHANNELS, 1, 1], coef_lo, dt=DATA_TYPE)
    kt("chi", [1, CHANNELS, 1, 1], coef_hi, dt=DATA_TYPE)
    kt("cbg", [1, CHANNELS, 1, 1], coef_bg, dt=DATA_TYPE)
    nodes.append(oh.make_node("Mul", ["lo_win", "clo"], ["tlo"]))
    nodes.append(oh.make_node("Mul", ["hi_win", "chi"], ["thi"]))
    nodes.append(oh.make_node("Mul", ["bg", "cbg"], ["tbg"]))
    nodes.append(oh.make_node("Add", ["tlo", "thi"], ["tlh"]))
    nodes.append(oh.make_node("Add", ["tlh", "tbg"], ["small"]))
    nodes.append(oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - 3, WIDTH - 3]))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# t214: rotate-block into fixed 3x11 layout                                   #
# --------------------------------------------------------------------------- #
def _rot90cw(M):
    return np.transpose(M[::-1, :])


def _detect_214(prs):
    """All pairs 3x11.  block = cols0-2; cols4-6 = rot90CW(block);
    cols8-10 = rot180(block); cols3,7 separators kept.  Returns True if exact."""
    for a, b in prs:
        if a.shape != (3, 11) or b.shape != (3, 11):
            return False
        blk = a[:, 0:3]
        exp = a.copy()
        exp[:, 4:7] = _rot90cw(blk)
        exp[:, 8:11] = blk[::-1, ::-1]
        if not np.array_equal(exp, b):
            return False
    return True


def _build_214():
    nodes, inits = [], []

    def kt(name, shape, vals, dt=INT64):
        inits.append(oh.make_tensor(name, dt, shape, vals)); return name

    # A = input[:,:,:,0:4]
    kt("a_s", [1], [0]); kt("a_e", [1], [4]); kt("a_ax", [1], [3])
    nodes.append(oh.make_node("Slice", ["input", "a_s", "a_e", "a_ax"], ["A"]))
    # C = input[:,:,:,7:8]
    kt("c_s", [1], [7]); kt("c_e", [1], [8]); kt("c_ax", [1], [3])
    nodes.append(oh.make_node("Slice", ["input", "c_s", "c_e", "c_ax"], ["C"]))
    # E = input[:,:,:,11:30]
    kt("e_s", [1], [11]); kt("e_e", [1], [30]); kt("e_ax", [1], [3])
    nodes.append(oh.make_node("Slice", ["input", "e_s", "e_e", "e_ax"], ["E"]))
    # block = input[:,:,0:3,0:3]
    kt("b_s", [2], [0, 0]); kt("b_e", [2], [3, 3]); kt("b_ax", [2], [2, 3])
    nodes.append(oh.make_node("Slice", ["input", "b_s", "b_e", "b_ax"], ["block"]))
    # rot180 = reverse axes 2,3
    kt("r180_s", [2], [2, 2]); kt("r180_e", [2], [_NEG, _NEG])
    kt("r180_ax", [2], [2, 3]); kt("r180_st", [2], [-1, -1])
    nodes.append(oh.make_node("Slice", ["block", "r180_s", "r180_e", "r180_ax", "r180_st"], ["rot180"]))
    # rot90cw = transpose(flipud(block))
    kt("fu_s", [1], [2]); kt("fu_e", [1], [_NEG]); kt("fu_ax", [1], [2]); kt("fu_st", [1], [-1])
    nodes.append(oh.make_node("Slice", ["block", "fu_s", "fu_e", "fu_ax", "fu_st"], ["blk_fu"]))
    nodes.append(oh.make_node("Transpose", ["blk_fu"], ["rot90cw"], perm=[0, 1, 3, 2]))
    # B = pad rot90cw rows to 30  -> [1,10,30,3]
    nodes.append(oh.make_node("Pad", ["rot90cw"], ["B"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - 3, 0]))
    # D = pad rot180 rows to 30
    nodes.append(oh.make_node("Pad", ["rot180"], ["D"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - 3, 0]))
    nodes.append(oh.make_node("Concat", ["A", "B", "C", "D", "E"], ["output"], axis=3))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# t24: dots -> full row / column lines (per-color orientation)                #
# --------------------------------------------------------------------------- #
def _apply_lines(a, hcolors, vcolors):
    """Build expected output: each row containing an H-color is filled with it;
    each column containing a V-color is filled with it; H overwrites V; cells with
    no line stay background.  Returns (out, ok) where ok=False if two same-
    orientation colors collide (sum-ambiguous) -> rejects."""
    H, W = a.shape
    out = np.zeros_like(a)
    hcount = np.zeros((H, W), int)
    vcount = np.zeros((H, W), int)
    for h in hcolors:
        rows = np.any(a == h, axis=1)            # rows containing h
        m = np.zeros((H, W), bool); m[rows, :] = True
        hcount += m.astype(int)
        out[m] = h
    for v in vcolors:
        cols = np.any(a == v, axis=0)
        m = np.zeros((H, W), bool); m[:, cols] = True
        vcount += m.astype(int)
        # only where no horizontal line
        sel = m & (hcount == 0)
        out[sel] = v
    if hcount.max() > 1 or vcount.max() > 1:
        return out, False
    return out, True


def _detect_24(prs):
    """Try every H/V assignment of the global nonzero color set; return
    (hcolors, vcolors) if one reproduces all outputs exactly, else None."""
    colors = sorted(set(int(v) for a, _ in prs for v in np.unique(a) if v != 0))
    if not colors or len(colors) > 6:
        return None
    from itertools import product
    for bits in product((0, 1), repeat=len(colors)):
        hcolors = [c for c, b in zip(colors, bits) if b == 0]
        vcolors = [c for c, b in zip(colors, bits) if b == 1]
        good = True
        for a, b in prs:
            if a.shape != b.shape:
                return None
            exp, ok = _apply_lines(a, hcolors, vcolors)
            if not ok or not np.array_equal(exp, b):
                good = False
                break
        if good:
            # require both orientations used or at least reproduces (avoid trivial)
            return (hcolors, vcolors)
    return None


def _build_24(hcolors, vcolors):
    nodes, inits = [], []

    def kt(name, shape, vals, dt=INT64):
        inits.append(oh.make_tensor(name, dt, shape, vals)); return name

    def onehot(name, c):
        v = [0.0] * CHANNELS; v[c] = 1.0
        return kt(name, [1, CHANNELS, 1, 1], v, dt=DATA_TYPE)

    def chan(name, c):
        kt(name + "_s", [1], [c]); kt(name + "_e", [1], [c + 1]); kt(name + "_a", [1], [1])
        nodes.append(oh.make_node("Slice", ["input", name + "_s", name + "_e", name + "_a"], [name]))

    # ingrid = sum over channels
    nodes.append(oh.make_node("ReduceSum", ["input"], ["ingrid"], axes=[1], keepdims=1))
    kt("one1", [1, 1, 1, 1], [1.0], dt=DATA_TYPE)

    terms = []       # (value_tensor, channel)
    hfills = []
    for i, h in enumerate(hcolors):
        nm = f"chh{h}"; chan(nm, h)
        nodes.append(oh.make_node("ReduceMax", [nm], [f"hp{h}"], axes=[3], keepdims=1))
        nodes.append(oh.make_node("Mul", [f"hp{h}", "ingrid"], [f"hf{h}"]))
        hfills.append(f"hf{h}")
        terms.append((f"hf{h}", h))
    # Hmask = sum of hfills
    if hfills:
        cur = hfills[0]
        for j in range(1, len(hfills)):
            nodes.append(oh.make_node("Add", [cur, hfills[j]], [f"hmask{j}"])); cur = f"hmask{j}"
        hmask = cur
    else:
        hmask = None
    if hmask is not None:
        nodes.append(oh.make_node("Sub", ["one1", hmask], ["nothmask"]))

    vfinals = []
    for v in vcolors:
        nm = f"chv{v}"; chan(nm, v)
        nodes.append(oh.make_node("ReduceMax", [nm], [f"vp{v}"], axes=[2], keepdims=1))
        nodes.append(oh.make_node("Mul", [f"vp{v}", "ingrid"], [f"vf{v}"]))
        if hmask is not None:
            nodes.append(oh.make_node("Mul", [f"vf{v}", "nothmask"], [f"vff{v}"]))
            vfinals.append(f"vff{v}")
            terms.append((f"vff{v}", v))
        else:
            vfinals.append(f"vf{v}")
            terms.append((f"vf{v}", v))

    # ch0 = ingrid - hmask - sum(vfinals)
    cov_parts = ([hmask] if hmask is not None else []) + vfinals
    cur = "ingrid"
    for j, p in enumerate(cov_parts):
        nodes.append(oh.make_node("Sub", [cur, p], [f"ch0_{j}"])); cur = f"ch0_{j}"
    terms.append((cur, 0))

    # assemble: output = sum(term * onehot(channel))
    placed = []
    for k, (t, c) in enumerate(terms):
        oc = onehot(f"oh{k}_{c}", c)
        nodes.append(oh.make_node("Mul", [t, oc], [f"pl{k}"]))
        placed.append(f"pl{k}")
    cur = placed[0]
    for j in range(1, len(placed)):
        last = "output" if j == len(placed) - 1 else f"acc{j}"
        nodes.append(oh.make_node("Add", [cur, placed[j]], [last])); cur = last
    if len(placed) == 1:
        nodes.append(oh.make_node("Identity", [placed[0]], ["output"]))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    try:
        r = _detect_43(prs)
        if r is not None:
            out.append(("proj43", _build_43(*r)))
    except Exception:
        pass

    try:
        if _detect_130(prs):
            out.append(("blockmax130", _build_blockmax(zero_color=5)))
    except Exception:
        pass

    try:
        r = _detect_271(prs)
        if r is not None:
            out.append(("blockmaj271", _build_blockmajority(*r)))
    except Exception:
        pass

    try:
        if _detect_214(prs):
            out.append(("rot214", _build_214()))
    except Exception:
        pass

    try:
        r = _detect_24(prs)
        if r is not None and (r[0] or r[1]):
            out.append(("lines24", _build_24(*r)))
    except Exception:
        pass

    return out
