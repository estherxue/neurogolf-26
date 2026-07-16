"""Golf slice 5: cheaper exact solvers for selected NeuroGolf targets.

Each candidate fires ONLY when the rule is reproduced EXACTLY on every available
train/test pair, and is built from a minimal opset-10 graph (few/small
intermediates, few params) so the integrator's auto-pick chooses it over the
existing (more expensive) family.
"""
from __future__ import annotations

import onnx
from onnx import helper as oh
import numpy as np

from builders import _model
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, HEIGHT, WIDTH, CHANNELS

INT64 = onnx.TensorProto.INT64


def _pairs(ex):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in ex.get("train", []) + ex.get("test", [])]


# ---------------------------------------------------------------------------
# task 135: output = top-right out_h x out_w block of the input (W fixed).
# Slice rows[0:out_h], cols[W-out_w:W] -> small, Pad back to 30x30 top-left.
# ---------------------------------------------------------------------------

def _fixed_window_crop(r0, r1, c0, c1):
    s = oh.make_tensor("c_starts", INT64, [2], [r0, c0])
    e = oh.make_tensor("c_ends", INT64, [2], [r1, c1])
    a = oh.make_tensor("c_axes", INT64, [2], [2, 3])
    sl = oh.make_node("Slice", ["input", "c_starts", "c_ends", "c_axes"], ["small"])
    oh_ = r1 - r0
    ow_ = c1 - c0
    pad = oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                       pads=[0, 0, 0, 0, 0, 0, HEIGHT - oh_, WIDTH - ow_])
    return _model([sl, pad], [s, e, a])


def _detect_topright_crop(prs):
    """output == input[0:oh, W-ow:W], with oh,ow,W constant across all pairs."""
    ws = {a.shape[1] for a, b in prs}
    if len(ws) != 1:
        return None
    W = ws.pop()
    ohs = {b.shape[0] for a, b in prs}
    ows = {b.shape[1] for a, b in prs}
    if len(ohs) != 1 or len(ows) != 1:
        return None
    oh_ = ohs.pop(); ow_ = ows.pop()
    if ow_ > W or oh_ > HEIGHT:
        return None
    c0 = W - ow_
    for a, b in prs:
        if a.shape[0] < oh_:
            return None
        if not np.array_equal(a[0:oh_, c0:W], b):
            return None
    return (0, oh_, c0, W)


# ---------------------------------------------------------------------------
# ray_down (task 322): forward-fill each column downward.  Works on the fixed
# HxW region (all cells real), so cumsum-down (MatMul lower-tri) is exact when
# every column has at most one foreground cell.
# ---------------------------------------------------------------------------

def _filldown_model(H, W):
    # lower-triangular [1,1,H,H] (L[i,k]=1 for k<=i) -> cumsum over rows
    Lv = [1.0 if k <= i else 0.0 for i in range(H) for k in range(H)]
    L = oh.make_tensor("L", DATA_TYPE, [1, 1, H, H], Lv)
    chmask = oh.make_tensor("chm", DATA_TYPE, [1, CHANNELS, 1, 1],
                            [0.0] + [1.0] * (CHANNELS - 1))
    one = oh.make_tensor("one", DATA_TYPE, [1], [1.0])
    s = oh.make_tensor("s0", INT64, [2], [0, 0])
    e = oh.make_tensor("e0", INT64, [2], [H, W])
    a = oh.make_tensor("a0", INT64, [2], [2, 3])
    nodes = [
        oh.make_node("Slice", ["input", "s0", "e0", "a0"], ["xs"]),
        oh.make_node("Mul", ["xs", "chm"], ["xsfg"]),
        oh.make_node("MatMul", ["L", "xsfg"], ["cs"]),
        oh.make_node("ReduceSum", ["cs"], ["fgsum"], axes=[1], keepdims=1),
        oh.make_node("Sub", ["one", "fgsum"], ["bg0"]),
        oh.make_node("Pad", ["bg0"], ["padbg"], mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, CHANNELS - 1, 0, 0]),
        oh.make_node("Add", ["cs", "padbg"], ["small"]),
        oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]),
    ]
    return _model(nodes, [L, chmask, one, s, e, a])


def _ffill_down(a):
    out = a.copy()
    H, W = a.shape
    for c in range(W):
        last = 0
        for r in range(H):
            if a[r, c] != 0:
                last = a[r, c]
            out[r, c] = last
    return out


def _detect_filldown(prs):
    hs = {a.shape[0] for a, b in prs}
    ws = {a.shape[1] for a, b in prs}
    if len(hs) != 1 or len(ws) != 1:
        return None
    H = hs.pop(); W = ws.pop()
    if H > HEIGHT or W > WIDTH:
        return None
    for a, b in prs:
        if a.shape != b.shape:
            return None
        # cumsum-down is exact only with <=1 fg per column
        if ((a != 0).sum(axis=0) > 1).any():
            return None
        if not np.array_equal(_ffill_down(a), b):
            return None
    return (H, W)


# ---------------------------------------------------------------------------
# task 60 (linefill): each row with end-markers A=col0, B=col(W-1) becomes
#   [A]*mid + [5] + [B]*(W-mid-1),  mid = W//2.  Fixed HxW.
# ---------------------------------------------------------------------------

def _linefill_model(H, W):
    mid = W // 2
    rightw = W - mid - 1
    sa = oh.make_tensor("sa", INT64, [2], [0, 0])
    ea = oh.make_tensor("ea", INT64, [2], [H, 1])
    sb = oh.make_tensor("sb", INT64, [2], [0, W - 1])
    eb = oh.make_tensor("eb", INT64, [2], [H, W])
    ax = oh.make_tensor("axc", INT64, [2], [2, 3])
    repL = oh.make_tensor("repL", INT64, [4], [1, 1, 1, mid])
    repR = oh.make_tensor("repR", INT64, [4], [1, 1, 1, rightw])
    s0 = oh.make_tensor("s0c", INT64, [1], [0])
    e0 = oh.make_tensor("e0c", INT64, [1], [1])
    ax1 = oh.make_tensor("ax1", INT64, [1], [1])
    d5 = oh.make_tensor("d5", DATA_TYPE, [1, CHANNELS, 1, 1],
                        [-1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0])
    v0 = oh.make_tensor("v0", DATA_TYPE, [1, CHANNELS, 1, 1],
                        [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    one = oh.make_tensor("one1", DATA_TYPE, [1], [1.0])
    nodes = [
        oh.make_node("Slice", ["input", "sa", "ea", "axc"], ["A"]),
        oh.make_node("Slice", ["input", "sb", "eb", "axc"], ["B"]),
        oh.make_node("Tile", ["A", "repL"], ["left"]),
        oh.make_node("Tile", ["B", "repR"], ["right"]),
        oh.make_node("Slice", ["A", "s0c", "e0c", "ax1"], ["a0"]),
        oh.make_node("Sub", ["one1", "a0"], ["act"]),
        oh.make_node("Mul", ["act", "d5"], ["cmul"]),
        oh.make_node("Add", ["cmul", "v0"], ["center"]),
        oh.make_node("Concat", ["left", "center", "right"], ["small"], axis=3),
        oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]),
    ]
    inits = [sa, ea, sb, eb, ax, repL, repR, s0, e0, ax1, d5, v0, one]
    return _model(nodes, inits)


def _detect_linefill(prs):
    hs = {a.shape[0] for a, b in prs}
    ws = {a.shape[1] for a, b in prs}
    if len(hs) != 1 or len(ws) != 1:
        return None
    H = hs.pop(); W = ws.pop()
    if H > HEIGHT or W > WIDTH or W < 3 or W % 2 == 0:
        return None
    mid = W // 2
    for a, b in prs:
        if a.shape != b.shape:
            return None
        if (a[:, 1:W - 1] != 0).any():
            return None
        pred = np.zeros_like(a)
        for r in range(H):
            A = a[r, 0]; Bv = a[r, W - 1]
            if A != 0 or Bv != 0:
                pred[r, 0:mid] = A
                pred[r, mid] = 5
                pred[r, mid + 1:W] = Bv
        if not np.array_equal(pred, b):
            return None
    return (H, W)


# ---------------------------------------------------------------------------
# task 350 (connect dots): cell becomes 8 when it has a `1` both above & below
# (same column) OR both left & right (same row).  Single fg colour 1, mark 8.
# Realised with two strict-triangular [30,30] matrices via MatMul (cumulative
# counts), all intermediates [1,1,30,30].
# ---------------------------------------------------------------------------

def _connectdots_model():
    Lv = [1.0 if k < r else 0.0 for r in range(HEIGHT) for k in range(HEIGHT)]
    Uv = [1.0 if k > r else 0.0 for r in range(HEIGHT) for k in range(HEIGHT)]
    L = oh.make_tensor("Ls", DATA_TYPE, [1, 1, HEIGHT, HEIGHT], Lv)
    U = oh.make_tensor("Us", DATA_TYPE, [1, 1, HEIGHT, HEIGHT], Uv)
    s1 = oh.make_tensor("s1", INT64, [1], [1])
    e1 = oh.make_tensor("e1", INT64, [1], [2])
    a1 = oh.make_tensor("a1", INT64, [1], [1])
    zero = oh.make_tensor("zc", DATA_TYPE, [1], [0.0])
    onec = oh.make_tensor("onec", DATA_TYPE, [1], [1.0])
    nodes = [
        oh.make_node("Slice", ["input", "s1", "e1", "a1"], ["ones"]),
        oh.make_node("ReduceSum", ["input"], ["M"], axes=[1], keepdims=1),
        oh.make_node("MatMul", ["Ls", "ones"], ["aboveC"]),
        oh.make_node("MatMul", ["Us", "ones"], ["belowC"]),
        oh.make_node("MatMul", ["ones", "Us"], ["leftC"]),
        oh.make_node("MatMul", ["ones", "Ls"], ["rightC"]),
        oh.make_node("Mul", ["aboveC", "belowC"], ["vsc"]),
        oh.make_node("Mul", ["leftC", "rightC"], ["hsc"]),
        oh.make_node("Add", ["vsc", "hsc"], ["betw"]),
        oh.make_node("Clip", ["betw"], ["bpos"], min=0.0, max=1.0),
        oh.make_node("Sub", ["onec", "ones"], ["notone"]),
        oh.make_node("Mul", ["bpos", "notone"], ["eight"]),
        oh.make_node("Sub", ["M", "ones"], ["m1"]),
        oh.make_node("Sub", ["m1", "eight"], ["ch0"]),
        oh.make_node("Mul", ["ones", "zc"], ["z"]),
        oh.make_node("Concat",
                     ["ch0", "ones", "z", "z", "z", "z", "z", "z", "eight", "z"],
                     ["output"], axis=1),
    ]
    inits = [L, U, s1, e1, a1, zero, onec]
    return _model(nodes, inits)


def _between_pred(a):
    H, W = a.shape
    ones = (a == 1)
    eight = np.zeros((H, W), bool)
    for c in range(W):
        rows = [r for r in range(H) if ones[r, c]]
        if len(rows) >= 2:
            for r in range(rows[0] + 1, rows[-1]):
                if not ones[r, c]:
                    eight[r, c] = True
    for r in range(H):
        cols = [c for c in range(W) if ones[r, c]]
        if len(cols) >= 2:
            for c in range(cols[0] + 1, cols[-1]):
                if not ones[r, c]:
                    eight[r, c] = True
    return np.where(ones, 1, np.where(eight, 8, 0))


def _detect_connectdots(prs):
    cols = set()
    for a, b in prs:
        cols |= set(np.unique(a).tolist()) | set(np.unique(b).tolist())
        if a.shape != b.shape:
            return False
        if not np.array_equal(_between_pred(a), b):
            return False
    return cols <= {0, 1, 8}


# ---------------------------------------------------------------------------
# connected-component size recolour on a fixed HxW region (top-left).
#   labels via min-index propagation (K iters, shift+Min, no big params),
#   per-cell size via an [N,N] label-equality reduction.
# Used by task 330 (colour-5 component, size==6 -> 2 else 1).
# ---------------------------------------------------------------------------

def _label_block(H, W, K, BIG, src, pre):
    """Emit min-index label propagation. Returns (nodes, inits, out_name)."""
    N = H * W
    idxmBIG = [float(r * W + c + 1 - BIG) for r in range(H) for c in range(W)]
    cI = oh.make_tensor(pre + "idx", DATA_TYPE, [1, 1, H, W], idxmBIG)
    cBIG = oh.make_tensor(pre + "BIG", DATA_TYPE, [1], [float(BIG)])
    cone = oh.make_tensor(pre + "one1", DATA_TYPE, [1], [1.0])
    # slice windows from the all-side padded label
    su = oh.make_tensor(pre + "su", INT64, [2], [0, 1]); eu = oh.make_tensor(pre + "eu", INT64, [2], [H, W + 1])
    sd = oh.make_tensor(pre + "sd", INT64, [2], [2, 1]); ed = oh.make_tensor(pre + "ed", INT64, [2], [H + 2, W + 1])
    sl = oh.make_tensor(pre + "sl", INT64, [2], [1, 0]); el = oh.make_tensor(pre + "el", INT64, [2], [H + 1, W])
    sr = oh.make_tensor(pre + "sr", INT64, [2], [1, 2]); er = oh.make_tensor(pre + "er", INT64, [2], [H + 1, W + 2])
    ax = oh.make_tensor(pre + "ax", INT64, [2], [2, 3])
    inits = [cI, cBIG, cone, su, eu, sd, ed, sl, el, sr, er, ax]
    nodes = [
        oh.make_node("Mul", [src, pre + "idx"], [pre + "lm"]),
        oh.make_node("Add", [pre + "lm", pre + "BIG"], [pre + "L0"]),
        # bigm = (1-FG)*BIG : background cells pinned at BIG so they never adopt
        # a neighbour's small label during propagation.
        oh.make_node("Sub", [pre + "one1", src], [pre + "nfg"]),
        oh.make_node("Mul", [pre + "nfg", pre + "BIG"], [pre + "bigm"]),
    ]
    cur = pre + "L0"
    for k in range(K):
        p = f"{pre}p{k}"
        nodes += [
            oh.make_node("Pad", [cur], [p + "P"], mode="constant", value=float(BIG),
                         pads=[0, 0, 1, 1, 0, 0, 1, 1]),
            oh.make_node("Slice", [p + "P", pre + "su", pre + "eu", pre + "ax"], [p + "u"]),
            oh.make_node("Slice", [p + "P", pre + "sd", pre + "ed", pre + "ax"], [p + "d"]),
            oh.make_node("Slice", [p + "P", pre + "sl", pre + "el", pre + "ax"], [p + "l"]),
            oh.make_node("Slice", [p + "P", pre + "sr", pre + "er", pre + "ax"], [p + "r"]),
            oh.make_node("Min", [cur, p + "u", p + "d", p + "l", p + "r"], [p + "m"]),
            oh.make_node("Max", [p + "m", pre + "bigm"], [p + "L"]),
        ]
        cur = p + "L"
    return nodes, inits, cur


def _size_block(H, W, label, pre):
    """Per-cell component size [1,1,H,W] from label [1,1,H,W]."""
    N = H * W
    shA = oh.make_tensor(pre + "shA", INT64, [2], [N, 1])
    shB = oh.make_tensor(pre + "shB", INT64, [2], [1, N])
    shG = oh.make_tensor(pre + "shG", INT64, [4], [1, 1, H, W])
    nh = oh.make_tensor(pre + "nh", DATA_TYPE, [1], [-0.5])
    ph = oh.make_tensor(pre + "ph", DATA_TYPE, [1], [0.5])
    nodes = [
        oh.make_node("Reshape", [label, pre + "shA"], [pre + "LA"]),
        oh.make_node("Reshape", [label, pre + "shB"], [pre + "LB"]),
        oh.make_node("Sub", [pre + "LA", pre + "LB"], [pre + "diff"]),
        oh.make_node("Greater", [pre + "diff", pre + "nh"], [pre + "gt"]),
        oh.make_node("Less", [pre + "diff", pre + "ph"], [pre + "lt"]),
        oh.make_node("And", [pre + "gt", pre + "lt"], [pre + "eqb"]),
        oh.make_node("Cast", [pre + "eqb"], [pre + "eqf"], to=DATA_TYPE),
        oh.make_node("ReduceSum", [pre + "eqf"], [pre + "sz"], axes=[1], keepdims=1),
        oh.make_node("Reshape", [pre + "sz", pre + "shG"], [pre + "szG"]),
    ]
    inits = [shA, shB, shG, nh, ph]
    return nodes, inits, pre + "szG"


def _ccsize330_model(H, W, C, K=16, BIG=99999):
    # FG = colour C in region
    sC = oh.make_tensor("sC", INT64, [3], [C, 0, 0])
    eC = oh.make_tensor("eC", INT64, [3], [C + 1, H, W])
    aC = oh.make_tensor("aC", INT64, [3], [1, 2, 3])
    inits = [sC, eC, aC]
    nodes = [oh.make_node("Slice", ["input", "sC", "eC", "aC"], ["FG"])]
    ln, li, lab = _label_block(H, W, K, BIG, "FG", "")
    nodes += ln; inits += li
    sn, si, szG = _size_block(H, W, lab, "")
    nodes += sn; inits += si
    six_lo = oh.make_tensor("slo", DATA_TYPE, [1], [5.5])
    six_hi = oh.make_tensor("shi", DATA_TYPE, [1], [6.5])
    one = oh.make_tensor("one", DATA_TYPE, [1], [1.0])
    zero = oh.make_tensor("zero", DATA_TYPE, [1], [0.0])
    inits += [six_lo, six_hi, one, zero]
    nodes += [
        oh.make_node("Greater", [szG, "slo"], ["g6"]),
        oh.make_node("Less", [szG, "shi"], ["l6"]),
        oh.make_node("And", ["g6", "l6"], ["is6b"]),
        oh.make_node("Cast", ["is6b"], ["is6f"], to=DATA_TYPE),
        oh.make_node("Mul", ["is6f", "FG"], ["ch2"]),
        oh.make_node("Sub", ["FG", "ch2"], ["ch1"]),
        oh.make_node("Sub", ["one", "FG"], ["ch0"]),
        oh.make_node("Mul", ["FG", "zero"], ["z"]),
        oh.make_node("Concat",
                     ["ch0", "ch1", "ch2", "z", "z", "z", "z", "z", "z", "z"],
                     ["small"], axis=1),
        oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]),
    ]
    return _model(nodes, inits)


def _label_np(mask):
    H, W = mask.shape
    lab = np.zeros((H, W), int); cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                cur += 1; st = [(i, j)]; lab[i, j] = cur
                while st:
                    r, c = st.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and lab[nr, nc] == 0:
                            lab[nr, nc] = cur; st.append((nr, nc))
    return lab, cur


# ---------------------------------------------------------------------------
# task 302 (enclosed-hole recolour): 0-holes enclosed by colour-5 walls are
# squares of size 1/4/9 -> colours 6/7/8.  Outer background (flood from border)
# stays 0; walls stay 5.  Hole size from a 5x5 box-sum conv (holes <=3x3 and
# isolated, validated).  No size matrix needed.
# ---------------------------------------------------------------------------

def _flood302_model(H, W, K=12):
    sZ = oh.make_tensor("sZ", INT64, [3], [0, 0, 0])
    eZ = oh.make_tensor("eZ", INT64, [3], [1, H, W])
    aZ = oh.make_tensor("aZ", INT64, [3], [1, 2, 3])
    s5 = oh.make_tensor("s5", INT64, [3], [5, 0, 0])
    e5 = oh.make_tensor("e5", INT64, [3], [6, H, W])
    border = [1.0 if (r in (0, H - 1) or c in (0, W - 1)) else 0.0
              for r in range(H) for c in range(W)]
    bm = oh.make_tensor("bm", DATA_TYPE, [1, 1, H, W], border)
    # shift slice windows
    su = oh.make_tensor("su", INT64, [2], [0, 1]); eu = oh.make_tensor("eu", INT64, [2], [H, W + 1])
    sd = oh.make_tensor("sd", INT64, [2], [2, 1]); ed = oh.make_tensor("ed", INT64, [2], [H + 2, W + 1])
    sl = oh.make_tensor("sl", INT64, [2], [1, 0]); el = oh.make_tensor("el", INT64, [2], [H + 1, W])
    sr = oh.make_tensor("sr", INT64, [2], [1, 2]); er = oh.make_tensor("er", INT64, [2], [H + 1, W + 2])
    ax = oh.make_tensor("axf", INT64, [2], [2, 3])
    box = oh.make_tensor("box", DATA_TYPE, [1, 1, 5, 5], [1.0] * 25)
    inits = [sZ, eZ, aZ, s5, e5, bm, su, eu, sd, ed, sl, el, sr, er, ax, box]
    nodes = [
        oh.make_node("Slice", ["input", "sZ", "eZ", "aZ"], ["ZG"]),
        oh.make_node("Slice", ["input", "s5", "e5", "aZ"], ["FG5"]),
        oh.make_node("Mul", ["ZG", "bm"], ["reach0"]),
    ]
    cur = "reach0"
    for k in range(K):
        p = f"f{k}"
        nodes += [
            oh.make_node("Pad", [cur], [p + "P"], mode="constant", value=0.0,
                         pads=[0, 0, 1, 1, 0, 0, 1, 1]),
            oh.make_node("Slice", [p + "P", "su", "eu", "axf"], [p + "u"]),
            oh.make_node("Slice", [p + "P", "sd", "ed", "axf"], [p + "d"]),
            oh.make_node("Slice", [p + "P", "sl", "el", "axf"], [p + "l"]),
            oh.make_node("Slice", [p + "P", "sr", "er", "axf"], [p + "r"]),
            oh.make_node("Max", [cur, p + "u", p + "d", p + "l", p + "r"], [p + "mx"]),
            oh.make_node("Mul", [p + "mx", "ZG"], [p + "R"]),
        ]
        cur = p + "R"
    reach = cur
    lo1 = oh.make_tensor("lo1", DATA_TYPE, [1], [0.5]); hi1 = oh.make_tensor("hi1", DATA_TYPE, [1], [1.5])
    lo4 = oh.make_tensor("lo4", DATA_TYPE, [1], [3.5]); hi4 = oh.make_tensor("hi4", DATA_TYPE, [1], [4.5])
    lo9 = oh.make_tensor("lo9", DATA_TYPE, [1], [8.5]); hi9 = oh.make_tensor("hi9", DATA_TYPE, [1], [9.5])
    zc = oh.make_tensor("zc302", DATA_TYPE, [1], [0.0])
    inits += [lo1, hi1, lo4, hi4, lo9, hi9, zc]
    nodes += [
        oh.make_node("Sub", ["ZG", reach], ["hole"]),
        oh.make_node("Conv", ["hole", "box"], ["bs"], kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
        oh.make_node("Greater", ["bs", "lo1"], ["g1"]),
        oh.make_node("Less", ["bs", "hi1"], ["l1"]),
        oh.make_node("And", ["g1", "l1"], ["i1b"]),
        oh.make_node("Cast", ["i1b"], ["i1"], to=DATA_TYPE),
        oh.make_node("Mul", ["i1", "hole"], ["c6"]),
        oh.make_node("Greater", ["bs", "lo4"], ["g4"]),
        oh.make_node("Less", ["bs", "hi4"], ["l4"]),
        oh.make_node("And", ["g4", "l4"], ["i4b"]),
        oh.make_node("Cast", ["i4b"], ["i4"], to=DATA_TYPE),
        oh.make_node("Mul", ["i4", "hole"], ["c7"]),
        oh.make_node("Greater", ["bs", "lo9"], ["g9"]),
        oh.make_node("Less", ["bs", "hi9"], ["l9"]),
        oh.make_node("And", ["g9", "l9"], ["i9b"]),
        oh.make_node("Cast", ["i9b"], ["i9"], to=DATA_TYPE),
        oh.make_node("Mul", ["i9", "hole"], ["c8"]),
        oh.make_node("Mul", ["ZG", "zc302"], ["z302"]),
        oh.make_node("Concat",
                     [reach, "z302", "z302", "z302", "z302", "FG5", "c6", "c7", "c8", "z302"],
                     ["small302"], axis=1),
        oh.make_node("Pad", ["small302"], ["output"], mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]),
    ]
    return _model(nodes, inits)


def _flood_np(zg):
    H, W = zg.shape
    reach = np.zeros((H, W), bool)
    st = []
    for c in range(W):
        for r in (0, H - 1):
            if zg[r, c]:
                reach[r, c] = True; st.append((r, c))
    for r in range(H):
        for c in (0, W - 1):
            if zg[r, c] and not reach[r, c]:
                reach[r, c] = True; st.append((r, c))
    while st:
        r, c = st.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and zg[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True; st.append((nr, nc))
    return reach


def _boxsum_np(m, K=5):
    H, W = m.shape; r = K // 2
    out = np.zeros((H, W), int)
    for i in range(H):
        for j in range(W):
            out[i, j] = m[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1].sum()
    return out


# ---------------------------------------------------------------------------
# task 254 (colheight): among columns of colour 5, the tallest -> 1, the
# shortest (non-zero) -> 2, all others -> 0.  Fixed HxW, per-column reductions.
# ---------------------------------------------------------------------------

def _colheight_model(H, W):
    s5 = oh.make_tensor("ch_s5", INT64, [3], [5, 0, 0])
    e5 = oh.make_tensor("ch_e5", INT64, [3], [6, H, W])
    a5 = oh.make_tensor("ch_a5", INT64, [3], [1, 2, 3])
    half = oh.make_tensor("ch_half", DATA_TYPE, [1], [0.5])
    BIG = oh.make_tensor("ch_BIG", DATA_TYPE, [1], [1000.0])
    one = oh.make_tensor("ch_one", DATA_TYPE, [1], [1.0])
    zero = oh.make_tensor("ch_zero", DATA_TYPE, [1], [0.0])
    inits = [s5, e5, a5, half, BIG, one, zero]
    nodes = [
        oh.make_node("Slice", ["input", "ch_s5", "ch_e5", "ch_a5"], ["cFG"]),
        oh.make_node("ReduceSum", ["cFG"], ["ccnt"], axes=[2], keepdims=1),
        # tallest
        oh.make_node("ReduceMax", ["ccnt"], ["cmax"], axes=[3], keepdims=1),
        oh.make_node("Sub", ["cmax", "ch_half"], ["cmaxh"]),
        oh.make_node("Greater", ["ccnt", "cmaxh"], ["tallb"]),
        oh.make_node("Cast", ["tallb"], ["tall"], to=DATA_TYPE),
        # shortest nonzero
        oh.make_node("Less", ["ccnt", "ch_half"], ["zb"]),
        oh.make_node("Cast", ["zb"], ["zf"], to=DATA_TYPE),
        oh.make_node("Mul", ["zf", "ch_BIG"], ["zbig"]),
        oh.make_node("Add", ["ccnt", "zbig"], ["nzcnt"]),
        oh.make_node("ReduceMin", ["nzcnt"], ["cmin"], axes=[3], keepdims=1),
        oh.make_node("Sub", ["cmin", "ch_half"], ["cminl"]),
        oh.make_node("Add", ["cmin", "ch_half"], ["cminh"]),
        oh.make_node("Greater", ["ccnt", "cminl"], ["sg"]),
        oh.make_node("Less", ["ccnt", "cminh"], ["sl"]),
        oh.make_node("And", ["sg", "sl"], ["shortb"]),
        oh.make_node("Cast", ["shortb"], ["short"], to=DATA_TYPE),
        # recolour
        oh.make_node("Mul", ["cFG", "tall"], ["k1"]),
        oh.make_node("Mul", ["cFG", "short"], ["k2"]),
        oh.make_node("Sub", ["ch_one", "k1"], ["t0a"]),
        oh.make_node("Sub", ["t0a", "k2"], ["k0"]),
        oh.make_node("Mul", ["cFG", "ch_zero"], ["zg254"]),
        oh.make_node("Concat",
                     ["k0", "k1", "k2", "zg254", "zg254", "zg254", "zg254",
                      "zg254", "zg254", "zg254"], ["sm254"], axis=1),
        oh.make_node("Pad", ["sm254"], ["output"], mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]),
    ]
    return _model(nodes, inits)


def _detect_colheight(prs):
    hs = {a.shape[0] for a, b in prs}; ws = {a.shape[1] for a, b in prs}
    if len(hs) != 1 or len(ws) != 1:
        return None
    H = hs.pop(); W = ws.pop()
    if H > HEIGHT or W > WIDTH:
        return None
    cols = np.arange(W)
    for a, b in prs:
        if a.shape != b.shape:
            return None
        if set(np.unique(a).tolist()) - {0, 5} or set(np.unique(b).tolist()) - {0, 1, 2, 5}:
            return None
        cnt = (a == 5).sum(0)
        if cnt.max() == 0:
            return None
        nz = cnt.astype(float); nz[nz == 0] = np.inf
        tallv = cnt.max(); minv = nz.min()
        # require unique max & unique min (== selection must be safe)
        if (cnt == tallv).sum() != 1 or (cnt == minv).sum() != 1:
            return None
        tall = (cnt == tallv); short = (cnt == minv)
        pred = np.zeros_like(a)
        pred[(a == 5) & tall[None, :]] = 1
        pred[(a == 5) & short[None, :]] = 2
        if not np.array_equal(pred, b):
            return None
    return (H, W)


def _detect_302(prs):
    hs = {a.shape[0] for a, b in prs}; ws = {a.shape[1] for a, b in prs}
    if len(hs) != 1 or len(ws) != 1:
        return None
    H = hs.pop(); W = ws.pop()
    if H > HEIGHT or W > WIDTH:
        return None
    mp = {1: 6, 4: 7, 9: 8}
    for a, b in prs:
        if a.shape != b.shape:
            return None
        if set(np.unique(a).tolist()) - {0, 5}:
            return None
        if set(np.unique(b).tolist()) - {0, 5, 6, 7, 8}:
            return None
        zg = (a == 0)
        reach = _flood_np(zg)
        hole = (zg & ~reach).astype(int)
        bs = _boxsum_np(hole, 5)
        pred = a.copy()
        ok = True
        for i in range(H):
            for j in range(W):
                if hole[i, j]:
                    if bs[i, j] not in mp:
                        ok = False; break
                    pred[i, j] = mp[bs[i, j]]
            if not ok:
                break
        if not ok or not np.array_equal(pred, b):
            return None
    return (H, W)


def _detect_ccsize6(prs):
    """colour-5 4-conn component: size==6 -> 2 else -> 1; bg stays 0; fixed HxW."""
    hs = {a.shape[0] for a, b in prs}; ws = {a.shape[1] for a, b in prs}
    if len(hs) != 1 or len(ws) != 1:
        return None
    H = hs.pop(); W = ws.pop()
    if H > HEIGHT or W > WIDTH:
        return None
    for a, b in prs:
        if a.shape != b.shape:
            return None
        if set(np.unique(a).tolist()) - {0, 5}:
            return None
        if set(np.unique(b).tolist()) - {0, 1, 2}:
            return None
        lab, n = _label_np(a == 5)
        pred = np.zeros_like(a)
        for i in range(1, n + 1):
            cells = lab == i
            pred[cells] = 2 if int(cells.sum()) == 6 else 1
        if not np.array_equal(pred, b):
            return None
    return (H, W)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    win = _detect_topright_crop(prs)
    if win is not None:
        r0, r1, c0, c1 = win
        out.append(("golf5_trcrop", _fixed_window_crop(r0, r1, c0, c1)))

    fd = _detect_filldown(prs)
    if fd is not None:
        out.append(("golf5_filldown", _filldown_model(*fd)))

    lf = _detect_linefill(prs)
    if lf is not None:
        out.append(("golf5_linefill", _linefill_model(*lf)))

    if _detect_connectdots(prs):
        out.append(("golf5_connectdots", _connectdots_model()))

    cc = _detect_ccsize6(prs)
    if cc is not None:
        out.append(("golf5_ccsize6", _ccsize330_model(cc[0], cc[1], 5, K=12)))

    h302 = _detect_302(prs)
    if h302 is not None:
        out.append(("golf5_holes302", _flood302_model(h302[0], h302[1])))

    chh = _detect_colheight(prs)
    if chh is not None:
        out.append(("golf5_colheight", _colheight_model(*chh)))

    return out
