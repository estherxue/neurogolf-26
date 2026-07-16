"""family_crack0 — a small set of exact, origin-anchored opset-10 solvers.

Two independent rules, each detected structurally from the train+test pairs and
emitted as a static ONNX graph (FLOAT[1,10,30,30] one-hot in/out):

1. ``cornerfold`` — constant input size (Hi,Wi) folds onto a smaller constant
   output (Ho,Wo) by OR-overlaying the four corner blocks of size Ho x Wo at
   offsets {0,Hi-Ho} x {0,Wi-Wo}.  One-hot OR = Max on colour channels 1..9 and
   product (AND of backgrounds) on channel 0.  (task 296: 5x7 -> 3x3.)

2. ``crossproj`` — constant square-ish grid with a few isolated colour markers.
   Each marker projects to a full row+column line of its colour; where two lines
   of *different* colours cross, the cell takes a fixed colour K (the "crossing"
   colour).  Built with ReduceMax row/col projection + a count of distinct line
   colours.  (task 47: 9x9, K=2.)

Detection verifies EXACT equality on every train+test pair (the harness then
re-checks arc-gen), so over-proposing is safe — wrong guesses are rejected.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, [1, CHANNELS, HEIGHT, WIDTH])
    y = oh.make_tensor_value_info("output", DATA_TYPE, [1, CHANNELS, HEIGHT, WIDTH])
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=10,
                         opset_imports=[oh.make_opsetid("", 10)])


def _slice(inp, out, starts, ends, axes, tag, steps=None):
    n = len(axes)
    s = oh.make_tensor(f"{tag}_s", INT64, [n], list(starts))
    e = oh.make_tensor(f"{tag}_e", INT64, [n], list(ends))
    a = oh.make_tensor(f"{tag}_a", INT64, [n], list(axes))
    inits = [s, e, a]
    names = [inp, f"{tag}_s", f"{tag}_e", f"{tag}_a"]
    if steps is not None:
        st = oh.make_tensor(f"{tag}_st", INT64, [n], list(steps))
        inits.append(st)
        names.append(f"{tag}_st")
    return oh.make_node("Slice", names, [out]), inits


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX graphs exactly)
# --------------------------------------------------------------------------- #

def _cornerfold_ref(a, Ho, Wo):
    Hi, Wi = a.shape
    roffs = sorted({0, Hi - Ho})
    coffs = sorted({0, Wi - Wo})
    blocks = [a[r:r + Ho, c:c + Wo] for r in roffs for c in coffs]
    out = np.zeros((Ho, Wo), int)
    for i in range(Ho):
        for j in range(Wo):
            nz = [int(b[i, j]) for b in blocks if b[i, j] != 0]
            if len(set(nz)) > 1:
                return None  # colour conflict -> not a clean fold
            out[i, j] = nz[0] if nz else 0
    return out


def _crossproj_ref(a, K):
    H, W = a.shape
    L = np.zeros((10, H, W), bool)
    for c in range(1, 10):
        rows = np.any(a == c, axis=1)
        cols = np.any(a == c, axis=0)
        L[c] = rows[:, None] | cols[None, :]
    count = L[1:].sum(axis=0)
    out = np.zeros((H, W), int)
    for c in range(1, 10):
        out[(count == 1) & L[c]] = c
    out[count >= 2] = K
    return out


def _boxcross_centers(a, F):
    """Centres of 5x5 rings of colour F (the 16 border cells all == F)."""
    H, W = a.shape
    center = np.zeros((H, W), bool)
    for i in range(2, H - 2):
        for j in range(2, W - 2):
            if all(a[i + di, j + dj] == F
                   for di in range(-2, 3) for dj in range(-2, 3)
                   if abs(di) == 2 or abs(dj) == 2):
                center[i, j] = True
    return center


def _boxcross_ref(a, F, C):
    center = _boxcross_centers(a, F)
    if not center.any():
        return None
    rows = np.any(center, axis=1)
    cols = np.any(center, axis=0)
    cross = rows[:, None] | cols[None, :]
    out = a.copy()
    out[cross & (a != F)] = C
    return out


# --------------------------------------------------------------------------- #
# ONNX builders
# --------------------------------------------------------------------------- #

def build_cornerfold(Hi, Wi, Ho, Wo):
    roffs = sorted({0, Hi - Ho})
    coffs = sorted({0, Wi - Wo})
    offs = [(r, c) for r in roffs for c in coffs]
    nodes, inits = [], []
    block_names = []
    for k, (r, c) in enumerate(offs):
        bn = f"blk{k}"
        nd, ci = _slice("input", bn, [r, c], [r + Ho, c + Wo], [2, 3], f"blk{k}")
        nodes.append(nd); inits += ci
        block_names.append(bn)

    # colours 1..9: Max over blocks
    if len(block_names) == 1:
        cmax = block_names[0]
    else:
        cmax = "cmax"
        nodes.append(oh.make_node("Max", block_names, [cmax]))
    nd, ci = _slice(cmax, "col", [1], [CHANNELS], [1], "col"); nodes.append(nd); inits += ci

    # channel 0: product (AND) of backgrounds across blocks
    bg_names = []
    for k, bn in enumerate(block_names):
        nd, ci = _slice(bn, f"bg{k}", [0], [1], [1], f"bg{k}")
        nodes.append(nd); inits += ci
        bg_names.append(f"bg{k}")
    if len(bg_names) == 1:
        bg = bg_names[0]
    else:
        prev = bg_names[0]
        for j in range(1, len(bg_names)):
            out = "bg" if j == len(bg_names) - 1 else f"bgmul{j}"
            nodes.append(oh.make_node("Mul", [prev, bg_names[j]], [out]))
            prev = out
        bg = "bg"

    nodes.append(oh.make_node("Concat", [bg, "col"], ["folded"], axis=1))
    nodes.append(oh.make_node("Pad", ["folded"], ["output"], mode="constant",
                              value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - Ho, WIDTH - Wo]))
    return _model(nodes, inits)


def build_crossproj(H, W, K):
    nodes, inits = [], []
    # restrict to HxW to keep content origin-anchored and intermediates small
    nd, ci = _slice("input", "x", [0, 0], [H, W], [2, 3], "x"); nodes.append(nd); inits += ci
    nd, ci = _slice("x", "colin", [1], [CHANNELS], [1], "colin"); nodes.append(nd); inits += ci

    nodes.append(oh.make_node("ReduceMax", ["colin"], ["rowproj"], axes=[3], keepdims=1))
    nodes.append(oh.make_node("ReduceMax", ["colin"], ["colproj"], axes=[2], keepdims=1))
    nodes.append(oh.make_node("Max", ["rowproj", "colproj"], ["L"]))
    nodes.append(oh.make_node("ReduceSum", ["L"], ["count"], axes=[1], keepdims=1))

    thr2 = oh.make_tensor("thr2", DATA_TYPE, [1, 1, 1, 1], [1.5])
    thr0 = oh.make_tensor("thr0", DATA_TYPE, [1, 1, 1, 1], [0.5])
    one = oh.make_tensor("one", DATA_TYPE, [1, 1, 1, 1], [1.0])
    inits += [thr2, thr0, one]

    nodes.append(oh.make_node("Greater", ["count", "thr2"], ["cross_b"]))
    nodes.append(oh.make_node("Cast", ["cross_b"], ["cross"], to=DATA_TYPE))
    nodes.append(oh.make_node("Greater", ["count", "thr0"], ["pres_b"]))
    nodes.append(oh.make_node("Cast", ["pres_b"], ["present"], to=DATA_TYPE))
    nodes.append(oh.make_node("Sub", ["one", "present"], ["bg"]))
    nodes.append(oh.make_node("Sub", ["one", "cross"], ["notcross"]))
    nodes.append(oh.make_node("Mul", ["L", "notcross"], ["colorpart"]))

    # place the crossing colour K (index K-1 in the 9-channel colour block)
    nodes.append(oh.make_node("Pad", ["cross"], ["crossadd"], mode="constant",
                              value=0.0,
                              pads=[0, K - 1, 0, 0, 0, (CHANNELS - 1) - K, 0, 0]))
    nodes.append(oh.make_node("Max", ["colorpart", "crossadd"], ["colors9"]))
    nodes.append(oh.make_node("Concat", ["bg", "colors9"], ["out9"], axis=1))
    nodes.append(oh.make_node("Pad", ["out9"], ["output"], mode="constant",
                              value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]))
    return _model(nodes, inits)


def build_boxcross(H, W, F, C):
    """Draw a full row+col cross of colour C through the centre of every 5x5
    ring of frame-colour F; frame cells (colour F) on the cross stay F, every
    other cell on the cross becomes C, off-cross cells are unchanged."""
    nodes, inits = [], []
    nd, ci = _slice("input", "x", [0, 0], [H, W], [2, 3], "x"); nodes.append(nd); inits += ci
    nd, ci = _slice("x", "Fm", [F], [F + 1], [1], "Fm"); nodes.append(nd); inits += ci

    # 5x5 border-ring detector
    ring = [1.0 if (abs(di) == 2 or abs(dj) == 2) else 0.0
            for di in range(-2, 3) for dj in range(-2, 3)]
    Kr = oh.make_tensor("Kring", DATA_TYPE, [1, 1, 5, 5], ring)
    inits.append(Kr)
    nodes.append(oh.make_node("Conv", ["Fm", "Kring"], ["ringsum"],
                              kernel_shape=[5, 5], pads=[2, 2, 2, 2]))
    thr = oh.make_tensor("thr", DATA_TYPE, [1, 1, 1, 1], [15.5])
    one = oh.make_tensor("one", DATA_TYPE, [1, 1, 1, 1], [1.0])
    inits += [thr, one]
    nodes.append(oh.make_node("Greater", ["ringsum", "thr"], ["center_b"]))
    nodes.append(oh.make_node("Cast", ["center_b"], ["center"], to=DATA_TYPE))

    nodes.append(oh.make_node("ReduceMax", ["center"], ["rowhas"], axes=[3], keepdims=1))
    nodes.append(oh.make_node("ReduceMax", ["center"], ["colhas"], axes=[2], keepdims=1))
    nodes.append(oh.make_node("Max", ["rowhas", "colhas"], ["cross"]))

    nodes.append(oh.make_node("Sub", ["one", "Fm"], ["notF"]))
    nodes.append(oh.make_node("Mul", ["cross", "notF"], ["put"]))
    nodes.append(oh.make_node("Sub", ["one", "put"], ["keep"]))
    nodes.append(oh.make_node("Mul", ["x", "keep"], ["kept"]))
    nodes.append(oh.make_node("Pad", ["put"], ["addC"], mode="constant", value=0.0,
                              pads=[0, C, 0, 0, 0, (CHANNELS - 1) - C, 0, 0]))
    nodes.append(oh.make_node("Add", ["kept", "addC"], ["out9"]))
    nodes.append(oh.make_node("Pad", ["out9"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]))
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# detection / candidate generation
# --------------------------------------------------------------------------- #

def _pairs(ex):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in ex.get("train", []) + ex.get("test", [])]


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    inshapes = {a.shape for a, _ in prs}
    outshapes = {b.shape for _, b in prs}

    # ---- corner-fold: constant in/out shapes, real folding -------------- #
    if len(inshapes) == 1 and len(outshapes) == 1:
        (Hi, Wi) = next(iter(inshapes))
        (Ho, Wo) = next(iter(outshapes))
        if (Hi >= Ho and Wi >= Wo and (Hi > Ho or Wi > Wo)
                and Ho <= HEIGHT and Wo <= WIDTH and Hi <= HEIGHT and Wi <= WIDTH):
            ok = True
            for a, b in prs:
                ref = _cornerfold_ref(a, Ho, Wo)
                if ref is None or not np.array_equal(ref, b):
                    ok = False
                    break
            if ok:
                out.append((f"cornerfold_{Hi}x{Wi}_{Ho}x{Wo}",
                            build_cornerfold(Hi, Wi, Ho, Wo)))

    # ---- cross-projection: same-shape, lines + crossing colour K -------- #
    if (len(inshapes) == 1 and inshapes == outshapes):
        (H, W) = next(iter(inshapes))
        if H <= HEIGHT and W <= WIDTH and any(not np.array_equal(a, b) for a, b in prs):
            for K in range(1, 10):
                good = all(np.array_equal(_crossproj_ref(a, K), b) for a, b in prs)
                if good:
                    out.append((f"crossproj_{H}x{W}_K{K}", build_crossproj(H, W, K)))
                    break

    # ---- box-centre cross: 5x5 frame rings, cross of colour C ----------- #
    if (len(inshapes) == 1 and inshapes == outshapes):
        (H, W) = next(iter(inshapes))
        if (5 <= H <= HEIGHT and 5 <= W <= WIDTH
                and any(not np.array_equal(a, b) for a, b in prs)):
            done = False
            for F in range(1, 10):
                if done:
                    break
                for C in range(1, 10):
                    if C == F:
                        continue
                    ok = True
                    for a, b in prs:
                        ref = _boxcross_ref(a, F, C)
                        if ref is None or not np.array_equal(ref, b):
                            ok = False
                            break
                    if ok:
                        out.append((f"boxcross_{H}x{W}_F{F}_C{C}",
                                    build_boxcross(H, W, F, C)))
                        done = True
                        break

    return out
