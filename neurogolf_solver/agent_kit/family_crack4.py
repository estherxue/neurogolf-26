"""family_crack4 — origin-anchored opset-10 solvers for a slice of unsolved tasks.

Implements three structural transforms detected from the train/test/arc-gen pairs:

  * stamp_conv  (task 15)   per-color local stamp: a seed of color T paints a fixed
                            4-neighbour pattern (orthogonal "plus" or diagonal "X")
                            of a fixed colour around it.  Expressed as ONE 3x3 Conv.
  * fractal     (task 217)  self-Kronecker: an NxN shape S at the origin maps to an
                            (N*N)x(N*N) grid where block (a,b)=S iff S[a,b] is
                            foreground.  Tile(S) * upscaled-foreground-mask.
  * countbar    (task 274)  measurement->bar: count = firstRow(fill) - firstRow(wall);
                            output is a 3x3 filled with `count` cells of `fill` in
                            boustrophedon order.

Each detector reconstructs every example exactly (numpy) before emitting, so the
family only fires where the rule is provably correct.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(ex):
    out = []
    for split in ("train", "test", "arc-gen"):
        for e in ex.get(split, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return None
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


# ===========================================================================
# Task 15 — per-colour local stamp  (single 3x3 Conv)
# ===========================================================================
_ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _stamp_reconstruct(a, mapping):
    H, W = a.shape
    out = a.copy()
    for r in range(H):
        for c in range(W):
            t = a[r, c]
            if t in mapping:
                cls, col = mapping[t]
                offs = _ORTH if cls == "orth" else _DIAG
                for dr, dc in offs:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and a[rr, cc] == 0:
                        out[rr, cc] = col
    return out


def _detect_stamp(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    # seeds preserved exactly
    for a, b in prs:
        m = a != 0
        if not np.array_equal(b[m], a[m]):
            return None
    mapping = {}
    for a, b in prs:
        H, W = a.shape
        for r in range(H):
            for c in range(W):
                if a[r, c] != 0 or b[r, c] == 0:
                    continue
                v = int(b[r, c])
                cand = set()
                for dr, dc in _ORTH:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and a[rr, cc] != 0:
                        cand.add((int(a[rr, cc]), "orth", v))
                for dr, dc in _DIAG:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and a[rr, cc] != 0:
                        cand.add((int(a[rr, cc]), "diag", v))
                if len(cand) == 1:
                    t, cls, col = next(iter(cand))
                    key = t
                    val = (cls, col)
                    if key in mapping and mapping[key] != val:
                        return None
                    mapping[key] = val
    if not mapping:
        return None
    # verify reconstruction on all pairs
    for a, b in prs:
        if not np.array_equal(_stamp_reconstruct(a, mapping), b):
            return None
    # avoid trivial (must actually add cells somewhere)
    if all(np.array_equal(a, b) for a, b in prs):
        return None
    return mapping


def _build_stamp(mapping):
    K = 3
    cen = (1, 1)
    orth_k = [(0, 1), (2, 1), (1, 0), (1, 2)]
    diag_k = [(0, 0), (0, 2), (2, 0), (2, 2)]
    W = np.zeros((CHANNELS, CHANNELS, K, K), np.float32)
    B = np.full(CHANNELS, -5.0, np.float32)
    # keep term for every channel
    for o in range(CHANNELS):
        W[o, o, cen[0], cen[1]] += 10.0
    # stamp terms
    targets = {}
    for t, (cls, s) in mapping.items():
        targets.setdefault(s, []).append((t, cls))
    for s, lst in targets.items():
        W[s, 0, cen[0], cen[1]] += 5.0  # background-of-centre gate (once per channel)
        for t, cls in lst:
            ks = orth_k if cls == "orth" else diag_k
            for (kr, kc) in ks:
                W[s, t, kr, kc] += 1.0
    # background channel: subtract any stamp trigger neighbours
    for t, (cls, s) in mapping.items():
        ks = orth_k if cls == "orth" else diag_k
        for (kr, kc) in ks:
            W[0, t, kr, kc] -= 11.0
    w = oh.make_tensor("W", DATA_TYPE, [CHANNELS, CHANNELS, K, K], W.ravel().tolist())
    bt = oh.make_tensor("B", DATA_TYPE, [CHANNELS], B.tolist())
    node = oh.make_node("Conv", ["input", "W", "B"], ["output"],
                        kernel_shape=[K, K], pads=[1, 1, 1, 1])
    return _model([node], [w, bt])


# ===========================================================================
# Task 217 — self-Kronecker fractal
# ===========================================================================
def _detect_fractal(prs):
    n = None
    for a, b in prs:
        nz = np.argwhere(a != 0)
        if len(nz) == 0:
            return None
        r0, c0 = nz.min(0)
        r1, c1 = nz.max(0)
        m = max(r1 - r0 + 1, c1 - c0 + 1)
        if n is None:
            n = m
        elif n != m:
            return None
    if n is None or n < 2 or n * n > 30:
        return None
    for a, b in prs:
        nz = np.argwhere(a != 0)
        r0, c0 = nz.min(0)
        # shape sits in one n-aligned block of an (n*n)x(n*n) grid
        if r0 % n != 0 or c0 % n != 0:
            return None
        if a.shape != (n * n, n * n) or b.shape != (n * n, n * n):
            return None
        S = a[r0:r0 + n, c0:c0 + n]
        if S.shape != (n, n):
            return None
        # only that single block is nonzero
        if (a != 0).sum() != (S != 0).sum():
            return None
        exp = np.zeros((n * n, n * n), int)
        for ai in range(n):
            for bj in range(n):
                if S[ai, bj] != 0:
                    exp[ai * n:ai * n + n, bj * n:bj * n + n] = S
        if not np.array_equal(exp, b):
            return None
    return n


def _build_fractal(n):
    nn = n * n
    nodes = []
    inits = []
    # 1) S = max over the n x n blocks of the (nn x nn) input -> [1,10,n,n].
    #    Exactly one block is non-background, so the max recovers the shape S
    #    regardless of where it sits (channel0 is junk but never used downstream).
    blocks = []
    for R in range(n):
        for C in range(n):
            nm = f"blk{R}_{C}"
            inits += [
                oh.make_tensor(f"{nm}_s", INT64, [2], [n * R, n * C]),
                oh.make_tensor(f"{nm}_e", INT64, [2], [n * R + n, n * C + n]),
                oh.make_tensor(f"{nm}_a", INT64, [2], [2, 3]),
            ]
            nodes.append(oh.make_node("Slice", ["input", f"{nm}_s", f"{nm}_e", f"{nm}_a"], [nm]))
            blocks.append(nm)
    nodes.append(oh.make_node("Max", blocks, ["S"]))
    # 2) T = Tile(S, [1,1,k,k]) then crop to 30x30
    k = (HEIGHT + n - 1) // n
    inits.append(oh.make_tensor("reps", INT64, [4], [1, 1, k, k]))
    nodes.append(oh.make_node("Tile", ["S", "reps"], ["Tfull"]))
    if k * n != HEIGHT:
        inits += [
            oh.make_tensor("t_s", INT64, [2], [0, 0]),
            oh.make_tensor("t_e", INT64, [2], [HEIGHT, WIDTH]),
            oh.make_tensor("t_a", INT64, [2], [2, 3]),
        ]
        nodes.append(oh.make_node("Slice", ["Tfull", "t_s", "t_e", "t_a"], ["T"]))
        tname = "T"
    else:
        tname = "Tfull"
    # 3) foreground F = ReduceSum over channels 1..9 of S  -> [1,1,n,n]
    inits += [
        oh.make_tensor("fc_s", INT64, [1], [1]),
        oh.make_tensor("fc_e", INT64, [1], [CHANNELS]),
        oh.make_tensor("fc_a", INT64, [1], [1]),
    ]
    nodes.append(oh.make_node("Slice", ["S", "fc_s", "fc_e", "fc_a"], ["Sfg"]))
    nodes.append(oh.make_node("ReduceSum", ["Sfg"], ["F"], axes=[1], keepdims=1))
    # 4) M = Resize(F, scale n) -> [1,1,nn,nn]
    inits.append(oh.make_tensor("scales", DATA_TYPE, [4], [1.0, 1.0, float(n), float(n)]))
    nodes.append(oh.make_node("Resize", ["F", "scales"], ["Mup"], mode="nearest"))
    # pad M to 30x30
    nodes.append(oh.make_node("Pad", ["Mup"], ["M"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - nn, WIDTH - nn]))
    # 5) Yfore = T * M
    nodes.append(oh.make_node("Mul", [tname, "M"], ["Yfore"]))
    # 6) sum_fore = ReduceSum over channels 1..9 of Yfore
    inits += [
        oh.make_tensor("yc_s", INT64, [1], [1]),
        oh.make_tensor("yc_e", INT64, [1], [CHANNELS]),
        oh.make_tensor("yc_a", INT64, [1], [1]),
    ]
    nodes.append(oh.make_node("Slice", ["Yfore", "yc_s", "yc_e", "yc_a"], ["Yfg"]))
    nodes.append(oh.make_node("ReduceSum", ["Yfg"], ["sumfore"], axes=[1], keepdims=1))
    # 7) W9 constant (1 on rows<nn, cols<nn)
    w9 = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    w9[0, 0, :nn, :nn] = 1.0
    inits.append(oh.make_tensor("W9", DATA_TYPE, [1, 1, HEIGHT, WIDTH], w9.ravel().tolist()))
    nodes.append(oh.make_node("Sub", ["W9", "sumfore"], ["y0"]))
    # 8) output = concat(y0, Yfore[:,1:10])
    inits += [
        oh.make_tensor("oc_s", INT64, [1], [1]),
        oh.make_tensor("oc_e", INT64, [1], [CHANNELS]),
        oh.make_tensor("oc_a", INT64, [1], [1]),
    ]
    nodes.append(oh.make_node("Slice", ["Yfore", "oc_s", "oc_e", "oc_a"], ["Yrest"]))
    nodes.append(oh.make_node("Concat", ["y0", "Yrest"], ["output"], axis=1))
    return _model(nodes, inits)


# ===========================================================================
# Task 274 — measurement -> boustrophedon count bar
# ===========================================================================
_BOUSTRO = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]


def _first_row_with(a, color):
    rows = np.where((a == color).any(axis=1))[0]
    return int(rows[0]) if len(rows) else None


def _detect_countbar(prs):
    if not all(b.shape == (3, 3) for a, b in prs):
        return None
    # fill colour = single nonzero colour appearing in outputs
    fillset = set()
    for a, b in prs:
        fillset |= set(int(v) for v in np.unique(b) if v != 0)
    if len(fillset) != 1:
        return None
    fill = next(iter(fillset))
    # wall colour: the other input colour (must be exactly two input colours overall)
    incolset = set()
    for a, b in prs:
        incolset |= set(int(v) for v in np.unique(a) if v != 0)
    walls = incolset - {fill}
    if len(walls) != 1:
        return None
    wall = next(iter(walls))
    # verify count rule + boustrophedon layout
    for a, b in prs:
        rf = _first_row_with(a, fill)
        rw = _first_row_with(a, wall)
        if rf is None or rw is None:
            return None
        count = rf - rw
        if count < 0 or count > 9:
            return None
        exp = np.zeros((3, 3), int)
        for k in range(count):
            exp[_BOUSTRO[k]] = fill
        if not np.array_equal(exp, b):
            return None
    return wall, fill


def _build_countbar(wall, fill):
    nodes = []
    inits = []

    def row_first(channel, name):
        # slice channel
        inits.append(oh.make_tensor(f"{name}_cs", INT64, [1], [channel]))
        inits.append(oh.make_tensor(f"{name}_ce", INT64, [1], [channel + 1]))
        inits.append(oh.make_tensor(f"{name}_ca", INT64, [1], [1]))
        nodes.append(oh.make_node("Slice", ["input", f"{name}_cs", f"{name}_ce", f"{name}_ca"], [f"{name}_ch"]))
        # rowmax over width axis -> [1,1,30,1]
        nodes.append(oh.make_node("ReduceMax", [f"{name}_ch"], [f"{name}_rm"], axes=[3], keepdims=1))
        # argmax over rows axis=2 -> [1,1,1,1] int64
        nodes.append(oh.make_node("ArgMax", [f"{name}_rm"], [f"{name}_amax"], axis=2, keepdims=1))
        return f"{name}_amax"

    f8 = row_first(fill, "fl")
    f5 = row_first(wall, "wl")
    nodes.append(oh.make_node("Sub", [f8, f5], ["count_i"]))
    nodes.append(oh.make_node("Cast", ["count_i"], ["count_f"], to=DATA_TYPE))
    # Kgrid constant: boustrophedon index on 3x3, BIG elsewhere
    kg = np.full((1, 1, HEIGHT, WIDTH), 100.0, np.float32)
    for k, (r, c) in enumerate(_BOUSTRO):
        kg[0, 0, r, c] = float(k)
    inits.append(oh.make_tensor("Kg", DATA_TYPE, [1, 1, HEIGHT, WIDTH], kg.ravel().tolist()))
    # mask = count_f > Kg
    nodes.append(oh.make_node("Greater", ["count_f", "Kg"], ["mask_b"]))
    nodes.append(oh.make_node("Cast", ["mask_b"], ["mask"], to=DATA_TYPE))
    # W3 = (Kg < 50)
    inits.append(oh.make_tensor("fifty", DATA_TYPE, [], [50.0]))
    nodes.append(oh.make_node("Less", ["Kg", "fifty"], ["w3_b"]))
    nodes.append(oh.make_node("Cast", ["w3_b"], ["w3"], to=DATA_TYPE))
    # ch0 = w3 - mask
    nodes.append(oh.make_node("Sub", ["w3", "mask"], ["ch0"]))
    # place ch0 at channel0 -> pad end channels by 9
    nodes.append(oh.make_node("Pad", ["ch0"], ["p0"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, CHANNELS - 1, 0, 0]))
    # place mask at channel `fill` -> pad begin by fill, end by (CHANNELS-1-fill)
    nodes.append(oh.make_node("Pad", ["mask"], ["pf"], mode="constant", value=0.0,
                              pads=[0, fill, 0, 0, 0, CHANNELS - 1 - fill, 0, 0]))
    nodes.append(oh.make_node("Add", ["p0", "pf"], ["output"]))
    return _model(nodes, inits)


# ===========================================================================
# Task 162 — fill 3x3 background "holes" in a noisy field (greedy top-left NMS)
# ===========================================================================
def _holefill_reconstruct(a, fill):
    H, W = a.shape
    Z = np.zeros((H, W), bool)  # top-left index of an all-background 3x3
    for r in range(H - 2):
        for c in range(W - 2):
            if np.all(a[r:r + 3, c:c + 3] == 0):
                Z[r, c] = True
    befores = [(dr, dc) for dr in (-2, -1) for dc in (-2, -1, 0, 1, 2)] + [(0, -2), (0, -1)]
    supp = np.zeros((H, W), bool)
    for r in range(H - 2):
        for c in range(W - 2):
            if not Z[r, c]:
                continue
            for dr, dc in befores:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < H and 0 <= c2 < W and Z[r2, c2]:
                    supp[r, c] = True
                    break
    out = a.copy()
    for r in range(H - 2):
        for c in range(W - 2):
            if Z[r, c] and not supp[r, c]:
                out[r:r + 3, c:c + 3] = fill
    return out


def _detect_holefill(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    fill = None
    changed = False
    for a, b in prs:
        diff = a != b
        if diff.any():
            changed = True
            if (a[diff] != 0).any():
                return None  # only background cells may change
            cols = set(int(v) for v in np.unique(b[diff]))
            if len(cols) != 1:
                return None
            fc = cols.pop()
            if fill is None:
                fill = fc
            elif fill != fc:
                return None
    if not changed or fill is None:
        return None
    for a, b in prs:
        if not np.array_equal(_holefill_reconstruct(a, fill), b):
            return None
    return fill


def _build_holefill(fill):
    nodes = []
    inits = []
    ones3 = np.ones((1, 1, 3, 3), np.float32)
    inits.append(oh.make_tensor("ones3", DATA_TYPE, [1, 1, 3, 3], ones3.ravel().tolist()))
    inits.append(oh.make_tensor("thr8", DATA_TYPE, [], [8.5]))
    inits.append(oh.make_tensor("thr_half", DATA_TYPE, [], [0.5]))
    # before-overlap 5x5 kernel
    bk = np.zeros((1, 1, 5, 5), np.float32)
    bk[0, 0, 0, :] = 1.0
    bk[0, 0, 1, :] = 1.0
    bk[0, 0, 2, 0] = 1.0
    bk[0, 0, 2, 1] = 1.0
    inits.append(oh.make_tensor("bk", DATA_TYPE, [1, 1, 5, 5], bk.ravel().tolist()))
    # slice channel0
    inits += [
        oh.make_tensor("c0_s", INT64, [1], [0]),
        oh.make_tensor("c0_e", INT64, [1], [1]),
        oh.make_tensor("c0_a", INT64, [1], [1]),
    ]
    nodes.append(oh.make_node("Slice", ["input", "c0_s", "c0_e", "c0_a"], ["ch0"]))
    # cnt0 = conv3x3(ch0); Z = cnt0 > 8.5  (centered window all background)
    nodes.append(oh.make_node("Conv", ["ch0", "ones3"], ["cnt0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(oh.make_node("Greater", ["cnt0", "thr8"], ["Zb"]))
    nodes.append(oh.make_node("Cast", ["Zb"], ["Z"], to=DATA_TYPE))
    # suppressed = conv(Z, before-kernel) > 0.5
    nodes.append(oh.make_node("Conv", ["Z", "bk"], ["sc"], kernel_shape=[5, 5], pads=[2, 2, 2, 2]))
    nodes.append(oh.make_node("Greater", ["sc", "thr_half"], ["suppb"]))
    nodes.append(oh.make_node("Cast", ["suppb"], ["supp"], to=DATA_TYPE))
    # chosen = Z * (1 - supp) = relu(Z - supp)
    nodes.append(oh.make_node("Sub", ["Z", "supp"], ["chs"]))
    nodes.append(oh.make_node("Relu", ["chs"], ["chosen"]))
    # fill = conv3x3(chosen) > 0.5
    nodes.append(oh.make_node("Conv", ["chosen", "ones3"], ["fc"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(oh.make_node("Greater", ["fc", "thr_half"], ["fillb"]))
    nodes.append(oh.make_node("Cast", ["fillb"], ["fillm"], to=DATA_TYPE))
    # assemble output: ch0_out = ch0 - fill ; ch_fill_out = ch_fill + fill
    nodes.append(oh.make_node("Sub", ["ch0", "fillm"], ["ch0o"]))
    # build channels list
    # place fill into channel `fill`
    if fill == 0:
        # degenerate (shouldn't happen): output ch0 = ch0 (fill is bg) -> just identity-ish
        nodes.append(oh.make_node("Identity", ["input"], ["output"]))
        return _model(nodes, inits)
    # slice the fill channel
    inits += [
        oh.make_tensor("cf_s", INT64, [1], [fill]),
        oh.make_tensor("cf_e", INT64, [1], [fill + 1]),
        oh.make_tensor("cf_a", INT64, [1], [1]),
    ]
    nodes.append(oh.make_node("Slice", ["input", "cf_s", "cf_e", "cf_a"], ["chf"]))
    nodes.append(oh.make_node("Add", ["chf", "fillm"], ["chfo"]))
    # gather remaining channels and concat in order 0..9
    # channels: 0 -> ch0o, fill -> chfo, others -> input slice
    parts = []
    for c in range(CHANNELS):
        if c == 0:
            parts.append("ch0o")
        elif c == fill:
            parts.append("chfo")
        else:
            nm = f"oc{c}"
            inits += [
                oh.make_tensor(f"{nm}_s", INT64, [1], [c]),
                oh.make_tensor(f"{nm}_e", INT64, [1], [c + 1]),
                oh.make_tensor(f"{nm}_a", INT64, [1], [1]),
            ]
            nodes.append(oh.make_node("Slice", ["input", f"{nm}_s", f"{nm}_e", f"{nm}_a"], [nm]))
            parts.append(nm)
    nodes.append(oh.make_node("Concat", parts, ["output"], axis=1))
    return _model(nodes, inits)


# ===========================================================================
# entry point
# ===========================================================================
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    try:
        mp = _detect_stamp(prs)
    except Exception:
        mp = None
    if mp is not None:
        try:
            out.append(("stamp_conv", _build_stamp(mp)))
        except Exception:
            pass

    try:
        n = _detect_fractal(prs)
    except Exception:
        n = None
    if n is not None:
        try:
            out.append((f"fractal{n}", _build_fractal(n)))
        except Exception:
            pass

    try:
        cb = _detect_countbar(prs)
    except Exception:
        cb = None
    if cb is not None:
        try:
            out.append(("countbar", _build_countbar(*cb)))
        except Exception:
            pass

    try:
        fc = _detect_holefill(prs)
    except Exception:
        fc = None
    if fc is not None:
        try:
            out.append(("holefill", _build_holefill(fc)))
        except Exception:
            pass

    return out
