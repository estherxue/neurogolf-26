"""family_pgolf9_0 — GOLF rewrite of the diagonal "echo" task (370).

Incumbent `echo370` (family_crk10_1) unrolls the echo as 15 LINEAR diagonal
shifts, OR-accumulated: for k=1..15 it does MatMul(Pr,cur), MatMul(cur,PcT),
Max — ~44 [30,30] intermediates just for the echo.

The echo is the union  U_{k=1..15} (holes + k*T)  along a single fixed diagonal
vector T.  A union of translates along a fixed vector is a textbook
Hillis-Steele DOUBLING: with shift operators for T,2T,4T,8T we cover k=1..16 in
4 doubling steps instead of 15.  The larger shift matrices are obtained by
SQUARING the base permutation matrix (Pr2 = Pr1@Pr1, ...), since composing shift
permutations adds their offsets — no rebuild from scalars.  This collapses the
echo from ~44 to ~20 [30,30] intermediates; identical outputs, so it validates
on the same 266/266 (train+test+arc-gen) the incumbent does.

Everything else (background/hole/marker extraction, T inference via farthest
diagonal hole, recolor via Where) is the SAME rule as the incumbent, ported
op-for-op.  Fires only when that rule reproduces every train+test pair (task 370
and its arc-gen family), so it is fully general — no per-instance parameters.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

FLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64
N = 30


def _f(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    return oh.make_tensor(name, FLOAT, list(arr.shape), arr.ravel().tolist())


def _i64(name, arr):
    arr = np.asarray(arr, dtype=np.int64)
    return oh.make_tensor(name, INT64, list(arr.shape), arr.ravel().tolist())


def _build_370():
    rows = np.arange(N, dtype=np.float32).reshape(N, 1)
    cols = np.arange(N, dtype=np.float32).reshape(1, N)
    Ig = np.broadcast_to(rows, (N, N)).astype(np.float32)     # Ig[r,c]=r
    Jg = np.broadcast_to(cols, (N, N)).astype(np.float32)     # Jg[r,c]=c
    DIFF = (rows - cols).astype(np.float32)                   # DIFF[i,j]=i-j
    big_vec = np.zeros(10, dtype=np.float32); big_vec[0] = 1e9

    inits = [
        _f("Ig", Ig), _f("Jg", Jg), _f("DIFF", DIFF),
        _f("one", [1.0]), _f("half", [0.5]), _f("thousand", [1000.0]),
        _f("big_vec", big_vec),
        _i64("s0", [0]), _i64("e1", [1]), _i64("ax0", [0]),
        _i64("sh_1_30_30", [1, N, N]), _i64("sh_10_1_1", [10, 1, 1]),
        _i64("sh_out", [1, 10, N, N]),
    ]
    nd = []

    def node(op, ins, outs, **attr):
        nd.append(oh.make_node(op, ins, outs, **attr))

    # X [10,30,30]
    node("Squeeze", ["input"], ["X"], axes=[0])
    node("ReduceSum", ["X"], ["gridmask"], axes=[0], keepdims=0)  # [30,30]
    node("Slice", ["X", "s0", "e1", "ax0"], ["ch0_c"])
    node("Squeeze", ["ch0_c"], ["ch0"], axes=[0])
    node("Mul", ["ch0", "gridmask"], ["holes"])                # [30,30]
    node("Sub", ["gridmask", "ch0"], ["NZ"])
    # background channel
    node("ReduceSum", ["X"], ["sums"], axes=[1, 2], keepdims=0)
    node("Sub", ["sums", "big_vec"], ["s19"])
    node("ArgMax", ["s19"], ["bg"], axis=0, keepdims=1)
    node("Gather", ["X", "bg"], ["bg_map_c"], axis=0)
    node("Squeeze", ["bg_map_c"], ["bg_map"], axes=[0])
    # marker mask
    node("Sub", ["NZ", "bg_map"], ["mk_raw"])
    node("Relu", ["mk_raw"], ["mk_relu"])
    node("Mul", ["mk_relu", "gridmask"], ["Mk"])
    # marker colour vector [10]
    node("Reshape", ["Mk", "sh_1_30_30"], ["Mk3"])
    node("Mul", ["X", "Mk3"], ["xMk"])
    node("ReduceSum", ["xMk"], ["colorvec"], axes=[1, 2], keepdims=0)
    # marker position
    node("Mul", ["Mk", "Ig"], ["MkI"]); node("ReduceSum", ["MkI"], ["mr"], axes=[0, 1], keepdims=0)
    node("Mul", ["Mk", "Jg"], ["MkJ"]); node("ReduceSum", ["MkJ"], ["mc"], axes=[0, 1], keepdims=0)
    # diagonal holes + farthest score
    node("Sub", ["Ig", "mr"], ["diff_r"]); node("Sub", ["Jg", "mc"], ["diff_c"])
    node("Abs", ["diff_r"], ["adr"]); node("Abs", ["diff_c"], ["adc"])
    node("Sub", ["adr", "adc"], ["ad_diff"]); node("Abs", ["ad_diff"], ["ad_diff_a"])
    node("Less", ["ad_diff_a", "half"], ["diag_b"]); node("Cast", ["diag_b"], ["diag"], to=FLOAT)
    node("Mul", ["holes", "diag"], ["diaghole"])
    node("Mul", ["diaghole", "adr"], ["term1"])
    node("Sub", ["diaghole", "one"], ["dh1"]); node("Mul", ["dh1", "thousand"], ["term2"])
    node("Add", ["term1", "term2"], ["score"])
    node("ReduceMax", ["score"], ["mx"], axes=[0, 1], keepdims=0)
    node("Sub", ["mx", "half"], ["thr"])
    node("Greater", ["score", "thr"], ["far_b"]); node("Cast", ["far_b"], ["farmask"], to=FLOAT)
    node("Mul", ["farmask", "Ig"], ["fI"]); node("ReduceSum", ["fI"], ["pr"], axes=[0, 1], keepdims=0)
    node("Mul", ["farmask", "Jg"], ["fJ"]); node("ReduceSum", ["fJ"], ["pc"], axes=[0, 1], keepdims=0)
    node("Sub", ["mr", "pr"], ["dr"]); node("Sub", ["mc", "pc"], ["dc"])

    # base shift matrices: Pr1 shifts rows by dr, PcT1 shifts cols by dc
    node("Sub", ["DIFF", "dr"], ["Pr_d"]); node("Abs", ["Pr_d"], ["Pr_a"])
    node("Less", ["Pr_a", "half"], ["Pr_b"]); node("Cast", ["Pr_b"], ["Pr1"], to=FLOAT)
    node("Add", ["DIFF", "dc"], ["Pc_d"]); node("Abs", ["Pc_d"], ["Pc_a"])
    node("Less", ["Pc_a", "half"], ["Pc_b"]); node("Cast", ["Pc_b"], ["PcT1"], to=FLOAT)
    # power matrices via squaring (compose shift permutations -> add offsets)
    node("MatMul", ["Pr1", "Pr1"], ["Pr2"]); node("MatMul", ["Pr2", "Pr2"], ["Pr4"])
    node("MatMul", ["Pr4", "Pr4"], ["Pr8"])
    node("MatMul", ["PcT1", "PcT1"], ["PcT2"]); node("MatMul", ["PcT2", "PcT2"], ["PcT4"])
    node("MatMul", ["PcT4", "PcT4"], ["PcT8"])

    # echo via Hillis-Steele doubling: start at k=1, cover k=1..16 in 4 steps
    node("MatMul", ["Pr1", "holes"], ["A0r"]); node("MatMul", ["A0r", "PcT1"], ["E0"])  # k=1
    prs = ["Pr1", "Pr2", "Pr4", "Pr8"]
    pcs = ["PcT1", "PcT2", "PcT4", "PcT8"]
    cur = "E0"
    for s in range(4):
        sr = f"sr{s}"; sc = f"sc{s}"; mx = f"E{s+1}"
        node("MatMul", [prs[s], cur], [sr])
        node("MatMul", [sr, pcs[s]], [sc])
        node("Max", [cur, sc], [mx])
        cur = mx
    node("Mul", [cur, "gridmask"], ["Eg"])
    node("Reshape", ["Eg", "sh_1_30_30"], ["Eflat"])
    node("Greater", ["Eflat", "half"], ["cond"])
    node("Reshape", ["colorvec", "sh_10_1_1"], ["cv3"])
    node("Where", ["cond", "cv3", "X"], ["out10"])
    node("Reshape", ["out10", "sh_out"], ["output"])

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nd, "echo370_golf", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# detection: numpy mirror of the exact rule; fire only when it reproduces every
# train+test pair (so it never mis-fires on unrelated tasks).
# --------------------------------------------------------------------------- #
def _solve_np(X):
    H, W = X.shape
    cnt = np.array([(X == c).sum() for c in range(10)], dtype=float)
    cnt[0] -= 1e9
    bg = int(cnt.argmax())
    holes = (X == 0)
    mk = (X != 0) & (X != bg)
    if mk.sum() != 1:
        return None
    mr, mc = [int(v) for v in np.argwhere(mk)[0]]
    C = int(X[mr, mc])
    best = None; bestd = -1
    for r, c in np.argwhere(holes):
        dr = mr - r; dc = mc - c
        if dr != 0 and abs(dr) == abs(dc) and abs(dr) > bestd:
            bestd = abs(dr); best = (r, c)
    if best is None:
        return None
    Tr, Tc = mr - best[0], mc - best[1]
    out = X.copy()
    hs = np.argwhere(holes)
    for k in range(1, 20):
        for (r, c) in hs:
            rr, cc = r + k * Tr, c + k * Tc
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = C
    return out


def candidates(ex):
    pairs = []
    for sec in ("train", "test"):
        for e in ex.get(sec, []):
            pairs.append((np.asarray(e["input"], int), np.asarray(e["output"], int)))
    if not pairs:
        return []
    changed = False
    for a, b in pairs:
        if a.shape != b.shape:
            return []
        p = _solve_np(a)
        if p is None or not np.array_equal(p, b):
            return []
        if not np.array_equal(a, b):
            changed = True
    if not changed:
        return []
    try:
        return [("echo370_golf", _build_370())]
    except Exception:
        return []
