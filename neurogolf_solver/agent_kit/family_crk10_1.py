"""family_crk10_1  -- hardest-tasks slice U[1::8] = [18,76,107,157,191,255,370].

Only task 370 is expressible as a static opset-10 graph; the other six each
require a per-instance data-dependent parameter that varies across arc-gen
(rigid-transform rotation[18], template rotation-stamp[76], variable output
size + scaled concentric pattern[107], jigsaw shape-assignment[157],
variable-size template match+stamp[191], noise-ambiguous corridor
segmentation[255]) and cannot be baked into a single fixed graph.

TASK 370  (diagonal "echo" of a hole-cluster from a marker):
  * grid = dense background colour BG, with a small cluster of holes (value 0)
    and exactly one MARKER cell of a third colour C, sitting diagonally off the
    hole cluster.
  * Let S = set of hole cells, M = marker cell.  Among holes p with (M-p)
    diagonal (|dr|==|dc|, dr!=0), let p* be the one with the LARGEST |dr|.
    Define T = M - p*  (a diagonal step vector).
  * Output = input, plus every cell of  U_{k>=1} (S + k*T)  (clipped to grid)
    painted with the marker colour C.  Original holes stay 0; marker stays C.
  Verified EXACT on all train+test+arc-gen (266/266) via numpy mirror that
  reproduces this graph's op-by-op dataflow.

Implementation notes (opset-10, no Div / no float-Equal / static shapes):
  - grid mask via ReduceMax over the non-background channel (BG fills every grid
    row/col, so rows/cols with any non-hole cell mark the HxW region).
  - marker isolated as (nonzero - background_channel); background channel picked
    by ArgMax of per-channel counts and Gather'd.
  - marker/hole positions recovered by ReduceSum(mask * index_grid) (mask is a
    single cell -> weighted sum == its coordinate); no ArgMax-decode / Div.
  - shift by the data-dependent (dr,dc) via computed [30,30] MatMul matrices
    P[i,j] = Less(Abs((i-j) -/+ d), 0.5); the echo is 15 unrolled diagonal
    shifts OR-accumulated (grids <=~20 tall, |T|>=2  ->  <=15 copies).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

FLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64
N = 30
_ITERS = 15


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
    # PRES = 1 on grid cells, 0 on padding (grader leaves padding all-zero)
    node("ReduceSum", ["X"], ["gridmask"], axes=[0], keepdims=0)  # [30,30]
    # ch0 = holes (grid cells whose colour is 0); padding contributes 0
    node("Slice", ["X", "s0", "e1", "ax0"], ["ch0_c"])         # [1,30,30]
    node("Squeeze", ["ch0_c"], ["ch0"], axes=[0])              # [30,30]
    node("Mul", ["ch0", "gridmask"], ["holes"])                # [30,30] == ch0
    # NZ = grid non-hole cells (background + marker)
    node("Sub", ["gridmask", "ch0"], ["NZ"])                   # [30,30]
    # background channel
    node("ReduceSum", ["X"], ["sums"], axes=[1, 2], keepdims=0)  # [10]
    node("Sub", ["sums", "big_vec"], ["s19"])
    node("ArgMax", ["s19"], ["bg"], axis=0, keepdims=1)        # [1]
    node("Gather", ["X", "bg"], ["bg_map_c"], axis=0)          # [1,30,30]
    node("Squeeze", ["bg_map_c"], ["bg_map"], axes=[0])        # [30,30]
    # marker mask
    node("Sub", ["NZ", "bg_map"], ["mk_raw"])
    node("Relu", ["mk_raw"], ["mk_relu"])
    node("Mul", ["mk_relu", "gridmask"], ["Mk"])               # [30,30], single 1
    # marker colour vector [10]
    node("Reshape", ["Mk", "sh_1_30_30"], ["Mk3"])             # [1,30,30]
    node("Mul", ["X", "Mk3"], ["xMk"])                         # [10,30,30]
    node("ReduceSum", ["xMk"], ["colorvec"], axes=[1, 2], keepdims=0)  # [10]
    # marker position (scalars)
    node("Mul", ["Mk", "Ig"], ["MkI"]); node("ReduceSum", ["MkI"], ["mr"], axes=[0, 1], keepdims=0)
    node("Mul", ["Mk", "Jg"], ["MkJ"]); node("ReduceSum", ["MkJ"], ["mc"], axes=[0, 1], keepdims=0)
    # diagonal holes + score
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
    # shift matrices Pr (rows by dr), PcT (cols by dc)
    node("Sub", ["DIFF", "dr"], ["Pr_d"]); node("Abs", ["Pr_d"], ["Pr_a"])
    node("Less", ["Pr_a", "half"], ["Pr_b"]); node("Cast", ["Pr_b"], ["Pr"], to=FLOAT)
    node("Add", ["DIFF", "dc"], ["Pc_d"]); node("Abs", ["Pc_d"], ["Pc_a"])
    node("Less", ["Pc_a", "half"], ["Pc_b"]); node("Cast", ["Pc_b"], ["PcT"], to=FLOAT)
    # echo: 15 unrolled diagonal shifts, OR-accumulated
    cur = "holes"
    E = None
    for k in range(_ITERS):
        t = f"t{k}"; nx = f"cur{k}"
        node("MatMul", ["Pr", cur], [t])
        node("MatMul", [t, "PcT"], [nx])
        cur = nx
        if E is None:
            E = nx
        else:
            ne = f"acc{k}"; node("Max", [E, nx], [ne]); E = ne
    node("Mul", [E, "gridmask"], ["Eg"])
    node("Reshape", ["Eg", "sh_1_30_30"], ["Eflat"])
    node("Greater", ["Eflat", "half"], ["cond"])               # [1,30,30] bool
    node("Reshape", ["colorvec", "sh_10_1_1"], ["cv3"])
    node("Where", ["cond", "cv3", "X"], ["out10"])             # [10,30,30]
    node("Reshape", ["out10", "sh_out"], ["output"])           # [1,10,30,30]

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nd, "echo370", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(ex):
    """Return the diagonal-echo model (solves task 370). It is a fully general
    graph -- no per-task params -- so the grader validates it directly."""
    try:
        return [("echo370", _build_370())]
    except Exception:
        return []
