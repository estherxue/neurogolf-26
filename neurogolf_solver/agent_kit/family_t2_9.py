"""family_t2_9 -- ENRICHED recompile of task 9 (hash 06df4c85, "gridlines").

Task: a size x size lattice of 2x2 colour cells separated by 1-wide `linecolor`
gridlines.  Pairs of equal-colour cells that share a row (or column) are joined
by a straight line of that colour; the single random "dot" and the gridlines are
left untouched.  The generator guarantees (a) <=2 cells of any colour per row and
per column, and (b) no two colour segments ever cross (`horiz_free`/`vert_free`),
so every cell is painted by at most ONE colour -> the join is exactly the
per-colour "bracketed" fill computed by the two Einsums below.

Incumbent = out_blend12/onnx/task009.onnx ("fp16_9_h", grader points 16.19).  It
is built at cell resolution (10x10) and expanded to 30x30 with a single
DepthToSpace, which is already memory-optimal for the dominant tensor
`onehot16` [1,9,10,10] f16 = 1800 B (the per-colour one-hot feeding both
connection Einsums; no cheaper exact form exists -- OneHot needs int64 indices,
which is larger).

HOWEVER the incumbent combines the horizontal fill / vertical fill / own-content
with a ternary **uint8 Max** (and builds the block-corner separator with a second
uint8 Max).  ORT 1.23.2 (the local scoring runtime, hard-gate a) does NOT
implement Max/Sum/Min for uint8|int8, so the incumbent does not even load
locally.  This module rebuilds the graph with only locally-supported ops:

  * content combine  ->  f16 ternary Max (h_between, v_between, cast(cgridV))
                         then a single Cast to u8.  (uint8 Max is unavailable and
                         there is no ternary uint8 op, so a wider f16 combine or a
                         binary Add-chain intermediate is unavoidable: +100 B.)
  * separator corner ->  the valid region is always a full square, hence
                         xsep[h,w] == linegrid[h+1,w+1]; all three separators
                         (v/h/x) are static slices of a single 255-padded
                         line-grid, which removes the uint8 Max, the col255/row255
                         params and two tail tensors (-59 B, -2 params).

Net vs the (locally-unloadable) incumbent: +41 B memory, cost 6738 vs 6699 ->
points 16.184.  The candidate is exact on train+test+arc-gen and on 3000 fresh
generator samples, opt-level invariant, 21 nodes.  It is the best result that
LOADS on local ORT; it is ~0.006 pt shy of the incumbent's grader score because
that score relies on an op the local runtime cannot execute.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, numpy_helper as onp, TensorProto as TP

from ng_utils_shim import GRID_SHAPE, CHANNELS, HEIGHT, WIDTH

F32, F16, U8, I64, BOOL = (
    TP.FLOAT, TP.FLOAT16, TP.UINT8, TP.INT64, TP.BOOL,
)


def _build_model():
    inits = []

    def cf32(name, dims, vals):
        inits.append(oh.make_tensor(name, F32, list(dims),
                                    np.asarray(vals, np.float32).ravel().tolist()))

    def cf16(name, dims, vals):
        a = np.asarray(vals, np.float16).reshape(list(dims))
        inits.append(onp.from_array(a, name))

    def cu8(name, dims, vals):
        a = np.asarray(vals, np.uint8).reshape(list(dims))
        inits.append(onp.from_array(a, name))

    def ci(name, vals):
        inits.append(oh.make_tensor(name, I64, [len(vals)], [int(v) for v in vals]))

    # -- colour / einsum constants -------------------------------------------
    cf32("conv_w", [1, CHANNELS, 1, 1], [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cf32("colorids", [1, 9, 1, 1], list(range(1, 10)))
    cf16("cv16", [9], list(range(1, 10)))
    cf16("sepA", [10, 10], np.triu(np.ones((10, 10), np.float16), k=1))
    cu8("all_color_ids", [1, CHANNELS, 1, 1], list(range(10)))
    cu8("u255", [], [255])
    cf32("zero", [], [0.0])

    # -- slice / pad index constants -----------------------------------------
    ci("ls", [2, 2]); ci("le", [3, 3]); ci("la", [2, 3])          # linecolor pixel
    ci("pads", [0, 0, 0, 0, 0, 0, 1, 1])                          # pad end of H,W by 1
    ci("v_s", [0, 1]); ci("v_e", [10, 11])                       # right neighbour
    ci("h_s", [1, 0]); ci("h_e", [11, 10])                       # bottom neighbour
    ci("x_s", [1, 1]); ci("x_e", [11, 11])                       # diagonal neighbour
    ci("ax23", [2, 3])

    N = oh.make_node
    nodes = [
        # cell-resolution colour value grid: 0=pad, 0.5=background, 1..9=colour
        N("Conv", ["input", "conv_w"], ["cgridV"],
          kernel_shape=[1, 1], strides=[3, 3]),
        # per-colour one-hot [1,9,10,10] (f16) -- the DOMINANT tensor (1800 B)
        N("Equal", ["cgridV", "colorids"], ["eq_b"]),
        N("Cast", ["eq_b"], ["onehot16"], to=F16),
        # horizontal / vertical "bracketed" fills (strictly between equal cells)
        N("Einsum", ["onehot16", "onehot16", "cv16", "sepA", "sepA"], ["h_between"],
          equation="ncya,ncyb,c,ax,xb->nyx"),
        N("Einsum", ["onehot16", "onehot16", "cv16", "sepA", "sepA"], ["v_between"],
          equation="ncax,ncbx,c,ay,yb->nyx"),
        # content = max(h_fill, v_fill, own_colour) -- disjoint, done in f16
        # because ORT 1.23.2 has no uint8 Max/Sum/Min.
        N("Cast", ["cgridV"], ["cgrid16"], to=F16),
        N("Max", ["h_between", "v_between", "cgrid16"], ["content16"]),
        N("Cast", ["content16"], ["content_u8"], to=U8),
        # mask padding cells to 255 (all-zero one-hot in the output)
        N("Greater", ["cgridV", "zero"], ["valid_b"]),
        N("Where", ["valid_b", "content_u8", "u255"], ["content_grid"]),
        # linecolor = colour at the (2,2) separator pixel
        N("Slice", ["input", "ls", "le", "la"], ["line_oh"]),
        N("ArgMax", ["line_oh"], ["line_i64"], axis=1, keepdims=1),
        N("Cast", ["line_i64"], ["line_u8"], to=U8),
        N("Where", ["valid_b", "line_u8", "u255"], ["line_grid"]),
        # separators = static shifts of a 255-padded line grid (no uint8 Max):
        #   v_sep[h,w]=line[h,w+1]  h_sep[h,w]=line[h+1,w]  x_sep[h,w]=line[h+1,w+1]
        # (valid region is always a full square so the diagonal shift is exact).
        N("Pad", ["line_grid", "pads", "u255"], ["line_grid_pad"], mode="constant"),
        N("Slice", ["line_grid_pad", "v_s", "v_e", "ax23"], ["v_sep"]),
        N("Slice", ["line_grid_pad", "h_s", "h_e", "ax23"], ["h_sep"]),
        N("Slice", ["line_grid_pad", "x_s", "x_e", "ax23"], ["x_sep"]),
        # assemble each cell's 3x3 block and expand 10x10 -> 30x30
        N("Concat", ["content_grid", "content_grid", "v_sep",
                     "content_grid", "content_grid", "v_sep",
                     "h_sep", "h_sep", "x_sep"], ["blocks"], axis=1),
        N("DepthToSpace", ["blocks"], ["colormap"], blocksize=3, mode="DCR"),
        N("Equal", ["colormap", "all_color_ids"], ["output"]),
    ]

    x = oh.make_tensor_value_info("input", F32, list(GRID_SHAPE))
    y = oh.make_tensor_value_info("output", BOOL,
                                  [1, CHANNELS, HEIGHT, WIDTH])
    graph = oh.make_graph(nodes, "family_t2_9", [x], [y], inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)],
                          ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    return model


def _to_onehot(grid):
    a = np.asarray(grid, np.int64)
    h, w = a.shape
    x = np.zeros((1, CHANNELS, HEIGHT, WIDTH), np.float32)
    for c in range(CHANNELS):
        x[0, c, :h, :w] = (a == c)
    return x


def _expected(grid):
    a = np.asarray(grid, np.int64)
    h, w = a.shape
    y = np.zeros((CHANNELS, HEIGHT, WIDTH), bool)
    for c in range(CHANNELS):
        y[c, :h, :w] = (a == c)
    return y


def _exact_on(model, pairs):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    for gi, go in pairs:
        out = np.asarray(sess.run(None, {"input": _to_onehot(gi)})[0])[0] > 0
        if not np.array_equal(out, _expected(go)):
            return False
    return True


def candidates(examples):
    """Return [(name, model)] iff the model reproduces every train+test pair."""
    pairs = [(e["input"], e["output"])
             for e in examples.get("train", []) + examples.get("test", [])]
    if not pairs:
        return []
    try:
        model = _build_model()
    except Exception:
        return []
    if not _exact_on(model, pairs):
        return []
    return [("t2_9", model)]
