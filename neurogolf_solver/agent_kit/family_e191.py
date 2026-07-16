"""ENTROPY-FLOOR rebuild of task191 (arc-gen hash 7df24a62).

Rule: a yellow motif is stamped at several spots (each in one of 8 D4
orientations); one reference spot carries a blue frame (its filled bbox). The
OUTPUT redraws the blue bbox around EVERY occurrence (motif kept yellow on top).

blend6 (the prior best) built 8 static orientation kernels and detected with an
f16 Conv, then Equal -> Cast -> ConvTranspose.  Its three [1,8,23,23] tensors
(score f16, valid bool, valid f16 = 21160 B) dominate the 35335 B cost.

This rebuild keeps blend6's cheap, statically-shaped bbox/motif front-end and its
8 static orientation kernels, but collapses the whole detection+paint core into
TWO integer convolutions:

  * yellow -> uint8; validity kernels = motif(+1) + rect_excl*(-100) as int8.
  * DETECT: valid = QLinearConv(yellow_u8, validity_8, bias = (1 - total_motif)).
    With uint8 output saturating at 0, an exact match scores total_motif and the
    bias maps it to exactly 1; every partial/violating anchor maps to <=0 -> 0.
    So `valid` is a binary [1,8,23,23] uint8 map with NO Equal/Cast needed.
  * PAINT: paint = QLinearConv(valid, flip(rect_excl) as [1,8,5,5]) -- the Conv
    adjoint of blend6's ConvTranspose -- summing the 8 bbox footprints into one
    [1,1,23,23] uint8 coverage map, all in integer arithmetic (no f16).

Detection+paint shrinks from 21160 B (3x f16/bool [1,8,*]) to one [1,8,23,23]
uint8 (4232) + one [1,1,23,23] uint8 (529).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

IR_VERSION = 8
OPSET = [oh.make_opsetid("", 17)]
F32, F16, U8, I8, I32, I64, BOOL = (TP.FLOAT, TP.FLOAT16, TP.UINT8, TP.INT8,
                                    TP.INT32, TP.INT64, TP.BOOL)


class _G:
    def __init__(self):
        self.nodes, self.inits, self._k = [], [], 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}_{self._k}"

    def init(self, name, dt, dims, vals):
        self.inits.append(oh.make_tensor(name, dt, list(dims), list(vals)))
        return name

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm(op.lower())
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def build():
    g = _G()
    S = 23  # generator fixes size=23

    # ---- constants -------------------------------------------------------- #
    g.init("range5", I64, [5], [0, 1, 2, 3, 4])
    g.init("rangeG", F32, [S], list(range(S)))
    g.init("one_i64", I64, [], [1])
    g.init("c100_i8", I8, [], [100])
    g.init("one_i32", I32, [], [1])
    g.init("z8_i32", I32, [8], [0] * 8)
    # QLinearConv scalars
    g.init("xs", F32, [], [1.0]); g.init("xz", U8, [], [0])
    g.init("ws", F32, [], [1.0]); g.init("wz", I8, [], [0])
    g.init("ys", F32, [], [1.0]); g.init("yz", U8, [], [0])
    g.init("zero_u8", U8, [], [0]); g.init("one_u8", U8, [], [1])
    g.init("four_u8", U8, [], [4]); g.init("ten_u8", U8, [], [10])
    g.init("col_idx", U8, [1, 10, 1, 1], list(range(10)))
    # slice specs
    g.init("s_y0", I64, [3], [4, 0, 0]); g.init("s_y1", I64, [3], [5, S, S])
    g.init("s_b0", I64, [3], [1, 0, 0]); g.init("s_b1", I64, [3], [2, S, S])
    g.init("ax123", I64, [3], [1, 2, 3])
    g.init("ax01", I64, [2], [0, 1])
    g.init("ax0", I64, [1], [0]); g.init("ax1", I64, [1], [1])
    g.init("ax23", I64, [2], [2, 3])
    g.init("pad_y", I64, [8], [0, 0, 0, 0, 0, 0, 2, 1])       # yellow23 -> 25x24
    g.init("pad30", I64, [8], [0, 0, 0, 0, 0, 0, 7, 7])       # color23 -> 30x30
    g.init("sh1155", I64, [4], [1, 1, 5, 5])
    g.init("sh1855", I64, [4], [1, 8, 5, 5])
    # reverse-slice specs (step -1)
    g.init("neg1", I64, [1], [-1]); g.init("neg6", I64, [1], [-6])
    g.init("axm2", I64, [1], [-2])
    g.init("neg1x2", I64, [2], [-1, -1]); g.init("neg6x2", I64, [2], [-6, -6])

    # ---- front-end: slices ------------------------------------------------ #
    yf = g.nd("Slice", ["input", "s_y0", "s_y1", "ax123"], "yellow_f32")   # [1,1,23,23]
    bf = g.nd("Slice", ["input", "s_b0", "s_b1", "ax123"], "blue_f32")
    yu8 = g.nd("Cast", [yf], "yellow_u8", to=U8)

    # ---- bbox of the blue frame ------------------------------------------ #
    rh = g.nd("Squeeze", [g.nd("ReduceMax", [bf], "row_has4d", axes=[3], keepdims=0),
                          "ax01"], "row_has")                              # [23]
    ch = g.nd("Squeeze", [g.nd("ReduceMax", [bf], "col_has4d", axes=[2], keepdims=0),
                          "ax01"], "col_has")
    r0 = g.nd("ArgMax", [rh], "r0", axis=0, keepdims=0)
    r1 = g.nd("ArgMax", [g.nd("Mul", [rh, "rangeG"], "row_score")], "r1", axis=0, keepdims=0)
    c0 = g.nd("ArgMax", [ch], "c0", axis=0, keepdims=0)
    c1 = g.nd("ArgMax", [g.nd("Mul", [ch, "rangeG"], "col_score")], "c1", axis=0, keepdims=0)
    th = g.nd("Add", [g.nd("Sub", [r1, r0], "rspan"), "one_i64"], "th")     # tall+2
    tw = g.nd("Add", [g.nd("Sub", [c1, c0], "cspan"), "one_i64"], "tw")     # wide+2

    # ---- extract 5x5 patch at bbox top-left via Gather (static shape) ----- #
    y27 = g.nd("Pad", [yu8, "pad_y", "zero_u8"], "yellow_u8_pad")           # [1,1,25,24]
    ridx = g.nd("Add", ["range5", r0], "row_idx")                          # [5]
    cidx = g.nd("Add", ["range5", c0], "col_idx5")                         # [5]
    prow = g.nd("Gather", [y27, ridx], "patch_rows", axis=2)               # [1,1,5,24]
    patch = g.nd("Gather", [prow, cidx], "patch_u8", axis=3)               # [1,1,5,5] u8

    # ---- box (rect) mask, top-left anchored (static) --------------------- #
    rin = g.nd("Less", ["range5", th], "row_in")                           # [5] bool
    cin = g.nd("Less", ["range5", tw], "col_in")
    box2d = g.nd("And", [g.nd("Unsqueeze", [rin, "ax1"], "row_in_5x1"),
                         g.nd("Unsqueeze", [cin, "ax0"], "col_in_1x5")], "box2d")  # [5,5]
    rect = g.nd("Reshape", [g.nd("Cast", [box2d], "box2d_i8", to=I8), "sh1155"], "rect_i8")
    patch_i8 = g.nd("Cast", [patch], "patch_i8", to=I8)
    motif = g.nd("Mul", [patch_i8, rect], "motif_i8")                      # [1,1,5,5] i8

    # total (rotation invariant) and detection bias = 1 - total
    total_i32 = g.nd("ReduceSum", [g.nd("Cast", [motif], "motif_i32", to=I32)],
                     "total_i32", keepdims=0)                              # scalar i32
    bias = g.nd("Add", ["z8_i32", g.nd("Sub", ["one_i32", total_i32], "one_minus_tot")],
                "det_bias")                                                # [8] i32

    # ---- 8 D4 orientations (constant reversals/transposes -> static) ----- #
    def revc(x): return g.nd("Slice", [x, "neg1", "neg6", "neg1", "neg1"], g.nm("revc"))
    def revr(x): return g.nd("Slice", [x, "neg1", "neg6", "axm2", "neg1"], g.nm("revr"))
    def transp(x): return g.nd("Transpose", [x], g.nm("T"), perm=[0, 1, 3, 2])

    def orient8(base):
        fH = revc(base); fV = revr(base); t2 = revr(fH)
        t6 = transp(base); t1 = revr(t6); t3 = revc(t6); t7 = transp(t2)
        return g.nd("Concat", [base, t1, t2, t3, fH, fV, t6, t7], g.nm("stack8"), axis=0)

    motif8 = orient8(motif)                                                # [8,1,5,5] i8
    rect8 = orient8(rect)                                                  # [8,1,5,5] i8
    excl8 = g.nd("Sub", [rect8, motif8], "rect_excl8")                     # box-minus-motif
    validity8 = g.nd("Sub", [motif8, g.nd("Mul", [excl8, "c100_i8"], "excl_neg")],
                     "validity8")                                         # motif - 100*excl

    # ---- DETECT: one QLinearConv -> binary [1,8,23,23] uint8 ------------- #
    valid = g.nd("QLinearConv",
                 [yu8, "xs", "xz", validity8, "ws", "wz", "ys", "yz", bias], "valid_u8",
                 group=1, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1])

    # ---- PAINT: Conv adjoint of blend6's ConvTranspose ------------------- #
    # flip rect_excl spatially, relayout [8,1,5,5] -> [1,8,5,5]
    excl_flip = g.nd("Slice", [excl8, "neg1x2", "neg6x2", "ax23", "neg1x2"], "excl_flip")
    paint_w = g.nd("Reshape", [excl_flip, "sh1855"], "paint_w")            # [1,8,5,5] i8
    paint = g.nd("QLinearConv",
                 [valid, "xs", "xz", paint_w, "ws", "wz", "ys", "yz"], "paint_u8",
                 group=1, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1])

    # ---- compose colours -------------------------------------------------- #
    # yellow (4) wins over painted blue (1) wins over black (0).
    yb = g.nd("Cast", [yu8], "yellow_b", to=BOOL)
    pb = g.nd("Cast", [paint], "paint_b", to=BOOL)                        # nonzero -> painted
    bg = g.nd("Where", [pb, "one_u8", "zero_u8"], "color_bg")            # blue / black
    col23 = g.nd("Where", [yb, "four_u8", bg], "color23")                # yellow overrides
    col30 = g.nd("Pad", [col23, "pad30", "ten_u8"], "color30")            # [1,1,30,30] u8
    g.nd("Equal", [col30, "col_idx"], "output")                          # [1,10,30,30] bool

    x = oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = oh.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = oh.make_graph(g.nodes, "e191", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def _to_onehot(grid):
    g = np.array(grid, int)
    H, W = g.shape
    x = np.zeros((1, 10, 30, 30), np.float32)
    for r in range(H):
        for c in range(W):
            x[0, g[r, c], r, c] = 1.0
    return x


def candidates(ex):
    import onnxruntime as ort
    try:
        model = build()
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    pairs = [(e["input"], e["output"]) for e in ex.get("train", []) + ex.get("test", [])]
    if not pairs:
        return []
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model.SerializeToString(), so)
    except Exception:
        return []
    try:
        for gin, gout in pairs:
            if max(len(gin), len(gin[0])) > 30:
                return []
            out = (sess.run(["output"], {"input": _to_onehot(gin)})[0] > 0.0).astype(float)
            exp = _to_onehot(gout)
            if out.shape != exp.shape or not (out == exp).all():
                return []
    except Exception:
        return []
    return [("e191", model)]
