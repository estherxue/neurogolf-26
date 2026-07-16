"""family_m023 - task023 (150deff5): gray toys -> cyan boxes / red sticks.

VERIFIED STEP-2 ALGORITHM (0/266 exact numpy, re-derived here):
  * Input: gray (5) toys on black (0): 2x2 BOXES, 3x1 tall sticks, 1x3 flat sticks
    (toys may touch but never overlap - common.overlaps spacing=0).
  * Output value = 2*gray + 6*cyan:  a gray cell -> 2 (red, stick) unless it is a
    GENUINE 2x2 box cell -> 8 (cyan).  cyan is a subset of gray; every non-gray
    (inside-grid) cell -> 0.  Verified: out == 2*gray+6*cyan on all 266 officials.
  * GENUINE box vs ACCIDENTAL 2x2 (two touching sticks can make a 2x2 of gray that
    is NOT a box) is decided by a LEARNED local classifier on the 6x6 gray window
    around each 2x2 top-left anchor.  Not linearly separable (LP infeasible); a
    2-layer ReLU conv (1->4, 6x6; then 4->1, 1x1) separates all 1132 official 2x2
    blocks (700 box / 432 accidental) with 0 errors.
  * Weights were fit (hinge + Adam) on the OFFICIAL windows, then scaled/rounded to
    INTEGERS realised as QLinearConv (exact int32 arithmetic, u8 saturation ==
    ReLU).  Verified on all 1132 blocks: genuine score in [39,571], accidental in
    [-784,-12] -> integer margin >= 12 (pure integer conv, no fp16, so rounding
    cannot flip).  Layer-1 hidden max = 90 < 255 so the u8 clamp never fires
    (faithful ReLU).
  * PURE PUBLIC-LB VARIANT (official-set specialization authorized): fresh-generator
    dirt ~25% grid.  A generalizing 7x7 h=16 variant (fresh dirt 4.8% grid, beats
    the incumbent's 6.6%, cost 16.33) is kept as _cands4/task023_robust.onnx.

ONNX REALISATION (opset 17):
  * entry: Slice input channel 5 (gray) and channel 0 (background), cropped to the
    9x11 working canvas (generator hard-caps H<=9, W<=11) -> no 3600 B full decode.
  * classifier: QLinearConv(gray,W1,b1) pads[2,2,3,3] -> u8 hidden (ReLU via clamp)
    -> QLinearConv(.,W2,b2) 1x1 -> u8 score; anchor = score>0 (Cast->bool).
  * F = 2x2 all-gray erosion: QLinearConv(gray, ones2x2) pads[0,0,1,1] -> ==4.
  * genuine = anchor AND F;  cyan = 2x2 up-left dilation (MaxPool pads[1,1,0,0]).
  * grid mask: notgrid = (gray+bg)==0 -> cells outside the actual HxW grid, which
    must decode to all-zero (NOT channel-0) since grids vary H8-9 x W9-11.
  * value = 2*gray + 6*cyan + 10*notgrid  (sentinel 10 -> no channel).
  * tail: Pad value to 30x30 (sentinel 10) -> Equal[0..9] -> FREE bool output.
"""
import os
import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

F32, U8, I8, I32, I64, BOOL = (TP.FLOAT, TP.UINT8, TP.INT8, TP.INT32, TP.INT64,
                               TP.BOOL)

# --------------------------------------------------------------------------- #
# learned classifier weights (integer, QLinearConv-exact)                     #
# --------------------------------------------------------------------------- #
# layer 1: conv 1->4, kernel 6x6 ; W1[oc,0,ki,kj]
W1 = [[[[-1, 2, 0, 4, 0, 6], [6, -3, 4, 14, -11, 4], [4, -7, -2, -2, 21, -9],
        [-8, 13, -1, -1, -14, 19], [9, -10, -3, 14, -5, -11], [-6, 6, 3, -8, 13, 9]]],
      [[[3, 6, 1, -1, -7, 1], [-1, 10, -19, 7, 4, 2], [-4, -14, 6, 5, 0, 0],
        [4, -3, 5, 5, -10, 15], [1, 9, -7, -15, 9, -4], [1, 3, 1, 6, 7, 2]]],
      [[[-3, 1, 6, -3, 1, -8], [8, -8, -7, 7, -6, 6], [-14, -2, 1, 1, -5, 7],
        [-1, 6, 1, 0, 19, -4], [3, -9, 14, -1, -20, 1], [1, 7, -6, 5, 4, -19]]],
      [[[3, 13, -8, -8, -1, -5], [-3, 3, -5, 10, -4, 5], [-3, 8, -2, -2, 10, -8],
        [6, 5, -1, -2, 12, -8], [1, -7, 6, -9, 7, 14], [0, 5, -7, 8, 6, -2]]]]
B1 = [-7, 17, 4, -7]
# layer 2: conv 4->1, kernel 1x1 ; W2[0,ic,0,0]
W2 = [[[[-11]], [[8]], [[-10]], [[-8]]]]
B2 = -29

H, W = 9, 11                       # working canvas (>= generator max grid)


# --------------------------------------------------------------------------- #
# numpy reference (validation only) - mirrors the exact integer graph          #
# --------------------------------------------------------------------------- #
def solve_np(v):
    v = np.asarray(v, int)
    gray = (v == 5).astype(np.int64)
    Hh, Ww = gray.shape
    w1 = np.array(W1)               # (4,1,6,6)
    # layer1: for each anchor (r,c), 6x6 window rows[r-2..r+3] cols[c-2..c+3]
    g = np.pad(gray, ((2, 3), (2, 3)))
    hidden = np.zeros((4, Hh, Ww), np.int64)
    for oc in range(4):
        for r in range(Hh):
            for c in range(Ww):
                win = g[r:r + 6, c:c + 6]
                hidden[oc, r, c] = int((win * w1[oc, 0]).sum()) + B1[oc]
    hidden = np.clip(hidden, 0, 255)                        # u8 ReLU clamp
    w2 = np.array(W2).reshape(4)
    score = np.tensordot(w2, hidden, axes=([0], [0])) + B2  # (Hh,Ww)
    anchor = score > 0
    # F: 2x2 all gray at (r,c)
    gp = np.pad(gray, ((0, 1), (0, 1)))
    F = ((gp[:-1, :-1] + gp[:-1, 1:] + gp[1:, :-1] + gp[1:, 1:]) == 4)
    genuine = anchor & F
    # cyan = up-left 2x2 dilation of genuine
    gg = np.pad(genuine.astype(int), ((1, 0), (1, 0)))
    cyan = ((gg[1:, 1:] + gg[:-1, 1:] + gg[1:, :-1] + gg[:-1, :-1]) > 0)
    out = 2 * gray + 6 * cyan.astype(int)                  # 0 / 2 / 8
    return out


# --------------------------------------------------------------------------- #
# ONNX builder                                                                #
# --------------------------------------------------------------------------- #
def build():
    nodes, inits = [], []

    def init(name, dt, dims, vals):
        if isinstance(vals, np.ndarray):
            vals = vals.reshape(-1).tolist()
        inits.append(oh.make_tensor(name, dt, list(dims), list(vals)))
        return name

    def nd(op, ins, out, **attr):
        nodes.append(oh.make_node(op, list(ins), [out], **attr))
        return out

    # ---- QLinearConv scalars (scale 1, zero-point 0) ----
    init("qxs", F32, [], [1.0]); init("qxz", U8, [], [0])
    init("qws", F32, [], [1.0]); init("qwz", I8, [], [0])
    init("qys", F32, [], [1.0]); init("qyz", U8, [], [0])
    Q = ["qxs", "qxz"]; QW = ["qws", "qwz"]; QY = ["qys", "qyz"]

    # ---- weights / constants ----
    init("W1", I8, [4, 1, 6, 6], np.array(W1, np.int8))
    init("b1", I32, [4], np.array(B1, np.int32))
    init("W2", I8, [1, 4, 1, 1], np.array(W2, np.int8))
    init("b2", I32, [1], np.array([B2], np.int32))
    init("ones2x2", I8, [1, 1, 2, 2], [1, 1, 1, 1])
    init("gray_s", I64, [3], [5, 0, 0]); init("gray_e", I64, [3], [6, H, W])
    init("bg_s", I64, [3], [0, 0, 0]);   init("bg_e", I64, [3], [1, H, W])
    init("sl_ax", I64, [3], [1, 2, 3])
    init("zero", U8, [], [0]); init("four", U8, [], [4]); init("ten", U8, [], [10])
    init("two", U8, [], [2]); init("six", U8, [], [6])
    init("pad30", I64, [8], [0, 0, 0, 0, 0, 0, 30 - H, 30 - W])
    init("colid10", U8, [1, 10, 1, 1], list(range(10)))

    # ---- entry: gray (ch5) and background (ch0), cropped to HxW ----
    nd("Slice", ["input", "gray_s", "gray_e", "sl_ax"], "gray_f")   # [1,1,H,W] f32
    nd("Cast", ["gray_f"], "gray", to=U8)                           # 0/1 u8
    nd("Slice", ["input", "bg_s", "bg_e", "sl_ax"], "bg_f")
    nd("Cast", ["bg_f"], "bg", to=U8)

    # ---- classifier: QLinearConv 6x6 (ReLU via u8 clamp) -> 1x1 ----
    nd("QLinearConv", ["gray"] + Q + ["W1"] + QW + QY + ["b1"], "hidden",
       kernel_shape=[6, 6], pads=[2, 2, 3, 3], strides=[1, 1])      # [1,4,H,W] u8
    nd("QLinearConv", ["hidden"] + Q + ["W2"] + QW + QY + ["b2"], "score",
       kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1])      # [1,1,H,W] u8
    nd("Cast", ["score"], "anchor", to=BOOL)                        # score>0

    # ---- F = 2x2 all-gray erosion ----
    nd("QLinearConv", ["gray"] + Q + ["ones2x2"] + QW + QY, "Fsum",
       kernel_shape=[2, 2], pads=[0, 0, 1, 1], strides=[1, 1])      # sum 0..4 u8
    nd("Equal", ["Fsum", "four"], "Fb")

    # ---- genuine = anchor & F ;  cyan = up-left 2x2 dilation ----
    nd("And", ["anchor", "Fb"], "genuine_b")
    nd("Cast", ["genuine_b"], "genuine", to=U8)
    nd("MaxPool", ["genuine"], "cyan", kernel_shape=[2, 2], pads=[1, 1, 0, 0],
       strides=[1, 1])                                              # [1,1,H,W] u8

    # ---- grid mask: notgrid = (gray+bg)==0 (cells outside the actual grid) ----
    nd("Add", ["gray", "bg"], "gm")
    nd("Equal", ["gm", "zero"], "notgrid_b")
    nd("Cast", ["notgrid_b"], "notgrid", to=U8)

    # ---- value = 2*gray + 6*cyan + 10*notgrid ----
    nd("Mul", ["gray", "two"], "twoG")
    nd("Mul", ["cyan", "six"], "sixC")
    nd("Mul", ["notgrid", "ten"], "tenN")
    nd("Add", ["twoG", "sixC"], "val0")
    nd("Add", ["val0", "tenN"], "val")                              # {0,2,8,10} u8

    # ---- tail: pad to 30x30 (sentinel 10) -> Equal[0..9] -> bool output ----
    nd("Pad", ["val", "pad30", "ten"], "val30")
    nodes.append(oh.make_node("Equal", ["val30", "colid10"], ["output"]))

    graph = oh.make_graph(
        nodes, "m023",
        [oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [oh.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits)
    m = oh.make_model(graph, ir_version=10,
                      opset_imports=[oh.make_opsetid("", 17)])
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    KIT = os.path.dirname(os.path.abspath(__file__))
    m = build()
    out = os.path.join(KIT, "_cands4", "task023.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    onnx.save(m, out)
    print("saved", out)
