"""family_m044 — task044 (228f6490): box-hole recolor by shape-matched sprite.

VERIFIED STEP-2 ALGORITHM (0/266 exact numpy):
  * Input: gray boxes (filled rects) with a BLACK creature-shaped hole carved in;
    free colored sprites (same creature shapes) elsewhere; scattered dust pixels.
  * Output: keep gray + dust; ERASE free sprites; FILL each box's hole with the
    color of the sprite whose SHAPE matches (same creature, translated).
  * Hole detection: a black cell is a hole iff gray is reachable BOTH left and
    right along its row (horizontal-only reach suffices: boxes never share a row,
    so gray-on-both-sides => interior of the one box in that row).  Verified: L&R
    reach gives identical holes to the 4-dir version.
  * Matching: translation-invariant integer shape signature (n, Srr, Scc) where
    Srr = n*Sum(r^2) - (Sum r)^2, Scc = n*Sum(c^2) - (Sum c)^2, uniquely pairs each
    hole with its sprite and rejects same-count dust.  Verified on 266 officials:
    count alone 227/266, (n,Srr) 264, (n,Scc) 265, (n,Srr,Scc) 266.

ONNX REALISATION (opset 17):
  * decode: dilated Conv (kernel 2x2, dilation 20) collapses one-hot -> 10x10
    value grid directly (400 B f32) instead of the 3600 B full-grid decode.
  * reach: two directional MaxPools on the u8 gray mask (cumulative-OR).
  * per-color signatures: Einsum contractions of the one-hot 'input' -> [1,10]
    (n, Sum r, Sum r^2, Sum c, Sum c^2) -> Srr, Scc.  No large intermediates.
  * per-hole signatures: split the two boxes at the always-empty gap row 5 with a
    strided ConvInteger (kernel 5x10 / 5x1, stride 5) -> row/col marginals per
    half -> (n,Srr,Scc) for the top and bottom hole.
  * match: Equal broadcast [1,10] vs [1,2] on the integer triple -> one-hot color
    per half -> k1 (top), k2 (bottom).
  * paint: erase sprite-colored cells; fill hole cells with row-dependent k1/k2.
  * tail: pad value plane to 30x30 (sentinel 10) -> Equal[0..9] -> bool output.
"""
import os
import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

F32, U8, I8, I32, I64, BOOL = (TP.FLOAT, TP.UINT8, TP.INT8, TP.INT32, TP.INT64,
                               TP.BOOL)


# --------------------------------------------------------------------------- #
# numpy reference (validation only)                                           #
# --------------------------------------------------------------------------- #
_R = (np.arange(10)[:, None] * np.ones((1, 10), int))
_C = _R.T


def _sig(mask):
    n = int(mask.sum())
    if n == 0:
        return None
    sr = int((mask * _R).sum())
    sc = int((mask * _C).sum())
    srr = n * int((mask * _R * _R).sum()) - sr * sr
    scc = n * int((mask * _C * _C).sum()) - sc * sc
    return (n, srr, scc)


def solve_np(v):
    v = np.asarray(v, int)
    gray = (v == 5)
    L = np.maximum.accumulate(gray, axis=1)
    Rr = np.maximum.accumulate(gray[:, ::-1], axis=1)[:, ::-1]
    hole = (L & Rr) & (v == 0)
    top = hole & (_R < 5)
    bot = hole & (_R >= 5)
    ts, bs = _sig(top), _sig(bot)
    k1 = k2 = 0
    for k in range(1, 10):
        if k == 5:
            continue
        m = (v == k)
        if not m.any():
            continue
        s = _sig(m)
        if s == ts:
            k1 = k
        if s == bs:
            k2 = k
    out = v.copy()
    out[(v == k1) | (v == k2)] = 0
    out[top] = k1
    out[bot] = k2
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

    # ---- constants ----
    # decode: weight[0,k,0,0] = k, other taps 0  (dilation 20 -> reads 10x10 region)
    w = np.zeros((1, 10, 2, 2), np.float32)
    for k in range(10):
        w[0, k, 0, 0] = k
    init("dec_w", F32, [1, 10, 2, 2], w)
    init("five", U8, [], [5])
    init("zero", U8, [], [0])
    # per-color coordinate vectors over the 30-wide axes
    init("rvec", F32, [30], [float(i) for i in range(30)])
    init("r2vec", F32, [30], [float(i * i) for i in range(30)])
    init("cvec", F32, [30], [float(i) for i in range(30)])
    init("c2vec", F32, [30], [float(i * i) for i in range(30)])
    # per-hole: strided-conv coordinate kernels over a 5x10 window; 5 output
    # channels pack (ones, r, r^2, c, c^2) so one ConvInteger yields all moments.
    Wr = np.zeros((5, 1, 5, 10), np.uint8)
    Wr[0] = 1
    for r in range(5):
        Wr[1, 0, r, :] = r
        Wr[2, 0, r, :] = r * r
    for c in range(10):
        Wr[3, 0, :, c] = c
        Wr[4, 0, :, c] = c * c
    init("Wr", U8, [5, 1, 5, 10], Wr)
    # matching / paint helpers
    init("s_10_1", I64, [3], [1, 10, 1])
    init("s_1_2", I64, [3], [1, 1, 2])
    init("s_1_2b", I64, [2], [1, 2])
    init("s_1111", I64, [4], [1, 1, 1, 1])
    init("ax2", I64, [1], [2])
    init("k0s", I64, [1], [0])
    init("k1s", I64, [1], [1])
    init("k2s", I64, [1], [2])
    rowtop = np.zeros((1, 1, 10, 1), bool)
    rowtop[0, 0, :5, 0] = True
    init("rowtop", BOOL, [1, 1, 10, 1], rowtop.astype(np.uint8).tolist())
    init("colid10", U8, [1, 10, 1, 1], list(range(10)))
    init("pad30", I64, [8], [0, 0, 0, 0, 0, 0, 20, 20])
    init("ten", U8, [], [10])

    # ---- decode + gray + hole ----
    nd("Conv", ["input", "dec_w"], "idx_f", dilations=[20, 20],
       kernel_shape=[2, 2])
    nd("Cast", ["idx_f"], "v", to=U8)                         # [1,1,10,10] value
    nd("Equal", ["v", "five"], "gray_b")
    nd("Cast", ["gray_b"], "gray", to=U8)
    nd("MaxPool", ["gray"], "reachL", kernel_shape=[1, 10], pads=[0, 9, 0, 0],
       strides=[1, 1])
    nd("MaxPool", ["gray"], "reachR", kernel_shape=[1, 10], pads=[0, 0, 0, 9],
       strides=[1, 1])
    nd("BitwiseAnd", ["reachL", "reachR"], "inside")
    nd("Sub", ["inside", "gray"], "hole")                     # u8 hole mask
    nd("Cast", ["hole"], "hole_b", to=BOOL)

    # ---- per-hole signatures (top/bottom via strided conv split at row 5) ----
    nd("ConvInteger", ["hole", "Wr"], "hstuff_i", kernel_shape=[5, 10],
       strides=[5, 1])                                        # [1,5,2,1] i32
    nd("Cast", ["hstuff_i"], "hstuff", to=F32)                # [1,5,2,1]
    nodes.append(oh.make_node(
        "Split", ["hstuff"], ["nH4", "SrH4", "Sr2H4", "ScH4", "Sc2H4"],
        axis=1, num_outputs=5))                               # each [1,1,2,1]
    nd("Reshape", ["nH4", "s_1_2b"], "nH")                    # [1,2]
    nd("Reshape", ["SrH4", "s_1_2b"], "SrH")
    nd("Reshape", ["Sr2H4", "s_1_2b"], "Sr2H")
    nd("Reshape", ["ScH4", "s_1_2b"], "ScH")
    nd("Reshape", ["Sc2H4", "s_1_2b"], "Sc2H")
    # central second moments per half
    nd("Mul", ["nH", "Sr2H"], "nSr2H")
    nd("Mul", ["SrH", "SrH"], "SrH2")
    nd("Sub", ["nSr2H", "SrH2"], "srrH")                      # [1,2]
    nd("Mul", ["nH", "Sc2H"], "nSc2H")
    nd("Mul", ["ScH", "ScH"], "ScH2")
    nd("Sub", ["nSc2H", "ScH2"], "sccH")

    # ---- per-color signatures over the one-hot input ----
    nd("Einsum", ["input"], "N", equation="bchw->bc")         # [1,10]
    nd("Einsum", ["input", "rvec"], "SR", equation="bchw,h->bc")
    nd("Einsum", ["input", "r2vec"], "SR2", equation="bchw,h->bc")
    nd("Einsum", ["input", "cvec"], "SC", equation="bchw,w->bc")
    nd("Einsum", ["input", "c2vec"], "SC2", equation="bchw,w->bc")
    nd("Mul", ["N", "SR2"], "nSR2")
    nd("Mul", ["SR", "SR"], "SR_2")
    nd("Sub", ["nSR2", "SR_2"], "SRR")                        # [1,10]
    nd("Mul", ["N", "SC2"], "nSC2")
    nd("Mul", ["SC", "SC"], "SC_2")
    nd("Sub", ["nSC2", "SC_2"], "SCC")

    # ---- match: [1,10] color vs [1,2] half ----
    nd("Reshape", ["N", "s_10_1"], "Ncol")
    nd("Reshape", ["SRR", "s_10_1"], "SRRcol")
    nd("Reshape", ["SCC", "s_10_1"], "SCCcol")
    nd("Reshape", ["nH", "s_1_2"], "nHrow")
    nd("Reshape", ["srrH", "s_1_2"], "srrHrow")
    nd("Reshape", ["sccH", "s_1_2"], "sccHrow")
    nd("Equal", ["Ncol", "nHrow"], "eqN")                     # [1,10,2]
    nd("Equal", ["SRRcol", "srrHrow"], "eqSRR")
    nd("Equal", ["SCCcol", "sccHrow"], "eqSCC")
    nd("And", ["eqN", "eqSRR"], "eq01")
    nd("And", ["eq01", "eqSCC"], "match")                     # [1,10,2] bool
    nd("Cast", ["match"], "match_u8", to=U8)                  # [1,10,2] u8
    nd("ArgMax", ["match_u8"], "kH_i", axis=1, keepdims=1)    # [1,1,2] i64
    nd("Cast", ["kH_i"], "kH", to=U8)                         # [1,1,2] u8
    nd("Slice", ["kH", "k0s", "k1s", "ax2"], "k1r")           # [1,1,1]
    nd("Slice", ["kH", "k1s", "k2s", "ax2"], "k2r")
    nd("Reshape", ["k1r", "s_1111"], "k1u")                   # [1,1,1,1] u8
    nd("Reshape", ["k2r", "s_1111"], "k2u")

    # ---- paint ----
    nd("Equal", ["v", "k1u"], "e1")
    nd("Equal", ["v", "k2u"], "e2")
    nd("Or", ["e1", "e2"], "sprite")
    nd("Where", ["sprite", "zero", "v"], "erased")            # erase free sprites
    nd("Where", ["rowtop", "k1u", "k2u"], "fill")             # [1,1,10,10] u8
    nd("Where", ["hole_b", "fill", "erased"], "out10")

    # ---- tail: pad to 30x30 (sentinel 10) -> Equal[0..9] -> bool output ----
    nd("Pad", ["out10", "pad30", "ten"], "out30")
    nodes.append(oh.make_node("Equal", ["out30", "colid10"], ["output"]))

    graph = oh.make_graph(
        nodes, "m044",
        [oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [oh.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits)
    m = oh.make_model(graph, ir_version=10,
                      opset_imports=[oh.make_opsetid("", 18)])
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    KIT = os.path.dirname(os.path.abspath(__file__))
    m = build()
    out = os.path.join(KIT, "_cands4", "task044.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    onnx.save(m, out)
    print("saved", out)
