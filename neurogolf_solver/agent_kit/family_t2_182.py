"""family_t2_182 — cheaper rebuild of task 182 (776ffc46).

Generator rule: a 20x20 grid holds 5-6 non-overlapping 5x5 sprites.  One sprite
(the "marker", colour 2 or 3) sits inside a 7x7 gray(5) box; every other sprite
is drawn in colour 1.  At least one colour-1 sprite has the SAME shape as the
marker.  In the output every colour-1 sprite whose shape exactly matches the
marker's shape is recoloured to the marker colour.

Pipeline (mirrors the incumbent's decode + box-find + template extraction, but
replaces the match/spread/recolour tail with an ADJOINT-CONV PAINT that drops a
full-grid [1,1,20,20] intermediate):

    cidf32 = Conv(one-hot, [0..9])            # id map, cropped to 20x20  (dominant f32 tensor)
    cid    = Cast(cidf32, u8)
    box    = AvgPool argmax over cidf32        # 7-wide band with the gray box
    win    = cid[box+1 : box+6]                # 5x5 sprite region (the marker)
    tmask  = win > 0                           # marker shape mask
    kern   = 2*tmask   (w_zp = 1  ->  +1 / -1 signed weights)
    hits   = QLinearConv(c1, kern, B = 1 - cnt)   # BINARY exact-match centres in ONE op
    paint  = QLinearConv(hits, flip(tmask))       # stamp the shape back = matched sprite cells
    out_id = Where(paint>0, marker, cid)
    output = Equal(Pad(out_id, 255), [0..9])       # one-hot, padded region -> all-zero

The incumbent (out_blend12/onnx/task182.onnx) carries the extra tail tensors
score / score_dil / dil_b / recolor_b (4 x [1,1,20,20]); the adjoint paint needs
only hits / paint (2), so one full-grid u8 tensor is eliminated.  The dominant
memory tensor is cidf32 (f32 [1,1,20,20] = 1600 B); it is irreducible because a
float decode of the whole 20x20 grid is required for the AveragePool box-finder.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

from ng_utils_shim import GRID_SHAPE

F = TP.FLOAT
F16 = TP.FLOAT16
I32 = TP.INT32
I64 = TP.INT64
U8 = TP.UINT8
BOOL = TP.BOOL


class G:
    def __init__(self):
        self.nodes, self.inits, self.vis, self._k = [], [], [], 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def c(self, arr, dt):
        a = np.asarray(arr)
        n = self.nm("c")
        if dt in (I64,):
            vals = a.astype(np.int64).ravel().tolist()
        elif dt == I32:
            vals = a.astype(np.int32).ravel().tolist()
        elif dt == U8:
            vals = a.astype(np.uint8).ravel().tolist()
        elif dt == F16:
            vals = a.astype(np.float16).ravel().tolist()
        else:
            vals = a.astype(np.float64).ravel().tolist()
        self.inits.append(oh.make_tensor(n, dt, list(a.shape) if a.shape else [], vals))
        return n

    def nd(self, op, ins, out=None, **kw):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **kw))
        return out

    def vi(self, name, dt, shape):
        self.vis.append(oh.make_tensor_value_info(name, dt, shape))

    def model(self, name, opset=18):
        x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
        y = oh.make_tensor_value_info("output", BOOL, GRID_SHAPE)
        used = {i for n in self.nodes for i in n.input}
        inits = [t for t in self.inits if t.name in used]
        g = oh.make_graph(self.nodes, name, [x], [y], inits)
        g.value_info.extend(self.vis)
        m = oh.make_model(g, ir_version=10, opset_imports=[oh.make_opsetid("", opset)])
        onnx.checker.check_model(m, full_check=True)
        return m


# --------------------------------------------------------------------------- #
# numpy reference (gates emission)                                            #
# --------------------------------------------------------------------------- #
def _ref(a):
    a = np.asarray(a, int)
    H, W = a.shape
    if H != 20 or W != 20:
        return None
    cf = a.astype(float)
    rscore = np.array([cf[i:i + 7, :].mean() for i in range(H - 6)])
    boxrow = int(np.argmax(rscore))
    cscore = np.array([cf[:, j:j + 7].mean() for j in range(W - 6)])
    boxcol = int(np.argmax(cscore))
    rt, ct = boxrow + 1, boxcol + 1
    win = a[rt:rt + 5, ct:ct + 5]
    tmask = (win > 0).astype(int)
    cnt = int(tmask.sum())
    marker = int(win[2, 2])
    c1 = (a == 1).astype(int)
    K = 2 * tmask - 1                       # +1 on shape, -1 elsewhere
    Cp = np.pad(c1, 2)
    S = np.zeros((H, W), int)
    for u in range(5):
        for v in range(5):
            S += K[u, v] * Cp[u:u + H, v:v + W]
    hits = (S == cnt).astype(int)           # exact-match centres
    Wf = tmask[::-1, ::-1]
    Hp = np.pad(hits, 2)
    paint = np.zeros((H, W), int)
    for u in range(5):
        for v in range(5):
            paint += Wf[u, v] * Hp[u:u + H, v:v + W]
    out = a.copy()
    out[paint > 0] = marker
    return out


# --------------------------------------------------------------------------- #
# onnx builder                                                                #
# --------------------------------------------------------------------------- #
def _build():
    g = G()

    # --- decode one-hot -> colour-id map, cropped to 20x20 (2x2 dil-10 conv) ---
    cw = np.zeros((1, 10, 2, 2), np.float32)
    cw[0, :, 0, 0] = np.arange(10)
    cwn = g.c(cw, F)
    cidf = g.nd("Conv", ["input", cwn], "cidf",
                dilations=[10, 10], kernel_shape=[2, 2], pads=[0, 0, 0, 0])
    cid = g.nd("Cast", [cidf], "cid", to=U8)                     # u8 [1,1,20,20]

    # --- locate the 7x7 gray box via 7-wide band averages ---
    rscore = g.nd("AveragePool", [cidf], kernel_shape=[7, 20])   # [1,1,14,1]
    cscore = g.nd("AveragePool", [cidf], kernel_shape=[20, 7])   # [1,1,1,14]
    ri = g.nd("ArgMax", [rscore], axis=2, keepdims=0)
    ci = g.nd("ArgMax", [cscore], axis=3, keepdims=0)
    ri32 = g.nd("Cast", [ri], to=I32)
    ci32 = g.nd("Cast", [ci], to=I32)
    sc1 = g.c([1], I64)
    rr = g.nd("Reshape", [ri32, sc1])
    cc = g.nd("Reshape", [ci32, sc1])
    one_i32 = g.c([1], I32)
    start_r = g.nd("Add", [rr, one_i32])
    start_c = g.nd("Add", [cc, one_i32])
    starts = g.nd("Concat", [start_r, start_c], "starts", axis=0)
    five = g.c([5, 5], I32)
    ends = g.nd("Add", [starts, five], "ends")
    axes23 = g.c([2, 3], I32)
    g.vi("starts", I32, [2])
    g.vi("ends", I32, [2])

    # --- extract the 5x5 marker window ---
    win = g.nd("Slice", [cid, starts, ends, axes23], "win")       # u8 [1,1,5,5]
    g.vi("win", U8, [1, 1, 5, 5])
    z_u8 = g.c(0, U8)
    one_u8 = g.c(1, U8)
    tmask_b = g.nd("Greater", [win, z_u8])                        # bool [1,1,5,5]
    tmask_u8 = g.nd("Cast", [tmask_b], "tmask_u8", to=U8)
    enc_s = g.c([2, 2], I32)
    enc_e = g.c([3, 3], I32)
    enc = g.nd("Slice", [win, enc_s, enc_e, axes23], "enc")       # u8 [1,1,1,1] = marker
    g.vi("enc", U8, [1, 1, 1, 1])

    # --- signed match kernel + runtime bias (1 - cnt) ---
    kern = g.nd("Add", [tmask_u8, tmask_u8], "kern")             # {0,2}; w_zp=1 -> -1/+1
    tmask_f16 = g.nd("Cast", [tmask_b], to=F16)
    red_axes = g.c([2, 3], I64)
    cnt_f16 = g.nd("ReduceSum", [tmask_f16, red_axes], keepdims=1)  # f16 [1,1,1,1]
    one_f16 = g.c([1.0], F16)
    bias_f16 = g.nd("Sub", [one_f16, cnt_f16])                    # 1 - cnt
    bias_i32 = g.nd("Cast", [bias_f16], to=I32)
    bshape = g.c([1], I64)
    bias = g.nd("Reshape", [bias_i32, bshape])                    # int32 [1]

    # --- colour-1 mask ---
    c1_b = g.nd("Equal", [cid, one_u8])
    c1_u8 = g.nd("Cast", [c1_b], "c1_u8", to=U8)

    # --- BINARY exact-match centres in one QLinearConv (arsenal #1) ---
    xsc = g.c(1.0, F)
    hits = g.nd("QLinearConv",
                [c1_u8, xsc, z_u8, kern, xsc, one_u8, xsc, z_u8, bias],
                "hits", kernel_shape=[5, 5], pads=[2, 2, 2, 2])   # u8 {0,1}

    # --- ADJOINT PAINT: stamp the shape back with the 180deg-flipped kernel ---
    fs = g.c([4, 4], I32)
    fe = g.c([-6, -6], I32)
    fstep = g.c([-1, -1], I32)
    wflip = g.nd("Slice", [tmask_u8, fs, fe, axes23, fstep], "wflip")
    g.vi("wflip", U8, [1, 1, 5, 5])
    paint = g.nd("QLinearConv",
                 [hits, xsc, z_u8, wflip, xsc, z_u8, xsc, z_u8],
                 "paint", kernel_shape=[5, 5], pads=[2, 2, 2, 2])  # u8 {0,1}
    paint_b = g.nd("Greater", [paint, z_u8])

    # --- recolour + one-hot output ---
    out_id = g.nd("Where", [paint_b, enc, cid], "out_id")         # u8 [1,1,20,20]
    pad255 = g.c(255, U8)
    pads = g.c([0, 0, 10, 10], I64)
    out_pad = g.nd("Pad", [out_id, pads, pad255, red_axes], "out_pad", mode="constant")
    chvals = g.c(np.arange(10).reshape(1, 10, 1, 1), U8)
    g.nd("Equal", [out_pad, chvals], "output")
    return g.model("t2_182")


# --------------------------------------------------------------------------- #
def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return []
            if max(a.shape) > 30 or max(b.shape) > 30:
                return []
            out.append((a, b))
    return out


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    for a, b in prs:
        o = _ref(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return
    try:
        yield ("t2_182", _build())
    except Exception:
        return
