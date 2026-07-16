"""family_bpk005b — value-exact memory-golf recompile of family_bpk005 (task005,
ARC 045e512c "rays").

Same algorithm as bpk005 (see that module), re-encoded cheaper.  Output is
byte-identical (as bool, i.e. after >0) to the bpk005 graph on every input:

  * T-ENCODING HEAD: one 1x1 Conv with weights [1..10] gives T=(V+1)*ingrid in a
    single f32 tensor (padding ring is all-zero one-hot, so T=0 there).  This
    replaces the separate V-conv + ReduceSum(ingrid) f32 pair (7200B -> 3600B):
      - nonbg mask N  = (T > 1.5)          (== V>0 exactly)
      - ingrid        = Clip(T, 0, 1)
      - marker colour = Clip(MaxPool3(T), min=1) - 1   (== MaxPool3(V) exactly:
        an all-padding sample box gives 0->1->0, an in-grid box gives maxV+1-1)
  * NO 8-CHANNEL RAY TENSOR: bpk005's MD=Conv(M,w8[8,1,9,9]) (14400B) + Einsum
    with cd (1800B) is linear in cd, so instead the coloured 9x9 kernel is built
    at runtime: wc[i,j] = cd9[dirmap[i,j]] via GatherND (dirmap maps each of the
    32 dilation-4 taps to its direction, everything else to the appended 0), and
    ONE single-channel Conv(M, wc, bias=1) gives R+1 directly (bias folds the +1
    that bpk005 applied after the Max).  Tap sets of the 8 directions are
    disjoint, so wc == sum_d cd[d]*w8[d] exactly; Conv is linear, all values are
    small integers, hence f16-exact equality with MD@cd.
  * corner coordinates r0,c0 via two tiny Einsums ('nchw,h->n'/'nchw,w->n')
    instead of the marginal ReduceSum/Mul chains; position math stays in f16
    (exact for ints <= 33), no f32 round-trip.
  * tail: out = Equal(Max(T, R+1) * Clip(T,0,1), [1..10]).  In-grid this is
    Equal(max(V,R)+1, ...) as in bpk005; on the padding ring the ingrid factor
    zeroes it.  Value-identical to bpk005's (Max(V,R)+1)*ingrid.

45848B/774p (14.25pt) -> ~25990B/~208p (~14.83pt).  Detection is behavioural
and unchanged, so it only ever fires on this task.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, numpy_helper as nh

from ng_utils_shim import GRID_SHAPE, IR_VERSION

FP32 = onnx.TensorProto.FLOAT
FP16 = onnx.TensorProto.FLOAT16
BOOL = onnx.TensorProto.BOOL
INT64 = onnx.TensorProto.INT64
H = 30
OPSET = [oh.make_opsetid("", 12)]

_DIRS8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


# --------------------------------------------------------------------------- #
# graph accumulator                                                           #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init(self, arr, dtype):
        n = self.nm("c")
        self.inits.append(nh.from_array(np.asarray(arr, dtype), n))
        return n

    def f16(self, arr):
        return self.init(arr, np.float16)

    def f32(self, arr):
        return self.init(arr, np.float32)

    def i64(self, arr):
        return self.init(arr, np.int64)

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _np_005(a):
    """Reference semantics (mirror of the ONNX numerics) — unchanged."""
    a = np.asarray(a, int)
    Hd, Wd = a.shape
    if Hd > 30 or Wd > 30 or Hd < 4 or Wd < 4:
        return None
    N = (a > 0).astype(int)
    Np = np.pad(N, ((1, 3), (1, 3)))
    S = np.zeros((Hd, Wd), int)
    for u in range(5):
        for v in range(5):
            wgt = 1 if (1 <= u <= 3 and 1 <= v <= 3) else -3
            S += wgt * Np[u:u + Hd, v:v + Wd]
    mx = S.max()
    pos = np.argwhere(S == mx)
    if len(pos) != 1 or mx < 4:
        return None
    r0, c0 = map(int, pos[0])
    if r0 + 2 >= Hd or c0 + 2 >= Wd:
        return None
    B = np.zeros((Hd, Wd), int)
    B[r0:r0 + 3, c0:c0 + 3] = 1
    M = N * B
    R = np.zeros((Hd, Wd), int)
    for di, dj in _DIRS8:
        colors = []
        for u in range(3):
            for v in range(3):
                y, x = r0 + u + 4 * di, c0 + v + 4 * dj
                if 0 <= y < Hd and 0 <= x < Wd and a[y, x] > 0:
                    colors.append(int(a[y, x]))
        cd = max(colors) if colors else 0
        if cd == 0:
            continue
        for k in range(1, 8):
            for u in range(3):
                for v in range(3):
                    if M[r0 + u, c0 + v]:
                        y, x = r0 + u + 4 * k * di, c0 + v + 4 * k * dj
                        if 0 <= y < Hd and 0 <= x < Wd:
                            R[y, x] = cd
    return np.maximum(a, R)


def _build_005b():
    g = _G()

    # T = (V+1)*ingrid in one conv (padding one-hot is all-zero -> T=0 there) -- #
    wT = np.arange(1, 11, dtype=np.float32).reshape(1, 10, 1, 1)
    T32 = g.nd("Conv", ["input", g.f32(wT)], kernel_shape=[1, 1])
    Th = g.nd("Cast", [T32], to=FP16)

    # nonbg mask N = (T > 1.5)  (== V > 0) -------------------------------------- #
    c1p5 = g.f16(1.5)
    Nh = g.nd("Cast", [g.nd("Greater", [Th, c1p5])], to=FP16)

    # 5x5 score conv -> unique bbox corner (identical to bpk005) ---------------- #
    k5 = np.full((1, 1, 5, 5), -3.0, np.float16)
    k5[0, 0, 1:4, 1:4] = 1.0
    S = g.nd("Conv", [Nh, g.f16(k5)], kernel_shape=[5, 5], pads=[1, 1, 3, 3])
    mx = g.nd("ReduceMax", [S], axes=[2, 3], keepdims=1)
    half = g.f16(0.5)
    Ph = g.nd("Cast", [g.nd("Greater", [S, g.nd("Sub", [mx, half])])], to=FP16)

    # sprite mask M = N inside the 3x3 box (identical) --------------------------- #
    B = g.nd("Conv", [Ph, g.f16(np.ones((1, 1, 3, 3), np.float16))],
             kernel_shape=[3, 3], pads=[2, 2, 0, 0])
    M = g.nd("Mul", [Nh, B])

    # corner coords + 8 sample points (f16 throughout, exact for ints <= 33;
    # batch_dims=2 gathers keep the index chain at (h,w) pairs only) ------------ #
    rows = g.f16(np.arange(H, dtype=np.float16))
    r0 = g.nd("Einsum", [Ph, rows], equation="nchw,h->n")            # [1]
    c0 = g.nd("Einsum", [Ph, rows], equation="nchw,w->n")            # [1]
    base = g.nd("Concat", [r0, c0], axis=0)                          # [2]
    offs = np.array([[1 + 4 * di, 1 + 4 * dj] for (di, dj) in _DIRS8],
                    np.float16).reshape(1, 1, 8, 2)
    pos = g.nd("Add", [g.f16(offs), base])                           # [1,1,8,2]
    c0s, c1s = g.f16(0.0), g.f16(1.0)
    pos = g.nd("Clip", [pos, c0s, g.f16(float(H - 1))])
    idx = g.nd("Cast", [pos], to=INT64)

    # 8 marker colours: cd = Clip(MaxPool3(T)@idx, min=1) - 1 (== MaxPool3(V)) -- #
    Vmax = g.nd("MaxPool", [Th], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                strides=[1, 1])
    cdp = g.nd("GatherND", [Vmax, idx], batch_dims=2)                # [1,1,8]
    cd = g.nd("Sub", [g.nd("Clip", [cdp, c1s]), c1s])                # [1,1,8]

    # runtime-coloured 9x9 kernel: wc[i,j] = cd9[dirmap[i,j]] ------------------- #
    dirmap = np.full((9, 9), 8, np.int64)
    for d, (di, dj) in enumerate(_DIRS8):
        for k in range(1, 5):
            dirmap[4 - k * di, 4 - k * dj] = d
    z1 = g.f16([[[0.0]]])
    cd9 = g.nd("Concat", [cd, z1], axis=2)                           # [1,1,9]
    wcf = g.nd("GatherND", [cd9, g.i64(dirmap.reshape(1, 1, 81, 1))],
               batch_dims=2)                                         # [1,1,81]
    wc = g.nd("Reshape", [wcf, g.i64([1, 1, 9, 9])])

    # R+1 in one conv (bias folds the +1); tail identical in value to bpk005 ---- #
    Rp = g.nd("Conv", [M, wc, g.f16([1.0])], kernel_shape=[9, 9],
              dilations=[4, 4], pads=[16, 16, 16, 16])
    ingrid = g.nd("Clip", [Th, c0s, c1s])
    S2 = g.nd("Mul", [g.nd("Max", [Th, Rp]), ingrid])
    idxP = g.f16(np.arange(1, 11, dtype=np.float16).reshape(1, 10, 1, 1))
    g.nd("Equal", [S2, idxP], "output")                              # BOOL[1,10,30,30]

    x = oh.make_tensor_value_info("input", FP32, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", BOOL, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, "bpk005b", [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET)
    onnx.checker.check_model(m, full_check=True)
    return m


def _pairs(examples):
    out = []
    for split in ("train", "test"):
        for e in examples.get(split, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                return []
            out.append((a, b))
    return out


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    for a, b in prs:
        o = _np_005(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return
    try:
        yield ("bpk005b_rays045e512c", _build_005b())
    except Exception:
        return
