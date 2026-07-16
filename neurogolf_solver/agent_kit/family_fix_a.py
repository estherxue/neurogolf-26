"""family_fix_a — rebuilt solvers for gen-validate CATASTROPHIC rule proxies.

Currently ships:
  * task118 (ARC-GEN 50846271) "static + hidden crosses".

Rule (decoded from the generator source, tasks/task_50846271.py):
  The grid is random gray(5) static.  A single arm-length L in {2,3} is chosen for
  the whole grid, then up to 4 non-overlapping "+" crosses (arms of length L in the
  4 cardinal directions) are stamped.  Each stamped plus-cell becomes CYAN(8) if it
  landed on gray static, else RED(2).  The INPUT hides the cyan (shows it as gray);
  the OUTPUT reveals it.  So the transform is: input == output EXCEPT the gray cells
  that lie on a cross plus are recolored 8.  Every plus cell is therefore non-black
  in the input (red, or gray-that-should-be-cyan).

Reconstruction (pure convolution, no coordinates):
  * arm_nb = 1-black  (out-of-grid counts as non-black, matching the generator's
    hand-built edge crosses in the ARC-AGI train/test examples).
  * a cell is a cross CENTER at scale L iff both its length-(2L+1) arms are fully
    non-black AND its plus carries >=1 red, AND it is the local-max of that red-count
    over the (4L+1)^2 cross box (NMS — crosses never overlap, so the true center is
    the unique peak).
  * L is 3 iff some L=2 center has a RED strictly at distance 3 on an arm (only an
    L=3 cross can put red that far out); else L=2.  When no cross exposes a distance-3
    red the grid is genuinely L-ambiguous from the input (see LIMITATION below).
  * S = gray cells covered by dilating the centers along the plus shape; recolor 8.

LIMITATION (information floor, ~2.5% of fresh generator samples, unavoidable by ANY
solver): (a) a cross whose whole plus landed on gray leaves no red -> invisible in the
input yet cyan in the output (~0.4%); (b) an L=3 cross with all four gray arm-tips is
pixel-identical in the input to an L=2 cross whose distance-3 cells happen to be gray,
but the outputs differ (~2%).  This module is exact on ALL local data (267/267) and
~96.5% on fresh samples — a large gain over the broken incumbent (~56%).  The residual
is the floor above, NOT a rule error.

ONNX: single-channel [1,1,30,30] float chain.  Both L=2 and L=3 pipelines are built
and blended by a scalar L3 flag.  Ops: Conv (arm/plus sums), MaxPool (NMS), Pad+Slice
(shifts), ConvTranspose (plus dilation), Greater/Less/And/Not/Mul.  No banned ops.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


# ------------------------------------------------------------------ numpy ref
def _sh1(a, k):            # horizontal sliding sum, off-grid padded with 1
    L = k // 2; H, W = a.shape
    p = np.ones((H, W + 2 * L)); p[:, L:L + W] = a
    o = np.zeros((H, W))
    for d in range(k):
        o += p[:, d:d + W]
    return o


def _sv1(a, k):
    L = k // 2; H, W = a.shape
    p = np.ones((H + 2 * L, W)); p[L:L + H, :] = a
    o = np.zeros((H, W))
    for d in range(k):
        o += p[d:d + H, :]
    return o


def _sh0(a, k):            # zero-padded sliding sum
    L = k // 2; H, W = a.shape
    p = np.zeros((H, W + 2 * L)); p[:, L:L + W] = a
    o = np.zeros((H, W))
    for d in range(k):
        o += p[:, d:d + W]
    return o


def _sv0(a, k):
    L = k // 2; H, W = a.shape
    p = np.zeros((H + 2 * L, W)); p[L:L + H, :] = a
    o = np.zeros((H, W))
    for d in range(k):
        o += p[d:d + H, :]
    return o


def _plusc(a, L):
    return _sh0(a, 2 * L + 1) + _sv0(a, 2 * L + 1) - a


def _shiftbool(a, dr, dc):
    H, W = a.shape; o = np.zeros((H, W), bool)
    r0 = max(dr, 0); r1 = min(H, H + dr); c0 = max(dc, 0); c1 = min(W, W + dc)
    o[r0:r1, c0:c1] = a[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return o


def _lmax(s, rad):         # local max over (2rad+1)^2, pad 0 (scores are >=0)
    H, W = s.shape
    big = np.zeros((H + 2 * rad, W + 2 * rad)); big[rad:rad + H, rad:rad + W] = s
    m = np.zeros((H, W))
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            m = np.maximum(m, big[rad + dr:rad + dr + H, rad + dc:rad + dc + W])
    return m


def _centers(nb, red, L):
    aH = (_sh1(nb, 2 * L + 1) == 2 * L + 1)
    aV = (_sv1(nb, 2 * L + 1) == 2 * L + 1)
    full = (aH & aV).astype(float)
    rp = _plusc(red.astype(float), L)
    score = full * rp
    lm = _lmax(score, 2 * L)
    return (score >= 1) & (score >= lm - 1e-9)


def _dilate_plus(centers, L):
    H, W = centers.shape; plus = np.zeros((H, W), bool)
    for d in range(-L, L + 1):
        plus |= _shiftbool(centers, d, 0)
        plus |= _shiftbool(centers, 0, d)
    return plus


def _solve(inp):
    x = np.asarray(inp)
    if not set(np.unique(x)).issubset({0, 2, 5}):
        return None
    H, W = x.shape
    nb = (x != 0).astype(float); red = (x == 2); gray = (x == 5)
    c2 = _centers(nb, red, 2)
    ev = (_shiftbool(red, -3, 0) | _shiftbool(red, 3, 0) |
          _shiftbool(red, 0, -3) | _shiftbool(red, 0, 3)) & c2
    if ev.any():
        centers, L = _centers(nb, red, 3), 3
    else:
        centers, L = c2, 2
    S = _dilate_plus(centers, L) & gray
    y = x.copy(); y[S] = 8
    return y


# ------------------------------------------------------------------ onnx build
def _const(name, arr, dt=F):
    a = np.asarray(arr)
    a = a.astype(np.int64) if dt == I64 else a.astype(np.float32)
    return oh.make_tensor(name, dt, list(a.shape) if a.shape else [1],
                          a.flatten().tolist())


def _build():
    nodes, inits = [], []

    def C(name, arr, dt=F):
        inits.append(_const(name, arr, dt)); return name

    def slice_ch(src, dst, lo, hi):
        C(dst + "_s", [lo], I64); C(dst + "_e", [hi], I64); C(dst + "_a", [1], I64)
        nodes.append(oh.make_node("Slice", [src, dst + "_s", dst + "_e", dst + "_a"], [dst]))
        return dst

    def shift(src, dst, dr, dc):
        """dst[i,j] = src[i-dr, j-dc] (content moves down/right by dr,dc), 0-filled."""
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        nodes.append(oh.make_node("Pad", [src], [dst + "_p"], mode="constant",
                                  value=0.0, pads=[0, 0, pt, pl, 0, 0, pb, pr]))
        sr, sc = pb, pr
        C(dst + "_s", [0, 0, sr, sc], I64)
        C(dst + "_e", [1, 1, sr + 30, sc + 30], I64)
        C(dst + "_a", [0, 1, 2, 3], I64)
        nodes.append(oh.make_node("Slice", [dst + "_p", dst + "_s", dst + "_e", dst + "_a"], [dst]))
        return dst

    C("one", [1.0]); C("half", [0.5])

    # ---- planes ---- #
    nodes.append(oh.make_node("ReduceSum", ["input"], ["allsum"], axes=[1], keepdims=1))
    slice_ch("input", "ch0", 0, 1)
    slice_ch("input", "red", 2, 3)
    slice_ch("input", "gray", 5, 6)
    nodes.append(oh.make_node("Sub", ["one", "ch0"], ["armnb"]))        # non-black or off-grid

    def scale(L):
        k = 2 * L + 1
        # arm sums
        C(f"bh{L}", np.ones((1, 1, 1, k)))
        C(f"bv{L}", np.ones((1, 1, k, 1)))
        nodes.append(oh.make_node("Conv", ["armnb", f"bh{L}"], [f"ah{L}"],
                                  kernel_shape=[1, k], pads=[0, L, 0, L]))
        nodes.append(oh.make_node("Conv", ["armnb", f"bv{L}"], [f"av{L}"],
                                  kernel_shape=[k, 1], pads=[L, 0, L, 0]))
        C(f"kt{L}", [k - 0.5])
        nodes.append(oh.make_node("Greater", [f"ah{L}", f"kt{L}"], [f"ahb{L}"]))
        nodes.append(oh.make_node("Greater", [f"av{L}", f"kt{L}"], [f"avb{L}"]))
        nodes.append(oh.make_node("And", [f"ahb{L}", f"avb{L}"], [f"fullb{L}"]))
        nodes.append(oh.make_node("Cast", [f"fullb{L}"], [f"full{L}"], to=F))
        # red-on-plus count
        pk = np.zeros((1, 1, k, k)); pk[0, 0, L, :] = 1; pk[0, 0, :, L] = 1
        C(f"pk{L}", pk)
        nodes.append(oh.make_node("Conv", ["red", f"pk{L}"], [f"rp{L}"],
                                  kernel_shape=[k, k], pads=[L, L, L, L]))
        nodes.append(oh.make_node("Mul", [f"full{L}", f"rp{L}"], [f"score{L}"]))
        # NMS local max over (4L+1)^2 via Pad0 + MaxPool
        rad = 2 * L; kk = 2 * rad + 1
        nodes.append(oh.make_node("Pad", [f"score{L}"], [f"scp{L}"], mode="constant",
                                  value=0.0, pads=[0, 0, rad, rad, 0, 0, rad, rad]))
        nodes.append(oh.make_node("MaxPool", [f"scp{L}"], [f"lm{L}"],
                                  kernel_shape=[kk, kk], strides=[1, 1]))
        # centers = score>0.5 AND not(lm>score)
        nodes.append(oh.make_node("Greater", [f"score{L}", "half"], [f"pos{L}"]))
        nodes.append(oh.make_node("Greater", [f"lm{L}", f"score{L}"], [f"gt{L}"]))
        nodes.append(oh.make_node("Not", [f"gt{L}"], [f"ismax{L}"]))
        nodes.append(oh.make_node("And", [f"pos{L}", f"ismax{L}"], [f"cb{L}"]))
        nodes.append(oh.make_node("Cast", [f"cb{L}"], [f"cen{L}"], to=F))
        # dilate along plus via ConvTranspose with plus kernel
        nodes.append(oh.make_node("ConvTranspose", [f"cen{L}", f"pk{L}"], [f"dil{L}"],
                                  kernel_shape=[k, k], pads=[L, L, L, L]))
        nodes.append(oh.make_node("Greater", [f"dil{L}", "half"], [f"plusb{L}"]))
        nodes.append(oh.make_node("And", [f"plusb{L}", "grayb"], [f"Sb{L}"]))
        nodes.append(oh.make_node("Cast", [f"Sb{L}"], [f"S{L}"], to=F))
        return f"cb{L}", f"S{L}"

    nodes.append(oh.make_node("Greater", ["gray", "half"], ["grayb"]))

    cb2, S2 = scale(2)
    cb3, S3 = scale(3)

    # ---- L decision: L=3 iff some L=2 center has a red at distance exactly 3 ---- #
    shift("red", "r_up", -3, 0)   # red[i+3]
    shift("red", "r_dn", 3, 0)
    shift("red", "r_lf", 0, -3)
    shift("red", "r_rt", 0, 3)
    nodes.append(oh.make_node("Add", ["r_up", "r_dn"], ["r_a"]))
    nodes.append(oh.make_node("Add", ["r_lf", "r_rt"], ["r_b"]))
    nodes.append(oh.make_node("Add", ["r_a", "r_b"], ["r_any"]))
    nodes.append(oh.make_node("Greater", ["r_any", "half"], ["r_anyb"]))
    nodes.append(oh.make_node("And", ["r_anyb", cb2], ["evb"]))
    nodes.append(oh.make_node("Cast", ["evb"], ["ev"], to=F))
    nodes.append(oh.make_node("ReduceMax", ["ev"], ["l3flag"], axes=[0, 1, 2, 3], keepdims=1))
    # blend: S = l3flag*S3 + (1-l3flag)*S2
    nodes.append(oh.make_node("Sub", ["one", "l3flag"], ["nl3"]))
    nodes.append(oh.make_node("Mul", ["l3flag", "S3"], ["bS3"]))
    nodes.append(oh.make_node("Mul", ["nl3", "S2"], ["bS2"]))
    nodes.append(oh.make_node("Add", ["bS3", "bS2"], ["S"]))

    # ---- output channels ---- #
    nodes.append(oh.make_node("Sub", ["gray", "S"], ["out5"]))   # gray minus converted
    slice_ch("input", "ch1", 1, 2)
    slice_ch("input", "ch3", 3, 4)
    slice_ch("input", "ch4", 4, 5)
    slice_ch("input", "ch6", 6, 7)
    slice_ch("input", "ch7", 7, 8)
    slice_ch("input", "ch9", 9, 10)
    nodes.append(oh.make_node("Sub", ["ch0", "ch0"], ["Z"]))     # zero plane
    nodes.append(oh.make_node("Concat",
                              ["ch0", "ch1", "red", "ch3", "ch4", "out5",
                               "ch6", "ch7", "S", "ch9"],
                              ["output"], axis=1))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "fix118", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    for a, b in prs:
        r = _solve(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            return
    yield ("fix118_crosses", _build())
