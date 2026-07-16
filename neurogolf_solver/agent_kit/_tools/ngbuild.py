"""ngbuild — shared ONNX construction primitives for NeuroGolf rebuilds.

The recurring graph idioms that every rebuild (e017/e191/e319/ed209b/...) kept
re-inventing, extracted as tested, grader-proven primitives. Each is a lesson:

  decode_head   Conv(input,[0..9]) -> value grid (the only unavoidable f32 3600B)
  onehot_tail   value plane -> Pad -> Equal[0..9] -> FREE bool output (hourglass)
  qdetect       QLinearConv exact-match with saturating bias -> BINARY u8 (e191:
                bias=1-total makes an exact match score 1, everything else -> 0)
  qpaint        QLinearConv adjoint paint (stamp a footprint at each anchor)
  bbox_coords   ArgMax-free bounding box: coordinates-as-values + ReduceMax/Min
  u8_reduce     MaxPool-as-reducer (ORT has no u8 ReduceMax/ReduceSum kernel)
  dir_reach     directional MaxPool -> distance-to-wall / boundary prefix (t145)
  finalize      make_graph+model, clamp ir<=10, checker

All ops are on the grader-proven path (Conv/QLinearConv/MaxPool/Equal/Where/Pad/
Slice/Cast/ReduceMax/Min). NO u8 Min/Max/CumSum/ReduceSum, NO i64 ArgMax/Gather
chains, NO int32 arithmetic in the emitted graph.

Usage:
    from ngbuild import G, decode_head, onehot_tail, qdetect, finalize
    g = G(S=21)                       # working canvas S (crop the generator bound)
    v = decode_head(g, crop=True)     # -> u8 value grid [1,1,S,S]
    ... your logic on u8 planes ...
    onehot_tail(g, value_plane)       # writes 'output'
    m = finalize(g)
"""
import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

F32, F16, U8, I8, I32, I64, BOOL = (TP.FLOAT, TP.FLOAT16, TP.UINT8, TP.INT8,
                                    TP.INT32, TP.INT64, TP.BOOL)


class G:
    """Graph accumulator. Pre-registers the QLinearConv scalars + a [0..9] color
    table + a decode kernel, since nearly every rebuild needs them."""

    def __init__(self, S=30, opset=17, ir=8):
        self.nodes, self.inits, self._k = [], [], 0
        self.S = S
        self.opset = opset
        self.ir = ir
        # ubiquitous constants
        self.init("dec_w", F32, [1, 10, 1, 1], [float(i) for i in range(10)])
        self.init("col_idx", U8, [1, 10, 1, 1], list(range(10)))
        self.init("zero_u8", U8, [], [0]); self.init("one_u8", U8, [], [1])
        self.init("ten_u8", U8, [], [10])
        # QLinearConv scalars (scale 1, zero-point 0) — the "integer conv" setup
        self.init("qxs", F32, [], [1.0]); self.init("qxz", U8, [], [0])
        self.init("qws", F32, [], [1.0]); self.init("qwz", I8, [], [0])
        self.init("qys", F32, [], [1.0]); self.init("qyz", U8, [], [0])

    def nm(self, p="t"):
        self._k += 1
        return "%s_%d" % (p, self._k)

    def init(self, name, dt, dims, vals):
        if isinstance(vals, np.ndarray):
            vals = vals.reshape(-1).tolist()
        self.inits.append(oh.make_tensor(name, dt, list(dims), list(vals)))
        return name

    def nd(self, op, ins, out=None, **attr):
        out = out or self.nm(op.lower())
        self.nodes.append(oh.make_node(op, list(ins), [out], **attr))
        return out


# --------------------------------------------------------------------------- #
# heads & tails                                                               #
# --------------------------------------------------------------------------- #
def decode_head(g, crop=True):
    """input one-hot [1,10,30,30] -> value grid [1,1,S,S] uint8.
    Conv with weights [0..9] collapses the one-hot channel to its color index.
    The 3600B f32 Conv output is the one unavoidable big tensor; Cast to u8 asap."""
    v30f = g.nd("Conv", ["input", "dec_w"], "v30f")           # [1,1,30,30] f32 3600B
    v30 = g.nd("Cast", [v30f], "v30u", to=U8)                 # u8 900B
    if crop and g.S < 30:
        g.init("_crop_s", I64, [2], [0, 0])
        g.init("_crop_e", I64, [2], [g.S, g.S])
        g.init("_crop_ax", I64, [2], [2, 3])
        return g.nd("Slice", [v30, "_crop_s", "_crop_e", "_crop_ax"], "v_u8")
    return v30


def onehot_tail(g, value_plane, out="output"):
    """value plane [1,1,S,S] u8 -> Pad to 30x30 with sentinel 10 -> Equal[0..9]
    -> FREE bool output [1,10,30,30]. The hourglass exit: a 900B->9000B expansion
    on the free boundary. Sentinel 10 matches no channel -> padding decodes to
    all-zero (background)."""
    pad = 30 - g.S
    if pad > 0:
        g.init("_pad30", I64, [8], [0, 0, 0, 0, 0, 0, pad, pad])
        c30 = g.nd("Pad", [value_plane, "_pad30", "ten_u8"], "c30")
    else:
        c30 = value_plane
    g.nodes.append(oh.make_node("Equal", [c30, "col_idx"], [out]))
    return out


# --------------------------------------------------------------------------- #
# runtime-weight QLinearConv detect / paint (e191)                            #
# --------------------------------------------------------------------------- #
def qdetect(g, signal_u8, weight_i8, kernel_shape, pads, bias_i32=None,
            group=1, strides=None, name=None):
    """Exact-match binary detector. QLinearConv(signal, weight, bias) with u8
    saturation: choose weight so an EXACT match scores `total` and bias=(1-total)
    maps it to exactly 1; any partial/violating anchor scores <=0 -> saturates to 0.
    Returns a BINARY u8 map — no Equal/Cast needed. `weight_i8` and `bias_i32` may
    be runtime tensor NAMES (not just constants) — that's the key trick."""
    ins = [signal_u8, "qxs", "qxz", weight_i8, "qws", "qwz", "qys", "qyz"]
    if bias_i32 is not None:
        ins.append(bias_i32)
    kw = dict(group=group, kernel_shape=list(kernel_shape), pads=list(pads),
              strides=list(strides or [1, 1]))
    return g.nd("QLinearConv", ins, name or g.nm("detect"), **kw)


def qpaint(g, anchor_u8, footprint_i8, kernel_shape, pads, group=1, name=None):
    """Adjoint paint: QLinearConv sums each anchor's footprint into a coverage map
    (the Conv adjoint of ConvTranspose). footprint should be the flipped stamp."""
    ins = [anchor_u8, "qxs", "qxz", footprint_i8, "qws", "qwz", "qys", "qyz"]
    return g.nd("QLinearConv", ins, name or g.nm("paint"),
                group=group, kernel_shape=list(kernel_shape), pads=list(pads),
                strides=[1, 1])


# --------------------------------------------------------------------------- #
# ArgMax-free bounding box (t145 / ed209b coordinates-as-values)              #
# --------------------------------------------------------------------------- #
def bbox_coords(g, mask_bool, prefix=None):
    """Bounding box of the True region of a [1,1,S,S] bool mask, WITHOUT ArgMax
    or i64 (both are grader-fragile in chains). Uses coordinates-as-values:
      rowmax = ReduceMax over rows of (rowidx if any-in-row else 0)
      rowmin = S-1 - ReduceMax of ((S-1-rowidx) if any-in-row else 0)
    Returns f16 scalar names dict(r0,r1,c0,c1) each [1,1,1,1]. Compose rectangles
    downstream by comparing against a positional grid (posgrid_row/col below)."""
    p = prefix or g.nm("bb")
    S = g.S
    g.init(p + "_ridx", F16, [1, 1, S, 1], [float(i) for i in range(S)])
    g.init(p + "_cidx", F16, [1, 1, 1, S], [float(i) for i in range(S)])
    g.init(p + "_ridxr", F16, [1, 1, S, 1], [float(S - 1 - i) for i in range(S)])
    g.init(p + "_cidxr", F16, [1, 1, 1, S], [float(S - 1 - i) for i in range(S)])
    g.init(p + "_smax", F16, [], [float(S - 1)])
    mf = g.nd("Cast", [mask_bool], p + "_mf", to=F16)
    rowhas = g.nd("ReduceMax", [mf], p + "_rowhas", axes=[3], keepdims=1)   # [1,1,S,1]
    colhas = g.nd("ReduceMax", [mf], p + "_colhas", axes=[2], keepdims=1)   # [1,1,1,S]
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rowhas, p + "_ridx"], p + "_rs"), ],
              p + "_r1", axes=[2], keepdims=1)
    c1 = g.nd("ReduceMax", [g.nd("Mul", [colhas, p + "_cidx"], p + "_cs")],
              p + "_c1", axes=[3], keepdims=1)
    r0 = g.nd("Sub", [p + "_smax", g.nd("ReduceMax", [g.nd("Mul", [rowhas, p + "_ridxr"], p + "_rsr")],
              p + "_r0r", axes=[2], keepdims=1)], p + "_r0")
    c0 = g.nd("Sub", [p + "_smax", g.nd("ReduceMax", [g.nd("Mul", [colhas, p + "_cidxr"], p + "_csr")],
              p + "_c0r", axes=[3], keepdims=1)], p + "_c0")
    return dict(r0=r0, r1=r1, c0=c0, c1=c1)


def posgrid(g, prefix="pos"):
    """Register [1,1,S,S] f16 row/col position grids for rectangle composition:
    a rect [r0..r1]x[c0..c1] = And(row>=r0, row<=r1, col>=c0, col<=c1)."""
    S = g.S
    if prefix + "_row" not in [i.name for i in g.inits]:
        g.init(prefix + "_row", F16, [1, 1, S, S],
               np.repeat(np.arange(S, dtype=np.float16)[:, None], S, 1))
        g.init(prefix + "_col", F16, [1, 1, S, S],
               np.repeat(np.arange(S, dtype=np.float16)[None, :], S, 0))
    return prefix + "_row", prefix + "_col"


# --------------------------------------------------------------------------- #
# u8 reduction & directional reach                                            #
# --------------------------------------------------------------------------- #
def u8_reduce_max(g, plane_u8, spatial, name=None):
    """u8 ReduceMax replacement (ORT has no u8 ReduceMax kernel): a full-window
    MaxPool -> [1,C,1,1]. `spatial` = (H,W) of the plane."""
    return g.nd("MaxPool", [plane_u8], name or g.nm("umax"),
                kernel_shape=list(spatial), strides=[1, 1])


def dir_reach(g, src_u8, direction, reach, name=None):
    """Directional MaxPool: for direction in {L,R,U,D}, propagate the max of src
    along that axis over `reach` cells (pads<kernel enforced). Used for wall-
    distance (t145) and boundary prefix-runs. src should carry position-indices
    on walls / 0 elsewhere (or 1/0 for reachability)."""
    S = g.S
    k = reach
    if direction in ("L", "R"):
        ks = [1, k]
        pads = [0, k - 1, 0, 0] if direction == "L" else [0, 0, 0, k - 1]
    else:
        ks = [k, 1]
        pads = [k - 1, 0, 0, 0] if direction == "U" else [0, 0, k - 1, 0]
    return g.nd("MaxPool", [src_u8], name or g.nm("reach_" + direction),
                kernel_shape=ks, pads=pads, strides=[1, 1])


# --------------------------------------------------------------------------- #
# finalize                                                                    #
# --------------------------------------------------------------------------- #
def finalize(g, name="rebuild", out_dtype=BOOL, check=True):
    graph = oh.make_graph(
        g.nodes, name,
        [oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [oh.make_tensor_value_info("output", out_dtype, [1, 10, 30, 30])],
        g.inits)
    m = oh.make_model(graph, ir_version=g.ir,
                      opset_imports=[oh.make_opsetid("", g.opset)])
    if g.ir > 10:
        m.ir_version = 10
    if check:
        onnx.checker.check_model(m, full_check=True)
    return m
