"""task277 (ARC b230c067) — "biggerisblue".

Rule decoded from the generator (task_b230c067.py): the input holds three well-separated
cyan(8) sprites — TWO identical "full" sprites and ONE "reduced" sprite that is a full
sprite with one column deleted (columns compacted). The two full sprites span a common
column-width W; the reduced sprite spans exactly W-1 columns (the full sprite touches every
column, so removing one always drops the span by 1). Recolour the two bigger/full sprites
BLUE (1) and the odd reduced one RED (2)  ("bigger is blue").

Discriminator (validated bit-exact on train+test+arc-gen, on the 7 known-fail inputs, and on
30000 fresh generator samples): per connected sprite, the bounding-box column-width. Both
full sprites achieve the unique global maximum width W; every part of the reduced sprite
spans <= W-1. So a cyan cell is BLUE iff its component's bbox width == global max width,
else RED.

Segmentation: the generator places boxes with bounding-box spacing >= 1 (overlaps(...,1)),
so distinct sprites are separated by a full empty band and never touch even 8-diagonally.
The full sprites can be only 8-connected (not 4-connected — see the generator's hand-authored
test), so 8-connectivity is required to keep each full sprite whole. The reduced sprite may
split into pieces, but every piece still spans <= W-1 columns, so all pieces stay RED.

ONNX (opset-10, static [1,10,30,30], float32 — the grader runs float32, all intermediates
lie in [-100, 9] << 2048): cyan = channel 8; 8-connected masked MAX-propagation of the
column index (Pad-with-NEG + MaxPool 3x3, 16 unrolled steps; components <=16 px so geodesic
<=15) yields the per-cell max column; the same on the negated column index yields the min
column; width = maxcol - mincol + 1; BLUE = width >= global-max, RED = cyan AND NOT blue;
background channel 0 passes through unchanged. No banned ops (only Slice/Sub/Mul/Add/Pad/
MaxPool/ReduceMax/Greater/Cast/Concat).
"""
import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64
STEPS = 16
NEG = -100.0
S = GRID_SHAPE[2]          # 30
ON_CHANNEL = 8


# --------------------------------------------------------------------------- #
# numpy reference (gate) — mirrors the ONNX op chain exactly                   #
# --------------------------------------------------------------------------- #
def _prop8(mask, val):
    """8-connected masked max-propagation of `val` over cyan `mask` (off-cells held at NEG)."""
    v = val.copy()
    for _ in range(STEPS):
        p = v.copy()
        n = np.full_like(v, NEG)
        n[1:, :] = np.maximum(n[1:, :], p[:-1, :]); n[:-1, :] = np.maximum(n[:-1, :], p[1:, :])
        n[:, 1:] = np.maximum(n[:, 1:], p[:, :-1]); n[:, :-1] = np.maximum(n[:, :-1], p[:, 1:])
        n[1:, 1:] = np.maximum(n[1:, 1:], p[:-1, :-1]); n[1:, :-1] = np.maximum(n[1:, :-1], p[:-1, 1:])
        n[:-1, 1:] = np.maximum(n[:-1, 1:], p[1:, :-1]); n[:-1, :-1] = np.maximum(n[:-1, :-1], p[1:, 1:])
        v = np.where(mask, np.maximum(p, n), p)   # off-cells keep NEG (p), on-cells take max
    return v


def _ref(grid):
    x = np.asarray(grid)
    if not set(np.unique(x)).issubset({0, 8}):
        return None
    H, W = x.shape
    cyan = (x == 8)
    cols = np.tile(np.arange(W), (H, 1)).astype(np.float64)
    base_max = np.where(cyan, cols, NEG)
    base_min = np.where(cyan, -cols, NEG)
    maxc = _prop8(cyan, base_max)
    minp = _prop8(cyan, base_min)                 # = -(min col) on cyan
    width = maxc + minp + 1.0                      # on cyan: maxcol - mincol + 1
    gmax = width.max()
    y = np.zeros_like(x)
    y[cyan & (width > gmax - 0.5)] = 1
    y[cyan & (width <= gmax - 0.5)] = 2
    return y


# --------------------------------------------------------------------------- #
# ONNX                                                                         #
# --------------------------------------------------------------------------- #
def _const(name, arr, dt=F):
    a = np.asarray(arr)
    a = a.astype(np.int64) if dt == I64 else a.astype(np.float32)
    return oh.make_tensor(name, dt, list(a.shape) if a.shape else [1], a.flatten().tolist())


def _build():
    nodes, inits = [], []

    def C(name, arr, dt=F):
        inits.append(_const(name, arr, dt)); return name

    # --- channel slices --- #
    C("s0", [0], I64); C("e1", [1], I64); C("ax1", [1], I64)
    nodes.append(oh.make_node("Slice", ["input", "s0", "e1", "ax1"], ["ch0"]))       # background
    C("s8", [ON_CHANNEL], I64); C("e9", [ON_CHANNEL + 1], I64)
    nodes.append(oh.make_node("Slice", ["input", "s8", "e9", "ax1"], ["cyan"]))      # [1,1,S,S]

    C("one", [1.0]); C("neg", [NEG])
    nodes.append(oh.make_node("Sub", ["one", "cyan"], ["ncyan"]))
    nodes.append(oh.make_node("Mul", ["ncyan", "neg"], ["negmask"]))                 # NEG off-cyan, 0 on-cyan

    cols = np.tile(np.arange(S), (S, 1)).reshape(1, 1, S, S).astype(np.float32)
    C("cols", cols)
    C("negcols", -cols)

    def prop(base, tag):
        # V0 = base*cyan + negmask
        nodes.append(oh.make_node("Mul", [base, "cyan"], [tag + "_m"]))
        cur = tag + "_0"
        nodes.append(oh.make_node("Add", [tag + "_m", "negmask"], [cur]))
        for s in range(STEPS):
            pd = f"{tag}_p{s}"
            nodes.append(oh.make_node("Pad", [cur], [pd], mode="constant", value=NEG,
                                      pads=[0, 0, 1, 1, 0, 0, 1, 1]))                 # [1,1,S+2,S+2]
            mp = f"{tag}_mp{s}"
            nodes.append(oh.make_node("MaxPool", [pd], [mp],
                                      kernel_shape=[3, 3], strides=[1, 1]))           # [1,1,S,S]
            mm = f"{tag}_mm{s}"
            nodes.append(oh.make_node("Mul", [mp, "cyan"], [mm]))
            nxt = f"{tag}_{s+1}"
            nodes.append(oh.make_node("Add", [mm, "negmask"], [nxt]))
            cur = nxt
        return cur

    maxc = prop("cols", "mx")          # on cyan: max col ; else NEG
    minp = prop("negcols", "mn")       # on cyan: -(min col) ; else NEG

    nodes.append(oh.make_node("Add", [maxc, minp], ["wsum"]))
    nodes.append(oh.make_node("Add", ["wsum", "one"], ["width"]))                    # on cyan: bbox width; else ~-199
    nodes.append(oh.make_node("ReduceMax", ["width"], ["gmax"], axes=[1, 2, 3], keepdims=1))
    C("half", [0.5])
    nodes.append(oh.make_node("Sub", ["gmax", "half"], ["thr"]))
    nodes.append(oh.make_node("Greater", ["width", "thr"], ["blueb"]))               # true only on max-width cyan
    nodes.append(oh.make_node("Cast", ["blueb"], ["blue"], to=F))
    nodes.append(oh.make_node("Sub", ["cyan", "blue"], ["red"]))                     # cyan AND NOT blue

    nodes.append(oh.make_node("Sub", ["cyan", "cyan"], ["Z"]))                       # zero plane
    nodes.append(oh.make_node("Concat",
                              ["ch0", "blue", "red", "Z", "Z", "Z", "Z", "Z", "Z", "Z"],
                              ["output"], axis=1))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "rb277", [x], [y], inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = []
    for s in ("train", "test"):
        for e in ex.get(s, []):
            a = np.array(e["input"]); b = np.array(e["output"])
            if a.ndim != 2 or b.ndim != 2 or a.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            prs.append((a, b))
    if not prs:
        return []
    for a, b in prs:
        r = _ref(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            return []
    return [("family_rb277", _build())]
