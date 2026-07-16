"""family_b1c — recompiled cheap ONNX for a slice of the hard-task backlog.

Gates fire only for the specific tasks whose numpy `_ref` is exact on train+test.

task064 (ARC 2c608aff) — "connect marker to rectangle".
  Grid has 3 colors: background (most frequent), one SOLID rectangle (middle color),
  and marker cells (least frequent single cells). Every marker that shares a row OR a
  column with the rectangle's extent draws a straight line, in the marker color, from
  itself up to (and not overwriting) the rectangle. Non-aligned markers are unchanged.
  Verified: _ref exact on 267/267 shipped (train+test+arc-gen) and 4000/4000 fresh.

ONNX design (single-channel working tensors + free one-hot output):
  * colour scalars (bg/rect/marker) chosen by per-channel pixel counts — all done on
    tiny [1,10,1,1] tensors (negligible memory).
  * R (rect mask), K (marker mask) built as [1,1,30,30].
  * rect row/col bands via ReduceMax; above/below/left/right half-planes from the bands.
  * marker "reach" via CumSum (int32) in 4 directions; fill mask M = union of the four
    band-gated reaches.
  * terminal: newcolour = Where(M, marker, colourgrid); output = Equal(newcolour, arange)
    broadcasts straight into the free [1,10,30,30] output — no 10-channel intermediate.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION

F = onnx.TensorProto.FLOAT
I32 = onnx.TensorProto.INT32
I64 = onnx.TensorProto.INT64
B = onnx.TensorProto.BOOL
OPSET = [oh.make_opsetid("", 12)]


# ---------------- numpy reference (gate) ---------------- #
def _ref(inp):
    x = np.asarray(inp)
    if x.ndim != 2:
        return None
    vals, cnts = np.unique(x, return_counts=True)
    cnt = dict(zip(vals.tolist(), cnts.tolist()))
    bg = int(vals[np.argmax(cnts)])
    others = [int(v) for v in vals.tolist() if v != bg]
    rect, ra = None, -1
    for v in others:
        m = (x == v)
        rr = np.where(m.any(1))[0]
        cc = np.where(m.any(0))[0]
        if len(rr) == 0:
            continue
        if m[rr.min():rr.max() + 1, cc.min():cc.max() + 1].all():
            area = (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
            if area > ra:
                ra, rect = area, v
    if rect is None:
        return x.copy()
    markers = [v for v in others if v != rect]
    if not markers:
        return x.copy()
    marker = min(markers, key=lambda v: cnt[v])
    R = (x == rect)
    rows = np.where(R.any(1))[0]
    cols = np.where(R.any(0))[0]
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    out = x.copy()
    for i, j in zip(*np.where(x == marker)):
        if c0 <= j <= c1:
            if i < r0:
                out[i:r0, j] = marker
            elif i > r1:
                out[r1 + 1:i + 1, j] = marker
        elif r0 <= i <= r1:
            if j < c0:
                out[i, j:c0] = marker
            elif j > c1:
                out[i, c1 + 1:j + 1] = marker
    return out


# ---------------- ONNX builder ---------------- #
def _c(name, arr, dt):
    a = np.asarray(arr)
    a = a.astype(np.int64) if dt == I64 else (
        a.astype(np.int32) if dt == I32 else a.astype(np.float32))
    return oh.make_tensor(name, dt, list(a.shape) if a.shape else [1], a.flatten().tolist())


def _build():
    nodes, inits = [], []

    def C(name, arr, dt=F):
        inits.append(_c(name, arr, dt)); return name

    def N(op, ins, outs, **kw):
        nodes.append(oh.make_node(op, ins, [outs] if isinstance(outs, str) else outs, **kw))
        return outs

    C("wcol", np.arange(10).reshape(1, 10, 1, 1))          # colour weights
    C("ar10", np.arange(10).reshape(1, 10, 1, 1))          # channel ids (one-hot terminal)
    C("half", [0.5])
    C("axr", [2], I64)                                     # rows axis
    C("axc", [3], I64)                                     # cols axis

    # ---- per-channel selection (all on tiny [1,10,1,1] tensors) ----
    C("zero", [0.0])
    C("BIG", [1e9])
    C("one_f", [1.0])
    N("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)          # pixel count / colour
    N("ReduceMax", ["cnt"], "cmax", axes=[1], keepdims=1)             # bg = most frequent
    N("Equal", ["cnt", "cmax"], "bg_sel")
    N("Cast", ["bg_sel"], "bg_self", to=F)
    N("Greater", ["cnt", "zero"], "present")
    N("Cast", ["present"], "presf", to=F)
    N("Sub", ["presf", "bg_self"], "nbp")                            # present & not-bg

    # solidity per colour: cnt == (#rows touched)*(#cols touched)  -> combinatorial rectangle
    N("ReduceMax", ["input"], "rpres", axes=[3], keepdims=1)         # [1,10,30,1]
    N("ReduceSum", ["rpres"], "nrows", axes=[2], keepdims=1)         # [1,10,1,1]
    N("ReduceMax", ["input"], "cpres", axes=[2], keepdims=1)         # [1,10,1,30]
    N("ReduceSum", ["cpres"], "ncols", axes=[3], keepdims=1)         # [1,10,1,1]
    N("Mul", ["nrows", "ncols"], "area")
    N("Equal", ["cnt", "area"], "solidb")
    N("Cast", ["solidb"], "solidf", to=F)
    N("Mul", ["solidf", "nbp"], "snb")                              # solid & non-bg present

    # rect = the solid non-bg colour of maximal area
    N("Mul", ["area", "snb"], "sarea")
    N("ReduceMax", ["sarea"], "smaxarea", axes=[1], keepdims=1)
    N("Equal", ["area", "smaxarea"], "areqb")
    N("Cast", ["areqb"], "areqf", to=F)
    N("Mul", ["snb", "areqf"], "rc_self0")
    N("Greater", ["rc_self0", "half"], "rc_selb")
    N("Cast", ["rc_selb"], "rc_self", to=F)

    # marker = non-bg, non-rect colour of minimal count (row-major tiebreak via ReduceMin)
    N("Sub", ["nbp", "rc_self"], "mk_cand")                         # marker candidates (0/1)
    N("Sub", ["one_f", "mk_cand"], "noncand")
    N("Mul", ["noncand", "BIG"], "candpen")
    N("Add", ["cnt", "candpen"], "cnt_adj")                         # non-candidates -> huge
    N("ReduceMin", ["cnt_adj"], "cmin", axes=[1], keepdims=1)
    N("Equal", ["cnt_adj", "cmin"], "mkeqb")
    N("Cast", ["mkeqb"], "mkeqf", to=F)
    N("Mul", ["mkeqf", "mk_cand"], "mk_self0")                      # keep only genuine candidate
    N("Greater", ["mk_self0", "half"], "mk_selb")
    N("Cast", ["mk_selb"], "mk_self", to=F)

    # ---- colour grid G = sum_c c*ch_c via 1x1 conv (no [1,10,30,30] intermediate) ----
    C("wconv", np.arange(10).reshape(1, 10, 1, 1))                  # conv weight [1,10,1,1]
    N("Conv", ["input", "wconv"], "G", kernel_shape=[1, 1])        # [1,1,30,30] float

    # ---- R (rect mask) and K (marker mask) via channel-contracting Einsum ----
    C("sh_1_10", [1, 10], I64)
    C("sh_1_1_30_30", [1, 1, 30, 30], I64)
    N("Reshape", ["rc_self", "sh_1_10"], "rc2d")
    N("Einsum", ["input", "rc2d"], "R3", equation="ncij,nc->nij")  # [1,30,30]
    N("Reshape", ["R3", "sh_1_1_30_30"], "R")
    N("Reshape", ["mk_self", "sh_1_10"], "mk2d")
    N("Einsum", ["input", "mk2d"], "K3", equation="ncij,nc->nij")
    N("Reshape", ["K3", "sh_1_1_30_30"], "Kf")
    # marker scalar value = sum_c c*mk_self  [1,1,1,1]
    N("Mul", ["mk_self", "ar10"], "mkv0")
    N("ReduceSum", ["mkv0"], "mkval", axes=[1], keepdims=1)        # [1,1,1,1]

    # ---- rect bands ----
    N("ReduceMax", ["R"], "rowband", axes=[3], keepdims=1)         # [1,1,30,1]
    N("ReduceMax", ["R"], "colband", axes=[2], keepdims=1)         # [1,1,1,30]
    # cumulative presence of the band (prefix max via CumSum>0)
    N("Cast", ["rowband"], "rowband_i", to=I32)
    N("Cast", ["colband"], "colband_i", to=I32)
    N("CumSum", ["rowband_i", "axr"], "rb_cd")                     # down-cumsum rows
    N("CumSum", ["rowband_i", "axr"], "rb_cu", reverse=1)          # up-cumsum
    N("CumSum", ["colband_i", "axc"], "cb_cl")                     # left->right
    N("CumSum", ["colband_i", "axc"], "cb_cr", reverse=1)
    # aboveRect = (rb_cd==0); belowRect=(rowband==0 & rb_cd>0); analog for cols
    C("i0", [0], I32)
    N("Equal", ["rb_cd", "i0"], "above")                          # rows strictly above rect
    N("Equal", ["rb_cu", "i0"], "below")                          # rows strictly below rect
    N("Equal", ["cb_cl", "i0"], "leftof")
    N("Equal", ["cb_cr", "i0"], "rightof")
    # colband/rowband as bool
    C("f0", [0.0])
    N("Greater", ["colband", "f0"], "colb_b")                     # [1,1,1,30]
    N("Greater", ["rowband", "f0"], "rowb_b")                     # [1,1,30,1]

    # ---- marker reach (int32 cumsum of Kf) in 4 directions ----
    N("Cast", ["Kf"], "Ki", to=I32)
    N("CumSum", ["Ki", "axr"], "reach_d")                        # any marker at/above (down-fill)
    N("CumSum", ["Ki", "axr"], "reach_u", reverse=1)
    N("CumSum", ["Ki", "axc"], "reach_l")
    N("CumSum", ["Ki", "axc"], "reach_r", reverse=1)
    N("Greater", ["reach_d", "i0"], "rd_b")
    N("Greater", ["reach_u", "i0"], "ru_b")
    N("Greater", ["reach_l", "i0"], "rl_b")
    N("Greater", ["reach_r", "i0"], "rr_b")

    # ---- fill mask M ----
    # vertical: colband column, above rect, marker at/above  (reach_d)  OR below+reach_u
    N("And", ["above", "rd_b"], "va")
    N("And", ["below", "ru_b"], "vb")
    N("Or", ["va", "vb"], "vert0")
    N("And", ["vert0", "colb_b"], "vert")                        # broadcast [1,1,1,30]
    # horizontal
    N("And", ["leftof", "rl_b"], "ha")
    N("And", ["rightof", "rr_b"], "hb")
    N("Or", ["ha", "hb"], "horiz0")
    N("And", ["horiz0", "rowb_b"], "horiz")
    N("Or", ["vert", "horiz"], "M")                             # [1,1,30,30] bool

    # ---- terminal ----
    # in-grid mask so out-of-grid cells decode to "no colour" (all channels 0)
    N("ReduceSum", ["input"], "pix", axes=[1], keepdims=1)       # [1,1,30,30]
    N("Greater", ["pix", "f0"], "ingrid")                       # bool
    N("Where", ["M", "mkval", "G"], "newcol")                   # marker where fill else colour
    C("neg1", [-1.0])
    N("Where", ["ingrid", "newcol", "neg1"], "gsel")            # -1 outside grid
    N("Equal", ["gsel", "ar10"], "output")                     # [1,10,30,30] bool -> free output

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", B, GRID_SHAPE)
    g = oh.make_graph(nodes, "b1c064", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET)


# The task064 build is EXACT (267/267 shipped, 4000/4000 fresh) but its cheapest
# verifiable form still needs 4 int32 CumSum reaches (4*3600 B = 14.4 KB) plus ~10
# single-channel grids, scoring 13.93 pts — WORSE than the specialised fp16 incumbent
# (15.55). It is therefore NOT emitted (would regress a naive override / lose a blend).
# Flip EMIT to True only if a cheaper reach formulation drops it below the incumbent.
EMIT = False


def candidates(examples):
    if not EMIT:
        return
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    for a, b in prs:
        r = _ref(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            return
    yield ("b1c064_connect", _build())
