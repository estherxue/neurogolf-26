"""Verifier-translated ONNX family (batch vc_3).

task023 (verify_150deff5): recolor gray(5) shapes into red(2)/azure(8). The verifier is an
iterative template-occurrence "peeling" CA (power-5). Decoded templates:
  * 8-template: an isolated 2x2 gray block (bg on 2 sides, one corner may connect) -> paint 2x2
    as 8;  4 rotations.
  * 2-template: a gray "tip" cell (bg up/left/right) -> paint a length-3 line of 2 going into the
    body;  4 rotations.
Each of 5 iterations: match 8-blocks and 2-tips on the current uncolored-gray mask W, colour them
(8 has priority over 2, first-colour-wins across iterations), then REMOVE coloured cells from W.
A fully-parallel per-iteration formulation reproduces the sequential verifier EXACTLY on all
266 train+test+arc-gen pairs. Implemented as a static opset-10 graph: 5 unrolled steps of Conv
template matching (kernel encodes FG=+1 / BG=-nFG so an exact hit scores nFG) + Conv painting.

task133 (verify_57aa92db) and task238 (verify_9aec4887): NOT emitted. Both are data-dependent
beyond a static graph — 133 stamps a variable number of scaled/recoloured copies of a template
at arbitrary marker positions (data-dependent count/placement/scale); 238 crops a
data-dependent-size subgrid around a frame object and fills its interior by a per-cell nearest
-object distance (Voronoi) computation. Neither is expressible with the static arsenal.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64

# ---- template offsets (anchor-relative) ------------------------------------- #
# 8-template: anchor = 2x2 block top-left. block=FG, plus required-bg L-frame, corner wildcard.
FG8 = [(0, 0), (0, 1), (1, 0), (1, 1)]
BG8 = [(-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -1), (0, 2), (1, -1), (2, -1), (2, 0)]
PAINT8 = [(0, 0), (0, 1), (1, 0), (1, 1)]
# 2-template: anchor = tip cell. FG tip, bg up/left/right; paint length-3 line downward.
FG2 = [(0, 0)]
BG2 = [(-1, 0), (0, -1), (0, 1)]
PAINT2 = [(0, 0), (1, 0), (2, 0)]


def _rot(offs):                       # rotate offsets 90deg: (r,c)->(c,-r)
    return [(c, -r) for (r, c) in offs]


def _allrot(fg, bg, pa):
    res = []
    for _ in range(4):
        res.append((fg, bg, pa)); fg, bg, pa = _rot(fg), _rot(bg), _rot(pa)
    return res


T8 = _allrot(FG8, BG8, PAINT8)
T2 = _allrot(FG2, BG2, PAINT2)


# ---- numpy reference (detection gate; == verifier on all 266) --------------- #
def _match(W, fg, bg):
    H, Wd = W.shape
    m = np.ones_like(W, bool)
    for (dr, dc) in fg:
        sh = np.zeros_like(W, bool)
        r0, r1 = max(0, -dr), min(H, H - dr); c0, c1 = max(0, -dc), min(Wd, Wd - dc)
        sh[r0:r1, c0:c1] = W[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        m &= sh
    for (dr, dc) in bg:
        val = np.zeros_like(W, bool)
        r0, r1 = max(0, -dr), min(H, H - dr); c0, c1 = max(0, -dc), min(Wd, Wd - dc)
        val[r0:r1, c0:c1] = W[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        m &= (~val)
    return m


def _paint(anchors, pa, H, Wd):
    out = np.zeros((H, Wd), bool)
    ar, ac = np.where(anchors)
    for (dr, dc) in pa:
        rr, cc = ar + dr, ac + dc
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < Wd)
        out[rr[ok], cc[ok]] = True
    return out


def _solve(I):
    I = np.asarray(I)
    if set(np.unique(I).tolist()) - {0, 5}:
        return None
    H, Wd = I.shape
    W = (I == 5)
    ACC = np.zeros((H, Wd), int)
    for _ in range(5):
        m8 = np.zeros((H, Wd), bool)
        for fg, bg, pa in T8:
            m8 |= _paint(_match(W, fg, bg), pa, H, Wd)
        m8 &= W
        m2 = np.zeros((H, Wd), bool)
        for fg, bg, pa in T2:
            m2 |= _paint(_match(W, fg, bg), pa, H, Wd)
        m2 &= W
        avail = (ACC == 0)
        new8 = m8 & avail
        new2 = m2 & avail & (~m8)
        ACC[new8] = 8; ACC[new2] = 2
        W = W & (~(new8 | new2))
    out = I.copy()
    out[ACC == 2] = 2; out[ACC == 8] = 8
    return out


# ---- ONNX builder ----------------------------------------------------------- #
def _kernels():
    """8 templates (4x 8-block then 4x 2-tip). match weights [8,1,5,5], grouped paint weights
    [8,1,5,5], per-channel thresholds [1,8,1,1]."""
    mw = np.zeros((8, 1, 5, 5), np.float32)
    pw = np.zeros((8, 1, 5, 5), np.float32)
    thr = np.zeros((1, 8, 1, 1), np.float32)
    templates = [(fg, bg, pa, 4) for fg, bg, pa in T8] + [(fg, bg, pa, 1) for fg, bg, pa in T2]
    for i, (fg, bg, pa, nfg) in enumerate(templates):
        for (dr, dc) in fg:
            mw[i, 0, 2 + dr, 2 + dc] += 1.0
        for (dr, dc) in bg:
            mw[i, 0, 2 + dr, 2 + dc] += -float(nfg)
        for (dr, dc) in pa:
            pw[i, 0, 2 - dr, 2 - dc] = 1.0     # out[q] gets match[q-off]
        thr[0, i, 0, 0] = nfg - 0.5
    return mw, pw, thr


def _build():
    nodes, inits = [], []

    def C(name, arr, dt=F):
        a = np.asarray(arr)
        a = a.astype(np.int64) if dt == I64 else a.astype(np.float32)
        inits.append(oh.make_tensor(name, dt, list(a.shape) if a.shape else [1],
                                    a.flatten().tolist()))
        return name

    mw, pw, thr = _kernels()
    C("MW", mw); C("PW", pw); C("THR", thr)
    C("half", [0.5]); C("one", [1.0])
    C("zero11", np.zeros((1, 1, 30, 30)))

    C("s5", [5], I64); C("e6", [6], I64); C("s0", [0], I64); C("e1", [1], I64); C("ax1", [1], I64)
    nodes.append(oh.make_node("Slice", ["input", "s5", "e6", "ax1"], ["W0"]))
    nodes.append(oh.make_node("Slice", ["input", "s0", "e1", "ax1"], ["ch0"]))
    C("s04", [0], I64); C("e04", [4], I64); C("s48", [4], I64); C("e48", [8], I64)

    W = "W0"; acc8 = "zero11"; acc2 = "zero11"; taken = "zero11"
    for it in range(5):
        p = f"i{it}_"
        nodes.append(oh.make_node("Conv", [W, "MW"], [p + "sc"], kernel_shape=[5, 5],
                                  pads=[2, 2, 2, 2]))
        nodes.append(oh.make_node("Greater", [p + "sc", "THR"], [p + "mb"]))
        nodes.append(oh.make_node("Cast", [p + "mb"], [p + "mf"], to=F))          # [1,8,30,30]
        nodes.append(oh.make_node("Conv", [p + "mf", "PW"], [p + "pc"], kernel_shape=[5, 5],
                                  pads=[2, 2, 2, 2], group=8))
        nodes.append(oh.make_node("Slice", [p + "pc", "s04", "e04", "ax1"], [p + "p8"]))
        nodes.append(oh.make_node("Slice", [p + "pc", "s48", "e48", "ax1"], [p + "p2"]))
        nodes.append(oh.make_node("ReduceSum", [p + "p8"], [p + "r8"], axes=[1], keepdims=1))
        nodes.append(oh.make_node("ReduceSum", [p + "p2"], [p + "r2"], axes=[1], keepdims=1))
        nodes.append(oh.make_node("Greater", [p + "r8", "half"], [p + "b8"]))
        nodes.append(oh.make_node("Greater", [p + "r2", "half"], [p + "b2"]))
        nodes.append(oh.make_node("Cast", [p + "b8"], [p + "f8"], to=F))
        nodes.append(oh.make_node("Cast", [p + "b2"], [p + "f2"], to=F))
        nodes.append(oh.make_node("Mul", [p + "f8", W], [p + "mask8"]))            # &W
        nodes.append(oh.make_node("Mul", [p + "f2", W], [p + "mask2"]))
        nodes.append(oh.make_node("Sub", ["one", taken], [p + "avail"]))
        nodes.append(oh.make_node("Mul", [p + "mask8", p + "avail"], [p + "new8"]))
        nodes.append(oh.make_node("Sub", ["one", p + "mask8"], [p + "not8"]))
        nodes.append(oh.make_node("Mul", [p + "mask2", p + "avail"], [p + "t2"]))
        nodes.append(oh.make_node("Mul", [p + "t2", p + "not8"], [p + "new2"]))
        nodes.append(oh.make_node("Add", [acc8, p + "new8"], [p + "acc8"]))
        nodes.append(oh.make_node("Add", [acc2, p + "new2"], [p + "acc2"]))
        nodes.append(oh.make_node("Add", [taken, p + "new8"], [p + "tk1"]))
        nodes.append(oh.make_node("Add", [p + "tk1", p + "new2"], [p + "taken"]))
        nodes.append(oh.make_node("Add", [p + "new8", p + "new2"], [p + "rem"]))
        nodes.append(oh.make_node("Sub", ["one", p + "rem"], [p + "keep"]))
        nodes.append(oh.make_node("Mul", [W, p + "keep"], [p + "W"]))
        W, acc8, acc2, taken = p + "W", p + "acc8", p + "acc2", p + "taken"

    chans = ["ch0", "zero11", acc2, "zero11", "zero11", W, "zero11", "zero11", acc8, "zero11"]
    nodes.append(oh.make_node("Concat", chans, ["output"], axis=1))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "vc3_150deff5", [x], [y], inits)
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
    yield ("vc3_150deff5", _build())
