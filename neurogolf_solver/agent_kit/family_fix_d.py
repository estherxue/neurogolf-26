"""family_fix_d — exact rebuilds of gen-failing incumbents (audit wave D).

Two tasks whose incumbent solvers pass the public set but FAIL fresh arc-gen samples
(so they would score 0 on the private set) are rebuilt here from the generator source,
exact on the FULL generator distribution:

  * task279  (gen: b2862040)  "loopfill".  All shapes are colour-1 on maroon(9).  A shape
    becomes cyan(8) iff its component contains a CYCLE (a closed rectangle ring); open
    boxes (a ring with one perimeter pixel removed) and their barnacles stay blue(1).
    The incumbent (family_golf2_0 loopfill / rfo_freeout / fp16_279) used an enclosed-bg
    flood which leaks through spurious 1-cell pockets wedged between two shapes -> ~0.2%
    gen failures.  Cycle detection (iterated leaf-trim, then flood the surviving cycle
    across the component) is exact.  Pure single-channel conv chain, opset-10.

  * task197  (gen: 82819916)  "rowpattern".  Every marked row is the SAME binary column
    pattern painted in that row's own (light,dark) colour pair.  Row 1 is fully revealed
    (the reference pattern); every other marked row is revealed only as a prefix (up to
    where both of its colours first appear).  Reconstruct each row from the reference
    pattern + that row's two colours.  The incumbent (golfe4_crop14_197 / g73_f16_197)
    cropped/approximated and failed ~0.5%.  Channel-wise ReduceMax/ReduceSum, opset-10.

Both are numpy-_ref exact on train+test+>=3000 fresh gen samples, and the hand-built ONNX
is validated bit-equal to the numpy _ref on all local data + fresh gen samples.

task066 (gen: 2dd70a9a) was ALSO assigned but is provably NOT exactly solvable: ~0.1% of
generator inputs are genuinely ambiguous (the random cyan static occasionally completes a
full false corner-marker set, so the same rendered grid is a valid rendering of TWO distinct
S/U path parameterisations that both satisfy every generator constraint).  See notes returned
to the orchestrator.  Not emitted here.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64
CH, HH, WW = 10, 30, 30

# fixed unrolled step counts (>= observed convergence depth on the gen distribution).
TRIM_STEPS = 16    # leaf-trim iterations (max real: ~8 on 16x16 grids)
FLOOD_STEPS = 6    # cycle->component flood iterations (max real: ~1, barnacles are dist 1)


# =========================================================================== #
# 279  loopfill via cycle detection                                           #
# =========================================================================== #
def _ref_279(a):
    a = np.asarray(a)
    if set(np.unique(a).tolist()) - {1, 9}:
        return None
    reg = (a == 1)

    def ncount(cur):
        H, W = cur.shape
        c = np.zeros((H, W), int)
        c[1:, :] += cur[:-1, :]; c[:-1, :] += cur[1:, :]
        c[:, 1:] += cur[:, :-1]; c[:, :-1] += cur[:, 1:]
        return c

    def dil(x):
        n = x.copy()
        n[1:, :] |= x[:-1, :]; n[:-1, :] |= x[1:, :]
        n[:, 1:] |= x[:, :-1]; n[:, :-1] |= x[:, 1:]
        return n

    cur = reg.copy()
    for _ in range(TRIM_STEPS):
        cur = cur & (ncount(cur) >= 2)
    loop = cur.copy()
    for _ in range(FLOOD_STEPS):
        loop = loop | (dil(loop) & reg)
    o = a.copy()
    o[loop] = 8
    return o


def _const(name, arr, dt=F):
    a = np.asarray(arr)
    a = a.astype(np.int64) if dt == I64 else a.astype(np.float32)
    return oh.make_tensor(name, dt, list(a.shape) if a.shape else [1], a.flatten().tolist())


def _build_279():
    nodes, inits = [], []

    def C(name, arr, dt=F):
        inits.append(_const(name, arr, dt)); return name

    C("s1", [1], I64); C("e2", [2], I64); C("ax1", [1], I64)
    nodes.append(oh.make_node("Slice", ["input", "s1", "e2", "ax1"], ["reg"]))  # [1,1,30,30]

    # neighbour-count kernel (4-neighbour, no centre) and full plus kernel
    C("k_nb", np.array([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], float))
    C("k_pl", np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], float))
    C("c15", [1.5]); C("c05", [0.5])

    # leaf-trim: cur = cur * (neighbours(cur) >= 2)
    cur = "reg"
    for i in range(TRIM_STEPS):
        nodes.append(oh.make_node("Conv", [cur, "k_nb"], [f"nb{i}"],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Greater", [f"nb{i}", "c15"], [f"g{i}"]))
        nodes.append(oh.make_node("Cast", [f"g{i}"], [f"gk{i}"], to=F))
        nodes.append(oh.make_node("Mul", [cur, f"gk{i}"], [f"cur{i}"]))
        cur = f"cur{i}"

    # flood the surviving cycle across the whole component (masked to region)
    loop = cur
    for i in range(FLOOD_STEPS):
        nodes.append(oh.make_node("Conv", [loop, "k_pl"], [f"d{i}"],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Greater", [f"d{i}", "c05"], [f"gg{i}"]))
        nodes.append(oh.make_node("Cast", [f"gg{i}"], [f"gd{i}"], to=F))
        nodes.append(oh.make_node("Mul", ["reg", f"gd{i}"], [f"loop{i}"]))
        loop = f"loop{i}"

    # delta: loop cells move channel 1 (-1) -> channel 8 (+1)
    dv = np.zeros((1, CH, 1, 1), float); dv[0, 1, 0, 0] = -1.0; dv[0, 8, 0, 0] = 1.0
    C("delta", dv)
    nodes.append(oh.make_node("Mul", [loop, "delta"], ["filled"]))  # broadcast [1,1,30,30]*[1,10,1,1]
    nodes.append(oh.make_node("Add", ["input", "filled"], ["output"]))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "fixd279", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# =========================================================================== #
# 197  rowpattern                                                             #
# =========================================================================== #
def _ref_197(x):
    x = np.asarray(x); H, W = x.shape
    marked = [r for r in range(H) if (x[r] != 0).any()]
    if not marked:
        return None
    ref = marked[0]
    refrow = x[ref]
    if (refrow == 0).any():          # reference row must be fully revealed
        return None
    y = np.zeros_like(x)
    for r in marked:
        row = x[r]
        m = {}
        for c in range(W):
            if row[c] != 0:
                m[refrow[c]] = row[c]
        if len(m) < 2:               # both reference colours must appear in this row
            return None
        for c in range(W):
            if refrow[c] not in m:   # reference row has a colour this row never reveals
                return None
            y[r][c] = m[refrow[c]]
    return y


def _build_197():
    nodes, inits = [], []

    def C(name, arr, dt=F):
        inits.append(_const(name, arr, dt)); return name

    # slices along axis 2 (rows) / axis 3 (cols) / axis 1 (channels)
    C("r1", [1], I64); C("r2", [2], I64); C("ax2", [2], I64)
    C("c0", [0], I64); C("c1", [1], I64); C("ax3", [3], I64); C("ax1", [1], I64)

    nodes.append(oh.make_node("Slice", ["input", "r1", "r2", "ax2"], ["Rr"]))    # [1,10,1,30] ref row
    nodes.append(oh.make_node("Slice", ["Rr", "c0", "c1", "ax3"], ["A"]))        # [1,10,1,1] col0 colour

    nodes.append(oh.make_node("Mul", ["Rr", "A"], ["RrA"]))
    nodes.append(oh.make_node("ReduceSum", ["RrA"], ["G0"], axes=[1], keepdims=1))  # [1,1,1,30]
    nodes.append(oh.make_node("ReduceSum", ["Rr"], ["M"], axes=[1], keepdims=1))    # [1,1,1,30]
    nodes.append(oh.make_node("Sub", ["M", "G0"], ["G1"]))                          # [1,1,1,30]

    # zero channel 0 so unrevealed (colour-0) cells in a marked row don't pollute the
    # per-row colour ReduceMax; the real colours live in channels 1-9.
    mask9 = np.ones((1, CH, 1, 1), float); mask9[0, 0, 0, 0] = 0.0
    C("mask9", mask9)
    nodes.append(oh.make_node("Mul", ["input", "mask9"], ["Xc"]))                   # [1,10,30,30]
    nodes.append(oh.make_node("Mul", ["Xc", "G0"], ["XG0"]))
    nodes.append(oh.make_node("ReduceMax", ["XG0"], ["C0"], axes=[3], keepdims=1))  # [1,10,30,1]
    nodes.append(oh.make_node("Mul", ["Xc", "G1"], ["XG1"]))
    nodes.append(oh.make_node("ReduceMax", ["XG1"], ["C1"], axes=[3], keepdims=1))  # [1,10,30,1]

    nodes.append(oh.make_node("Mul", ["C0", "G0"], ["P0"]))                         # [1,10,30,30]
    nodes.append(oh.make_node("Mul", ["C1", "G1"], ["P1"]))
    nodes.append(oh.make_node("Add", ["P0", "P1"], ["out19"]))                      # colours 1-9, ch0=0

    # background (colour-0) plane = in-grid AND not a marked row
    nodes.append(oh.make_node("ReduceSum", ["input"], ["ingrid"], axes=[1], keepdims=1))  # [1,1,30,30]
    nodes.append(oh.make_node("Slice", ["input", "c0", "c1", "ax1"], ["ch0"]))            # [1,1,30,30]
    nodes.append(oh.make_node("Sub", ["ingrid", "ch0"], ["colored"]))                     # colour!=0 mask
    nodes.append(oh.make_node("ReduceMax", ["colored"], ["rowmark"], axes=[3], keepdims=1))  # [1,1,30,1]
    C("one", [1.0])
    nodes.append(oh.make_node("Sub", ["one", "rowmark"], ["unmark"]))
    nodes.append(oh.make_node("Mul", ["ingrid", "unmark"], ["bg0"]))                      # [1,1,30,30]

    e0 = np.zeros((1, CH, 1, 1), float); e0[0, 0, 0, 0] = 1.0
    C("e0", e0)
    nodes.append(oh.make_node("Mul", ["bg0", "e0"], ["bg0c"]))                            # [1,10,30,30]
    nodes.append(oh.make_node("Add", ["out19", "bg0c"], ["output"]))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "fixd197", [x], [y], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# =========================================================================== #
def _pairs(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    return prs


def _exact(prs, ref):
    if not prs:
        return False
    for a, b in prs:
        r = ref(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            return False
    return True


def candidates(examples):
    prs = _pairs(examples)
    if _exact(prs, _ref_279):
        yield ("fixd_loopfill279", _build_279())
    if _exact(prs, _ref_197):
        yield ("fixd_rowpattern197", _build_197())
