"""Hand-golf of task 392 (concentric Chebyshev rings). The incumbent crk9_concentric
searches every doubled centre on the FULL 30x30 grid -> a [3600,30,30] tensor (~139MB
intermediate, 6.25 pts). All 392 grids are <=10x10, so we do the identical search on a
S=12 work area: Slice input->[1,10,12,12], run the concentric search with NC=24, Pad the
[1,10,12,12] result back to 30x30. Memory ~[576,12,12] -> ~40x smaller. Value-exact
(same algorithm, smaller canvas); validated on all train+test+arc-gen by the grader.
"""
import numpy as np
import onnx
from onnx import helper as oh
import family_crk9_0 as base   # reuse the numpy reference _solve_concentric for detection
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

FLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64
BIG = 1e6
S = 12
NC = 2 * S


def _const(name, arr, dtype=FLOAT):
    a = np.asarray(arr)
    a = a.astype(np.int64) if dtype == INT64 else a.astype(np.float32)
    return oh.make_tensor(name, dtype, list(a.shape) if a.shape else [1], a.flatten().tolist())


def _build():
    nodes, inits = [], []
    G = S

    def C(name, arr, dtype=FLOAT):
        inits.append(_const(name, arr, dtype)); return name

    # crop input [1,10,30,30] -> [1,10,S,S]
    C("cs", [0, 0], INT64); C("ce", [S, S], INT64); C("ca", [2, 3], INT64)
    nodes.append(oh.make_node("Slice", ["input", "cs", "ce", "ca"], ["inp"]))

    nodes.append(oh.make_node("ReduceSum", ["inp"], ["grid_mask"], axes=[1], keepdims=1))
    C("ax0_s", [0], INT64); C("ax0_e", [1], INT64); C("ax0_a", [1], INT64)
    nodes.append(oh.make_node("Slice", ["inp", "ax0_s", "ax0_e", "ax0_a"], ["ch0"]))
    nodes.append(oh.make_node("Sub", ["grid_mask", "ch0"], ["M"]))
    C("half", [0.5])
    nodes.append(oh.make_node("Greater", ["M", "half"], ["Mbool"]))
    nodes.append(oh.make_node("Greater", ["grid_mask", "half"], ["gridbool"]))
    nodes.append(oh.make_node("ReduceSum", ["inp"], ["chan_sum"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Greater", ["chan_sum", "half"], ["chan_pos"]))
    nodes.append(oh.make_node("Cast", ["chan_pos"], ["chan_pos_f"], to=FLOAT))
    C("notch0", np.array([0.] + [1.] * 9).reshape(1, CHANNELS, 1, 1))
    nodes.append(oh.make_node("Mul", ["chan_pos_f", "notch0"], ["e_Xc"]))
    e5 = np.zeros((1, CHANNELS, 1, 1)); e5[0, 5, 0, 0] = 1.0; C("e5", e5)

    Rcoord = (2.0 * np.arange(G)).reshape(1, 1, G, 1)
    Ccoord = (2.0 * np.arange(G)).reshape(1, 1, 1, G)
    cr2c = np.arange(NC, dtype=float).reshape(NC, 1, 1, 1)
    cc2c = np.arange(NC, dtype=float).reshape(1, NC, 1, 1)
    C("Rcoord", Rcoord); C("Ccoord", Ccoord); C("cr2c", cr2c); C("cc2c", cc2c)
    nodes.append(oh.make_node("Sub", ["Rcoord", "cr2c"], ["drow_s"]))
    nodes.append(oh.make_node("Abs", ["drow_s"], ["drow"]))
    nodes.append(oh.make_node("Sub", ["Ccoord", "cc2c"], ["dcol_s"]))
    nodes.append(oh.make_node("Abs", ["dcol_s"], ["dcol"]))
    nodes.append(oh.make_node("Max", ["drow", "dcol"], ["rho"]))

    C("BIG", [BIG]); C("negone", [-1.0])
    nodes.append(oh.make_node("Where", ["Mbool", "rho", "BIG"], ["rad_at_M"]))
    nodes.append(oh.make_node("ReduceMin", ["rad_at_M"], ["phi2"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Add", ["phi2", "half"], ["phi2_h"]))
    nodes.append(oh.make_node("Greater", ["rho", "phi2_h"], ["gt"]))
    nodes.append(oh.make_node("And", ["gt", "Mbool"], ["sel"]))
    nodes.append(oh.make_node("Where", ["sel", "rho", "BIG"], ["rad_gt"]))
    nodes.append(oh.make_node("ReduceMin", ["rad_gt"], ["second"], axes=[2, 3], keepdims=1))
    nodes.append(oh.make_node("Sub", ["second", "phi2"], ["s2"]))
    nodes.append(oh.make_node("Where", ["Mbool", "rho", "negone"], ["rad_or_neg"]))
    nodes.append(oh.make_node("ReduceMax", ["rad_or_neg"], ["rmax"], axes=[2, 3], keepdims=1))

    nodes.append(oh.make_node("Sub", ["rho", "phi2"], ["g"]))
    nodes.append(oh.make_node("Mod", ["g", "s2"], ["modg"], fmod=1))
    nodes.append(oh.make_node("Abs", ["modg"], ["amodg"]))
    nodes.append(oh.make_node("Less", ["amodg", "half"], ["on"]))
    nodes.append(oh.make_node("Add", ["rmax", "half"], ["rmax_h"]))
    nodes.append(oh.make_node("Less", ["rho", "rmax_h"], ["clip"]))
    nodes.append(oh.make_node("And", ["on", "clip"], ["pc1"]))
    nodes.append(oh.make_node("And", ["pc1", "gridbool"], ["predclip"]))
    nodes.append(oh.make_node("Xor", ["predclip", "Mbool"], ["mm_cell"]))
    nodes.append(oh.make_node("Cast", ["mm_cell"], ["mm_cell_f"], to=FLOAT))
    nodes.append(oh.make_node("ReduceSum", ["mm_cell_f"], ["mm"], axes=[2, 3], keepdims=0))
    nodes.append(oh.make_node("Neg", ["mm"], ["score"]))
    C("flatNN", [NC * NC], INT64)
    nodes.append(oh.make_node("Reshape", ["score", "flatNN"], ["score_flat"]))
    nodes.append(oh.make_node("ArgMax", ["score_flat"], ["best"], axis=0, keepdims=1))

    C("rho_shape", [NC * NC, G, G], INT64)
    nodes.append(oh.make_node("Reshape", ["rho", "rho_shape"], ["rho_flat"]))
    nodes.append(oh.make_node("Gather", ["rho_flat", "best"], ["rho_best"], axis=0))
    nodes.append(oh.make_node("Reshape", ["phi2", "flatNN"], ["phi2_flat"]))
    nodes.append(oh.make_node("Gather", ["phi2_flat", "best"], ["phi2_b"], axis=0))
    nodes.append(oh.make_node("Reshape", ["s2", "flatNN"], ["s2_flat"]))
    nodes.append(oh.make_node("Gather", ["s2_flat", "best"], ["s2_b"], axis=0))
    C("sc111", [1, 1, 1], INT64)
    nodes.append(oh.make_node("Reshape", ["phi2_b", "sc111"], ["phi2_b3"]))
    nodes.append(oh.make_node("Reshape", ["s2_b", "sc111"], ["s2_b3"]))
    nodes.append(oh.make_node("Sub", ["rho_best", "phi2_b3"], ["gb"]))
    nodes.append(oh.make_node("Mod", ["gb", "s2_b3"], ["modb"], fmod=1))
    nodes.append(oh.make_node("Abs", ["modb"], ["amodb"]))
    nodes.append(oh.make_node("Less", ["amodb", "half"], ["rawon"]))
    nodes.append(oh.make_node("Cast", ["rawon"], ["rawon_f"], to=FLOAT))
    C("sh11GG", [1, 1, G, G], INT64)
    nodes.append(oh.make_node("Reshape", ["rawon_f", "sh11GG"], ["rawon4"]))
    nodes.append(oh.make_node("Mul", ["rawon4", "grid_mask"], ["on_f"]))
    nodes.append(oh.make_node("Sub", ["grid_mask", "on_f"], ["off_f"]))
    nodes.append(oh.make_node("Mul", ["on_f", "e_Xc"], ["term1"]))
    nodes.append(oh.make_node("Mul", ["off_f", "e5"], ["term2"]))
    nodes.append(oh.make_node("Add", ["term1", "term2"], ["out_s"]))
    nodes.append(oh.make_node("Pad", ["out_s"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - S, WIDTH - S]))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "golf392", [x], [y], inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    for a, b in prs:
        if a.shape[0] > S or a.shape[1] > S:   # grid must fit the work area
            return
    try:
        for a, b in prs:
            o = base._solve_concentric(a)
            if o is None or o.shape != b.shape or not np.array_equal(o, b):
                return
    except Exception:
        return
    yield ("golf392_crop", _build())
