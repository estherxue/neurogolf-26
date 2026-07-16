"""task079 — output the most-frequent isolated 3x3 object shape (user's sliding-window method).
Fixed 14x14 input, 3x3 output. Per color c, per 3x3 anchor: valid iff window has c AND the
16-cell surrounding ring has no c (isolation). target color = argmax object_count. Extract that
color's 3x3 patch (via a Conv whose WEIGHT is the runtime-computed valid-anchor map = template
cross-correlation). Fully static opset-10. Validated 266/266 on train+test+arc-gen.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64


# ---- numpy reference (detection gate) --------------------------------------- #
def _solve(x):
    """Per color: extract the object pattern from ONE isolated (16-ring-empty) instance; object
    count = total_cells / cells_per_object (robust — counts non-isolated objects too). Target =
    argmax count. Same color has only one pattern (invariant)."""
    x = np.asarray(x); H, W = x.shape
    if (H, W) != (14, 14):
        return None
    ring = [(-1, 0), (-1, 1), (-1, 2), (3, 0), (3, 1), (3, 2), (0, -1), (1, -1), (2, -1),
            (0, 3), (1, 3), (2, 3), (-1, -1), (-1, 3), (3, -1), (3, 3)]  # 16-cell ring
    score, patch = {}, {}
    for c in range(1, 10):
        m = (x == c).astype(int)
        tot = m.sum()
        if tot == 0:
            continue
        P = None
        for a in range(H - 2):
            for b in range(W - 2):
                win = m[a:a + 3, b:b + 3]
                if win.sum() == 0:
                    continue
                if all(not (0 <= a + dr < H and 0 <= b + dc < W and m[a + dr, b + dc])
                       for dr, dc in ring):
                    P = win.copy(); break
            if P is not None:
                break
        if P is None:
            continue
        score[c] = tot / P.sum()          # object count = total cells / cells-per-object
        patch[c] = P * c
    if not score:
        return None
    return patch[max(score, key=lambda k: score[k])]


def _const(name, arr, dt=F):
    a = np.asarray(arr)
    a = a.astype(np.int64) if dt == I64 else a.astype(np.float32)
    return oh.make_tensor(name, dt, list(a.shape) if a.shape else [1], a.flatten().tolist())


def _build():
    nodes, inits = [], []

    def C(name, arr, dt=F):
        inits.append(_const(name, arr, dt)); return name

    # crop 30x30 -> 14x14, take colour channels 1..9
    C("cs", [0, 0], I64); C("ce", [14, 14], I64); C("ca", [2, 3], I64)
    nodes.append(oh.make_node("Slice", ["input", "cs", "ce", "ca"], ["inp"]))
    C("chs", [1], I64); C("che", [10], I64); C("cha", [1], I64)
    nodes.append(oh.make_node("Slice", ["inp", "chs", "che", "cha"], ["M9"]))   # [1,9,14,14]

    # winsum = 3x3 ones conv (grouped) -> [1,9,12,12]
    C("ones3", np.ones((9, 1, 3, 3)))
    nodes.append(oh.make_node("Conv", ["M9", "ones3"], ["winsum"], group=9,
                              kernel_shape=[3, 3]))
    # ring16 = 5x5 frame conv on pad-1 input -> [1,9,12,12]
    ringk = np.ones((5, 5)); ringk[1:4, 1:4] = 0
    C("ring5", np.tile(ringk, (9, 1, 1, 1)))
    nodes.append(oh.make_node("Pad", ["M9"], ["M9p"], mode="constant", value=0.0,
                              pads=[0, 0, 1, 1, 0, 0, 1, 1]))
    nodes.append(oh.make_node("Conv", ["M9p", "ring5"], ["ring"], group=9,
                              kernel_shape=[5, 5]))

    C("half", [0.5])
    nodes.append(oh.make_node("Greater", ["winsum", "half"], ["has"]))
    nodes.append(oh.make_node("Less", ["ring", "half"], ["iso"]))
    nodes.append(oh.make_node("And", ["has", "iso"], ["valid"]))
    nodes.append(oh.make_node("Cast", ["valid"], ["validf"], to=F))            # [1,9,12,12]

    # object count = total colour-c cells / cells-per-object (robust: counts non-isolated too)
    nodes.append(oh.make_node("ReduceSum", ["M9"], ["total"], axes=[2, 3], keepdims=0))       # [1,9]
    nodes.append(oh.make_node("Mul", ["validf", "winsum"], ["vw"]))
    nodes.append(oh.make_node("ReduceMax", ["vw"], ["psize"], axes=[2, 3], keepdims=0))        # [1,9] |P| per colour
    nodes.append(oh.make_node("Greater", ["psize", "half"], ["pm"]))
    nodes.append(oh.make_node("Cast", ["pm"], ["pmf"], to=F))
    C("one_f", [1.0])
    nodes.append(oh.make_node("Sub", ["one_f", "pmf"], ["notpm"]))
    nodes.append(oh.make_node("Add", ["psize", "notpm"], ["safeP"]))                            # avoid /0
    nodes.append(oh.make_node("Div", ["total", "safeP"], ["ratio"]))
    nodes.append(oh.make_node("Mul", ["ratio", "pmf"], ["score"]))                              # 0 if no isolated
    nodes.append(oh.make_node("ArgMax", ["score"], ["tgt"], axis=1, keepdims=0))               # [1]

    nodes.append(oh.make_node("Gather", ["validf", "tgt"], ["vt"], axis=1))    # [1,1,12,12]
    nodes.append(oh.make_node("Gather", ["M9", "tgt"], ["mt"], axis=1))        # [1,1,14,14]
    # extract 3x3 = cross-correlation of M_target with valid_target (DATA-DEPENDENT conv weight)
    nodes.append(oh.make_node("Conv", ["mt", "vt"], ["extract"], kernel_shape=[12, 12]))     # [1,1,3,3]

    nodes.append(oh.make_node("Greater", ["extract", "half"], ["shp"]))
    nodes.append(oh.make_node("Less", ["extract", "half"], ["bgm"]))
    nodes.append(oh.make_node("Cast", ["shp"], ["shpf"], to=F))
    nodes.append(oh.make_node("Cast", ["bgm"], ["bgf"], to=F))

    # target output channel = tgt+1 ; build one-hot e_col over the 10 output channels
    C("one1", [1], I64)
    nodes.append(oh.make_node("Add", ["tgt", "one1"], ["tcol"]))               # [1] channel 1..9
    C("ar10", list(range(10)), I64)
    nodes.append(oh.make_node("Equal", ["ar10", "tcol"], ["eqc"]))            # [10] bool
    nodes.append(oh.make_node("Cast", ["eqc"], ["ecf"], to=F))
    C("sh1011", [1, 10, 1, 1], I64)
    nodes.append(oh.make_node("Reshape", ["ecf", "sh1011"], ["ecol"]))         # [1,10,1,1]
    e0 = np.zeros((1, 10, 1, 1)); e0[0, 0, 0, 0] = 1.0; C("e0", e0)

    nodes.append(oh.make_node("Mul", ["shpf", "ecol"], ["termC"]))             # [1,10,3,3]
    nodes.append(oh.make_node("Mul", ["bgf", "e0"], ["termB"]))
    nodes.append(oh.make_node("Add", ["termC", "termB"], ["out3"]))
    nodes.append(oh.make_node("Pad", ["out3"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, 27, 27]))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "hc079", [x], [y], inits)
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
    yield ("hc079_sliding3x3", _build())
