"""family_b1b — cheap ONNX recompiles (fully uint8) for enclosed-region tasks.

task002 (00d62c1b): fill enclosed background(black) regions with yellow(4).
task187 (7b6016b9): enclosed black -> red(2); other black -> green(3); colours kept.

Both = 4-connectivity flood-fill reachability from the grid border through 'passable'
cells (non-coloured: inside-black OR outside-grid); enclosed = inside-black AND NOT
reachable.

Cost model = sum of every named node-output tensor (no reuse). Tricks:
  * EVERYTHING in uint8 (1 byte) — half of fp16, quarter of fp32,
  * canvas CROPPED to (G+2)x(G+2) with G = proven max grid side,
  * exterior seeded free by a Pad ring (constant_value=1),
  * 4-conn dilation via SEPARABLE masked MaxPool (converges in fewer iterations than a
    plus-Conv): R = ((MaxPool_{3x1} R) & P) then MaxPool_{1x3} & P — no diagonal leak
    because the mask P is re-applied between the two 1-D dilations,
  * planes assembled at GxG, Concat, then ONE Pad writes straight to the free 'output'
    (grader only checks sign per channel).
Verified: exact on all local train+test+arc-gen and on 600 fresh generator draws
(matching the flood's irreducible ambiguity rate — same as the incumbent).
"""
import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

F = TP.FLOAT
U8 = TP.UINT8
I64 = TP.INT64

GRID_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = [oh.make_opsetid("", 18)]


# ---------------- numpy reference ---------------- #
def _enclosed(g):
    bg = (g == 0)
    R = np.zeros_like(bg)
    R[0, :] |= bg[0, :]; R[-1, :] |= bg[-1, :]
    R[:, 0] |= bg[:, 0]; R[:, -1] |= bg[:, -1]
    while True:
        Rd = R.copy()
        Rd[1:, :] |= R[:-1, :]; Rd[:-1, :] |= R[1:, :]
        Rd[:, 1:] |= R[:, :-1]; Rd[:, :-1] |= R[:, 1:]
        Rd &= bg
        if (Rd == R).all():
            break
        R = Rd
    return bg & ~R


def _ref2(g):
    o = g.copy(); o[_enclosed(g)] = 4
    return o


def _ref187(g):
    o = g.copy(); o[g == 0] = 3; o[_enclosed(g)] = 2
    return o


# ---------------- ONNX helpers ---------------- #
def _i64(name, vals):
    return oh.make_tensor(name, I64, [len(vals)], list(vals))


def _u8s(name, v):
    return oh.make_tensor(name, U8, [1], [v])


class B:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._n = 0

    def add(self, *a, **k):
        self.nodes.append(oh.make_node(*a, **k))

    def init(self, t):
        self.inits.append(t)

    def slice(self, src, starts, ends, out):
        s = out + "_s"; e = out + "_e"; a = out + "_a"
        self.init(_i64(s, starts)); self.init(_i64(e, ends))
        self.init(_i64(a, list(range(len(starts)))))
        self.add("Slice", [src, s, e, a], [out])
        return out

    def pad(self, src, pads, val, out):
        p = out + "_p"
        self.init(_i64(p, pads)); self.init(_u8s(out + "_v", val))
        self.add("Pad", [src, p, out + "_v"], [out], mode="constant")
        return out


def _flood(b, passG, ch0G, G, K):
    """passG, ch0G uint8 [1,1,G,G]. Returns enc, reach uint8 [1,1,G,G]."""
    b.init(_u8s("one", 1)); b.init(_u8s("zero", 0))
    ring = [0, 0, 1, 1, 0, 0, 1, 1]
    b.pad(passG, ring, 1, "Ppad")
    b.add("Mul", [passG, "zero"], ["Zc"])          # GxG zeros
    b.pad("Zc", ring, 1, "Rseed")
    cur = "Rseed"
    for s in range(K):
        b.add("MaxPool", [cur], [f"v{s}"], kernel_shape=[3, 1], pads=[1, 0, 1, 0])
        b.add("Mul", [f"v{s}", "Ppad"], [f"vm{s}"])
        b.add("MaxPool", [f"vm{s}"], [f"h{s}"], kernel_shape=[1, 3], pads=[0, 1, 0, 1])
        b.add("Mul", [f"h{s}", "Ppad"], [f"R{s}"])
        cur = f"R{s}"
    b.slice(cur, [0, 0, 1, 1], [1, 1, G + 1, G + 1], "Rc")
    b.add("Sub", ["one", "Rc"], ["nRc"])
    b.add("Mul", [ch0G, "nRc"], ["enc"])
    b.add("Mul", [ch0G, "Rc"], ["reach"])
    return "enc", "reach"


def _model(b, name):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", U8, GRID_SHAPE)
    g = oh.make_graph(b.nodes, name, [x], [y], b.inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET)


def _build2(G, K):
    b = B()
    b.slice("input", [0, 0, 0, 0], [1, 1, G, G], "ch0f")
    b.slice("input", [0, 3, 0, 0], [1, 4, G, G], "grnf")
    b.add("Cast", ["ch0f"], ["ch0G"], to=U8)
    b.add("Cast", ["grnf"], ["grn"], to=U8)
    b.init(_u8s("one2", 1))
    b.add("Sub", ["one2", "grn"], ["passG"])
    enc, reach = _flood(b, "passG", "ch0G", G, K)
    b.add("Mul", ["grn", "zero"], ["Z"])
    planes = [reach, "Z", "Z", "grn", enc, "Z", "Z", "Z", "Z", "Z"]
    b.add("Concat", planes, ["cat"], axis=1)
    b.pad("cat", [0, 0, 0, 0, 0, 0, 30 - G, 30 - G], 0, "output")
    return _model(b, "b1b_flood2")


def _build187(G, K):
    b = B()
    # single uint8 copy of the whole input; everything slices from it (cheap)
    b.add("Cast", ["input"], ["Xu8"], to=U8)
    b.init(_i64("rsax", [1]))
    b.add("ReduceSum", ["input", "rsax"], ["totF"], keepdims=1)   # [1,1,30,30] fp32
    b.slice("totF", [0, 0, 0, 0], [1, 1, G, G], "totGf")
    b.add("Cast", ["totGf"], ["totG"], to=U8)
    b.slice("Xu8", [0, 0, 0, 0], [1, 1, G, G], "ch0G")
    b.init(_u8s("one2", 1))
    # passable = 1 - tot + ch0
    b.add("Sub", ["one2", "totG"], ["pm"])
    b.add("Add", ["pm", "ch0G"], ["passG"])
    enc, reach = _flood(b, "passG", "ch0G", G, K)
    b.slice("Xu8", [0, 1, 0, 0], [1, 2, G, G], "ch1u")
    b.slice("Xu8", [0, 4, 0, 0], [1, 10, G, G], "ch49u")
    b.add("Mul", ["ch0G", "zero"], ["Z"])
    # ch0=Z, ch1=ch1u, ch2=enc(red), ch3=reach(green), ch4..9=ch49u
    b.add("Concat", ["Z", "ch1u", enc, reach, "ch49u"], ["cat"], axis=1)
    b.pad("cat", [0, 0, 0, 0, 0, 0, 30 - G, 30 - G], 0, "output")
    return _model(b, "b1b_flood187")


_SPECS = [
    (_ref2, _build2, 20, 20),
    (_ref187, _build187, 25, 20),
]


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    for ref, build, G, K in _SPECS:
        if all(ref(a).shape == b.shape and np.array_equal(ref(a), b) for a, b in prs):
            yield (build.__name__, build(G, K))
            return
