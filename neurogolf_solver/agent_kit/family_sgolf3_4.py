"""family_sgolf3_4 -- cheaper EXACT solvers via FIXED-SIZE work-area rebuilds.

Some FIXED-size baselines run their whole computation on the full 30x30 tensor even
though every train+test+arc-gen grid is one fixed square GxG. Their builders hardcode
30 in every shift/slice, so the generic "patch a size global + crop" trick (family_sgolf_0)
cannot shrink them. Here we instead REBUILD the byte-identical algorithm at a small work
resolution S (declared input sliced to [1,10,S,S], all interior shifts/reductions done at
S, result padded back to [1,10,30,30]). Same ops, same numerics, far fewer intermediate
bytes -> more points. Correctness is re-checked EXACTLY against the family's own numpy
reference (detection) and the grader validates on every split, so a wrong S is rejected.

Covered target:
  - task 34 (t34, family_crk3_2): thick diagonal beams from a marked 2x2 block. Its build
    uses only shift-by-Pad/Slice + channel-vector constants (NO 30x30 initializers), so it
    re-expresses cleanly at any S. Baseline is full-30 (12.10 pts); at S=G=9 the interior
    is 1/11th the area.

ANTI-OVERFIT: fire ONLY when every train+test+arc-gen input AND output is one square GxG
(a truly fixed-size generator). Emit S = G .. G+3 and let the harness keep the cheapest
EXACT one; the grader re-validates exactness on all splits.
"""
from __future__ import annotations

import numpy as np
import onnx
import onnx.checker as _chk
from onnx import helper as oh, TensorProto as TP

import family_crk3_2 as _c32
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE


# --------------------------------------------------------------------------- #
# size-parametrized graph accumulator (mirror of family_crk3_2._G at width S)  #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self, S):
        self.S = S
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, TP.INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    # shift content of a [1,C,S,S] tensor by (dr,dc); zero-fill at edges
    def shift(self, x, dr, dc):
        S = self.S
        pt = max(dr, 0); pb = max(-dr, 0)
        pl = max(dc, 0); pr = max(-dc, 0)
        pads = [0, 0, pt, pl, 0, 0, pb, pr]
        p = self.nd("Pad", [x], mode="constant", pads=pads, value=0.0)
        s = self.i64([0, 0, pb, pr]); e = self.i64([1, 99, pb + S, pr + S])
        ax = self.i64([0, 1, 2, 3])
        return self.nd("Slice", [p, s, e, ax])


def _slc(g, src, lo, hi, axis):
    s = g.i64([lo]); e = g.i64([hi]); a = g.i64([axis])
    return g.nd("Slice", [src, s, e, a])


# --------------------------------------------------------------------------- #
# t34 rebuilt at size S (interior operates on the SxS work area)               #
# --------------------------------------------------------------------------- #
def _t34_build_S(S):
    g = _G(S)
    inp = "inp_s"                                                    # crop-wrapper feeds [1,10,S,S]
    grid = g.nd("ReduceSum", [inp], axes=[1], keepdims=1)
    col = _slc(g, inp, 1, 10, 1)
    B = g.nd("ReduceSum", [col], axes=[1], keepdims=1)
    T = _slc(g, inp, 2, 3, 1)
    cnt = g.nd("ReduceSum", [inp], axes=[2, 3], keepdims=1)
    half = g.f([1, 1, 1, 1], [0.5])
    pos = g.nd("Cast", [g.nd("Greater", [cnt, half])], to=F)
    allow = g.f([1, 10, 1, 1], [0, 1, 0, 1, 1, 1, 1, 1, 1, 1])
    chsel = g.nd("Mul", [pos, allow])
    Br = g.shift(B, 0, -1)
    Bl = g.shift(B, 0, 1)
    Bd = g.shift(B, -1, 0)
    Bu = g.shift(B, 1, 0)

    def enable(a, b):
        p = g.nd("Mul", [T, a]); p = g.nd("Mul", [p, b])
        return g.nd("ReduceMax", [p], axes=[2, 3], keepdims=1)
    e_ul = enable(Br, Bd)
    e_ur = enable(Bl, Bd)
    e_dl = enable(Br, Bu)
    e_dr = enable(Bl, Bu)

    # doubling shifts must cover the full SxS propagation
    steps = []
    s = 1
    while s < S:
        steps.append(s)
        s *= 2

    def prop(seed, dr, dc):
        beam = seed
        for st in steps:
            sh = g.shift(beam, dr * st, dc * st)
            beam = g.nd("Max", [beam, sh])
        return beam
    beams = []
    for (dr, dc), en in (((-1, -1), e_ul), ((-1, 1), e_ur), ((1, -1), e_dl), ((1, 1), e_dr)):
        bm = prop(B, dr, dc)
        beams.append(g.nd("Mul", [bm, en]))
    tot = B
    for bm in beams:
        tot = g.nd("Max", [tot, bm])
    tot = g.nd("Mul", [tot, grid])
    one = g.f([1, 1, 1, 1], [1.0])
    notbeam = g.nd("Sub", [one, tot])
    bg = g.nd("Mul", [grid, notbeam])
    bgvec = g.f([1, 10, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    bgpart = g.nd("Mul", [bg, bgvec])
    fpart = g.nd("Mul", [tot, chsel])
    g.nd("Add", [bgpart, fpart], "out_s")                           # crop-wrapper pads -> [1,10,30,30]
    return g


# --------------------------------------------------------------------------- #
# crop-wrap: Slice input -> [1,10,S,S], run SxS graph, Pad out_s -> 30x30       #
# --------------------------------------------------------------------------- #
def _wrap(g, S):
    nodes = list(g.nodes)
    inits = list(g.inits)
    inits += [
        oh.make_tensor("cwS", TP.INT64, [2], [0, 0]),
        oh.make_tensor("cwE", TP.INT64, [2], [S, S]),
        oh.make_tensor("cwA", TP.INT64, [2], [2, 3]),
    ]
    nodes.insert(0, oh.make_node("Slice", ["input", "cwS", "cwE", "cwA"], ["inp_s"], name="cw_s"))
    nodes.append(oh.make_node("Pad", ["out_s"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], name="cw_p"))
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], inits)
    m = oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    _chk.check_model(m, full_check=True)
    return m


def _grid_size(examples):
    """Return G if every train+test+arc-gen input AND output is the same square GxG, else None."""
    sizes = set()
    saw = False
    for sec in ("train", "test", "arc-gen"):
        for e in examples.get(sec, []):
            try:
                a = np.array(e["input"], int)
                b = np.array(e["output"], int)
            except Exception:
                return None
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return None
            sizes.add(a.shape)
            sizes.add(b.shape)
            saw = True
    if not saw or len(sizes) != 1:
        return None
    (h, w), = sizes
    if h != w or not (1 <= h <= 27):
        return None
    return h


def candidates(examples):
    G = _grid_size(examples)
    if G is None:
        return []
    # reuse family_crk3_2's exact numpy reference for t34 as the detector
    prs = _c32._pairs(examples)
    if not prs:
        return []
    try:
        matched = _c32._t34_detect(prs)
    except Exception:
        matched = False
    if not matched:
        return []
    out = []
    for S in range(G, min(G + 4, 30)):
        try:
            g = _t34_build_S(S)
            m = _wrap(g, S)
        except Exception:
            continue
        out.append((f"t34_S{S}", m))
    return out
