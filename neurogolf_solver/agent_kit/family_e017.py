"""ENTROPY rebuild of task017 (arc-gen hash 0dfd9992).

Rule: the whole 21x21 grid is a doubly-periodic texture v(r,c)=((o+r)%L-L//2)^2+
((o+c)%L-L//2)^2) % mod + 1 with period L in [4,9]; black rectangles (<=5, each
<=5x5) hide parts.  Output = the texture restored.

Constraint exploited (task145 lesson): we never infer (offset, mod) or evaluate
the formula.  Periodicity alone means every cell's value lives in its residue
class mod L, and every class always keeps >=1 visible member (cutouts are too
small to cover a class except a ~0.07% 4-cutout conspiracy at L=9, which is
information-theoretically unrecoverable).  So:

  * completion under candidate L = ONE dilated MaxPool (dilation L): the max
    over the residue class = the class's unique visible value (cutouts are 0).
  * candidate L is correct iff its completion agrees with every visible cell.
  * any consistent L yields the same completion; prefer the smallest.

ORT requires MaxPool pads < kernel, so the +-20 reach cannot come from pool
pads.  Instead ONE shared Pad(v, 18) -> [1,1,57,57] u8 feeds all six pools;
per-L kernel k and small legal pads pp are chosen so the center tap of every
window sits exactly on the output cell: (k-1)*L/2 == pp + 18, pp < k.

~46 nodes, ~20.1k bytes static memory (vs incumbent bpk017 49.5k), and CLEANER:
2/3000 fresh dirt (inherent 4-cutout conspiracies at L=9) vs incumbent 0.25%.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

IR_VERSION = 8
OPSET = [oh.make_opsetid("", 17)]
F32, U8, I64, BOOL = TP.FLOAT, TP.UINT8, TP.INT64, TP.BOOL

# L -> (kernel, pp) pooling the shared [57,57] pad: (k-1)*L/2 == pp + 18, pp < k
KS = {4: (11, 2), 5: (9, 2), 6: (7, 0), 7: (7, 3), 8: (7, 6), 9: (5, 0)}


class _G:
    def __init__(self):
        self.nodes, self.inits, self._k = [], [], 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}_{self._k}"

    def init(self, name, dt, dims, vals):
        self.inits.append(oh.make_tensor(name, dt, list(dims), list(vals)))
        return name

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm(op.lower())
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def build():
    g = _G()
    S = 21  # generator fixes size=21

    g.init("dec_w", F32, [1, 10, 1, 1], [float(i) for i in range(10)])
    g.init("s0", I64, [2], [0, 0]); g.init("s1", I64, [2], [S, S])
    g.init("ax23", I64, [2], [2, 3])
    g.init("zero_u8", U8, [], [0]); g.init("one_u8", U8, [], [1])
    g.init("ten_u8", U8, [], [10])
    g.init("pad30", I64, [8], [0, 0, 0, 0, 0, 0, 30 - S, 30 - S])
    g.init("pad57", I64, [8], [0, 0, 18, 18, 0, 0, 18, 18])
    g.init("col_idx", U8, [1, 10, 1, 1], list(range(10)))

    # ---- decode one-hot -> value grid, cropped to 21x21, uint8 ------------ #
    v30 = g.nd("Conv", ["input", "dec_w"], "v30")                    # [1,1,30,30] f32
    v30u = g.nd("Cast", [v30], "v30_u8", to=U8)                      # [1,1,30,30] u8
    v = g.nd("Slice", [v30u, "s0", "s1", "ax23"], "v_u8")            # [1,1,21,21] u8
    is_cut = g.nd("Equal", [v, "zero_u8"], "is_cut")                 # bool
    viz1 = g.nd("Where", [is_cut, "zero_u8", "one_u8"], "viz1")      # u8: 1 at visible
    vpad = g.nd("Pad", [v, "pad57", "zero_u8"], "vpad")              # [1,1,57,57] u8

    # ---- per-L: dilated-MaxPool completion + visible-consistency ---------- #
    comps, oks = {}, {}
    for L, (k, pp) in KS.items():
        comp = g.nd("MaxPool", [vpad], f"comp{L}",
                    kernel_shape=[k, k], dilations=[L, L],
                    pads=[pp, pp, pp, pp], strides=[1, 1])           # [1,1,21,21] u8
        eq = g.nd("Equal", [comp, v], f"eq{L}")                      # bool
        bad = g.nd("Where", [eq, "zero_u8", "viz1"], f"bad{L}")      # u8: 1 at visible mismatch
        badmax = g.nd("MaxPool", [bad], f"badmax{L}",
                      kernel_shape=[S, S], strides=[1, 1])           # [1,1,1,1] u8
        oks[L] = g.nd("Equal", [badmax, "zero_u8"], f"ok{L}")        # bool scalar
        comps[L] = comp

    # ---- select smallest consistent L ------------------------------------- #
    sel = comps[9]
    for L in (8, 7, 6, 5, 4):
        sel = g.nd("Where", [oks[L], comps[L], sel], f"sel{L}")

    # ---- re-encode: pad with sentinel 10, one-hot via Equal ---------------- #
    c30 = g.nd("Pad", [sel, "pad30", "ten_u8"], "c30")               # [1,1,30,30] u8
    g.nodes.append(oh.make_node("Equal", [c30, "col_idx"], ["output"]))

    graph = oh.make_graph(
        g.nodes, "e017",
        [oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [oh.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        g.inits)
    m = oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET)
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    m = build()
    onnx.save(m, "/private/tmp/claude-501/e017.onnx")
    print("nodes:", len(m.graph.node))
