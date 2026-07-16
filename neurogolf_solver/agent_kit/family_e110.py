"""family_e110 -- residue-class completion for task110 (arc-gen 484b58aa).

task110 is the SAME texture family as task017 (0dfd9992): the whole grid is a
doubly-periodic texture v(r,c) = ((offset+r)%L - L//2)^2 + ((offset+c)%L - L//2)^2)
mod m + 1 with period L in [4,9]; black rectangles hide parts. (The generator's
`colors` override never fires in random generation, so the pure-formula texture
is what fresh/private samples carry.)

So the task017 solution transfers verbatim: every residue class mod L holds one
value; completion = a dilated MaxPool over the class; the correct L is the smallest
whose completion agrees with every visible cell. Canvas is 29 (vs 21 for t017);
the shared-Pad geometry recomputes to P=14 -> 57x57 (same 3249B as t017 by luck)
with per-L (k, pp) solving (k-1)*L/2 == pp + 14, reach (k-1)/2 >= 2:
  L4:k9pp2  L5:k7pp1  L6:k7pp4  L7:k5pp0  L8:k5pp2  L9:k5pp4
Verified 0/2000 fresh (cleaner than t017 -- the 29-canvas dodges the 4-cutout
conspiracy). Built with the shared ngbuild primitives.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tools"))
import onnx
from ngbuild import G, decode_head, onehot_tail, u8_reduce_max, finalize, I64, U8

S = 29
P = 14
KS = {4: (9, 2), 5: (7, 1), 6: (7, 4), 7: (5, 0), 8: (5, 2), 9: (5, 4)}


def build():
    g = G(S=S, opset=17)
    v = decode_head(g, crop=True)                                 # [1,1,29,29] u8
    is_cut = g.nd("Equal", [v, "zero_u8"], "is_cut")
    viz1 = g.nd("Where", [is_cut, "zero_u8", "one_u8"], "viz1")    # 1 at visible
    g.init("padP", I64, [8], [0, 0, P, P, 0, 0, P, P])
    vpad = g.nd("Pad", [v, "padP", "zero_u8"], "vpad")            # [1,1,57,57]
    comps, oks = {}, {}
    for L, (k, pp) in KS.items():
        comp = g.nd("MaxPool", [vpad], "comp%d" % L,
                    kernel_shape=[k, k], dilations=[L, L],
                    pads=[pp, pp, pp, pp], strides=[1, 1])         # [1,1,29,29] u8
        eq = g.nd("Equal", [comp, v], "eq%d" % L)
        bad = g.nd("Where", [eq, "zero_u8", "viz1"], "bad%d" % L)
        badmax = u8_reduce_max(g, bad, (S, S), "bm%d" % L)
        oks[L] = g.nd("Equal", [badmax, "zero_u8"], "ok%d" % L)
        comps[L] = comp
    sel = comps[9]
    for L in (8, 7, 6, 5, 4):
        sel = g.nd("Where", [oks[L], comps[L], sel], "sel%d" % L)
    onehot_tail(g, sel)
    return finalize(g, "e110")


def candidates(ex):
    # signature-gated adoption hook (mirrors sibling families); the integrator
    # only emits this when the task matches. Kept simple: build unconditionally.
    try:
        return [("e110_residue", build())]
    except Exception:
        return []


if __name__ == "__main__":
    m = build()
    onnx.save(m, "/private/tmp/claude-501/e110.onnx")
    print("nodes:", len(m.graph.node))
