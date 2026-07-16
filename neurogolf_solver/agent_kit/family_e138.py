"""FROM-SCRATCH ONNX rebuild of task138 (5daaa586), matching _cands3/t138_ref.solve.

RULE: 4 full border lines (distinct colors) form a frame [up..down]x[left..right];
interior seed pixels of `drawcolor` (== one line's color) each cast a ray toward
the matching line's edge; OUTPUT = the frame region cropped+translated to (0,0),
with rays filled.

ONNX (grader-proven ops only; no i64 ArgMax/Gather, no u8 ReduceSum/Min/Max):
  * decode -> u8 value plane v; nb = (v>0).
  * frame cols/rows: colcount/rowcount via ones-kernel QLinearConv; a full line's
    count == max count (== grid H / W).  Equal(count,max) flags exactly {left,
    right}/{up,down} (equivalent on-distribution to the ref's >=0.7*H density).
  * left/right/up/down: coordinate-as-value MaxPool (ArgMax-free).
  * line colors cL/cR/cU/cD sampled at (0,left),(0,right),(up,0),(down,0) via
    one-hot * value MaxPool (corner-free samples).
  * drawcolor dc = max interior value.  seed = interior & (v==dc) & (v>0).
  * ray = directional prefix-max (dir_reach) in each of 4 dirs; select the dir
    whose line color == dc (Equal masks, sum-of-products); clip to strict interior.
  * vray = Where(ray, dc, v).
  * CROP+TRANSLATE: shift vray to (0,0) by (up,left) with a data-dependent delta
    kernel [1,1,8,8] (Equal(krow,up)&Equal(kcol,left)) via a copy-QLinearConv,
    pads=[0,0,7,7] -> out[y,x]=vray[y+up,x+left].
  * valid = (row<=down-up)&(col<=right-left); outside -> sentinel 10 -> onehot
    decodes to all-zero background.
"""
from __future__ import annotations

import numpy as np


def build():
    import sys
    from onnx import helper as oh, TensorProto as TP
    kit = __file__.rsplit("/", 1)[0]
    if kit + "/_tools" not in sys.path:
        sys.path.insert(0, kit + "/_tools")
    from ngbuild import G, decode_head, onehot_tail, finalize

    U8, I8 = TP.UINT8, TP.INT8
    K = 8  # delta-kernel size; up,left in [1,7]

    g = G(S=30)
    z, o = "zero_u8", "one_u8"
    g.init("c29", U8, [], [29])
    g.init("rowidx", U8, [1, 1, 30, 1], list(range(30)))
    g.init("colidx", U8, [1, 1, 1, 30], list(range(30)))
    g.init("revrow", U8, [1, 1, 30, 1], list(range(29, -1, -1)))
    g.init("revcol", U8, [1, 1, 1, 30], list(range(29, -1, -1)))
    g.init("ones_col_w", I8, [1, 1, 30, 1], [1] * 30)   # column-sum kernel
    g.init("ones_row_w", I8, [1, 1, 1, 30], [1] * 30)   # row-sum kernel
    g.init("krowK", U8, [1, 1, K, 1], list(range(K)))
    g.init("kcolK", U8, [1, 1, 1, K], list(range(K)))
    g.init("r0_s", TP.INT64, [1], [0]); g.init("r0_e", TP.INT64, [1], [1])
    g.init("ax2", TP.INT64, [1], [2]); g.init("ax3", TP.INT64, [1], [3])
    nd = g.nd

    def mp(x, ks, pads=None, nm=None):
        return nd("MaxPool", [x], nm, kernel_shape=list(ks),
                  pads=list(pads or [0, 0, 0, 0]), strides=[1, 1])

    def qconv(x, w, nm=None, **a):
        return nd("QLinearConv",
                  [x, "qxs", "qxz", w, "qws", "qwz", "qys", "qyz"], nm, **a)

    def u8(cond, nm=None):
        return nd("Cast", [cond], nm, to=U8)

    # ---- decode ---- #
    v = decode_head(g, crop=False)                       # [1,1,30,30] u8
    nb = u8(nd("Greater", [v, z], "nbb"), "nb")

    # ---- frame lines: count==max flags exactly the two full lines ---- #
    colcount = qconv(nb, "ones_col_w", "colcount", kernel_shape=[30, 1])   # [1,1,1,30]
    rowcount = qconv(nb, "ones_row_w", "rowcount", kernel_shape=[1, 30])   # [1,1,30,1]
    Hmax = mp(colcount, [1, 30], nm="Hmax")
    Wmax = mp(rowcount, [30, 1], nm="Wmax")
    fc = u8(nd("Equal", [colcount, Hmax], "fcb"), "fc")   # [1,1,1,30]
    fr = u8(nd("Equal", [rowcount, Wmax], "frb"), "fr")   # [1,1,30,1]

    right = mp(nd("Mul", [fc, "colidx"], "rmul"), [1, 30], nm="right")
    left = nd("Sub", ["c29", mp(nd("Mul", [fc, "revcol"], "lmul"), [1, 30])], "left")
    down = mp(nd("Mul", [fr, "rowidx"], "dmul"), [30, 1], nm="down")
    up = nd("Sub", ["c29", mp(nd("Mul", [fr, "revrow"], "umul"), [30, 1])], "up")

    # ---- line colors (corner-free samples) ---- #
    vrow0 = nd("Slice", [v, "r0_s", "r0_e", "ax2"], "vrow0")   # [1,1,1,30]
    vcol0 = nd("Slice", [v, "r0_s", "r0_e", "ax3"], "vcol0")   # [1,1,30,1]
    ohL = u8(nd("Equal", ["colidx", left], "ohLb"))
    ohR = u8(nd("Equal", ["colidx", right], "ohRb"))
    ohU = u8(nd("Equal", ["rowidx", up], "ohUb"))
    ohD = u8(nd("Equal", ["rowidx", down], "ohDb"))
    cL = mp(nd("Mul", [vrow0, ohL], "cLm"), [1, 30], nm="cL")
    cR = mp(nd("Mul", [vrow0, ohR], "cRm"), [1, 30], nm="cR")
    cU = mp(nd("Mul", [vcol0, ohU], "cUm"), [30, 1], nm="cU")
    cD = mp(nd("Mul", [vcol0, ohD], "cDm"), [30, 1], nm="cD")

    # ---- strict interior, drawcolor, seed ---- #
    rowin = nd("And", [nd("Greater", ["rowidx", up]), nd("Less", ["rowidx", down])], "rowin")
    colin = nd("And", [nd("Greater", ["colidx", left]), nd("Less", ["colidx", right])], "colin")
    interior = nd("And", [rowin, colin], "interior")      # [1,1,30,30] bool
    intu = u8(interior, "intu")
    dc = mp(nd("Mul", [v, intu], "vint"), [30, 30], nm="dc")   # max interior value
    seed = nd("And", [nd("And", [nd("Equal", [v, dc], "vdc"), interior], "sd1"),
                      nd("Greater", [v, z], "vpos")], "seed")
    seedu = u8(seed, "seedu")

    # ---- directional prefix-max rays ---- #
    fR = mp(seedu, [1, 30], [0, 29, 0, 0], "fR")   # spread right
    fL = mp(seedu, [1, 30], [0, 0, 0, 29], "fL")   # spread left
    fD = mp(seedu, [30, 1], [29, 0, 0, 0], "fD")   # spread down
    fU = mp(seedu, [30, 1], [0, 0, 29, 0], "fU")   # spread up
    mL = u8(nd("Equal", [dc, cL], "mLb"))
    mR = u8(nd("Equal", [dc, cR], "mRb"))
    mU = u8(nd("Equal", [dc, cU], "mUb"))
    mD = u8(nd("Equal", [dc, cD], "mDb"))
    sel = nd("Add", [nd("Add", [nd("Mul", [mL, fL], "pL"), nd("Mul", [mR, fR], "pR")], "s01"),
                     nd("Add", [nd("Mul", [mU, fU], "pU"), nd("Mul", [mD, fD], "pD")], "s23")], "sel")
    fillu = nd("Mul", [sel, intu], "fillu")               # clip to strict interior
    fillb = nd("Greater", [fillu, z], "fillb")
    vray = nd("Where", [fillb, dc, v], "vray")            # [1,1,30,30] u8

    # ---- crop+translate: shift by (up,left) via delta-kernel copy-conv ---- #
    dk = nd("And", [nd("Equal", ["krowK", up], "keqr"),
                    nd("Equal", ["kcolK", left], "keqc")], "dkb")
    delta = nd("Cast", [dk], "delta", to=I8)             # [1,1,K,K] i8
    shifted = qconv(vray, delta, "shifted", kernel_shape=[K, K],
                    pads=[0, 0, K - 1, K - 1])            # out[y,x]=vray[y+up,x+left]

    # ---- valid region -> sentinel 10 outside ---- #
    oh_h = nd("Sub", [down, up], "oh_h")
    ow = nd("Sub", [right, left], "ow")
    rvalid = nd("Not", [nd("Greater", ["rowidx", oh_h])], "rvalid")
    cvalid = nd("Not", [nd("Greater", ["colidx", ow])], "cvalid")
    valid = nd("And", [rvalid, cvalid], "valid")
    finalv = nd("Where", [valid, shifted, "ten_u8"], "finalv")   # [1,1,30,30] u8

    onehot_tail(g, finalv)
    return finalize(g, name="e138")


def _to_onehot(grid):
    gg = np.array(grid, int); hh, ww = gg.shape
    x = np.zeros((1, 10, 30, 30), np.float32)
    for r in range(hh):
        for c in range(ww):
            x[0, gg[r, c], r, c] = 1.0
    return x


def candidates(ex):
    import onnxruntime as ort
    try:
        model = build()
    except Exception:
        return []
    pairs = [(e["input"], e["output"])
             for e in ex.get("train", []) + ex.get("test", [])]
    if not pairs:
        return []
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model.SerializeToString(), so)
        for gin, gout in pairs:
            if max(len(gin), len(gin[0])) > 30:
                return []
            out = (sess.run(["output"], {"input": _to_onehot(gin)})[0] > 0).astype(float)
            exp = _to_onehot(gout)
            if out.shape != exp.shape or not (out == (exp > 0)).all():
                return []
    except Exception:
        return []
    return [("e138", model)]


if __name__ == "__main__":
    import onnx
    m = build()
    onnx.save(m, __file__.rsplit("/", 1)[0] + "/_cands3/task138.onnx")
    print("nodes:", len(m.graph.node),
          "params:", sum(int(np.prod(t.dims) or 1) for t in m.graph.initializer))
