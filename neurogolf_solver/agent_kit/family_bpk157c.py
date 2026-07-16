"""family_bpk157c — value-exact canvas/structure shrink of the task157 (6a1e5592) incumbent.

Same algorithm as out_blend15/onnx/task157.onnx (= family_bpk157b), re-encoded against
generator invariants (all verified on official data + fresh gens; flood bound PROVEN by
brute force over every 4-connected creature shape in a 4x4 box):

  G1. grids are always 10x15; input colors are {0, 2(red), 5(gray)} only.
  G2. gray creatures live in rows 6..9 (grayrows = 9 - max_row, tall <= 4), are
      column-separated by >= 1 empty column (all gray boxes reach row 9, so
      common.overlaps(...,1) forces column separation), fit in a 4x4 box, are
      4-connected and contain their box origin (continuous_creature starts at (0,0)).
  G3. ch0 (color-0 plane) is deterministic except rows 1..2:
      row 0 = red everywhere -> 0; rows 1..2 = 1 - red (creature cells in the band);
      rows 3..5 = all ones; rows 6..9 = 1 - gray.
  G4. matches: the true creature match sits at row bluerow in {1,2}; row 0 has no
      0-cells, so the topmost match row is always <= 2 (empty comp => all-ones row 0).
  G5. creature bounding box <= 4x4 -> the incumbent's 6x6 sprite canvas rows/cols 4..5
      and its 8x8 padded-kernel rows/cols 6..7 are identically zero.

Re-encodings (each block value-exact vs the incumbent tensor it replaces):
  1. head: gray plane sliced to [1,1,4,15] (rows 6..9); ch0 rebuilt from the 2-row band
     slice + constant rows + (1 - gray row 6); the fused match input becomes
     [1,2,7,15] (kernel rows >= 5 are provably zero, so rows 7..9 are never read).
  2. seed cell: same toprow/leftcol cumulative picks, on the 4-row plane.
  3. flood fill: the 8x [Conv 3x3, Mul] 2-D flood is replaced by a 3-step 1-D column
     flood on the column-occupancy vector (creatures are column-separated intervals of
     width <= 4 whose leftmost column is the seed column); comp = mask * colmask.
     Proven equal to the incumbent's 8-step flood (max 8-geodesic over all creature
     shapes = 7 <= 8, no diagonal leaks across a >= 1 column gap).
  4. sprite: the one-hot (top row, min col) cell of comp (top row from the seed pick,
     min col = leftmost column of the colmask) is used as a dynamic Conv kernel to
     shift comp to the origin; a [4,4] sprite canvas replaces the incumbent's [6,6]
     (G5).  NB: official hand-written examples have creatures without the box-origin
     cell, so min col must be taken over the whole comp, not the seed row.
  5. match/pick/pass-2 batched over the 4 components as channels: one 4-out-channel
     Conv against Concat(sprite,ring) kernels, channelwise toprow/leftcol picks
     (MatMul broadcasts over channels), one 4-in-channel ConvTranspose replaces the
     four paints + Add tree (sums are small ints, exact in f16 in any order).
  6. match planes cropped to rows 0..3 (G4); paint planes to [6,15] (stamp row <= 2 +
     sprite rows <= 3).
  7. tail: bg = t4 + t8 == [0-row; band; ones(7x15)]; red ch == [ones-row; 1-band;
     zeros] -- both rebuilt by Concat with constant initializers; output = same free
     Pad to [1,10,30,30].
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

import family_gw2_f16 as _gw2

F16 = TP.FLOAT16


def _f16(name, arr):
    a = np.asarray(arr, dtype=np.float16)
    return oh.make_tensor(name, F16, list(a.shape), a.tobytes(), raw=True)


def _i64(name, vals):
    return oh.make_tensor(name, TP.INT64, [len(vals)], list(vals))


def _build():
    nodes = []
    k = [0]

    def N(op, ins, out, **attrs):
        k[0] += 1
        nodes.append(oh.make_node(op, ins, [out], name=f"n{k[0]}", **attrs))
        return out

    inits = [
        # slice indices
        _i64("stA", [0, 1, 0]), _i64("enA", [1, 3, 15]), _i64("axA", [1, 2, 3]),
        _i64("stB", [5, 6, 0]), _i64("enB", [6, 10, 15]),
        _i64("st0", [0]), _i64("en1", [1]), _i64("ax2", [2]),
        _i64("en4", [4]), _i64("ax3", [3]),
        # small f16 constants
        _f16("k13", 0.25 * np.ones((1, 1, 1, 3))),               # 1-D flood kernel
        _f16("kt16", 16.0 * np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]],
                                     np.float16).reshape(1, 1, 3, 3)),
        _f16("cL4", np.tril(np.ones((4, 4)), -1)),               # strict lower 4x4
        _f16("cU", np.triu(np.ones((15, 15)), 1)),               # strict upper 15x15
        _f16("c1", np.ones((1, 1, 1, 1))),
        _f16("ch", 0.5 * np.ones((1, 1, 1, 1))),
        _f16("cz", np.zeros((1, 1, 1, 1))),
        # constant plane rows (G3)
        _f16("zr1", np.zeros((1, 1, 1, 15))),
        _f16("on3", np.ones((1, 1, 3, 15))),
        _f16("on7", np.ones((1, 1, 7, 15))),
        _f16("or1", np.ones((1, 1, 1, 15))),
        _f16("zr7", np.zeros((1, 1, 7, 15))),
    ]

    # ---------------- head ----------------
    N("Slice", ["input", "stA", "enA", "axA"], "sl12")           # ch0 rows 1..2 (f32)
    bnd = N("Cast", ["sl12"], "bnd", to=F16)                     # [1,1,2,15] band black
    N("Slice", ["input", "stB", "enB", "axA"], "sl5")            # ch5 rows 6..9 (f32)
    t8 = N("Cast", ["sl5"], "t8", to=F16)                        # [1,1,4,15] gray
    t8r = N("Slice", [t8, "st0", "en1", "ax2"], "t8r")           # gray row 6 [1,1,1,15]
    iv1 = N("Sub", ["c1", t8r], "iv1")                           # ch0 row 6
    t4c = N("Concat", ["zr1", bnd, "on3", iv1], "t4c", axis=2)   # ch0 rows 0..6 [1,1,7,15]
    t10c = N("Pad", [bnd], "t10c", pads=[0, 0, 1, 0, 0, 0, 4, 0])  # band mask rows 0..6
    cat2 = N("Concat", [t4c, t10c], "cat2", axis=1)              # [1,2,7,15]

    def toprow(x, p, h):
        """[1,C,h,1] indicator of topmost nonzero row per channel of x [1,C,h,W]."""
        ro = N("ReduceMax", [x], f"{p}ro", axes=[3], keepdims=1)
        rc = N("MatMul", ["cL4", ro], f"{p}rc")
        rl = N("Clip", [rc], f"{p}rl", min=0.0, max=1.0)
        rn = N("Sub", ["c1", rl], f"{p}rn")
        return N("Mul", [ro, rn], f"{p}rp")

    def leftcol(occ, p):
        """[1,C,1,15] indicator of leftmost nonzero entry per channel of occ."""
        cc = N("MatMul", [occ, "cU"], f"{p}cc")
        cl = N("Clip", [cc], f"{p}cl", min=0.0, max=1.0)
        cn = N("Sub", ["c1", cl], f"{p}cn")
        return N("Mul", [occ, cn], f"{p}cp")

    # ---------------- per-component seed / flood / sprite ----------------
    s44s, szs = [], []
    mask = t8
    for c in range(4):
        p = f"a{c}"
        rp = toprow(mask, p, 4)                                  # topmost mask row
        co = N("Conv", [mask, rp], f"{p}co", kernel_shape=[4, 1])  # that row [1,1,1,15]
        cp = leftcol(co, p)                                      # seed col one-hot
        oc = N("ReduceMax", [mask], f"{p}oc", axes=[2], keepdims=1)  # col occupancy
        x = cp
        for i in range(3):                                       # 1-D column flood
            v = N("Conv", [x, "k13"], f"{p}d{i}",
                  kernel_shape=[1, 3], pads=[0, 1, 0, 1])
            x = N("Mul", [v, oc], f"{p}x{i}")
        fb = N("Greater", [x, "cz"], f"{p}fb")
        cm = N("Cast", [fb], f"{p}cm", to=F16)                   # component columns
        comp = N("Mul", [mask, cm], f"{p}cq")                    # == incumbent comp rows 6..9
        if c < 3:
            mask = N("Sub", [mask, comp], f"{p}nm")
        cp2 = leftcol(cm, p + "e")                               # min col of comp
        ce2 = N("Mul", [rp, cp2], f"{p}ce")                      # (top row, min col) cell
        sf = N("Conv", [comp, ce2], f"{p}sf",
               kernel_shape=[4, 15], pads=[0, 0, 3, 14])         # comp shifted to origin
        s44s.append(N("Slice", [sf, "st0", "en4", "ax3"], f"{p}s4"))  # sprite [1,1,4,4]
        szs.append(N("ReduceSum", [comp], f"{p}sz", axes=[2, 3], keepdims=1))

    # ---------------- batched sprite kernels ----------------
    s4c = N("Concat", s44s, "s4c", axis=0)                       # [4,1,4,4]
    sprE = N("Pad", ["s4c"], "sprE", pads=[0, 0, 1, 1, 0, 0, 1, 1])  # [4,1,6,6]
    spr4 = s4c                                                   # paint kernel [4,1,4,4]
    n3 = N("Conv", [sprE, "kt16"], "n3",
           kernel_shape=[3, 3], pads=[1, 1, 1, 1])               # 16*(below/left/right count)
    sm1 = N("Sub", [sprE, "c1"], "sm1")                          # sprite - 1
    ng = N("Mul", [n3, sm1], "ng")                               # -16*nb3*(1-sprite)
    K2 = N("Concat", [sprE, ng], "K2", axis=1)                   # [4,2,6,6]
    szc = N("Concat", szs, "szc", axis=1)                        # [1,4,1,1]
    thr = N("Sub", [szc, "ch"], "thr")

    # ---------------- fused match (rows 0..3 suffice, G4) ----------------
    mv = N("Conv", [cat2, K2], "mv",
           kernel_shape=[6, 6], pads=[1, 1, 1, 4])               # [1,4,4,15]
    mb = N("Greater", [mv, thr], "mb")
    mm = N("Cast", [mb], "mm", to=F16)                           # == incumbent mm rows 0..3

    # ---------------- pass-1 pick + paint ----------------
    qrp = toprow(mm, "q", 4)
    mt = N("Mul", [mm, qrp], "mt")                               # topmost matching row
    ms = N("ReduceSum", [mt], "ms", axes=[2, 3], keepdims=1)
    md = N("Sub", [ms, "c1"], "md")
    ma = N("Abs", [md], "ma")
    ub = N("Less", [ma, "ch"], "ub")
    uq = N("Cast", [ub], "uq", to=F16)                           # uniqueness [1,4,1,1]
    mtu = N("Mul", [mt, uq], "mtu")
    ps1 = N("ConvTranspose", [mtu, spr4], "ps1",
            kernel_shape=[4, 4], pads=[0, 0, 1, 3])              # pass-1 sum [1,1,6,15]

    # ---------------- pass-2 ----------------
    cov = N("Conv", [ps1, spr4], "cov",
            kernel_shape=[4, 4], pads=[0, 0, 1, 3])              # coverage [1,4,4,15]
    cb = N("Greater", [cov, "ch"], "cb")
    cf = N("Cast", [cb], "cf", to=F16)
    nu = N("Sub", ["c1", uq], "nu")
    c2m = N("Mul", [cf, nu], "c2m")
    al = N("Sub", ["c1", c2m], "al")                             # allow
    ca = N("Mul", [mt, al], "ca")                                # candidates
    brp = toprow(ca, "b", 4)
    crp = N("Mul", [ca, brp], "crp")
    co4 = N("ReduceMax", [crp], "co4", axes=[2], keepdims=1)     # selected row [1,4,1,15]
    bcp = leftcol(co4, "b")
    ce4 = N("Mul", [brp, bcp], "ce4")                            # cells [1,4,4,15]
    p2s = N("ConvTranspose", [ce4, spr4], "p2s",
            kernel_shape=[4, 4], pads=[0, 0, 1, 3])              # pass-2 sum [1,1,6,15]

    # ---------------- tail ----------------
    blu = N("Clip", [p2s], "blu", min=0.0, max=1.0)              # blue [1,1,6,15]
    pbl = N("Pad", [blu], "pbl", pads=[0, 0, 0, 0, 0, 0, 4, 0])  # [1,1,10,15]
    bg = N("Concat", ["zr1", bnd, "on7"], "bg", axis=2)          # t4+t8 [1,1,10,15]
    c0o = N("Sub", [bg, pbl], "c0o")
    nb2 = N("Sub", ["c1", bnd], "nb2")                           # red rows 1..2
    ch2 = N("Concat", ["or1", nb2, "zr7"], "ch2", axis=2)        # red ch [1,1,10,15]
    cc = N("Concat", [c0o, pbl, ch2], "cc", axis=1)              # [1,3,10,15]
    N("Pad", ["cc"], "output",
      pads=[0, 0, 0, 0, 0, 7, 20, 15])                           # -> [1,10,30,30] (free)

    graph = oh.make_graph(
        nodes, "bpk157c",
        [oh.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [oh.make_tensor_value_info("output", F16, [1, 10, 30, 30])],
        inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


_MODEL = None


def candidates(examples):
    global _MODEL
    fp = _gw2._fp(examples.get("train", []))
    e = _gw2._E.get(fp)
    if not e or e[0] != 157:
        return []
    if _MODEL is None:
        _MODEL = _build()
    return [("bpk157c", _MODEL)]
