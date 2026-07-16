"""family_bpk157b — value-exact memory-golf rebuild of the task157 (6a1e5592) incumbent.

Same algorithm as out_blend13/onnx/task157.onnx (= family_bpk157), re-encoded cheaper.
Output tensor is VALUE-identical (not just after >0 threshold) on every generator input.

Re-encodings (each value-exact at its block boundary):
 1. dead code: t168 (Sub after 4th component) removed.
 2. flood fill: kernel 0.25*ones, 8x [Conv, Mul(mask)] UNCLIPPED, one Greater(.,0)+Cast at
    the end.  Support is preserved exactly each step (values stay in (0, 2.25^8], no
    overflow/underflow; positives cannot round to 0), so the final {0,1} component mask
    equals the incumbent's t53 exactly.  24 -> 18 tensors per component.
 3. seed / pass-2 cell pick: the [10,15] "selected row" tensor (t20/t489) is replaced by a
    Conv with the 0/1 single-row indicator as a [1,1,10,1] kernel, projecting the row to
    [1,1,1,15]; the single cell is then rebuilt by an outer-product broadcast Mul.
 4. sprite extraction: rectangular permutations P_row [6,10] / P_col [15,6] replace the
    square [10,10]/[15,15] ones plus the [0:6,0:6] Slice; index tables (j-i, 1000-idx)
    folded into initializers.  t200[r,c] = comp[min_row+r, min_col+c] exactly as before.
 5. ring: the 3 shifted copies (below/left/right) become one Conv with a 3-tap kernel.
    The clip on the neighbour count is dropped: ring weight -16*nb3*(1-sprite) has the
    same zero set as the incumbent ring, which is all the threshold test consumes
    (sums stay < 2048, exact in f16).
 6. match: the two Convs (sprite hits on full black > size-0.5 AND ring hits on band-masked
    black < 0.5) fuse into ONE 2-channel Conv on Concat(black, band-black) with kernel
    Concat(embed(sprite), -16*nb3*(1-sprite)):  fused = t202 - 16*t227w, and
    fused > size-0.5  <=>  t202 = size AND t227w = 0  <=>  incumbent t230.
    The band mask (incumbent c9 = rows 0-2) is built by Slice+Pad instead of 150 params.
 7. Pad-before-Conv and Slice-after-ConvTranspose folded into Conv/ConvTranspose `pads`
    attributes (t11, t12, t475 and the paint Slices gone).
 8. clips dropped where provably value-identical: before Mul-by-uniqueness-flag
    (flag=1 => paint already {0,1}; flag=0 => 0 either way), after single-cell
    ConvTranspose paints, and on the pass-1 sum (only feeds >0.5 threshold Convs which
    yield identical booleans).  The FINAL blue clip is kept (exact output values).
 9. t482 = t241 + (1-t241)(1-t478) rewritten as 1 - t478*(1-t241) (same algebra).
10. tail: output = Pad(Concat([ch0, blue, red] [1,3,10,15]), channel+spatial) as the free
    graph output.  Kills the 30x30 slices/pads and the 900-param zero plane (generator
    grids are always 10x15, so channel 2 is zero outside [0:10,0:15]).
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

    # ---------------- initializers ----------------
    inits = [
        _i64("sA", [0, 0, 0]), _i64("eA", [1, 10, 15]),
        _i64("sB", [5, 0, 0]), _i64("eB", [6, 10, 15]),
        _i64("sC", [2, 0, 0]), _i64("eC", [3, 10, 15]),
        _i64("ax", [1, 2, 3]),
        _i64("s3", [0]), _i64("e3", [3]), _i64("ax2", [2]),
        _f16("kq", 0.25 * np.ones((1, 1, 3, 3))),                # flood dilate
        _f16("kt", np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]],
                            np.float16).reshape(1, 1, 3, 3)),    # below/left/right taps
        _f16("cL", np.tril(np.ones((10, 10)), -1)),              # strict lower (c13)
        _f16("cU", np.triu(np.ones((15, 15)), 1)),               # strict upper (c21)
        _f16("c1", np.ones((1, 1, 1, 1))),
        _f16("ch", 0.5 * np.ones((1, 1, 1, 1))),
        _f16("c16", 16.0 * np.ones((1, 1, 1, 1))),
        _f16("cz", np.zeros((1, 1, 1, 1))),
        _f16("cM", np.array([1000.0])),
        _f16("wr", (1000.0 - np.arange(10)).reshape(1, 1, 10, 1)),
        _f16("wc", (1000.0 - np.arange(15)).reshape(1, 1, 1, 15)),
        _f16("pr", (np.arange(10)[None, :] - np.arange(6)[:, None]).reshape(1, 1, 6, 10)),
        _f16("pc", (np.arange(15)[:, None] - np.arange(6)[None, :]).reshape(1, 1, 15, 6)),
    ]

    # ---------------- head ----------------
    N("Slice", ["input", "sA", "eA", "ax"], "sl0")
    t4 = N("Cast", ["sl0"], "t4", to=F16)                        # black channel [1,1,10,15]
    N("Slice", ["input", "sB", "eB", "ax"], "sl5")
    t8 = N("Cast", ["sl5"], "t8", to=F16)                        # gray channel
    # band-masked black (incumbent t10 = t4 * c9, c9 = rows 0-2)
    b3 = N("Slice", [t4, "s3", "e3", "ax2"], "b3")               # [1,1,3,15]
    t10 = N("Pad", [b3], "t10", pads=[0, 0, 0, 0, 0, 0, 7, 0], value=0.0)
    cat2 = N("Concat", [t4, t10], "cat2", axis=1)                # [1,2,10,15]

    def toprow_pick(x, p):
        """indicator of topmost nonzero row of x ([1,1,10,*] {0,1}) -> [1,1,10,1]"""
        occ = N("ReduceMax", [x], f"{p}_ro", axes=[3], keepdims=1)
        cum = N("MatMul", ["cL", occ], f"{p}_rc")
        cl = N("Clip", [cum], f"{p}_rl", min=0.0, max=1.0)
        no = N("Sub", ["c1", cl], f"{p}_rn")
        return N("Mul", [occ, no], f"{p}_rp")

    def leftcol_pick(occ, p):
        """indicator of leftmost nonzero entry of occ ([1,1,1,15] {0,1}) -> [1,1,1,15]"""
        cum = N("MatMul", [occ, "cU"], f"{p}_cc")
        cl = N("Clip", [cum], f"{p}_cl", min=0.0, max=1.0)
        no = N("Sub", ["c1", cl], f"{p}_cn")
        return N("Mul", [occ, no], f"{p}_cp")

    def single_cell(x, p):
        """x [1,1,10,15] {0,1}: 1 at its topmost-then-leftmost cell -> [1,1,10,15]"""
        rp = toprow_pick(x, p)
        occ = N("Conv", [x, rp], f"{p}_co", kernel_shape=[10, 1])  # row rp of x [1,1,1,15]
        cp = leftcol_pick(occ, p)
        cc = N("Mul", [occ, cp], f"{p}_cs")                        # single 1 [1,1,1,15]
        return N("Mul", [rp, cc], f"{p}_ce")                       # outer product

    comps, uqs, mtops, sprs, paints1 = [], [], [], [], []
    mask = t8
    for c in range(4):
        p = f"a{c}"
        # ---- seed: topmost-then-leftmost cell of mask (== incumbent t29) ----
        x = single_cell(mask, p)
        # ---- flood fill: 8 unclipped dilate+mask steps ----
        for i in range(8):
            v = N("Conv", [x, "kq"], f"{p}_d{i}",
                  kernel_shape=[3, 3], pads=[1, 1, 1, 1])
            x = N("Mul", [v, mask], f"{p}_x{i}")
        fb = N("Greater", [x, "cz"], f"{p}_fb")
        comp = N("Cast", [fb], f"{p}_cm", to=F16)                # == incumbent t53
        comps.append(comp)
        if c < 3:
            mask = N("Sub", [mask, comp], f"{p}_nm")             # t54
        # ---- sprite: normalize component into 6x6 top-left frame ----
        ro = N("ReduceMax", [comp], f"{p}_mr", axes=[3], keepdims=1)
        rw = N("Mul", [ro, "wr"], f"{p}_mrw")
        rm = N("ReduceMax", [rw], f"{p}_mrm", axes=[2], keepdims=1)
        minr = N("Sub", ["cM", rm], f"{p}_ir")                   # t177 (min row; 1000 if empty)
        co = N("ReduceMax", [comp], f"{p}_mc", axes=[2], keepdims=1)
        cw = N("Mul", [co, "wc"], f"{p}_mcw")
        cm2 = N("ReduceMax", [cw], f"{p}_mcm", axes=[3], keepdims=1)
        minc = N("Sub", ["cM", cm2], f"{p}_ic")                  # t181
        rd = N("Sub", ["pr", minr], f"{p}_prd")
        ra = N("Abs", [rd], f"{p}_pra")
        rb = N("Less", [ra, "ch"], f"{p}_prb")
        Prow = N("Cast", [rb], f"{p}_prw", to=F16)               # [1,1,6,10]
        cd = N("Sub", ["pc", minc], f"{p}_pcd")
        ca = N("Abs", [cd], f"{p}_pca")
        cb = N("Less", [ca, "ch"], f"{p}_pcb")
        Pcol = N("Cast", [cb], f"{p}_pcw", to=F16)               # [1,1,15,6]
        r6 = N("MatMul", [Prow, comp], f"{p}_r6")                # [1,1,6,15]
        spr = N("MatMul", [r6, Pcol], f"{p}_sp")                 # t200 [1,1,6,6]
        sprs.append(spr)
        # ---- match: fused sprite/ring 2-channel Conv ----
        size = N("ReduceSum", [comp], f"{p}_sz", axes=[2, 3], keepdims=1)
        thr = N("Sub", [size, "ch"], f"{p}_th")                  # t203
        sprE = N("Pad", [spr], f"{p}_se",
                 pads=[0, 0, 1, 1, 0, 0, 1, 1], value=0.0)       # t206 [1,1,8,8]
        nb3 = N("Conv", [sprE, "kt"], f"{p}_n3",
                kernel_shape=[3, 3], pads=[1, 1, 1, 1])          # t223 (0..3)
        m16 = N("Mul", [sprE, "c16"], f"{p}_m16")
        n16 = N("Sub", [m16, "c16"], f"{p}_n16")                 # -16*(1-sprite)
        negr = N("Mul", [nb3, n16], f"{p}_ng")                   # -16*nb3*(1-sprite)
        K2 = N("Concat", [sprE, negr], f"{p}_K2", axis=1)        # [1,2,8,8]
        mc = N("Conv", [cat2, K2], f"{p}_mv",
               kernel_shape=[8, 8], pads=[1, 1, 6, 6])           # t202 - 16*t227w
        mb = N("Greater", [mc, thr], f"{p}_mb")
        mm = N("Cast", [mb], f"{p}_mm", to=F16)                  # t230
        # ---- pick topmost matching row; uniqueness flag ----
        pp = toprow_pick(mm, p + "q")
        mtop = N("Mul", [mm, pp], f"{p}_mt")                     # t236
        mtops.append(mtop)
        ms = N("ReduceSum", [mtop], f"{p}_msu", axes=[2, 3], keepdims=1)
        md = N("Sub", [ms, "c1"], f"{p}_md")
        ma = N("Abs", [md], f"{p}_ma")
        ub = N("Less", [ma, "ch"], f"{p}_ub")
        uq = N("Cast", [ub], f"{p}_uq", to=F16)                  # t241
        uqs.append(uq)
        # ---- pass-1 paint (only when unique) ----
        pt = N("ConvTranspose", [mtop, spr], f"{p}_pt",
               kernel_shape=[6, 6], pads=[0, 0, 5, 5])           # t246 (clip dropped)
        paints1.append(N("Mul", [pt, uq], f"{p}_p1"))            # t248

    s12 = N("Add", [paints1[0], paints1[1]], "s12")
    s34 = N("Add", [paints1[2], paints1[3]], "s34")
    pass1 = N("Add", [s12, s34], "ps1")                          # t473 (clip dropped)

    # ---------------- pass 2 ----------------
    paints2 = []
    for c in range(4):
        p = f"b{c}"
        cov = N("Conv", [pass1, sprs[c]], f"{p}_cv",
                kernel_shape=[6, 6], pads=[0, 0, 5, 5])          # t476
        cb2 = N("Greater", [cov, "ch"], f"{p}_cb")
        cvf = N("Cast", [cb2], f"{p}_cf", to=F16)                # t478
        nuq = N("Sub", ["c1", uqs[c]], f"{p}_nu")                # t479
        cnu = N("Mul", [cvf, nuq], f"{p}_cn2")
        allow = N("Sub", ["c1", cnu], f"{p}_al")                 # t482
        cand = N("Mul", [mtops[c], allow], f"{p}_ca")            # t483
        cell = single_cell(cand, p)                              # t495 (single cell)
        paints2.append(N("ConvTranspose", [cell, sprs[c]], f"{p}_p2",
                         kernel_shape=[6, 6], pads=[0, 0, 5, 5]))  # t500 (clip dropped)

    u12 = N("Add", [paints2[0], paints2[1]], "u12")
    u34 = N("Add", [paints2[2], paints2[3]], "u34")
    tsum = N("Add", [u12, u34], "ts")                            # t582
    blue = N("Clip", [tsum], "blu", min=0.0, max=1.0)            # t583 (kept: exact output)
    bg = N("Add", [t4, t8], "bg")                                # t584
    ch0 = N("Sub", [bg, blue], "c0o")                            # t585
    N("Slice", ["input", "sC", "eC", "ax"], "sl2")
    ch2 = N("Cast", ["sl2"], "c2o", to=F16)                      # red channel
    cc = N("Concat", [ch0, blue, ch2], "cc", axis=1)             # [1,3,10,15]
    N("Pad", [cc], "output",
      pads=[0, 0, 0, 0, 0, 7, 20, 15], value=0.0)                # -> [1,10,30,30] (free)

    graph = oh.make_graph(
        nodes, "bpk157b",
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
    return [("bpk157b", _MODEL)]
