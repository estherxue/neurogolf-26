"""ENTROPY rebuild of task319 (arc-gen hash ce602527).

Rule: background bg + two sprites (colors a,b; dense conway sprites, 3..5 x
3..5 bbox, every bbox line occupied, diagonally connected) + a x2-magnified
copy of sprite 0 in a fourth color, ALWAYS partially clipped by one grid edge
(clip k>=1 pixels off exactly one side).  Output = sprite 0's bbox crop.

Pipeline (numpy-verified 99.77% on 3000 fresh gens; misses are inherent
ambiguities where both sprites are valid interpretations):

 1. DECODE  v = Conv(input, 0..9) -> u8, cropped to 19x19 (grids are 15..19).
 2. COLORS  counts = ReduceSum(input); bg = argmax; the 3 object colors by
    count-rank (key = count + idx/16 breaks ties; floor(key) = count).
 3. PER CANDIDATE (3 masks): bbox scalars via desc/asc coordinate MaxPools;
    block phase: windows start at rmin-adj where adj = (span odd) & (rmin==0)
    (the clipped side is flush; an odd span means a half-block line there).
    s4 = aligned 2x2 sums via TWO strided QLinearConvs with runtime 2-tap
    one-hot kernels (taps t,t+1; t = rmin+2-adj; pads top/left 2).
    blocky (mag test) = no odd s4 AND some s4==4.   D-cells = (s4 >= 1).
    X0 = translated 5x5 bbox crop via two 1-tap one-hot QLinearConvs.
 4. MAG = first blocky candidate.  D u8 [5,5]; nD by ones-QLinearConv.
    Allowed clip sides = grid-edge-flush sides of the mag bbox.
 5. MATCH: candidate is sprite-0-compatible for side s iff exists a placement
    of D in X0 with X0 == D everywhere except strictly beyond D's s-edge.
    One QLinearConv per side on X0: 9x9 u8 weight (zp 16): D-cells 17 (+1),
    penalty 0 (-16), free side 16 (0); max score == nD  <->  match.
 6. WINNER among the two non-mag: matching candidate with the smaller count
    (tie/none -> first).  Output values composed from its X0 + bbox h/w with
    sentinel 10 outside; one-hot via Equal; bool Pad to 30x30.

All logic bits are u8 0/1 (ORT Where lacks int8/bool value support):
AND = Mul, OR = Add + Greater, select = Where(bool cond, u8, u8).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

IR_VERSION = 8
OPSET = [oh.make_opsetid("", 17)]
F32, U8, I64, BOOL = TP.FLOAT, TP.UINT8, TP.INT64, TP.BOOL


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

    def qlc(self, x, w, out=None, wz="wzp0", **a):
        return self.nd("QLinearConv",
                       [x, "xsc", "xzp", w, "wsc", wz, "ysc", "yzp"],
                       out, **a)


def build():
    g = _G()

    # ---------------- constants ---------------- #
    g.init("dec_w", F32, [1, 10, 1, 1], [float(i) for i in range(10)])
    g.init("idx16", F32, [1, 10, 1, 1], [i / 16.0 for i in range(10)])
    g.init("colidx", U8, [1, 10, 1, 1], list(range(10)))
    g.init("s_st", I64, [2], [0, 0])
    g.init("s_en", I64, [2], [19, 19])
    g.init("ax23", I64, [2], [2, 3])
    g.init("idx19r", U8, [1, 1, 19, 1], list(range(19)))
    g.init("idx19c", U8, [1, 1, 1, 19], list(range(19)))
    g.init("asc19r", U8, [1, 1, 19, 1], list(range(1, 20)))
    g.init("asc19c", U8, [1, 1, 1, 19], list(range(1, 20)))
    g.init("desc19r", U8, [1, 1, 19, 1], list(range(19, 0, -1)))
    g.init("desc19c", U8, [1, 1, 1, 19], list(range(19, 0, -1)))
    g.init("idx21r", U8, [1, 1, 21, 1], list(range(21)))
    g.init("idx21c", U8, [1, 1, 1, 21], list(range(21)))
    g.init("idx5rf", F32, [1, 1, 5, 1], [float(i) for i in range(5)])
    g.init("idx5cf", F32, [1, 1, 1, 5], [float(i) for i in range(5)])
    g.init("ones55", U8, [1, 1, 5, 5], [1] * 25)
    g.init("zeros4r", U8, [1, 1, 4, 1], [0] * 4)
    g.init("zeros4c", U8, [1, 1, 1, 4], [0] * 4)
    # stored-weight pen vectors: free -> 16 (eff 0), penalty -> 0 (eff -16)
    g.init("penT9", U8, [1, 1, 9, 1], [16] * 4 + [0] * 5)
    g.init("penL9", U8, [1, 1, 1, 9], [16] * 4 + [0] * 5)
    g.init("asc5r", U8, [1, 1, 5, 1], list(range(1, 6)))
    g.init("asc5c", U8, [1, 1, 1, 5], list(range(1, 6)))
    g.init("pad_dpad", I64, [8], [0, 0, 4, 4, 0, 0, 0, 0])
    g.init("pad_out", I64, [8], [0, 0, 0, 0, 0, 0, 25, 25])
    for n, v in (("zero", 0), ("one", 1), ("two", 2), ("three", 3),
                 ("four", 4), ("ten", 10), ("c16", 16), ("c17", 17),
                 ("c19", 19)):
        g.init(n, U8, [], [v])
    g.init("negone", F32, [], [-1.0])
    g.init("half", F32, [], [0.5])
    g.init("c16f", F32, [], [16.0])
    g.init("xsc", F32, [], [1.0]); g.init("ysc", F32, [], [1.0])
    g.init("wsc", F32, [], [1.0])
    g.init("xzp", U8, [], [0]); g.init("yzp", U8, [], [0])
    g.init("wzp0", U8, [], [0]); g.init("wzp16", U8, [], [16])
    g.init("fv", BOOL, [], [0])

    def sel(cond_b, x, y, out=None):
        return g.nd("Where", [cond_b, x, y], out)

    def u8bit(cond_b, out=None):
        return sel(cond_b, "one", "zero", out)

    # ---------------- 1. decode ---------------- #
    v30 = g.nd("Conv", ["input", "dec_w"], "v30")            # [1,1,30,30] f32
    v30u = g.nd("Cast", [v30], "v30u", to=U8)                # 900
    v19 = g.nd("Slice", [v30u, "s_st", "s_en", "ax23"], "v19")   # 361

    # ---------------- grid dims ---------------- #
    def grid_dim(kern, asc):
        mx = g.nd("MaxPool", [v19], g.nm("gmax"), kernel_shape=kern)
        anyb = g.nd("Greater", [mx, "zero"], g.nm("ganyb"))
        anyu = u8bit(anyb)
        mm = g.nd("Mul", [anyu, asc], g.nm("gmm"))
        return g.nd("MaxPool", [mm], g.nm("gdim"), kernel_shape=[19, 19]
                    if False else ([19, 1] if kern == [1, 19] else [1, 19]))

    H = grid_dim([1, 19], "asc19r")     # [1,1,1,1] u8 == grid height
    W = grid_dim([19, 1], "asc19c")     # == grid width

    # ---------------- 2. colors ---------------- #
    counts = g.nd("ReduceSum", ["input", "ax23"], "counts", keepdims=1)
    cmax = g.nd("ReduceMax", [counts], "cmax", axes=[1], keepdims=1)
    eqbg = g.nd("Equal", [counts, cmax], "eqbg")             # [1,10,1,1] bool
    eqbgf = g.nd("Cast", [eqbg], "eqbgf", to=F32)
    bg = g.nd("Cast", [g.nd("Conv", ["eqbgf", "dec_w"], "bgf")], "bg", to=U8)
    keys = g.nd("Add", [counts, "idx16"], "keys")
    kcur = g.nd("Where", ["eqbg", "negone", keys], "kobj1")
    vals, cnts = [], []
    for r in (1, 2, 3):
        km = g.nd("ReduceMax", [kcur], f"km{r}", axes=[1], keepdims=1)
        n_u = g.nd("Cast", [km], f"n{r}", to=U8)             # floor = count
        nf = g.nd("Cast", [n_u], g.nm("nf"), to=F32)
        fr = g.nd("Sub", [km, nf], g.nm("fr"))
        val = g.nd("Cast", [g.nd("Mul", [fr, "c16f"], g.nm("vf"))],
                   f"val{r}", to=U8)
        vals.append(val); cnts.append(n_u)
        if r < 3:
            eqk = g.nd("Equal", [kcur, km], g.nm("eqk"))
            kcur = g.nd("Where", [eqk, "negone", kcur], f"kobj{r+1}")

    # ---------------- 3. per-candidate ---------------- #
    cand = []
    for i, val in enumerate(vals, 1):
        c = {}
        eqm = g.nd("Equal", [v19, val], f"eqm{i}")           # 361b
        m = u8bit(eqm, f"m{i}")                              # 361
        rh = g.nd("MaxPool", [m], f"rh{i}", kernel_shape=[1, 19])
        ch = g.nd("MaxPool", [m], f"ch{i}", kernel_shape=[19, 1])

        def axis(hasv, desc, asc, idx21, kern1, tag):
            qd = g.nd("MaxPool", [g.nd("Mul", [hasv, desc], g.nm("qdm"))],
                      g.nm("qd"), kernel_shape=kern1)
            qa = g.nd("MaxPool", [g.nd("Mul", [hasv, asc], g.nm("qam"))],
                      g.nm("qa"), kernel_shape=kern1)        # max+1
            mn = g.nd("Sub", ["c19", qd], f"mn{tag}{i}")
            span = g.nd("Sub", [g.nd("Add", [qa, qd], g.nm("sp0")), "c19"],
                        g.nm("span"))
            spf = g.nd("Cast", [span], g.nm("spf"), to=F32)
            sph = g.nd("Cast", [g.nd("Mul", [spf, "half"], g.nm("sph"))],
                       g.nm("spt"), to=U8)
            spb = g.nd("Mul", [sph, "two"], g.nm("spb"))
            even_u = u8bit(g.nd("Equal", [spb, span], g.nm("evb")))
            mn0_u = u8bit(g.nd("Equal", [mn, "zero"], g.nm("mn0b")))
            adj = g.nd("Mul", [g.nd("Sub", ["one", even_u], g.nm("nev")),
                               mn0_u], g.nm("adj"))
            t0 = g.nd("Sub", [g.nd("Add", [mn, "two"], g.nm("mp2")), adj],
                      g.nm("t0"))
            t1v = g.nd("Add", [t0, "one"], g.nm("t1v"))
            e_a = g.nd("Equal", [idx21, t0], g.nm("ea"))
            e_b = g.nd("Equal", [idx21, t1v], g.nm("eb"))
            ker = sel(e_a, "one", sel(e_b, "one", "zero"))   # 21+21 u8
            dim = g.nd("Sub", [qa, mn], f"dim{tag}{i}")      # bbox extent
            return qa, mn, ker, even_u, dim

        qar, rmin, rker, evr, hX = axis(rh, "desc19r", "asc19r",
                                        "idx21r", [19, 1], "r")
        qac, cmin, cker, evc, wX = axis(ch, "desc19c", "asc19c",
                                        "idx21c", [1, 19], "c")

        s1 = g.qlc(m, rker, f"s1_{i}", kernel_shape=[21, 1],
                   strides=[2, 1], pads=[2, 0, 8, 0])        # [1,1,5,19]
        s4 = g.qlc(s1, cker, f"s4_{i}", kernel_shape=[1, 21],
                   strides=[1, 2], pads=[0, 2, 0, 8])        # [1,1,5,5]

        e1 = g.nd("Equal", [s4, "one"], g.nm("e1"))
        e3 = g.nd("Equal", [s4, "three"], g.nm("e3"))
        odd = sel(e1, "one", sel(e3, "one", "zero"))         # 25+25
        anyodd = g.nd("MaxPool", [odd], g.nm("anyodd"), kernel_shape=[5, 5])
        noodd = u8bit(g.nd("Equal", [anyodd, "zero"], g.nm("nob")))
        e4u = u8bit(g.nd("Equal", [s4, "four"], g.nm("e4b")))
        any4 = g.nd("MaxPool", [e4u], g.nm("any4"), kernel_shape=[5, 5])
        blocky = g.nd("Mul", [noodd, any4], f"blocky{i}")    # u8 0/1
        Du = u8bit(g.nd("Greater", [s4, "zero"], g.nm("gt0")), f"Du{i}")

        ro = u8bit(g.nd("Equal", ["idx19r", rmin], g.nm("roe")))
        co = u8bit(g.nd("Equal", ["idx19c", cmin], g.nm("coe")))
        t1 = g.qlc(m, ro, f"tr1_{i}", kernel_shape=[19, 1],
                   pads=[0, 0, 4, 0])                        # [1,1,5,19]
        X0 = g.qlc(t1, co, f"X0_{i}", kernel_shape=[1, 19],
                   pads=[0, 0, 0, 4])                        # [1,1,5,5]

        c.update(val=val, n=cnts[i - 1], blocky=blocky, Du=Du, X0=X0,
                 rmin=rmin, cmin=cmin, qar=qar, qac=qac,
                 evr=evr, evc=evc, hX=hX, wX=wX)
        cand.append(c)

    # ---------------- 4. pick mag; D; allowed sides ---------------- #
    b1, b2 = cand[0]["blocky"], cand[1]["blocky"]
    nb1 = g.nd("Sub", ["one", b1], "nb1")
    p2u = g.nd("Mul", [nb1, b2], "p2u")
    p3u = g.nd("Mul", [nb1, g.nd("Sub", ["one", b2], "nb2")], "p3u")
    p1b = g.nd("Greater", [b1, "zero"], "p1b")
    p2b = g.nd("Greater", [p2u, "zero"], "p2b")
    p3b = g.nd("Greater", [p3u, "zero"], "p3b")

    def magsel(key, out=None):
        return sel(p1b, cand[0][key], sel(p2b, cand[1][key], cand[2][key]),
                   out)

    D = magsel("Du", "D")                                    # [1,1,5,5] u8
    nD = g.qlc(D, "ones55", "nD", kernel_shape=[5, 5])       # [1,1,1,1]
    rminM = magsel("rmin"); cminM = magsel("cmin")
    qarM = magsel("qar"); qacM = magsel("qac")
    aT = u8bit(g.nd("Equal", [rminM, "zero"], g.nm("aTb")))
    aB = u8bit(g.nd("Equal", [qarM, H], g.nm("aBb")))
    aL = u8bit(g.nd("Equal", [cminM, "zero"], g.nm("aLb")))
    aR = u8bit(g.nd("Equal", [qacM, W], g.nm("aRb")))

    # ---------------- 5. side weights + matches ---------------- #
    rhD = g.nd("MaxPool", [D], "rhD", kernel_shape=[1, 5])   # [1,1,5,1]
    fwd = g.nd("MaxPool", [rhD], g.nm("fwd"), kernel_shape=[5, 1],
               pads=[4, 0, 0, 0])
    bwd = g.nd("MaxPool", [rhD], g.nm("bwd"), kernel_shape=[5, 1],
               pads=[0, 0, 4, 0])
    rext = g.nd("Mul", [fwd, bwd], "rext")                   # rows < dh
    chD = g.nd("MaxPool", [D], "chD", kernel_shape=[5, 1])   # [1,1,1,5]
    fwc = g.nd("MaxPool", [chD], g.nm("fwc"), kernel_shape=[1, 5],
               pads=[0, 4, 0, 0])
    bwc = g.nd("MaxPool", [chD], g.nm("bwc"), kernel_shape=[1, 5],
               pads=[0, 0, 0, 4])
    cext = g.nd("Mul", [fwc, bwc], "cext")

    Dpad = g.nd("Pad", [D, "pad_dpad", "zero"], "Dpad")      # [1,1,9,9]
    Deq = g.nd("Equal", [Dpad, "one"], "Deq")                # 81b

    # D extents + mag clip-axis parity (even extent => k even >= 2 => the
    # true sprite is strictly bigger than D on that axis)
    dh = g.nd("MaxPool", [g.nd("Mul", [rhD, "asc5r"], g.nm("dhm"))], "dh",
              kernel_shape=[5, 1])
    dw = g.nd("MaxPool", [g.nd("Mul", [chD, "asc5c"], g.nm("dwm"))], "dw",
              kernel_shape=[1, 5])
    evR = magsel("evr"); evC = magsel("evc")
    midB = g.nd("Mul", [g.nd("Sub", ["one", rext], g.nm("nrx")), "c16"],
                "midB")                                      # [1,1,5,1]
    vecB = g.nd("Concat", ["zeros4r", midB], "vecB", axis=2)  # [1,1,9,1]
    midR = g.nd("Mul", [g.nd("Sub", ["one", cext], g.nm("ncx")), "c16"],
                "midR")
    vecR = g.nd("Concat", ["zeros4c", midR], "vecR", axis=3)  # [1,1,1,9]
    wS = {"T": sel(Deq, "c17", "penT9", "wT"),
          "B": sel(Deq, "c17", vecB, "wB"),
          "L": sel(Deq, "c17", "penL9", "wL"),
          "R": sel(Deq, "c17", vecR, "wR")}                  # [1,1,9,9] u8
    aS = {"T": aT, "B": aB, "L": aL, "R": aR}

    for i, c in enumerate(cand, 1):
        eqh = u8bit(g.nd("Equal", [c["hX"], dh], g.nm("eqhb")))
        eqw = u8bit(g.nd("Equal", [c["wX"], dw], g.nm("eqwb")))
        vTB = g.nd("Sub", ["one", g.nd("Mul", [evR, eqh], g.nm("beh"))],
                   g.nm("vTB"))
        vLR = g.nd("Sub", ["one", g.nd("Mul", [evC, eqw], g.nm("bew"))],
                   g.nm("vLR"))
        vS = {"T": vTB, "B": vTB, "L": vLR, "R": vLR}
        parts = []
        for s in "TBLR":
            sc = g.qlc(c["X0"], wS[s], f"sc{s}_{i}", kernel_shape=[9, 9],
                       pads=[4, 4, 4, 4], wz="wzp16")        # [1,1,5,5]
            mx = g.nd("MaxPool", [sc], g.nm("mx"), kernel_shape=[5, 5])
            mb = u8bit(g.nd("Equal", [mx, nD], g.nm("mbb")))
            gate = g.nd("Mul", [aS[s], vS[s]], g.nm("gate"))
            parts.append(g.nd("Mul", [mb, gate], g.nm("and")))
        c["match"] = g.nd("Add", [g.nd("Add", parts[:2], g.nm("m01")),
                                  g.nd("Add", parts[2:], g.nm("m23"))],
                          f"match{i}")                       # u8 0..4

    # ---------------- 6. winner + compose ---------------- #
    def fsel(key, out=None):     # first non-mag candidate's <key>
        return sel(p1b, cand[1][key], cand[0][key], out)

    def ssel(key, out=None):     # second non-mag candidate's <key>
        return sel(p3b, cand[1][key], cand[2][key], out)

    mf = fsel("match"); ms = ssel("match")
    mfu = u8bit(g.nd("Greater", [mf, "zero"], g.nm("mfb")))
    msu = u8bit(g.nd("Greater", [ms, "zero"], g.nm("msb")))
    nf_f = g.nd("Cast", [fsel("n")], g.nm("nff"), to=F32)
    ns_f = g.nd("Cast", [ssel("n")], g.nm("nsf"), to=F32)
    le = sel(g.nd("Greater", [nf_f, ns_f], g.nm("gtb")), "zero", "one")
    pf = g.nd("Add", [g.nd("Sub", ["one", msu], g.nm("nms")),
                      g.nd("Mul", [mfu, le], g.nm("mfle"))], "pfsum")
    pfb = g.nd("Greater", [pf, "zero"], "pfb")

    wcol = sel(pfb, fsel("val"), ssel("val"), "wcol")
    X0W = sel(pfb, fsel("X0"), ssel("X0"), "X0W")            # [1,1,5,5]
    h = sel(pfb, fsel("hX"), ssel("hX"), "h")
    w = sel(pfb, fsel("wX"), ssel("wX"), "w")

    winb = g.nd("Greater", [X0W, "zero"], "winb")            # 25b
    colored = sel(winb, wcol, bg, "colored")                 # 25
    hf = g.nd("Cast", [h], "hf", to=F32)
    wf = g.nd("Cast", [w], "wf", to=F32)
    rin = u8bit(g.nd("Less", ["idx5rf", hf], g.nm("rinb")))  # [1,1,5,1]
    cin = u8bit(g.nd("Less", ["idx5cf", wf], g.nm("cinb")))  # [1,1,1,5]
    ins = g.nd("Mul", [rin, cin], "ins")                     # 25
    insb = g.nd("Greater", [ins, "zero"], "insb")
    final5 = sel(insb, colored, "ten", "final5")             # [1,1,5,5] u8
    oh5 = g.nd("Equal", [final5, "colidx"], "oh5")           # [1,10,5,5] bool
    g.nodes.append(oh.make_node("Pad", [oh5, "pad_out", "fv"], ["output"],
                                mode="constant"))

    graph = oh.make_graph(
        g.nodes, "e319",
        [oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [oh.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        g.inits)
    m = oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET)
    onnx.checker.check_model(m, full_check=True)
    return m


def _to_onehot(grid):
    gg = np.array(grid, int)
    hh, ww = gg.shape
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
            if max(len(gin), len(gin[0])) > 19:
                return []
            out = (sess.run(["output"], {"input": _to_onehot(gin)})[0]
                   > 0).astype(float)
            exp = _to_onehot(gout)
            if out.shape != exp.shape or not (out == (exp > 0)).all():
                return []
    except Exception:
        return []
    return [("e319", model)]


if __name__ == "__main__":
    m = build()
    onnx.save(m, "_cands2/task319.onnx")
    print("nodes:", len(m.graph.node),
          "params:", sum(int(np.prod(t.dims) or 1) for t in
                         m.graph.initializer))
