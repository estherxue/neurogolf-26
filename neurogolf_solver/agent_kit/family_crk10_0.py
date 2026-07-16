"""family_crk10_0 — hardest-remaining slice U[0::8] = [5,69,101,153,182,238,367].

Solved here:
  * task153 — "jigsaw assembly".  Input has exactly two non-background colours,
    each drawn as a partial shape.  The two shapes are complementary pieces that
    tile a 3x3 square.  Output (always 3x3, top-left) = place the *anchor* piece
    (the one that occupies output cell (0,0)) normalised to the origin, colour it
    with its colour, and colour every other cell of the 3x3 with the other colour.

    Anchor rule (verified EXACT on all 265 train+test+arc-gen pairs): a piece P is
    the anchor iff  normalise(3x3 \\ P_at_origin) == other piece normalised.  When
    both pieces qualify the two placements yield the identical grid, so picking the
    smallest colour index among the valid pieces is always correct.

The ONNX graph is fully channel-parallel (batched MatMul shift matrices realise a
per-channel, data-dependent top-left alignment via the reflection trick
Less(|(j-i)-k|,0.5)).  No Loop/Scan/NonZero.  Static [1,10,30,30] throughout.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT
H = W = 30
C = 10


# ----------------------------------------------------------------------------
# numpy reference (mirror of the ONNX graph's rule) — used only for detection
# ----------------------------------------------------------------------------
def _norm(mask):
    ys, xs = np.where(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _pad3(m):
    r = np.zeros((3, 3), bool)
    r[:m.shape[0], :m.shape[1]] = m
    return r


def _solve153(a):
    a = np.asarray(a)
    cols = [c for c in np.unique(a) if c != 0]
    if len(cols) != 2:
        return None
    NM = {}
    for c in cols:
        m = _norm(a == c)
        if m.shape[0] > 3 or m.shape[1] > 3:
            return None
        NM[c] = _pad3(m)
    R = np.ones((3, 3), bool)
    for c in sorted(cols):          # smallest colour first
        o = cols[0] if cols[1] == c else cols[1]
        comp = R & ~NM[c]
        if np.array_equal(_norm(comp), _norm(NM[o])):
            return np.where(NM[c], c, o)
    return None


# ----------------------------------------------------------------------------
# ONNX construction
# ----------------------------------------------------------------------------
def _consts():
    """Shared initializers."""
    i = np.arange(H, dtype=np.float32)
    Dcol = (i[None, :] - i[:, None]).astype(np.float32)      # [30,30]  j-i
    Drow = (i[:, None] - i[None, :]).astype(np.float32)      # [30,30]  i-j
    R = np.zeros((H, W), np.float32); R[:3, :3] = 1.0
    L = np.zeros((C, C), np.float32)                         # strictly lower tri
    for r in range(C):
        for cc in range(r):
            L[r, cc] = 1.0
    chmask = np.array([0] + [1] * 9, np.float32)
    inits = [
        oh.make_tensor("Dcol", FLOAT, [1, 1, H, W], Dcol.ravel()),
        oh.make_tensor("Drow", FLOAT, [1, 1, H, W], Drow.ravel()),
        oh.make_tensor("R30", FLOAT, [1, 1, H, W], R.ravel()),
        oh.make_tensor("idxRow", FLOAT, [1, 1, H, 1], i.tolist()),
        oh.make_tensor("idxCol", FLOAT, [1, 1, 1, W], i.tolist()),
        oh.make_tensor("chIndex", FLOAT, [1, C, 1, 1], np.arange(C, dtype=np.float32).tolist()),
        oh.make_tensor("chmask", FLOAT, [1, C, 1, 1], chmask.tolist()),
        oh.make_tensor("big", FLOAT, [1, 1, 1, 1], [100.0]),
        oh.make_tensor("half", FLOAT, [1, 1, 1, 1], [0.5]),
        oh.make_tensor("one", FLOAT, [1, 1, 1, 1], [1.0]),
        oh.make_tensor("Ltri", FLOAT, [C, C], L.ravel()),
        oh.make_tensor("shp_col", INT64, [2], [C, 1]),
        oh.make_tensor("shp_ch", INT64, [4], [1, C, 1, 1]),
    ]
    return inits


def _normalize(nodes, x, pfx):
    """Emit nodes that top-left align every channel of x ([1,10,30,30])."""
    def n(op, ins, out, **kw):
        nodes.append(oh.make_node(op, ins, [out], name=out, **kw)); return out
    rowhas = n("ReduceMax", [x], f"{pfx}_rowhas", axes=[3], keepdims=1)
    rmb = n("Greater", [rowhas, "half"], f"{pfx}_rmb")
    rmask = n("Where", [rmb, "idxRow", "big"], f"{pfx}_rmask")
    rmin = n("ReduceMin", [rmask], f"{pfx}_rmin", axes=[2], keepdims=1)
    sd = n("Sub", ["Dcol", rmin], f"{pfx}_sd")
    sda = n("Abs", [sd], f"{pfx}_sda")
    slt = n("Less", [sda, "half"], f"{pfx}_slt")
    Sup = n("Cast", [slt], f"{pfx}_Sup", to=FLOAT)
    up = n("MatMul", [Sup, x], f"{pfx}_up")
    colhas = n("ReduceMax", [x], f"{pfx}_colhas", axes=[2], keepdims=1)
    cmb = n("Greater", [colhas, "half"], f"{pfx}_cmb")
    cmask = n("Where", [cmb, "idxCol", "big"], f"{pfx}_cmask")
    cmin = n("ReduceMin", [cmask], f"{pfx}_cmin", axes=[3], keepdims=1)
    cd = n("Sub", ["Drow", cmin], f"{pfx}_cd")
    cda = n("Abs", [cd], f"{pfx}_cda")
    clt = n("Less", [cda, "half"], f"{pfx}_clt")
    Sl = n("Cast", [clt], f"{pfx}_Sl", to=FLOAT)
    return n("MatMul", [up, Sl], f"{pfx}_norm")


def build_model():
    nodes = []

    def n(op, ins, out, **kw):
        nodes.append(oh.make_node(op, ins, [out], name=out, **kw)); return out

    NM = _normalize(nodes, "input", "nm")                       # [1,10,30,30]
    present = n("ReduceMax", ["input"], "present", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    act = n("Mul", [present, "chmask"], "act")                  # active non-bg [1,10,1,1]

    # TOTAL_map = sum_c act[c]*NM[c]  (= NM[p]+NM[q])
    nm_act = n("Mul", [NM, "act"], "nm_act")
    total = n("ReduceSum", [nm_act], "total", axes=[1], keepdims=1)   # [1,1,30,30]

    # comp[c] = R \ NM[c] ; ncomp = normalise(comp)
    one_m_nm = n("Sub", ["one", NM], "one_m_nm")
    comp = n("Mul", ["R30", one_m_nm], "comp")
    ncomp = _normalize(nodes, comp, "nc")

    other = n("Sub", [total, NM], "othernm")                   # NM of the other piece
    d = n("Sub", [ncomp, other], "d")
    da = n("Abs", [d], "da")
    diff = n("ReduceSum", [da], "diff", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    vlt = n("Less", [diff, "half"], "vlt")
    vraw = n("Cast", [vlt], "vraw", to=FLOAT)
    vact = n("Mul", [vraw, "act"], "vact")                     # valid & active [1,10,1,1]

    # pick smallest-index valid channel
    vr = n("Reshape", [vact, "shp_col"], "vr")                 # [10,1]
    pref = n("MatMul", ["Ltri", vr], "pref")                   # [10,1]
    pref4 = n("Reshape", [pref, "shp_ch"], "pref4")            # [1,10,1,1]
    isfirst = n("Less", [pref4, "half"], "isfirst")
    isf = n("Cast", [isfirst], "isf", to=FLOAT)
    anc = n("Mul", [vact, isf], "anc")                         # one-hot anchor channel

    # colours
    anc_idx = n("Mul", [anc, "chIndex"], "anc_idx")
    colorA = n("ReduceSum", [anc_idx], "colorA", axes=[1], keepdims=1)   # [1,1,1,1]
    act_idx = n("Mul", ["act", "chIndex"], "act_idx")
    sumcol = n("ReduceSum", [act_idx], "sumcol", axes=[1], keepdims=1)
    colorB = n("Sub", [sumcol, colorA], "colorB")

    # anchor mask + complement region
    anc_nm = n("Mul", [anc, NM], "anc_nm")
    maskA = n("ReduceSum", [anc_nm], "maskA", axes=[1], keepdims=1)      # [1,1,30,30]
    one_m_a = n("Sub", ["one", maskA], "one_m_a")
    maskB = n("Mul", ["R30", one_m_a], "maskB")

    # channel equality selectors
    eqA = n("Cast", [n("Less", [n("Abs", [n("Sub", ["chIndex", colorA], "cA_d")], "cA_da"), "half"], "cA_lt")], "eqA", to=FLOAT)
    eqB = n("Cast", [n("Less", [n("Abs", [n("Sub", ["chIndex", colorB], "cB_d")], "cB_da"), "half"], "cB_lt")], "eqB", to=FLOAT)

    outA = n("Mul", [maskA, eqA], "outA")
    outB = n("Mul", [maskB, eqB], "outB")
    nodes.append(oh.make_node("Add", [outA, outB], ["output"], name="output"))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "crk10_0", [x], [y], _consts())
    m = oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


# ============================================================================
# task069 — template-stamp recolour
#   The single connected multi-colour object (colours other than 0 and 8) is a
#   "key".  Every colour-8 object is an exact translate of the key's shape; it is
#   recoloured cell-by-cell to the key's colours; the key itself is erased.
#   ONNX pipeline: align key to origin (global shift) -> crop KxK key colour
#   kernels + key-shape kernel -> Conv-correlate the 8-mask with the key shape to
#   locate object top-lefts (corr == |shape|) -> ConvTranspose the colour kernels
#   at those top-lefts -> re-add background (original bg + erased key cells).
#   Verified EXACT (numpy mirror of this graph) on all 264 train+test+arc-gen.
# ============================================================================
_K69 = 4


def _oh69(a):
    x = np.zeros((10, H, W), np.float32)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            x[a[r, c], r, c] = 1.0
    return x


def _shup(x, k):
    o = np.zeros_like(x)
    if k < H:
        o[:, :H - k, :] = x[:, k:, :]
    return o


def _shleft(x, k):
    o = np.zeros_like(x)
    if k < W:
        o[:, :, :W - k] = x[:, :, k:]
    return o


def _corr69(M, Wk):
    K = _K69
    Mp = np.zeros((H + K - 1, W + K - 1), np.float32)
    Mp[:H, :W] = M
    out = np.zeros((H, W), np.float32)
    for i in range(K):
        for j in range(K):
            if Wk[i, j]:
                out += Mp[i:i + H, j:j + W] * Wk[i, j]
    return out


def _convtr69(T, Wk):
    K = _K69
    big = np.zeros((H + K - 1, W + K - 1), np.float32)
    for p, q in np.argwhere(T > 0.5):
        big[p:p + K, q:q + K] += Wk
    return big[:H, :W]


def _solve69(a):
    """numpy mirror of the ONNX graph (used for detection)."""
    a = np.asarray(a)
    x = _oh69(a)
    chk = np.array([1 if (c != 0 and c != 8) else 0 for c in range(10)], np.float32)
    KM = (x * chk[:, None, None]).sum(0)
    if KM.sum() == 0 or x[8].sum() == 0:
        return None
    ys, xs = np.where(KM > 0.5)
    rmin, cmin = int(ys.min()), int(xs.min())
    al = _shleft(_shup(x, rmin), cmin)
    Kall = al[:, :_K69, :_K69] * chk[:, None, None]
    S = _shleft(_shup(KM[None], rmin), cmin)[0, :_K69, :_K69]
    Sn = S.sum()
    match = (_corr69(x[8], S) > Sn - 0.5).astype(np.float32)
    stamped = np.stack([_convtr69(match, Kall[c]) for c in range(10)])
    chnobg = np.array([0] + [1] * 9, np.float32)
    out = stamped * chnobg[:, None, None]
    out[0] = x[0] + KM
    # decode to grid
    g = np.zeros(a.shape, int)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            nz = np.where(out[:, r, c] > 0.5)[0]
            if len(nz) != 1:
                return None
            g[r, c] = nz[0]
    return g


def build_model_69():
    K = _K69
    nodes = []

    def n(op, ins, out, **kw):
        nodes.append(oh.make_node(op, ins, [out], name=out, **kw)); return out

    i = np.arange(H, dtype=np.float32)
    Dcol = (i[None, :] - i[:, None]).astype(np.float32)
    Drow = (i[:, None] - i[None, :]).astype(np.float32)
    chk = [0.0] + [1.0] * 7 + [0.0, 1.0]                 # zero at ch0 and ch8
    inits = [
        oh.make_tensor("Dcol", FLOAT, [1, 1, H, W], Dcol.ravel()),
        oh.make_tensor("Drow", FLOAT, [1, 1, H, W], Drow.ravel()),
        oh.make_tensor("idxRow", FLOAT, [1, 1, H, 1], i.tolist()),
        oh.make_tensor("idxCol", FLOAT, [1, 1, 1, W], i.tolist()),
        oh.make_tensor("big", FLOAT, [1, 1, 1, 1], [100.0]),
        oh.make_tensor("half", FLOAT, [1, 1, 1, 1], [0.5]),
        oh.make_tensor("chk", FLOAT, [1, 10, 1, 1], chk),
        oh.make_tensor("ch0", FLOAT, [1, 10, 1, 1], [1.0] + [0.0] * 9),
        oh.make_tensor("s_ch8_s", INT64, [1], [8]),
        oh.make_tensor("s_ch8_e", INT64, [1], [9]),
        oh.make_tensor("s_ch_ax", INT64, [1], [1]),
        oh.make_tensor("s_ch0_s", INT64, [1], [0]),
        oh.make_tensor("s_ch0_e", INT64, [1], [1]),
        oh.make_tensor("s_kk_s", INT64, [2], [0, 0]),
        oh.make_tensor("s_kk_e", INT64, [2], [K, K]),
        oh.make_tensor("s_hw_ax", INT64, [2], [2, 3]),
        oh.make_tensor("s_30_e", INT64, [2], [H, W]),
    ]

    # key mask KM and its bbox origin
    xk = n("Mul", ["input", "chk"], "xk")
    KM = n("ReduceSum", [xk], "KM", axes=[1], keepdims=1)          # [1,1,30,30]
    rowhas = n("ReduceMax", [KM], "rowhas", axes=[3], keepdims=1)
    rmb = n("Greater", [rowhas, "half"], "rmb")
    rmask = n("Where", [rmb, "idxRow", "big"], "rmask")
    rmin = n("ReduceMin", [rmask], "rmin", axes=[2], keepdims=1)   # [1,1,1,1]
    colhas = n("ReduceMax", [KM], "colhas", axes=[2], keepdims=1)
    cmb = n("Greater", [colhas, "half"], "cmb")
    cmask = n("Where", [cmb, "idxCol", "big"], "cmask")
    cmin = n("ReduceMin", [cmask], "cmin", axes=[3], keepdims=1)

    Sup = n("Cast", [n("Less", [n("Abs", [n("Sub", ["Dcol", rmin], "sd")], "sda"), "half"], "slt")], "Sup", to=FLOAT)
    Sl = n("Cast", [n("Less", [n("Abs", [n("Sub", ["Drow", cmin], "cd")], "cda"), "half"], "clt")], "Sl", to=FLOAT)

    # align all channels to key origin, crop KxK -> colour kernels + shape kernel
    up = n("MatMul", [Sup, "input"], "up")                         # [1,10,30,30]
    al = n("MatMul", [up, Sl], "al")
    alc = n("Slice", [al, "s_kk_s", "s_kk_e", "s_hw_ax"], "alc")   # [1,10,K,K]
    Kall = n("Mul", [alc, "chk"], "Kall")                          # ConvTranspose W [1,10,K,K]

    kmup = n("MatMul", [Sup, KM], "kmup")
    kmal = n("MatMul", [kmup, Sl], "kmal")
    Sker = n("Slice", [kmal, "s_kk_s", "s_kk_e", "s_hw_ax"], "Sker")   # [1,1,K,K]
    Sn = n("ReduceSum", [Sker], "Sn", axes=[2, 3], keepdims=1)     # [1,1,1,1]

    # correlate 8-mask with key shape -> match top-lefts
    M8 = n("Slice", ["input", "s_ch8_s", "s_ch8_e", "s_ch_ax"], "M8")   # [1,1,30,30]
    corr = n("Conv", [M8, Sker], "corr", kernel_shape=[K, K], strides=[1, 1],
             pads=[0, 0, K - 1, K - 1])
    thr = n("Sub", [Sn, "half"], "thr")
    match = n("Cast", [n("Greater", [corr, thr], "mgt")], "match", to=FLOAT)  # [1,1,30,30]

    # stamp colour kernels at matches
    st = n("ConvTranspose", [match, Kall], "st", kernel_shape=[K, K], strides=[1, 1])  # [1,10,29+K,..]
    stamped = n("Slice", [st, "s_kk_s", "s_30_e", "s_hw_ax"], "stamped")   # crop -> [1,10,30,30]
    stamped_m = n("Mul", [stamped, "chk"], "stamped_m")

    X0 = n("Slice", ["input", "s_ch0_s", "s_ch0_e", "s_ch_ax"], "X0")
    bg = n("Add", [X0, KM], "bg")
    bg_ch0 = n("Mul", [bg, "ch0"], "bg_ch0")
    nodes.append(oh.make_node("Add", [stamped_m, bg_ch0], ["output"], name="output"))

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "crk10_69", [x], [y], inits)
    m = oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


# ----------------------------------------------------------------------------
def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    # task153 — jigsaw
    if all((lambda s: s is not None and s.shape == b.shape and np.array_equal(s, b))(_solve153(a))
           for a, b in prs):
        out.append(("jigsaw153", build_model()))
    # task069 — template stamp
    if all((lambda s: s is not None and s.shape == b.shape and np.array_equal(s, b))(_solve69(a))
           for a, b in prs):
        out.append(("stamp69", build_model_69()))
    return out
