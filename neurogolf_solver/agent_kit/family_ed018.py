"""ENRICHED-ARSENAL deep rebuild of task18 (arc-gen hash 0e206a2e): "complete the
rotated clone".

Semantics (from the generator, mirrored by family_rb018 which this reproduces bit
for bit): the grid holds 1-2 SPRITES.  Each sprite is a small 4-connected creature
whose body cells share the global MODE colour and whose 3 remaining cells are 3
DISTINCT marker colours.  Every sprite is drawn twice — an ORIGINAL (body + markers)
and a CLONE that shows ONLY its 3 markers, rotated by one of 4 fixed D4 maps.  The
OUTPUT keeps ONLY the clones, drawn FULLY (markers in place + body filled), rotated;
the originals are dropped.

Because the clone body is always the MODE colour and the clone markers are already
present in the input, the output is exactly:  clone-markers (copied) + mode painted
on the union of every valid rotated-body footprint.  No per-cell priority is needed
(all bodies are one colour), which collapses the whole detect+paint core.

ONNX core (opset-17, static, all uint8/int8/bool — no banned ops):
  * value image via 1x1 Conv, cropped to 24x24 (grids are <=24).
  * mode via per-channel ReduceSum + ArgMax.
  * ONE flood (MaxPool(7)-radius-3 x3, clipped with Mul) isolates original #0; the
    r=3 dilation of the mode mask separates clone markers (>=4 from any mode cell)
    from original markers (<=3), so clones/original#1 come from set algebra with no
    2nd flood.
  * per-original 7x7 value kernel is a plain Gather window (originals/clones are >=4
    apart, so a radius-3 window auto-isolates one creature) centred on the bbox
    centre (an on-grid pivot => match centres stay on-grid).
  * DETECT (arsenal trick #1): one RUNTIME-weight QLinearConv over [clone^2, clone]
    with weights [-1_marker , +2*marker_value] and bias (1 - sum marker^2) makes an
    exact colour match saturate to a BINARY [1,8,24,24] uint8 map — the sum-of-
    squares distance D>=0 hits 1-D==1 only at D==0.  Invalid branches get bias -1000.
  * PAINT (arsenal trick #2): one group-1 QLinearConv is the Conv adjoint of the 8
    body-footprint ConvTransposes, summing every rotated body into one uint8 cover.
  * output = Where(clone-marker, clone-value, mode*cover), masked to the real grid
    (beyond-grid padded with 10) then one-hot via Equal.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh, TensorProto as TP

IR_VERSION = 8
OPSET = [oh.make_opsetid("", 17)]
F32, F16, U8, I8, I32, I64, BOOL = (TP.FLOAT, TP.FLOAT16, TP.UINT8, TP.INT8,
                                    TP.INT32, TP.INT64, TP.BOOL)
S = 24          # working grid (arc-gen grids are 12..24)
KS = 7
CTR = 3

# --------------------------------------------------------------------------- #
# numpy reference (mirrors the ONNX numerics; == family_rb018 on all inputs)   #
# --------------------------------------------------------------------------- #
_REV = np.zeros((KS, KS), np.float32)
for _i in range(KS):
    _REV[_i, KS - 1 - _i] = 1.0


FA_ITERS = 4     # floodAll: mode -> all originals (markers <=3 from mode)
C0_ITERS = 8     # single-seed: fill one creature (radius-1 8-geodesic <=6)


def _dil(m):
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for di in range(3):
        for dj in range(3):
            out = np.maximum(out, p[di:di + m.shape[0], dj:dj + m.shape[1]])
    return out


def _flood(seed, nonbg, iters):
    T = seed.copy()
    for _ in range(iters):
        T = np.minimum(_dil(T), nonbg)      # radius-1: a 1-cell gap always blocks it
    return (T > 0.5).astype(np.float32)


def _first(m):
    ys, xs = np.where(m > 0.5)
    o = np.zeros_like(m)
    if len(ys):
        o[ys[0], xs[0]] = 1.0
    return o


def _corr(X, K):
    H, W = X.shape
    Xp = np.pad(X, CTR)
    out = np.zeros((H, W), np.float32)
    ys, xs = np.where(K != 0)
    for ki, kj in zip(ys, xs):
        out += K[ki, kj] * Xp[ki:ki + H, kj:kj + W]
    return out


def _stamp(valid, K):
    H, W = valid.shape
    acc = np.zeros((H + 2 * CTR, W + 2 * CTR), np.float32)
    ys, xs = np.where(K != 0)
    for ki, kj in zip(ys, xs):
        acc[ki:ki + H, kj:kj + W] += K[ki, kj] * valid
    return acc[CTR:CTR + H, CTR:CTR + W]


def _ref(a):
    a = np.asarray(a, int)
    if a.ndim != 2 or a.size == 0 or max(a.shape) > S:
        return None
    H, W = a.shape
    Vg = a.astype(np.float32)
    nonbg = (Vg > 0.5).astype(np.float32)
    if nonbg.sum() < 0.5:
        return np.zeros((H, W), int)
    counts = np.array([(nonbg * (np.abs(Vg - c) < 0.5)).sum() for c in range(10)])
    counts[0] = -1
    mode = int(np.argmax(counts))
    modeMask = (np.abs(Vg - mode) < 0.5).astype(np.float32) * nonbg
    floodAll = _flood(modeMask, nonbg, FA_ITERS)          # union of originals
    comp0 = _flood(_first(modeMask), nonbg, C0_ITERS)     # one creature
    creature1 = floodAll * (1.0 - comp0)                  # the other creature
    cloneMk = nonbg * (1.0 - floodAll)                    # clone markers
    cloneVal = Vg * cloneMk
    cloneSq = cloneVal * cloneVal
    bodyCount = np.zeros((H, W), np.float32)
    for comp in (comp0, creature1):
        ys, xs = np.where(comp > 0.5)
        if len(ys) == 0:
            continue
        cr = (int(ys.min()) + int(ys.max())) // 2
        cc = (int(xs.min()) + int(xs.max())) // 2
        Vp = np.pad(Vg * comp, CTR)                       # MASK to this creature
        Kc = Vp[cr:cr + KS, cc:cc + KS]
        nmark = (comp * (1.0 - modeMask)).sum()
        gate = abs(nmark - 3) < 0.5
        Krots = {1: _REV @ Kc.T, 2: Kc.T @ _REV, 3: _REV @ Kc, 4: Kc.T}
        for rr in (1, 2, 3, 4):
            KR = Krots[rr]
            ismode_k = np.abs(KR - mode) < 0.5
            markerK = np.where(ismode_k, 0.0, KR) * (KR > 0.5)
            binm = (markerK > 0.5).astype(np.float32)
            valm = markerK
            Cc = (valm * valm).sum()
            oneMinusD = 1.0 - _corr(cloneSq, binm) + 2.0 * _corr(cloneVal, valm) - Cc
            valid = (oneMinusD > 0.5).astype(np.float32)
            if not gate:
                valid = valid * 0.0
            bodyCount = bodyCount + _stamp(valid, ismode_k.astype(np.float32))
    bodyCov = (bodyCount > 0.5).astype(np.float32)
    out = np.where(cloneMk > 0.5, cloneVal, mode * bodyCov)
    return out.astype(int)


# --------------------------------------------------------------------------- #
# ONNX graph builder                                                           #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes, self.inits, self._k = [], [], 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}_{self._k}"

    def init(self, name, dt, dims, vals):
        self.inits.append(oh.make_tensor(name, dt, list(dims), list(np.asarray(vals).ravel())))
        return name

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm(op.lower())
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _flip(g, x, axes):
    # reverse the given spatial axes of a [.,.,KS,KS] (or [.,.,S,S]) tensor
    n = len(axes)
    starts = g.init(g.nm("fs"), I64, [n], [-1] * n)
    ends = g.init(g.nm("fe"), I64, [n], [-(KS + 1)] * n)
    ax = g.init(g.nm("fa"), I64, [n], axes)
    st = g.init(g.nm("ft"), I64, [n], [-1] * n)
    return g.nd("Slice", [x, starts, ends, ax, st])


def _bbox_center(g, comp):
    """Return (ridx7, cidx7) int64 gather indices for the radius-3 window centred on
    the bbox centre of `comp` [1,1,S,S] uint8, addressing a comp-padded (pad 3) grid."""
    rowhas = g.nd("ReduceMax", [comp], axes=[3], keepdims=1)          # [1,1,S,1]
    colhas = g.nd("ReduceMax", [comp], axes=[2], keepdims=1)          # [1,1,1,S]
    rmin = g.nd("ArgMax", [rowhas], axis=2, keepdims=1, select_last_index=0)  # [1,1,1,1]
    rmax = g.nd("ArgMax", [rowhas], axis=2, keepdims=1, select_last_index=1)
    cmin = g.nd("ArgMax", [colhas], axis=3, keepdims=1, select_last_index=0)
    cmax = g.nd("ArgMax", [colhas], axis=3, keepdims=1, select_last_index=1)

    def centre(mn, mx):
        s = g.nd("Add", [mn, mx])                                    # i64 [1,1,1,1]
        sf = g.nd("Cast", [s], to=F32)
        cf = g.nd("Floor", [g.nd("Mul", [sf, "half"])])
        ci = g.nd("Cast", [cf], to=I64)
        c1 = g.nd("Reshape", [ci, "sh1"])                            # [1]
        return g.nd("Add", ["range7", c1])                          # [7]
    return centre(rmin, rmax), centre(cmin, cmax)


def _kernel(g, vg, comp):
    """Extract the centred masked 7x7 value kernel of one original, then its 4 D4
    rotations stacked into [4,1,KS,KS] uint8.  Masking (Vg*comp) isolates one
    creature even when a neighbour is only 1 empty cell away."""
    VgCpad = g.nd("Pad", [g.nd("Mul", [vg, comp]), "padK", "zero_u8"], mode="constant")  # [1,1,30,30]
    ridx, cidx = _bbox_center(g, comp)
    prow = g.nd("Gather", [VgCpad, ridx], axis=2)                    # [1,1,KS,S+6]
    Kc = g.nd("Gather", [prow, cidx], axis=3)                        # [1,1,KS,KS]
    KcT = g.nd("Transpose", [Kc], perm=[0, 1, 3, 2])
    rot1 = _flip(g, KcT, [2])       # flipud(KcT)  = REV @ Kc^T
    rot2 = _flip(g, KcT, [3])       # fliplr(KcT)  = Kc^T @ REV
    rot3 = _flip(g, Kc, [2])        # flipud(Kc)   = REV @ Kc
    rot4 = KcT                      # Kc^T
    return g.nd("Concat", [rot1, rot2, rot3, rot4], axis=0)          # [4,1,KS,KS]


def build():
    g = _G()
    # ---- constants -------------------------------------------------------- #
    g.init("valw", F32, [1, 10, 1, 1], list(range(10)))
    g.init("colidx", U8, [1, 10, 1, 1], list(range(10)))
    g.init("ch0pen", F32, [1, 10, 1, 1], [1e9] + [0.0] * 9)
    g.init("half", F32, [], [0.5])
    g.init("range7", I64, [7], [0, 1, 2, 3, 4, 5, 6])
    g.init("sh1", I64, [1], [1])
    g.init("rowidx", U8, [1, 1, S, 1], list(range(S)))
    g.init("colidx24", U8, [1, 1, 1, S], list(range(S)))
    g.init("zero_u8", U8, [], [0])
    g.init("one_u8", U8, [], [1])
    g.init("two_u8", U8, [], [2])
    g.init("ten_u8", U8, [], [10])
    g.init("zero_i8", I8, [], [0])
    g.init("negone_i8", I8, [], [-1])
    g.init("one_i32", I32, [], [1])
    g.init("zero_i32", I32, [], [0])
    g.init("negbig_i32", I32, [], [-1000])
    g.init("three_u8", U8, [], [3])
    # QLinearConv scalars
    g.init("xs", F32, [], [1.0]); g.init("xz", U8, [], [0])
    g.init("ws", F32, [], [1.0]); g.init("wz", I8, [], [0])
    g.init("ys", F32, [], [1.0]); g.init("yz", U8, [], [0])
    # slice specs
    g.init("s0", I64, [2], [0, 0]); g.init("s24", I64, [2], [S, S])
    g.init("ax23", I64, [2], [2, 3])
    g.init("padK", I64, [8], [0, 0, CTR, CTR, 0, 0, CTR, CTR])
    g.init("pad30", I64, [8], [0, 0, 0, 0, 0, 0, 30 - S, 30 - S])
    g.init("axr23", I64, [2], [2, 3])
    g.init("axr123", I64, [3], [1, 2, 3])

    # ---- value image + crop to 24 ---------------------------------------- #
    vg30f = g.nd("Conv", ["input", "valw"], kernel_shape=[1, 1])     # [1,1,30,30] f32
    vg30u = g.nd("Cast", [vg30f], to=U8)
    vg = g.nd("Slice", [vg30u, "s0", "s24", "ax23"])                 # [1,1,24,24] u8

    # ---- real (within-grid) mask ----------------------------------------- #
    rcu = g.nd("Cast", [g.nd("ReduceMax", ["input"], axes=[1, 3], keepdims=1)], to=U8)  # [1,1,30,1]
    ccu = g.nd("Cast", [g.nd("ReduceMax", ["input"], axes=[1, 2], keepdims=1)], to=U8)  # [1,1,1,30]
    g.init("z1", I64, [1], [0]); g.init("s24_1", I64, [1], [S])
    rc24 = g.nd("Slice", [rcu, "z1", "s24_1", g.init("ax2", I64, [1], [2])])
    cc24 = g.nd("Slice", [ccu, "z1", "s24_1", g.init("ax3", I64, [1], [3])])
    real = g.nd("Mul", [rc24, cc24])                                 # [1,1,24,24] u8

    # ---- nonbg / mode / modeMask ----------------------------------------- #
    nonbg = g.nd("Cast", [g.nd("Greater", [vg, "zero_u8"])], to=U8)  # [1,1,24,24] u8
    counts = g.nd("ReduceSum", ["input", "axr23"], keepdims=1)       # [1,10,1,1]
    counts_m = g.nd("Sub", [counts, "ch0pen"])
    modeArg = g.nd("ArgMax", [counts_m], axis=1, keepdims=1)         # [1,1,1,1] i64
    modeu = g.nd("Cast", [modeArg], to=U8)                           # [1,1,1,1]
    modeMask = g.nd("Cast", [g.nd("Equal", [vg, modeu])], to=U8)     # [1,1,24,24]

    # ---- flood original #0 (seed = first mode cell) ---------------------- #
    rowhas = g.nd("ReduceMax", [modeMask], axes=[3], keepdims=1)     # [1,1,24,1]
    rmin = g.nd("Cast", [g.nd("ArgMax", [rowhas], axis=2, keepdims=1)], to=U8)  # [1,1,1,1]
    rowsel = g.nd("Cast", [g.nd("Equal", ["rowidx", rmin])], to=U8)  # [1,1,24,1]
    maskrow = g.nd("Mul", [modeMask, rowsel])                        # [1,1,24,24]
    colhas = g.nd("ReduceMax", [maskrow], axes=[2], keepdims=1)      # [1,1,1,24]
    cmin = g.nd("Cast", [g.nd("ArgMax", [colhas], axis=3, keepdims=1)], to=U8)
    colsel = g.nd("Cast", [g.nd("Equal", ["colidx24", cmin])], to=U8)  # [1,1,1,24]
    seed0 = g.nd("Mul", [rowsel, colsel])                            # [1,1,24,24] one cell

    def flood(seed, iters):                                          # radius-1 (gap-safe)
        T = seed
        for _ in range(iters):
            mp = g.nd("MaxPool", [T], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
            T = g.nd("Mul", [mp, nonbg])
        return T
    floodAll = flood(modeMask, 4)                                    # union of originals (markers <=3 from mode)
    comp0 = flood(seed0, 7)                                          # one creature (radius-1 geodesic <=6)

    # ---- clone markers / other original via set algebra ------------------ #
    creature1 = g.nd("Sub", [floodAll, comp0])                       # the other creature
    cloneMk = g.nd("Sub", [nonbg, floodAll])                         # clone markers
    cloneVal = g.nd("Mul", [vg, cloneMk])
    cloneSq = g.nd("Mul", [cloneVal, cloneVal])
    clone2ch = g.nd("Concat", [cloneSq, cloneVal], axis=1)           # [1,2,24,24]

    # ---- per-original kernels -> KR8 [8,1,KS,KS] ------------------------- #
    kr0 = _kernel(g, vg, comp0)
    kr1 = _kernel(g, vg, creature1)
    KR8 = g.nd("Concat", [kr0, kr1], axis=0)                         # [8,1,KS,KS] u8

    # ---- derive match weights + paint footprints ------------------------- #
    ismode = g.nd("Equal", [KR8, modeu])                            # bool [8,1,KS,KS]
    markerV = g.nd("Where", [ismode, "zero_u8", KR8])               # markers only, u8
    ch1 = g.nd("Cast", [g.nd("Mul", [markerV, "two_u8"])], to=I8)    # 2*marker value, i8
    gtm = g.nd("Greater", [markerV, "zero_u8"])                     # bool
    ch0 = g.nd("Neg", [g.nd("Cast", [gtm], to=I8)])                 # -1 at marker, i8
    weight8 = g.nd("Concat", [ch0, ch1], axis=1)                     # [8,2,KS,KS] i8
    sq8 = g.nd("Mul", [markerV, markerV])                           # u8 (<=81)
    Cconst = g.nd("ReduceSum", [g.nd("Cast", [sq8], to=I32), "axr123"], keepdims=0)  # [8]
    bias_valid = g.nd("Sub", ["one_i32", Cconst])                   # [8] i32
    # gate: real creatures have exactly 3 markers so Cconst>=3; an empty flood -> 0.
    gate8 = g.nd("Greater", [Cconst, "zero_i32"])
    bias8 = g.nd("Where", [gate8, bias_valid, "negbig_i32"])         # [8] i32

    # paint weight = flip(body-footprint), relaid [8,1,KS,KS] -> [1,8,KS,KS]
    bodyflip = _flip(g, g.nd("Cast", [ismode], to=I8), [2, 3])       # [8,1,KS,KS] i8
    paint8 = g.nd("Reshape", [bodyflip, g.init("sh18", I64, [4], [1, 8, KS, KS])])

    # ---- DETECT: binary [1,8,24,24] uint8 -------------------------------- #
    valid8 = g.nd("QLinearConv",
                  [clone2ch, "xs", "xz", weight8, "ws", "wz", "ys", "yz", bias8],
                  group=1, kernel_shape=[KS, KS], pads=[CTR, CTR, CTR, CTR], strides=[1, 1])

    # ---- PAINT: sum the 8 body footprints -> cover ----------------------- #
    cover = g.nd("QLinearConv",
                 [valid8, "xs", "xz", paint8, "ws", "wz", "ys", "yz"],
                 group=1, kernel_shape=[KS, KS], pads=[CTR, CTR, CTR, CTR], strides=[1, 1])

    # ---- compose output --------------------------------------------------- #
    body = g.nd("Where", [g.nd("Greater", [cover, "zero_u8"]), modeu, "zero_u8"])
    outv = g.nd("Where", [g.nd("Greater", [cloneMk, "zero_u8"]), cloneVal, body])  # markers win
    outv = g.nd("Where", [g.nd("Greater", [real, "zero_u8"]), outv, "ten_u8"])     # beyond-grid -> 10
    out30 = g.nd("Pad", [outv, "pad30", "ten_u8"], mode="constant")               # [1,1,30,30]
    g.nd("Equal", [out30, "colidx"], "output")                                    # [1,10,30,30] bool

    x = oh.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = oh.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    graph = oh.make_graph(g.nodes, "ed018", [x], [y], inits)
    m = oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET)
    onnx.checker.check_model(m, full_check=True)
    return m


# --------------------------------------------------------------------------- #
# detection / candidates                                                       #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return []
            if max(a.shape) > S or max(b.shape) > S:
                return []
            out.append((a, b))
    return out


def _matches(prs):
    if not prs:
        return False
    for a, b in prs:
        try:
            o = _ref(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not _matches(prs):
        return []
    try:
        return [("ed018", build())]
    except Exception:
        return []
