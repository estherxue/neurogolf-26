"""family_ed209b — task209 / 8a004b2b : "magnify sprite into box" (uint8 golf v2).

Byte-identical transform to family_ed209 (= the out_blend13 incumbent ONNX; see
family_rb209 for the full spec + numpy reference `_ref`), recompiled so every
scored intermediate is uint8/bool (opset-14 u8 Add/Sub/Mul) except the single
fp32 Conv output.  87297 B / 394 params  ->  ~12.1 kB / ~1.6 k params.

Levers vs ed209 (all value-exact on generator-produced inputs — verified
byte-identical to the incumbent ONNX on the official 266 + 3000 fresh gen +
opt-invariance; the generator guarantees exploited beyond ed209's are: nothing
above/beside the box, sprite strictly below it, shown strictly interior,
colours {1,2,3,8}, t>=2):
  1. the 1x1 value Conv becomes an 11x11 Conv whose only nonzero kernel cell
     is (0,0): its output IS the 20x20 parse crop (params for memory).  Yellow
     weighs 10, so yellow rows/cols are exactly the rows/cols with max == 10,
     and colour rows are those with 0 < max < 10 — no [20,20] yellow/colour/
     inbox/Vsh/Vspr masks at all, only V u8 and [20,1]/[1,20] vectors.
  2. Gather-crops with clamped u8 indices replace every dyncrop MatMul: sprite
     rows (below mr, pure sprite) -> [1,1,3,20] strip -> Psmall [1,1,3,5]; the
     shown band (12 rows from aR, rows >= mr masked) doubles as the source of
     aC and of the [1,1,12,12] shown window.
  3. detect = block counting: Qcnt[m] = 4-group m-strided QLinearConv of the
     query one-hot (= per-colour cell counts of every m x m block); using it as
     the (runtime, uint8) weight of a QLinearConv over the [1,4,3,5] template
     one-hot with pads=k-1 gives all 15 offset match-counts per mag in one
     [1,1,3,5] map — no resized/padded templates, no Gather of corr maps.
     match <=> count == #shown (#shown = two tiny all-ones convs over Qcnt[4]).
  4. hierarchical arg-min in u8 over the [3,1,3,5] (mag,rt,cl) lattice: min
     mag, then min Or+16, then min Oc+16, carrying valid*100 penalties; fits
     tests are exact integer u8 compares (wraps only occur on already-invalid
     candidates).  Identical to the incumbent's fp32 key argmin because
     key = mag*10000+Or*100+Oc is lexicographic in (mag,Or,Oc).
  5. winner compose: mask Psmall by (mag==sel), Resize per mag, cascade-pad
     the running sum ([6,10]->[9,15]->[12,20]); placement = two Gathers from
     the zero-guard-padded [1,1,13,21] canvas with (d+16)-shifted u8 indices —
     junk picks hit the guard row/col, so no data masks.
  6. corners/in-grid compose in u8 with sentinel 255 (OUTv is provably 0
     outside the box); Pad(255) to [1,1,30,30] and the final one-hot Equal
     writes the FREE graph output directly.

opset 14 (u8 Add/Sub/Mul; all other ops at their opset-13 semantics), IR 10,
static [1,10,30,30] f32 in / bool out, no banned ops, op set is a strict
subset of the incumbent's.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import GRID_SHAPE, IR_VERSION

F32 = onnx.TensorProto.FLOAT
F16 = onnx.TensorProto.FLOAT16
U8 = onnx.TensorProto.UINT8
I8 = onnx.TensorProto.INT8
I32 = onnx.TensorProto.INT32
I64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
H30 = 30
S = 20       # parse canvas (input W,H <= 20)
OHR = 16     # output canvas rows (box height tall <= 16)
WIN = 12     # search window (magnified creature <= 12x12)
OPSET = [oh.make_opsetid("", 14)]

_NP = {U8: np.uint8, I8: np.int8, F16: np.float16, F32: np.float32,
       I64: np.int64, I32: np.int32}


# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def c(self, dt, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, dt, list(dims), np.asarray(vals, _NP[dt]).tobytes(), raw=True))
        return n

    def u8(self, dims, vals):
        return self.c(U8, dims, vals)

    def u8s(self, v):
        return self.c(U8, [], [v])

    def h(self, dims, vals):
        return self.c(F16, dims, vals)

    def hs(self, v):
        return self.c(F16, [], [v])

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, I64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out

    def cast(self, x, dt):
        return self.nd("Cast", [x], to=dt)


# ---- u8 span helpers ---------------------------------------------------------
def _minidx(g, has, rev):
    """min index of a 0/1 u8 presence vector `has` (255-trick)."""
    return g.nd("Sub", [g.c255, g.nd("ReduceMax", [g.nd("Mul", [has, rev])], keepdims=0)])


def _maxmin(g, has, idx, rev):
    """(min,max) index of a 0/1 u8 presence vector `has`."""
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], keepdims=0)
    return _minidx(g, has, rev), mx


def _gidx(g, kconst, base, gate=None, limit=None):
    """Clamped gather indices (base+k), masked to 0 where k>=gate or idx>=limit.

    Returns (i32 index vector, u8 mask vector or None)."""
    t = g.nd("Add", [kconst, base])
    m = None
    if gate is not None:
        m = g.cast(g.nd("Less", [kconst, gate]), U8)
        t = g.nd("Mul", [t, m])
    if limit is not None:
        lm = g.cast(g.nd("Less", [t, limit]), U8)
        t = g.nd("Mul", [t, lm])
        m = lm if m is None else m
    return g.cast(t, I32), m


def _consts(g):
    g.c0 = g.u8s(0)
    g.c1 = g.u8s(1)
    g.c4u = g.u8s(4)
    g.c20s = g.u8s(20)
    g.c255 = g.u8s(255)
    g.r20u = g.u8([1, 1, S, 1], list(range(S)))            # row idx
    g.c20u = g.u8([1, 1, 1, S], list(range(S)))            # col idx
    g.revr = g.u8([1, 1, S, 1], [255 - r for r in range(S)])
    g.revc = g.u8([1, 1, 1, S], [255 - r for r in range(S)])
    g.r20p1 = g.u8([1, 1, S, 1], [r + 1 for r in range(S)])
    g.c20p1 = g.u8([1, 1, 1, S], [r + 1 for r in range(S)])
    g.or16u = g.u8([1, 1, OHR, 1], list(range(OHR)))
    g.k3 = g.u8([3], [0, 1, 2])
    g.k5 = g.u8([5], list(range(5)))
    g.k12 = g.u8([WIN], list(range(WIN)))
    g.colv = g.u8([1, 4, 1, 1], [1, 2, 3, 8])
    g.valvec = g.u8([1, 10, 1, 1], list(range(10)))
    g.half = g.hs(0.5)
    g.onef = g.hs(1.0)
    g.c1000f = g.hs(1000.0)
    g.or16f = g.h([OHR], list(range(OHR)))
    g.c20f = g.h([S], list(range(S)))
    # QLinearConv quantization params (identity)
    g.xs = g.c(F32, [], [1.0])
    g.xz = g.u8s(0)
    g.ws = g.c(F32, [], [1.0])
    g.wz = g.c(I8, [], [0])
    g.wzu = g.u8s(0)
    g.ys = g.c(F32, [], [1.0])
    g.yz = g.u8s(0)


def _qlc(g, x, w, **a):
    return g.nd("QLinearConv", [x, g.xs, g.xz, w, g.ws, g.wz, g.ys, g.yz], **a)


def _qlcu(g, x, w, **a):  # u8 (runtime) weight variant
    return g.nd("QLinearConv", [x, g.xs, g.xz, w, g.ws, g.wzu, g.ys, g.yz], **a)


# =========================================================================== #
def build_209(mode="conv"):
    g = _G()
    _consts(g)

    # ---- value image: 11x11 Conv crops to the 20x20 parse canvas for free ----
    # weight lives at kernel cell (0,0): out[r,c] = sum_ch w[ch]*x[ch,r,c];
    # yellow(4) maps to 10 so a row/col contains yellow iff its max == 10.
    wv = np.zeros((1, 10, 11, 11), np.float32)
    for ch in range(10):
        wv[0, ch, 0, 0] = 10.0 if ch == 4 else float(ch)
    wconv = g.c(F32, [1, 10, 11, 11], wv)
    Vf = g.nd("Conv", ["input", wconv], kernel_shape=[11, 11])  # [1,1,20,20] f32
    V = g.cast(Vf, U8)                                          # [1,1,20,20] u8

    # ---- yellow box bbox (yellow == 10 dominates every row/col max) ----
    c10 = g.u8s(10)
    rmx = g.nd("ReduceMax", [V], axes=[3], keepdims=1)         # [20,1] row max
    cmx = g.nd("ReduceMax", [V], axes=[2], keepdims=1)         # [1,20] col max
    yhr = g.cast(g.nd("Equal", [rmx, c10]), U8)
    yhc = g.cast(g.nd("Equal", [cmx, c10]), U8)
    br, mr = _maxmin(g, yhr, g.r20u, g.revr)
    bc, mc = _maxmin(g, yhc, g.c20u, g.revc)
    tall = g.nd("Add", [g.nd("Sub", [mr, br]), g.c1])
    wide = g.nd("Add", [g.nd("Sub", [mc, bc]), g.c1])

    # ---- colour row presence; shown top row aR and sprite row span ----
    # a row holds colour iff 0 < max < 10 (frame rows max out at 10); rows in
    # [br,mr] with colour are shown rows, rows below mr with colour are sprite.
    hr_col = g.nd("Mul", [g.cast(g.nd("Greater", [rmx, g.c0]), U8),
                          g.cast(g.nd("Less", [rmx, c10]), U8)])       # [20,1]
    grm = g.cast(g.nd("Greater", [g.r20u, mr]), U8)            # rows below the box
    sprh = g.nd("Mul", [hr_col, grm])                          # sprite rows
    aR = _minidx(g, g.nd("Sub", [hr_col, sprh]), g.revr)       # shown top row
    sr0, sr1 = _maxmin(g, sprh, g.r20u, g.revr)
    tp = g.nd("Add", [g.nd("Sub", [sr1, sr0]), g.c1])

    # ---- sprite -> Psmall [1,1,3,5]: row Gather from V (sprite rows are pure
    # sprite: below the box there is no yellow/shown), junk rows masked ----
    ridx, rm3 = _gidx(g, g.k3, sr0, gate=tp)        # k<tp -> sr0+k<=sr1<=19
    G1 = g.nd("Mul", [g.nd("Gather", [V, ridx], axis=2),
                      g.nd("Reshape", [rm3, g.i64([3, 1])])])  # [1,1,3,20]
    hs_c = g.cast(g.nd("Greater", [g.nd("ReduceMax", [G1], axes=[2], keepdims=1),
                                   g.c0]), U8)                 # [1,20] sprite cols
    sc0, sc1 = _maxmin(g, hs_c, g.c20u, g.revc)
    wp = g.nd("Add", [g.nd("Sub", [sc1, sc0]), g.c1])
    cidx, cm = _gidx(g, g.k5, sc0, gate=wp)
    Psm = g.nd("Mul", [g.nd("Gather", [G1, cidx], axis=3), cm])

    # ---- template one-hot [1,4,3,5] ----
    Poh = g.cast(g.nd("Equal", [Psm, g.colv]), U8)

    # ---- shown band [1,1,12,20] from V (rows aR..aR+11, strictly above mr):
    # its cells are exactly the shown values, so it also yields aC ----
    qrt = g.nd("Add", [g.k12, aR])
    qr = g.cast(g.nd("Mul", [qrt, g.cast(g.nd("Less", [qrt, g.c20s]), U8)]), I32)
    rin = g.nd("Reshape", [g.cast(g.nd("Less", [qrt, mr]), U8), g.i64([WIN, 1])])
    VRm = g.nd("Mul", [g.nd("Gather", [V, qr], axis=2), rin])  # [1,1,12,20]
    hasc = g.cast(g.nd("Greater", [g.nd("ReduceMax", [VRm], axes=[2], keepdims=1),
                                   g.c0]), U8)                 # [1,20] shown cols
    aC = _minidx(g, hasc, g.revc)                              # shown left col
    qct = g.nd("Add", [g.k12, aC])
    qc = g.cast(g.nd("Mul", [qct, g.cast(g.nd("Less", [qct, g.c20s]), U8)]), I32)
    cin = g.cast(g.nd("Less", [qct, mc]), U8)                  # [12]
    Wsb = g.nd("Mul", [g.nd("Gather", [VRm, qc], axis=3), cin])  # [1,1,12,12]
    Qoh = g.cast(g.nd("Equal", [Wsb, g.colv]), U8)             # [1,4,12,12] one-hot

    # ---- 45-way match via block counting ----
    # Qcnt[m][c,u,v] = #shown cells of colour c in the m x m block (u,v); the
    # corr of Poh with Qcnt (pads k-1) counts shown cells whose block colour
    # matches P at (rt+u, cl+v) — identical to the incumbent's cell-aligned
    # one-hot correlation at offsets (rt*m, cl*m).
    parts = []
    Qcnt = {}
    for mag in (2, 3, 4):
        k = WIN // mag
        wsum = g.c(I8, [4, 1, mag, mag], [1] * (4 * mag * mag))
        Qcnt[mag] = _qlc(g, Qoh, wsum, group=4, strides=[mag, mag])  # [1,4,k,k] u8
        parts.append(_qlcu(g, Poh, Qcnt[mag], pads=[0, 0, k - 1, k - 1]))  # [1,1,3,5]
    cc = g.nd("Concat", parts, axis=0)                         # [3,1,3,5] u8
    # shown count = sum of Qcnt[4] (two tiny all-ones convs)
    cnt = _qlc(g, _qlc(g, Qcnt[4], g.c(I8, [1, 4, 3, 1], [1] * 12)),
               g.c(I8, [1, 1, 1, 3], [1] * 3))                 # [1,1,1,1]
    mok = g.cast(g.nd("Equal", [cc, cnt]), U8)                 # [3,1,3,5] 0/1

    # ---- fits (u8 0/1 over the [mag,1,rt,cl] lattice; exact int compares) ----
    MAGu = g.u8([3, 1, 1, 1], [2, 3, 4])
    RMvu = g.u8([3, 1, 3, 1], [rt * m for m in (2, 3, 4) for rt in range(3)])
    CMvu = g.u8([3, 1, 1, 5], [cl * m for m in (2, 3, 4) for cl in range(5)])
    aRb = g.nd("Sub", [aR, br])                                # >=1 (shown interior)
    aCb = g.nd("Sub", [aC, bc])
    c1b = g.nd("Less", [RMvu, aRb])                            # Or = aRb-RM >= 1
    c2b = g.nd("Less", [CMvu, aCb])                            # Oc >= 1
    # Or+mag*tp<=tall  <=>  mag*tp+aRb-RM < tall+1 (no wrap when c1 holds)
    l3 = g.nd("Sub", [g.nd("Add", [g.nd("Mul", [MAGu, tp]), aRb]), RMvu])
    c3b = g.nd("Less", [l3, g.nd("Add", [tall, g.c1])])        # [3,1,3,1]
    l4 = g.nd("Sub", [g.nd("Add", [g.nd("Mul", [MAGu, wp]), aCb]), CMvu])
    c4b = g.nd("Less", [l4, g.nd("Add", [wide, g.c1])])        # [3,1,1,5]
    fr = g.nd("Mul", [g.cast(c1b, U8), g.cast(c3b, U8)])       # [3,1,3,1]
    fc = g.nd("Mul", [g.cast(c2b, U8), g.cast(c4b, U8)])       # [3,1,1,5]
    valid = g.nd("Mul", [g.nd("Mul", [fr, fc]), mok])          # [3,1,3,5] u8 0/1

    # ---- hierarchical arg-min: min mag, then min Or+16, then min Oc+16 ----
    # vXc carries valid*100 through the stages (invalid candidates get +100).
    c16 = g.u8s(16)
    c100 = g.u8s(100)
    M100 = g.u8([3, 1, 1, 1], [102, 103, 104])
    v1c = g.nd("Mul", [valid, c100])                           # [3,1,3,5]
    mag_sel = g.nd("ReduceMin", [g.nd("Sub", [M100, v1c])], keepdims=0)
    v2c = g.nd("Mul", [v1c, g.cast(g.nd("Equal", [MAGu, mag_sel]), U8)])
    Orb = g.nd("Sub", [g.nd("Add", [aRb, c16]), RMvu])         # Or+16, [3,1,3,1]
    Or_sel = g.nd("ReduceMin", [g.nd("Sub", [g.nd("Add", [Orb, c100]), v2c])],
                  keepdims=0)                                  # (Or+16 of winner)
    v3c = g.nd("Mul", [v2c, g.cast(g.nd("Equal", [Orb, Or_sel]), U8)])
    Ocb = g.nd("Sub", [g.nd("Add", [aCb, c16]), CMvu])         # Oc+16, [3,1,1,5]
    Oc_sel = g.nd("ReduceMin", [g.nd("Sub", [g.nd("Add", [Ocb, c100]), v3c])],
                  keepdims=0)

    # ---- winner template on a [1,1,12,20] u8 canvas (mask, resize, then
    # cascade-pad the running sum through the growing canvases) ----
    Tm = {}
    for mag in (2, 3, 4):
        emu = g.cast(g.nd("Equal", [mag_sel, g.u8s(mag)]), U8)
        pm = g.nd("Mul", [Psm, emu])                           # [1,1,3,5]
        Tm[mag] = g.nd("Resize", [pm, "", g.c(F32, [4], [1, 1, mag, mag])],
                       mode="nearest")                         # [1,1,3m,5m]
    s2 = g.nd("Pad", [Tm[2], g.i64([0, 0, 0, 0, 0, 0, 3, 5]), g.c0],
              mode="constant")                                 # [1,1,9,15]
    s3 = g.nd("Pad", [g.nd("Add", [s2, Tm[3]]), g.i64([0, 0, 0, 0, 0, 0, 3, 5]),
                      g.c0], mode="constant")                  # [1,1,12,20]
    Msel = g.nd("Add", [s3, Tm[4]])                            # [1,1,12,20]
    # zero guard row/col in front: clamped (out-of-sprite) picks read zeros,
    # so the placement Gathers need no data masks.
    Msel2 = g.nd("Pad", [Msel, g.i64([0, 0, 1, 1, 0, 0, 0, 0]), g.c0],
                 mode="constant")                              # [1,1,13,21]

    # ---- place at (Or_sel-16, Oc_sel-16) on the 16x20 canvas ----
    # d = (r+16) - Or_sel = r - Or; u8 wraparound (r < Or, or the no-valid
    # sentinel offsets) lands >= the mask bound and picks the zero guard.
    def _pidx(coords16, off, bound):
        d = g.nd("Sub", [coords16, off])
        m = g.cast(g.nd("Less", [d, bound]), U8)
        return g.cast(g.nd("Mul", [g.nd("Add", [d, g.c1]), m]), I32)  # 1+d or 0

    or16p = g.u8([OHR], [r + 16 for r in range(OHR)])
    c20p = g.u8([S], [c + 16 for c in range(S)])
    ridxp = _pidx(or16p, Or_sel, g.u8s(WIN))                   # [16] in [0,12]
    cidxp = _pidx(c20p, Oc_sel, g.c20s)                        # [20] in [0,20]
    Placed = g.nd("Gather", [g.nd("Gather", [Msel2, ridxp], axis=2), cidxp], axis=3)

    # ---- corners + compose (u8, sentinel 255/254 outside the box) ----
    tm1 = g.nd("Sub", [tall, g.c1])
    wm1 = g.nd("Sub", [wide, g.c1])

    def _edge(idx, last1):
        e0 = g.cast(g.nd("Equal", [idx, g.c0]), U8)
        e1 = g.cast(g.nd("Equal", [idx, last1]), U8)
        return g.nd("Add", [e0, e1])

    cr4 = g.nd("Mul", [_edge(g.or16u, tm1), g.c4u])            # [16,1]
    corners = g.nd("Mul", [cr4, _edge(g.c20u, wm1)])           # [16,20] 0/4
    ez = g.cast(g.nd("Equal", [Placed, g.c0]), U8)
    OUTv = g.nd("Add", [Placed, g.nd("Mul", [corners, ez])])
    # OUTv is provably 0 outside the box (fits => sprite strictly inside;
    # corners live on the box frame), so adding 255*(r>=tall) + 255*(c>=wide)
    # leaves box cells intact and turns everything else into 255/254 (no
    # colour channel matches either sentinel).
    outr = g.nd("Mul", [g.cast(g.nd("Greater", [g.or16u, tm1]), U8), g.c255])  # [16,1]
    outc = g.nd("Mul", [g.cast(g.nd("Greater", [g.c20u, wm1]), U8), g.c255])   # [1,20]
    OUT2 = g.nd("Add", [OUTv, g.nd("Add", [outr, outc])])

    # ---- FREE one-hot output: Pad(255) to 30x30, Equal writes `output` ----
    padded = g.nd("Pad", [OUT2, g.i64([0, 0, 0, 0, 0, 0, H30 - OHR, H30 - S]),
                          g.c255], mode="constant")            # [1,1,30,30] u8
    g.nodes.append(oh.make_node("Equal", [padded, g.valvec], ["output"]))

    x = oh.make_tensor_value_info("input", F32, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", BOOL, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, "ed209b", [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET)
    onnx.checker.check_model(m, full_check=True)
    return m


# =========================================================================== #
# detection / candidates  (reuse the reference numerics from family_rb209)      #
# =========================================================================== #
from family_rb209 import _ref, _pairs, _matches  # noqa: E402


def candidates(ex):
    prs = _pairs(ex)
    if not _matches(prs):
        return []
    try:
        return [("ed209b", build_209())]
    except Exception:
        return []
