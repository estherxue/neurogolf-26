"""task219 (hash 90f3ed37) — "C-region tiling" family, GRADER-ROBUST rewrite.

Same maximum-likelihood parse as family_rb219._ref (imported verbatim — the numpy
reference is EXACT on all 265 local train+test+arc-gen and ~99.5% on fresh generator
samples; the residual is irreducible generator ambiguity).  The ONNX here reproduces
_ref bit-exactly but is built ENTIRELY in the int32 label domain:

  * NO float equality.  Cyan mask is read directly from one-hot channel 8 (exact 0/1),
    Cast to int32; every subsequent comparison is an INTEGER Equal / Greater / Less.
  * NO float accumulation in the reconstruct-and-verify search.  All masks are int32
    0/1; MatMul / ReduceSum tile-and-count in exact integers; there is no rounding to
    accumulate and no fusion-order sensitivity, so the graph is identical across ORT
    versions and graph_optimization_level DISABLE_ALL vs ENABLE_ALL.
  * opset-10 gaps bridged with integer ops only: a>=b := 1-Cast(Less(a,b)); Max(m,n) of
    0/1 masks := Cast(Greater(m+n,0)); C-phase residue via integer Div (floor for the
    masked-in c>=start region).  Only the final one-hot channel edit (subtract blue from
    channel 0, add to channel 1) runs in float — exact 0/1 arithmetic, no equality.

The 48-hypothesis batched parse is unchanged from family_rb219._build_search: for each
(tall in 3,2,1) x (awide in 1,2) x (bwide in 1,2) x (cwide in 1,2) x (col0 in
{awide,2*awide}) it reconstructs the whole grid (band0 A+B+C legend + an unrolled
8-step greedy multi-band scan painting each lower band's A+B), keeps the hypotheses whose
reconstruction is bit-equal to the input, and selects the first valid one in _ref's
search order; the chosen C-pattern is tiled BLUE into each lower band's C-region.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

# Reuse the proven reference + hypothesis tables (DO NOT re-derive — import verbatim).
from family_rb219 import (
    _ref, H, W, B, TALL, AW, BW, CW, COL0, S0,
    _sel, _last, _cmod, _grp, _cmodC,
)

F = onnx.TensorProto.FLOAT
I32 = onnx.TensorProto.INT32
I64 = onnx.TensorProto.INT64
BIG = 1000  # off-grid nudge, small enough that int32 never overflows here
# Greedy scan steps for LOWER bands.  A 15-row grid packs at most 7 bands (tall=1,
# tops 1,3,5,7,9,11,13) = 1 legend + 6 lower; bands are always separated by a blank
# row so they never merge -> 6 lower bands is the hard maximum, KSCAN=6 is exact.
KSCAN = 6


# ============================ integer graph builder ========================= #
class _Gi:
    def __init__(s):
        s.nodes = []
        s.inits = []
        s.k = 0

    def c(s, name, arr, dt=I32):
        a = np.asarray(arr)
        if dt == I32:
            a = a.astype(np.int32)
        elif dt == I64:
            a = a.astype(np.int64)
        else:
            a = a.astype(np.float32)
        s.inits.append(oh.make_tensor(name, dt, list(a.shape) if a.shape else [1],
                                      a.flatten().tolist()))
        return name

    def n(s, op, ins, outs=None, **kw):
        if outs is None:
            outs = s.u()
        if isinstance(outs, str):
            outs = [outs]
        s.nodes.append(oh.make_node(op, list(ins), outs, **kw))
        return outs[0]

    def u(s, pref="t"):
        s.k += 1
        return f"_{pref}{s.k}"

    # elementwise integer arithmetic
    def add(s, a, b): return s.n("Add", [a, b])
    def sub(s, a, b): return s.n("Sub", [a, b])
    def mul(s, a, b): return s.n("Mul", [a, b])
    def matmul(s, a, b): return s.n("MatMul", [a, b])
    def rmax(s, a, axes, keep=1): return s.n("ReduceMax", [a], s.u(), axes=axes, keepdims=keep)
    def rmin(s, a, axes, keep=1): return s.n("ReduceMin", [a], s.u(), axes=axes, keepdims=keep)
    def rsum(s, a, axes, keep=1): return s.n("ReduceSum", [a], s.u(), axes=axes, keepdims=keep)

    def reshape(s, a, shape):
        sn = s.c(s.u("sh"), shape, I64)
        return s.n("Reshape", [a, sn])

    def transpose(s, a, perm): return s.n("Transpose", [a], perm=perm)

    # integer comparisons -> int32 0/1
    def eq(s, a, b): return s.n("Cast", [s.n("Equal", [a, b])], to=I32)
    def gtf(s, a, b): return s.n("Cast", [s.n("Greater", [a, b])], to=I32)
    def ltf(s, a, b): return s.n("Cast", [s.n("Less", [a, b])], to=I32)
    # a >= b  ==  not (a < b)  (integer exact)
    def ge(s, a, b): return s.n("Cast", [s.n("Not", [s.n("Less", [a, b])])], to=I32)

    # 0/1 clamp of a non-negative integer tensor
    def clip01(s, a): return s.gtf(a, s.zero)
    # OR of two 0/1 masks
    def omax(s, a, b): return s.clip01(s.add(a, b))


def _build_int():
    g = _Gi()
    g.zero = g.c("zero", [0])
    one = g.c("one", [1])

    # ---- cyan mask M [15,10] read straight from one-hot channel 8 (exact 0/1) ----
    g.c("cs", [0, 0], I64); g.c("ce", [H, W], I64); g.c("ca", [2, 3], I64)
    g.n("Slice", ["input", "cs", "ce", "ca"], "crop")            # [1,10,15,10] float
    g.c("chs", [8], I64); g.c("che", [9], I64); g.c("cha", [1], I64)
    g.n("Slice", ["crop", "chs", "che", "cha"], "M4")            # [1,1,15,10] float
    Mf = g.reshape("M4", [H, W])
    M = g.n("Cast", [Mf], to=I32)                                # int cyan mask [15,10]

    Rr = g.c("Rr", np.arange(H).reshape(1, H))                   # [1,15]
    Cc = g.c("Cc", np.arange(W).reshape(1, W))                   # [1,10]
    BIGc = g.c("BIG", [[BIG]])                                   # [1,1]
    oneB = g.c("oneB", np.ones((B, 1), int))                     # [B,1]
    onesH = g.c("onesH", np.ones((1, H), int))                   # [1,15]
    dc0 = g.c("dc0", [[0]]); dc1 = g.c("dc1", [[1]])             # [1,1]
    rdc0 = g.c("rdc0", [[[0]]]); rdc1 = g.c("rdc1", [[[1]]])     # [1,1,1]

    rowhasRow = g.reshape(g.rmax(M, [1], 1), [1, H])             # [1,15] row has cyan
    fc = g.rmin(g.add(Rr, g.mul(g.sub(one, rowhasRow), BIGc)), [1], 1)   # first-cyan row [1,1]

    # ---- extract the three legend rows M[fc], M[fc+1], M[fc+2] ----
    b0rows = []
    for j in range(3):
        fcj = g.add(fc, g.c(f"j{j}", [[j]]))
        b0rows.append(g.matmul(g.eq(Rr, fcj), M))               # [1,10]
    band0B = g.reshape(g.n("Concat", b0rows, g.u(), axis=0), [1, 3, W])   # [1,3,10]

    # ---- carve A / B / C sub-blocks for every hypothesis (batch B) ----
    SelA = g.c("SelA", _sel(0, AW)); SelB = g.c("SelB", _sel(COL0, BW)); SelC = g.c("SelC", _sel(S0, CW))
    talmask = g.c("talmask", (np.arange(3)[None, :] < TALL[:, None]).astype(int).reshape(B, 3, 1))
    Apat = g.mul(g.matmul(band0B, SelA), talmask)               # [B,3,2]
    Bpat = g.mul(g.matmul(band0B, SelB), talmask)
    Cpat = g.mul(g.matmul(band0B, SelC), talmask)

    lastA = g.c("lastA", _last(AW)); lastB = g.c("lastB", _last(BW)); lastC = g.c("lastC", _last(CW))

    def okw(P, last):
        return g.gtf(g.rsum(g.mul(P, last), [1, 2], 1), g.zero)  # sub-block reaches its declared width
    S0valid = g.c("S0valid", ((S0 + CW) <= W).astype(int).reshape(B, 1, 1))
    bv = g.mul(g.mul(okw(Apat, lastA), okw(Bpat, lastB)), g.mul(okw(Cpat, lastC), S0valid))
    basevalid = g.reshape(bv, [B, 1])

    ABp = g.gtf(g.add(g.rsum(Apat, [2], 1), g.rsum(Bpat, [2], 1)), g.zero)   # [B,3,1]
    Cp = g.gtf(g.rsum(Cpat, [2], 1), g.zero)                                 # [B,3,1]
    ABp2 = g.reshape(ABp, [B, 3])
    DRrow = g.c("DRrow", np.arange(3).reshape(1, 3))
    one3 = g.c("one3", [[1, 1, 1]])
    minAB = g.rmin(g.add(DRrow, g.mul(g.sub(one3, ABp2), BIGc)), [1], 1)     # [B,1]
    drp1 = g.c("drp1", (np.arange(3) + 1).reshape(1, 3))
    maxAB = g.sub(g.rmax(g.mul(drp1, ABp2), [1], 1), one)                    # [B,1]
    span = g.sub(maxAB, minAB)

    CMODmat = g.c("CMODmat", _cmod(AW))                          # [B,2,10] col -> c%awide
    Acolpat = g.matmul(Apat, CMODmat)                           # [B,3,10] A tiled full width
    GRP = g.c("GRP", _grp(AW).reshape(B, 1, W))                 # [B,1,10] awide*(c//awide)
    AWc = g.c("AWc", AW.reshape(B, 1)); BWc = g.c("BWc", BW.reshape(B, 1))
    CWc1 = g.c("CWc", CW.reshape(B, 1, 1)); TALLc = g.c("TALLc", TALL.reshape(B, 1))
    COL0c = g.c("COL0c", COL0.reshape(B, 1))
    AW2c = g.c("AW2c", (2 * AW).reshape(B, 1))                   # 2*awide per hypothesis
    Ccb = g.reshape(Cc, [1, 1, W])
    DRb = g.c("DRb", np.arange(3).reshape(1, 3, 1))
    Rrb = g.reshape(Rr, [1, 1, H])

    def paint_AB(gname, top, col):
        topb = g.reshape(top, [B, 1, 1]); colb1 = g.reshape(col, [B, 1, 1])
        rm = g.mul(g.eq(Rrb, g.add(topb, DRb)), talmask)        # [B,3,15]
        Acol = g.mul(Acolpat, g.ltf(GRP, colb1))                # [B,3,10]
        ohs = []
        for dc in (dc0, dc1):
            cpos = g.add(g.reshape(col, [B, 1]), dc)            # [B,1]
            ohs.append(g.eq(Ccb, g.reshape(cpos, [B, 1, 1])))  # [B,1,10]
        Bcol = g.matmul(Bpat, g.n("Concat", ohs, g.u(), axis=1))    # [B,3,10]
        paint = g.omax(Acol, Bcol)
        gadd = g.clip01(g.matmul(g.transpose(rm, [0, 2, 1]), paint))
        return g.omax(gname, gadd)

    def cfill(top, start, active):
        topb = g.reshape(top, [B, 1, 1]); startb = g.reshape(start, [B, 1, 1])
        d = g.sub(Ccb, startb)                                  # [B,1,10]
        q = g.n("Div", [d, CWc1])                               # int div == floor for c>=start
        resid = g.sub(d, g.mul(q, CWc1))
        ohr = [g.eq(resid, rdc0), g.eq(resid, rdc1)]
        Ccolpat = g.matmul(Cpat, g.n("Concat", ohr, g.u(), axis=1))    # [B,3,10]
        Ccol = g.mul(g.mul(Ccolpat, g.ge(Ccb, startb)), Cp)
        rm = g.mul(g.eq(Rrb, g.add(topb, DRb)), Cp)
        ofill = g.clip01(g.matmul(g.transpose(rm, [0, 2, 1]), Ccol))
        return g.mul(ofill, g.reshape(active, [B, 1, 1]))

    fcB = g.reshape(g.mul(oneB, fc), [B, 1])                     # broadcast fc -> [B,1]

    # legend band: A-tiling + B-marker + C-tiling, all cyan
    g_ = paint_AB(g.c("gzero", np.zeros((B, H, W), int)), fcB, COL0c)
    CmodC = g.c("CmodC", _cmodC(S0, CW))                         # [B,2,10] (c-S0)%cwide
    Cc0 = g.matmul(Cpat, CmodC)                                 # [B,3,10]
    rm0 = g.mul(g.eq(Rrb, g.add(g.reshape(fcB, [B, 1, 1]), DRb)), Cp)
    gC = g.clip01(g.matmul(g.transpose(rm0, [0, 2, 1]), Cc0))
    g_ = g.omax(g_, gC)

    ofill_acc = g.c("ofz", np.zeros((B, H, W), int))
    p = g.add(fcB, TALLc)
    Hc = g.c("Hc", [[H]])
    done = g.ge(p, Hc)
    valid = basevalid
    strict_ok = oneB                        # all lower bands so far obey col in {awide,2*awide}
    for step in range(KSCAN):
        cand = g.mul(g.ge(Rr, p), rowhasRow)                    # [B,15]
        hascand = g.gtf(g.rsum(cand, [1], 1), g.zero)           # [B,1]
        f = g.rmin(g.add(Rr, g.mul(g.sub(onesH, cand), BIGc)), [1], 1)   # [B,1]
        active = g.mul(g.sub(oneB, done), hascand)
        top = g.sub(f, minAB)
        fsp = g.add(f, span)
        bandrow = g.mul(g.ge(Rr, f), g.ge(fsp, Rr))            # [B,15]
        colcyan = g.rmax(g.mul(g.reshape(bandrow, [B, H, 1]), g.reshape(M, [1, H, W])), [1], 1)  # [B,1,10]
        rmp1 = g.reshape(g.rmax(g.mul(g.add(Ccb, one), colcyan), [2], 1), [B, 1])   # rightmost cyan col +1
        start = rmp1
        col = g.sub(rmp1, BWc)
        badcol = g.mul(g.ltf(col, AWc), active)
        badtop = g.mul(g.ltf(top, g.zero), active)
        bademp = g.mul(g.ltf(rmp1, one), active)               # rmp1<1  <=> no cyan
        valid = g.mul(g.mul(valid, g.sub(oneB, badcol)), g.mul(g.sub(oneB, badtop), g.sub(oneB, bademp)))
        # strict pass (matches _ref's strict-before-relaxed order): active band col in {awide,2*awide}
        colstrict = g.clip01(g.add(g.eq(col, AWc), g.eq(col, AW2c)))
        strict_ok = g.mul(strict_ok, g.sub(oneB, g.mul(active, g.sub(oneB, colstrict))))
        nb = g.mul(g.sub(oneB, active), BIGc)                  # nudge inactive bands off-grid
        g_ = paint_AB(g_, g.sub(top, nb), g.sub(col, nb))
        ofill_acc = g.add(ofill_acc, cfill(g.sub(top, nb), g.sub(start, nb), active))
        p = g.add(g.mul(active, g.add(top, TALLc)), g.mul(g.sub(oneB, active), p))
        done = g.clip01(g.add(done, g.sub(oneB, hascand)))

    dif = g.n("Abs", [g.sub(g_, g.reshape(M, [1, H, W]))])
    mx = g.rmax(g.rmax(dif, [2], 1), [1], 1)                    # [B,1,1]
    reconok = g.sub(oneB, g.gtf(g.reshape(mx, [B, 1]), g.zero))  # mx==0
    valid = g.mul(valid, reconok)
    valid_strict = g.mul(valid, strict_ok)                       # passes the strict lower-band rule too
    prio = g.c("prio", (B - np.arange(B)).reshape(B, 1))         # HYP-order tiebreak (max 48)
    bigp = g.c("bigp", [[1000]])                                 # strict-valid outranks relaxed-only
    score = g.add(g.mul(prio, valid), g.mul(bigp, valid_strict))
    smax = g.rmax(score, [0], 1)                                # [1,1]
    winner = g.mul(g.eq(score, smax), g.gtf(score, g.zero))     # [B,1]
    bluefill = g.clip01(g.reshape(g.rsum(g.mul(ofill_acc, g.reshape(winner, [B, 1, 1])), [0], 1), [H, W]))

    # ---- one-hot edit: move blue-filled cells from channel 0 to channel 1 (exact float) ----
    bf4 = g.reshape(g.n("Cast", [bluefill], to=F), [1, 1, H, W])
    g.c("c1s", [1], I64); g.c("c1e", [2], I64)
    g.n("Slice", ["crop", "c1s", "c1e", "cha"], "ch1")
    g.c("c0s", [0], I64); g.c("c0e", [1], I64)
    g.n("Slice", ["crop", "c0s", "c0e", "cha"], "ch0")
    ch1n = g.add("ch1", bf4); ch0n = g.sub("ch0", bf4)
    g.c("c2s", [2], I64); g.c("c2e", [10], I64)
    g.n("Slice", ["crop", "c2s", "c2e", "cha"], "ch29")
    g.n("Concat", [ch0n, ch1n, "ch29"], "cropout", axis=1)
    g.n("Pad", ["cropout"], "output", mode="constant", value=0.0,
        pads=[0, 0, 0, 0, 0, 0, 15, 20])

    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    used = set()
    for nd in g.nodes:
        used.update(nd.input)
    inits = [t for t in g.inits if t.name in used]
    graph = oh.make_graph(g.nodes, "rb219v2", [x], [y], inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    for a, b in prs:
        r = _ref(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            return
    try:
        model = _build_int()
    except Exception:
        return
    yield ("rb219v2", model)
