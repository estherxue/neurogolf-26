"""task219 (hash 90f3ed37) — "C-region tiling" family, GRAPH-COMPRESSED rewrite.

Bit-identical to family_rb219v2 (and hence to family_rb219._ref, imported verbatim):
same 48-hypothesis reconstruct-and-verify parse, pure int32, NO float equality.  v3
only shrinks the GRAPH so the grader's profiled trace (one event per node per example,
265 examples) stays small and measurable:

  * The greedy multi-band scan (KSCAN=6 sequential steps) is split.  A LEAN recurrence
    computes only the scalar-ish per-step quantities that genuinely depend on the
    previous step: f_k = first cyan row at/after the running search cursor p, and the
    active flag.  p_{k+1}=f_k+(tall-minAB) reproduces v2's p=active*(top+tall)+... update
    exactly (inactive steps push p off-grid, so they stay inactive and their f is unused).
  * Everything else — top/col/start, the three validity gates, the strict-col gate, the
    A+B paint and the blue C-fill — is INPUT to the recurrence only through f_k, so it is
    lifted OUT of the loop and evaluated ONCE, batched over a length-K step axis (single
    big tensor ops instead of 6 unrolled copies).  The legend band is folded in as an
    extra paint slot.  Node count drops ~937 -> <500 with identical outputs.
  * The per-band "rightmost cyan column" scan is replaced by a per-ROW precompute
    (rowRight[r] = max cyan col+1 in row r); a band's rmp1 is then max(rowRight) over its
    rows — identical value, no [B,H,W] intermediate.

All comparisons remain integer Equal/Greater/Less(+Cast); the only float arithmetic is
the final one-hot channel edit (exact 0/1).  Output is invariant across ORT versions and
graph_optimization_level DISABLE_ALL vs ENABLE_ALL.
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
KSCAN = 6   # 15-row grid, blank-separated bands -> <=6 lower bands (see v2 header)
KALL = KSCAN + 1  # + legend slot


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
    onesH = g.c("onesH", np.ones((1, H), int))                   # [1,15]

    rowhasRow = g.reshape(g.rmax(M, [1], 1), [1, H])             # [1,15] row has cyan
    fc = g.rmin(g.add(Rr, g.mul(g.sub(one, rowhasRow), BIGc)), [1], 1)   # first-cyan row [1,1]

    # per-row rightmost cyan col + 1 (0 if the row has no cyan) -> [1,15]
    rowRight = g.reshape(
        g.rmax(g.mul(M, g.add(Cc, one)), [1], 1), [1, H])        # [1,15]

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
    span = g.sub(maxAB, minAB)                                               # [B,1]

    CMODmat = g.c("CMODmat", _cmod(AW))                          # [B,2,10] col -> c%awide
    Acolpat = g.matmul(Apat, CMODmat)                           # [B,3,10] A tiled full width
    GRP = g.c("GRP", _grp(AW).reshape(B, 1, W))                 # [B,1,10] awide*(c//awide)
    AWc = g.c("AWc", AW.reshape(B, 1)); BWc = g.c("BWc", BW.reshape(B, 1))
    TALLc = g.c("TALLc", TALL.reshape(B, 1))
    COL0c = g.c("COL0c", COL0.reshape(B, 1))
    AW2c = g.c("AW2c", (2 * AW).reshape(B, 1))                   # 2*awide per hypothesis

    # ============ LEAN sequential recurrence: only f_k and active_k ============ #
    # p_1 = fc + tall ; p_{k+1} = f_k + (tall - minAB)  (== v2's p update when active).
    delta = g.sub(TALLc, minAB)                                  # [B,1]
    Hc = g.c("Hc", [[H]])
    p = g.add(fc, TALLc)                                         # [B,1]
    done = g.ge(p, Hc)                                           # [B,1]
    fs, acts = [], []
    for step in range(KSCAN):
        mask_ge = g.ge(Rr, p)                                    # [B,15]
        cand = g.mul(mask_ge, rowhasRow)                        # [B,15]
        hascand = g.gtf(g.rsum(cand, [1], 1), g.zero)          # [B,1]
        f = g.rmin(g.add(Rr, g.mul(g.sub(onesH, cand), BIGc)), [1], 1)   # [B,1]
        active = g.mul(g.sub(one, done), hascand)              # [B,1]
        fs.append(f); acts.append(active)
        if step + 1 < KSCAN:
            p = g.add(f, delta)                                 # off-grid once inactive
            done = g.clip01(g.add(done, g.sub(one, hascand)))

    # stack the per-step scalars along a length-K step axis -> [B,K]
    Fstk = g.n("Concat", fs, g.u(), axis=1)                     # [B,6]
    ACT = g.n("Concat", acts, g.u(), axis=1)                    # [B,6]

    # ================= batched per-band geometry over the K axis ================ #
    Rk = g.c("Rk", np.arange(H).reshape(1, 1, H))              # [1,1,15]
    Fk = g.reshape(Fstk, [B, KSCAN, 1])                        # [B,6,1]
    spanK = g.reshape(span, [B, 1, 1])
    top = g.sub(Fstk, minAB)                                    # [B,6]
    fsp = g.add(Fk, spanK)                                      # [B,6,1]
    bandrow = g.mul(g.ge(Rk, Fk), g.ge(fsp, Rk))              # [B,6,15]
    rowRightK = g.reshape(rowRight, [1, 1, H])                 # [1,1,15]
    rmp1 = g.rmax(g.mul(bandrow, rowRightK), [2], 1)          # [B,6,1]
    rmp1_2 = g.reshape(rmp1, [B, KSCAN])                       # [B,6]
    col = g.sub(rmp1_2, BWc)                                    # [B,6]
    start = rmp1_2                                              # [B,6]

    # ---- validity gates (== v2 badcol/badtop/bademp, but batched & AND-reduced) ----
    step_ok = g.mul(g.mul(g.ge(col, AWc), g.ge(top, g.zero)),
                    g.ge(rmp1_2, one))                          # [B,6]
    good = g.sub(one, g.mul(ACT, g.sub(one, step_ok)))         # 1-active*(1-ok) [B,6]
    scan_ok = g.reshape(g.rmin(good, [1], 1), [B, 1])          # AND over K [B,1]
    # ---- strict col in {awide, 2*awide} gate ----
    colstrict = g.clip01(g.add(g.eq(col, AWc), g.eq(col, AW2c)))   # [B,6]
    sgood = g.sub(one, g.mul(ACT, g.sub(one, colstrict)))      # [B,6]
    strict_ok = g.reshape(g.rmin(sgood, [1], 1), [B, 1])       # [B,1]

    # ---- off-grid nudge for inactive lower bands ----
    nb = g.mul(g.sub(one, ACT), BIGc)                          # [B,6]
    top_nb = g.sub(top, nb)                                     # [B,6]
    col_nb = g.sub(col, nb)                                     # [B,6]
    start_nb = g.sub(start, nb)                                 # [B,6]

    # ================= batched A+B reconstruction paint =================== #
    # slots 0..5 = lower bands (nudged), slot 6 = legend (top=fc, col=COL0).
    fcB = g.reshape(g.mul(g.c("oneB", np.ones((B, 1), int)), fc), [B, 1])   # [B,1]
    tops_all = g.n("Concat", [top_nb, fcB], g.u(), axis=1)     # [B,7]
    cols_all = g.n("Concat", [col_nb, COL0c], g.u(), axis=1)   # [B,7]
    Rk4 = g.c("Rk4", np.arange(H).reshape(1, 1, 1, H))        # [1,1,1,15]
    DR4 = g.c("DR4", np.arange(3).reshape(1, 1, 3, 1))       # [1,1,3,1]
    Cc4 = g.reshape(Cc, [1, 1, 1, W])                          # [1,1,1,10]
    talm4 = g.reshape(talmask, [B, 1, 3, 1])                   # [B,1,3,1]
    topsA = g.reshape(tops_all, [B, KALL, 1, 1])              # [B,7,1,1]
    colsA = g.reshape(cols_all, [B, KALL, 1, 1])              # [B,7,1,1]
    rm = g.mul(g.eq(Rk4, g.add(topsA, DR4)), talm4)           # [B,7,3,15]
    Acolpat4 = g.reshape(Acolpat, [B, 1, 3, W])               # [B,1,3,10]
    GRP4 = g.reshape(GRP, [B, 1, 1, W])                        # [B,1,1,10]
    Acol = g.mul(Acolpat4, g.ltf(GRP4, colsA))               # [B,7,3,10]
    dc0 = g.c("dc0", [[[[0]]]]); dc1 = g.c("dc1", [[[[1]]]])   # [1,1,1,1]
    ohB = g.n("Concat", [g.eq(Cc4, g.add(colsA, dc0)),
                         g.eq(Cc4, g.add(colsA, dc1))], g.u(), axis=2)  # [B,7,2,10]
    Bpat4 = g.reshape(Bpat, [B, 1, 3, 2])                      # [B,1,3,2]
    Bcol = g.matmul(Bpat4, ohB)                                # [B,7,3,10]
    paint = g.clip01(g.add(Acol, Bcol))                        # [B,7,3,10]
    gadd = g.clip01(g.matmul(g.transpose(rm, [0, 1, 3, 2]), paint))   # [B,7,15,10]
    g_AB = g.rmax(gadd, [1], 0)                                # [B,15,10]

    # ---- legend C tiling (cyan), added to reconstruction ----
    CmodC = g.c("CmodC", _cmodC(S0, CW))                       # [B,2,10] (c-S0)%cwide
    Cc0 = g.matmul(Cpat, CmodC)                                # [B,3,10]
    DRb = g.c("DRb", np.arange(3).reshape(1, 3, 1))
    Rrb = g.reshape(Rr, [1, 1, H])
    rm0 = g.mul(g.eq(Rrb, g.add(g.reshape(fcB, [B, 1, 1]), DRb)), Cp)   # [B,3,15]
    gC = g.clip01(g.matmul(g.transpose(rm0, [0, 2, 1]), Cc0))  # [B,15,10]
    g_ = g.omax(g_AB, gC)                                       # [B,15,10]

    # ---- reconstruction must equal M exactly ----
    dif = g.n("Abs", [g.sub(g_, g.reshape(M, [1, H, W]))])
    mx = g.rmax(g.rmax(dif, [2], 1), [1], 1)                    # [B,1,1]
    reconok = g.sub(one, g.gtf(g.reshape(mx, [B, 1]), g.zero))  # mx==0 -> [B,1]

    valid = g.mul(g.mul(basevalid, scan_ok), reconok)          # [B,1]
    valid_strict = g.mul(valid, strict_ok)                     # [B,1]
    prio = g.c("prio", (B - np.arange(B)).reshape(B, 1))        # HYP-order tiebreak
    bigp = g.c("bigp", [[1000]])                                # strict outranks relaxed
    score = g.add(g.mul(prio, valid), g.mul(bigp, valid_strict))
    smax = g.rmax(score, [0], 1)                                # [1,1]
    winner = g.mul(g.eq(score, smax), g.gtf(score, g.zero))     # [B,1]

    # ================= batched blue C-fill for the 6 lower bands ============= #
    CWc4 = g.c("CWc4", CW.reshape(B, 1, 1, 1))                 # [B,1,1,1]
    Cp4 = g.reshape(Cp, [B, 1, 3, 1])                          # [B,1,3,1]
    tops6 = g.reshape(top_nb, [B, KSCAN, 1, 1])               # [B,6,1,1]
    starts6 = g.reshape(start_nb, [B, KSCAN, 1, 1])           # [B,6,1,1]
    d = g.sub(Cc4, starts6)                                     # [B,6,1,10]
    q = g.n("Div", [d, CWc4])                                   # int div (floor for c>=start)
    resid = g.sub(d, g.mul(q, CWc4))                           # [B,6,1,10]
    rdc0 = g.c("rdc0", [[[[0]]]]); rdc1 = g.c("rdc1", [[[[1]]]])
    ohr = g.n("Concat", [g.eq(resid, rdc0), g.eq(resid, rdc1)], g.u(), axis=2)  # [B,6,2,10]
    Cpat4 = g.reshape(Cpat, [B, 1, 3, 2])                      # [B,1,3,2]
    Ccolpat = g.matmul(Cpat4, ohr)                             # [B,6,3,10]
    Ccol = g.mul(g.mul(Ccolpat, g.ge(Cc4, starts6)), Cp4)     # [B,6,3,10]
    rmc = g.mul(g.eq(Rk4, g.add(tops6, DR4)), Cp4)            # [B,6,3,15]
    ofill = g.clip01(g.matmul(g.transpose(rmc, [0, 1, 3, 2]), Ccol))   # [B,6,15,10]
    ofill = g.mul(ofill, g.reshape(ACT, [B, KSCAN, 1, 1]))    # gate inactive
    ofill_acc = g.rsum(ofill, [1], 0)                          # [B,15,10]
    bluefill = g.clip01(g.reshape(
        g.rsum(g.mul(ofill_acc, g.reshape(winner, [B, 1, 1])), [0], 1), [H, W]))

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
    graph = oh.make_graph(g.nodes, "rb219v3", [x], [y], inits)
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
    yield ("rb219v3", model)
