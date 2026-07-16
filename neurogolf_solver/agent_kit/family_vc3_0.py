"""family_vc3_0 — task096 / 4290ef0e : static ONNX for the D4 frame-mandala.

The exact numpy reference (D4-symmetric frame mandala, 266/266) lives in
family_vc2_5.solve096.  This module ships the matching static opset-10 ONNX.

Rule recap (see family_vc2_5 for the full write-up):
  * background = majority colour; centre dot = the count-1 colour (if any);
    frame colours = the rest; d = #frames  (output size (2d+1), d in {3,4,5}).
  * each frame colour's input fragment is a rectangular crop of a canonical
    corner-frame CAN(r,L)  (r in 1..d, arm length L in 2..r+1).
  * feasible (r,L) per colour = templates whose exact rectangular sub-window is
    the fragment; a colour with a UNIQUE feasible radius claims it; the single
    ambiguous elbow [[1,1],[1,0]] takes the one leftover radius; L = max feasible.
  * redraw d concentric CAN(r,L) frames centred at (d,d) + the centre dot.

ONNX realisation (all origin-anchored, fully static):
  1. per-channel origin-crop of each fragment (MatMul selection) -> 11x11 kernel.
  2. sub-window feasibility = batched Conv of every fragment kernel against the
     15 fixed CAN templates (exact match via corr==|frag| AND bboxcorr==|frag|).
  3. radius assignment: unique-radius claim + ascending leftover match (cumsum
     via a triangular MatMul) — reproduces the numpy tie-break.
  4. per-cell redraw from centred coordinates (u,v): ring = max(|u|,|v|); a ring-r
     cell is painted iff it lies within arm length L_r of a corner.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS
from family_vc2_5 import solve096, _canonical

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64

# fixed canonical template set: (r, L), r=1..5, L=2..r+1  -> 15 templates
_TPL = [(r, L) for r in range(1, 6) for L in range(2, r + 2)]
_NT = len(_TPL)                 # 15
_CANV = 21                      # image canvas for sliding
_KS = 11                        # fragment kernel size / max frame
# slice ranges on the template axis, grouped by radius r=1..5
_RSL = {1: (0, 1), 2: (1, 3), 3: (3, 6), 4: (6, 10), 5: (10, 15)}


class _G:
    def __init__(self):
        self.nodes, self.inits, self._k = [], [], 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, F, list(dims), [float(v) for v in np.asarray(vals, np.float64).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, "vc3_096", [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


def build_096():
    g = _G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [1000.0])
    chidx = g.f([1, 10, 1, 1], list(range(10)))
    rowidx = g.f([1, 1, 30, 1], list(range(30)))
    colidx = g.f([1, 1, 1, 30], list(range(30)))
    ri11 = g.f([1, 1, _KS, 1], list(range(_KS)))
    ci11 = g.f([1, 1, 1, _KS], list(range(_KS)))

    def gt(a, b):
        return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)

    def lt(a, b):
        return g.nd("Cast", [g.nd("Less", [a, b])], to=F)

    def eqm(a, b):
        return lt(g.nd("Abs", [g.nd("Sub", [a, b])]), half)

    def rsum(a, axes):
        return g.nd("ReduceSum", [a], axes=axes, keepdims=1)

    def rmax(a, axes):
        return g.nd("ReduceMax", [a], axes=axes, keepdims=1)

    # ---- counts / background / centre / frames / d -------------------------
    counts = rsum("input", [2, 3])                              # [1,10,1,1]
    bg_arg = g.nd("ArgMax", [counts], axis=1, keepdims=1)
    bggate = g.nd("Cast", [g.nd("Equal",
                  [bg_arg, g.nd("Cast", [chidx], to=INT64)])], to=F)   # [1,10,1,1]
    notbg = g.nd("Sub", [one, bggate])
    present = gt(counts, half)
    iscenter = eqm(counts, one)                                 # count==1
    isframe = g.nd("Mul", [g.nd("Mul", [present, gt(counts, g.f([1, 1, 1, 1], [1.5]))]), notbg])
    d = rsum(isframe, [1])                                      # [1,1,1,1] in {3,4,5}
    bg_val = rsum(g.nd("Mul", [bggate, chidx]), [1])            # [1,1,1,1]
    center_present = rmax(iscenter, [1])
    center_color = rsum(g.nd("Mul", [iscenter, chidx]), [1])

    # ---- per-channel origin crop -> 11x11 kernel ---------------------------
    rowhas = rmax("input", [3])                                 # [1,10,30,1]
    colhas = rmax("input", [2])                                 # [1,10,1,30]
    maxr = rmax(g.nd("Mul", [rowhas, rowidx]), [2])             # [1,10,1,1]
    minr = g.nd("Sub", [cbig, rmax(g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowidx])]), [2])])
    maxc = rmax(g.nd("Mul", [colhas, colidx]), [3])
    minc = g.nd("Sub", [cbig, rmax(g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])]), [3])])
    fh = g.nd("Add", [g.nd("Sub", [maxr, minr]), one])          # [1,10,1,1]
    fw = g.nd("Add", [g.nd("Sub", [maxc, minc]), one])

    Srow = eqm(colidx, g.nd("Add", [rowidx, minr]))             # [1,10,30,30]
    Scol = eqm(rowidx, g.nd("Add", [colidx, minc]))
    crC = g.nd("MatMul", [g.nd("MatMul", [Srow, "input"]), Scol])   # [1,10,30,30]
    crC11 = g.nd("Slice", [crC, g.i64([0, 0]), g.i64([_KS, _KS]), g.i64([2, 3])])
    crC_w = g.nd("Reshape", [crC11, g.i64([10, 1, _KS, _KS])])

    bboxK = g.nd("Mul", [lt(ri11, fh), lt(ci11, fw)])           # [1,10,11,11]
    bbox_w = g.nd("Reshape", [bboxK, g.i64([10, 1, _KS, _KS])])

    # ---- sub-window feasibility via batched Conv ---------------------------
    Timg = np.zeros((_NT, 1, _CANV, _CANV), np.float32)
    Lvals = []
    for t, (r, L) in enumerate(_TPL):
        n = 2 * r + 1
        Timg[t, 0, :n, :n] = _canonical(r, L)
        Lvals.append(L)
    Timg_c = g.f([_NT, 1, _CANV, _CANV], Timg)
    corr = g.nd("Conv", [Timg_c, crC_w], kernel_shape=[_KS, _KS])   # [15,10,11,11]
    bb = g.nd("Conv", [Timg_c, bbox_w], kernel_shape=[_KS, _KS])
    tgt = counts                                                   # [1,10,1,1]
    mfull = g.nd("Mul", [eqm(corr, tgt), eqm(bb, tgt)])            # [15,10,11,11]
    feas_t = rmax(mfull, [2, 3])                                   # [15,10,1,1]
    Lc = g.f([_NT, 1, 1, 1], Lvals)
    FTL = g.nd("Mul", [feas_t, Lc])                               # [15,10,1,1]

    # ---- per-radius feasibility / maxL, restricted to r<=d -----------------
    feas_r = {}
    maxL_r = {}
    validr = {}
    for r in range(1, 6):
        a, b = _RSL[r]
        sl_f = g.nd("Slice", [feas_t, g.i64([a]), g.i64([b]), g.i64([0])])
        sl_L = g.nd("Slice", [FTL, g.i64([a]), g.i64([b]), g.i64([0])])
        vr = gt(d, g.f([1, 1, 1, 1], [r - 0.5]))                  # [1,1,1,1] (r<=d)
        validr[r] = vr
        feas_r[r] = g.nd("Mul", [rmax(sl_f, [0]), vr])            # [1,10,1,1]
        maxL_r[r] = g.nd("Mul", [rmax(sl_L, [0]), vr])

    nfeas = feas_r[1]
    for r in range(2, 6):
        nfeas = g.nd("Add", [nfeas, feas_r[r]])                   # [1,10,1,1]
    unique = g.nd("Mul", [isframe, eqm(nfeas, one)])
    amb = g.nd("Mul", [isframe, gt(nfeas, g.f([1, 1, 1, 1], [1.5]))])

    # ambrank[c] = #ambiguous colours c' < c  (strictly-lower triangular MatMul)
    P = np.zeros((10, 10), np.float32)
    for cp in range(10):
        for c in range(10):
            if cp < c:
                P[cp, c] = 1.0
    Pc = g.f([10, 10], P)
    ambrow = g.nd("Reshape", [amb, g.i64([1, 10])])
    ambrank = g.nd("Reshape", [g.nd("MatMul", [ambrow, Pc]), g.i64([1, 10, 1, 1])])

    # ---- assignment per radius --------------------------------------------
    color_of_r = {}
    L_of_r = {}
    mrank = g.f([1, 1, 1, 1], [0.0])          # missingrank accumulator (=0 at r=1)
    for r in range(1, 6):
        uassign = g.nd("Mul", [unique, feas_r[r]])               # [1,10,1,1]
        claimed = rsum(uassign, [1])                             # [1,1,1,1]
        missing = g.nd("Mul", [validr[r], lt(claimed, half)])    # [1,1,1,1]
        lassign = g.nd("Mul", [g.nd("Mul", [amb, missing]), eqm(ambrank, mrank)])
        assign = g.nd("Add", [uassign, lassign])                 # [1,10,1,1]
        color_of_r[r] = rsum(g.nd("Mul", [assign, chidx]), [1])  # [1,1,1,1]
        L_of_r[r] = rsum(g.nd("Mul", [assign, maxL_r[r]]), [1])
        mrank = g.nd("Add", [mrank, missing])                    # cumulate for next r

    # ---- redraw from centred coordinates -----------------------------------
    u = g.nd("Sub", [rowidx, d])                                 # [1,1,30,1]
    v = g.nd("Sub", [colidx, d])                                 # [1,1,1,30]
    au = g.nd("Abs", [u])
    av = g.nd("Abs", [v])
    ring = g.nd("Max", [au, av])                                 # [1,1,30,30]
    valid = lt(ring, g.nd("Add", [d, half]))                     # [1,1,30,30]

    framehit = g.f([1, 1, 1, 1], [0.0])
    framecolor = g.f([1, 1, 1, 1], [0.0])
    for r in range(1, 6):
        rc = g.f([1, 1, 1, 1], [float(r)])
        # thr = r - L + 1 = (r+1) - L ; compare x >= thr  <=>  x > thr-0.5
        thr = g.nd("Sub", [g.f([1, 1, 1, 1], [r + 1.0]), L_of_r[r]])
        thrm = g.nd("Sub", [thr, half])                          # (r+1-L) - 0.5
        c1 = g.nd("Mul", [eqm(au, rc), gt(av, thrm)])            # [1,1,30,30]
        c2 = g.nd("Mul", [eqm(av, rc), gt(au, thrm)])
        orc = gt(g.nd("Add", [c1, c2]), half)
        fg = g.nd("Mul", [g.nd("Mul", [eqm(ring, rc), orc]), validr[r]])
        framehit = g.nd("Add", [framehit, fg])
        framecolor = g.nd("Add", [framecolor, g.nd("Mul", [fg, color_of_r[r]])])

    centerhit = eqm(ring, g.f([1, 1, 1, 1], [0.0]))
    centerval = g.nd("Add", [g.nd("Mul", [center_present, center_color]),
                             g.nd("Mul", [g.nd("Sub", [one, center_present]), bg_val])])
    bghit = g.nd("Mul", [g.nd("Mul", [valid, g.nd("Sub", [one, framehit])]),
                         g.nd("Sub", [one, centerhit])])

    out_frame = g.nd("Mul", [framehit, eqm(framecolor, chidx)])   # [1,10,30,30]
    out_center = g.nd("Mul", [centerhit, eqm(centerval, chidx)])
    out_bg = g.nd("Mul", [bghit, eqm(bg_val, chidx)])
    g.nd("Add", [g.nd("Add", [out_frame, out_center]), out_bg], "output")
    return _model(g)


def _pairs(examples):
    out = []
    for s in ("train", "test"):
        for e in examples.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return []
            if max(a.shape) > 30 or max(b.shape) > 30:
                return []
            out.append((a, b))
    return out


def candidates(examples):
    prs = _pairs(examples)
    if not prs:
        return
    for a, b in prs:
        try:
            o = solve096(a)
        except Exception:
            return
        if o is None or np.array(o).shape != b.shape or not np.array_equal(np.array(o), b):
            return
    try:
        yield ("vc3_4290ef0e", build_096())
    except Exception:
        return
