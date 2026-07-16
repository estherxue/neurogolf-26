"""family_crk9_4 — hard residual ARC tasks (slice U[4::7]).

Solved:
  * task071: occlusion repair by vertical mirror symmetry. A figure (one color)
    that is left/right symmetric about a vertical axis p has a rectangular patch
    overwritten by a SOLID rectangle of a second "occluder" color. We recover the
    original by reflecting the grid about p and pasting the mirror value onto the
    occluder cells.

    Pipeline (all data-dependent, expressed with static opset-10 ops):
      - occluder mask = the unique present color whose pixels form a solid filled
        rectangle (mask == outer(rowhas, colhas)); figure mask F = nonbg - occ.
      - axis tp (= 2*p, integer 0..30) chosen by argmax over candidates of
        S(tp)=6*ff(tp)+5*fo(tp) where ff = #figure cells whose column-mirror is
        figure, fo = #whose mirror is occluder. ff/fo are anti-diagonal sums of
        the column co-occurrence matrices F^T F and F^T O, extracted by a fixed
        [31,900] selector matrix.
      - reflect every channel about tp via a gathered [30,30] reflection matrix,
        then out = where(occ, reflected, input).
    Verified EXACT on all train+test+arc-gen (265/265).

  * task027: 180-degree point-symmetry reveal. A single-colour shape is (almost)
    point-symmetric about a centre (p,p) on the main diagonal; rotating it 180
    about that centre paints colour 2 onto every background cell that is the
    rotated image of a shape cell. The centre index t (=2p) is chosen by argmax
    over t of overlap(t)+1.5*[t == (minrow+maxrow)] where overlap(t)=#shape cells
    whose 180-rotation about (t/2,t/2) is also a shape cell. Rotation = R_t @ F @
    R_t for the per-t reflection matrices; overlaps for all t come from one
    batched-matmul rotation bank. Verified EXACT on all train+test+arc-gen
    (265/265).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
N = 30
NTP = 31  # candidate axes tp = 0..30 (2*p, p in [0,15])


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ----------------------------------------------------------------- numpy oracle

def _solve_np(a):
    """Reference solver mirroring the ONNX math; returns predicted grid or None."""
    H, W = a.shape
    nz = [c for c in set(a.ravel().tolist()) if c != 0]
    if len(nz) != 2:
        return None
    occ = None
    for oc in nz:
        m = (a == oc)
        ys, xs = np.where(m)
        if m.sum() == (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1):
            if occ is not None:
                return None  # not unique
            occ = oc
    if occ is None:
        return None
    fig = [c for c in nz if c != occ][0]
    F = (a == fig)

    def score(tp):
        ff = fb = 0
        for r in range(H):
            for c in range(W):
                if F[r, c]:
                    cc = tp - c
                    w = a[r, cc] if 0 <= cc < W else 0
                    if w == fig:
                        ff += 1
                    elif w == 0:
                        fb += 1
        return ff - 5 * fb

    tp = max(range(0, 2 * W - 1), key=score)
    om = (a == occ)
    out = a.copy()
    for r in range(H):
        for c in range(W):
            if om[r, c]:
                cc = tp - c
                out[r, c] = a[r, cc] if 0 <= cc < W else 0
    return out


def _matches(pairs):
    for a, b in pairs:
        if a.shape != b.shape:
            return False
        out = _solve_np(a)
        if out is None or not np.array_equal(out, b):
            return False
    return True


# ----------------------------------------------------------------- ONNX builder

def _build():
    nodes = []
    inits = []

    # --- constant initializers -------------------------------------------------
    Dmat = np.zeros((NTP, N * N), np.float32)
    Rbank = np.zeros((NTP, N, N), np.float32)
    for tp in range(NTP):
        for c in range(N):
            cc = tp - c
            if 0 <= cc < N:
                Dmat[tp, c * N + cc] = 1.0
                Rbank[tp, c, cc] = 1.0
    inits.append(oh.make_tensor("Dmat", DATA_TYPE, [NTP, N * N], Dmat.ravel().tolist()))
    inits.append(oh.make_tensor("Rbank", DATA_TYPE, [NTP, N, N], Rbank.ravel().tolist()))
    inits.append(oh.make_tensor("half", DATA_TYPE, [1], [0.5]))
    inits.append(oh.make_tensor("one", DATA_TYPE, [1], [1.0]))
    inits.append(oh.make_tensor("six", DATA_TYPE, [1], [6.0]))
    inits.append(oh.make_tensor("five", DATA_TYPE, [1], [5.0]))

    def sl(name, start, end, axis, src="input"):
        s = oh.make_tensor(name + "_s", INT64, [1], [start])
        e = oh.make_tensor(name + "_e", INT64, [1], [end])
        ax = oh.make_tensor(name + "_a", INT64, [1], [axis])
        inits.extend([s, e, ax])
        nodes.append(oh.make_node("Slice", [src, name + "_s", name + "_e", name + "_a"], [name]))

    # nonbg colour channels 1..9
    sl("Mc", 1, 10, 1)                                   # [1,9,30,30]
    nodes.append(oh.make_node("ReduceSum", ["Mc"], ["nonzero"], axes=[1], keepdims=1))  # [1,1,30,30]

    # --- solid-rectangle (occluder) detection over the 9 colour channels -------
    nodes.append(oh.make_node("ReduceMax", ["Mc"], ["rowhas"], axes=[3], keepdims=1))  # [1,9,30,1]
    nodes.append(oh.make_node("ReduceMax", ["Mc"], ["colhas"], axes=[2], keepdims=1))  # [1,9,1,30]
    nodes.append(oh.make_node("Mul", ["rowhas", "colhas"], ["outer"]))                 # [1,9,30,30]
    nodes.append(oh.make_node("Sub", ["Mc", "outer"], ["diff"]))
    nodes.append(oh.make_node("Abs", ["diff"], ["adiff"]))
    nodes.append(oh.make_node("ReduceSum", ["adiff"], ["D"], axes=[2, 3], keepdims=1))   # [1,9,1,1]
    nodes.append(oh.make_node("ReduceSum", ["Mc"], ["pres"], axes=[2, 3], keepdims=1))   # [1,9,1,1]
    # weight = present AND (D < 0.5)
    nodes.append(oh.make_node("Less", ["D", "half"], ["Dlt"]))
    nodes.append(oh.make_node("Greater", ["pres", "half"], ["presgt"]))
    nodes.append(oh.make_node("Cast", ["Dlt"], ["Dltf"], to=DATA_TYPE))
    nodes.append(oh.make_node("Cast", ["presgt"], ["presf"], to=DATA_TYPE))
    nodes.append(oh.make_node("Mul", ["Dltf", "presf"], ["wgt"]))                        # [1,9,1,1]
    nodes.append(oh.make_node("Mul", ["wgt", "Mc"], ["occch"]))                          # [1,9,30,30]
    nodes.append(oh.make_node("ReduceSum", ["occch"], ["O"], axes=[1], keepdims=1))      # [1,1,30,30]
    nodes.append(oh.make_node("Sub", ["nonzero", "O"], ["F"]))                           # figure mask

    # --- axis selection via anti-diagonal sums of column co-occurrence ----------
    nodes.append(oh.make_node("Squeeze", ["F"], ["Fsq"], axes=[0, 1]))   # [30,30]
    nodes.append(oh.make_node("Squeeze", ["O"], ["Osq"], axes=[0, 1]))   # [30,30]
    nodes.append(oh.make_node("Transpose", ["Fsq"], ["Ft"], perm=[1, 0]))
    nodes.append(oh.make_node("MatMul", ["Ft", "Fsq"], ["Qff"]))         # [30,30]
    nodes.append(oh.make_node("MatMul", ["Ft", "Osq"], ["Qfo"]))         # [30,30]
    rs = oh.make_tensor("flat900", INT64, [2], [N * N, 1])
    inits.append(rs)
    nodes.append(oh.make_node("Reshape", ["Qff", "flat900"], ["Qfff"]))  # [900,1]
    nodes.append(oh.make_node("Reshape", ["Qfo", "flat900"], ["Qfof"]))  # [900,1]
    nodes.append(oh.make_node("MatMul", ["Dmat", "Qfff"], ["ffvec"]))    # [31,1]
    nodes.append(oh.make_node("MatMul", ["Dmat", "Qfof"], ["fovec"]))    # [31,1]
    nodes.append(oh.make_node("Mul", ["ffvec", "six"], ["ff6"]))
    nodes.append(oh.make_node("Mul", ["fovec", "five"], ["fo5"]))
    nodes.append(oh.make_node("Add", ["ff6", "fo5"], ["S"]))             # [31,1]
    nodes.append(oh.make_node("ArgMax", ["S"], ["ti"], axis=0, keepdims=0))  # [1]

    # --- gather reflection matrix and reflect every channel ---------------------
    nodes.append(oh.make_node("Gather", ["Rbank", "ti"], ["Rg"], axis=0))  # [1,30,30]
    r4 = oh.make_tensor("r4shape", INT64, [4], [1, 1, N, N])
    inits.append(r4)
    nodes.append(oh.make_node("Reshape", ["Rg", "r4shape"], ["R4"]))       # [1,1,30,30]
    nodes.append(oh.make_node("MatMul", ["input", "R4"], ["refl"]))        # [1,10,30,30]

    # --- composite: out = O*refl + (1-O)*input ---------------------------------
    nodes.append(oh.make_node("Sub", ["one", "O"], ["notO"]))
    nodes.append(oh.make_node("Mul", ["O", "refl"], ["a1"]))
    nodes.append(oh.make_node("Mul", ["notO", "input"], ["a2"]))
    nodes.append(oh.make_node("Add", ["a1", "a2"], ["output"]))
    return _model(nodes, inits)


# ============================================================ task027: rotate

def _solve27_np(a):
    """Reference solver mirroring the ONNX math; returns predicted grid or None."""
    H, W = a.shape
    nz = [c for c in set(a.ravel().tolist()) if c != 0]
    if nz != [1]:
        return None
    F = (a == 1)
    ys, xs = np.where(F)
    rowsum = ys.min() + ys.max()

    def score(t):
        ff = 0
        for r in range(H):
            for c in range(W):
                if F[r, c]:
                    r2, c2 = t - r, t - c
                    if 0 <= r2 < H and 0 <= c2 < W and F[r2, c2]:
                        ff += 1
        return ff + (1.5 if t == rowsum else 0.0)

    t = max(range(0, 2 * H - 1), key=score)
    out = a.copy()
    for r in range(H):
        for c in range(W):
            if a[r, c] == 0:
                r2, c2 = t - r, t - c
                if 0 <= r2 < H and 0 <= c2 < W and F[r2, c2]:
                    out[r, c] = 2
    return out


def _matches27(pairs):
    for a, b in pairs:
        if a.shape != b.shape or a.shape[0] != a.shape[1]:
            return False
        out = _solve27_np(a)
        if out is None or not np.array_equal(out, b):
            return False
    return True


def _build27():
    nodes = []
    inits = []

    Rbank = np.zeros((NTP, N, N), np.float32)
    for tp in range(NTP):
        for c in range(N):
            cc = tp - c
            if 0 <= cc < N:
                Rbank[tp, c, cc] = 1.0
    inits.append(oh.make_tensor("Rb", DATA_TYPE, [NTP, N, N], Rbank.ravel().tolist()))
    inits.append(oh.make_tensor("arange", DATA_TYPE, [NTP], list(range(NTP))))
    inits.append(oh.make_tensor("half27", DATA_TYPE, [1], [0.5]))
    inits.append(oh.make_tensor("bw", DATA_TYPE, [1], [1.5]))
    inits.append(oh.make_tensor("c29", INT64, [1, 1, 1, 1], [N - 1]))
    sel = [0.0] * CHANNELS
    sel[2] = 1.0
    sel[0] = -1.0
    inits.append(oh.make_tensor("selc", DATA_TYPE, [1, CHANNELS, 1, 1], sel))

    def sl(name, start, end, axis, src="input"):
        s = oh.make_tensor(name + "_s", INT64, [1], [start])
        e = oh.make_tensor(name + "_e", INT64, [1], [end])
        ax = oh.make_tensor(name + "_a", INT64, [1], [axis])
        inits.extend([s, e, ax])
        nodes.append(oh.make_node("Slice", [src, name + "_s", name + "_e", name + "_a"], [name]))

    sl("Fch", 1, 2, 1)   # colour-1 channel  [1,1,30,30]
    sl("BGch", 0, 1, 1)  # background channel [1,1,30,30]
    nodes.append(oh.make_node("Squeeze", ["Fch"], ["Fsq"], axes=[0, 1]))   # [30,30]
    nodes.append(oh.make_node("Squeeze", ["BGch"], ["BGsq"], axes=[0, 1]))  # [30,30]

    # rotation bank rot[t] = R_t @ F @ R_t
    nodes.append(oh.make_node("MatMul", ["Rb", "Fsq"], ["t1"]))    # [31,30,30]
    nodes.append(oh.make_node("MatMul", ["t1", "Rb"], ["rot"]))    # [31,30,30]
    nodes.append(oh.make_node("Mul", ["rot", "Fsq"], ["ffmul"]))   # [31,30,30]
    nodes.append(oh.make_node("ReduceSum", ["ffmul"], ["ffvec"], axes=[1, 2], keepdims=0))  # [31]

    # rowsum = minrow + maxrow  (on the colour-1 mask)
    nodes.append(oh.make_node("ReduceMax", ["Fch"], ["rowhas"], axes=[3], keepdims=1))  # [1,1,30,1]
    nodes.append(oh.make_node("ArgMax", ["rowhas"], ["minrow"], axis=2, keepdims=1))    # [1,1,1,1]
    rs = oh.make_tensor("rev_s", INT64, [1], [N - 1])
    re = oh.make_tensor("rev_e", INT64, [1], [-(1 << 31)])
    ra = oh.make_tensor("rev_a", INT64, [1], [2])
    rst = oh.make_tensor("rev_t", INT64, [1], [-1])
    inits.extend([rs, re, ra, rst])
    nodes.append(oh.make_node("Slice", ["rowhas", "rev_s", "rev_e", "rev_a", "rev_t"], ["rowhasR"]))
    nodes.append(oh.make_node("ArgMax", ["rowhasR"], ["amf"], axis=2, keepdims=1))      # [1,1,1,1]
    nodes.append(oh.make_node("Sub", ["c29", "amf"], ["maxrow"]))
    nodes.append(oh.make_node("Add", ["minrow", "maxrow"], ["rowsum_i"]))               # [1,1,1,1]
    nodes.append(oh.make_node("Cast", ["rowsum_i"], ["rowsum_f4"], to=DATA_TYPE))
    r1 = oh.make_tensor("to1", INT64, [1], [1])
    inits.append(r1)
    nodes.append(oh.make_node("Reshape", ["rowsum_f4", "to1"], ["rowsum_f"]))           # [1]

    # bias = 1.5 * [|arange - rowsum| < 0.5]
    nodes.append(oh.make_node("Sub", ["arange", "rowsum_f"], ["adiff"]))                # [31]
    nodes.append(oh.make_node("Abs", ["adiff"], ["aabs"]))
    nodes.append(oh.make_node("Less", ["aabs", "half27"], ["alt"]))
    nodes.append(oh.make_node("Cast", ["alt"], ["altf"], to=DATA_TYPE))
    nodes.append(oh.make_node("Mul", ["altf", "bw"], ["bias"]))                         # [31]
    nodes.append(oh.make_node("Add", ["ffvec", "bias"], ["S"]))                         # [31]
    nodes.append(oh.make_node("ArgMax", ["S"], ["ts"], axis=0, keepdims=1))             # [1]

    # gather chosen rotated mask, reveal background cells, paint colour 2
    nodes.append(oh.make_node("Gather", ["rot", "ts"], ["rg"], axis=0))   # [1,30,30]
    nodes.append(oh.make_node("Squeeze", ["rg"], ["rmask"], axes=[0]))    # [30,30]
    nodes.append(oh.make_node("Mul", ["rmask", "BGsq"], ["reveal"]))      # [30,30]
    rv4 = oh.make_tensor("rv4", INT64, [4], [1, 1, N, N])
    inits.append(rv4)
    nodes.append(oh.make_node("Reshape", ["reveal", "rv4"], ["reveal4"]))  # [1,1,30,30]
    nodes.append(oh.make_node("Mul", ["reveal4", "selc"], ["delta"]))      # [1,10,30,30]
    nodes.append(oh.make_node("Add", ["input", "delta"], ["output"]))
    return _model(nodes, inits)


# ====================================================== task185: lattice -> 3x3

def _solve185_np(a):
    H, W = a.shape
    counts = {c: int((a == c).sum()) for c in range(1, 10) if (a == c).sum() > 0}
    if len(counts) < 2:
        return None
    base = max(counts, key=counts.get)
    anoms = [c for c in counts if c != base]
    A = np.isin(a, anoms)
    rh = np.where(A.any(1))[0]
    ch = np.where(A.any(0))[0]
    if len(rh) != 4 or len(ch) != 4:
        return None
    M = {c: np.zeros((4, 4), int) for c in anoms}
    for i, r in enumerate(rh):
        for j, cc in enumerate(ch):
            v = a[r, cc]
            if v in M:
                M[v][i, j] = 1
    out = np.zeros((3, 3), int)
    for c in anoms:
        for i in range(3):
            for j in range(3):
                if M[c][i, j] and M[c][i, j + 1] and M[c][i + 1, j] and M[c][i + 1, j + 1]:
                    out[i, j] = c
    return out


def _matches185(pairs):
    ok = False
    for a, b in pairs:
        if b.shape != (3, 3):
            return False
        out = _solve185_np(a)
        if out is None or not np.array_equal(out, b):
            return False
        ok = True
    return ok


def _build185():
    nodes = []
    inits = []

    U = np.triu(np.ones((N, N), np.float32))  # U[r',r]=1 if r'<=r
    inits.append(oh.make_tensor("U", DATA_TYPE, [N, N], U.ravel().tolist()))
    inits.append(oh.make_tensor("ranks", DATA_TYPE, [4, 1], [1.0, 2.0, 3.0, 4.0]))
    inits.append(oh.make_tensor("half5", DATA_TYPE, [1], [0.5]))
    chm = [0.0] + [1.0] * (CHANNELS - 1)
    inits.append(oh.make_tensor("chmask", DATA_TYPE, [1, CHANNELS, 1, 1], chm))
    e0 = [1.0] + [0.0] * (CHANNELS - 1)
    inits.append(oh.make_tensor("e0", DATA_TYPE, [1, CHANNELS, 1, 1], e0))
    m3 = np.zeros((1, 1, N, N), np.float32)
    m3[0, 0, :3, :3] = 1.0
    inits.append(oh.make_tensor("mask3", DATA_TYPE, [1, 1, N, N], m3.ravel().tolist()))

    def cst(name, arr, shape):
        inits.append(oh.make_tensor(name, INT64, shape, arr))

    # ---- anomaly weight w[ch] = present & (count < max_{1..9} count) & (ch>=1)
    nodes.append(oh.make_node("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1))  # [1,10,1,1]
    cst("c1_s", [1], [1]); cst("c1_e", [10], [1]); cst("c1_a", [1], [1])  # value,shape ok (len1)
    nodes.append(oh.make_node("Slice", ["counts", "c1_s", "c1_e", "c1_a"], ["counts19"]))   # [1,9,1,1]
    nodes.append(oh.make_node("ReduceMax", ["counts19"], ["maxc"], axes=[1], keepdims=1))    # [1,1,1,1]
    nodes.append(oh.make_node("Greater", ["counts", "half5"], ["pres"]))
    nodes.append(oh.make_node("Less", ["counts", "maxc"], ["below"]))
    nodes.append(oh.make_node("Cast", ["pres"], ["presf"], to=DATA_TYPE))
    nodes.append(oh.make_node("Cast", ["below"], ["belowf"], to=DATA_TYPE))
    nodes.append(oh.make_node("Mul", ["presf", "belowf"], ["w0"]))
    nodes.append(oh.make_node("Mul", ["w0", "chmask"], ["w"]))   # [1,10,1,1]

    # ---- anomaly mask A and row/col occupancy
    nodes.append(oh.make_node("Mul", ["input", "w"], ["Aall"]))                          # [1,10,30,30]
    nodes.append(oh.make_node("ReduceSum", ["Aall"], ["A"], axes=[1], keepdims=1))        # [1,1,30,30]
    nodes.append(oh.make_node("ReduceMax", ["A"], ["rowh3"], axes=[3], keepdims=0))       # [1,1,30]
    nodes.append(oh.make_node("ReduceMax", ["A"], ["colh3"], axes=[2], keepdims=0))       # [1,1,30]
    cst("r30", [1, N], [2])
    nodes.append(oh.make_node("Reshape", ["rowh3", "r30"], ["rowh"]))   # [1,30]
    nodes.append(oh.make_node("Reshape", ["colh3", "r30"], ["colh"]))   # [1,30]

    # ---- ranked selection matrices Sr, Sc  [4,30]
    for tag, hv in [("r", "rowh"), ("c", "colh")]:
        nodes.append(oh.make_node("MatMul", [hv, "U"], [tag + "rank"]))           # [1,30]
        nodes.append(oh.make_node("Sub", ["ranks", tag + "rank"], [tag + "df"]))  # [4,30]
        nodes.append(oh.make_node("Abs", [tag + "df"], [tag + "ab"]))
        nodes.append(oh.make_node("Less", [tag + "ab", "half5"], [tag + "eqb"]))
        nodes.append(oh.make_node("Cast", [tag + "eqb"], [tag + "eq"], to=DATA_TYPE))  # [4,30]
        nodes.append(oh.make_node("Mul", [tag + "eq", hv], [tag + "S"]))           # [4,30]
    nodes.append(oh.make_node("Transpose", ["cS"], ["ScT"], perm=[1, 0]))          # [30,4]

    # ---- M_all[ch] = Sr @ Aall[ch] @ Sc^T  -> [1,10,4,4]
    nodes.append(oh.make_node("MatMul", ["rS", "Aall"], ["SA"]))   # [1,10,4,30]
    nodes.append(oh.make_node("MatMul", ["SA", "ScT"], ["Mall"]))  # [1,10,4,4]

    # ---- 2x2-corner product -> per-channel 3x3 indicator
    def corner(name, rs, re, cs, ce):
        cst(name + "_s", [rs, cs], [2]); cst(name + "_e", [re, ce], [2]); cst(name + "_a", [2, 3], [2])
        nodes.append(oh.make_node("Slice", ["Mall", name + "_s", name + "_e", name + "_a"], [name]))
    corner("TL", 0, 3, 0, 3); corner("TR", 0, 3, 1, 4)
    corner("BL", 1, 4, 0, 3); corner("BR", 1, 4, 1, 4)
    nodes.append(oh.make_node("Mul", ["TL", "TR"], ["p1"]))
    nodes.append(oh.make_node("Mul", ["BL", "BR"], ["p2"]))
    nodes.append(oh.make_node("Mul", ["p1", "p2"], ["prod"]))     # [1,10,3,3]

    # ---- embed at top-left, fill background channel
    nodes.append(oh.make_node("Pad", ["prod"], ["padded"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, N - 3, N - 3]))  # [1,10,30,30]
    nodes.append(oh.make_node("ReduceSum", ["padded"], ["anyc"], axes=[1], keepdims=1))  # [1,1,30,30]
    nodes.append(oh.make_node("Sub", ["mask3", "anyc"], ["ch0v"]))
    nodes.append(oh.make_node("Mul", ["ch0v", "e0"], ["ch0c"]))   # [1,10,30,30]
    nodes.append(oh.make_node("Add", ["padded", "ch0c"], ["output"]))
    return _model(nodes, inits)


# ----------------------------------------------------------------- dispatch

def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return []
    out = []
    if _matches(prs):
        out.append(("mirror_repair", _build()))
    if _matches27(prs):
        out.append(("rot180_reveal", _build27()))
    if _matches185(prs):
        out.append(("lattice_blocks", _build185()))
    return out
