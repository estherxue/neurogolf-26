"""family_crk9_3 -- "light through a slit" ray-casting family (task 268).

Rule (verified EXACT on all 266 train/test/arc-gen pairs)
---------------------------------------------------------
The grid contains a single rectangular box OUTLINE drawn in one colour, with a
gap (opening) on exactly one of its four sides; the rest of the grid is
background (0).  Light of colour 4 fills the box interior and shines out through
the opening:

  * INTERIOR FILL: every background cell strictly inside the box bounding-box
    becomes 4.
  * STRAIGHT BEAM: light leaves the opening travelling straight outward,
    keeping the opening's width, until the grid edge.
  * DIFFRACTION RAYS: the two extreme cells of the opening additionally emit a
    45-degree diagonal ray outward (one to each side), to the grid edge.

Everything is data-dependent (box colour, position, size, opening side/width).

ONNX construction (opset-10, static [1,10,30,30])
-------------------------------------------------
All geometry is computed on 1-channel [1,1,30,30] float masks.

  gridmask = ReduceSum_c(input)          1 at real cells, 0 at padding
  bg       = channel-0                   1 at real background cells
  wall     = gridmask - bg               1 at the box-outline cells

Bounding box / borders are obtained with TRIANGULAR [30,30] MatMul prefix
matrices (row prefixes via MatMul(T, x); column prefixes via MatMul(x, T)).
Neighbour shifts use single-tap 3x3 Conv kernels.  Light is then propagated by
masked directional relaxation (L = Max(L, shift(L)*bg)) -- four axis directions
seeded from the interior (interior+beam), and four diagonal directions seeded
from the opening corners (diffraction).  Finally the lit cells (a subset of bg)
are recoloured 4.

The family validates the EXACT rule (incl. arc-gen) in numpy before emitting and
stays silent otherwise.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT
H = HEIGHT
W = WIDTH
NPROP = 30  # propagation steps (covers full 30x30 grid)


# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "crk9_3", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# =========================================================================== #
# numpy mirror of the exact ONNX graph (used for detection)                   #
# =========================================================================== #
def _onehot(g):
    g = np.asarray(g, int); h, w = g.shape
    o = np.zeros((CHANNELS, H, W), np.float32)
    for c in range(CHANNELS):
        o[c, :h, :w] = (g[:h, :w] == c)
    return o, h, w


def _shift(L, di, dj):
    o = np.zeros_like(L)
    rs0 = max(di, 0); rs1 = H + min(di, 0)
    cs0 = max(dj, 0); cs1 = W + min(dj, 0)
    # o[r,c] = L[r-di, c-dj]
    o[rs0:rs1, cs0:cs1] = L[rs0 - di:rs1 - di, cs0 - dj:cs1 - dj]
    return o


def _tri(kind):
    M = np.zeros((H, W), np.float32)
    for i in range(H):
        for j in range(W):
            if (kind == 'lt' and j < i) or (kind == 'gt' and j > i) or \
               (kind == 'le' and j <= i) or (kind == 'ge' and j >= i):
                M[i, j] = 1.0
    return M


_A_lt = _tri('lt'); _A_gt = _tri('gt'); _A_le = _tri('le'); _A_ge = _tri('ge')
_C_le = np.zeros((H, W), np.float32); _C_ge = np.zeros((H, W), np.float32)
_B_lt = np.zeros((H, W), np.float32); _B_gt = np.zeros((H, W), np.float32)
for _j in range(H):
    for _c in range(W):
        if _j <= _c: _C_le[_j, _c] = 1.0
        if _j >= _c: _C_ge[_j, _c] = 1.0
        if _j < _c: _B_lt[_j, _c] = 1.0
        if _j > _c: _B_gt[_j, _c] = 1.0
_ALL = np.ones((H, W), np.float32)


def _g0(x):
    return (x > 0.5).astype(np.float32)


def _prop(seed, bg, di, dj, n=NPROP):
    L = seed.copy()
    for _ in range(n):
        L = np.maximum(L, _shift(L, di, dj) * bg)
    return L


def _simulate(g):
    o, h, w = _onehot(g)
    gridmask = o.sum(0)
    bg = o[0]
    wall = gridmask - bg
    rowHas = _g0(wall @ _ALL)          # row sums broadcast over cols
    colHas = _g0(_ALL @ wall)          # col sums broadcast over rows
    rowAA = _g0(_A_le @ rowHas); rowAB = _g0(_A_ge @ rowHas)
    rowIn = rowAA * rowAB
    colAA = _g0(colHas @ _C_le); colAB = _g0(colHas @ _C_ge)
    colIn = colAA * colAB
    isTop = rowIn - rowIn * _shift(rowIn, 1, 0)
    isBot = rowIn - rowIn * _shift(rowIn, -1, 0)
    isLeft = colIn - colIn * _shift(colIn, 0, 1)
    isRight = colIn - colIn * _shift(colIn, 0, -1)
    inBbox = rowIn * colIn
    border = inBbox * _g0(isTop + isBot + isLeft + isRight)
    gap = border * bg
    rowStrict = rowIn * (1 - isTop) * (1 - isBot)
    colStrict = colIn * (1 - isLeft) * (1 - isRight)
    interior = rowStrict * colStrict * bg
    wallAbove = _shift(wall, 1, 0); wallBelow = _shift(wall, -1, 0)
    wallLeft = _shift(wall, 0, 1); wallRight = _shift(wall, 0, -1)
    rightGap = gap * isRight; leftGap = gap * isLeft
    topGap = gap * isTop; botGap = gap * isBot
    seedUR = _g0(rightGap * wallAbove + topGap * wallRight)
    seedUL = _g0(leftGap * wallAbove + topGap * wallLeft)
    seedDR = _g0(rightGap * wallBelow + botGap * wallRight)
    seedDL = _g0(leftGap * wallBelow + botGap * wallLeft)
    beam = np.zeros((H, W), np.float32)
    for di, dj in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        beam = np.maximum(beam, _prop(interior, bg, di, dj))
    diag = np.zeros((H, W), np.float32)
    diag = np.maximum(diag, _prop(seedUR, bg, -1, 1))
    diag = np.maximum(diag, _prop(seedUL, bg, -1, -1))
    diag = np.maximum(diag, _prop(seedDR, bg, 1, 1))
    diag = np.maximum(diag, _prop(seedDL, bg, 1, -1))
    lit = _g0(beam + diag)
    out = o.copy()
    out[4] = out[4] + lit
    out[0] = out[0] - lit
    pos = out > 0
    res = np.full((h, w), -99, int)
    for r in range(h):
        for c in range(w):
            idx = np.where(pos[:, r, c])[0]
            if idx.size == 1:
                res[r, c] = idx[0]
    return res


# =========================================================================== #
# ONNX builder                                                                #
# =========================================================================== #
def _c(name, arr, dtype=FLOAT):
    arr = np.asarray(arr)
    return oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist())


def _shiftkernel(di, dj):
    """3x3 single-tap kernel so Conv(L,W)[r,c] = L[r-di, c-dj]."""
    k = np.zeros((1, 1, 3, 3), np.float32)
    # cross-correlation: out[r,c]=sum w[i,j] in[r+i-1,c+j-1]; want in[r-di,c-dj]
    k[0, 0, 1 - di, 1 - dj] = 1.0
    return k


def _build():
    nodes, inits = [], []
    n = nodes.append
    def C(name, arr): inits.append(_c(name, arr))

    # 4D matrices [1,1,30,30]
    for nm, M in (("A_le", _A_le), ("A_ge", _A_ge),
                  ("C_le", _C_le), ("C_ge", _C_ge), ("ALL", _ALL)):
        C(nm, M.reshape(1, 1, H, W))
    C("HALF", np.array([0.5], np.float32).reshape(1, 1, 1, 1))
    C("ONE", np.array([1.0], np.float32).reshape(1, 1, 1, 1))
    # channel selectors
    av = np.zeros((1, CHANNELS, 1, 1), np.float32); av[0, 4, 0, 0] = 1.0
    sv = np.zeros((1, CHANNELS, 1, 1), np.float32); sv[0, 0, 0, 0] = 1.0
    C("ADDV", av); C("SUBV", sv)
    w_ch0 = np.zeros((1, CHANNELS, 1, 1), np.float32); w_ch0[0, 0, 0, 0] = 1.0
    C("wch0", w_ch0)
    # shift kernels
    shifts = {"sD": (1, 0), "sU": (-1, 0), "sR": (0, 1), "sL": (0, -1),
              "sUR": (-1, 1), "sUL": (-1, -1), "sDR": (1, 1), "sDL": (1, -1)}
    for nm, (di, dj) in shifts.items():
        C("k_" + nm, _shiftkernel(di, dj))

    def conv(inp, kern, out):
        n(oh.make_node("Conv", [inp, kern], [out],
                       kernel_shape=[3, 3], pads=[1, 1, 1, 1]))

    def g0(inp, out):
        n(oh.make_node("Greater", [inp, "HALF"], [out + "_b"]))
        n(oh.make_node("Cast", [out + "_b"], [out], to=FLOAT))

    # gridmask, bg, wall
    n(oh.make_node("ReduceSum", ["input"], ["gridmask"], axes=[1], keepdims=1))
    n(oh.make_node("Conv", ["input", "wch0"], ["bg"], kernel_shape=[1, 1],
                   pads=[0, 0, 0, 0]))
    n(oh.make_node("Sub", ["gridmask", "bg"], ["wall"]))
    # row/col has-wall
    n(oh.make_node("MatMul", ["wall", "ALL"], ["rowHasCnt"]))
    g0("rowHasCnt", "rowHas")
    n(oh.make_node("MatMul", ["ALL", "wall"], ["colHasCnt"]))
    g0("colHasCnt", "colHas")
    # row in-bbox
    n(oh.make_node("MatMul", ["A_le", "rowHas"], ["rowAAc"])); g0("rowAAc", "rowAA")
    n(oh.make_node("MatMul", ["A_ge", "rowHas"], ["rowABc"])); g0("rowABc", "rowAB")
    n(oh.make_node("Mul", ["rowAA", "rowAB"], ["rowIn"]))
    n(oh.make_node("MatMul", ["colHas", "C_le"], ["colAAc"])); g0("colAAc", "colAA")
    n(oh.make_node("MatMul", ["colHas", "C_ge"], ["colABc"])); g0("colABc", "colAB")
    n(oh.make_node("Mul", ["colAA", "colAB"], ["colIn"]))
    # borders
    conv("rowIn", "k_sD", "rowInPrev")  # rowIn[r-1]
    conv("rowIn", "k_sU", "rowInNext")
    conv("colIn", "k_sR", "colInPrev")
    conv("colIn", "k_sL", "colInNext")
    n(oh.make_node("Mul", ["rowIn", "rowInPrev"], ["tA"]))
    n(oh.make_node("Sub", ["rowIn", "tA"], ["isTop"]))
    n(oh.make_node("Mul", ["rowIn", "rowInNext"], ["tB"]))
    n(oh.make_node("Sub", ["rowIn", "tB"], ["isBot"]))
    n(oh.make_node("Mul", ["colIn", "colInPrev"], ["tC"]))
    n(oh.make_node("Sub", ["colIn", "tC"], ["isLeft"]))
    n(oh.make_node("Mul", ["colIn", "colInNext"], ["tD"]))
    n(oh.make_node("Sub", ["colIn", "tD"], ["isRight"]))
    n(oh.make_node("Mul", ["rowIn", "colIn"], ["inBbox"]))
    n(oh.make_node("Add", ["isTop", "isBot"], ["bs1"]))
    n(oh.make_node("Add", ["isLeft", "isRight"], ["bs2"]))
    n(oh.make_node("Add", ["bs1", "bs2"], ["bs"]))
    g0("bs", "bsel")
    n(oh.make_node("Mul", ["inBbox", "bsel"], ["border"]))
    n(oh.make_node("Mul", ["border", "bg"], ["gap"]))
    # strict interior
    n(oh.make_node("Sub", ["ONE", "isTop"], ["nT"]))
    n(oh.make_node("Sub", ["ONE", "isBot"], ["nB"]))
    n(oh.make_node("Sub", ["ONE", "isLeft"], ["nL"]))
    n(oh.make_node("Sub", ["ONE", "isRight"], ["nR"]))
    n(oh.make_node("Mul", ["rowIn", "nT"], ["rs0"]))
    n(oh.make_node("Mul", ["rs0", "nB"], ["rowStrict"]))
    n(oh.make_node("Mul", ["colIn", "nL"], ["cs0"]))
    n(oh.make_node("Mul", ["cs0", "nR"], ["colStrict"]))
    n(oh.make_node("Mul", ["rowStrict", "colStrict"], ["strict0"]))
    n(oh.make_node("Mul", ["strict0", "bg"], ["interior"]))
    # wall neighbours
    conv("wall", "k_sD", "wallAbove")
    conv("wall", "k_sU", "wallBelow")
    conv("wall", "k_sR", "wallLeft")
    conv("wall", "k_sL", "wallRight")
    n(oh.make_node("Mul", ["gap", "isRight"], ["rightGap"]))
    n(oh.make_node("Mul", ["gap", "isLeft"], ["leftGap"]))
    n(oh.make_node("Mul", ["gap", "isTop"], ["topGap"]))
    n(oh.make_node("Mul", ["gap", "isBot"], ["botGap"]))

    def seed(out, terms):
        # terms: list of (a,b) -> sum a*b then >0.5
        accs = []
        for k, (a, b) in enumerate(terms):
            n(oh.make_node("Mul", [a, b], [f"{out}_m{k}"])); accs.append(f"{out}_m{k}")
        n(oh.make_node("Add", accs, [f"{out}_s"]))
        g0(f"{out}_s", out)
    seed("seedUR", [("rightGap", "wallAbove"), ("topGap", "wallRight")])
    seed("seedUL", [("leftGap", "wallAbove"), ("topGap", "wallLeft")])
    seed("seedDR", [("rightGap", "wallBelow"), ("botGap", "wallRight")])
    seed("seedDL", [("leftGap", "wallBelow"), ("botGap", "wallLeft")])

    # propagation
    def propagate(seedname, kern, tag):
        prev = seedname
        for i in range(NPROP):
            sh = f"{tag}_sh{i}"; mk = f"{tag}_mk{i}"; nx = f"{tag}_L{i}"
            conv(prev, kern, sh)
            n(oh.make_node("Mul", [sh, "bg"], [mk]))
            n(oh.make_node("Max", [prev, mk], [nx]))
            prev = nx
        return prev

    beamfields = []
    for kern, tag in (("k_sR", "br"), ("k_sL", "bl"), ("k_sU", "bu"), ("k_sD", "bd")):
        beamfields.append(propagate("interior", kern, tag))
    diagfields = []
    for sd, kern, tag in (("seedUR", "k_sUR", "dur"), ("seedUL", "k_sUL", "dul"),
                          ("seedDR", "k_sDR", "ddr"), ("seedDL", "k_sDL", "ddl")):
        diagfields.append(propagate(sd, kern, tag))

    allf = beamfields + diagfields
    cur = allf[0]
    for i, f in enumerate(allf[1:]):
        nxt = f"lit_acc{i}"
        n(oh.make_node("Max", [cur, f], [nxt])); cur = nxt
    n(oh.make_node("Identity", [cur], ["lit"]))

    # write colour 4
    n(oh.make_node("Mul", ["lit", "ADDV"], ["add4"]))
    n(oh.make_node("Mul", ["lit", "SUBV"], ["sub0"]))
    n(oh.make_node("Add", ["input", "add4"], ["o1"]))
    n(oh.make_node("Sub", ["o1", "sub0"], ["output"]))
    return _model(nodes, inits)


# =========================================================================== #
# detection                                                                   #
# =========================================================================== #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0:
                continue
            if max(a.shape) > H or max(b.shape) > W:
                continue
            out.append((a, b))
    return out


def _cand_268(prs):
    # quick gate: same shape, only 0 -> 4 changes, and 4 is introduced
    introduces4 = False
    for a, b in prs:
        if a.shape != b.shape:
            return []
        d = a != b
        if d.any():
            if (a[d] != 0).any() or (b[d] != 4).any():
                return []
            introduces4 = True
    if not introduces4:
        return []
    try:
        if not all(np.array_equal(_simulate(a), b) for a, b in prs):
            return []
        model = _build()
    except Exception:
        return []
    return [("crk9_3_slitlight", model)]


def candidates(ex):
    try:
        prs = _pairs(ex)
    except Exception:
        return []
    if len(prs) < 2:
        return []
    out = []
    try:
        out += _cand_268(prs)
    except Exception:
        pass
    try:
        out += _cand_184(prs)
    except Exception:
        pass
    try:
        out += _cand_90(prs)
    except Exception:
        pass
    try:
        out += _cand_134(prs)
    except Exception:
        pass
    return out


# =========================================================================== #
# Sub-family: block-grid colour reading (task 184)                            #
# --------------------------------------------------------------------------- #
# The grid is partitioned by single all-zero separator rows/columns into a    #
# K x L grid of monochrome noise blocks.  Output is the K x L grid whose cell #
# (i,j) is the colour of block (i,j).  Implemented with data-dependent        #
# stripe-index one-hot ASSIGNMENT matrices RA[i,r], CA[j,c] (computed via      #
# triangular MatMul prefix sums + band selection) and                          #
#   present[ch,i,j] = (RA @ input[ch] @ CA^T)  ;  output = present * CHMASK.   #
# =========================================================================== #
_A_lt_full = np.zeros((H, W), np.float32)   # [i,j]=1 if j<i (row prefix, strict)
_B_lt_full = np.zeros((H, W), np.float32)   # [j,c]=1 if j<c (col prefix, strict)
for _i in range(H):
    for _j in range(W):
        if _j < _i:
            _A_lt_full[_i, _j] = 1.0
for _jj in range(H):
    for _cc in range(W):
        if _jj < _cc:
            _B_lt_full[_jj, _cc] = 1.0


def _simulate184(g):
    o, h, w = _onehot(g)
    gridmask = o.sum(0); bg = o[0]; wall = gridmask - bg
    rowContent = wall.sum(1); rowInGrid = _g0(gridmask.sum(1))
    zerorow = rowInGrid * (1 - _g0(rowContent)); blockrow = _g0(rowContent)
    stripeRow = _A_lt_full @ zerorow
    colContent = wall.sum(0); colInGrid = _g0(gridmask.sum(0))
    zerocol = colInGrid * (1 - _g0(colContent)); blockcol = _g0(colContent)
    stripeCol = zerocol @ _B_lt_full
    IDX = np.arange(H).reshape(H, 1)
    diffR = stripeRow.reshape(1, H) - IDX
    RA = _g0((diffR > -0.5).astype(np.float32) * (diffR < 0.5).astype(np.float32)) * blockrow.reshape(1, H)
    diffC = stripeCol.reshape(1, H) - IDX
    CA = _g0((diffC > -0.5).astype(np.float32) * (diffC < 0.5).astype(np.float32)) * blockcol.reshape(1, H)
    CAT = CA.T
    out = np.zeros((CHANNELS, H, W), np.float32)
    for ch in range(1, CHANNELS):
        out[ch] = (RA @ o[ch] @ CAT > 0.5).astype(np.float32)
    if blockrow.sum() < 0.5 or blockcol.sum() < 0.5:
        return None
    K = int(stripeRow[blockrow > 0.5].max()) + 1
    L = int(stripeCol[blockcol > 0.5].max()) + 1
    if K > H or L > W:
        return None
    pos = out > 0
    res = np.full((K, L), -99, int)
    for i in range(K):
        for j in range(L):
            idx = np.where(pos[:, i, j])[0]
            if idx.size == 1:
                res[i, j] = idx[0]
    return res


def _build184():
    wch0 = np.zeros((1, CHANNELS, 1, 1), np.float32); wch0[0, 0, 0, 0] = 1.0
    CHM = np.ones((1, CHANNELS, 1, 1), np.float32); CHM[0, 0, 0, 0] = 0.0
    IDX = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)
    inits = [_c("A_lt2", _A_lt_full.reshape(1, 1, H, W)),
             _c("B_lt2", _B_lt_full.reshape(1, 1, H, W)),
             _c("wch0b", wch0), _c("CHM", CHM), _c("IDX", IDX),
             _c("ONEb", np.array([1.0], np.float32).reshape(1, 1, 1, 1)),
             _c("HALFb", np.array([0.5], np.float32).reshape(1, 1, 1, 1)),
             _c("NHALFb", np.array([-0.5], np.float32).reshape(1, 1, 1, 1))]
    nodes = []; n = nodes.append

    def g0(inp, out, thr="HALFb"):
        n(oh.make_node("Greater", [inp, thr], [out + "_b"]))
        n(oh.make_node("Cast", [out + "_b"], [out], to=FLOAT))

    n(oh.make_node("ReduceSum", ["input"], ["gm"], axes=[1], keepdims=1))
    n(oh.make_node("Conv", ["input", "wch0b"], ["bgb"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    n(oh.make_node("Sub", ["gm", "bgb"], ["wll"]))
    n(oh.make_node("ReduceSum", ["wll"], ["rowContent"], axes=[3], keepdims=1))
    n(oh.make_node("ReduceSum", ["gm"], ["rowGrid"], axes=[3], keepdims=1))
    g0("rowGrid", "rowInGrid"); g0("rowContent", "blockrow")
    n(oh.make_node("Sub", ["ONEb", "blockrow"], ["nbr"]))
    n(oh.make_node("Mul", ["rowInGrid", "nbr"], ["zerorow"]))
    n(oh.make_node("MatMul", ["A_lt2", "zerorow"], ["stripeRow"]))
    n(oh.make_node("ReduceSum", ["wll"], ["colContent"], axes=[2], keepdims=1))
    n(oh.make_node("ReduceSum", ["gm"], ["colGrid"], axes=[2], keepdims=1))
    g0("colGrid", "colInGrid"); g0("colContent", "blockcol")
    n(oh.make_node("Sub", ["ONEb", "blockcol"], ["nbc"]))
    n(oh.make_node("Mul", ["colInGrid", "nbc"], ["zerocol"]))
    n(oh.make_node("MatMul", ["zerocol", "B_lt2"], ["stripeCol"]))
    n(oh.make_node("Transpose", ["stripeRow"], ["stripeRowT"], perm=[0, 1, 3, 2]))
    n(oh.make_node("Sub", ["stripeRowT", "IDX"], ["diffR"]))
    g0("diffR", "gA", "NHALFb")
    n(oh.make_node("Less", ["diffR", "HALFb"], ["gB_b"]))
    n(oh.make_node("Cast", ["gB_b"], ["gB"], to=FLOAT))
    n(oh.make_node("Mul", ["gA", "gB"], ["RA0"]))
    n(oh.make_node("Transpose", ["blockrow"], ["blockrowT"], perm=[0, 1, 3, 2]))
    n(oh.make_node("Mul", ["RA0", "blockrowT"], ["RA"]))
    n(oh.make_node("Sub", ["stripeCol", "IDX"], ["diffC"]))
    g0("diffC", "hA", "NHALFb")
    n(oh.make_node("Less", ["diffC", "HALFb"], ["hB_b"]))
    n(oh.make_node("Cast", ["hB_b"], ["hB"], to=FLOAT))
    n(oh.make_node("Mul", ["hA", "hB"], ["CA0"]))
    n(oh.make_node("Mul", ["CA0", "blockcol"], ["CA"]))
    n(oh.make_node("MatMul", ["RA", "input"], ["tmp"]))
    n(oh.make_node("Transpose", ["CA"], ["CAT"], perm=[0, 1, 3, 2]))
    n(oh.make_node("MatMul", ["tmp", "CAT"], ["present"]))
    n(oh.make_node("Mul", ["present", "CHM"], ["output"]))
    return _model(nodes, inits)


def _cand_184(prs):
    # gate: output strictly smaller than input on at least one axis, never larger,
    # output uses only non-zero colours, inputs contain all-zero separator lines.
    saw_shrink = False
    for a, b in prs:
        if b.shape[0] > a.shape[0] or b.shape[1] > a.shape[1]:
            return []
        if b.shape != a.shape:
            saw_shrink = True
        if (b == 0).any():
            return []
    if not saw_shrink:
        return []
    try:
        for a, b in prs:
            r = _simulate184(a)
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                return []
        model = _build184()
    except Exception:
        return []
    return [("crk9_3_blockgrid", model)]


# =========================================================================== #
# Sub-family: maximal all-zero (>=2x2) rectangle -> fill colour 6 (task 90)   #
# --------------------------------------------------------------------------- #
# A noise grid (0/1) hides one solid rectangular block of 0s that is the       #
# largest axis-aligned all-zero rectangle whose height AND width are both >=2. #
# That block is filled with colour 6.  Implemented by enumerating rectangle    #
# sizes (h in 2..6, w in 2..12): for each size a box-sum Conv finds all-zero   #
# windows; the maximum present area is reduced over sizes; the winning size's  #
# windows are stamped back to full rectangles with a second (look-back) Conv.  #
# =========================================================================== #
_HS90 = range(2, 7)
_WS90 = range(2, 13)


def _integral(x):
    return np.pad(np.cumsum(np.cumsum(x, 0), 1), ((1, 0), (1, 0)))


def _boxsum_tl(bg, h, w):
    I = _integral(bg)
    r = np.arange(H); c = np.arange(W)
    rend = np.minimum(r + h, H); cend = np.minimum(c + w, W)
    A = I[np.ix_(rend, cend)]; B = I[np.ix_(r, cend)]
    Cc = I[np.ix_(rend, c)]; D = I[np.ix_(r, c)]
    return (A - B - Cc + D).astype(np.float32)


def _cover90(v, h, w):
    I = _integral(v)
    r = np.arange(H); c = np.arange(W)
    rs = np.maximum(r - h + 1, 0); cs = np.maximum(c - w + 1, 0)
    A = I[np.ix_(r + 1, c + 1)]; B = I[np.ix_(rs, c + 1)]
    Cc = I[np.ix_(r + 1, cs)]; D = I[np.ix_(rs, cs)]
    return (A - B - Cc + D).astype(np.float32)


def _simulate90(g):
    o, h, w = _onehot(g); bg = o[0]
    valids = {}; best = 0.0
    for hh in _HS90:
        for ww in _WS90:
            v = (_boxsum_tl(bg, hh, ww) > hh * ww - 0.5).astype(np.float32)
            valids[(hh, ww)] = v
            if v.max() > 0.5:
                best = max(best, float(hh * ww))
    paint = np.zeros((H, W), np.float32)
    for (hh, ww), v in valids.items():
        if hh * ww > best - 0.5 and v.max() > 0.5:
            paint = np.maximum(paint, (_cover90(v, hh, ww) > 0.5).astype(np.float32))
    out = o.copy(); out[6] = out[6] + paint; out[0] = out[0] - paint
    pos = out > 0; res = np.full((h, w), -99, int)
    for r in range(h):
        for c in range(w):
            idx = np.where(pos[:, r, c])[0]
            if idx.size == 1:
                res[r, c] = idx[0]
    return res


def _build90():
    inits = []; nodes = []; n = nodes.append
    def C(nm, a): inits.append(_c(nm, np.asarray(a, np.float32)))
    wch0 = np.zeros((1, CHANNELS, 1, 1), np.float32); wch0[0, 0, 0, 0] = 1.0; C("wch0c", wch0)
    C("HALFc", np.array([0.5]).reshape(1, 1, 1, 1))
    av6 = np.zeros((1, CHANNELS, 1, 1), np.float32); av6[0, 6, 0, 0] = 1.0; C("ADDV6", av6)
    sv0 = np.zeros((1, CHANNELS, 1, 1), np.float32); sv0[0, 0, 0, 0] = 1.0; C("SUBV0", sv0)
    n(oh.make_node("Conv", ["input", "wch0c"], ["bgc"], kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
    pairs = [(h, w) for h in _HS90 for w in _WS90]
    for h, w in pairs:
        t = f"{h}_{w}"
        C(f"k_{t}", np.ones((1, 1, h, w), np.float32))
        C(f"thr_{t}", np.array([h * w - 0.5]).reshape(1, 1, 1, 1))
        C(f"area_{t}", np.array([float(h * w)]).reshape(1, 1, 1, 1))
    areaif = []
    for h, w in pairs:
        t = f"{h}_{w}"
        n(oh.make_node("Conv", ["bgc", f"k_{t}"], [f"box_{t}"], kernel_shape=[h, w], pads=[0, 0, h - 1, w - 1]))
        n(oh.make_node("Greater", [f"box_{t}", f"thr_{t}"], [f"vb_{t}"]))
        n(oh.make_node("Cast", [f"vb_{t}"], [f"valid_{t}"], to=FLOAT))
        n(oh.make_node("ReduceMax", [f"valid_{t}"], [f"vmax_{t}"], axes=[2, 3], keepdims=1))
        n(oh.make_node("Mul", [f"vmax_{t}", f"area_{t}"], [f"aif_{t}"]))
        areaif.append(f"aif_{t}")
    cur = areaif[0]
    for i, a in enumerate(areaif[1:]):
        nm = f"best{i}"; n(oh.make_node("Max", [cur, a], [nm])); cur = nm
    n(oh.make_node("Sub", [cur, "HALFc"], ["bestMinus"]))
    covers = []
    for h, w in pairs:
        t = f"{h}_{w}"
        n(oh.make_node("Greater", [f"aif_{t}", "bestMinus"], [f"iwb_{t}"]))
        n(oh.make_node("Cast", [f"iwb_{t}"], [f"iw_{t}"], to=FLOAT))
        n(oh.make_node("Mul", [f"valid_{t}", f"iw_{t}"], [f"win_{t}"]))
        n(oh.make_node("Conv", [f"win_{t}", f"k_{t}"], [f"cov_{t}"], kernel_shape=[h, w], pads=[h - 1, w - 1, 0, 0]))
        covers.append(f"cov_{t}")
    cur = covers[0]
    for i, cv in enumerate(covers[1:]):
        nm = f"covacc{i}"; n(oh.make_node("Add", [cur, cv], [nm])); cur = nm
    n(oh.make_node("Greater", [cur, "HALFc"], ["paint_b"]))
    n(oh.make_node("Cast", ["paint_b"], ["paint"], to=FLOAT))
    n(oh.make_node("Mul", ["paint", "ADDV6"], ["add6"]))
    n(oh.make_node("Mul", ["paint", "SUBV0"], ["sub0_90"]))
    n(oh.make_node("Add", ["input", "add6"], ["o1_90"]))
    n(oh.make_node("Sub", ["o1_90", "sub0_90"], ["output"]))
    return _model(nodes, inits)


def _cand_90(prs):
    saw = False
    for a, b in prs:
        if a.shape != b.shape:
            return []
        d = a != b
        if d.any():
            if (a[d] != 0).any() or (b[d] != 6).any():
                return []
            saw = True
    if not saw:
        return []
    try:
        for a, b in prs:
            r = _simulate90(a)
            if r.shape != b.shape or not np.array_equal(r, b):
                return []
        model = _build90()
    except Exception:
        return []
    return [("crk9_3_maxrect6", model)]


# =========================================================================== #
# Sub-family: big-shape silhouette downsampled to 3x3 in noise colour (134)   #
# --------------------------------------------------------------------------- #
# Two non-zero colours: one forms a big connected blob, the other is scattered #
# "noise".  Output is the big shape's bounding box split into a 3x3 grid of    #
# bands; a band is ON (>=50% covered by the big colour) and drawn in the noise #
# colour, OFF bands are 0.  The big colour is the one with the most same-colour #
# 4-adjacency; bbox + band membership use computed scalars + assignment        #
# matrices; majority is 2*count >= area.                                       #
# =========================================================================== #
def _colpref(kind):
    M = np.zeros((H, W), np.float32)
    for j in range(H):
        for c in range(W):
            if (kind == 'le' and j <= c) or (kind == 'ge' and j >= c):
                M[j, c] = 1.0
    return M


def _shiftmat(d):
    M = np.zeros((H, W), np.float32)
    for i in range(H):
        j = i - d
        if 0 <= j < W:
            M[i, j] = 1.0
    return M


def _colshift(d):
    M = np.zeros((H, W), np.float32)
    for c in range(W):
        j = c - d
        if 0 <= j < H:
            M[j, c] = 1.0
    return M


def _adjcount(a, col):
    m = (a == col).astype(int)
    s = np.zeros_like(m)
    s[1:, :] += m[:-1, :]; s[:-1, :] += m[1:, :]
    s[:, 1:] += m[:, :-1]; s[:, :-1] += m[:, 1:]
    return int((m * (s > 0)).sum())


def _simulate134(g):
    a = np.asarray(g, int); h, w = a.shape
    cols = sorted(set(a.flatten().tolist()) - {0})
    if len(cols) != 2:
        return None
    big = max(cols, key=lambda c: _adjcount(a, c))
    noise = [c for c in cols if c != big][0]
    m = (a == big)
    if not m.any():
        return None
    ys, xs = np.where(m)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    Hb = r1 - r0 + 1; Wb = c1 - c0 + 1
    out = np.zeros((3, 3), int)
    cnt = np.zeros((3, 3)); area = np.zeros((3, 3))
    for r in range(r0, r1 + 1):
        bi = (3 * (r - r0)) // Hb
        for c in range(c0, c1 + 1):
            bj = (3 * (c - c0)) // Wb
            area[bi, bj] += 1
            if m[r, c]:
                cnt[bi, bj] += 1
    for i in range(3):
        for j in range(3):
            out[i, j] = noise if (area[i, j] > 0 and 2 * cnt[i, j] >= area[i, j]) else 0
    return out


def _build134():
    inits = []; nodes = []; n = nodes.append
    def C(nm, a): inits.append(_c(nm, np.asarray(a, np.float32)))
    plusK = np.zeros((CHANNELS, 1, 3, 3), np.float32)
    for ch in range(CHANNELS):
        plusK[ch, 0] = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    C("plusK", plusK)
    ch0neg = np.zeros((1, CHANNELS, 1, 1), np.float32); ch0neg[0, 0, 0, 0] = -1e9; C("CH0NEG", ch0neg)
    chnz = np.ones((1, CHANNELS, 1, 1), np.float32); chnz[0, 0, 0, 0] = 0.0; C("CHNZ", chnz)
    ch0vec = np.zeros((1, CHANNELS, 1, 1), np.float32); ch0vec[0, 0, 0, 0] = 1.0; C("CH0VEC", ch0vec)
    C("A_le2", _A_le.reshape(1, 1, H, W)); C("A_ge2", _A_ge.reshape(1, 1, H, W))
    C("Cc_le", _colpref('le').reshape(1, 1, H, W)); C("Cc_ge", _colpref('ge').reshape(1, 1, H, W))
    C("SDOWN", _shiftmat(1).reshape(1, 1, H, W)); C("SUP", _shiftmat(-1).reshape(1, 1, H, W))
    C("SRIGHT", _colshift(1).reshape(1, 1, H, W)); C("SLEFT", _colshift(-1).reshape(1, 1, H, W))
    C("ROWIDX", np.arange(H, dtype=np.float32).reshape(1, 1, H, 1))
    C("COLIDX", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W))
    C("IDX2", np.arange(H, dtype=np.float32).reshape(1, 1, H, 1))
    C("ONE2", np.array([1.0]).reshape(1, 1, 1, 1)); C("HALF2", np.array([0.5]).reshape(1, 1, 1, 1))
    C("NHALF2", np.array([-0.5]).reshape(1, 1, 1, 1)); C("THREE", np.array([3.0]).reshape(1, 1, 1, 1))
    C("TWO", np.array([2.0]).reshape(1, 1, 1, 1))
    g3 = np.zeros((1, 1, H, W), np.float32); g3[0, 0, :3, :3] = 1.0; C("GRID3", g3)

    def g0(inp, out, thr="HALF2"):
        n(oh.make_node("Greater", [inp, thr], [out + "_b"]))
        n(oh.make_node("Cast", [out + "_b"], [out], to=FLOAT))

    n(oh.make_node("Conv", ["input", "plusK"], ["nbr"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=CHANNELS))
    g0("nbr", "nbrPos")
    n(oh.make_node("Mul", ["input", "nbrPos"], ["hasNbr"]))
    n(oh.make_node("ReduceSum", ["hasNbr"], ["adj"], axes=[2, 3], keepdims=1))
    n(oh.make_node("Add", ["adj", "CH0NEG"], ["adjM"]))
    n(oh.make_node("ReduceMax", ["adjM"], ["maxadj"], axes=[1], keepdims=1))
    n(oh.make_node("Sub", ["maxadj", "HALF2"], ["maxadjm"]))
    g0("adjM", "gBig", "maxadjm")
    n(oh.make_node("ReduceSum", ["input"], ["chsum"], axes=[2, 3], keepdims=1))
    g0("chsum", "present")
    n(oh.make_node("Mul", ["present", "CHNZ"], ["nzcol"]))
    n(oh.make_node("Sub", ["ONE2", "gBig"], ["notbig"]))
    n(oh.make_node("Mul", ["nzcol", "notbig"], ["gNoise"]))
    n(oh.make_node("Mul", ["input", "gBig"], ["bigsel"]))
    n(oh.make_node("ReduceSum", ["bigsel"], ["bigmask"], axes=[1], keepdims=1))
    n(oh.make_node("ReduceSum", ["bigmask"], ["rowHas"], axes=[3], keepdims=1))
    g0("rowHas", "rowHasB")
    n(oh.make_node("MatMul", ["A_le2", "rowHasB"], ["anyAbv"])); g0("anyAbv", "anyAbvB")
    n(oh.make_node("MatMul", ["A_ge2", "rowHasB"], ["anyBlw"])); g0("anyBlw", "anyBlwB")
    n(oh.make_node("Mul", ["anyAbvB", "anyBlwB"], ["rowIn"]))
    n(oh.make_node("MatMul", ["SDOWN", "rowIn"], ["rowInPrev"]))
    n(oh.make_node("MatMul", ["SUP", "rowIn"], ["rowInNext"]))
    n(oh.make_node("Mul", ["rowIn", "rowInPrev"], ["tA"])); n(oh.make_node("Sub", ["rowIn", "tA"], ["isTop"]))
    n(oh.make_node("Mul", ["rowIn", "rowInNext"], ["tB"])); n(oh.make_node("Sub", ["rowIn", "tB"], ["isBot"]))
    n(oh.make_node("Mul", ["ROWIDX", "isTop"], ["r0m"])); n(oh.make_node("ReduceSum", ["r0m"], ["r0"], axes=[2], keepdims=1))
    n(oh.make_node("Mul", ["ROWIDX", "isBot"], ["r1m"])); n(oh.make_node("ReduceSum", ["r1m"], ["r1"], axes=[2], keepdims=1))
    n(oh.make_node("Sub", ["r1", "r0"], ["hbm1"])); n(oh.make_node("Add", ["hbm1", "ONE2"], ["Hb"]))
    n(oh.make_node("ReduceSum", ["bigmask"], ["colHas"], axes=[2], keepdims=1))
    g0("colHas", "colHasB")
    n(oh.make_node("MatMul", ["colHasB", "Cc_le"], ["anyLft"])); g0("anyLft", "anyLftB")
    n(oh.make_node("MatMul", ["colHasB", "Cc_ge"], ["anyRgt"])); g0("anyRgt", "anyRgtB")
    n(oh.make_node("Mul", ["anyLftB", "anyRgtB"], ["colIn"]))
    n(oh.make_node("MatMul", ["colIn", "SRIGHT"], ["colInPrev"]))
    n(oh.make_node("MatMul", ["colIn", "SLEFT"], ["colInNext"]))
    n(oh.make_node("Mul", ["colIn", "colInPrev"], ["uA"])); n(oh.make_node("Sub", ["colIn", "uA"], ["isLeft"]))
    n(oh.make_node("Mul", ["colIn", "colInNext"], ["uB"])); n(oh.make_node("Sub", ["colIn", "uB"], ["isRight"]))
    n(oh.make_node("Mul", ["COLIDX", "isLeft"], ["c0m"])); n(oh.make_node("ReduceSum", ["c0m"], ["c0"], axes=[3], keepdims=1))
    n(oh.make_node("Mul", ["COLIDX", "isRight"], ["c1m"])); n(oh.make_node("ReduceSum", ["c1m"], ["c1"], axes=[3], keepdims=1))
    n(oh.make_node("Sub", ["c1", "c0"], ["wbm1"])); n(oh.make_node("Add", ["wbm1", "ONE2"], ["Wb"]))
    n(oh.make_node("Sub", ["ROWIDX", "r0"], ["relr0"])); n(oh.make_node("Mul", ["relr0", "THREE"], ["relr"]))
    n(oh.make_node("Sub", ["relr", "Hb"], ["d1r"])); g0("d1r", "b1r", "NHALF2")
    n(oh.make_node("Mul", ["Hb", "TWO"], ["Hb2"])); n(oh.make_node("Sub", ["relr", "Hb2"], ["d2r"])); g0("d2r", "b2r", "NHALF2")
    n(oh.make_node("Add", ["b1r", "b2r"], ["bandRow"]))
    n(oh.make_node("Transpose", ["bandRow"], ["bandRowT"], perm=[0, 1, 3, 2]))
    n(oh.make_node("Sub", ["bandRowT", "IDX2"], ["diffR2"]))
    g0("diffR2", "gAr", "NHALF2")
    n(oh.make_node("Less", ["diffR2", "HALF2"], ["gBr_b"])); n(oh.make_node("Cast", ["gBr_b"], ["gBr"], to=FLOAT))
    n(oh.make_node("Mul", ["gAr", "gBr"], ["RA0c"]))
    n(oh.make_node("Transpose", ["rowIn"], ["rowInT"], perm=[0, 1, 3, 2]))
    n(oh.make_node("Mul", ["RA0c", "rowInT"], ["RB"]))
    n(oh.make_node("Sub", ["COLIDX", "c0"], ["relc0"])); n(oh.make_node("Mul", ["relc0", "THREE"], ["relc"]))
    n(oh.make_node("Sub", ["relc", "Wb"], ["d1c"])); g0("d1c", "b1c", "NHALF2")
    n(oh.make_node("Mul", ["Wb", "TWO"], ["Wb2"])); n(oh.make_node("Sub", ["relc", "Wb2"], ["d2c"])); g0("d2c", "b2c", "NHALF2")
    n(oh.make_node("Add", ["b1c", "b2c"], ["bandCol"]))
    n(oh.make_node("Sub", ["bandCol", "IDX2"], ["diffC2"]))
    g0("diffC2", "gAc", "NHALF2")
    n(oh.make_node("Less", ["diffC2", "HALF2"], ["gBc_b"])); n(oh.make_node("Cast", ["gBc_b"], ["gBc"], to=FLOAT))
    n(oh.make_node("Mul", ["gAc", "gBc"], ["CB0c"]))
    n(oh.make_node("Mul", ["CB0c", "colIn"], ["CB"]))
    n(oh.make_node("MatMul", ["RB", "bigmask"], ["rc"]))
    n(oh.make_node("Transpose", ["CB"], ["CBT"], perm=[0, 1, 3, 2]))
    n(oh.make_node("MatMul", ["rc", "CBT"], ["count"]))
    n(oh.make_node("Mul", ["rowIn", "colIn"], ["inbb"]))
    n(oh.make_node("MatMul", ["RB", "inbb"], ["ra"])); n(oh.make_node("MatMul", ["ra", "CBT"], ["area"]))
    n(oh.make_node("Mul", ["count", "TWO"], ["count2"])); n(oh.make_node("Sub", ["count2", "area"], ["dfill"]))
    g0("dfill", "f1", "NHALF2"); g0("area", "f2")
    n(oh.make_node("Mul", ["f1", "f2"], ["bandfill"]))
    n(oh.make_node("Mul", ["bandfill", "gNoise"], ["noisePart"]))
    n(oh.make_node("Sub", ["ONE2", "bandfill"], ["nofill"]))
    n(oh.make_node("Mul", ["GRID3", "nofill"], ["ch0fill"]))
    n(oh.make_node("Mul", ["ch0fill", "CH0VEC"], ["zeroPart"]))
    n(oh.make_node("Add", ["noisePart", "zeroPart"], ["output"]))
    return _model(nodes, inits)


def _cand_134(prs):
    for a, b in prs:
        if b.shape != (3, 3):
            return []
        if len(set(a.flatten().tolist()) - {0}) != 2:
            return []
    try:
        for a, b in prs:
            r = _simulate134(a)
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                return []
        model = _build134()
    except Exception:
        return []
    return [("crk9_3_silhouette3x3", model)]
