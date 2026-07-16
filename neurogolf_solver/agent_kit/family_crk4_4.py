"""family_crk4_4 -- hard-tail slice IDX=4 of NeuroGolf 2026 unsolved tasks.

Solvers contributed here (all opset-10, static [1,10,30,30], origin-anchored):

  * task270  "pull aligned markers to a plus around each centre"
      Centres of colour 1 (marker 7) and colour 2 (marker 3).  Far markers sit in
      the same row/column as their centre (N/S/E/W, some directions may be absent).
      Output keeps the centres, draws ONE marker-coloured cell immediately adjacent
      to the centre in every direction that had a marker, and erases the far markers.
      Built with MatMul shift matrices (neighbour-of-centre) and triangular MatMuls
      (any marker in that half-row/half-col), batched over the two colour pairs.

  * task381  "horizontal fill-between colour-2 walls with colour 9"
      A background cell becomes 9 iff there is a 2 to its left AND a 2 to its right
      in the same row.  Prefix-OR along the width via two triangular MatMuls.

Every candidate is gated by a numpy mirror that reproduces the exact ONNX
semantics on the 30x30 padded grid, and is emitted only when it matches all
train+test+arc-gen pairs.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
G = HEIGHT  # 30 (square grid window)


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                          [float(v) for v in np.asarray(vals, np.float32).ravel()]))
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
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _plane(g, ch):
    return g.nd("Slice", ["input", g.i64([ch]), g.i64([ch + 1]), g.i64([1])])


# --------------------------------------------------------------------------- #
# shared index-built matrices (numpy)                                          #
# --------------------------------------------------------------------------- #
_idx = np.arange(G)
_DIFF = _idx[:, None] - _idx[None, :]      # [i,k] = i-k
MAT_A = (_DIFF >= 0).astype(np.float32)    # lower tri incl diag  (i>=k)
MAT_B = (_DIFF <= 0).astype(np.float32)    # upper tri incl diag  (i<=k)
MAT_SP = (_DIFF == 1).astype(np.float32)   # sub-diagonal         (k=i-1)
MAT_SM = (_DIFF == -1).astype(np.float32)  # super-diagonal       (k=i+1)


def _shiftmat(d):
    """Matrix S s.t. MatMul(S, X) shifts X's rows DOWN by d (up if d<0), zero-fill.
    S[i,k] = 1 iff i-k == d.  Also: MatMul(X, S) shifts COLUMNS LEFT by d."""
    return (_DIFF == d).astype(np.float32)


# row index plane RI[i,j] = i  (constant)
_RI = (np.arange(G)[:, None] * np.ones((1, G))).astype(np.float32)


# =========================================================================== #
# task 270                                                                     #
# =========================================================================== #
def build_270():
    g = _G()
    c1 = _plane(g, 1)
    c2 = _plane(g, 2)
    c3 = _plane(g, 3)
    c7 = _plane(g, 7)
    Cc = g.nd("Concat", [c1, c2], axis=1)     # centres: ch0->c1(arm7), ch1->c2(arm3)
    Mk = g.nd("Concat", [c7, c3], axis=1)     # markers matching the centres

    A = g.f([1, 1, G, G], MAT_A)
    B = g.f([1, 1, G, G], MAT_B)
    SP = g.f([1, 1, G, G], MAT_SP)
    SM = g.f([1, 1, G, G], MAT_SM)

    # neighbour-of-centre planes
    Cup = g.nd("MatMul", [SM, Cc])            # value at (r+1,c): centre below -> north arm
    Cdown = g.nd("MatMul", [SP, Cc])          # value at (r-1,c): centre above -> south arm
    Cleft = g.nd("MatMul", [Cc, SP])          # value at (r,c+1): centre right -> west arm
    Cright = g.nd("MatMul", [Cc, SM])         # value at (r,c-1): centre left  -> east arm

    # "any marker in that half-row / half-col" (clip count -> 0/1)
    north = g.nd("Clip", [g.nd("MatMul", [A, Mk])], min=0.0, max=1.0)
    south = g.nd("Clip", [g.nd("MatMul", [B, Mk])], min=0.0, max=1.0)
    west = g.nd("Clip", [g.nd("MatMul", [Mk, B])], min=0.0, max=1.0)
    east = g.nd("Clip", [g.nd("MatMul", [Mk, A])], min=0.0, max=1.0)

    armN = g.nd("Mul", [Cup, north])
    armS = g.nd("Mul", [Cdown, south])
    armW = g.nd("Mul", [Cleft, west])
    armE = g.nd("Mul", [Cright, east])
    arm = g.nd("Add", [g.nd("Add", [armN, armS]), g.nd("Add", [armW, armE])])
    arm = g.nd("Clip", [arm], min=0.0, max=1.0)          # [1,2,30,30]

    arm7 = g.nd("Slice", [arm, g.i64([0]), g.i64([1]), g.i64([1])])
    arm3 = g.nd("Slice", [arm, g.i64([1]), g.i64([2]), g.i64([1])])

    real = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    out0 = g.nd("Sub", [g.nd("Sub", [g.nd("Sub", [g.nd("Sub", [real, c1]), c2]), arm7]), arm3])
    z = g.nd("Sub", [c1, c1])

    g.nd("Concat", [out0, c1, c2, arm3, z, z, z, arm7, z, z], "output", axis=1)
    return _model(g)


def _ref_270(a):
    h, w = a.shape
    def emb(m):
        z = np.zeros((G, G), np.float32)
        z[:h, :w] = m
        return z
    c1 = emb(a == 1); c2 = emb(a == 2); c3 = emb(a == 3); c7 = emb(a == 7)
    real = emb(np.ones_like(a))
    out0 = real.copy()
    arm3 = np.zeros((G, G), np.float32)
    arm7 = np.zeros((G, G), np.float32)
    for Cc, Mk, dst in ((c1, c7, "7"), (c2, c3, "3")):
        Cup = MAT_SM @ Cc; Cdown = MAT_SP @ Cc
        Cleft = Cc @ MAT_SP; Cright = Cc @ MAT_SM
        north = np.clip(MAT_A @ Mk, 0, 1); south = np.clip(MAT_B @ Mk, 0, 1)
        west = np.clip(Mk @ MAT_B, 0, 1); east = np.clip(Mk @ MAT_A, 0, 1)
        arm = np.clip(Cup * north + Cdown * south + Cleft * west + Cright * east, 0, 1)
        if dst == "7":
            arm7 = arm
        else:
            arm3 = arm
    out0 = real - c1 - c2 - arm7 - arm3
    planes = {0: out0, 1: c1, 2: c2, 3: arm3, 7: arm7}
    out = np.zeros((h, w), int)
    for col, pl in planes.items():
        m = pl[:h, :w] > 0
        out[m] = col
    return out


# =========================================================================== #
# task 381                                                                     #
# =========================================================================== #
def build_381():
    g = _G()
    c0 = _plane(g, 0)
    c2 = _plane(g, 2)
    A = g.f([1, 1, G, G], MAT_A)
    B = g.f([1, 1, G, G], MAT_B)
    hasLeft = g.nd("Clip", [g.nd("MatMul", [c2, B])], min=0.0, max=1.0)   # any 2 at col<=c
    hasRight = g.nd("Clip", [g.nd("MatMul", [c2, A])], min=0.0, max=1.0)  # any 2 at col>=c
    fill = g.nd("Mul", [g.nd("Mul", [hasLeft, hasRight]), c0])            # 0/1
    ch0 = g.nd("Sub", [c0, fill])
    z = g.nd("Sub", [c2, c2])
    g.nd("Concat", [ch0, z, c2, z, z, z, z, z, z, fill], "output", axis=1)
    return _model(g)


def _ref_381(a):
    h, w = a.shape
    def emb(m):
        z = np.zeros((G, G), np.float32)
        z[:h, :w] = m
        return z
    c0 = emb(a == 0); c2 = emb(a == 2)
    hasLeft = np.clip(c2 @ MAT_B, 0, 1); hasRight = np.clip(c2 @ MAT_A, 0, 1)
    fill = hasLeft * hasRight * c0
    ch0 = c0 - fill
    planes = {0: ch0, 2: c2, 9: fill}
    out = np.zeros((h, w), int)
    for col, pl in planes.items():
        m = pl[:h, :w] > 0
        out[m] = col
    return out


# =========================================================================== #
# task 139 : fill the bounding box of every 8-connected colour-4 blob with 7   #
# =========================================================================== #
_K139 = 8


def build_139(K=_K139):
    g = _G()
    c0 = _plane(g, 0)
    c4 = _plane(g, 4)
    SD = g.f([1, 1, G, G], _shiftmat(1))    # MatMul(SD,X) rows down1 / MatMul(X,SD) cols left1
    SU = g.f([1, 1, G, G], _shiftmat(-1))   # rows up1 / cols right1
    RIp1 = g.f([1, 1, G, G], _RI + 1.0)
    Lp31 = g.f([1, 1, G, G], 31.0 - _RI)

    # stacked propagation field [1,2,30,30]: ch0 seeds row+1, ch1 seeds 31-row, masked to 4-cells
    H0 = g.nd("Mul", [c4, RIp1])
    L0 = g.nd("Mul", [c4, Lp31])
    field = g.nd("Concat", [H0, L0], axis=1)
    for _ in range(K):
        vd = g.nd("Max", [field, g.nd("MatMul", [SD, field]), g.nd("MatMul", [SU, field])])
        hd = g.nd("Max", [vd, g.nd("MatMul", [vd, SD]), g.nd("MatMul", [vd, SU])])
        field = g.nd("Mul", [hd, c4])
    HR = g.nd("Slice", [field, g.i64([0]), g.i64([1]), g.i64([1])])   # Rmax+1 at 4-cells
    LR = g.nd("Slice", [field, g.i64([1]), g.i64([2]), g.i64([1])])   # 31-Rmin at 4-cells

    DM = HR
    for s in (1, 2, 4, 8, 16):                     # prefix-max DOWN the column
        DM = g.nd("Max", [DM, g.nd("MatMul", [g.f([1, 1, G, G], _shiftmat(s)), DM])])
    UM = LR
    for s in (1, 2, 4, 8, 16):                     # suffix-max UP the column
        UM = g.nd("Max", [UM, g.nd("MatMul", [g.f([1, 1, G, G], _shiftmat(-s)), UM])])

    RIh = g.f([1, 1, G, G], _RI + 0.5)
    UMt = g.f([1, 1, G, G], 30.5 - _RI)
    term1 = g.nd("Cast", [g.nd("Greater", [DM, RIh])], to=int(F))     # Rmax >= r (4 above)
    term2 = g.nd("Cast", [g.nd("Greater", [UM, UMt])], to=int(F))     # Rmin <= r (4 below)
    covered = g.nd("Max", [term1, term2])
    fill7 = g.nd("Mul", [covered, c0])

    ch0 = g.nd("Sub", [c0, fill7])
    z = g.nd("Sub", [c4, c4])
    g.nd("Concat", [ch0, z, z, z, c4, z, z, fill7, z, z], "output", axis=1)
    return _model(g)


def _ref_139(a, K=_K139):
    h, w = a.shape
    def emb(m):
        z = np.zeros((G, G), np.float32)
        z[:h, :w] = np.asarray(m, np.float32)
        return z
    c4 = emb(a == 4); c0 = emb(a == 0)
    def shr(X, d): return _shiftmat(d) @ X            # rows down d (zero fill)
    def shc(X, d): return X @ _shiftmat(d)            # cols left d (zero fill)
    def vdil(H): return np.maximum.reduce([H, shr(H, 1), shr(H, -1)])
    def hdil(V): return np.maximum.reduce([V, shc(V, 1), shc(V, -1)])
    H = c4 * (_RI + 1.0); L = c4 * (31.0 - _RI)
    for _ in range(K):
        H = hdil(vdil(H)) * c4
        L = hdil(vdil(L)) * c4
    DM = H.copy()
    for s in (1, 2, 4, 8, 16): DM = np.maximum(DM, shr(DM, s))
    UM = L.copy()
    for s in (1, 2, 4, 8, 16): UM = np.maximum(UM, shr(UM, -s))
    term1 = (DM >= _RI + 1.0).astype(np.float32)
    term2 = (UM >= 31.0 - _RI).astype(np.float32)
    fill7 = np.clip(term1 + term2, 0, 1) * c0
    out = np.zeros((h, w), int)
    out[c4[:h, :w] > 0] = 4
    out[fill7[:h, :w] > 0] = 7
    return out


# =========================================================================== #
# task 250 : slide every colour-5 marker until it touches the 2x2 colour-2 box #
#            (project each marker onto the box's perimeter ring -- separable    #
#            row/col remap built as data-dependent select matrices)             #
# =========================================================================== #
_RIcol = (np.arange(G).reshape(G, 1)).astype(np.float32)   # [30,1]
_RIrow = (np.arange(G).reshape(1, G)).astype(np.float32)   # [1,30]


def build_250():
    g = _G()
    c2 = _plane(g, 2)
    c5 = _plane(g, 5)
    real = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)

    RIc = g.f([1, 1, G, 1], _RIcol)
    RIr = g.f([1, 1, 1, G], _RIrow)
    ONE = g.f([1], [1.0]); K = g.f([1], [1000.0]); H = g.f([1], [0.5])

    c2rows = g.nd("ReduceMax", [c2], axes=[3], keepdims=1)   # [1,1,30,1]
    c2cols = g.nd("ReduceMax", [c2], axes=[2], keepdims=1)   # [1,1,1,30]
    notrow = g.nd("Sub", [ONE, c2rows])
    notcol = g.nd("Sub", [ONE, c2cols])

    r0 = g.nd("ReduceMin", [g.nd("Add", [RIc, g.nd("Mul", [notrow, K])])], axes=[2], keepdims=1)
    r1 = g.nd("ReduceMax", [g.nd("Sub", [RIc, g.nd("Mul", [notrow, K])])], axes=[2], keepdims=1)
    lo_r = g.nd("Sub", [r0, ONE]); hi_r = g.nd("Add", [r1, ONE])
    Dr = g.nd("Max", [g.nd("Min", [RIr, hi_r]), lo_r])      # [1,1,1,30] dest-row per src-row
    PR = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [RIc, Dr])]), H])], to=int(F))

    c0 = g.nd("ReduceMin", [g.nd("Add", [RIr, g.nd("Mul", [notcol, K])])], axes=[3], keepdims=1)
    c1 = g.nd("ReduceMax", [g.nd("Sub", [RIr, g.nd("Mul", [notcol, K])])], axes=[3], keepdims=1)
    lo_c = g.nd("Sub", [c0, ONE]); hi_c = g.nd("Add", [c1, ONE])
    Dc = g.nd("Max", [g.nd("Min", [RIr, hi_c]), lo_c])      # [1,1,1,30] dest-col per src-col
    Dc_col = g.nd("Transpose", [Dc], perm=[0, 1, 3, 2])      # [1,1,30,1]
    PC = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [Dc_col, RIr])]), H])], to=int(F))

    moved = g.nd("MatMul", [g.nd("MatMul", [PR, c5]), PC])
    moved5 = g.nd("Cast", [g.nd("Greater", [moved, H])], to=int(F))

    ch0 = g.nd("Sub", [g.nd("Sub", [real, c2]), moved5])
    z = g.nd("Sub", [c2, c2])
    g.nd("Concat", [ch0, z, c2, z, z, moved5, z, z, z, z], "output", axis=1)
    return _model(g)


def _ref_250(a):
    h, w = a.shape
    def emb(m):
        z = np.zeros((G, G), np.float32); z[:h, :w] = np.asarray(m, np.float32); return z
    c5 = emb(a == 5); c2 = emb(a == 2)
    if c2.sum() == 0:
        return a.copy()
    RI = np.arange(G).astype(np.float32)
    c2rows = c2.max(axis=1); c2cols = c2.max(axis=0)
    notrow = 1 - c2rows; notcol = 1 - c2cols
    r0 = (RI + notrow * 1000).min(); r1 = (RI - notrow * 1000).max()
    c0 = (RI + notcol * 1000).min(); c1 = (RI - notcol * 1000).max()
    Dr = np.maximum(np.minimum(RI, r1 + 1), r0 - 1)
    Dc = np.maximum(np.minimum(RI, c1 + 1), c0 - 1)
    PR = (np.abs(RI[:, None] - Dr[None, :]) < 0.5).astype(np.float32)
    PC = (np.abs(Dc[:, None] - RI[None, :]) < 0.5).astype(np.float32)
    moved5 = ((PR @ c5 @ PC) > 0.5).astype(np.float32)
    out = np.zeros((h, w), int)
    out[c2[:h, :w] > 0] = 2
    out[moved5[:h, :w] > 0] = 5
    return out


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _matches(prs, ref):
    for a, b in prs:
        try:
            o = ref(a)
        except Exception:
            return False
        if o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    same = all(a.shape == b.shape for a, b in prs)

    # task 270 : centres {1,2}, markers {3,7}
    if same:
        incol = set()
        outcol = set()
        for a, b in prs:
            incol |= set(np.unique(a).tolist())
            outcol |= set(np.unique(b).tolist())
        if incol <= {0, 1, 2, 3, 7} and outcol <= {0, 1, 2, 3, 7}:
            if _matches(prs, _ref_270):
                try:
                    m = build_270()
                    onnx.checker.check_model(m, full_check=True)
                    out.append(("plus_markers", m))
                except Exception:
                    pass

    # task 381 : fill-between colour-2 walls with 9
    if same:
        incol = set()
        outcol = set()
        for a, b in prs:
            incol |= set(np.unique(a).tolist())
            outcol |= set(np.unique(b).tolist())
        if incol <= {0, 2} and outcol <= {0, 2, 9}:
            if _matches(prs, _ref_381):
                try:
                    m = build_381()
                    onnx.checker.check_model(m, full_check=True)
                    out.append(("fill_between_h", m))
                except Exception:
                    pass

    # task 139 : bounding-box fill (colour 4 -> add 7 inside each blob's bbox)
    if same:
        incol = set(); outcol = set()
        for a, b in prs:
            incol |= set(np.unique(a).tolist())
            outcol |= set(np.unique(b).tolist())
        if incol <= {0, 4} and outcol <= {0, 4, 7}:
            if _matches(prs, _ref_139):
                try:
                    m = build_139()
                    onnx.checker.check_model(m, full_check=True)
                    out.append(("bbox_fill7", m))
                except Exception:
                    pass

    # task 250 : pull colour-5 markers onto the 2x2 colour-2 box
    if same:
        incol = set(); outcol = set()
        for a, b in prs:
            incol |= set(np.unique(a).tolist())
            outcol |= set(np.unique(b).tolist())
        if incol <= {0, 2, 5} and outcol <= {0, 2, 5}:
            if _matches(prs, _ref_250):
                try:
                    m = build_250()
                    onnx.checker.check_model(m, full_check=True)
                    out.append(("pull_to_box", m))
                except Exception:
                    pass

    return out
