"""family_vc2_4 — two RETRY tasks translated to static opset-10 ONNX.

task191 (verify_7df24a62): a solid template block (frame colour 1, marker colour 4)
sits in a field of scattered single-cell colour-4 noise.  Wherever a set of noise
markers exactly reproduces (some D4 orientation of) the template's marker pattern
AND the surrounding bbox is otherwise empty, draw the template frame around them.
Implemented with a runtime-extracted KxK marker kernel, 8 D4-oriented correlation
kernels (built via runtime permutation-matrix MatMuls), a single 8-out-channel Conv
that does the exact positive+negative match test, and a single 8-in-channel
ConvTranspose that stamps the bbox rectangles.

task366 (verify_e6721834): NOT emitted.  The rule (split into two equal halves along
the axis with more monochrome frontiers; base = fewer-colour half; stamp each source
object's solid-rectangle border around its matching interior in the base) is itself
static-expressible (4-combo axis/base blend + flood-peel object separation + runtime
kernels).  The irreducible blocker is the tie-break: when a source object's interior
matches the base panel at MORE THAN ONE location, the verifier stamps only the FIRST
via first(frozenset(occurrences)) — CPython set hash-order, which is non-geometric.
On the NeuroGolf data this fires on 4/266 arc-gen examples (148,151,217,25) and the
picked occurrence is inconsistent with any spatial order (row-major, col-major, etc.),
so no static tensor computation can reproduce it and exact grading is impossible.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = onnx.TensorProto.FLOAT
I64 = onnx.TensorProto.INT64
H = W = 30
K = 5
BIG = 1000.0


# --------------------------------------------------------------------------- #
# numpy reference for task191 (exact vs Hodel verifier on all 267 examples)   #
# --------------------------------------------------------------------------- #
def _solve191(I):
    I = np.asarray(I)
    h, w = I.shape
    P = np.zeros((h + 2, w + 2), int)
    P[1:h + 1, 1:w + 1] = I
    HH, WW = P.shape
    Fp = (P == 1).astype(float)
    Mp = (P == 4).astype(float)
    ys, xs = np.where(Fp > 0)
    if len(ys) == 0:
        return I.copy()
    rmin, rmax, cmin, cmax = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = rmax - rmin + 1, cmax - cmin + 1
    if bh > K or bw > K:
        return None
    Mpad = np.zeros((HH + K, WW + K))
    Mpad[:HH, :WW] = Mp
    Kid = np.zeros((K, K))
    for dr in range(K):
        for dc in range(K):
            r, c = rmin + dr, cmin + dc
            if dr < bh and dc < bw and r < HH and c < WW:
                Kid[dr, dc] = Mp[r, c]

    def rowflip(A, hh):
        B = A.copy(); B[:hh] = A[:hh][::-1]; return B

    def colflip(A, ww):
        B = A.copy(); B[:, :ww] = A[:, :ww][:, ::-1]; return B

    base = [Kid, rowflip(Kid, bh), colflip(Kid, bw), colflip(rowflip(Kid, bh), bw)]
    oris = [(b, bh, bw) for b in base] + [(b.T, bw, bh) for b in base]
    n = Kid.sum()
    cover = np.zeros((HH, WW))
    for Kg, rh, rw in oris:
        for i in range(HH - rh + 1):
            for j in range(WW - rw + 1):
                win_m = Mpad[i:i + K, j:j + K]
                mh = (win_m * Kg).sum()
                rectM = Mp[i:i + rh, j:j + rw].sum()
                rectF = Fp[i:i + rh, j:j + rw].sum()
                if n > 0 and mh == n and rectM == n and rectF == 0:
                    cover[i:i + rh, j:j + rw] = 1
    resP = P.copy()
    resP[(cover > 0) & (P == 0)] = 1
    return resP[1:h + 1, 1:w + 1]


# --------------------------------------------------------------------------- #
# ONNX graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes, self.inits, self._k = [], [], 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        n = self.nm("c")
        self.inits.append(oh.make_tensor(
            n, F, list(dims), [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return n

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, I64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **attrs):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out

    def clip01(self, x):
        return self.nd("Clip", [x], min=0.0, max=1.0)

    def eq0(self, x):
        return self.nd("Cast", [self.nd("Less", [self.nd("Abs", [x]), self.f([1, 1, 1, 1], [0.5])])], to=F)

    def chan(self, x, a, b):
        return self.nd("Slice", [x, self.i64([a]), self.i64([b]), self.i64([1])])


def _model(g, name):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, name, [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# builder: task191                                                            #
# --------------------------------------------------------------------------- #
def _build191():
    g = _G()
    rowidx = g.f([1, 1, H, 1], list(range(H)))    # index over dim2
    colidx = g.f([1, 1, 1, W], list(range(W)))    # index over dim3
    drk = g.f([1, 1, K, 1], list(range(K)))
    dck = g.f([1, 1, 1, K], list(range(K)))
    one = g.f([1, 1, 1, 1], [1.0])

    # shift content by (1,1): pad top/left by 1 then crop to 30x30
    def shift_pos(x):  # +1,+1
        p = g.nd("Pad", [x], mode="constant", value=0.0, pads=[0, 0, 1, 1, 0, 0, 0, 0])
        return g.nd("Slice", [p, g.i64([0, 0]), g.i64([H, W]), g.i64([2, 3])])

    F0 = g.chan("input", 1, 2)
    M0 = g.chan("input", 4, 5)
    Fsh = shift_pos(F0)          # [1,1,30,30]
    Msh = shift_pos(M0)

    # --- bbox of frame on shifted canvas ---
    rowHas = g.nd("ReduceMax", [Fsh], axes=[3], keepdims=1)          # [1,1,30,1]
    colHas = g.nd("ReduceMax", [Fsh], axes=[2], keepdims=1)          # [1,1,1,30]
    BIGc = g.f([1, 1, 1, 1], [1e4])
    # rmin = min index with rowHas=1
    rmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [rowidx, rowHas]),
              g.nd("Mul", [BIGc, g.nd("Sub", [one, rowHas])])])], axes=[2], keepdims=1)
    rmax = g.nd("ReduceMax", [g.nd("Mul", [rowidx, rowHas])], axes=[2], keepdims=1)
    cmin = g.nd("ReduceMin", [g.nd("Add", [g.nd("Mul", [colidx, colHas]),
              g.nd("Mul", [BIGc, g.nd("Sub", [one, colHas])])])], axes=[3], keepdims=1)
    cmax = g.nd("ReduceMax", [g.nd("Mul", [colidx, colHas])], axes=[3], keepdims=1)
    bh = g.nd("Add", [g.nd("Sub", [rmax, rmin]), one])    # [1,1,1,1]
    bw = g.nd("Add", [g.nd("Sub", [cmax, cmin]), one])

    # --- extract KxK marker patch at (rmin,cmin) via selection matmuls ---
    # R[.,k,j] = 1 iff j == rmin + k    (shape [1,1,K,30])
    Rsel = g.eq0(g.nd("Sub", [g.nd("Sub", [colidx, drk]), rmin]))       # broadcast [1,1,K,30]
    # Cm[.,c,k'] = 1 iff c == cmin + k' (shape [1,1,30,K])
    Cm = g.eq0(g.nd("Sub", [g.nd("Sub", [rowidx, dck]), cmin]))         # [1,1,30,K]
    Kid_raw = g.nd("MatMul", [g.nd("MatMul", [Rsel, Msh]), Cm])         # [1,1,K,K]

    # rect masks
    rowlt = g.nd("Cast", [g.nd("Less", [drk, bh])], to=F)   # [1,1,K,1]
    collt = g.nd("Cast", [g.nd("Less", [dck, bw])], to=F)   # [1,1,1,K]
    Ra = g.nd("Mul", [rowlt, collt])                        # [1,1,K,K]  bh x bw
    Rb = g.nd("Transpose", [Ra], perm=[0, 1, 3, 2])         # bw x bh
    Kid = g.nd("Mul", [Kid_raw, Ra])                        # clean template pattern
    n = g.nd("ReduceSum", [Kid], axes=[2, 3], keepdims=1)   # [1,1,1,1]

    # permutation matrices (reverse first bh / bw rows/cols)
    Ii = drk           # [1,1,K,1]
    Jj = dck           # [1,1,1,K]
    def perm(dim):     # P[i,j]=(i<dim)&(j<dim)&(i+j==dim-1)
        ilt = g.nd("Cast", [g.nd("Less", [Ii, dim])], to=F)   # [1,1,K,1]
        jlt = g.nd("Cast", [g.nd("Less", [Jj, dim])], to=F)   # [1,1,1,K]
        s = g.nd("Add", [Ii, Jj])                             # [1,1,K,K]
        seq = g.eq0(g.nd("Sub", [s, g.nd("Sub", [dim, one])]))
        return g.nd("Mul", [g.nd("Mul", [ilt, jlt]), seq])
    Prow = perm(bh)
    Pcol = perm(bw)

    b0 = Kid
    b1 = g.nd("MatMul", [Prow, Kid])
    b2 = g.nd("MatMul", [Kid, Pcol])
    b3 = g.nd("MatMul", [b1, Pcol])
    bases = [b0, b1, b2, b3]
    Ks = bases + [g.nd("Transpose", [b], perm=[0, 1, 3, 2]) for b in bases]
    Rects = [Ra, Ra, Ra, Ra, Rb, Rb, Rb, Rb]

    n1 = g.nd("Add", [n, one])
    n2 = g.nd("Add", [n1, one])
    negBIG = g.f([1, 1, 1, 1], [-BIG])
    # build conv weight [8,2,K,K]
    chans = []
    for Kg, Rg in zip(Ks, Rects):
        SK = g.nd("Sub", [g.nd("Mul", [n2, Kg]), g.nd("Mul", [n1, Rg])])  # [1,1,K,K]
        nr = g.nd("Mul", [negBIG, Rg])
        chans.append(g.nd("Concat", [SK, nr], axis=1))                    # [1,2,K,K]
    Wconv = g.nd("Concat", chans, axis=0)                                 # [8,2,K,K]

    # conv input [1,2,34,34]
    Mpad = g.nd("Pad", [Msh], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, K - 1, K - 1])
    Fpad = g.nd("Pad", [Fsh], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, K - 1, K - 1])
    stacked = g.nd("Concat", [Mpad, Fpad], axis=1)                        # [1,2,34,34]
    score = g.nd("Conv", [stacked, Wconv], kernel_shape=[K, K], group=1)  # [1,8,30,30]

    thresh = g.nd("Sub", [n, g.f([1, 1, 1, 1], [0.5])])                   # [1,1,1,1]
    match8 = g.nd("Cast", [g.nd("Greater", [score, thresh])], to=F)       # [1,8,30,30]

    # stamp rects: ConvTranspose weight [8,1,K,K]
    rectW = g.nd("Concat", Rects, axis=0)                                 # [8,1,K,K]
    cover_raw = g.nd("ConvTranspose", [match8, rectW], kernel_shape=[K, K])  # [1,1,34,34]
    cover = g.nd("Slice", [cover_raw, g.i64([0, 0]), g.i64([H, W]), g.i64([2, 3])])
    covered = g.clip01(cover)

    bg_sh = g.clip01(g.nd("Sub", [g.nd("Sub", [one, Fsh]), Msh]))         # 1 where background
    frameAdd_sh = g.nd("Mul", [covered, bg_sh])                          # [1,1,30,30]

    # shift back (-1,-1): crop top-left 1, pad bottom-right 1
    fa_c = g.nd("Slice", [frameAdd_sh, g.i64([1, 1]), g.i64([H, W]), g.i64([2, 3])])  # [1,1,29,29]
    fa_p = g.nd("Pad", [fa_c], mode="constant", value=0.0, pads=[0, 0, 0, 0, 0, 0, 1, 1])
    # only add frame at genuine background cells of the ORIGINAL grid (kills padding/border)
    frameAdd = g.nd("Mul", [fa_p, g.chan("input", 0, 1)])

    # assemble output: ch0 -= frameAdd, ch1 += frameAdd, rest unchanged
    in01 = g.chan("input", 0, 2)          # [1,2,30,30]
    rest = g.chan("input", 2, 10)         # [1,8,30,30]
    negFA = g.nd("Sub", [g.f([1, 1, 1, 1], [0.0]), frameAdd])
    d01 = g.nd("Concat", [negFA, frameAdd], axis=1)      # [1,2,30,30]
    new01 = g.nd("Add", [in01, d01])
    g.nd("Concat", [new01, rest], "output", axis=1)
    return _model(g, "vc2_4_191")


# --------------------------------------------------------------------------- #
def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    # task191
    ok = True
    for a, b in prs:
        if set(np.unique(a).tolist()) - {0, 1, 4}:
            ok = False; break
        r = _solve191(a)
        if r is None or r.shape != b.shape or not np.array_equal(r, b):
            ok = False; break
    if ok:
        yield ("vc2_4_191", _build191())
