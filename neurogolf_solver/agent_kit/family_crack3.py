"""family_crack3 -- a grab-bag of statically-expressible ARC transforms.

Each rule below is detected from the train/test/arc-gen pairs (numpy, raw
variable-size grids) and, when it reproduces EVERY available pair exactly, an
opset-10 ONNX graph realising the same rule is emitted.  All graphs keep content
anchored at the top-left origin so they generalise across the (variable, zero
padded) 30x30 layout.

Rules
-----
bboxcrop      Crop to the bounding box of the non-background content and move it
              to the origin.  Realised with two data-dependent shift matrices
              (built from scalar leading-empty counts via Relu/Abs, no Equal) and
              two clip masks that zero everything outside the cropped rectangle.
htile2        Duplicate the content horizontally: output = [grid | grid].  The
              second copy is placed via a data-dependent right shift of W columns
              (W = grid width), again a Relu/Abs shift matrix.
bboxarea2x2   Emit a 2x2 block of the color whose per-color BOUNDING-BOX AREA is
              largest (background excluded).  Pure per-channel reductions.
diagrepeat    Repeat the small origin motif along the main diagonal (3x3 -> 6x6)
              with a single depthwise diagonal Conv, then re-derive the
              background channel and clip to the 6x6 window.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_BIG = 1.0e6


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def const(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         [float(v) for v in np.ravel(vals).tolist()]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _pairs(ex):
    out = []
    for split in ("train", "test"):
        for e in ex.get(split, []):
            out.append((np.array(e["input"]), np.array(e["output"])))
    return out


def _all(ex):
    out = []
    for split in ("train", "test", "arc-gen"):
        for e in ex.get(split, []):
            out.append((np.array(e["input"]), np.array(e["output"])))
    return out


# --------------------------------------------------------------------------- #
# shared building blocks                                                       #
# --------------------------------------------------------------------------- #
def _chmask_nbg(g):
    """[1,10,1,1] : 0 for background channel, 1 for colours 1..9."""
    return g.const([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))


def _e0(g):
    """[1,10,1,1] one-hot for the background channel."""
    return g.const([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))


def _shift_matrix(g, scalar_k, kind):
    """[1,1,30,30] matrix M with M[a,b]=1 where (b-a)==k (kind='row', left mult)
    or (a-b)==k (kind='col', right mult), built from scalar k (>=0) with no Equal.

    For a row shift up by k we want output row i = input row i+k, i.e. left-multiply
    by S with S[i,j]=1 iff j-i==k.  For a column shift left by k we want output col
    c = input col c+k, i.e. right-multiply by T with T[j,c]=1 iff j-c==k.
    Both reduce to Relu(1 - |diff - k|) with the appropriate constant diff matrix.
    """
    a = np.arange(30).reshape(30, 1)
    b = np.arange(30).reshape(1, 30)
    if kind == "row":
        diffm = (b - a).astype(float)      # diff[i,j] = j - i
    else:
        diffm = (a - b).astype(float)      # diff[j,c] = j - c
    diff_c = g.const([1, 1, 30, 30], diffm)
    d = g.node("Sub", [diff_c, scalar_k])          # diff - k   (broadcast)
    ad = g.node("Abs", [d])
    one = g.const([1, 1, 1, 1], [1.0])
    return g.node("Relu", [g.node("Sub", [one, ad])])


def _fgcell(g):
    """[1,1,30,30] : 1 at non-background real cells, else 0."""
    masked = g.node("Mul", ["input", _chmask_nbg(g)])
    return g.node("ReduceSum", [masked], axes=[1], keepdims=1)


def _leading_empty(g, has, axis):
    """Given `has` (1 where a row/col contains non-bg) over `axis` (2 or 3),
    return scalar [1,1,1,1] = number of leading all-empty rows/cols (r0/c0)."""
    n = HEIGHT if axis == 2 else WIDTH
    # cumulative sum cs along `axis` via lower-triangular matmul
    Lvals = np.tril(np.ones((30, 30))).astype(float)          # L[i,j]=1 if j<=i
    if axis == 2:                                             # rows: S = L @ has
        L = g.const([1, 1, 30, 30], Lvals)
        cs = g.node("MatMul", [L, has])
        red_axis = 2
    else:                                                    # cols: cs = has @ U
        U = g.const([1, 1, 30, 30], Lvals.T)                  # U[j,c]=1 if j<=c
        cs = g.node("MatMul", [has, U])
        red_axis = 3
    one = g.const([1, 1, 1, 1], [1.0])
    zind = g.node("Relu", [g.node("Sub", [one, cs])])         # 1 where cs==0
    return g.node("ReduceSum", [zind], axes=[red_axis], keepdims=1)


def _revmat(g, length):
    """[1,1,30,30] anti-diagonal R with R[a,b]=1 iff a+b == length-1 (length scalar).
    Reverses the first `length` rows/cols; indices >= length map to nothing -> 0."""
    a = np.arange(30).reshape(30, 1)
    b = np.arange(30).reshape(1, 30)
    sumidx = g.const([1, 1, 30, 30], (a + b).astype(float))
    one = g.const([1, 1, 1, 1], [1.0])
    target = g.node("Sub", [length, one])               # length-1
    d = g.node("Abs", [g.node("Sub", [sumidx, target])])
    return g.node("Relu", [g.node("Sub", [one, d])])


def _trailing_empty(g, has, axis):
    """Scalar [1,1,1,1] = #indices i in 0..29 with no non-bg at/after i (== 29-last)."""
    Uvals = np.triu(np.ones((30, 30))).astype(float)          # U[i,j]=1 if j>=i
    if axis == 2:
        U = g.const([1, 1, 30, 30], Uvals)
        rcs = g.node("MatMul", [U, has])
        red_axis = 2
    else:
        L = g.const([1, 1, 30, 30], Uvals.T)                  # L[j,c]=1 if j>=c
        rcs = g.node("MatMul", [has, L])
        red_axis = 3
    one = g.const([1, 1, 1, 1], [1.0])
    zind = g.node("Relu", [g.node("Sub", [one, rcs])])
    return g.node("ReduceSum", [zind], axes=[red_axis], keepdims=1)


# --------------------------------------------------------------------------- #
# RULE: bboxcrop                                                               #
# --------------------------------------------------------------------------- #
def _bbox(a):
    nz = np.argwhere(a != 0)
    if len(nz) == 0:
        return None
    r0, c0 = nz.min(0)
    r1, c1 = nz.max(0)
    return a[r0:r1 + 1, c0:c1 + 1]


_POST = {
    "id":  lambda c: c,
    "lr":  lambda c: c[:, ::-1],
    "ud":  lambda c: c[::-1],
    "180": lambda c: c[::-1, ::-1],
    "T":   lambda c: c.T,
}


def _detect_bbox(prs, post):
    """post(bbox(input)) == output for every pair.  Returns 'shift' flag (some crop
    actually trimmed something) so we can drop the trivial identity case."""
    fn = _POST[post]
    seen_shift = False
    for a, b in prs:
        crop = _bbox(a)
        if crop is None:
            return None
        e = fn(crop)
        if e.shape != b.shape or not np.array_equal(e, b):
            return None
        if crop.shape != a.shape:
            seen_shift = True
    return seen_shift


def _build_bbox(post):
    g = _G()
    fg = _fgcell(g)                                            # [1,1,30,30]
    rowhas = g.node("Clip", [g.node("ReduceSum", [fg], axes=[3], keepdims=1)],
                    min=0.0, max=1.0)                          # [1,1,30,1]
    colhas = g.node("Clip", [g.node("ReduceSum", [fg], axes=[2], keepdims=1)],
                    min=0.0, max=1.0)                          # [1,1,1,30]

    r0 = _leading_empty(g, rowhas, 2)
    c0 = _leading_empty(g, colhas, 3)
    rtrail = _trailing_empty(g, rowhas, 2)
    ctrail = _trailing_empty(g, colhas, 3)

    c30 = g.const([1, 1, 1, 1], [30.0])
    ch = g.node("Sub", [g.node("Sub", [c30, r0]), rtrail])    # output real height
    cw = g.node("Sub", [g.node("Sub", [c30, c0]), ctrail])    # output real width

    S = _shift_matrix(g, r0, "row")                           # [1,1,30,30]
    T = _shift_matrix(g, c0, "col")                           # [1,1,30,30]
    x1 = g.node("MatMul", [S, "input"])                       # shift rows up by r0
    x2 = g.node("MatMul", [x1, T])                            # shift cols left by c0

    ar = g.const([1, 1, 30, 1], np.arange(30))
    ac = g.const([1, 1, 1, 30], np.arange(30))
    maskR = g.node("Clip", [g.node("Sub", [ch, ar])], min=0.0, max=1.0)
    maskC = g.node("Clip", [g.node("Sub", [cw, ac])], min=0.0, max=1.0)

    if post == "id":
        out = g.node("Mul", [g.node("Mul", [x2, maskR]), maskC])
    elif post == "lr":                                        # reverse cols -> mask rows
        Rc = _revmat(g, cw)
        out = g.node("Mul", [g.node("MatMul", [x2, Rc]), maskR])
    elif post == "ud":                                        # reverse rows -> mask cols
        Rr = _revmat(g, ch)
        out = g.node("Mul", [g.node("MatMul", [Rr, x2]), maskC])
    elif post == "180":
        Rr = _revmat(g, ch)
        Rc = _revmat(g, cw)
        out = g.node("MatMul", [g.node("MatMul", [Rr, x2]), Rc])
    else:                                                     # transpose (origin-safe)
        masked = g.node("Mul", [g.node("Mul", [x2, maskR]), maskC])
        out = g.node("Transpose", [masked], perm=[0, 1, 3, 2])
    g.nodes[-1].output[0] = "output"
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# RULE: htile2  (output = [grid | grid])                                       #
# --------------------------------------------------------------------------- #
def _detect_htile2(prs):
    ok = False
    for a, b in prs:
        til = np.concatenate([a, a], axis=1)
        if til.shape != b.shape or not np.array_equal(til, b):
            return False
        ok = True
    return ok


def _build_htile2():
    g = _G()
    real = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    colsum = g.node("ReduceSum", [real], axes=[2], keepdims=1)    # [1,1,1,30]
    colhas = g.node("Clip", [colsum], min=0.0, max=1.0)
    W = g.node("ReduceSum", [colhas], axes=[3], keepdims=1)       # scalar grid width
    a = np.arange(30).reshape(30, 1)
    b = np.arange(30).reshape(1, 30)
    diff = g.const([1, 1, 30, 30], (a - b).astype(float))         # diff[j,c]=j-c
    one = g.const([1, 1, 1, 1], [1.0])
    # want T2[j,c]=1 iff j-c == -W  ->  Relu(1 - |diff + W|)
    s = g.node("Add", [diff, W])
    T2 = g.node("Relu", [g.node("Sub", [one, g.node("Abs", [s])])])
    copy2 = g.node("MatMul", ["input", T2])
    g.node("Add", ["input", copy2], out="output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# RULE: bboxarea2x2                                                            #
# --------------------------------------------------------------------------- #
def _area_vec(a):
    H, W = a.shape
    area = np.full(10, -_BIG)
    for c in range(1, 10):
        nz = np.argwhere(a == c)
        if len(nz) == 0:
            continue
        r0, c0 = nz.min(0)
        r1, c1 = nz.max(0)
        area[c] = (r1 - r0 + 1) * (c1 - c0 + 1)
    return area


def _detect_bboxarea2x2(prs):
    ok = False
    for a, b in prs:
        if b.shape != (2, 2):
            return False
        area = _area_vec(a)
        mx = area.max()
        if np.sum(np.abs(area - mx) < 0.5) != 1:
            return False
        win = int(np.argmax(area))
        exp = np.full((2, 2), win)
        if not np.array_equal(b, exp):
            return False
        ok = True
    return ok


def _build_bboxarea2x2():
    g = _G()
    ar = g.const([1, 1, 30, 1], np.arange(30))
    ac = g.const([1, 1, 1, 30], np.arange(30))
    one = g.const([1, 1, 1, 1], [1.0])
    big = g.const([1, 1, 1, 1], [_BIG])

    rowpres = g.node("ReduceMax", ["input"], axes=[3], keepdims=1)   # [1,10,30,1]
    rmax = g.node("ReduceMax", [g.node("Mul", [rowpres, ar])], axes=[2], keepdims=1)
    inv_rp = g.node("Sub", [one, rowpres])
    rmin = g.node("ReduceMin",
                  [g.node("Add", [g.node("Mul", [rowpres, ar]),
                                  g.node("Mul", [inv_rp, big])])],
                  axes=[2], keepdims=1)
    height = g.node("Add", [g.node("Sub", [rmax, rmin]), one])

    colpres = g.node("ReduceMax", ["input"], axes=[2], keepdims=1)   # [1,10,1,30]
    cmax = g.node("ReduceMax", [g.node("Mul", [colpres, ac])], axes=[3], keepdims=1)
    inv_cp = g.node("Sub", [one, colpres])
    cmin = g.node("ReduceMin",
                  [g.node("Add", [g.node("Mul", [colpres, ac]),
                                  g.node("Mul", [inv_cp, big])])],
                  axes=[3], keepdims=1)
    width = g.node("Add", [g.node("Sub", [cmax, cmin]), one])

    present = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    area_raw = g.node("Mul", [g.node("Mul", [height, width]), present])
    inv_pr = g.node("Sub", [one, present])
    area = g.node("Sub", [area_raw, g.node("Mul", [inv_pr, big])])
    area = g.node("Sub", [area, g.node("Mul", [_e0(g), big])])       # drop background

    mx = g.node("ReduceMax", [area], axes=[1], keepdims=1)
    ind = g.node("Relu", [g.node("Sub", [one, g.node("Abs", [g.node("Sub", [area, mx])])])])

    mask2 = np.zeros((30, 30)); mask2[:2, :2] = 1.0
    sm = g.const([1, 1, 30, 30], mask2)
    g.node("Mul", [ind, sm], out="output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# RULE: diagrepeat (small origin motif tiled along main diagonal, NxN -> 2Nx2N)#
# --------------------------------------------------------------------------- #
def _diag_emul(a, oH, oW, K):
    H, W = a.shape
    big = np.zeros((oH, oW), dtype=int)
    for r in range(H):
        for c in range(W):
            if a[r, c] != 0:
                for k in range(K + 1):
                    rr, cc = r + k, c + k
                    if rr < oH and cc < oW:
                        big[rr, cc] = a[r, c]
    return big


def _detect_diagrepeat(prs_all):
    """Require constant in/out sizes (NxN -> 2Nx2N) and non-overlapping diagonal
    tiling that reproduces every pair (matching the depthwise-conv summation)."""
    insh = set(); outsh = set()
    for a, b in prs_all:
        insh.add(a.shape); outsh.add(b.shape)
    if len(insh) != 1 or len(outsh) != 1:
        return None
    (H, W), = insh
    (oH, oW), = outsh
    if (oH, oW) != (2 * H, 2 * W) or H != W:
        return None
    K = oH - 1
    for a, b in prs_all:
        # conv-sum exactness: every output cell gets at most one colour
        emul = _diag_emul(a, oH, oW, K)
        if not np.array_equal(emul, b):
            return None
        # verify no overlap of differently-coloured diagonal copies (conv sums!)
        cover = np.zeros((oH, oW), dtype=int)
        for r in range(H):
            for c in range(W):
                if a[r, c] != 0:
                    for k in range(K + 1):
                        rr, cc = r + k, c + k
                        if rr < oH and cc < oW:
                            cover[rr, cc] += 1
        if cover.max() > 1:
            return None
    return (oH, oW, K)


def _build_diagrepeat(oH, oW, K):
    g = _G()
    ksz = K + 1
    # depthwise diagonal conv: weight[o,0,i,j] = 1 if i==j
    wv = np.zeros((CHANNELS, 1, ksz, ksz), dtype=float)
    for i in range(ksz):
        wv[:, 0, i, i] = 1.0
    g.inits.append(oh.make_tensor("dW", DATA_TYPE, [CHANNELS, 1, ksz, ksz],
                                  wv.ravel().tolist()))
    D = g.node("Conv", ["input", "dW"], group=CHANNELS, kernel_shape=[ksz, ksz],
               pads=[K, K, 0, 0], strides=[1, 1])
    chm = _chmask_nbg(g)
    mask = np.zeros((30, 30)); mask[:oH, :oW] = 1.0
    m6 = g.const([1, 1, 30, 30], mask)
    fpos = g.node("Mul", [g.node("Mul", [D, chm]), m6])        # colours 1..9, windowed
    fgsum = g.node("ReduceSum", [fpos], axes=[1], keepdims=1)
    bg0 = g.node("Sub", [m6, fgsum])
    bgchan = g.node("Mul", [_e0(g), bg0])
    g.node("Add", [fpos, bgchan], out="output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# RULE: pinwheel  (square NxN -> 2Nx2N rotational kaleidoscope)                 #
#   [[ input , rot90CW  ],                                                      #
#    [rot90CCW, rot180   ]]                                                     #
# --------------------------------------------------------------------------- #
def _detect_pinwheel(prs_all):
    ok = False
    for a, b in prs_all:
        H, W = a.shape
        if H != W or b.shape != (2 * H, 2 * W):
            return False
        cw = np.rot90(a, 3)
        ccw = np.rot90(a, 1)
        r180 = np.rot90(a, 2)
        blk = np.block([[a, cw], [ccw, r180]])
        if not np.array_equal(blk, b):
            return False
        ok = True
    return ok


def _build_pinwheel():
    g = _G()
    real = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)      # [1,1,30,30]
    colsum = g.node("ReduceSum", [real], axes=[2], keepdims=1)
    colhas = g.node("Clip", [colsum], min=0.0, max=1.0)
    N = g.node("ReduceSum", [colhas], axes=[3], keepdims=1)          # grid size scalar
    zero = g.const([1, 1, 1, 1], [0.0])
    negN = g.node("Sub", [zero, N])

    R = _revmat(g, N)                                                # reverse first N
    Sd = _shift_matrix(g, negN, "row")                              # shift down by N
    Tr = _shift_matrix(g, negN, "col")                             # shift right by N

    Tp = g.node("Transpose", ["input"], perm=[0, 1, 3, 2])
    rcw = g.node("MatMul", [Tp, R])                                 # rot90 CW
    rccw = g.node("MatMul", [R, Tp])                                # rot90 CCW
    r180 = g.node("MatMul", [R, g.node("MatMul", ["input", R])])     # rot180

    TL = "input"
    TR = g.node("MatMul", [rcw, Tr])
    BL = g.node("MatMul", [Sd, rccw])
    BR = g.node("MatMul", [Sd, g.node("MatMul", [r180, Tr])])
    s = g.node("Add", [TL, TR])
    s = g.node("Add", [s, BL])
    g.node("Add", [s, BR], out="output")
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    prs_all = _all(ex)
    for post in ("id", "lr", "ud", "180", "T"):
        try:
            shift = _detect_bbox(prs_all, post)
        except Exception:
            shift = None
        if shift is None:
            continue
        if post == "id" and not shift:        # pure identity -> skip trivial
            continue
        try:
            out.append((f"bbox_{post}", _build_bbox(post)))
        except Exception:
            pass
    try:
        if _detect_htile2(prs):
            out.append(("htile2", _build_htile2()))
    except Exception:
        pass
    try:
        if _detect_bboxarea2x2(prs):
            out.append(("bboxarea2x2", _build_bboxarea2x2()))
    except Exception:
        pass
    try:
        dr = _detect_diagrepeat(_all(ex))
        if dr is not None:
            out.append(("diagrepeat", _build_diagrepeat(*dr)))
    except Exception:
        pass
    try:
        if _detect_pinwheel(prs_all):
            out.append(("pinwheel", _build_pinwheel()))
    except Exception:
        pass
    return out
