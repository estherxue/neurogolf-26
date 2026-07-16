"""Verifier-decoded static-ONNX family for tasks 089/080/201 (opset 10).

task089 = verify_3e980e27 (template stamp at lone markers):
    A multicolor 8-connected object containing color 3 is stamped (as-is) at every
    isolated 3-cell; the multicolor object containing color 2 is stamped VMIRRORED
    at every isolated 2-cell.  ONNX: anchor = the 3/2 cell with a nonbg 8-neighbour;
    9x9 patch extracted around it with runtime MatMul selection matrices, masked to
    the connected object by a bounded 8-conn flood, then stamped at the isolated
    markers with a runtime-weight ConvTranspose.  Paint order: 2-stamps then 3-stamps.

task080 = verify_39e1d7f9 (block-lattice template stamp):
    Grid = lattice of hxw blocks separated by 1-cell frontier lines.  The connected
    multicolor block-cluster is the template; the colour of the "lonely" blocks (no
    block neighbours) is the key; the template is stamped (key-block ulcorner
    aligned) at the ulcorner of EVERY key-coloured block.  ONNX: frontier lines =
    monochrome full rows/cols (per-colour row-count == row-length); block adjacency
    = distance-2 jumps across frontier cells; lonely detection by bounded 4-conn
    flood; 25x25 patch extraction + block-adjacency flood mask + ConvTranspose stamp.

task201 = verify_846bdb03 (move scattered shape into corner-marked box):
    Colour 4 forms exactly the 4 corners of a box; the box has two colour stripes on
    its left/right (or top/bottom) edges.  Output = the box crop with the scattered
    two-colour shape (all nonbg outside the box) painted at (1,1), transposed to the
    stripe orientation and vmirrored unless the shape's colour order matches the
    stripe order.  ONNX: conditional Transpose blend, dyncrop-style MatMul crops of
    box and content, runtime anti-diagonal reflection matrix, cumulative-OR gate.

All three numpy references mirror the ONNX numerics exactly and are the detection
gate (must reproduce every train+test pair before a model is emitted).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
I64 = onnx.TensorProto.INT64
H = W = 30
CBIG = 1000.0

K89, T89 = 9, 20          # window + flood iters for task089
K80, T80A, T80B = 25, 12, 16   # window, lonely-flood iters, patch-flood iters


# --------------------------------------------------------------------------- #
# numpy helpers                                                               #
# --------------------------------------------------------------------------- #
def _np_shift(m, dr, dc):
    """out[p] = m[p - (dr,dc)] with zero fill."""
    h, w = m.shape
    out = np.zeros_like(m)
    src = m[max(-dr, 0):h - max(dr, 0), max(-dc, 0):w - max(dc, 0)]
    out[max(dr, 0):h - max(-dr, 0), max(dc, 0):w - max(-dc, 0)] = src
    return out


def _flood(seed, mask, kern, iters):
    """bounded dilate-and-mask flood; kern = list of (dr,dc) reach offsets."""
    f = (seed & mask).astype(np.uint8)
    mask = mask.astype(np.uint8)
    for _ in range(iters):
        d = f.copy()
        for dr, dc in kern:
            d |= _np_shift(f, dr, dc)
        f = d & mask
    return f.astype(bool)


_K8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
_K4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_KBLK = _K4 + [(0, 2), (0, -2), (2, 0), (-2, 0)]


# --------------------------------------------------------------------------- #
# task089 numpy reference                                                     #
# --------------------------------------------------------------------------- #
def _solve89(a):
    a = np.asarray(a, int)
    h, w = a.shape
    nonbg = a != 0
    nb = np.zeros((h, w), int)
    for dr, dc in _K8:
        nb += _np_shift(nonbg.astype(int), dr, dc)
    C = K89 // 2
    out = a.copy()
    for c, mirror in ((2, True), (3, False)):
        cm = a == c
        markers = np.argwhere(cm & (nb == 0))
        anchors = np.argwhere(cm & (nb > 0))
        if len(anchors) > 1:
            return None                       # rule needs a unique template anchor
        if len(anchors) == 0 or len(markers) == 0:
            continue
        ar, ac = anchors[0]
        pg = np.pad(a, C)
        pn = np.pad(nonbg, C)
        win = pg[ar:ar + K89, ac:ac + K89]
        wn = pn[ar:ar + K89, ac:ac + K89]
        seed = np.zeros((K89, K89), bool)
        seed[C, C] = True
        fl = _flood(seed, wn, _K8, T89)
        patch = win * fl
        if mirror:
            patch = patch[:, ::-1]
        stamp = np.zeros((h, w), int)
        for mr, mc in markers:
            ys, xs = np.nonzero(patch)
            for i, j in zip(ys, xs):
                rr, cc = mr + i - C, mc + j - C
                if 0 <= rr < h and 0 <= cc < w:
                    if stamp[rr, cc] not in (0, patch[i, j]):
                        return None           # conflicting overlap -> not static-safe
                    stamp[rr, cc] = patch[i, j]
        out = np.where(stamp > 0, stamp, out)
    return out


# --------------------------------------------------------------------------- #
# task080 numpy reference                                                     #
# --------------------------------------------------------------------------- #
def _solve80(a):
    a = np.asarray(a, int)
    h, w = a.shape
    frows = np.array([len(set(a[i, :].tolist())) == 1 for i in range(h)])
    fcols = np.array([len(set(a[:, j].tolist())) == 1 for j in range(w)])
    if not frows.any() or not fcols.any():
        return None
    fmask = frows[:, None] | fcols[None, :]
    nonbg = a != 0
    nonf = nonbg & ~fmask
    seed = np.zeros((h, w), bool)
    for dr, dc in _K4:
        seed |= (_np_shift(fmask.astype(int), -dr, -dc).astype(bool)
                 & _np_shift(nonf.astype(int), -2 * dr, -2 * dc).astype(bool))
    seed &= nonf
    hasnb = _flood(seed, nonf, _K4, T80A)
    lonely = nonf & ~hasnb
    keyset = set(a[lonely].tolist())
    if len(keyset) > 1:
        return None
    if len(keyset) == 0:
        return a.copy()                       # no lonely blocks -> identity
    key = keyset.pop()
    keyb = (a == key) & nonf
    ul = keyb & ~_np_shift(keyb.astype(int), 1, 0).astype(bool) \
              & ~_np_shift(keyb.astype(int), 0, 1).astype(bool)
    anchors = np.argwhere(ul & hasnb)
    if len(anchors) != 1:
        return None
    ar, ac = anchors[0]
    C = K80 // 2
    pg = np.pad(a, C)
    pn = np.pad(nonf, C)
    win = pg[ar:ar + K80, ac:ac + K80]
    wn = pn[ar:ar + K80, ac:ac + K80]
    s0 = np.zeros((K80, K80), bool)
    s0[C, C] = True
    fl = _flood(s0, wn, _KBLK, T80B)
    patch = win * fl
    out = a.copy()
    stamp = np.zeros((h, w), int)
    for mr, mc in np.argwhere(ul):
        ys, xs = np.nonzero(patch)
        for i, j in zip(ys, xs):
            rr, cc = mr + i - C, mc + j - C
            if 0 <= rr < h and 0 <= cc < w:
                if stamp[rr, cc] not in (0, patch[i, j]):
                    return None
                stamp[rr, cc] = patch[i, j]
    return np.where(stamp > 0, stamp, out)


# --------------------------------------------------------------------------- #
# task201 numpy reference                                                     #
# --------------------------------------------------------------------------- #
def _solve201(a):
    a = np.asarray(a, int)
    # marker colour: exactly 4 cells forming the corners of a proper rectangle;
    # data (all 266 examples) fixes it to colour 4 -- require that, and uniqueness.
    mcol = None
    for c in range(10):
        cells = np.argwhere(a == c)
        if len(cells) != 4:
            continue
        r0, r1 = cells[:, 0].min(), cells[:, 0].max()
        c0, c1 = cells[:, 1].min(), cells[:, 1].max()
        if r0 != r1 and c0 != c1 and \
                set(map(tuple, cells)) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            if mcol is not None:
                return None
            mcol = c
    if mcol != 4:
        return None
    cells = np.argwhere(a == 4)
    r0, r1 = cells[:, 0].min(), cells[:, 0].max()
    c0, c1 = cells[:, 1].min(), cells[:, 1].max()
    box = a[r0:r1 + 1, c0:c1 + 1].copy()
    cover = a.copy()
    cover[r0:r1 + 1, c0:c1 + 1] = 0
    t = box.shape[0] > box.shape[1]           # == "box has horizontal frontiers" on data
    if t:
        box, cover = box.T, cover.T
    bh, bw = box.shape
    if bh < 3 or bw < 3:
        return None
    ys, xs = np.nonzero(cover)
    if len(ys) == 0:
        return None
    cont = cover[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ca, cb = box[1, 0], box[1, bw - 1]
    if ca == 0 or cb == 0:
        return None
    la = np.argwhere(cont == ca)
    lb = np.argwhere(cont == cb)
    lac = la[:, 1].min() if len(la) else 10 ** 9
    lbc = lb[:, 1].min() if len(lb) else 10 ** 9
    if not (lac < lbc):
        cont = cont[:, ::-1]
    out = box.copy()
    ch, cw = cont.shape
    if 1 + ch > bh or 1 + cw > bw:            # content must fit inside the box
        return None
    m = cont != 0
    out[1:1 + ch, 1:1 + cw][m] = cont[m]
    return out.T if t else out


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

    # ---- common composites ----
    def consts(self):
        self.rowidx = self.f([1, 1, H, 1], list(range(H)))
        self.colidx = self.f([1, 1, 1, W], list(range(W)))
        self.half = self.f([1, 1, 1, 1], [0.5])
        self.one = self.f([1, 1, 1, 1], [1.0])
        self.cbig = self.f([1, 1, 1, 1], [CBIG])

    def clip01(self, x):
        return self.nd("Clip", [x], min=0.0, max=1.0)

    def not_(self, x):
        return self.nd("Sub", [self.one, x])

    def shift(self, x, dr, dc):
        """out[p] = x[p-(dr,dc)], zero fill, spatial dims of [1,C,30,30]."""
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        p = self.nd("Pad", [x], mode="constant", value=0.0,
                    pads=[0, 0, pt, pl, 0, 0, pb, pr])
        st = self.i64([max(-dr, 0), max(-dc, 0)])
        en = self.i64([max(-dr, 0) + H, max(-dc, 0) + W])
        ax = self.i64([2, 3])
        return self.nd("Slice", [p, st, en, ax])

    def chan(self, x, c):
        return self.nd("Slice", [x, self.i64([c]), self.i64([c + 1]), self.i64([1])])

    def scalar_min(self, has, idx):
        """min index where has==1 (has [1,1,30,1] or [1,1,1,30], idx matching)."""
        ax = 2 if idx == self.rowidx else 3
        comp = self.nd("Sub", [self.cbig, idx])
        m = self.nd("ReduceMax", [self.nd("Mul", [has, comp])], axes=[ax], keepdims=1)
        return self.nd("Sub", [self.cbig, m])

    def scalar_max(self, has, idx):
        ax = 2 if idx == self.rowidx else 3
        return self.nd("ReduceMax", [self.nd("Mul", [has, idx])], axes=[ax], keepdims=1)

    def eq0(self, x):
        """|x| < 0.5 -> 1.0 else 0.0 (x integer-valued float)."""
        return self.nd("Cast", [self.nd("Less", [self.nd("Abs", [x]), self.half])], to=F)


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "vc0", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _extract_patch(g, src, arow, acol, K):
    """[1,C,K,K] patch of src around scalar anchor (arow,acol), zero outside grid."""
    rowk = g.f([1, 1, K, 1], list(range(K)))
    colk = g.f([1, 1, 1, K], list(range(K)))
    off = K // 2
    # R[k,j] = (j - k == arow - off)
    dR = g.nd("Sub", [g.nd("Sub", [g.colidx, rowk]),
                      g.nd("Sub", [arow, g.f([1, 1, 1, 1], [off])])])
    R = g.eq0(dR)                                             # [1,1,K,30]
    # C[j,k] = (j - k == acol - off)
    dC = g.nd("Sub", [g.nd("Sub", [g.rowidx, colk]),
                      g.nd("Sub", [acol, g.f([1, 1, 1, 1], [off])])])
    Cm = g.eq0(dC)                                            # [1,1,30,K]
    t1 = g.nd("MatMul", [src, Cm])                            # [1,C,30,K]
    return g.nd("MatMul", [R, t1])                            # [1,C,K,K]


def _patch_flood(g, maskpatch, K, kern_arr, iters):
    """bounded flood from patch centre within maskpatch ([1,1,K,K])."""
    seed = np.zeros((K, K), np.float32)
    seed[K // 2, K // 2] = 1.0
    f = g.nd("Mul", [g.f([1, 1, K, K], seed), maskpatch])
    kk = kern_arr.shape[-1]
    kern = g.f([1, 1, kk, kk], kern_arr)
    pad = kk // 2
    for _ in range(iters):
        d = g.nd("Conv", [f, kern], kernel_shape=[kk, kk], pads=[pad] * 4)
        f = g.nd("Mul", [g.clip01(d), maskpatch])
    return f


def _stamp_combine(g, base, markers, patch, K):
    """ConvTranspose stamp of patch at markers; returns (stamped, coverage)."""
    off = K // 2
    st = g.nd("ConvTranspose", [markers, patch], kernel_shape=[K, K],
              pads=[off] * 4)                                 # [1,10,30,30]
    cov = g.clip01(g.nd("ReduceSum", [st], axes=[1], keepdims=1))
    out = g.nd("Add", [g.nd("Mul", [base, g.not_(cov)]), st])
    return out


_CHMASK = np.ones((1, 10, 1, 1), np.float32)
_CHMASK[0, 0, 0, 0] = 0.0


# --------------------------------------------------------------------------- #
# builder: task089                                                            #
# --------------------------------------------------------------------------- #
def _build89():
    g = _G()
    g.consts()
    K, C = K89, K89 // 2
    anyc = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    nonbg = g.nd("Sub", [anyc, g.chan("input", 0)])
    ring = np.ones((3, 3), np.float32)
    ring[1, 1] = 0.0
    nb = g.nd("Conv", [nonbg, g.f([1, 1, 3, 3], ring)], kernel_shape=[3, 3],
              pads=[1, 1, 1, 1])
    nbc = g.clip01(nb)
    iso = g.not_(nbc)
    chmask = g.f([1, 10, 1, 1], _CHMASK)
    k8 = np.ones((3, 3), np.float32)
    cur = "input"
    for c, mirror in ((2, True), (3, False)):
        chc = g.chan("input", c)
        markers = g.nd("Mul", [chc, iso])
        anchor = g.nd("Mul", [chc, nbc])
        hasA = g.clip01(g.nd("ReduceSum", [anchor], axes=[2, 3], keepdims=1))
        ar = g.nd("ReduceSum", [g.nd("Mul", [anchor, g.rowidx])], axes=[2, 3], keepdims=1)
        ac = g.nd("ReduceSum", [g.nd("Mul", [anchor, g.colidx])], axes=[2, 3], keepdims=1)
        praw = _extract_patch(g, "input", ar, ac, K)          # [1,10,K,K]
        npatch = _extract_patch(g, nonbg, ar, ac, K)          # [1,1,K,K]
        fl = _patch_flood(g, npatch, K, k8, T89)
        patch = g.nd("Mul", [g.nd("Mul", [g.nd("Mul", [praw, fl]), chmask]), hasA])
        if mirror:
            patch = g.nd("Slice", [patch, g.i64([K - 1]), g.i64([-K - 1]),
                                   g.i64([3]), g.i64([-1])])
        cur = _stamp_combine(g, cur, markers, patch, K)
    g.nd("Mul", [cur, anyc], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# builder: task080                                                            #
# --------------------------------------------------------------------------- #
def _build80():
    g = _G()
    g.consts()
    K, C = K80, K80 // 2
    anyc = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)
    nonbg = g.nd("Sub", [anyc, g.chan("input", 0)])
    # frontier rows/cols: per-colour count == line length (>0)
    rowcnt = g.nd("ReduceSum", ["input"], axes=[3], keepdims=1)      # [1,10,30,1]
    rowlen = g.nd("ReduceSum", [rowcnt], axes=[1], keepdims=1)       # [1,1,30,1]
    monor = g.nd("Cast", [g.nd("Greater", [rowcnt, g.nd("Sub", [rowlen, g.half])])], to=F)
    fr = g.nd("Mul", [g.nd("ReduceMax", [monor], axes=[1], keepdims=1),
                      g.nd("Cast", [g.nd("Greater", [rowlen, g.half])], to=F)])
    colcnt = g.nd("ReduceSum", ["input"], axes=[2], keepdims=1)      # [1,10,1,30]
    collen = g.nd("ReduceSum", [colcnt], axes=[1], keepdims=1)
    monoc = g.nd("Cast", [g.nd("Greater", [colcnt, g.nd("Sub", [collen, g.half])])], to=F)
    fc = g.nd("Mul", [g.nd("ReduceMax", [monoc], axes=[1], keepdims=1),
                      g.nd("Cast", [g.nd("Greater", [collen, g.half])], to=F)])
    fmask = g.clip01(g.nd("Add", [fr, fc]))                          # [1,1,30,30]
    nonf = g.nd("Mul", [nonbg, g.not_(fmask)])
    # block-neighbour seed: fmask[p+d] & nonf[p+2d]
    seeds = []
    for dr, dc in _K4:
        seeds.append(g.nd("Mul", [g.shift(fmask, -dr, -dc),
                                  g.shift(nonf, -2 * dr, -2 * dc)]))
    seed = g.nd("Mul", [g.clip01(g.nd("Add", [g.nd("Add", [seeds[0], seeds[1]]),
                                              g.nd("Add", [seeds[2], seeds[3]])])), nonf])
    # lonely flood (4-conn within blocks)
    cross = np.zeros((3, 3), np.float32)
    cross[1, :] = 1.0
    cross[:, 1] = 1.0
    ck = g.f([1, 1, 3, 3], cross)
    hasnb = seed
    for _ in range(T80A):
        d = g.nd("Conv", [hasnb, ck], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        hasnb = g.nd("Mul", [g.clip01(d), nonf])
    lonely = g.nd("Mul", [nonf, g.not_(hasnb)])
    keyvec = g.nd("ReduceMax", [g.nd("Mul", ["input", lonely])], axes=[2, 3],
                  keepdims=1)                                        # [1,10,1,1]
    chkey = g.nd("ReduceSum", [g.nd("Mul", ["input", keyvec])], axes=[1], keepdims=1)
    keyb = g.nd("Mul", [chkey, nonf])
    ul = g.nd("Mul", [g.nd("Mul", [keyb, g.not_(g.shift(keyb, 1, 0))]),
                      g.not_(g.shift(keyb, 0, 1))])
    anchor = g.nd("Mul", [ul, hasnb])
    hasA = g.clip01(g.nd("ReduceSum", [anchor], axes=[2, 3], keepdims=1))
    ar = g.nd("ReduceSum", [g.nd("Mul", [anchor, g.rowidx])], axes=[2, 3], keepdims=1)
    ac = g.nd("ReduceSum", [g.nd("Mul", [anchor, g.colidx])], axes=[2, 3], keepdims=1)
    praw = _extract_patch(g, "input", ar, ac, K)
    npatch = _extract_patch(g, nonf, ar, ac, K)
    blk = np.zeros((5, 5), np.float32)
    blk[2, :] = 1.0
    blk[:, 2] = 1.0
    fl = _patch_flood(g, npatch, K, blk, T80B)
    chmask = g.f([1, 10, 1, 1], _CHMASK)
    patch = g.nd("Mul", [g.nd("Mul", [g.nd("Mul", [praw, fl]), chmask]), hasA])
    out = _stamp_combine(g, "input", ul, patch, K)
    g.nd("Mul", [out, anyc], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# builder: task201                                                            #
# --------------------------------------------------------------------------- #
def _build201():
    g = _G()
    g.consts()
    ch4 = g.chan("input", 4)
    rh = g.nd("ReduceMax", [ch4], axes=[3], keepdims=1)              # [1,1,30,1]
    chh = g.nd("ReduceMax", [ch4], axes=[2], keepdims=1)             # [1,1,1,30]
    r0 = g.scalar_min(rh, g.rowidx)
    r1 = g.scalar_max(rh, g.rowidx)
    c0 = g.scalar_min(chh, g.colidx)
    c1 = g.scalar_max(chh, g.colidx)
    bh = g.nd("Add", [g.nd("Sub", [r1, r0]), g.one])
    bw = g.nd("Add", [g.nd("Sub", [c1, c0]), g.one])
    t = g.nd("Cast", [g.nd("Greater", [bh, bw])], to=F)              # [1,1,1,1]
    tn = g.not_(t)
    Xt = g.nd("Transpose", ["input"], perm=[0, 1, 3, 2])
    Xc = g.nd("Add", [g.nd("Mul", [Xt, t]), g.nd("Mul", ["input", tn])])

    def blend(p, q):   # t ? p : q  (scalars)
        return g.nd("Add", [g.nd("Mul", [p, t]), g.nd("Mul", [q, tn])])

    r0c, c0c = blend(c0, r0), blend(r0, c0)
    bhc, bwc = blend(bw, bh), blend(bh, bw)

    # ---- box crop to origin: Sr[r,k]=(k==r+r0c)&(r<bhc); Sc[k,j]=(k==j+c0c)&(j<bwc)
    Sr = g.nd("Mul", [g.eq0(g.nd("Sub", [g.nd("Sub", [g.colidx, g.rowidx]), r0c])),
                      g.nd("Cast", [g.nd("Less", [g.rowidx, bhc])], to=F)])
    Sc = g.nd("Mul", [g.eq0(g.nd("Sub", [g.nd("Sub", [g.rowidx, g.colidx]), c0c])),
                      g.nd("Cast", [g.nd("Less", [g.colidx, bwc])], to=F)])
    boxO = g.nd("MatMul", [Sr, g.nd("MatMul", [Xc, Sc])])            # [1,10,30,30]

    # ---- content mask (nonbg outside box backdrop, canonical coords)
    anyc = g.nd("ReduceSum", [Xc], axes=[1], keepdims=1)
    nonbg = g.nd("Sub", [anyc, g.chan(Xc, 0)])
    rge = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [g.rowidx, g.nd("Sub", [r0c, g.half])])], to=F),
                       g.nd("Cast", [g.nd("Less", [g.rowidx, g.nd("Sub", [g.nd("Add", [r0c, bhc]), g.half])])], to=F)])
    cge = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [g.colidx, g.nd("Sub", [c0c, g.half])])], to=F),
                       g.nd("Cast", [g.nd("Less", [g.colidx, g.nd("Sub", [g.nd("Add", [c0c, bwc]), g.half])])], to=F)])
    bmask = g.nd("Mul", [rge, cge])
    cmask = g.nd("Mul", [nonbg, g.not_(bmask)])
    contX = g.nd("Mul", [Xc, cmask])
    crh = g.nd("ReduceMax", [cmask], axes=[3], keepdims=1)
    cch = g.nd("ReduceMax", [cmask], axes=[2], keepdims=1)
    cr0 = g.scalar_min(crh, g.rowidx)
    cc0 = g.scalar_min(cch, g.colidx)
    cc1 = g.scalar_max(cch, g.colidx)
    cw = g.nd("Add", [g.nd("Sub", [cc1, cc0]), g.one])
    Sr2 = g.eq0(g.nd("Sub", [g.nd("Sub", [g.colidx, g.rowidx]), cr0]))
    Sc2 = g.eq0(g.nd("Sub", [g.nd("Sub", [g.rowidx, g.colidx]), cc0]))
    contO = g.nd("MatMul", [Sr2, g.nd("MatMul", [contX, Sc2])])      # [1,10,30,30]

    # ---- stripe colours: va at box (1,0); vb at box (1,bw-1)
    row1 = g.nd("Slice", [boxO, g.i64([1, 0]), g.i64([2, 30]), g.i64([2, 3])])  # [1,10,1,30]
    va = g.nd("Slice", [row1, g.i64([0]), g.i64([1]), g.i64([3])])              # [1,10,1,1]
    cbsel = g.eq0(g.nd("Sub", [g.colidx, g.nd("Sub", [bwc, g.one])]))           # [1,1,1,30]
    vb = g.nd("ReduceSum", [g.nd("Mul", [row1, cbsel])], axes=[3], keepdims=1)  # [1,10,1,1]

    # ---- gate: leftmost(a) < leftmost(b) in content  <=>  identity
    chA = g.nd("ReduceSum", [g.nd("Mul", [contO, va])], axes=[1], keepdims=1)
    chB = g.nd("ReduceSum", [g.nd("Mul", [contO, vb])], axes=[1], keepdims=1)
    pa = g.nd("ReduceMax", [chA], axes=[2], keepdims=1)              # [1,1,1,30]
    pb = g.nd("ReduceMax", [chB], axes=[2], keepdims=1)
    triu = np.triu(np.ones((W, W), np.float32))                      # [k<=j]
    TR = g.f([W, W], triu)
    ca = g.clip01(g.nd("MatMul", [pa, TR]))
    cb = g.clip01(g.nd("MatMul", [pb, TR]))
    suma = g.nd("ReduceSum", [ca], axes=[3], keepdims=1)
    sumb = g.nd("ReduceSum", [cb], axes=[3], keepdims=1)
    m = g.nd("Cast", [g.nd("Greater", [suma, sumb])], to=F)          # 1 -> identity

    # ---- mirrored content: M[k,j] = (k + j == cw-1)
    Mre = g.eq0(g.nd("Sub", [g.nd("Add", [g.rowidx, g.colidx]),
                             g.nd("Sub", [cw, g.one])]))             # [1,1,30,30]
    contM = g.nd("MatMul", [contO, Mre])
    contF = g.nd("Add", [g.nd("Mul", [contO, m]), g.nd("Mul", [contM, g.not_(m)])])
    contS = g.shift(contF, 1, 1)

    # ---- compose inside box extent, un-transpose
    cov = g.clip01(g.nd("ReduceSum", [contS], axes=[1], keepdims=1))
    ext = g.nd("Mul", [g.nd("Cast", [g.nd("Less", [g.rowidx, bhc])], to=F),
                       g.nd("Cast", [g.nd("Less", [g.colidx, bwc])], to=F)])
    outC = g.nd("Mul", [g.nd("Add", [g.nd("Mul", [boxO, g.not_(cov)]), contS]), ext])
    outT = g.nd("Transpose", [outC], perm=[0, 1, 3, 2])
    g.nd("Add", [g.nd("Mul", [outT, t]), g.nd("Mul", [outC, tn])], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def _gate(prs, fn):
    for a, b in prs:
        o = fn(a)
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(examples):
    prs = [(np.array(e["input"], int), np.array(e["output"], int))
           for e in examples.get("train", []) + examples.get("test", [])]
    if not prs:
        return
    same = all(a.shape == b.shape for a, b in prs)
    if same and any((a != b).any() for a, b in prs):
        if _gate(prs, _solve89):
            yield ("vc0_stamp23", _build89())
        if _gate(prs, _solve80):
            yield ("vc0_blockstamp", _build80())
    if all(b.shape[0] <= a.shape[0] and b.shape[1] <= a.shape[1] for a, b in prs) \
            and any(b.shape != a.shape for a, b in prs):
        if _gate(prs, _solve201):
            yield ("vc0_boxmove", _build201())
