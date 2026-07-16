"""family_vc2_2 — two verifier-translated static ONNX solvers (retry wave).

task238 / verify_9aec4887  (bg=0): a hollow rectangular FRAME sits somewhere (4
    solid single-colour edges T/B/L/R, empty interior); a separate mono-colour
    SHAPE sits elsewhere.  Output = the frame with the shape dropped at offset
    (1,1) inside it, every shape cell RECOLOURED to the colour of its nearest
    frame edge (manhattan); ties keep the shape colour.  Because the frame is an
    axis-aligned rectangle the "nearest object" Voronoi collapses to position
    distances d_top=r, d_bot=(H-1)-r, d_left=c, d_right=(W-1)-c in cropped frame
    coords -> argmin with a tie count.  ONNX: per-colour line test (nrows==1 /
    ncols==1) splits edges from the shape; dyncrop MatMul crops the frame to the
    origin (ring + empty interior); a second MatMul + Pad(1,1) drops the shape
    mask in place; edge colours read by masked ReduceMax of the cropped value img;
    Min/Equal distance planes give the recolour; value-image one-hot + ingrid clip.

task170 / verify_6ecd11f4  (bg=0): a big MONO-colour blob is a scaled (sr x sc)
    up-blow of an NxN block grid; a small solid multi-colour PALETTE (3x3/4x4)
    sits elsewhere.  Output = the palette, each cell KEPT iff the blob block at
    that grid position is present, else 0.  ONNX: palette found by flooding
    "differing-neighbour" seed cells through the non-bg mask (12 dilate steps);
    blob = non-bg minus palette; strided-sample MatMuls (Drow/Dcol built from
    blob-origin + floor(bh/ph),floor(bw/pw)) read block presence at the block
    top-left corners; palette dyncrop gives the colours; presence masks it;
    value-image one-hot + ingrid clip.

Both detection gates are numpy mirrors of the ONNX numerics, validated 266/266
against train+test+arc-gen expected outputs.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
INT64 = onnx.TensorProto.INT64
H30 = 30


# --------------------------------------------------------------------------- #
# graph accumulator + helpers                                                  #
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
        self.inits.append(oh.make_tensor(
            n, F, list(dims), [float(v) for v in np.asarray(vals, np.float64).ravel()]))
        return n

    def f1(self, v):
        return self.f([1, 1, 1, 1], [v])

    def i64(self, vals):
        n = self.nm("i")
        self.inits.append(oh.make_tensor(n, INT64, [len(vals)], [int(v) for v in vals]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g, name):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    used = {i for n in g.nodes for i in n.input}
    inits = [t for t in g.inits if t.name in used]
    m = oh.make_model(oh.make_graph(g.nodes, name, [x], [y], inits),
                      ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(m, full_check=True)
    return m


def _consts(g):
    g.rowidx = g.f([1, 1, H30, 1], list(range(H30)))
    g.colidx = g.f([1, 1, 1, H30], list(range(H30)))
    g.half = g.f([1, 1, 1, 1], [0.5])
    g.one = g.f([1, 1, 1, 1], [1.0])
    g.cbig = g.f([1, 1, 1, 1], [1000.0])


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _eqm(g, a, b):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


def _slice_ch(g, c):
    return g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])


def _value_img(g):
    w = g.f([1, 10, 1, 1], list(range(10)))
    return g.nd("Conv", ["input", w], kernel_shape=[1, 1])


def _value_of(g, x):
    w = g.f([1, 10, 1, 1], list(range(10)))
    return g.nd("Conv", [x, w], kernel_shape=[1, 1])


def _minmax(g, has, idx, axis):
    mx = g.nd("ReduceMax", [g.nd("Mul", [has, idx])], axes=[axis], keepdims=1)
    inv = g.nd("Mul", [has, g.nd("Sub", [g.cbig, idx])])
    mn = g.nd("Sub", [g.cbig, g.nd("ReduceMax", [inv], axes=[axis], keepdims=1)])
    return mn, mx


def _rowspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[3], keepdims=1)
    return _minmax(g, has, g.rowidx, 2)


def _colspan(g, mask):
    has = g.nd("ReduceMax", [mask], axes=[2], keepdims=1)
    return _minmax(g, has, g.colidx, 3)


def _shift(g, x, dr, dc):
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0,
             pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st = g.i64([max(-dr, 0), max(-dc, 0)])
    en = g.i64([max(-dr, 0) + H30, max(-dc, 0) + H30])
    ax = g.i64([2, 3])
    return g.nd("Slice", [p, st, en, ax])


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_dims(g, r0, r1, c0, c1):
    h = g.nd("Add", [g.nd("Sub", [r1, r0]), g.one])
    w = g.nd("Add", [g.nd("Sub", [c1, c0]), g.one])
    return h, w


def _crop_matmul(g, src, r0, c0, h, w):
    """Crop region [r0:r0+h, c0:c0+w] of src to the top-left origin."""
    Rrow = g.nd("Mul", [_eqm(g, g.colidx, g.nd("Add", [g.rowidx, r0])),
                        _lt(g, g.rowidx, g.nd("Sub", [h, g.half]))])
    Rcol = g.nd("Mul", [_eqm(g, g.rowidx, g.nd("Add", [g.colidx, c0])),
                        _lt(g, g.colidx, g.nd("Sub", [w, g.half]))])
    return g.nd("MatMul", [Rrow, g.nd("MatMul", [src, Rcol])])


# =========================================================================== #
# task238 — 9aec4887                                                          #
# =========================================================================== #
def build_238():
    g = _G()
    _consts(g)
    V = _value_img(g)

    line_terms, shape_terms = [], []
    for c in range(1, 10):
        Mc = _slice_ch(g, c)
        pres = g.nd("ReduceMax", [Mc], axes=[2, 3], keepdims=1)
        nrows = g.nd("ReduceSum", [g.nd("ReduceMax", [Mc], axes=[3], keepdims=1)],
                     axes=[2], keepdims=1)
        ncols = g.nd("ReduceSum", [g.nd("ReduceMax", [Mc], axes=[2], keepdims=1)],
                     axes=[3], keepdims=1)
        p = _gt(g, pres, g.half)
        r_ge2 = _gt(g, nrows, g.f1(1.5))
        c_ge2 = _gt(g, ncols, g.f1(1.5))
        r_le1 = _lt(g, nrows, g.f1(1.5))
        c_le1 = _lt(g, ncols, g.f1(1.5))
        is_hor = g.nd("Mul", [g.nd("Mul", [p, r_le1]), c_ge2])
        is_ver = g.nd("Mul", [g.nd("Mul", [p, c_le1]), r_ge2])
        is_shape = g.nd("Mul", [g.nd("Mul", [p, r_ge2]), c_ge2])
        is_line = g.nd("Add", [is_hor, is_ver])
        line_terms.append(g.nd("Mul", [is_line, Mc]))
        shape_terms.append(g.nd("Mul", [is_shape, Mc]))
    frame_mask = g.nd("Sum", line_terms)
    shape_mask = g.nd("Sum", shape_terms)
    shapecolor = g.nd("ReduceMax", [g.nd("Mul", [V, shape_mask])], axes=[2, 3], keepdims=1)

    fr0, fr1 = _rowspan(g, frame_mask)
    fc0, fc1 = _colspan(g, frame_mask)
    Hf, Wf = _bbox_dims(g, fr0, fr1, fc0, fc1)

    RV = _crop_matmul(g, V, fr0, fc0, Hf, Wf)

    eqrow0 = _eqm(g, g.rowidx, g.f1(0.0))
    eqcol0 = _eqm(g, g.colidx, g.f1(0.0))
    eqrowB = _eqm(g, g.rowidx, g.nd("Sub", [Hf, g.one]))
    eqcolR = _eqm(g, g.colidx, g.nd("Sub", [Wf, g.one]))
    T = g.nd("ReduceMax", [g.nd("Mul", [RV, eqrow0])], axes=[2, 3], keepdims=1)
    B = g.nd("ReduceMax", [g.nd("Mul", [RV, eqrowB])], axes=[2, 3], keepdims=1)
    L = g.nd("ReduceMax", [g.nd("Mul", [RV, eqcol0])], axes=[2, 3], keepdims=1)
    R = g.nd("ReduceMax", [g.nd("Mul", [RV, eqcolR])], axes=[2, 3], keepdims=1)

    sr0, sr1 = _rowspan(g, shape_mask)
    sc0, sc1 = _colspan(g, shape_mask)
    sh_h, sh_w = _bbox_dims(g, sr0, sr1, sc0, sc1)
    SMc = _crop_matmul(g, shape_mask, sr0, sc0, sh_h, sh_w)
    SP = _shift(g, SMc, 1, 1)

    Dtop = g.rowidx
    Dbot = g.nd("Sub", [g.nd("Sub", [Hf, g.one]), g.rowidx])
    Dleft = g.colidx
    Dright = g.nd("Sub", [g.nd("Sub", [Wf, g.one]), g.colidx])
    Dmin = g.nd("Min", [Dtop, Dbot, Dleft, Dright])
    eqt = _eqm(g, Dtop, Dmin)
    eqb = _eqm(g, Dbot, Dmin)
    eql = _eqm(g, Dleft, Dmin)
    eqr = _eqm(g, Dright, Dmin)
    cnt = g.nd("Sum", [eqt, eqb, eql, eqr])
    edgecolor = g.nd("Sum", [g.nd("Mul", [eqt, T]), g.nd("Mul", [eqb, B]),
                             g.nd("Mul", [eql, L]), g.nd("Mul", [eqr, R])])
    tie = _gt(g, cnt, g.f1(1.5))
    rc = g.nd("Add", [g.nd("Mul", [edgecolor, g.nd("Sub", [g.one, tie])]),
                      g.nd("Mul", [shapecolor, tie])])
    RCval = g.nd("Mul", [SP, rc])
    OUTv = g.nd("Add", [g.nd("Mul", [RV, g.nd("Sub", [g.one, SP])]), RCval])

    ingrid = g.nd("Mul", [_lt(g, g.rowidx, g.nd("Sub", [Hf, g.half])),
                          _lt(g, g.colidx, g.nd("Sub", [Wf, g.half]))])
    OH = _eqm(g, OUTv, g.f([1, 10, 1, 1], list(range(10))))
    g.nd("Mul", [OH, ingrid], "output")
    return _model(g, "vc2_238")


# =========================================================================== #
# task170 — 6ecd11f4                                                          #
# =========================================================================== #
def build_170():
    g = _G()
    _consts(g)
    V = _value_img(g)
    N = _gt(g, V, g.half)

    seeds = []
    for dr, dc in _DIRS4:
        Vsh = _shift(g, V, dr, dc)
        nb = _gt(g, Vsh, g.half)
        diff = _gt(g, g.nd("Abs", [g.nd("Sub", [V, Vsh])]), g.half)
        seeds.append(g.nd("Mul", [g.nd("Mul", [N, nb]), diff]))
    pal = g.nd("Max", seeds)
    for _ in range(12):
        sh = [_shift(g, pal, dr, dc) for dr, dc in _DIRS4]
        pal = g.nd("Mul", [g.nd("Max", [pal] + sh), N])
    blob = g.nd("Mul", [N, g.nd("Sub", [g.one, pal])])

    pr0, pr1 = _rowspan(g, pal)
    pc0, pc1 = _colspan(g, pal)
    ph, pw = _bbox_dims(g, pr0, pr1, pc0, pc1)
    br0, br1 = _rowspan(g, blob)
    bc0, bc1 = _colspan(g, blob)
    bh, bw = _bbox_dims(g, br0, br1, bc0, bc1)
    bcv = g.nd("ReduceMax", [g.nd("Mul", [V, blob])], axes=[2, 3], keepdims=1)
    sr = g.nd("Floor", [g.nd("Div", [bh, ph])])
    sc = g.nd("Floor", [g.nd("Div", [bw, pw])])

    PV = _crop_matmul(g, V, pr0, pc0, ph, pw)

    Dcol = g.nd("Mul", [_eqm(g, g.rowidx, g.nd("Add", [bc0, g.nd("Mul", [g.colidx, sc])])),
                        _lt(g, g.colidx, g.nd("Sub", [pw, g.half]))])
    Drow = g.nd("Mul", [_eqm(g, g.colidx, g.nd("Add", [br0, g.nd("Mul", [g.rowidx, sr])])),
                        _lt(g, g.rowidx, g.nd("Sub", [ph, g.half]))])
    sampled = g.nd("MatMul", [Drow, g.nd("MatMul", [V, Dcol])])
    pres = _eqm(g, sampled, bcv)
    OUTv = g.nd("Mul", [PV, pres])

    ingrid = g.nd("Mul", [_lt(g, g.rowidx, g.nd("Sub", [ph, g.half])),
                          _lt(g, g.colidx, g.nd("Sub", [pw, g.half]))])
    OH = _eqm(g, OUTv, g.f([1, 10, 1, 1], list(range(10))))
    g.nd("Mul", [OH, ingrid], "output")
    return _model(g, "vc2_170")


# =========================================================================== #
# numpy references (mirror the ONNX numerics exactly)                          #
# =========================================================================== #
def _ref238(a):
    a = np.array(a, int)
    H, W = a.shape
    hor, ver, other = [], [], []
    for c in range(1, 10):
        m = (a == c)
        if not m.any():
            continue
        ys, xs = np.where(m)
        nr = len(set(ys.tolist()))
        nc = len(set(xs.tolist()))
        if nr == 1 and nc > 1:
            hor.append((int(ys[0]), c))
        elif nc == 1 and nr > 1:
            ver.append((int(xs[0]), c))
        else:
            other.append(c)
    if len(hor) != 2 or len(ver) != 2 or len(other) != 1:
        return None
    hor.sort()
    ver.sort()
    r0, T = hor[0]
    r1, B = hor[-1]
    c0, L = ver[0]
    c1, R = ver[-1]
    Hf, Wf = r1 - r0 + 1, c1 - c0 + 1
    if Hf < 3 or Wf < 3:
        return None
    if not (a[r0 + 1:r1, c0 + 1:c1] == 0).all():
        return None
    sc = other[0]
    sm = (a == sc)
    sys, sxs = np.where(sm)
    sr0, scc0 = sys.min(), sxs.min()
    out = np.zeros((Hf, Wf), int)
    out[0, 1:Wf - 1] = T
    out[Hf - 1, 1:Wf - 1] = B
    out[1:Hf - 1, 0] = L
    out[1:Hf - 1, Wf - 1] = R
    cols = [T, B, L, R]
    for r, c in zip(sys, sxs):
        rr, cc = r - sr0 + 1, c - scc0 + 1
        if not (1 <= rr <= Hf - 2 and 1 <= cc <= Wf - 2):
            return None
        ds = [rr, (Hf - 1) - rr, cc, (Wf - 1) - cc]
        m = min(ds)
        win = [i for i in range(4) if ds[i] == m]
        out[rr, cc] = cols[win[0]] if len(win) == 1 else sc
    return out


def _shift_np(m, dr, dc):
    H, W = m.shape
    o = np.zeros_like(m)
    r0, r1 = max(dr, 0), H + min(dr, 0)
    c0, c1 = max(dc, 0), W + min(dc, 0)
    o[r0:r1, c0:c1] = m[max(-dr, 0):H + min(-dr, 0), max(-dc, 0):W + min(-dc, 0)]
    return o


def _ref170(a):
    a = np.array(a, int)
    H, W = a.shape
    nonbg = (a != 0)
    if nonbg.sum() == 0:
        return None
    seed = np.zeros((H, W), bool)
    for dr, dc in _DIRS4:
        sh = _shift_np(a, dr, dc)
        seed |= nonbg & (sh != 0) & (sh != a)
    pal = seed.copy()
    for _ in range(12):
        nxt = pal.copy()
        for dr, dc in _DIRS4:
            nxt |= _shift_np(pal.astype(int), dr, dc).astype(bool)
        pal = nxt & nonbg
    if pal.sum() == 0:
        return None
    ys, xs = np.where(pal)
    pr0, pr1, pc0, pc1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = pr1 - pr0 + 1, pc1 - pc0 + 1
    P = a[pr0:pr1 + 1, pc0:pc1 + 1].copy()
    blob = nonbg & ~pal
    if blob.sum() == 0:
        return None
    bcnt = np.array([(blob & (a == c)).sum() for c in range(10)])
    bcnt[0] = -1
    bc = int(bcnt.argmax())
    bys, bxs = np.where(blob)
    br0, br1, bc0, bc1 = bys.min(), bys.max(), bxs.min(), bxs.max()
    bh, bw = br1 - br0 + 1, bc1 - bc0 + 1
    sr, scl = bh // ph, bw // pw
    if sr < 1 or scl < 1:
        return None
    out = np.zeros((ph, pw), int)
    for i in range(ph):
        for j in range(pw):
            ri, ci = br0 + i * sr, bc0 + j * scl
            if ri >= H or ci >= W:
                return None
            out[i, j] = P[i, j] if a[ri, ci] == bc else 0
    return out


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


def _matches(prs, fn):
    if not prs:
        return False
    for a, b in prs:
        try:
            o = fn(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    if _matches(prs, _ref238):
        try:
            out.append(("vc2_238", build_238()))
        except Exception:
            pass
    if _matches(prs, _ref170):
        try:
            out.append(("vc2_170", build_170()))
        except Exception:
            pass
    return out
