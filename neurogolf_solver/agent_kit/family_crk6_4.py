"""crk6_4 -- a small grab-bag of independent crackers for the U[4::6] slice.

Each sub-solver detects its own rule (numpy reference mirroring the ONNX numerics
EXACTLY) and only emits a candidate after reproducing every available
train+test+arc-gen pair, so wrong hypotheses are never scored.

Sub-solvers
-----------
  periodic   : variable-period 2-D periodic completion with a background-0 hole.
               Delegates to ``family_dynperiod`` (already validated machinery).
  cropsym    : crop to the bounding box of the unique colour whose bbox-content is
               left-right (horizontally) symmetric.  Realised with the
               ``family_dyncrop`` MatMul-shift crop, fed a data-dependent content
               mask selected by a per-colour reflection-equality test.
  stamp4     : four single-cell markers laid out on a rectangle (two colours on a
               checkerboard); each marker becomes a 3x3 ring of the OTHER colour
               with its own colour in the centre, and the four rectangle sides are
               joined by dashed (period-2) connector lines of a fixed colour.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

import family_dynperiod as dp
import family_dyncrop as dc
from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
F = DATA_TYPE
H, W = HEIGHT, WIDTH


# --------------------------------------------------------------------------- #
# shared pair helper                                                           #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits=("train", "test", "arc-gen")):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


# =========================================================================== #
# sub-solver: cropsym (task 174 family)                                        #
# =========================================================================== #
def _bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _cropsym_ref(a):
    """Crop to the bbox of the unique colour whose bbox content is h-symmetric."""
    sel = []
    for c in range(1, CHANNELS):
        m = (a == c)
        if not m.any():
            continue
        bb = _bbox(m)
        sub = (a[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1] == c)
        if np.array_equal(sub, sub[:, ::-1]):
            sel.append(c)
    if len(sel) != 1:
        return None
    c = sel[0]
    bb = _bbox(a == c)
    return a[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1]


def _build_cropsym():
    g = dc._G()
    dc._consts(g)
    rowidx, colidx, half, one = g.rowidx, g.colidx, g.half, g.one
    cbig = g.cbig
    # j+k index grid for the reflection selection matrix
    jk = g.nd("Add", [rowidx, colidx])                               # [1,1,30,30]

    content = None
    for c in range(1, CHANNELS):
        Mc = g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])  # [1,1,30,30]
        colhas = g.nd("ReduceMax", [Mc], axes=[2], keepdims=1)        # [1,1,1,30]
        maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colidx])], axes=[3], keepdims=1)
        mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                      [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
        Sc = g.nd("Add", [maxcol, mincol])                           # [1,1,1,1]
        Scol = g.nd("Cast", [g.nd("Less",
                  [g.nd("Abs", [g.nd("Sub", [jk, Sc])]), half])], to=F)   # [1,1,30,30]
        refl = g.nd("MatMul", [Mc, Scol])                            # [1,1,30,30]
        diff = g.nd("ReduceSum", [g.nd("Abs", [g.nd("Sub", [Mc, refl])])],
                    axes=[2, 3], keepdims=1)                          # [1,1,1,1]
        present = g.nd("ReduceSum", [Mc], axes=[2, 3], keepdims=1)    # [1,1,1,1]
        sym = g.nd("Cast", [g.nd("Less", [diff, half])], to=F)
        pres = g.nd("Cast", [g.nd("Greater", [present, half])], to=F)
        sel = g.nd("Mul", [sym, pres])                               # [1,1,1,1]
        term = g.nd("Mul", [sel, Mc])                                # [1,1,30,30]
        content = term if content is None else g.nd("Add", [content, term])

    dc._finish_crop(g, content)
    return dc._model(g)


def _cropsym(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    # crop must genuinely shrink at least one pair
    if not any(b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1] for a, b in det):
        return []

    def ok(plist):
        for a, b in plist:
            o = _cropsym_ref(a)
            if o is None or o.shape != b.shape or not np.array_equal(o, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    try:
        m = _build_cropsym()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("crk6_4_cropsym", m)]


# =========================================================================== #
# sub-solver: stamp4 (task 387 family)                                         #
# =========================================================================== #
def _stamp4_ref(a, conn=5):
    H, Wd = a.shape
    ys, xs = np.where(a != 0)
    if len(ys) != 4:
        return None
    rows = sorted(set(int(y) for y in ys))
    cols = sorted(set(int(x) for x in xs))
    if len(rows) != 2 or len(cols) != 2:
        return None
    r0, r1 = rows
    c0, c1 = cols
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if set((int(y), int(x)) for y, x in zip(ys, xs)) != corners:
        return None
    col_of = {(int(y), int(x)): int(a[y, x]) for y, x in zip(ys, xs)}
    p = col_of[(r0, c0)]
    q = col_of[(r0, c1)]
    if col_of[(r1, c1)] != p or col_of[(r1, c0)] != q or p == q:
        return None
    if p == conn or q == conn:          # keep colour 5 reserved for connectors
        return None
    out = np.zeros((H, Wd), int)
    other = {p: q, q: p}
    for (r, c), m in col_of.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < Wd:
                    out[rr, cc] = m if (dr == 0 and dc == 0) else other[m]
    for r in (r0, r1):
        for c in range(c0 + 1, c1):
            d = min(c - c0, c1 - c)
            if d >= 2 and d % 2 == 0:
                out[r, c] = conn
    for c in (c0, c1):
        for r in range(r0 + 1, r1):
            d = min(r - r0, r1 - r)
            if d >= 2 and d % 2 == 0:
                out[r, c] = conn
    return out


def _build_stamp4(conn=5):
    g = dp._G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    two = g.f([1, 1, 1, 1], [2.0])
    onehalf = g.f([1, 1, 1, 1], [1.5])
    cbig = g.f([1, 1, 1, 1], [1000.0])
    rowidx = g.f([1, 1, H, 1], list(range(H)))
    colidx = g.f([1, 1, 1, W], list(range(W)))
    chmask = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    e_conn = g.f([1, CHANNELS, 1, 1], [1.0 if c == conn else 0.0 for c in range(CHANNELS)])

    realmask = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)            # [1,1,30,30]
    nonbg = g.nd("ReduceSum", [g.nd("Mul", ["input", chmask])], axes=[1], keepdims=1)

    # dilations (3x3 box)
    wd = g.f([1, 1, 3, 3], [1.0] * 9)
    D_all = g.nd("Conv", [nonbg, wd], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    wdep = g.f([CHANNELS, 1, 3, 3], [1.0] * (CHANNELS * 9))
    Dch = g.nd("Conv", ["input", wdep], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
               group=CHANNELS)                                              # [1,10,30,30]
    otherD = g.nd("Sub", [D_all, Dch])                                      # [1,10,30,30]
    ringpos = g.nd("Cast", [g.nd("Greater", [otherD, half])], to=F)
    notmarker = g.nd("Sub", [one, nonbg])                                   # [1,1,30,30]
    present = g.nd("Cast", [g.nd("Greater",
                  [g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1), half])], to=F)
    ring = g.nd("Mul", [g.nd("Mul", [g.nd("Mul", [ringpos, notmarker]), present]), chmask])

    # connectors
    rowmask = g.nd("ReduceMax", [nonbg], axes=[3], keepdims=1)              # [1,1,30,1]
    colmask = g.nd("ReduceMax", [nonbg], axes=[2], keepdims=1)              # [1,1,1,30]
    c1 = g.nd("ReduceMax", [g.nd("Mul", [colmask, colidx])], axes=[3], keepdims=1)
    c0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [colmask, g.nd("Sub", [cbig, colidx])])], axes=[3], keepdims=1)])
    r1 = g.nd("ReduceMax", [g.nd("Mul", [rowmask, rowidx])], axes=[2], keepdims=1)
    r0 = g.nd("Sub", [cbig, g.nd("ReduceMax",
              [g.nd("Mul", [rowmask, g.nd("Sub", [cbig, rowidx])])], axes=[2], keepdims=1)])

    def edge(idx, lo, hi):
        dlo = g.nd("Sub", [idx, lo])
        dhi = g.nd("Sub", [hi, idx])
        between = g.nd("Mul", [g.nd("Cast", [g.nd("Greater", [dlo, half])], to=F),
                              g.nd("Cast", [g.nd("Greater", [dhi, half])], to=F)])
        dmin = g.nd("Min", [dlo, dhi])
        ge2 = g.nd("Cast", [g.nd("Greater", [dmin, onehalf])], to=F)
        evenm = g.nd("Cast", [g.nd("Less", [g.nd("Mod", [dmin, two], fmod=1), half])], to=F)
        return g.nd("Mul", [g.nd("Mul", [between, ge2]), evenm])

    cond_c = edge(colidx, c0, c1)                                          # [1,1,1,30]
    cond_r = edge(rowidx, r0, r1)                                          # [1,1,30,1]
    horiz = g.nd("Mul", [rowmask, cond_c])                                 # [1,1,30,30]
    vert = g.nd("Mul", [cond_r, colmask])                                  # [1,1,30,30]
    conn_mask = g.nd("Max", [horiz, vert])                                 # [1,1,30,30]

    colored = g.nd("Add", [g.nd("Mul", ["input", chmask]),
                           g.nd("Add", [ring, g.nd("Mul", [conn_mask, e_conn])])])
    painted = g.nd("ReduceSum", [colored], axes=[1], keepdims=1)
    out0 = g.nd("Mul", [realmask, g.nd("Sub", [one, painted])])            # [1,1,30,30]
    g.nd("Add", [colored, g.nd("Mul", [out0, e0])], "output")
    return dp._model(g)


def _stamp4(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    if all(np.array_equal(a, b) for a, b in allp):
        return []

    def ok(plist):
        for a, b in plist:
            if a.shape != b.shape:
                return False
            o = _stamp4_ref(a)
            if o is None or not np.array_equal(o, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    try:
        m = _build_stamp4()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("crk6_4_stamp4", m)]


# =========================================================================== #
# sub-solver: fold308 (task 308 family -- overlay symmetric colour orbits)      #
# =========================================================================== #
def _fold308_ref(a):
    vals, counts = np.unique(a, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    colors = [int(c) for c in vals if c != bg]
    if not colors:
        return None
    info = {}
    hr = hc = 0
    for c in colors:
        ys, xs = np.where(a == c)
        r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h % 2 == 0 or w % 2 == 0:
            return None
        info[c] = (r0, r1, c0, c1, (r0 + r1) // 2, (c0 + c1) // 2)
        hr = max(hr, (h - 1) // 2)
        hc = max(hc, (w - 1) // 2)
    H, Wd = 2 * hr + 1, 2 * hc + 1
    out = np.full((H, Wd), bg)
    cnt = np.zeros((H, Wd), int)
    for c in colors:
        r0, r1, c0, c1, cr, cc = info[c]
        for r in range(r0, r1 + 1):
            for cx in range(c0, c1 + 1):
                if a[r, cx] == c:
                    oi, oj = r - cr + hr, cx - cc + hc
                    if not (0 <= oi < H and 0 <= oj < Wd):
                        return None
                    out[oi, oj] = c
                    cnt[oi, oj] += 1
    if (cnt > 1).any():
        return None
    return out


def _build_fold308():
    g = dp._G()
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    cbig = g.f([1, 1, 1, 1], [1000.0])
    BIG = g.f([1, 1, 1, 1], [1.0e6])
    rowg = g.f([1, 1, H, 1], list(range(H)))           # i / row
    colg = g.f([1, 1, 1, W], list(range(W)))           # j / col
    imk = g.nd("Sub", [rowg, colg])                    # [1,1,30,30] (i-k)
    jml = g.nd("Sub", [colg, rowg])                    # [1,1,30,30] (j-l)
    chidx = g.nm("i")
    g.inits.append(oh.make_tensor(chidx, INT64, [1, CHANNELS, 1, 1], list(range(CHANNELS))))
    ones_c = g.f([1, CHANNELS, 1, 1], [1.0] * CHANNELS)

    counts = g.nd("ReduceSum", ["input"], axes=[2, 3], keepdims=1)          # [1,10,1,1]
    amax = g.nd("ArgMax", [counts], axis=1, keepdims=1)                     # int64 [1,1,1,1]
    bg_oh = g.nd("Cast", [g.nd("Equal", [amax, chidx])], to=F)              # [1,10,1,1]
    notbg = g.nd("Sub", [ones_c, bg_oh])
    present = g.nd("Cast", [g.nd("Greater", [counts, half])], to=F)
    active = g.nd("Mul", [present, notbg])                                  # [1,10,1,1]

    rowhas = g.nd("ReduceMax", ["input"], axes=[3], keepdims=1)             # [1,10,30,1]
    colhas = g.nd("ReduceMax", ["input"], axes=[2], keepdims=1)            # [1,10,1,30]
    maxrow = g.nd("ReduceMax", [g.nd("Mul", [rowhas, rowg])], axes=[2], keepdims=1)
    minrow = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [rowhas, g.nd("Sub", [cbig, rowg])])], axes=[2], keepdims=1)])
    maxcol = g.nd("ReduceMax", [g.nd("Mul", [colhas, colg])], axes=[3], keepdims=1)
    mincol = g.nd("Sub", [cbig, g.nd("ReduceMax",
                  [g.nd("Mul", [colhas, g.nd("Sub", [cbig, colg])])], axes=[3], keepdims=1)])
    cr = g.nd("Mul", [g.nd("Add", [minrow, maxrow]), half])                 # [1,10,1,1]
    cc = g.nd("Mul", [g.nd("Add", [mincol, maxcol]), half])
    hr_c = g.nd("Mul", [g.nd("Sub", [maxrow, minrow]), half])
    hc_c = g.nd("Mul", [g.nd("Sub", [maxcol, mincol]), half])
    pen = g.nd("Mul", [g.nd("Sub", [ones_c, active]), BIG])
    hr = g.nd("ReduceMax", [g.nd("Sub", [hr_c, pen])], axes=[1], keepdims=1)   # [1,1,1,1]
    hc = g.nd("ReduceMax", [g.nd("Sub", [hc_c, pen])], axes=[1], keepdims=1)
    twohr = g.nd("Add", [hr, hr])
    twohc = g.nd("Add", [hc, hc])
    shift_r = g.nd("Sub", [hr, cr])                                         # [1,10,1,1]
    shift_c = g.nd("Sub", [hc, cc])

    colored = None
    for c in range(CHANNELS):
        Mc = g.nd("Slice", ["input", g.i64([c]), g.i64([c + 1]), g.i64([1])])  # [1,1,30,30]
        sr = g.nd("Slice", [shift_r, g.i64([c]), g.i64([c + 1]), g.i64([1])])  # [1,1,1,1]
        sc = g.nd("Slice", [shift_c, g.i64([c]), g.i64([c + 1]), g.i64([1])])
        ac = g.nd("Slice", [active, g.i64([c]), g.i64([c + 1]), g.i64([1])])
        Srow = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [imk, sr])]), half])], to=F)
        Scol = g.nd("Cast", [g.nd("Less", [g.nd("Abs", [g.nd("Sub", [jml, sc])]), half])], to=F)
        sh = g.nd("MatMul", [Srow, g.nd("MatMul", [Mc, Scol])])             # [1,1,30,30]
        gated = g.nd("Mul", [sh, ac])                                      # [1,1,30,30]
        ec = g.f([1, CHANNELS, 1, 1], [1.0 if k == c else 0.0 for k in range(CHANNELS)])
        term = g.nd("Mul", [gated, ec])                                    # [1,10,30,30]
        colored = term if colored is None else g.nd("Add", [colored, term])

    painted = g.nd("ReduceSum", [colored], axes=[1], keepdims=1)            # [1,1,30,30]
    within_r = g.nd("Cast", [g.nd("Less", [rowg, g.nd("Add", [twohr, half])])], to=F)  # [1,1,30,1]
    within_c = g.nd("Cast", [g.nd("Less", [colg, g.nd("Add", [twohc, half])])], to=F)  # [1,1,1,30]
    within = g.nd("Mul", [within_r, within_c])                             # [1,1,30,30]
    outbg = g.nd("Mul", [within, g.nd("Sub", [one, painted])])             # [1,1,30,30]
    g.nd("Add", [colored, g.nd("Mul", [outbg, bg_oh])], "output")
    return dp._model(g)


def _fold308(ex):
    det = _pairs(ex, ("train", "test"))
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    if not det or not allp:
        return []
    # this family always shrinks to a small folded figure
    if not any(b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1] for a, b in det):
        return []

    def ok(plist):
        for a, b in plist:
            o = _fold308_ref(a)
            if o is None or o.shape != b.shape or not np.array_equal(o, b):
                return False
        return True

    if not ok(det) or not ok(allp):
        return []
    try:
        m = _build_fold308()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return []
    return [("crk6_4_fold308", m)]


# =========================================================================== #
# entry point                                                                  #
# =========================================================================== #
def candidates(ex):
    out = []
    try:
        out += dp.candidates(ex)            # periodic completion (61, 110, ...)
    except Exception:
        pass
    for fn in (_cropsym, _stamp4, _fold308):
        try:
            out += fn(ex)
        except Exception:
            pass
    return out
