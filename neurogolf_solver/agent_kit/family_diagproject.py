"""DIAGONAL & DIRECTIONAL PROJECTION (rays / shadows / bounding box), opset 10.

Every rule here is ORIGIN-ANCHORED: it is computed per-cell from the (top-left
anchored) real content and gated by the real-cell mask R = ReduceSum(input,
axis=channel) so nothing ever leaks into the zero-padding and the rules
generalise to grids of ANY size.

Two mechanisms
--------------
1.  SHADOW / RAY PROJECTION along a fixed set of directions D (subset of the 8
    axis/diagonal directions).  A background cell takes the colour of the NEAREST
    source cell found by tracing back along -d for some d in D (sources keep
    their own colour).  Special direction sets give the familiar ARC shapes:

        ray_down/up/left/right          a single straight shadow (gravity ray)
        ray_dr/dl/ur/ul                 a single diagonal shadow
        proj_row  = {left,right}        fill the whole row through each cell
        proj_col  = {up,down}           fill the whole column
        proj_cross= {U,D,L,R}           full plus through each cell
        proj_main = {ul,dr}             full main diagonal
        proj_anti = {ur,dl}             full anti diagonal
        proj_xdiag= {ul,ur,dl,dr}       full X (both diagonals)
        proj_star = all 8               full star

    Realisation: a single per-direction Hillis-Steele PREFIX-MAX (offsets
    1,2,4,8,16 -> covers all 30 cells) of an integer "key" field
        key = nbg * (10*pos_d) + value          (pos_d strictly increases along d)
    where value = colour 0..9.  Because 10*pos_d dominates, the prefix-max picks
    the nearest source behind, and the colour is recovered division-free via
        w = floor(K/10) = (int)(K*0.1);  val = K - 10*w
    (verified exact over the whole key range).  Per-direction value fields are
    merged with Max and expanded to a one-hot with a single int Equal against the
    channel indices, then gated by R.  Only [1,1,30,30] intermediates are used
    inside the scan, so the cost stays modest.

2.  BOUNDING BOX of all coloured cells.  rmin/rmax/cmin/cmax come from
    ReduceMax / ReduceMin of position masks; a rectangle (filled or outline) mask
    is formed with Greater/Less and the background cells inside it are recoloured
    to an inferred fill colour while the original content is kept.

Detection mirrors the ONNX semantics exactly (nearest-source-wins, Max merge)
and only emits a candidate when it reproduces EVERY available train/test/arc-gen
pair, so wrong hypotheses are dropped before the grader sees them.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT32 = onnx.TensorProto.INT32
INT64 = onnx.TensorProto.INT64

# 8 directions (dr, dc)
DIRS = {
    "down": (1, 0), "up": (-1, 0), "right": (0, 1), "left": (0, -1),
    "dr": (1, 1), "dl": (1, -1), "ur": (-1, 1), "ul": (-1, -1),
}

# named direction SETS the family understands
RULESETS = {
    "ray_down": ["down"], "ray_up": ["up"], "ray_right": ["right"], "ray_left": ["left"],
    "ray_dr": ["dr"], "ray_dl": ["dl"], "ray_ur": ["ur"], "ray_ul": ["ul"],
    "proj_row": ["left", "right"], "proj_col": ["up", "down"],
    "proj_cross": ["up", "down", "left", "right"],
    "proj_main": ["ul", "dr"], "proj_anti": ["ur", "dl"],
    "proj_xdiag": ["ul", "ur", "dl", "dr"],
    "proj_star": ["up", "down", "left", "right", "ul", "ur", "dl", "dr"],
}

_OFF = 600  # keeps 10*pos_d positive for the whole 30x30 grid


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

    def f(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         np.asarray(vals, np.float32).ravel().tolist()))
        return nm

    def i32(self, dims, vals):
        nm = self.name("k")
        self.inits.append(oh.make_tensor(nm, INT32, list(dims),
                                         [int(v) for v in np.asarray(vals).ravel().tolist()]))
        return nm

    def i64(self, dims, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, list(dims),
                                         [int(v) for v in np.asarray(vals).ravel().tolist()]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# shared front-end tensors                                                    #
# --------------------------------------------------------------------------- #
def _valgrid(g):
    """[1,1,30,30] colour-value field (sum_c c*input_c) via a 1x1 conv (10 params)."""
    wv = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    return g.node("Conv", ["input", wv], kernel_shape=[1, 1], pads=[0, 0, 0, 0])


def _realmask(g):
    """[1,1,30,30] 1 on real cells, 0 on padding."""
    return g.node("ReduceSum", ["input"], axes=[1], keepdims=1)


# --------------------------------------------------------------------------- #
# directional prefix-max (shadow / ray)                                       #
# --------------------------------------------------------------------------- #
def _shift(g, x, sr, sc):
    """Shift content so new[i,j] = x[i-sr, j-sc] (zero fill), staying [.,.,30,30]."""
    pt, pb = max(sr, 0), max(-sr, 0)
    pl, pr = max(sc, 0), max(-sc, 0)
    p = g.node("Pad", [x], mode="constant", value=0.0,
               pads=[0, 0, pt, pl, 0, 0, pb, pr])
    s = g.i64([2], [pb, pr])
    e = g.i64([2], [pb + HEIGHT, pr + WIDTH])
    ax = g.i64([2], [2, 3])
    return g.node("Slice", [p, s, e, ax])


def _prefixmax(g, x, dr, dc):
    cur = x
    for s in (1, 2, 4, 8, 16):
        cur = g.node("Max", [cur, _shift(g, cur, s * dr, s * dc)])
    return cur


def build_shadow(dir_names):
    g = _G()
    valgrid = _valgrid(g)
    R = _realmask(g)
    half = g.f([1, 1, 1, 1], [0.5])
    nbg = g.node("Cast", [g.node("Greater", [valgrid, half])], to=DATA_TYPE)  # presence
    c01 = g.f([1, 1, 1, 1], [0.1])
    c10 = g.f([1, 1, 1, 1], [10.0])

    vals = []
    for nm in dir_names:
        dr, dc = DIRS[nm]
        rowv = g.f([1, 1, HEIGHT, 1], [10 * dr * i for i in range(HEIGHT)])
        colv = g.f([1, 1, 1, WIDTH], [10 * dc * j + _OFF for j in range(WIDTH)])
        w10 = g.node("Add", [rowv, colv])                       # [1,1,30,30]
        key = g.node("Add", [g.node("Mul", [nbg, w10]), valgrid])
        K = _prefixmax(g, key, dr, dc)
        fw = g.node("Cast", [g.node("Cast", [g.node("Mul", [K, c01])], to=INT32)],
                    to=DATA_TYPE)
        vals.append(g.node("Sub", [K, g.node("Mul", [fw, c10])]))   # nearest colour

    V = vals[0]
    for v in vals[1:]:
        V = g.node("Max", [V, v])
    Vint = g.node("Cast", [V], to=INT32)                        # [1,1,30,30]
    idx = g.i32([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    onehot = g.node("Cast", [g.node("Equal", [Vint, idx])], to=DATA_TYPE)  # [1,10,30,30]
    g.node("Mul", [onehot, R], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# bounding box                                                                #
# --------------------------------------------------------------------------- #
def _box_mask(g, pres, rowi, coli, half, one, big, mode):
    """Rectangle (filled / outline) mask [1,1,30,30] of the cells flagged by the
    0/1 presence plane `pres` [1,1,30,30]."""
    notp = g.node("Sub", [one, pres])
    prow = g.node("Mul", [pres, rowi])
    pcol = g.node("Mul", [pres, coli])
    rmax = g.node("ReduceMax", [prow], axes=[2, 3], keepdims=1)            # [1,1,1,1]
    cmax = g.node("ReduceMax", [pcol], axes=[2, 3], keepdims=1)
    rmin = g.node("ReduceMin", [g.node("Add", [prow, g.node("Mul", [notp, big])])],
                  axes=[2, 3], keepdims=1)
    cmin = g.node("ReduceMin", [g.node("Add", [pcol, g.node("Mul", [notp, big])])],
                  axes=[2, 3], keepdims=1)

    def cge(idxt, lo):
        return g.node("Cast", [g.node("Greater", [idxt, g.node("Sub", [lo, half])])],
                      to=DATA_TYPE)

    def cle(idxt, hi):
        return g.node("Cast", [g.node("Less", [idxt, g.node("Add", [hi, half])])],
                      to=DATA_TYPE)

    rowmask = g.node("Mul", [cge(rowi, rmin), cle(rowi, rmax)])            # [1,1,30,1]
    colmask = g.node("Mul", [cge(coli, cmin), cle(coli, cmax)])           # [1,1,1,30]
    box = g.node("Mul", [rowmask, colmask])                               # [1,1,30,30]
    if mode != "outline":
        return box

    def eqc(idxt, v):
        return g.node("Cast", [g.node("Less", [g.node("Abs", [g.node("Sub", [idxt, v])]),
                                               half])], to=DATA_TYPE)
    rb = g.node("Max", [eqc(rowi, rmin), eqc(rowi, rmax)])                # [1,1,30,1]
    cb = g.node("Max", [eqc(coli, cmin), eqc(coli, cmax)])               # [1,1,1,30]
    border = g.node("Clip", [g.node("Add", [rb, cb])], min=0.0, max=1.0)  # [1,1,30,30]
    return g.node("Mul", [box, border])


def build_bbox(mode, F):
    """Single bounding box over ALL colours; background cells inside it -> colour F,
    original content kept."""
    g = _G()
    valgrid = _valgrid(g)
    R = _realmask(g)
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    rowi = g.f([1, 1, HEIGHT, 1], list(range(HEIGHT)))
    coli = g.f([1, 1, 1, WIDTH], list(range(WIDTH)))
    nbg = g.node("Cast", [g.node("Greater", [valgrid, half])], to=DATA_TYPE)  # [1,1,30,30]
    mask = _box_mask(g, nbg, rowi, coli, half, one, big, mode)

    bgmask = g.node("Sub", [R, nbg])                 # 1 at real background cells
    fillcells = g.node("Mul", [mask, bgmask])        # [1,1,30,30]
    vec = g.f([1, CHANNELS, 1, 1],
              [(-1.0 if c == 0 else (1.0 if c == F else 0.0)) for c in range(CHANNELS)])
    g.node("Add", ["input", g.node("Mul", [fillcells, vec])], "output")
    return _model(g)


def build_percolor_bbox(mode):
    """Per-colour bounding box: each colour's cells -> the (filled / outline)
    rectangle of that colour's own bounding box; everything else background."""
    g = _G()
    R = _realmask(g)
    half = g.f([1, 1, 1, 1], [0.5])
    one = g.f([1, 1, 1, 1], [1.0])
    big = g.f([1, 1, 1, 1], [1000.0])
    rowi = g.f([1, 1, HEIGHT, 1], list(range(HEIGHT)))
    coli = g.f([1, 1, 1, WIDTH], list(range(WIDTH)))

    chans = [None] * CHANNELS
    acc = None
    for c in range(1, CHANNELS):
        s = g.i64([1], [c]); e = g.i64([1], [c + 1]); ax = g.i64([1], [1])
        Pc = g.node("Slice", ["input", s, e, ax])                # [1,1,30,30]
        mask = _box_mask(g, Pc, rowi, coli, half, one, big, mode)
        mask = g.node("Mul", [mask, R])
        chans[c] = mask
        acc = mask if acc is None else g.node("Add", [acc, mask])
    chans[0] = g.node("Sub", [R, acc])               # real cells in no box -> background
    g.node("Concat", chans, "output", axis=1)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics for detection)                  #
# --------------------------------------------------------------------------- #
def _shift_np(x, sr, sc):
    h, w = x.shape
    out = np.zeros_like(x)
    rs0, rs1 = max(sr, 0), h - max(-sr, 0)
    cs0, cs1 = max(sc, 0), w - max(-sc, 0)
    if rs0 < rs1 and cs0 < cs1:
        out[rs0:rs1, cs0:cs1] = x[rs0 - sr:rs1 - sr, cs0 - sc:cs1 - sc]
    return out


def _val_dir(a, dr, dc):
    """Nearest-source colour propagated forward along (dr,dc)."""
    cur = a.copy()
    for _ in range(max(a.shape)):
        sh = _shift_np(cur, dr, dc)
        cur = np.where(cur == 0, sh, cur)
    return cur


def _ref_shadow(a, dir_names):
    V = np.zeros_like(a)
    for nm in dir_names:
        V = np.maximum(V, _val_dir(a, *DIRS[nm]))
    return V


def _bbox(a):
    nz = np.argwhere(a != 0)
    if nz.size == 0:
        return None
    r0, c0 = nz.min(0)
    r1, c1 = nz.max(0)
    return int(r0), int(r1), int(c0), int(c1)


def _ref_bbox(a, mode, F):
    bb = _bbox(a)
    if bb is None:
        return None
    r0, r1, c0, c1 = bb
    out = a.copy()
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            if mode == "outline" and not (i == r0 or i == r1 or j == c0 or j == c1):
                continue
            if a[i, j] == 0:
                out[i, j] = F
    return out


def _ref_percolor_bbox(a, mode):
    """Per-colour bbox (filled / outline); returns None if two colours' boxes
    overlap (the ONNX one-hot would then be ambiguous)."""
    h, w = a.shape
    out = np.zeros_like(a)
    cov = np.zeros_like(a)
    for c in range(1, CHANNELS):
        nz = np.argwhere(a == c)
        if nz.size == 0:
            continue
        r0, c0 = nz.min(0)
        r1, c1 = nz.max(0)
        for i in range(int(r0), int(r1) + 1):
            for j in range(int(c0), int(c1) + 1):
                if mode == "outline" and not (i == r0 or i == r1 or j == c0 or j == c1):
                    continue
                cov[i, j] += 1
                out[i, j] = c
    if (cov > 1).any():
        return None
    return out


# --------------------------------------------------------------------------- #
# entry point                                                                 #
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


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):       # every rule preserves shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []

    out = []

    # ---- shadow / ray / line projections ----------------------------------- #
    for name, dirset in RULESETS.items():
        ok = True
        for a, b in prs:
            if not np.array_equal(_ref_shadow(a, dirset), b):
                ok = False
                break
        if ok:
            try:
                out.append((name, build_shadow(dirset)))
            except Exception:
                pass

    # ---- bounding box (filled / outline) ----------------------------------- #
    for mode in ("fill", "outline"):
        # infer the single fill colour from background cells that change inside the box
        fills = set()
        bad = False
        for a, b in prs:
            bb = _bbox(a)
            if bb is None:
                bad = True
                break
            r0, r1, c0, c1 = bb
            for i in range(r0, r1 + 1):
                for j in range(c0, c1 + 1):
                    if mode == "outline" and not (i == r0 or i == r1 or j == c0 or j == c1):
                        continue
                    if a[i, j] == 0 and b[i, j] != 0:
                        fills.add(int(b[i, j]))
        if bad or len(fills) != 1:
            continue
        F = next(iter(fills))
        ok = True
        for a, b in prs:
            r = _ref_bbox(a, mode, F)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            try:
                out.append((f"bbox_{mode}_F{F}", build_bbox(mode, F)))
            except Exception:
                pass

    # ---- per-colour bounding boxes (filled / outline) ---------------------- #
    for mode in ("fill", "outline"):
        ok = True
        for a, b in prs:
            r = _ref_percolor_bbox(a, mode)
            if r is None or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            try:
                out.append((f"pcbbox_{mode}", build_percolor_bbox(mode)))
            except Exception:
                pass

    return out
