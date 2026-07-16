"""DIAGONAL-SPECIFIC transforms (opset 10).

This family collects the transforms whose defining structure is a *diagonal* of
the grid (main diagonal = cells with row==col, anti diagonal = row+col==const):

  reflect_main        Reflect across the MAIN diagonal == matrix transpose.
                      ``Transpose`` perm=[0,1,3,2] swaps the H/W axes and keeps
                      the top-left origin for grids of ANY size (the real
                      [0:H,0:W] block maps to [0:W,0:H], padding stays zero), so
                      it is fully origin-safe.  0 params, 0 intermediate.

  reflect_main_map    Transpose followed by a per-colour recolour (1x1 Conv).

  reflect_anti        Reflect across the ANTI diagonal == anti-transpose,
                      ``B[i,j] = A[H-1-j, W-1-i]`` (== rot180 then transpose).
                      rot180 is NOT origin-safe for grids < 30, so this is only
                      emitted when the input size is CONSTANT across every split:
                      we then crop to the exact HxW, reverse both axes (Slice
                      step=-1), Transpose, and Pad the WxH result back to the
                      top-left of a 30x30 grid.

  reflect_anti_map    Anti-transpose followed by a per-colour recolour.

  diag rays/shadows   From every coloured seed, shoot a diagonal RAY (shadow):
                      a background cell takes the colour of the nearest source
                      traced back along -d, for d in a diagonal direction set:
                        ray_dr/dl/ur/ul   one diagonal shadow
                        diag_main = {ul,dr}   full main-diagonal line per seed
                        diag_anti = {ur,dl}   full anti-diagonal line per seed
                        diag_x    = {ul,ur,dl,dr}  the full X through each seed
                      Realised origin-safe via a Hillis-Steele prefix-max of an
                      integer key field (offsets 1,2,4,8,16 cover all 30 cells),
                      merged with Max and gated by the real-cell mask so nothing
                      leaks into the zero-padding.

  keep_main / keep_anti
                      EXTRACT the diagonal: keep the colours lying on the main
                      (any size) / anti (constant size) diagonal, every other
                      real cell becomes background.

Detection mirrors the ONNX semantics exactly and only emits a candidate when it
reproduces EVERY available train/test/arc-gen pair (the grader's gate), so wrong
hypotheses are dropped before scoring.  Geometry-dependent rules (anti diagonal)
are emitted only when the grid size is constant across ALL splits.
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
_NEG = -(1 << 31)            # full-axis reverse Slice sentinel
_OFF = 600                   # keeps 10*pos positive over the whole 30x30 grid

# diagonal directions (dr, dc) -- content travels this way
DIRS = {"dr": (1, 1), "dl": (1, -1), "ur": (-1, 1), "ul": (-1, -1)}

# named diagonal direction SETS understood by the family
RAYSETS = {
    "ray_dr": ["dr"], "ray_dl": ["dl"], "ray_ur": ["ur"], "ray_ul": ["ul"],
    "diag_main": ["ul", "dr"], "diag_anti": ["ur", "dl"],
    "diag_x": ["ul", "ur", "dl", "dr"],
}


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


def _onehot(k):
    return [1.0 if c == k else 0.0 for c in range(CHANNELS)]


# --------------------------------------------------------------------------- #
# reflections                                                                 #
# --------------------------------------------------------------------------- #
def build_transpose():
    """Reflect across the main diagonal (transpose). Origin-safe, 0 params."""
    g = _G()
    g.node("Transpose", ["input"], "output", perm=[0, 1, 3, 2])
    return _model(g)


def _recolor_conv(g, src, color_map, out):
    """1x1 Conv recolour: output[:,o] = sum_i [color_map[i]==o] input[:,i]."""
    W = np.zeros((CHANNELS, CHANNELS, 1, 1), np.float32)
    for i, o in enumerate(color_map):
        W[o, i, 0, 0] = 1.0
    wt = g.f([CHANNELS, CHANNELS, 1, 1], W)
    return g.node("Conv", [src, wt], out, kernel_shape=[1, 1], pads=[0, 0, 0, 0])


def build_transpose_map(color_map):
    g = _G()
    t = g.node("Transpose", ["input"], perm=[0, 1, 3, 2])
    _recolor_conv(g, t, color_map, "output")
    return _model(g)


def build_antitranspose(h, w, color_map=None):
    """Reflect across the anti diagonal for a CONSTANT HxW input.
    B[i,j] = A[H-1-j, W-1-i]; realised as crop -> reverse(both) -> Transpose -> Pad."""
    g = _G()
    crop = g.node("Slice", ["input", g.i64([2], [0, 0]), g.i64([2], [h, w]),
                            g.i64([2], [2, 3])])                       # [1,10,h,w]
    rev = g.node("Slice", [crop, g.i64([2], [h - 1, w - 1]),
                           g.i64([2], [_NEG, _NEG]), g.i64([2], [2, 3]),
                           g.i64([2], [-1, -1])])                       # a[::-1,::-1]
    tr = g.node("Transpose", [rev], perm=[0, 1, 3, 2])                  # [1,10,w,h]
    if color_map is not None:
        tr = _recolor_conv(g, tr, color_map, None)
    if w == HEIGHT and h == WIDTH:
        g.node("Identity", [tr], "output")
    else:
        g.node("Pad", [tr], "output", mode="constant", value=0.0,
               pads=[0, 0, 0, 0, 0, 0, HEIGHT - w, WIDTH - h])
    return _model(g)


# --------------------------------------------------------------------------- #
# diagonal rays / shadows (prefix-max)                                        #
# --------------------------------------------------------------------------- #
def _valgrid(g):
    """[1,1,30,30] colour-value field (sum_c c*input_c) via a 1x1 conv (10 params)."""
    wv = g.f([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    return g.node("Conv", ["input", wv], kernel_shape=[1, 1], pads=[0, 0, 0, 0])


def _realmask(g):
    return g.node("ReduceSum", ["input"], axes=[1], keepdims=1)


def _shift(g, x, sr, sc):
    """new[i,j] = x[i-sr, j-sc] (zero fill), staying [.,.,30,30]."""
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
    nbg = g.node("Cast", [g.node("Greater", [valgrid, half])], to=DATA_TYPE)
    c01 = g.f([1, 1, 1, 1], [0.1])
    c10 = g.f([1, 1, 1, 1], [10.0])

    vals = []
    for nm in dir_names:
        dr, dc = DIRS[nm]
        rowv = g.f([1, 1, HEIGHT, 1], [10 * dr * i for i in range(HEIGHT)])
        colv = g.f([1, 1, 1, WIDTH], [10 * dc * j + _OFF for j in range(WIDTH)])
        w10 = g.node("Add", [rowv, colv])
        key = g.node("Add", [g.node("Mul", [nbg, w10]), valgrid])
        K = _prefixmax(g, key, dr, dc)
        fw = g.node("Cast", [g.node("Cast", [g.node("Mul", [K, c01])], to=INT32)],
                    to=DATA_TYPE)
        vals.append(g.node("Sub", [K, g.node("Mul", [fw, c10])]))

    V = vals[0]
    for v in vals[1:]:
        V = g.node("Max", [V, v])
    Vint = g.node("Cast", [V], to=INT32)
    idx = g.i32([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    onehot = g.node("Cast", [g.node("Equal", [Vint, idx])], to=DATA_TYPE)
    g.node("Mul", [onehot, R], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# diagonal extraction (keep the cells on a diagonal, rest -> background)       #
# --------------------------------------------------------------------------- #
def _diag_mask(kind, h, w):
    """Fixed [1,1,30,30] mask flagging the chosen diagonal.  The main diagonal
    (i==j) is size-independent and spans the full 30x30 (the real-cell mask gates
    it to the real region at runtime).  The anti diagonal (i+j==w-1) depends on
    the width, so it is only used when the grid size is constant."""
    m = np.zeros((1, 1, HEIGHT, WIDTH), np.float32)
    if kind == "main":
        for i in range(HEIGHT):
            m[0, 0, i, i] = 1.0
    else:  # anti
        for i in range(min(h, HEIGHT)):
            j = w - 1 - i
            if 0 <= j < WIDTH:
                m[0, 0, i, j] = 1.0
    return m


def build_keep_diag(kind, h, w):
    g = _G()
    D = g.f([1, 1, HEIGHT, WIDTH], _diag_mask(kind, h, w))
    R = _realmask(g)                                   # [1,1,30,30]
    kept = g.node("Mul", ["input", D])                 # keep colours on diagonal
    onD = g.node("Mul", [R, D])                        # real cells on diagonal
    offD = g.node("Sub", [R, onD])                     # real cells off diagonal
    e0 = g.f([1, CHANNELS, 1, 1], _onehot(0))
    add0 = g.node("Mul", [offD, e0])                   # set those to background
    g.node("Add", [kept, add0], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX semantics for detection)                  #
# --------------------------------------------------------------------------- #
def _antitranspose(a):
    return a[::-1, ::-1].T


def _shift_np(x, sr, sc):
    h, w = x.shape
    out = np.zeros_like(x)
    rs0, rs1 = max(sr, 0), h - max(-sr, 0)
    cs0, cs1 = max(sc, 0), w - max(-sc, 0)
    if rs0 < rs1 and cs0 < cs1:
        out[rs0:rs1, cs0:cs1] = x[rs0 - sr:rs1 - sr, cs0 - sc:cs1 - sc]
    return out


def _val_dir(a, dr, dc):
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


def _ref_keep(a, kind):
    h, w = a.shape
    out = np.zeros_like(a)
    for i in range(h):
        for j in range(w):
            if (kind == "main" and i == j) or (kind == "anti" and i + j == w - 1):
                out[i, j] = a[i, j]
    return out


def _color_map(prs, base):
    """Consistent per-colour map applied AFTER base(a); None if inconsistent."""
    mp = {}
    for a, b in prs:
        r = base(a)
        if r is None or r.shape != b.shape:
            return None
        for x, y in zip(r.ravel().tolist(), b.ravel().tolist()):
            if x in mp and mp[x] != y:
                return None
            mp[x] = y
    full = list(range(CHANNELS))
    for s, d in mp.items():
        if 0 <= s < CHANNELS:
            full[s] = d
    return full


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


def _emit(out, name, builder):
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    out.append((name, m))


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if all(np.array_equal(a, b) for a, b in prs):     # identity -> not our family
        return []

    out = []
    const_size = len({a.shape for a, _ in prs}) == 1
    shape_pres = all(a.shape == b.shape for a, b in prs)

    # ---- reflect across the main diagonal (transpose) ---------------------- #
    if all(b.shape == a.T.shape and np.array_equal(a.T, b) for a, b in prs):
        _emit(out, "reflect_main", build_transpose)
    else:
        cmap = _color_map(prs, lambda a: a.T)
        if cmap is not None and cmap != list(range(CHANNELS)):
            _emit(out, "reflect_main_map", lambda cm=cmap: build_transpose_map(cm))

    # ---- reflect across the anti diagonal (constant size only) ------------- #
    if const_size:
        h0, w0 = prs[0][0].shape
        if all(b.shape == _antitranspose(a).shape and np.array_equal(_antitranspose(a), b)
               for a, b in prs):
            _emit(out, "reflect_anti", lambda: build_antitranspose(h0, w0))
        else:
            cmap = _color_map(prs, _antitranspose)
            if cmap is not None and cmap != list(range(CHANNELS)):
                _emit(out, "reflect_anti_map",
                      lambda cm=cmap: build_antitranspose(h0, w0, cm))

    # ---- diagonal rays / shadows ------------------------------------------- #
    if shape_pres:
        # additive: every non-background input cell is preserved
        additive = all(np.array_equal(b[a != 0], a[a != 0]) for a, b in prs)
        if additive:
            for name, dirset in RAYSETS.items():
                if all(np.array_equal(_ref_shadow(a, dirset), b) for a, b in prs):
                    _emit(out, name, lambda d=dirset: build_shadow(d))
                    break

    # ---- extract a diagonal (keep diagonal colours, rest -> background) ---- #
    if shape_pres:
        if all(np.array_equal(_ref_keep(a, "main"), b) for a, b in prs):
            h0, w0 = prs[0][0].shape
            _emit(out, "keep_main", lambda h=h0, w=w0: build_keep_diag("main", h, w))
        if const_size:
            h0, w0 = prs[0][0].shape
            if all(np.array_equal(_ref_keep(a, "anti"), b) for a, b in prs):
                _emit(out, "keep_anti", lambda h=h0, w=w0: build_keep_diag("anti", h, w))

    return out
