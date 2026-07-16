"""family_golf_4 -- cheaper EXACT solvers for golf slice IDX=4.

Each detector re-derives the rule from train+test pairs (numpy), then emits a
minimal opset-10 ONNX graph (few/small intermediates -> low cost). The integrator
auto-picks the cheapest correct solver, so we just need cheaper-but-exact graphs.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


# --------------------------------------------------------------------------- #
# infra                                                                        #
# --------------------------------------------------------------------------- #
def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _ft(name, arr):
    arr = np.asarray(arr, np.float32)
    return oh.make_tensor(name, DATA_TYPE, list(arr.shape), arr.ravel().tolist())


def _onehot(grid):
    """grid (H,W) ints -> (CHANNELS,30,30) float one-hot, top-left anchored."""
    t = np.zeros((CHANNELS, HEIGHT, WIDTH), np.float32)
    h, w = grid.shape
    for c in range(CHANNELS):
        t[c, :h, :w] = (grid == c)
    return t


def _pairs(ex):
    out = []
    for e in ex.get("train", []) + ex.get("test", []):
        a = np.array(e["input"], int)
        b = np.array(e["output"], int)
        if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
            return None
        if max(a.shape) > 30 or max(b.shape) > 30:
            return None
        out.append((a, b))
    return out


# bg one-hot vector [1,10,1,1] (= color 0)
def _bg_onehot():
    v = np.zeros((1, CHANNELS, 1, 1), np.float32)
    v[0, 0] = 1.0
    return v


# --------------------------------------------------------------------------- #
# Task 97 family: denoise box-3, remove fg pixel with < T fg in its 3x3 box   #
#   value = 9*center_fg - neighbours_fg ; remove iff value > (10.5 - T)         #
# --------------------------------------------------------------------------- #
def _denoise_predict(a, T):
    h, w = a.shape
    fg = (a != 0).astype(int)
    cnt = np.zeros_like(fg)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            sl = np.zeros_like(fg)
            si0, si1 = max(0, di), min(h, h + di)
            sj0, sj1 = max(0, dj), min(w, w + dj)
            sl[max(0, -di):h - max(0, di) if di > 0 else h,
               max(0, -dj):w - max(0, dj) if dj > 0 else w] = \
                fg[si0:si1, sj0:sj1]
            cnt += sl
    out = a.copy()
    out[(fg == 1) & (cnt < T)] = 0
    return out


def _detect_denoise(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    # must actually remove something somewhere
    if not any((a != b).any() for a, b in prs):
        return None
    for T in (2, 3, 4):
        if all(np.array_equal(_denoise_predict(a, T), b) for a, b in prs):
            return T
    return None


def _build_denoise(T):
    thr = 10.5 - T
    W = np.zeros((1, CHANNELS, 3, 3), np.float32)
    for c in range(1, CHANNELS):
        W[0, c, :, :] = -1.0
        W[0, c, 1, 1] = 9.0
    nodes = [
        oh.make_node("Conv", ["input", "Wd"], ["v"], kernel_shape=[3, 3],
                     pads=[1, 1, 1, 1]),
        oh.make_node("Greater", ["v", "thr"], ["rm"]),
        oh.make_node("Where", ["rm", "bg", "input"], ["output"]),
    ]
    inits = [_ft("Wd", W), _ft("thr", np.array([thr], np.float32)),
             _ft("bg", _bg_onehot())]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# Task 50 family: connect collinear same-colour marker pairs with fill colour  #
# --------------------------------------------------------------------------- #
def _connect_predict(a, fill):
    h, w = a.shape
    M = (a != 0).astype(int)
    out = a.copy()
    # horizontal
    for i in range(h):
        cols = np.where(M[i] > 0)[0]
        if cols.size >= 2:
            for j in range(cols[0] + 1, cols[-1]):
                if a[i, j] == 0:
                    out[i, j] = fill
    # vertical
    for j in range(w):
        rows = np.where(M[:, j] > 0)[0]
        if rows.size >= 2:
            for i in range(rows[0] + 1, rows[-1]):
                if a[i, j] == 0:
                    out[i, j] = fill
    return out


def _detect_connect(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    marker, fill = set(), set()
    for a, b in prs:
        ina = set(np.unique(a)) - {0}
        new = set(np.unique(b[a != b]))
        marker |= ina
        fill |= new
    fill -= {0}
    if len(fill) != 1:
        return None
    fc = fill.pop()
    if any((a == fc).any() for a, _ in prs):  # fill colour must be new
        return None
    if all(np.array_equal(_connect_predict(a, fc), b) for a, b in prs):
        return fc
    return None


def _build_connect(fc):
    # directional fg-count convs (kernel sums fg channels along one axis)
    KwLR = np.zeros((1, CHANNELS, 1, WIDTH), np.float32)
    KwLR[0, 1:, 0, :] = 1.0
    KhUD = np.zeros((1, CHANNELS, HEIGHT, 1), np.float32)
    KhUD[0, 1:, :, 0] = 1.0
    Wbg = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Wbg[0, 0, 0, 0] = 1.0
    fo = np.zeros((1, CHANNELS, 1, 1), np.float32)
    fo[0, fc] = 1.0
    half = WIDTH - 1
    halfh = HEIGHT - 1
    nodes = [
        oh.make_node("Conv", ["input", "KwLR"], ["HL"], kernel_shape=[1, WIDTH],
                     pads=[0, half, 0, 0]),
        oh.make_node("Conv", ["input", "KwLR"], ["HR"], kernel_shape=[1, WIDTH],
                     pads=[0, 0, 0, half]),
        oh.make_node("Conv", ["input", "KhUD"], ["HU"], kernel_shape=[HEIGHT, 1],
                     pads=[halfh, 0, 0, 0]),
        oh.make_node("Conv", ["input", "KhUD"], ["HD"], kernel_shape=[HEIGHT, 1],
                     pads=[0, 0, halfh, 0]),
        oh.make_node("Conv", ["input", "Wbg"], ["BG"], kernel_shape=[1, 1]),
        oh.make_node("Greater", ["HL", "z"], ["bHL"]),
        oh.make_node("Greater", ["HR", "z"], ["bHR"]),
        oh.make_node("Greater", ["HU", "z"], ["bHU"]),
        oh.make_node("Greater", ["HD", "z"], ["bHD"]),
        oh.make_node("Greater", ["BG", "z"], ["bBG"]),
        oh.make_node("And", ["bHL", "bHR"], ["hor"]),
        oh.make_node("And", ["bHU", "bHD"], ["ver"]),
        oh.make_node("Or", ["hor", "ver"], ["bet"]),
        oh.make_node("And", ["bet", "bBG"], ["fillb"]),
        oh.make_node("Where", ["fillb", "fo", "input"], ["output"]),
    ]
    inits = [_ft("KwLR", KwLR), _ft("KhUD", KhUD), _ft("Wbg", Wbg),
             _ft("fo", fo), _ft("z", np.array([0.5], np.float32))]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# Task 141 family: draw X (both diagonals) through each foreground dot         #
# --------------------------------------------------------------------------- #
def _diagx_predict(a):
    h, w = a.shape
    out = a.copy()
    ys, xs = np.where(a != 0)
    colors = set(a[a != 0].tolist())
    if len(colors) != 1:
        return None
    c = colors.pop()
    for y, x in zip(ys, xs):
        for i in range(h):
            for j in range(w):
                if (i - j) == (y - x) or (i + j) == (y + x):
                    out[i, j] = c
    return out


def _detect_diagx(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    if not any((a != b).any() for a, b in prs):
        return None
    for a, b in prs:
        p = _diagx_predict(a)
        if p is None or not np.array_equal(p, b):
            return None
    return True


def _build_diagx():
    Wfg = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Wfg[0, 1:, 0, 0] = 1.0
    K = 2 * max(HEIGHT, WIDTH) - 1  # 59
    Kx = np.zeros((1, 1, K, K), np.float32)
    for i in range(K):
        Kx[0, 0, i, i] = 1.0
        Kx[0, 0, i, K - 1 - i] = 1.0
    pad = K // 2
    fgchan = np.zeros((1, CHANNELS, 1, 1), np.float32)
    fgchan[0, 1:] = 1.0
    nodes = [
        oh.make_node("Conv", ["input", "Wfg"], ["M"], kernel_shape=[1, 1]),
        oh.make_node("Conv", ["M", "Kx"], ["X"], kernel_shape=[K, K],
                     pads=[pad, pad, pad, pad]),
        oh.make_node("ReduceSum", ["input"], ["real"], axes=[1], keepdims=1),
        oh.make_node("ReduceMax", ["input"], ["cv"], axes=[2, 3], keepdims=1),
        oh.make_node("Mul", ["cv", "fgchan"], ["cvf"]),
        oh.make_node("Greater", ["X", "z"], ["bX"]),
        oh.make_node("Greater", ["real", "z"], ["bR"]),
        oh.make_node("And", ["bX", "bR"], ["fillb"]),
        oh.make_node("Where", ["fillb", "cvf", "input"], ["output"]),
    ]
    inits = [_ft("Wfg", Wfg), _ft("Kx", Kx), _ft("fgchan", fgchan),
             _ft("z", np.array([0.5], np.float32))]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# Task 151 family: stamp a 3x3 ring of marker colour around a cross-intersection #
#   centre = fg cell whose 4 orthogonal neighbours are all fg                   #
# --------------------------------------------------------------------------- #
def _cstamp_predict(a, sc):
    h, w = a.shape
    fg = (a != 0)
    out = a.copy()
    centers = []
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if fg[i, j] and fg[i - 1, j] and fg[i + 1, j] and fg[i, j - 1] and fg[i, j + 1]:
                centers.append((i, j))
    for (i, j) in centers:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                out[i + di, j + dj] = sc
    return out, len(centers)


def _detect_cstamp(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    newc = set()
    for a, b in prs:
        newc |= set(np.unique(b[a != b]).tolist())
    newc -= {0}
    if len(newc) != 1:
        return None
    sc = newc.pop()
    ok_any = False
    for a, b in prs:
        pred, nctr = _cstamp_predict(a, sc)
        if nctr == 0:
            continue
        ok_any = True
        if not np.array_equal(pred, b):
            return None
    return sc if ok_any else None


def _build_cstamp(sc):
    Wfg = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Wfg[0, 1:, 0, 0] = 1.0
    Kcross = np.zeros((1, 1, 3, 3), np.float32)
    Kcross[0, 0, 0, 1] = Kcross[0, 0, 2, 1] = 1.0
    Kcross[0, 0, 1, 0] = Kcross[0, 0, 1, 2] = 1.0
    Kbox = np.ones((1, 1, 3, 3), np.float32)
    four = np.zeros((1, CHANNELS, 1, 1), np.float32)
    four[0, sc] = 1.0
    nodes = [
        oh.make_node("Conv", ["input", "Wfg"], ["M"], kernel_shape=[1, 1]),
        oh.make_node("Conv", ["M", "Kcross"], ["c4"], kernel_shape=[3, 3],
                     pads=[1, 1, 1, 1]),
        oh.make_node("Greater", ["c4", "t35"], ["bN"]),
        oh.make_node("Greater", ["M", "z"], ["bM"]),
        oh.make_node("And", ["bN", "bM"], ["ctr"]),
        oh.make_node("Cast", ["ctr"], ["ctrf"], to=DATA_TYPE),
        oh.make_node("Conv", ["ctrf", "Kbox"], ["box"], kernel_shape=[3, 3],
                     pads=[1, 1, 1, 1]),
        oh.make_node("Greater", ["box", "z"], ["bBox"]),
        oh.make_node("Not", ["ctr"], ["nctr"]),
        oh.make_node("And", ["bBox", "nctr"], ["ring"]),
        oh.make_node("Where", ["ring", "four", "input"], ["output"]),
    ]
    inits = [_ft("Wfg", Wfg), _ft("Kcross", Kcross), _ft("Kbox", Kbox),
             _ft("four", four), _ft("t35", np.array([3.5], np.float32)),
             _ft("z", np.array([0.5], np.float32))]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# Task 256 family: anti-diagonal triangle from a left-edge bar, 3 colour bands  #
#   filled = {(r,c): r+c <= K}, K = max(r'+c') over fg; colour by row vs bar row #
# --------------------------------------------------------------------------- #
def _tri_predict(a, cbar, cab, cbe):
    h, w = a.shape
    fg = a != 0
    rows, cols = np.where(fg)
    if rows.size == 0 or len(set(rows.tolist())) != 1:
        return None
    R = int(rows[0])
    K = int((rows + cols).max())
    out = np.zeros_like(a)
    for r in range(h):
        for c in range(w):
            if r + c <= K:
                out[r, c] = cab if r < R else (cbe if r > R else cbar)
    return out


def _detect_tri(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    cbar = cab = cbe = None
    for a, b in prs:
        fg = a != 0
        inc = set(np.unique(a[fg]).tolist())
        if len(inc) != 1:
            return None
        cb = inc.pop()
        rows, cols = np.where(fg)
        if len(set(rows.tolist())) != 1:
            return None
        R = int(rows[0])
        ab = set(np.unique(b[:R][b[:R] != 0]).tolist()) if R > 0 else set()
        be = set(np.unique(b[R + 1:][b[R + 1:] != 0]).tolist())
        if len(ab) > 1 or len(be) > 1:
            return None
        if cbar is None:
            cbar = cb
        if ab:
            cab = ab.pop() if cab is None else cab
        if be:
            cbe = be.pop() if cbe is None else cbe
    if None in (cbar, cab, cbe):
        return None
    for a, b in prs:
        p = _tri_predict(a, cbar, cab, cbe)
        if p is None or not np.array_equal(p, b):
            return None
    return cbar, cab, cbe


def _oh_col(c):
    v = np.zeros((1, CHANNELS, 1, 1), np.float32)
    v[0, c] = 1.0
    return v


def _build_tri(cbar, cab, cbe):
    Wfg = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Wfg[0, 1:, 0, 0] = 1.0
    K = 2 * max(HEIGHT, WIDTH) - 1
    Ktri = np.zeros((1, 1, K, K), np.float32)
    for i in range(K):
        for j in range(K):
            if i + j >= K - 1:
                Ktri[0, 0, i, j] = 1.0
    pad = K // 2
    Kup = np.ones((1, 1, HEIGHT, 1), np.float32)
    nodes = [
        oh.make_node("Conv", ["input", "Wfg"], ["fg"], kernel_shape=[1, 1]),
        oh.make_node("Conv", ["fg", "Ktri"], ["tri"], kernel_shape=[K, K],
                     pads=[pad, pad, pad, pad]),
        oh.make_node("ReduceSum", ["input"], ["real"], axes=[1], keepdims=1),
        oh.make_node("ReduceMax", ["fg"], ["rowfg"], axes=[3], keepdims=1),
        oh.make_node("Conv", ["rowfg", "Kup"], ["aI"], kernel_shape=[HEIGHT, 1],
                     pads=[HEIGHT - 1, 0, 0, 0]),
        oh.make_node("Conv", ["rowfg", "Kup"], ["bI"], kernel_shape=[HEIGHT, 1],
                     pads=[0, 0, HEIGHT - 1, 0]),
        oh.make_node("Greater", ["aI", "z"], ["gA"]),
        oh.make_node("Greater", ["bI", "z"], ["gB"]),
        oh.make_node("Not", ["gA"], ["nA"]),
        oh.make_node("Not", ["gB"], ["nB"]),
        oh.make_node("And", ["gA", "gB"], ["gbar"]),
        oh.make_node("Cast", ["nA"], ["fA"], to=DATA_TYPE),
        oh.make_node("Cast", ["nB"], ["fB"], to=DATA_TYPE),
        oh.make_node("Cast", ["gbar"], ["fbar"], to=DATA_TYPE),
        oh.make_node("Mul", ["ohAB", "fA"], ["pA"]),
        oh.make_node("Mul", ["ohBE", "fB"], ["pB"]),
        oh.make_node("Mul", ["ohBAR", "fbar"], ["pbar"]),
        oh.make_node("Add", ["pA", "pB"], ["pAB"]),
        oh.make_node("Add", ["pAB", "pbar"], ["rowcolor"]),
        oh.make_node("Greater", ["tri", "z"], ["tb"]),
        oh.make_node("Greater", ["real", "z"], ["rb"]),
        oh.make_node("And", ["tb", "rb"], ["fillb"]),
        oh.make_node("Where", ["fillb", "rowcolor", "input"], ["output"]),
    ]
    inits = [_ft("Wfg", Wfg), _ft("Ktri", Ktri), _ft("Kup", Kup),
             _ft("ohAB", _oh_col(cab)), _ft("ohBE", _oh_col(cbe)),
             _ft("ohBAR", _oh_col(cbar)), _ft("z", np.array([0.5], np.float32))]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# Task 306 family: periodic completion (tile a single quadrant over the grid)   #
#   output[r,c] = unique non-bg colour over residue class (r,c) mod p           #
# --------------------------------------------------------------------------- #
def _tile_predict(a, p):
    h, w = a.shape
    out = np.zeros_like(a)
    for r in range(h):
        for c in range(w):
            vals = set()
            for r2 in range(r % p, h, p):
                for c2 in range(c % p, w, p):
                    if a[r2, c2] != 0:
                        vals.add(int(a[r2, c2]))
            if len(vals) == 1:
                out[r, c] = vals.pop()
            elif len(vals) > 1:
                return None
    return out


def _detect_tile(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    # need genuine completion (some bg -> fg) and non-trivial period
    if not any((a != b).any() for a, b in prs):
        return None
    for p in range(2, 16):
        if all(max(a.shape) > p for a, _ in prs) and \
           all((pp := _tile_predict(a, p)) is not None and np.array_equal(pp, b)
               for a, b in prs):
            return p
    return None


def _build_tile(p):
    md = max(HEIGHT, WIDTH)
    half = p * ((md - 1) // p)
    K = 2 * half + 1
    Kp = np.zeros((CHANNELS, 1, K, K), np.float32)
    offs = list(range(-half, half + 1, p))
    for ch in range(1, CHANNELS):
        for dr in offs:
            for dc in offs:
                Kp[ch, 0, half + dr, half + dc] = 1.0
    nodes = [
        oh.make_node("Conv", ["input", "Kp"], ["til"], kernel_shape=[K, K],
                     pads=[half, half, half, half], group=CHANNELS),
        oh.make_node("ReduceSum", ["input"], ["real"], axes=[1], keepdims=1),
        oh.make_node("ReduceSum", ["til"], ["tp"], axes=[1], keepdims=1),
        oh.make_node("Greater", ["tp", "z"], ["tb"]),
        oh.make_node("Greater", ["real", "z"], ["rb"]),
        oh.make_node("And", ["tb", "rb"], ["fb"]),
        oh.make_node("Where", ["fb", "til", "input"], ["output"]),
    ]
    inits = [_ft("Kp", Kp), _ft("z", np.array([0.5], np.float32))]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# Task 232 family: each (single-per-row) dot emits a rightward ray alternating  #
#   dot-colour (even offset) and a fixed fill-colour (odd offset)               #
# --------------------------------------------------------------------------- #
def _alt_predict(a, fillc):
    h, w = a.shape
    out = a.copy()
    changed = False
    for r in range(h):
        cols = np.where(a[r] != 0)[0]
        if cols.size == 0:
            continue
        if cols.size != 1:
            return None, False
        c0 = int(cols[0])
        d = int(a[r, c0])
        if d == fillc:
            return None, False
        for c in range(c0, w):
            out[r, c] = d if (c - c0) % 2 == 0 else fillc
        if w - 1 > c0:
            changed = True
    return out, changed


def _detect_alt(prs):
    if not all(a.shape == b.shape for a, b in prs):
        return None
    inc, outc = set(), set()
    for a, b in prs:
        inc |= set(np.unique(a).tolist())
        outc |= set(np.unique(b).tolist())
    newc = (outc - inc) - {0}
    if len(newc) != 1:
        return None
    fc = newc.pop()
    any_change = False
    for a, b in prs:
        p, ch = _alt_predict(a, fc)
        if p is None or not np.array_equal(p, b):
            return None
        any_change |= ch
    return fc if any_change else None


def _build_alt(fc):
    Keven = np.zeros((CHANNELS, 1, 1, WIDTH), np.float32)
    for ch in range(1, CHANNELS):
        for j in range(WIDTH - 1, -1, -2):   # odd j -> even left-offset
            Keven[ch, 0, 0, j] = 1.0
    Wfg = np.zeros((1, CHANNELS, 1, 1), np.float32)
    Wfg[0, 1:, 0, 0] = 1.0
    Kodd = np.zeros((1, 1, 1, WIDTH), np.float32)
    for j in range(WIDTH - 2, -1, -2):       # even j -> odd left-offset
        Kodd[0, 0, 0, j] = 1.0
    five = _oh_col(fc)
    hw = WIDTH - 1
    nodes = [
        oh.make_node("Conv", ["input", "Keven"], ["even"], kernel_shape=[1, WIDTH],
                     pads=[0, hw, 0, 0], group=CHANNELS),
        oh.make_node("Conv", ["input", "Wfg"], ["fg"], kernel_shape=[1, 1]),
        oh.make_node("Conv", ["fg", "Kodd"], ["op"], kernel_shape=[1, WIDTH],
                     pads=[0, hw, 0, 0]),
        oh.make_node("ReduceSum", ["even"], ["ep"], axes=[1], keepdims=1),
        oh.make_node("ReduceSum", ["input"], ["real"], axes=[1], keepdims=1),
        oh.make_node("Greater", ["real", "z"], ["rb"]),
        oh.make_node("Greater", ["ep", "z"], ["eb"]),
        oh.make_node("Greater", ["op", "z"], ["ob"]),
        oh.make_node("And", ["eb", "rb"], ["ebr"]),
        oh.make_node("And", ["ob", "rb"], ["obr"]),
        oh.make_node("Where", ["obr", "five", "input"], ["inner"]),
        oh.make_node("Where", ["ebr", "even", "inner"], ["output"]),
    ]
    inits = [_ft("Keven", Keven), _ft("Wfg", Wfg), _ft("Kodd", Kodd),
             _ft("five", five), _ft("z", np.array([0.5], np.float32))]
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    T = _detect_denoise(prs)
    if T is not None:
        out.append((f"denoise_box3_T{T}", _build_denoise(T)))

    fc = _detect_connect(prs)
    if fc is not None:
        out.append((f"connect_fill{fc}", _build_connect(fc)))

    if _detect_diagx(prs):
        out.append(("diagx", _build_diagx()))

    sc = _detect_cstamp(prs)
    if sc is not None:
        out.append((f"cstamp{sc}", _build_cstamp(sc)))

    tri = _detect_tri(prs)
    if tri is not None:
        out.append(("triangle", _build_tri(*tri)))

    p = _detect_tile(prs)
    if p is not None:
        out.append((f"tile{p}", _build_tile(p)))

    fa = _detect_alt(prs)
    if fa is not None:
        out.append((f"altfill{fa}", _build_alt(fa)))

    return out
