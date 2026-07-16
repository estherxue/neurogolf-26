"""family_golf2_1 -- cheaper exact solvers for a slice of golf targets.

Each candidate re-derives the task's rule from train+test+arc-gen pairs, verifies
EXACT equality against a numpy reference on every available pair, and only then
emits a minimal opset-10 ONNX graph.  The integrator auto-picks the cheapest
correct solver, so we just need these to be exact and cheaper than the incumbent.

Cost levers used throughout:
  * operate at the (constant) grid size H x W instead of the full 30x30 frame
  * single-channel [1,1,H,W] reductions / triangular matmuls instead of [1,9,..]
  * cumulative-OR via small triangular [W,W]/[H,H] matmuls (no log-unroll memory)
  * recolour via a tiny [1,10,1,1] colour vector + broadcast multiply
  * write the final result straight into the FREE "output" tensor via Pad
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


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

    def iconst(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], list(vals)))
        return nm

    def fconst(self, vals, shape):
        nm = self.name("f")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(shape),
                                         [float(v) for v in vals]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _slice(g, src, starts, ends, axes, steps=None):
    ins = [src, g.iconst(starts), g.iconst(ends), g.iconst(axes)]
    if steps is not None:
        ins.append(g.iconst(steps))
    return g.node("Slice", ins)


def _conv3(g, src, kernel):
    """3x3 same-padded single-channel convolution. `kernel` is a length-9 list."""
    w = g.fconst(kernel, [1, 1, 3, 3])
    return g.node("Conv", [src, w], kernel_shape=[3, 3], pads=[1, 1, 1, 1])


def _shift(g, src, dr, dc, H, W):
    """Shift a [1,1,H,W] tensor so out[r,c] = src[r-dr, c-dc] with zero fill."""
    pads = [0, 0, max(dr, 0), max(dc, 0), 0, 0, max(-dr, 0), max(-dc, 0)]
    pad = g.node("Pad", [src], mode="constant", value=0.0, pads=pads)
    sr, sc = max(-dr, 0), max(-dc, 0)
    return _slice(g, pad, [sr, sc], [sr + H, sc + W], [2, 3])


# --------------------------------------------------------------------------- #
# pairs                                                                        #
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


# ===========================================================================
# 356 connect1:  for each colour, fill the segment between the extreme markers
#                in every row and every column.
# ===========================================================================
def _ref_connect_collinear(a):
    H, W = a.shape
    o = a.copy()
    for col in range(1, 10):
        M = (a == col)
        for r in range(H):
            idx = np.where(M[r])[0]
            if len(idx) >= 2:
                o[r, idx.min():idx.max() + 1] = col
        for c in range(W):
            idx = np.where(M[:, c])[0]
            if len(idx) >= 2:
                o[idx.min():idx.max() + 1, c] = col
    return o


def _build_connect_collinear(H, W):
    g = _G()
    # work inside the H x W window
    in_small = _slice(g, "input", [0, 0], [H, W], [2, 3])          # [1,10,H,W]
    allsum = g.node("ReduceSum", [in_small], axes=[1], keepdims=1)  # [1,1,H,W]
    bg = _slice(g, in_small, [0], [1], [1])                        # [1,1,H,W]
    nonbg = g.node("Sub", [allsum, bg])                            # [1,1,H,W] markers

    # cumulative OR (as counts) via triangular matmuls
    Lw = g.fconst([1.0 if k <= c else 0.0 for k in range(W) for c in range(W)],
                  [W, W])
    Rw = g.fconst([1.0 if k >= c else 0.0 for k in range(W) for c in range(W)],
                  [W, W])
    Th = g.fconst([1.0 if k <= r else 0.0 for r in range(H) for k in range(H)],
                  [H, H])
    Bh = g.fconst([1.0 if k >= r else 0.0 for r in range(H) for k in range(H)],
                  [H, H])
    leftc = g.node("MatMul", [nonbg, Lw])     # markers at/left of c
    rightc = g.node("MatMul", [nonbg, Rw])    # markers at/right of c
    minh = g.node("Min", [leftc, rightc])     # >0 -> between in row
    topc = g.node("MatMul", [Th, nonbg])      # markers at/above r
    botc = g.node("MatMul", [Bh, nonbg])      # markers at/below r
    minv = g.node("Min", [topc, botc])        # >0 -> between in col
    fill = g.node("Max", [minh, minv])        # [1,1,H,W] cells to colour

    # colour vector: +1 at the marker colour channel, -1 at bg channel
    cvec = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    m01 = g.fconst([0.0] + [1.0] * (CHANNELS - 1), [1, CHANNELS, 1, 1])
    e0 = g.fconst([1.0] + [0.0] * (CHANNELS - 1), [1, CHANNELS, 1, 1])
    cvm = g.node("Mul", [cvec, m01])
    P = g.node("Sub", [cvm, e0])                                    # [1,10,1,1]

    delta = g.node("Mul", [fill, P])                               # [1,10,H,W]
    out_small = g.node("Add", [in_small, delta])                   # [1,10,H,W]
    g.node("Pad", [out_small], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 278 frame_c2_m3:  every 2 that is 4-adjacent to another 2 (a domino) gets a
#                   colour-3 frame painted on the surrounding bg cells.
# ===========================================================================
def _ref_frame_dominoes(a):
    from_conv = None
    H, W = a.shape
    two = (a == 2).astype(int)

    def shift(m, dr, dc):
        o = np.zeros_like(m)
        rs = slice(max(0, dr), H + min(0, dr)); rd = slice(max(0, -dr), H + min(0, -dr))
        cs = slice(max(0, dc), W + min(0, dc)); cd = slice(max(0, -dc), W + min(0, -dc))
        o[rd, cd] = m[rs, cs]
        return o

    nb = sum(shift(two, dr, dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    active = (two * (nb >= 1)).astype(int)
    dil = sum(shift(active, dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    o = a.copy()
    o[(a == 0) & (dil >= 1)] = 3
    return o


def _build_frame_dominoes():
    g = _G()
    two = _slice(g, "input", [2], [3], [1])                # [1,1,30,30] colour-2
    nb = _conv3(g, two, [0, 1, 0, 1, 0, 1, 0, 1, 0])       # 4-neighbour count
    active = g.node("Min", [two, nb])                      # 2s with a 2-neighbour
    dil = _conv3(g, active, [1] * 9)                       # 8-neighbourhood (+center)
    bg = _slice(g, "input", [0], [1], [1])                 # [1,1,30,30] bg channel
    frame = g.node("Min", [bg, dil])                       # bg cells in a frame -> 3
    ch0 = g.node("Sub", [bg, frame])                       # turn bg off under frame
    zero = g.fconst([0.0] * (HEIGHT * WIDTH), [1, 1, HEIGHT, WIDTH])
    # channels: 0=ch0, 2=two (kept), 3=frame, all others zero
    g.node("Concat", [ch0, zero, two, frame, zero, zero, zero, zero, zero, zero],
           "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 38 fieldbar:  count the 2x2 all-colour-1 blocks (capped at 5); emit a 1x5 row
#               with that many leading 1s.
# ===========================================================================
def _ref_count_bar(a):
    H, W = a.shape
    ones = (a == 1).astype(int)
    cnt = 0
    for r in range(H - 1):
        for c in range(W - 1):
            if ones[r:r + 2, c:c + 2].sum() == 4:
                cnt += 1
    N = min(cnt, 5)
    o = np.zeros((1, 5), int)
    o[0, :N] = 1
    return o


def _build_count_bar():
    g = _G()
    ones = _slice(g, "input", [1], [2], [1])               # [1,1,30,30] colour-1
    k = g.fconst([1, 1, 1, 1], [1, 1, 2, 2])
    c22 = g.node("Conv", [ones, k], kernel_shape=[2, 2], pads=[0, 0, 0, 0])  # [1,1,29,29]
    eq4 = g.node("Greater", [c22, g.fconst([3.5], [1])])   # ==4 -> True
    eq4f = g.node("Cast", [eq4], to=DATA_TYPE)
    N = g.node("ReduceSum", [eq4f], axes=[2, 3], keepdims=1)  # [1,1,1,1]
    thr = g.fconst([0, 1, 2, 3, 4], [1, 1, 1, 5])
    bar1 = g.node("Greater", [N, thr])                     # N>i -> leading 1s
    bar1f = g.node("Cast", [bar1], to=DATA_TYPE)
    bar0 = g.node("Sub", [g.fconst([1.0], [1]), bar1f])    # bg where not 1
    z = g.fconst([0.0] * 5, [1, 1, 1, 5])
    small = g.node("Concat", [bar0, bar1f] + [z] * 8, axis=1)  # [1,10,1,5]
    g.node("Pad", [small], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - 1, WIDTH - 5])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 246 L-connect:  single 2 and single 3 joined by an L of colour 8 -- the
#                 horizontal leg lies on the 2's row, the vertical leg on the
#                 3's column (corner at (row2, col3)).  Endpoints kept.
# ===========================================================================
def _ref_lconnect(a):
    p2 = np.argwhere(a == 2); p3 = np.argwhere(a == 3)
    if len(p2) != 1 or len(p3) != 1:
        return None
    r2, c2 = p2[0]; r3, c3 = p3[0]
    o = a.copy()
    lo, hi = sorted((c2, c3))
    for c in range(lo, hi + 1):
        if o[r2, c] == 0:
            o[r2, c] = 8
    lo, hi = sorted((r2, r3))
    for r in range(lo, hi + 1):
        if o[r, c3] == 0:
            o[r, c3] = 8
    return o


def _tri(g, pred, n):
    return g.fconst([1.0 if pred(i, j) else 0.0 for i in range(n) for j in range(n)],
                    [n, n])


def _build_lconnect():
    g = _G()
    H = W = HEIGHT  # 30 (sizes vary -> use the full frame)
    in2 = _slice(g, "input", [2], [3], [1])               # [1,1,30,30]
    in3 = _slice(g, "input", [3], [4], [1])               # [1,1,30,30]
    bg = _slice(g, "input", [0], [1], [1])                # [1,1,30,30]
    col2 = g.node("ReduceMax", [in2], axes=[2], keepdims=1)   # [1,1,1,30]
    col3 = g.node("ReduceMax", [in3], axes=[2], keepdims=1)   # [1,1,1,30]
    row2 = g.node("ReduceMax", [in2], axes=[3], keepdims=1)   # [1,1,30,1]
    row3 = g.node("ReduceMax", [in3], axes=[3], keepdims=1)   # [1,1,30,1]
    Lw = _tri(g, lambda k, c: k <= c, W)
    Rw = _tri(g, lambda k, c: k >= c, W)
    Th = _tri(g, lambda r, k: k <= r, H)
    Bh = _tri(g, lambda r, k: k >= r, H)
    colmark = g.node("Max", [col2, col3])
    bc = g.node("Min", [g.node("MatMul", [colmark, Lw]),
                        g.node("MatMul", [colmark, Rw])])    # [1,1,1,30] between cols
    rowmark = g.node("Max", [row2, row3])
    br = g.node("Min", [g.node("MatMul", [Th, rowmark]),
                        g.node("MatMul", [Bh, rowmark])])    # [1,1,30,1] between rows
    horiz = g.node("Mul", [row2, bc])                     # [1,1,30,30] leg on row2
    vert = g.node("Mul", [col3, br])                      # [1,1,30,30] leg on col3
    path = g.node("Max", [horiz, vert])                   # [1,1,30,30]
    pathbg = g.node("Min", [path, bg])                    # 8 only on bg cells
    ch0 = g.node("Sub", [bg, pathbg])                     # bg off under the path
    zero = g.fconst([0.0] * (HEIGHT * WIDTH), [1, 1, HEIGHT, WIDTH])
    g.node("Concat",
           [ch0, zero, in2, in3, zero, zero, zero, zero, pathbg, zero],
           "output", axis=1)
    return _model(g.nodes, g.inits)


# ===========================================================================
# 190 diag-rays:  a 2x2 block has single "seed" cells (no orthogonal same-colour
#                 neighbour) at its diagonal corners; each seed shoots a diagonal
#                 ray outward (away from the block) to the grid edge.
# ===========================================================================
_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _ref_diag_rays(a):
    H, W = a.shape
    nz = (a != 0).astype(int)
    if not nz.any():
        return a.copy()
    color = int(np.bincount(a[a != 0]).argmax())

    def sh(m, dr, dc):
        o = np.zeros_like(m)
        rs = slice(max(0, -dr), H + min(0, -dr)); rd = slice(max(0, dr), H + min(0, dr))
        cs = slice(max(0, -dc), W + min(0, -dc)); cd = slice(max(0, dc), W + min(0, dc))
        o[rd, cd] = m[rs, cs]
        return o

    orth = sum(sh(nz, dr, dc) for dr, dc in _ORTH)
    seed = nz * (orth == 0)
    allray = np.zeros((H, W), int)
    for dr, dc in _DIAG:
        seedD = seed * sh(nz, dr, dc)
        for k in range(1, max(H, W)):
            allray = allray | sh(seedD, k * dr, k * dc)
    o = a.copy()
    o[(allray == 1) & (a == 0)] = color
    return o


def _build_diag_rays(H, W):
    g = _G()
    L = max(H, W) - 1
    in_small = _slice(g, "input", [0, 0], [H, W], [2, 3])         # [1,10,H,W]
    allsum = g.node("ReduceSum", [in_small], axes=[1], keepdims=1)  # [1,1,H,W]
    bg = _slice(g, in_small, [0], [1], [1])                       # [1,1,H,W]
    nz = g.node("Sub", [allsum, bg])                              # markers
    orth = _conv3(g, nz, [0, 1, 0, 1, 0, 1, 0, 1, 0])            # 4-neighbour count
    seedf = g.node("Cast", [g.node("Less", [orth, g.fconst([0.5], [1])])], to=DATA_TYPE)
    seed = g.node("Mul", [nz, seedf])                            # isolated cells
    rays = []
    for dr, dc in _DIAG:
        seedD = g.node("Mul", [seed, _shift(g, nz, dr, dc, H, W)])
        ksz = 2 * L + 1
        kern = [0.0] * (ksz * ksz)
        for k in range(1, L + 1):
            kern[(L - k * dr) * ksz + (L - k * dc)] = 1.0
        w = g.fconst(kern, [1, 1, ksz, ksz])
        rays.append(g.node("Conv", [seedD, w], kernel_shape=[ksz, ksz],
                           pads=[L, L, L, L]))
    allray = rays[0]
    for r in rays[1:]:
        allray = g.node("Max", [allray, r])
    raybg = g.node("Min", [allray, bg])                          # ray cells (bg only)
    cvec = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)
    m01 = g.fconst([0.0] + [1.0] * (CHANNELS - 1), [1, CHANNELS, 1, 1])
    e0 = g.fconst([1.0] + [0.0] * (CHANNELS - 1), [1, CHANNELS, 1, 1])
    P = g.node("Sub", [g.node("Mul", [cvec, m01]), e0])          # [1,10,1,1]
    delta = g.node("Mul", [raybg, P])                            # [1,10,H,W]
    out_small = g.node("Add", [in_small, delta])
    g.node("Pad", [out_small], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 168 L-rays:  an L-tromino (3 cells of a 2x2 box) shoots a diagonal ray out of
#              its missing corner (away from the filled cells) to the grid edge.
# ===========================================================================
def _ref_l_rays(a):
    H, W = a.shape
    nz = (a != 0).astype(int)
    if not nz.any():
        return a.copy()
    color = int(np.bincount(a[a != 0]).argmax())
    bg = (a == 0).astype(int)

    def sh(m, dr, dc):
        o = np.zeros_like(m)
        rs = slice(max(0, -dr), H + min(0, -dr)); rd = slice(max(0, dr), H + min(0, dr))
        cs = slice(max(0, -dc), W + min(0, -dc)); cd = slice(max(0, dc), W + min(0, dc))
        o[rd, cd] = m[rs, cs]
        return o

    allray = np.zeros((H, W), int)
    for qr, qc in _DIAG:
        Lm = bg * sh(nz, -qr, 0) * sh(nz, 0, -qc) * sh(nz, -qr, -qc)
        for k in range(1, max(H, W)):
            allray = allray | sh(Lm, -k * qr, -k * qc)
    o = a.copy()
    o[(allray == 1) & (a == 0)] = color
    return o


def _build_l_rays(H, W):
    g = _G()
    L = max(H, W) - 1
    in_small = _slice(g, "input", [0, 0], [H, W], [2, 3])
    allsum = g.node("ReduceSum", [in_small], axes=[1], keepdims=1)
    bg = _slice(g, in_small, [0], [1], [1])
    nz = g.node("Sub", [allsum, bg])
    rays = []
    for qr, qc in _DIAG:
        Lm = g.node("Mul", [bg, _shift(g, nz, -qr, 0, H, W)])
        Lm = g.node("Mul", [Lm, _shift(g, nz, 0, -qc, H, W)])
        Lm = g.node("Mul", [Lm, _shift(g, nz, -qr, -qc, H, W)])
        ksz = 2 * L + 1
        kern = [0.0] * (ksz * ksz)
        for k in range(1, L + 1):
            kern[(L + k * qr) * ksz + (L + k * qc)] = 1.0
        w = g.fconst(kern, [1, 1, ksz, ksz])
        rays.append(g.node("Conv", [Lm, w], kernel_shape=[ksz, ksz],
                           pads=[L, L, L, L]))
    allray = rays[0]
    for r in rays[1:]:
        allray = g.node("Max", [allray, r])
    raybg = g.node("Min", [allray, bg])
    cvec = g.node("ReduceMax", ["input"], axes=[2, 3], keepdims=1)
    m01 = g.fconst([0.0] + [1.0] * (CHANNELS - 1), [1, CHANNELS, 1, 1])
    e0 = g.fconst([1.0] + [0.0] * (CHANNELS - 1), [1, CHANNELS, 1, 1])
    P = g.node("Sub", [g.node("Mul", [cvec, m01]), e0])
    delta = g.node("Mul", [raybg, P])
    out_small = g.node("Add", [in_small, delta])
    g.node("Pad", [out_small], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W])
    return _model(g.nodes, g.inits)


# ===========================================================================
# 327 diag-smear:  NxN -> 2Nx2N; every cell is copied down-right along its main
#                  diagonal (a diagonal forward-fill / smear).
# ===========================================================================
def _ref_diag_smear(a):
    H, W = a.shape
    OH, OW = 2 * H, 2 * W
    o = np.zeros((OH, OW), int)
    for i in range(OH):
        for j in range(OW):
            for t in range(min(i, j) + 1):
                r, c = i - t, j - t
                if 0 <= r < H and 0 <= c < W and a[r, c] != 0:
                    o[i, j] = a[r, c]
                    break
    return o


def _build_diag_smear(N):
    g = _G()
    OUT = 2 * N
    L = OUT - 1
    ksz = 2 * L + 1
    colors = _slice(g, "input", [1, 0, 0], [CHANNELS, N, N], [1, 2, 3])  # [1,9,N,N]
    padded = g.node("Pad", [colors], mode="constant", value=0.0,
                    pads=[0, 0, 0, 0, 0, 0, N, N])                       # [1,9,2N,2N]
    kern = [0.0] * (CHANNELS - 1) * ksz * ksz
    plane = ksz * ksz
    for grp in range(CHANNELS - 1):
        for k in range(0, L + 1):
            kern[grp * plane + (L - k) * ksz + (L - k)] = 1.0
    w = g.fconst(kern, [CHANNELS - 1, 1, ksz, ksz])
    conv = g.node("Conv", [padded, w], kernel_shape=[ksz, ksz],
                  pads=[L, L, L, L], group=CHANNELS - 1)                 # [1,9,2N,2N]
    anyfill = g.node("ReduceSum", [conv], axes=[1], keepdims=1)
    one = g.fconst([1.0], [1])
    ch0 = g.node("Sub", [one, g.node("Min", [anyfill, one])])           # bg where empty
    full = g.node("Concat", [ch0, conv], axis=1)                        # [1,10,2N,2N]
    g.node("Pad", [full], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - OUT, WIDTH - OUT])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# detection / candidate generation                                            #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    def emit(name, fn):
        try:
            out.append((name, fn()))
        except Exception:
            pass

    shapes = {a.shape for a, _ in prs}
    same_size = all(a.shape == b.shape for a, b in prs)
    changes = any(not np.array_equal(a, b) for a, b in prs)

    # ---- 356 connect collinear markers --------------------------------------
    if changes and same_size and len(shapes) == 1:
        H, W = next(iter(shapes))
        if 1 < H <= HEIGHT and 1 < W <= WIDTH and \
                all(np.array_equal(_ref_connect_collinear(a), b) for a, b in prs):
            emit("connect_collinear", lambda: _build_connect_collinear(H, W))

    # ---- 278 frame around 2-dominoes ----------------------------------------
    if changes and same_size and \
            all(not (a == 3).any() for a, _ in prs) and \
            all(np.array_equal(_ref_frame_dominoes(a), b) for a, b in prs):
        emit("frame_dominoes", _build_frame_dominoes)

    # ---- 38 count 2x2 blocks -> 1x5 bar -------------------------------------
    if all(b.shape == (1, 5) for _, b in prs) and \
            all(np.array_equal(_ref_count_bar(a), b) for a, b in prs):
        emit("count_bar", _build_count_bar)

    # ---- 246 L-connect a 2 and a 3 with colour 8 ----------------------------
    if changes and same_size and \
            all(not (a == 8).any() for a, _ in prs) and \
            all((_ref_lconnect(a) is not None and np.array_equal(_ref_lconnect(a), b))
                for a, b in prs):
        emit("lconnect", _build_lconnect)

    # ---- 190 diagonal rays from isolated seeds ------------------------------
    if changes and same_size and len(shapes) == 1 and \
            all(len(set(np.unique(a)) - {0}) <= 1 for a, _ in prs) and \
            all(np.array_equal(_ref_diag_rays(a), b) for a, b in prs):
        H, W = next(iter(shapes))
        if 1 < H <= HEIGHT and 1 < W <= WIDTH:
            emit("diag_rays", lambda: _build_diag_rays(H, W))

    # ---- 168 diagonal rays from L-tromino missing corners -------------------
    if changes and same_size and len(shapes) == 1 and \
            all(len(set(np.unique(a)) - {0}) <= 1 for a, _ in prs) and \
            all(np.array_equal(_ref_l_rays(a), b) for a, b in prs):
        H, W = next(iter(shapes))
        if 1 < H <= HEIGHT and 1 < W <= WIDTH:
            emit("l_rays", lambda: _build_l_rays(H, W))

    # ---- 327 diagonal smear NxN -> 2Nx2N ------------------------------------
    in_sq = {a.shape for a, _ in prs}
    out_dbl = all(b.shape == (2 * a.shape[0], 2 * a.shape[1]) and a.shape[0] == a.shape[1]
                  for a, b in prs)
    if changes and len(in_sq) == 1 and out_dbl:
        N = next(iter(in_sq))[0]
        if 1 < 2 * N <= HEIGHT and \
                all(np.array_equal(_ref_diag_smear(a), b) for a, b in prs):
            emit("diag_smear", lambda: _build_diag_smear(N))

    return out
