"""family_rb324 — task324 / d07ae81c : "diagonals from dots".

Rule (decoded + validated against the generator task_d07ae81c.py):
  The true grid is a solid background bg (=bgcolors[0]) overlaid with full-width
  horizontal STRIPES and full-height vertical STRIPES in a single stripe colour
  s (=bgcolors[1]).  2-3 "dots" sit on the grid; each dot is recoloured colors[0]
  if it lands on plain bg, colors[1] if it lands on a stripe.
  INPUT  = bg + stripes + the recoloured dots.
  OUTPUT = for EVERY dot, BOTH of its diagonals (all cells with r+c == dr+dc OR
  r-c == dr-dc) are drawn; each drawn cell becomes colors[0] if the underlying
  cell is plain bg, or colors[1] if it is on a stripe.  Stripes stay; dots lie on
  their own diagonals so they are included.

Colour identification (robust, no count-ordering assumptions):
  * rows 0,1 and cols 0,1 are ALWAYS non-stripe (the generator starts stripes at
    index >= 2), so the top-left 2x2 block is underlying bg; at most one of its 4
    cells can be a dot (dots are pairwise Chebyshev->1 apart via
    remove_diagonal_neighbors), hence >= 3 of the 4 are bg  ->  bg = majority
    colour of the top-left 2x2 block.
  * a stripe fills a whole row/col, so a row is a stripe-row iff it contains NO bg
    cell (a non-stripe row always keeps a bg cell at col 0 or 1 — both can't be
    dots at once); same for columns via rows 0,1.  underlying = s on any
    stripe-row/stripe-col cell, else bg.
  * the two dot colours are the remaining present colours; s (the stripe colour)
    is the most common non-bg colour (a stripe is >= ~18 cells, a dot colour <= 2).
    A dot colour whose cells sit on a stripe is colors[1]; on bg is colors[0].

This makes input->output a total function; the numpy _ref is EXACT on
train+test+arc-gen and on fresh generator samples.

ONNX (opset-10, static [1,10,30,30], no banned ops): all per-cell masking.
Colours are selected as one-hot [1,10,1,1] vectors via ReduceSum/ReduceMax over
the input one-hot channels; stripe rows/cols via row/col bg-sum thresholds;
diagonals via two static [1,59,30,30] "same anti-diagonal / same diagonal" bucket
tensors reduced twice (scatter dots -> per-diagonal count -> gather back). Output
is a sum of four (cellmask x colour-onehot) products — inherently one-hot.
"""
import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F = DATA_TYPE
H30 = 30
KD = 2 * H30 - 1  # 59 diagonal buckets


# --------------------------------------------------------------------------- #
# numpy reference (gate + ground truth mirror)                                 #
# --------------------------------------------------------------------------- #
def _ref(a):
    a = np.array(a, int)
    if a.ndim != 2 or min(a.shape) < 2:
        return None
    H, W = a.shape
    corner = [a[0, 0], a[0, 1], a[1, 0], a[1, 1]]
    vals, cnts = np.unique(corner, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    uv, uc = np.unique(a, return_counts=True)
    cnt = {int(v): int(c) for v, c in zip(uv, uc)}
    nonbg = [v for v in cnt if v != bg]
    if not nonbg:
        return None
    s = max(nonbg, key=lambda v: cnt[v])
    dot_colors = [v for v in cnt if v != bg and v != s]
    bgmap = (a == bg)
    stripe_row = ~bgmap.any(axis=1)
    stripe_col = ~bgmap.any(axis=0)
    stripe_cell = stripe_row[:, None] | stripe_col[None, :]
    dotmask = np.isin(a, dot_colors)
    dr, dc = np.where(dotmask)
    c0 = c1 = None
    for r, c in zip(dr.tolist(), dc.tolist()):
        if stripe_cell[r, c]:
            c1 = int(a[r, c])
        else:
            c0 = int(a[r, c])
    sums = set((r + c) for r, c in zip(dr.tolist(), dc.tolist()))
    diffs = set((r - c) for r, c in zip(dr.tolist(), dc.tolist()))
    R, C = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    D = np.isin(R + C, list(sums)) | np.isin(R - C, list(diffs))
    out = np.where(stripe_cell, s, bg).astype(int)
    if c1 is not None:
        out = np.where(D & stripe_cell, c1, out)
    if c0 is not None:
        out = np.where(D & ~stripe_cell, c0, out)
    return out


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


def _gt(g, a, b):
    return g.nd("Cast", [g.nd("Greater", [a, b])], to=F)


def _lt(g, a, b):
    return g.nd("Cast", [g.nd("Less", [a, b])], to=F)


def _eqm(g, a, b):
    return _lt(g, g.nd("Abs", [g.nd("Sub", [a, b])]), g.half)


# =========================================================================== #
# ONNX builder                                                                 #
# =========================================================================== #
def build_324():
    g = _G()
    g.half = g.f1(0.5)
    g.one = g.f1(1.0)

    # diagonal bucket tensors ------------------------------------------------ #
    rr = np.arange(H30)
    R = rr[:, None] * np.ones((1, H30))
    C = np.ones((H30, 1)) * rr[None, :]
    sumeq = np.zeros((1, KD, H30, H30), np.float32)
    difeq = np.zeros((1, KD, H30, H30), np.float32)
    for k in range(KD):
        sumeq[0, k] = ((R + C) == k)
        difeq[0, k] = ((R - C + (H30 - 1)) == k)
    SUMEQ = g.f([1, KD, H30, H30], sumeq)
    DIFEQ = g.f([1, KD, H30, H30], difeq)

    cmask = np.zeros((1, 1, H30, H30), np.float32)
    cmask[0, 0, 0, 0] = cmask[0, 0, 0, 1] = cmask[0, 0, 1, 0] = cmask[0, 0, 1, 1] = 1.0
    CORNER = g.f([1, 1, H30, H30], cmask)

    X = "input"

    # in-grid mask + per-colour counts -------------------------------------- #
    present = g.nd("ReduceSum", [X], axes=[1], keepdims=1)          # [1,1,30,30]
    counts = g.nd("ReduceSum", [X], axes=[2, 3], keepdims=1)         # [1,10,1,1]

    # bg = majority colour of top-left 2x2 --------------------------------- #
    cc = g.nd("ReduceSum", [g.nd("Mul", [X, CORNER])], axes=[2, 3], keepdims=1)
    maxcc = g.nd("ReduceMax", [cc], axes=[1], keepdims=1)
    BG = _eqm(g, cc, maxcc)                                          # [1,10,1,1]

    # s = most common non-bg colour ---------------------------------------- #
    cnb = g.nd("Mul", [counts, g.nd("Sub", [g.one, BG])])
    maxnb = g.nd("ReduceMax", [cnb], axes=[1], keepdims=1)
    S = _eqm(g, cnb, maxnb)

    present_sel = _gt(g, counts, g.half)
    DOT = g.nd("Mul", [present_sel, g.nd("Mul", [g.nd("Sub", [g.one, BG]),
                                                 g.nd("Sub", [g.one, S])])])

    # colour maps ----------------------------------------------------------- #
    bg_map = g.nd("ReduceSum", [g.nd("Mul", [X, BG])], axes=[1], keepdims=1)
    dot_map = g.nd("ReduceSum", [g.nd("Mul", [X, DOT])], axes=[1], keepdims=1)

    # stripe rows / cols ---------------------------------------------------- #
    bg_row = g.nd("ReduceSum", [bg_map], axes=[3], keepdims=1)       # [1,1,30,1]
    row_pres = g.nd("ReduceMax", [present], axes=[3], keepdims=1)
    stripe_row = g.nd("Mul", [_lt(g, bg_row, g.half), row_pres])
    bg_col = g.nd("ReduceSum", [bg_map], axes=[2], keepdims=1)       # [1,1,1,30]
    col_pres = g.nd("ReduceMax", [present], axes=[2], keepdims=1)
    stripe_col = g.nd("Mul", [_lt(g, bg_col, g.half), col_pres])
    stripe_any = g.nd("Add", [stripe_row, stripe_col])              # [1,1,30,30]
    stripe_cell = g.nd("Mul", [present, _gt(g, stripe_any, g.half)])
    bg_cell = g.nd("Mul", [present, _lt(g, stripe_any, g.half)])

    # split dot colours into colors[0] (on bg) / colors[1] (on stripe) ------ #
    on_str = g.nd("ReduceSum", [g.nd("Mul", [X, stripe_cell])], axes=[2, 3], keepdims=1)
    C1 = g.nd("Mul", [DOT, _gt(g, on_str, g.half)])
    C0 = g.nd("Mul", [DOT, _lt(g, on_str, g.half)])

    # diagonals from dots --------------------------------------------------- #
    anti_has = g.nd("ReduceSum", [g.nd("Mul", [dot_map, SUMEQ])], axes=[2, 3], keepdims=1)
    hit_anti = g.nd("ReduceSum", [g.nd("Mul", [anti_has, SUMEQ])], axes=[1], keepdims=1)
    diff_has = g.nd("ReduceSum", [g.nd("Mul", [dot_map, DIFEQ])], axes=[2, 3], keepdims=1)
    hit_diff = g.nd("ReduceSum", [g.nd("Mul", [diff_has, DIFEQ])], axes=[1], keepdims=1)
    D = _gt(g, g.nd("Add", [hit_anti, hit_diff]), g.half)            # [1,1,30,30]
    Dp = g.nd("Mul", [D, present])
    notDp = g.nd("Sub", [g.one, Dp])

    # compose output one-hot ------------------------------------------------ #
    m_c0 = g.nd("Mul", [Dp, bg_cell])
    m_c1 = g.nd("Mul", [Dp, stripe_cell])
    m_bg = g.nd("Mul", [bg_cell, notDp])
    m_s = g.nd("Mul", [stripe_cell, notDp])

    o0 = g.nd("Mul", [m_c0, C0])
    o1 = g.nd("Mul", [m_c1, C1])
    o2 = g.nd("Mul", [m_bg, BG])
    o3 = g.nd("Mul", [m_s, S])
    g.nd("Add", [g.nd("Add", [o0, o1]), g.nd("Add", [o2, o3])], "output")
    return _model(g, "rb324")


# =========================================================================== #
# detection / candidates                                                       #
# =========================================================================== #
def _pairs(ex):
    out = []
    for split in ("train", "test"):
        for e in ex.get(split, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _matches(prs):
    if not prs:
        return False
    for a, b in prs:
        try:
            o = _ref(a)
        except Exception:
            return False
        if o is None or o.shape != b.shape or not np.array_equal(o, b):
            return False
    return True


def candidates(ex):
    if not _matches(_pairs(ex)):
        return []
    try:
        return [("rb324", build_324())]
    except Exception:
        return []
