"""REPEAT / SCALE BY A (CONSTANT) COUNT -- tiling and self-fractal (opset 10).

Every rule here replicates the input grid a *fixed* number of times, laid out in a
constant R x C block grid, and is realised with static ``Slice`` / ``Tile`` /
``Concat`` / ``Resize`` plus a final origin-anchored ``Pad`` back to 30x30.  The
one-hot tensor is zero-padded to 30x30 with the grid anchored at (0,0); a plain
``Tile`` over the full 30x30 tensor would repeat at period 30 (not the real grid
period), so a static graph can only express a repeat when the input size is
CONSTANT across every train/test/arc-gen pair.  We therefore ``Slice`` out the
exact h x w content block, build the layout from it, and ``Pad`` the assembled
(R*h) x (C*w) result back to the top-left of a 30x30 frame.

Rules (the matching one is inferred from the pairs)
--------------------------------------------------
  blocktile       Output is the input replicated in a constant R x C grid, each
                  block being a fixed per-position transform of the input
                  (identity / horizontal flip / vertical flip / 180 / transpose /
                  90 / 270, or a blank background tile).  Plain replication uses a
                  single ``Tile``; flipped / rotated layouts use reverse ``Slice``
                  + ``Transpose`` + ``Concat``.  The factor R, C and the
                  per-position transform map are constant -> a static graph.

  upscale         Magnify every cell into a constant r x c block (nearest /
                  pixel upscale).  This is size-INDEPENDENT (a single ``Resize``
                  over the zero-padded tensor keeps the content anchored at the
                  origin and the padding zero), so it generalises to any input
                  size as long as the (constant) factor is the same everywhere.

  fractal         The classic ARC self-fractal: the output is (h*h) x (w*w) and
                  the input MOTIF is STAMPED ONCE PER MARKER cell -- block (bi,bj)
                  is a copy of the whole input where the input cell (bi,bj) is a
                  "marker", otherwise that block is plain background.  The marker
                  predicate is detected structurally:

                    nonbg     -> stamp where the input cell is non-background;
                    majority  -> stamp where the cell holds the most-frequent
                                 non-bg colour (data-dependent, via ArgMax);
                    minority  -> the least-frequent non-bg colour;
                    color C   -> stamp where the cell equals a fixed colour C
                                 (used when the marker colour is constant across
                                 every available pair).

                  Realised by upscaling the (small) marker plane with ``Resize``
                  (nearest, factor h/w -> one value per block), ``Tile``-ing the
                  motif into the (h*h) x (w*w) frame, and selecting copy-vs-
                  background per block; non-marker blocks become real background
                  (channel 0 = 1), not padding.

Detection mirrors the ONNX semantics exactly and only emits a candidate when it
reproduces EVERY available pair (the grader's gate), so wrong hypotheses are
dropped before scoring; the predicate forms (nonbg / majority / minority / fixed
colour) are structural so they generalise to the held-out set.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)            # full-axis reverse Slice sentinel
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

    def f(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         [float(v) for v in np.asarray(vals, np.float32).ravel()]))
        return nm

    def i1(self, vals):
        nm = self.name("i")
        vals = list(vals)
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], [int(v) for v in vals]))
        return nm

    def id(self, dims, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, list(dims),
                                         [int(v) for v in np.asarray(vals).ravel()]))
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


def _pad_out(g, src, Hh, Ww):
    """[1,10,Hh,Ww] -> [1,10,30,30], zero-padded, origin anchored."""
    if Hh == HEIGHT and Ww == WIDTH:
        g.node("Identity", [src], "output")
    else:
        g.node("Pad", [src], "output", mode="constant", value=0.0,
               pads=[0, 0, 0, 0, 0, 0, HEIGHT - Hh, WIDTH - Ww])


def _crop(g, h, w):
    """input[:, :, 0:h, 0:w] -> [1,10,h,w]."""
    return g.node("Slice", ["input", g.i1([0, 0]), g.i1([h, w]), g.i1([2, 3])])


# --------------------------------------------------------------------------- #
# numpy transforms (mirror the ONNX block transforms exactly)                 #
# --------------------------------------------------------------------------- #
def _transforms_np(a):
    h, w = a.shape
    d = {"ID": a, "FW": a[:, ::-1], "FH": a[::-1, :], "R180": a[::-1, ::-1]}
    if h == w:
        d["T"] = a.T
        d["R90"] = a.T[::-1, :]
        d["R270"] = a.T[:, ::-1]
    return d


# --------------------------------------------------------------------------- #
# block-tiling builder                                                        #
# --------------------------------------------------------------------------- #
def build_blocktile(grid, h, w, R, C):
    g = _G()
    M = _crop(g, h, w)
    flat = [v for row in grid for v in row]

    if set(flat) == {"ID"}:                              # plain replication
        asm = g.node("Tile", [M, g.i1([1, 1, R, C])])
    else:
        cache = {}

        def rev(t, axes, lens):
            return g.node("Slice", [t, g.i1([l - 1 for l in lens]),
                                    g.i1([_NEG] * len(axes)),
                                    g.i1(list(axes)), g.i1([-1] * len(axes))])

        def Tt():
            if "T" not in cache:
                cache["T"] = g.node("Transpose", [M], perm=[0, 1, 3, 2])
            return cache["T"]

        def var(nm):
            if nm in cache:
                return cache[nm]
            if nm == "ID":
                r = M
            elif nm == "FW":
                r = rev(M, [3], [w])
            elif nm == "FH":
                r = rev(M, [2], [h])
            elif nm == "R180":
                r = rev(M, [2, 3], [h, w])
            elif nm == "T":
                r = Tt()
            elif nm == "R90":                            # transpose then reverse rows
                r = rev(Tt(), [2], [w])
            elif nm == "R270":                           # transpose then reverse cols
                r = rev(Tt(), [3], [h])
            else:                                        # ZERO -> background tile
                realmask = g.node("ReduceSum", [M], axes=[1], keepdims=1)
                e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
                r = g.node("Mul", [realmask, e0])
            cache[nm] = r
            return r

        rows = []
        for bi in range(R):
            tiles = [var(grid[bi][bj]) for bj in range(C)]
            row = tiles[0] if C == 1 else g.node("Concat", tiles, axis=3)
            rows.append(row)
        asm = rows[0] if R == 1 else g.node("Concat", rows, axis=2)

    _pad_out(g, asm, R * h, C * w)
    return _model(g)


# --------------------------------------------------------------------------- #
# upscale builder (size-independent pixel magnification)                       #
# --------------------------------------------------------------------------- #
def build_upscale(r, c):
    g = _G()
    scales = g.f([4], [1.0, 1.0, float(r), float(c)])
    up = g.node("Resize", ["input", scales], mode="nearest")              # [1,10,30r,30c]
    g.node("Slice", [up, g.i1([0, 0]), g.i1([HEIGHT, WIDTH]), g.i1([2, 3])], "output")
    return _model(g)


# --------------------------------------------------------------------------- #
# fractal builder                                                             #
# --------------------------------------------------------------------------- #
def _mask_plane(g, M, pred, C):
    """[1,1,h,w] marker plane (1 where a copy of the motif should be stamped)."""
    if pred == "nonbg":
        ch = g.node("Slice", [M, g.i1([1]), g.i1([CHANNELS]), g.i1([1])])  # channels 1..9
        return g.node("ReduceSum", [ch], axes=[1], keepdims=1)
    if pred == "color":
        return g.node("Slice", [M, g.i1([C]), g.i1([C + 1]), g.i1([1])])

    counts = g.node("ReduceSum", [M], axes=[2, 3], keepdims=1)             # [1,10,1,1]
    if pred == "majority":
        bgneg = g.f([1, CHANNELS, 1, 1], [-_BIG] + [0.0] * (CHANNELS - 1))
        scored = g.node("Add", [counts, bgneg])
        amax = g.node("ArgMax", [scored], axis=1, keepdims=1)
    else:                                                                  # minority
        nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
        present = g.node("Cast", [g.node("Greater", [counts, g.f([1, 1, 1, 1], [0.5])])],
                         to=DATA_TYPE)
        present_nbg = g.node("Mul", [present, nbg])
        one = g.f([1, 1, 1, 1], [1.0])
        big = g.f([1, 1, 1, 1], [_BIG])
        push = g.node("Mul", [g.node("Sub", [one, present_nbg]), big])     # absent/bg -> huge
        scored = g.node("Add", [counts, push])
        neg = g.node("Mul", [scored, g.f([1, 1, 1, 1], [-1.0])])
        amax = g.node("ArgMax", [neg], axis=1, keepdims=1)
    idx = g.id([1, CHANNELS, 1, 1], list(range(CHANNELS)))
    gate = g.node("Cast", [g.node("Equal", [amax, idx])], to=DATA_TYPE)    # [1,10,1,1]
    gated = g.node("Mul", [M, gate])                                       # [1,10,h,w]
    return g.node("ReduceSum", [gated], axes=[1], keepdims=1)              # [1,1,h,w]


def build_fractal(pred, C, h, w):
    g = _G()
    M = _crop(g, h, w)                                   # [1,10,h,w] motif
    mask = _mask_plane(g, M, pred, C)                    # [1,1,h,w]
    scales = g.f([4], [1.0, 1.0, float(h), float(w)])
    mask_up = g.node("Resize", [mask, scales], mode="nearest")            # [1,1,h*h,w*w]
    tiled = g.node("Tile", [M, g.i1([1, 1, h, w])])                       # [1,10,h*h,w*w]
    term1 = g.node("Mul", [tiled, mask_up])
    e0 = g.f([1, CHANNELS, 1, 1], [1.0] + [0.0] * (CHANNELS - 1))
    one = g.f([1, 1, 1, 1], [1.0])
    inv = g.node("Sub", [one, mask_up])                                   # non-marker blocks
    term2 = g.node("Mul", [e0, inv])                                      # -> background
    res = g.node("Add", [term1, term2])
    _pad_out(g, res, h * h, w * w)
    return _model(g)


# --------------------------------------------------------------------------- #
# detection                                                                   #
# --------------------------------------------------------------------------- #
def _detect_blocktile(prs):
    shapes = {a.shape for a, _ in prs}
    if len(shapes) != 1:                                 # constant input size only
        return None
    h, w = next(iter(shapes))
    if h == 0 or w == 0:
        return None
    ratios = set()
    for a, b in prs:
        H, W = b.shape
        if H % h or W % w:
            return None
        ratios.add((H // h, W // w))
    if len(ratios) != 1:
        return None
    R, C = next(iter(ratios))
    if R * C < 2 or R * h > HEIGHT or C * w > WIDTH:
        return None
    names = ["ID", "FW", "FH", "R180"] + (["T", "R90", "R270"] if h == w else [])
    grid = [[None] * C for _ in range(R)]
    for bi in range(R):
        for bj in range(C):
            cand = set(names) | {"ZERO"}
            for a, b in prs:
                blk = b[bi * h:(bi + 1) * h, bj * w:(bj + 1) * w]
                tn = _transforms_np(a)
                ok = set(nm for nm in names
                         if tn[nm].shape == blk.shape and np.array_equal(tn[nm], blk))
                if (blk == 0).all():
                    ok.add("ZERO")
                cand &= ok
                if not cand:
                    return None
            for nm in names + ["ZERO"]:
                if nm in cand:
                    grid[bi][bj] = nm
                    break
    return grid, h, w, R, C


def _detect_upscale(prs):
    rcs = set()
    for a, b in prs:
        h, w = a.shape
        H, W = b.shape
        if h == 0 or w == 0 or H % h or W % w:
            return None
        rcs.add((H // h, W // w))
    if len(rcs) != 1:
        return None
    r, c = next(iter(rcs))
    if r * c < 2:
        return None
    for a, b in prs:
        if not np.array_equal(np.repeat(np.repeat(a, r, 0), c, 1), b):
            return None
    return r, c


def _sel(a, f):
    h, w = a.shape
    return {(bi, bj) for bi in range(h) for bj in range(w) if f(a[bi, bj])}


def _detect_fractal(prs):
    shapes = {a.shape for a, _ in prs}
    if len(shapes) != 1:                                 # constant input size only
        return None
    h, w = next(iter(shapes))
    if h == 0 or w == 0 or h * h > HEIGHT or w * w > WIDTH:
        return None
    for a, b in prs:
        if b.shape != (h * h, w * w):
            return None
    css = []
    for a, b in prs:
        cs = set()
        for bi in range(h):
            for bj in range(w):
                blk = b[bi * h:(bi + 1) * h, bj * w:(bj + 1) * w]
                if np.array_equal(blk, a):
                    cs.add((bi, bj))
                elif (blk == 0).all():
                    pass
                else:
                    return None
        css.append(cs)
    if not any(css):                                     # nothing stamped -> not fractal
        return None

    def matches(predfn):
        for (a, _), cs in zip(prs, css):
            ps = predfn(a)
            if ps is None or ps != cs:
                return False
        return True

    if matches(lambda a: _sel(a, lambda c: c != 0)):
        return ("nonbg", None, h, w)

    def _extreme(a, want_max):
        nbg = {c: int((a == c).sum()) for c in range(1, CHANNELS) if (a == c).any()}
        if not nbg:
            return set()
        m = (max if want_max else min)(nbg.values())
        win = [c for c, v in nbg.items() if v == m]
        if len(win) != 1:
            return None
        return _sel(a, lambda c, W=win[0]: c == W)

    if matches(lambda a: _extreme(a, True)):
        return ("majority", None, h, w)
    if matches(lambda a: _extreme(a, False)):
        return ("minority", None, h, w)
    for C in range(1, CHANNELS):
        if matches(lambda a, C=C: _sel(a, lambda c: c == C)):
            return ("color", C, h, w)
    return None


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
    if all(np.array_equal(a, b) for a, b in prs):        # identity -> not our family
        return []

    out = []

    # ---- self-fractal: stamp the motif once per marker cell ----------------- #
    fr = _detect_fractal(prs)
    if fr:
        pred, C, h, w = fr
        tag = f"color{C}" if pred == "color" else pred
        _emit(out, f"fractal_{tag}_{h}x{w}",
              lambda pred=pred, C=C, h=h, w=w: build_fractal(pred, C, h, w))

    # ---- constant R x C block tiling (plain / mirror / rotate) -------------- #
    bt = _detect_blocktile(prs)
    if bt:
        grid, h, w, R, C = bt
        flat = "_".join(v for row in grid for v in row)
        _emit(out, f"tile_{R}x{C}_{flat}",
              lambda grid=grid, h=h, w=w, R=R, C=C: build_blocktile(grid, h, w, R, C))

    # ---- size-independent integer upscale (magnify each cell) --------------- #
    up = _detect_upscale(prs)
    if up:
        r, c = up
        _emit(out, f"upscale_{r}x{c}", lambda r=r, c=c: build_upscale(r, c))

    return out
