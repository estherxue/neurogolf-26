"""SYMMETRY GENERATION / QUADRANT PASTE (origin-anchored, size-changing, opset 10).

The output is LARGER than the input and is a regular n x m mosaic whose blocks are
DIHEDRAL transforms (the 8 reflections/rotations) of the single input grid, with
the assembled (n*h) x (m*w) region anchored at the top-left of the 30x30 canvas.
This covers the classic 2x2 "kaleidoscope" (id|lr / ud|r180), the 4-fold ROTATIONAL
and POINT-SYMMETRIC 2x2 mosaics (id|rot270 / rot90|rot180), the 1x2 / 2x1 single
mirror, and larger symmetric mosaics such as a 3x2 reflection cross.

How it differs from family_mirror_concat
-----------------------------------------
* it is not limited to n,m in {1,2}; it detects general (small) n x m layouts, so
  it picks up symmetric mosaics like the 3x2 reflection cross that the 2x2-only
  family cannot express;
* for the 180-degree-symmetric 2x2 mosaics it uses a strictly CHEAPER construction
  (build the top strip [TL|TR], then take its 180-rotation as the bottom strip),
  which materialises one fewer half-size intermediate than the per-block build and
  therefore scores higher.

Realisability / origin safety (see CONTEXT padding gotcha)
---------------------------------------------------------
The one-hot tensor is zero-padded to 30x30 with the grid at (0,0) and grid sizes
VARY per example.  A reflect-paste places the mirrored copies at column/row offsets
that depend on the (data-dependent) input size, so a STATIC graph can only realise
it when the input size is CONSTANT across every split we can see
(train+test+arc-gen).  We therefore:

  * Slice the exact h x w content block out of the (constant-size) top-left region;
  * build each dihedral block FROM that block with reverse-Slice (steps=-1) and
    Transpose (perm [0,1,3,2]) -- both origin-preserving;
  * Concat the blocks into the n x m layout and Pad the assembled region back to
    30x30, so everything stays anchored at (0,0) for the (fixed) grid size.

Detection mirrors the ONNX semantics exactly and validates EVERY available pair
(train+test+arc-gen, the grader's gate) before emitting, so wrong hypotheses are
dropped; the rule is structural (one transform grid for all pairs), so it
generalises to the held-out arc-gen.  At least one block must be a real reflection
(else it is plain replication -> the tiling family's job).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)            # "to the beginning" sentinel for a reverse Slice

# variant name -> numpy dihedral transform of a raw (h, w) grid.  The plain flips
# preserve (h, w); the transpose-based ones swap to (w, h) and so only match a
# block when the input is square (the shape check in _detect rejects them
# otherwise).  Together these are the eight elements of the dihedral group:
#   id        identity                T       main-diagonal mirror (transpose)
#   lr        horizontal flip         Tlr     rot270 (transpose then col-flip)
#   ud        vertical flip           Tud     rot90  (transpose then row-flip)
#   r180      rot180                  Tr180   anti-diagonal mirror
_VAR = {
    "id":    lambda a: a,
    "lr":    lambda a: a[:, ::-1],
    "ud":    lambda a: a[::-1, :],
    "r180":  lambda a: a[::-1, ::-1],
    "T":     lambda a: a.T,
    "Tlr":   lambda a: a.T[:, ::-1],
    "Tud":   lambda a: a.T[::-1, :],
    "Tr180": lambda a: a.T[::-1, ::-1],
}

_MAXNM = 6                   # search bound for the layout factor (n*h<=30 caps it)


# --------------------------------------------------------------------------- #
# pairs                                                                        #
# --------------------------------------------------------------------------- #
def _pairs(ex, splits=("train", "test")):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            a = np.array(e["input"]); b = np.array(e["output"])
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > HEIGHT or max(b.shape) > HEIGHT:
                continue
            out.append((a, b))
    return out


# --------------------------------------------------------------------------- #
# detection: infer (n, m, grid-of-variant-names)                              #
# --------------------------------------------------------------------------- #
def _detect(prs):
    """Return (n, m, grid) for a consistent dihedral mosaic, or None.

    grid[i][j] is a variant name reproducing block (i,j) of EVERY pair's output.
    The input need not be the top-left block; the build slices the content and
    pads the bottom-right, so the assembled region is always origin-anchored."""
    nms = set()
    for a, b in prs:
        h, w = a.shape
        H, W = b.shape
        if h == 0 or w == 0 or H % h or W % w:
            return None
        n, m = H // h, W // w
        if n < 1 or m < 1 or n > _MAXNM or m > _MAXNM or (n, m) == (1, 1):
            return None
        nms.add((n, m))
    if len(nms) != 1:
        return None
    n, m = nms.pop()

    grid = [[None] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            found = None
            for vn, vf in _VAR.items():
                ok = True
                for a, b in prs:
                    h, w = a.shape
                    blk = b[i * h:(i + 1) * h, j * w:(j + 1) * w]
                    t = vf(a)
                    if t.shape != blk.shape or not np.array_equal(t, blk):
                        ok = False
                        break
                if ok:
                    found = vn
                    break
            if found is None:
                return None
            grid[i][j] = found
    return n, m, grid


# --------------------------------------------------------------------------- #
# numpy references (mirror the ONNX builders for validation)                   #
# --------------------------------------------------------------------------- #
def _ref_general(a, n, m, grid):
    """Place _VAR[grid[i][j]](a) into an n x m block layout."""
    rows = []
    for i in range(n):
        rows.append(np.concatenate([_VAR[grid[i][j]](a) for j in range(m)], axis=1))
    return np.concatenate(rows, axis=0)


def _ref_pointsym_2x2(a, grid):
    """Cheap 2x2 build: top = [TL|TR], bottom = rot180(top)."""
    top = np.concatenate([_VAR[grid[0][0]](a), _VAR[grid[0][1]](a)], axis=1)
    bot = top[::-1, ::-1]
    return np.concatenate([top, bot], axis=0)


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                       #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def init(self, dtype, dims, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, dtype, list(dims), list(vals)))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _reverse(g, src, axes, dims):
    """Slice that reverses `src` (steps=-1) along the given spatial axes."""
    starts = g.init(INT64, [len(axes)], [dims[a] - 1 for a in axes])
    ends = g.init(INT64, [len(axes)], [_NEG] * len(axes))
    ax = g.init(INT64, [len(axes)], list(axes))
    steps = g.init(INT64, [len(axes)], [-1] * len(axes))
    return g.node("Slice", [src, starts, ends, ax, steps])


def _content(g, h, w):
    """Slice the origin-anchored h x w content block input[:, :, :h, :w]."""
    cs = g.init(INT64, [2], [0, 0])
    ce = g.init(INT64, [2], [h, w])
    ca = g.init(INT64, [2], [2, 3])
    return g.node("Slice", ["input", cs, ce, ca])


def _make_variant_fn(g, c, h, w):
    """Return a closure name->tensor that builds each dihedral block from c,
    caching shared sub-results (transpose, single flips)."""
    cache = {"id": c}

    def variant(vn):
        if vn in cache:
            return cache[vn]
        if vn == "lr":
            r = _reverse(g, c, [3], {2: h, 3: w})
        elif vn == "ud":
            r = _reverse(g, c, [2], {2: h, 3: w})
        elif vn == "r180":
            r = _reverse(g, c, [2, 3], {2: h, 3: w})
        elif vn.startswith("T"):
            t = cache.get("T")
            if t is None:
                t = g.node("Transpose", [c], perm=[0, 1, 3, 2])
                cache["T"] = t
            dims = {2: w, 3: h}                 # transposed block is (w, h)
            if vn == "T":
                r = t
            elif vn == "Tlr":
                r = _reverse(g, t, [3], dims)
            elif vn == "Tud":
                r = _reverse(g, t, [2], dims)
            else:                               # Tr180
                r = _reverse(g, t, [2, 3], dims)
        else:
            raise ValueError(vn)
        cache[vn] = r
        return r

    return variant


def _finish(g, asm, nh, mw):
    """asm is the [1,10,nh,mw] assembled region; emit it as 'output' (free) when
    it already fills the canvas, else Pad it back to 30x30 (origin anchored)."""
    if nh == HEIGHT and mw == WIDTH:
        g.nodes[-1].output[0] = "output"       # asm node IS the output
    else:
        g.node("Pad", [asm], "output", mode="constant", value=0.0,
               pads=[0, 0, 0, 0, 0, 0, HEIGHT - nh, WIDTH - mw])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# ONNX builders                                                                #
# --------------------------------------------------------------------------- #
def _build_general(h, w, n, m, grid):
    """Per-block construction: build each dihedral block, Concat into n x m."""
    g = _G()
    c = _content(g, h, w)
    variant = _make_variant_fn(g, c, h, w)
    rows = []
    for i in range(n):
        tiles = [variant(grid[i][j]) for j in range(m)]
        row = tiles[0] if m == 1 else g.node("Concat", tiles, axis=3)
        rows.append(row)
    asm = rows[0] if n == 1 else g.node("Concat", rows, axis=2)
    return _finish(g, asm, n * h, m * w)


def _build_pointsym_2x2(h, w, grid):
    """Cheaper 2x2: build the top strip [TL|TR] then its 180-rotation as the
    bottom strip (valid iff the mosaic is 180-degree symmetric)."""
    g = _G()
    c = _content(g, h, w)
    variant = _make_variant_fn(g, c, h, w)
    top = g.node("Concat", [variant(grid[0][0]), variant(grid[0][1])], axis=3)
    bot = _reverse(g, top, [2, 3], {2: h, 3: 2 * w})       # rot180 of the strip
    asm = g.node("Concat", [top, bot], axis=2)
    return _finish(g, asm, 2 * h, 2 * w)


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex, ("train", "test"))
    if not prs:
        return []
    if all(a.shape == b.shape for a, b in prs):           # not size-changing
        return []

    det = _detect(prs)
    if det is None:
        return []
    n, m, grid = det
    if all(v == "id" for row in grid for v in row):       # plain replication
        return []

    # a static reflect-paste needs ONE constant input size across all splits.
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    shapes = {a.shape for a, _ in allp}
    if len(shapes) != 1:
        return []
    h, w = next(iter(shapes))
    if h * w == 0 or n * h > HEIGHT or m * w > WIDTH:
        return []

    # validate the per-block reconstruction on EVERY available pair (grader gate).
    for a, b in allp:
        if a.shape != (h, w):
            return []
        r = _ref_general(a, n, m, grid)
        if r.shape != b.shape or not np.array_equal(r, b):
            return []

    flat = "_".join(v for row in grid for v in row)
    out = []

    # cheaper point-symmetric 2x2 build (preferred when it reproduces all pairs).
    if n == 2 and m == 2 and all(
            np.array_equal(_ref_pointsym_2x2(a, grid), b) for a, b in allp):
        try:
            model = _build_pointsym_2x2(h, w, grid)
            onnx.checker.check_model(model, full_check=True)
            out.append((f"reflectpaste_2x2sym_{flat}", model))
        except Exception:
            pass

    # general per-block build (always correct; backup / non-symmetric layouts).
    try:
        model = _build_general(h, w, n, m, grid)
        onnx.checker.check_model(model, full_check=True)
        out.append((f"reflectpaste_{n}x{m}_{flat}", model))
    except Exception:
        pass

    return out
