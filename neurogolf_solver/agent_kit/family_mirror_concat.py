"""Mirror-concat family: output = the input assembled with REFLECTED copies of
itself in a 1x2 / 2x1 / 2x2 block layout, with the FIRST (top-left) block being
the original input -- so the construction is origin-anchored.

Examples (h x w input -> output):
    1x2  [ id | lr ]                       width doubles, left half == input
    2x1  [ id ; ud ]                       height doubles, top half == input
    2x2  [ id  | lr ;  ud | r180 ]         classic four-fold "kaleidoscope"

Why this is origin-safe (see CONTEXT padding gotcha): the one-hot tensor is
zero-padded to 30x30 with content at the TOP-LEFT, and grid sizes vary, so a
full-tensor flip would scatter content to the far edge. Instead we Slice out the
exact h x w content block, build each reflected sub-block FROM THAT BLOCK via a
reverse-Slice (steps=-1), Concat them in the n x m layout, then Pad the assembled
(n*h) x (m*w) region back to 30x30 -- everything stays anchored at (0,0).

Realizability: tiling/mirroring repeats content at the (data-dependent) grid
period, which a static 30x30 graph cannot track. So a model is emitted only when
the input size is CONSTANT across every split we can see (train+test+arc-gen);
otherwise the graph could not match arc-gen and the harness would reject it. We
also require at least one reflected (non-id) block so this stays "mirroring"
rather than plain replication (handled by the tiling family).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)  # sentinel "to the beginning" for a reverse (steps=-1) Slice

# variant name -> numpy reflection of a raw (h, w) grid. The plain flips (lr/ud/
# r180) preserve block dims, so they assemble for any rectangle. The transpose-
# based reflections (T = main-diagonal mirror, and its flips) swap dims to (w,h)
# and therefore only match a block when the input is square -- the shape check in
# _detect rejects them otherwise. Together id+lr+ud+r180+T-family span the eight
# dihedral reflections that produce a "kaleidoscope" tiling.
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


def _pairs(ex, splits=("train", "test")):
    out = []
    for s in splits:
        for e in ex.get(s, []):
            out.append((np.array(e["input"]), np.array(e["output"])))
    return out


def _detect(prs):
    """Return (n, m, grid) for a mirror layout, or None.

    grid[i][j] is a variant name. At least one block must be a reflection (else
    it is plain replication -> tiling family's job). The original need not be the
    first block: the build slices the content block and pads the bottom-right, so
    the assembled (n*h) x (m*w) region is always anchored at the origin (0,0)."""
    nms = set()
    for a, b in prs:
        if a.ndim != 2 or b.ndim != 2:
            return None
        h, w = a.shape
        H, W = b.shape
        if h == 0 or w == 0 or H % h or W % w:
            return None
        n, m = H // h, W // w
        if n not in (1, 2) or m not in (1, 2) or (n, m) == (1, 1):
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

    if all(v == "id" for row in grid for v in row):
        return None  # plain replication -> tiling family's job
    return n, m, grid


class _G:
    """Tiny node/initializer accumulator with auto-named tensors."""

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
    starts = g.init(INT64, [len(axes)], [dims[a] - 1 for a in axes])
    ends = g.init(INT64, [len(axes)], [_NEG] * len(axes))
    ax = g.init(INT64, [len(axes)], list(axes))
    steps = g.init(INT64, [len(axes)], [-1] * len(axes))
    return g.node("Slice", [src, starts, ends, ax, steps])


def _build(h, w, n, m, grid):
    g = _G()
    # content block: input[:, :, 0:h, 0:w] -- the origin-anchored grid content.
    cs = g.init(INT64, [2], [0, 0])
    ce = g.init(INT64, [2], [h, w])
    ca = g.init(INT64, [2], [2, 3])
    c = g.node("Slice", ["input", cs, ce, ca])

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
            dims = {2: w, 3: h}  # transposed block is (w, h)
            if vn == "T":
                r = t
            elif vn == "Tlr":
                r = _reverse(g, t, [3], dims)
            elif vn == "Tud":
                r = _reverse(g, t, [2], dims)
            else:  # Tr180
                r = _reverse(g, t, [2, 3], dims)
        else:
            raise ValueError(vn)
        cache[vn] = r
        return r

    rows = []
    for i in range(n):
        tiles = [variant(grid[i][j]) for j in range(m)]
        row = tiles[0] if m == 1 else g.node("Concat", tiles, axis=3)
        rows.append(row)
    asm = rows[0] if n == 1 else g.node("Concat", rows, axis=2)

    nh, mw = n * h, m * w
    g.node("Pad", [asm], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - nh, WIDTH - mw])
    return _model(g.nodes, g.inits)


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    det = _detect(prs)
    if det is None:
        return []
    n, m, grid = det

    # static graph requires a single constant input size across everything.
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    shapes = set(a.shape for a, _ in allp)
    if len(shapes) != 1:
        return []
    h, w = next(iter(shapes))
    if h * w == 0 or n * h > HEIGHT or m * w > WIDTH:
        return []

    flat = "_".join(v for row in grid for v in row)
    try:
        model = _build(h, w, n, m, grid)
    except Exception:
        return []
    return [(f"mirror_{n}x{m}_{flat}", model)]
