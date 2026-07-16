"""Tiling / replication family: output is the input replicated in an n x m grid,
all ORIGIN-ANCHORED (assembled content stays top-left, within 30x30).

Supported per-tile transforms: identity, horizontal flip (lr), vertical flip (ud),
180 rotation (r180), transpose (T) and its three flips (Tlr, Tud, Tr180), plus a
blank/zero tile and a constant per-tile recolor (color permutation).

Realizability note: tiling repeats content at period (h, w) -- the *actual* grid
size, which is data-dependent. The one-hot tensor is zero-padded to 30x30, so a
plain onnx Tile over the full 30x30 tensor would repeat at period 30, not period
w/h -> wrong for grids smaller than 30. We therefore only emit a model when the
input size is CONSTANT across all available examples (train+test+arc-gen): we
Slice out the exact h x w content block, build each tile from it (Tile for plain
replication, or Transpose/reverse-Slice + Concat for flipped/transposed layouts),
then Pad the assembled (n*h) x (m*w) block back to 30x30 -- everything anchored
top-left. Non-constant-size tiling is not expressible with a static graph.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model
from ng_utils_shim import DATA_TYPE, CHANNELS, HEIGHT, WIDTH

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)  # sentinel "to the beginning" for reverse Slice

# variant name -> numpy transform on a raw (h, w) grid
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
    """Return (n, m, grid) where grid[i][j] is a variant name, or None."""
    nms = set()
    for a, b in prs:
        h, w = a.shape
        H, W = b.shape
        if h == 0 or w == 0 or H % h or W % w:
            return None
        n, m = H // h, W // w
        if n * m <= 1:
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
                # blank tile?
                if all(np.array_equal(
                        b[i * a.shape[0]:(i + 1) * a.shape[0],
                          j * a.shape[1]:(j + 1) * a.shape[1]],
                        np.zeros_like(a)) for a, b in prs):
                    found = "zero"
            if found is None:
                return None
            grid[i][j] = found
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
    # content block: input[:, :, 0:h, 0:w]
    cs = g.init(INT64, [2], [0, 0])
    ce = g.init(INT64, [2], [h, w])
    ca = g.init(INT64, [2], [2, 3])
    c = g.node("Slice", ["input", cs, ce, ca])

    nh, mw = n * h, m * w
    flat = [v for row in grid for v in row]

    if set(flat) == {"id"}:
        # plain replication via Tile (cheapest)
        rep = g.init(INT64, [4], [1, 1, n, m])
        asm = g.node("Tile", [c, rep])
    else:
        cache = {}

        def variant(vn):
            if vn in cache:
                return cache[vn]
            if vn == "id":
                r = c
            elif vn == "lr":
                r = _reverse(g, c, [3], {2: h, 3: w})
            elif vn == "ud":
                r = _reverse(g, c, [2], {2: h, 3: w})
            elif vn == "r180":
                r = _reverse(g, c, [2, 3], {2: h, 3: w})
            elif vn == "zero":
                r = g.node("Sub", [c, c])  # h x w zeros, no params
            elif vn.startswith("T"):
                t = cache.get("T")
                if t is None:
                    t = g.node("Transpose", [c], perm=[0, 1, 3, 2])
                    cache["T"] = t
                # transposed block dims are (w, h)
                dims = {2: w, 3: h}
                if vn == "T":
                    r = t
                elif vn == "Tlr":
                    r = _reverse(g, t, [3], dims)
                elif vn == "Tud":
                    r = _reverse(g, t, [2], dims)
                else:  # Tr180
                    r = _reverse(g, t, [2, 3], dims)
            cache[vn] = r
            return r

        rows = []
        for i in range(n):
            tiles = [variant(grid[i][j]) for j in range(m)]
            row = tiles[0] if m == 1 else g.node("Concat", tiles, axis=3)
            rows.append(row)
        asm = rows[0] if n == 1 else g.node("Concat", rows, axis=2)

    # pad assembled (nh x mw) up to 30 x 30, anchored top-left
    pad = g.node("Pad", [asm], "output", mode="constant", value=0.0,
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
    # require constant input size across everything we can see (else a static
    # graph cannot match arc-gen and the model would be rejected anyway).
    allp = _pairs(ex, ("train", "test", "arc-gen"))
    shapes = set(a.shape for a, _ in allp)
    if len(shapes) != 1:
        return []
    h, w = next(iter(shapes))
    if n * h > HEIGHT or m * w > WIDTH or h * w == 0:
        return []
    flat = "_".join(v for row in grid for v in row)
    try:
        model = _build(h, w, n, m, grid)
    except Exception:
        return []
    return [(f"tile_{n}x{m}_{flat}", model)]
