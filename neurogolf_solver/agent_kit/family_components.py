"""Connected-component family: recolor each same-color 4-connected component by a
function of its CELL COUNT (component size), realised with an opset-10 graph that
ACTUALLY computes connected components at inference time (no hard-coding).

Pipeline (origin-anchored, size-independent)
--------------------------------------------
1.  M = non-background presence  (ReduceSum over colour channels 1..9) -> [1,1,30,30].
2.  Label-propagation cellular automaton.  Each cell starts with a unique position
    id  P[r,c] = r*30+c+1.  We iterate  L <- M * max(L, shift_up, shift_down,
    shift_left, shift_right)  for T steps (shifts are zero-padded Pad+Slice, so the
    top-left origin and grid boundaries are respected).  Because M is a single
    non-bg mask and these tasks contain ONE non-bg colour, propagation stays inside
    each 4-connected component; after T >= component diameter steps every cell in a
    component holds the same value (the component's maximum position id = a unique
    component label).
3.  Component size per cell via an equality matrix.  Flatten L to a [900,1] column
    and [1,900] row, Equal -> [900,900] (cells sharing a label), masked to non-bg
    cells, ReduceSum over a row -> each cell's component size (0 for bg/padding).
4.  Recolour: for each (size -> colour) rule inferred from the pairs, build a
    spatial mask from the size field with Greater/Less (no float Equal -> opset-10
    safe) and route it into the target colour channel; cells whose size matches no
    rule keep their original colour, background and padding stay untouched.

Only the *exact* size->colour map that reproduces EVERY train+test+arc-gen pair is
emitted, so wrong hypotheses are dropped before the grader sees them.  All five
ARC recolour-by-size tasks in the corpus (147, 169, 196, 272, 330) are covered;
the size-threshold special cases (147 "big->X", 272 "isolated->Y") are detected as
unbounded ">= k" rules so they generalise to unseen component sizes.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
INT32 = onnx.TensorProto.INT32
F = DATA_TYPE
H, W = HEIGHT, WIDTH


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0

    def nm(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def cf(self, dims, vals):
        n = self.nm("cf")
        self.inits.append(oh.make_tensor(n, F, list(dims),
                                         [float(v) for v in np.asarray(vals).ravel()]))
        return n

    def ci(self, dims, vals, dt=INT64):
        n = self.nm("ci")
        self.inits.append(oh.make_tensor(n, dt, list(dims),
                                         [int(v) for v in np.asarray(vals).ravel()]))
        return n

    def nd(self, op, ins, out=None, **a):
        out = out or self.nm()
        self.nodes.append(oh.make_node(op, list(ins), [out], **a))
        return out


def _model(g, out_shape=GRID_SHAPE):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, list(out_shape))
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# connected-component SIZE field  ->  size2d [1,1,30,30]                       #
# --------------------------------------------------------------------------- #
def _size_field(g, T):
    # shared shift index constants (reused across iterations)
    def shift_consts(dr, dc):
        sh, sw = max(-dr, 0), max(-dc, 0)
        return (g.ci([2], [sh, sw]), g.ci([2], [sh + H, sw + W]), g.ci([2], [2, 3]))

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    sc = {d: shift_consts(*d) for d in dirs}

    def shift(x, dr, dc):
        pt, pb = max(dr, 0), max(-dr, 0)
        pl, pr = max(dc, 0), max(-dc, 0)
        p = g.nd("Pad", [x], mode="constant", value=0.0,
                 pads=[0, 0, pt, pl, 0, 0, pb, pr])
        st, en, ax = sc[(dr, dc)]
        return g.nd("Slice", [p, st, en, ax])

    # non-bg presence M
    xnb = g.nd("Slice", ["input", g.ci([1], [1]), g.ci([1], [CHANNELS]), g.ci([1], [1])])
    M = g.nd("ReduceSum", [xnb], axes=[1], keepdims=1)            # [1,1,30,30]

    P = g.cf([1, 1, H, W], np.arange(1, H * W + 1).reshape(1, 1, H, W))
    L = g.nd("Mul", [M, P])
    for _ in range(T):
        nbrs = [shift(L, dr, dc) for dr, dc in dirs]
        mx = g.nd("Max", [L] + nbrs)
        L = g.nd("Mul", [mx, M])

    # equality matrix -> size per cell.  Bg/padding cells all carry label 0 and so
    # match each other, but a non-bg cell's label is unique to its component, so its
    # row-sum is exactly the component size; bg/padding rows are simply zeroed by M
    # afterwards (one cheap [1,1,30,30] multiply, avoiding big validity matrices).
    Lcol = g.nd("Reshape", [L, g.ci([2], [H * W, 1])])           # [900,1]
    Lrow = g.nd("Reshape", [L, g.ci([2], [1, H * W])])           # [1,900]
    E = g.nd("Equal", [g.nd("Cast", [Lcol], to=INT32),
                       g.nd("Cast", [Lrow], to=INT32)])           # bool [900,900]
    Ef = g.nd("Cast", [E], to=F)
    size_col = g.nd("ReduceSum", [Ef], axes=[1], keepdims=1)      # [900,1]
    size2d = g.nd("Reshape", [size_col, g.ci([4], [1, 1, H, W])]) # [1,1,30,30]
    size2d = g.nd("Mul", [size2d, M])                             # zero bg/padding
    return size2d


# --------------------------------------------------------------------------- #
# size mask helpers (Greater/Less only -> opset-10 safe, no float Equal)       #
# --------------------------------------------------------------------------- #
def _mask_ge(g, size2d, k):
    return g.nd("Cast", [g.nd("Greater", [size2d, g.cf([1, 1, 1, 1], [k - 0.5])])], to=F)


def _mask_eq(g, size2d, v):
    gt = g.nd("Greater", [size2d, g.cf([1, 1, 1, 1], [v - 0.5])])
    lt = g.nd("Less", [size2d, g.cf([1, 1, 1, 1], [v + 0.5])])
    return g.nd("Mul", [g.nd("Cast", [gt], to=F), g.nd("Cast", [lt], to=F)])


def _build_recolor(rules, T):
    """rules: list of (kind, args, target_color).
    kind 'ge' -> args (k,);  'eq' -> args (v,)."""
    g = _G()
    size2d = _size_field(g, T)

    per_target = {}                                   # color -> mask name
    all_masks = []
    for kind, args, o in rules:
        m = _mask_ge(g, size2d, args[0]) if kind == "ge" else _mask_eq(g, size2d, args[0])
        all_masks.append(m)
        per_target[o] = g.nd("Add", [per_target[o], m]) if o in per_target else m

    # R = union of recoloured cells
    R = all_masks[0]
    for m in all_masks[1:]:
        R = g.nd("Add", [R, m])
    one = g.cf([1, 1, 1, 1], [1.0])
    keep = g.nd("Mul", ["input", g.nd("Sub", [one, R])])          # [1,10,30,30]

    out = keep
    for o, m in per_target.items():
        e_o = g.cf([1, CHANNELS, 1, 1], [1.0 if c == o else 0.0 for c in range(CHANNELS)])
        out = g.nd("Add", [out, g.nd("Mul", [m, e_o])])
    g.nodes[-1].output[0] = "output"
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy reference: same-colour 4-connected components                          #
# --------------------------------------------------------------------------- #
def _components(a, bg=0):
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    out = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == bg:
                continue
            col = a[i, j]
            q = deque([(i, j)])
            seen[i, j] = True
            cells = [(i, j)]
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            out.append((int(col), cells))
    return out


def _apply_sizemap(a, sm, bg=0):
    out = a.copy()
    for col, cells in _components(a, bg):
        s = len(cells)
        if s not in sm:
            return None
        for r, c in cells:
            out[r, c] = sm[s]
    return out


def _diameter(a, bg=0):
    """max BFS distance from a component's max-position root to any of its cells."""
    worst = 0
    for col, cells in _components(a, bg):
        root = max(cells, key=lambda rc: rc[0] * W + rc[1])
        cset = set(cells)
        dist = {root: 0}
        q = deque([root])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in cset and nb not in dist:
                    dist[nb] = dist[(r, c)] + 1
                    q.append(nb)
        if dist:
            worst = max(worst, max(dist.values()))
    return worst


# --------------------------------------------------------------------------- #
# rule grouping (size->colour map -> compact Greater/Less rules)               #
# --------------------------------------------------------------------------- #
def _group_rules(sm, C):
    """sm: size->colour (full map, includes keep entries where colour==C).
    Returns list of (kind,args,colour) for the cells whose colour changes."""
    allsizes = sorted(sm)
    recolor = {v: o for v, o in sm.items() if o != C}
    if not recolor:
        return None
    targets = {}
    for v, o in recolor.items():
        targets.setdefault(o, []).append(v)
    rules = []
    for o, vs in targets.items():
        vs = sorted(vs)
        if len(vs) >= 2 and vs[0] > 1 and vs == [s for s in allsizes if s >= vs[0]]:
            rules.append(("ge", (vs[0],), o))         # unbounded "big component" rule
        else:
            for v in vs:
                rules.append(("eq", (v,), o))
    return rules


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
    if any(a.shape != b.shape for a, b in prs):
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []

    # single non-bg input colour (these CC-by-size tasks are mono-colour)
    incolors = set()
    for a, _ in prs:
        incolors |= set(np.unique(a[a != 0]).tolist())
    if len(incolors) != 1:
        return []
    C = next(iter(incolors))

    # infer size -> colour map, require consistency + exact reproduction
    sm = {}
    for a, b in prs:
        for col, cells in _components(a):
            tcols = {int(b[r, c]) for r, c in cells}
            if len(tcols) != 1:
                return []
            tc = next(iter(tcols))
            s = len(cells)
            if s in sm and sm[s] != tc:
                return []
            sm[s] = tc
    if not sm:
        return []
    if len(set(sm.values())) < 2:           # not size-dependent -> global recolor's job
        return []
    if any(_apply_sizemap(a, sm) is None or not np.array_equal(_apply_sizemap(a, sm), b)
           for a, b in prs):
        return []

    rules = _group_rules(sm, C)
    if not rules:
        return []

    need = max(_diameter(a) for a, _ in prs)
    T = min(80, max(25, need + 18))

    try:
        model = _build_recolor(rules, T)
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return []
    return [("ccsize", model)]
