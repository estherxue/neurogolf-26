"""RECOLOR 4-CONNECTED COMPONENTS BY A PER-COMPONENT PROPERTY (opset 10).

This family extends family_components: it labels same-colour 4-connected
components with an UNROLLED max-propagation cellular automaton (no hard-coding)
and then recolours every cell by a *property of its whole component*:

  * SIZE      -- the component cell-count (small -> A, large -> B, an inferred
                 size->colour map; the "big component" / "isolated pixel" cases
                 are detected as unbounded ">= k" rules so they generalise).
  * RANK      -- the component's cell-count RANK (0 = largest); the i-th largest
                 component becomes an inferred colour.  Computed structurally as
                 "number of distinct components strictly larger than mine".
  * BORDER    -- whether the component touches the (real, variable-size) grid
                 border, found with a border-seeded reach CA; touch -> A,
                 interior -> B.

Unlike family_components (which hard-codes background = colour 0 and a single
non-bg presence mask), this module DETECTS the background colour, so it also
covers mono-colour tasks whose canvas is some other colour (e.g. fg = colour 0).

Pipeline (origin-anchored, size-independent)
--------------------------------------------
1.  M = presence of the single foreground colour C  (one channel Slice) ->
    [1,1,30,30].  Padding is all-zero so M is 0 there; the real background
    colour also has M = 0, so labels never leak across the canvas.
2.  Label CA: each cell starts with a unique position id P = r*30+c+1; iterate
    L <- M * max(L, up, down, left, right) for T >= component-diameter steps
    (shifts are zero-padded Pad+Slice -> origin & grid boundary respected).
    Every cell of a 4-connected component ends holding that component's maximum
    position id = a unique component label.
3.  SIZE field: flatten L, Equal -> [900,900] same-label matrix, ReduceSum a row
    -> each cell's component size.  RANK field: the unique "root" cell of every
    component is where L == P; count root cells whose size exceeds mine.
4.  Recolour: per inferred (property-value -> colour) rule build a spatial mask
    with Greater/Less (no float Equal) gated by M and route it into the target
    colour channel; unmatched cells (incl. background, padding) keep their input.

Only the EXACT property->colour map that reproduces EVERY train+test+arc-gen pair
is emitted, so wrong hypotheses are dropped before the grader's held-out gate.
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
HW = H * W


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


def _model(g):
    x = oh.make_tensor_value_info("input", F, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", F, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------- #
# label-propagation CA  ->  L (component labels) and M (fg presence)           #
# --------------------------------------------------------------------------- #
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _shift_consts(g):
    sc = {}
    for dr, dc in _DIRS:
        sh, sw = max(-dr, 0), max(-dc, 0)
        sc[(dr, dc)] = (g.ci([2], [sh, sw]), g.ci([2], [sh + H, sw + W]), g.ci([2], [2, 3]))
    return sc


def _shift(g, sc, x, dr, dc):
    pt, pb = max(dr, 0), max(-dr, 0)
    pl, pr = max(dc, 0), max(-dc, 0)
    p = g.nd("Pad", [x], mode="constant", value=0.0,
             pads=[0, 0, pt, pl, 0, 0, pb, pr])
    st, en, ax = sc[(dr, dc)]
    return g.nd("Slice", [p, st, en, ax])


def _label_M(g, T, Cfg):
    """Return (L, M): L = per-cell component label, M = fg presence mask."""
    sc = _shift_consts(g)
    M = g.nd("Slice", ["input", g.ci([1], [Cfg]), g.ci([1], [Cfg + 1]), g.ci([1], [1])])
    P = g.cf([1, 1, H, W], np.arange(1, HW + 1).reshape(1, 1, H, W))
    L = g.nd("Mul", [M, P])
    for _ in range(T):
        nbrs = [_shift(g, sc, L, dr, dc) for dr, dc in _DIRS]
        mx = g.nd("Max", [L] + nbrs)
        L = g.nd("Mul", [mx, M])
    return L, M


def _size_field(g, L):
    """Per-cell component size -> [1,1,30,30] (bg/padding hold the label-0 size)."""
    Lcol = g.nd("Reshape", [L, g.ci([2], [HW, 1])])                 # [900,1]
    Lrow = g.nd("Reshape", [L, g.ci([2], [1, HW])])                 # [1,900]
    E = g.nd("Equal", [g.nd("Cast", [Lcol], to=INT32),
                       g.nd("Cast", [Lrow], to=INT32)])             # bool [900,900]
    Ef = g.nd("Cast", [E], to=F)
    size_col = g.nd("ReduceSum", [Ef], axes=[1], keepdims=1)        # [900,1]
    size2d = g.nd("Reshape", [size_col, g.ci([4], [1, 1, H, W])])   # [1,1,30,30]
    return size2d


def _rank_field(g, L):
    """Per-cell component rank (0 = largest) -> [1,1,30,30].

    rank = number of distinct components with a STRICTLY larger cell-count.
    The unique root of every component is the cell whose label equals its own
    position id (the component max), so counting roots larger than me counts
    distinct larger components."""
    Lcol = g.nd("Reshape", [L, g.ci([2], [HW, 1])])                 # [900,1]
    Lrow = g.nd("Reshape", [L, g.ci([2], [1, HW])])                 # [1,900]
    Li_col = g.nd("Cast", [Lcol], to=INT32)
    Li_row = g.nd("Cast", [Lrow], to=INT32)
    E = g.nd("Equal", [Li_col, Li_row])                            # bool [900,900]
    Ef = g.nd("Cast", [E], to=F)
    size_col = g.nd("ReduceSum", [Ef], axes=[1], keepdims=1)        # [900,1] size[i]
    size_row = g.nd("ReduceSum", [Ef], axes=[0], keepdims=1)        # [1,900] size[j]

    # root mask: L == position id
    P = g.cf([1, 1, H, W], np.arange(1, HW + 1).reshape(1, 1, H, W))
    isroot = g.nd("Cast", [g.nd("Equal", [g.nd("Cast", [L], to=INT32),
                                          g.nd("Cast", [P], to=INT32)])], to=F)
    isroot_row = g.nd("Reshape", [isroot, g.ci([2], [1, HW])])     # [1,900]

    bigger = g.nd("Cast", [g.nd("Greater", [size_row, size_col])], to=F)  # [900,900]
    weighted = g.nd("Mul", [bigger, isroot_row])                   # only larger roots
    rank_col = g.nd("ReduceSum", [weighted], axes=[1], keepdims=1) # [900,1]
    rank2d = g.nd("Reshape", [rank_col, g.ci([4], [1, 1, H, W])])  # [1,1,30,30]
    return rank2d


# --------------------------------------------------------------------------- #
# spatial masks (Greater/Less only -> opset-10 safe)                          #
# --------------------------------------------------------------------------- #
def _mask_ge(g, field, k):
    return g.nd("Cast", [g.nd("Greater", [field, g.cf([1, 1, 1, 1], [k - 0.5])])], to=F)


def _mask_eq(g, field, v):
    gt = g.nd("Greater", [field, g.cf([1, 1, 1, 1], [v - 0.5])])
    lt = g.nd("Less", [field, g.cf([1, 1, 1, 1], [v + 0.5])])
    return g.nd("Mul", [g.nd("Cast", [gt], to=F), g.nd("Cast", [lt], to=F)])


def _compose(g, field, M, rules):
    """rules: list of (kind, args, target_color); kind 'ge'->(k,), 'eq'->(v,).
    Every mask is gated by M so only foreground cells are recoloured."""
    per_target = {}
    all_masks = []
    for kind, args, o in rules:
        m = _mask_ge(g, field, args[0]) if kind == "ge" else _mask_eq(g, field, args[0])
        m = g.nd("Mul", [m, M])                                     # fg only
        all_masks.append(m)
        per_target[o] = g.nd("Add", [per_target[o], m]) if o in per_target else m

    R = all_masks[0]
    for m in all_masks[1:]:
        R = g.nd("Add", [R, m])
    one = g.cf([1, 1, 1, 1], [1.0])
    keep = g.nd("Mul", ["input", g.nd("Sub", [one, R])])           # bg + unchanged fg
    out = keep
    for o, m in per_target.items():
        e_o = g.cf([1, CHANNELS, 1, 1], [1.0 if c == o else 0.0 for c in range(CHANNELS)])
        out = g.nd("Add", [out, g.nd("Mul", [m, e_o])])
    g.nodes[-1].output[0] = "output"
    return _model(g)


def _build_size(rules, Cfg, T):
    g = _G()
    L, M = _label_M(g, T, Cfg)
    return _compose(g, _size_field(g, L), M, rules)


def _build_rank(rules, Cfg, T):
    g = _G()
    L, M = _label_M(g, T, Cfg)
    return _compose(g, _rank_field(g, L), M, rules)


# --------------------------------------------------------------------------- #
# BORDER reach: 1 on every cell of a component touching the real grid border   #
# --------------------------------------------------------------------------- #
def _build_border(touch_color, interior_color, Cfg, T):
    g = _G()
    sc = _shift_consts(g)
    M = g.nd("Slice", ["input", g.ci([1], [Cfg]), g.ci([1], [Cfg + 1]), g.ci([1], [1])])
    R = g.nd("ReduceSum", ["input"], axes=[1], keepdims=1)          # real-cell mask
    # neighbour-real count via cross Conv on R (4 orthogonal neighbours)
    ker = np.zeros((1, 1, 3, 3), np.float32)
    ker[0, 0, 0, 1] = ker[0, 0, 2, 1] = ker[0, 0, 1, 0] = ker[0, 0, 1, 2] = 1.0
    wt = g.cf([1, 1, 3, 3], ker)
    nb = g.nd("Conv", [R, wt], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    border_real = g.nd("Mul", [R, g.nd("Cast", [g.nd("Less", [nb, g.cf([1, 1, 1, 1], [3.5])])], to=F)])
    seed = g.nd("Mul", [M, border_real])                           # fg border cells
    reach = seed
    for _ in range(T):
        nbrs = [_shift(g, sc, reach, dr, dc) for dr, dc in _DIRS]
        reach = g.nd("Mul", [g.nd("Max", [reach] + nbrs), M])
    interior = g.nd("Mul", [M, g.nd("Sub", [g.cf([1, 1, 1, 1], [1.0]), reach])])

    masks = []
    per_target = {}
    for mask, col in ((reach, touch_color), (interior, interior_color)):
        masks.append(mask)
        per_target[col] = g.nd("Add", [per_target[col], mask]) if col in per_target else mask
    R2 = masks[0]
    for m in masks[1:]:
        R2 = g.nd("Add", [R2, m])
    one = g.cf([1, 1, 1, 1], [1.0])
    keep = g.nd("Mul", ["input", g.nd("Sub", [one, R2])])
    out = keep
    for col, m in per_target.items():
        e_o = g.cf([1, CHANNELS, 1, 1], [1.0 if c == col else 0.0 for c in range(CHANNELS)])
        out = g.nd("Add", [out, g.nd("Mul", [m, e_o])])
    g.nodes[-1].output[0] = "output"
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references                                                            #
# --------------------------------------------------------------------------- #
def _components(a, bg):
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
                for dr, dc in _DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            out.append((int(col), cells))
    return out


def _diameter(a, bg):
    worst = 0
    for col, cells in _components(a, bg):
        root = max(cells, key=lambda rc: rc[0] * W + rc[1])
        cset = set(cells)
        dist = {root: 0}
        q = deque([root])
        while q:
            r, c = q.popleft()
            for dr, dc in _DIRS:
                nb = (r + dr, c + dc)
                if nb in cset and nb not in dist:
                    dist[nb] = dist[(r, c)] + 1
                    q.append(nb)
        if dist:
            worst = max(worst, max(dist.values()))
    return worst


def _key_size(s, sizes, cells, shape):
    return s


def _key_rank(s, sizes, cells, shape):
    return sum(1 for s2 in sizes if s2 > s)


def _key_border(s, sizes, cells, shape):
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _group_rules(mp, keep):
    """mp: value->colour (full map incl. keep where colour==keep).
    Returns ge/eq rules for the cells whose colour CHANGES, or None."""
    allvals = sorted(mp)
    recolor = {v: o for v, o in mp.items() if o != keep}
    if not recolor:
        return None
    targets = {}
    for v, o in recolor.items():
        targets.setdefault(o, []).append(v)
    lo = min(allvals)
    rules = []
    for o, vs in targets.items():
        vs = sorted(vs)
        if len(vs) >= 2 and vs[0] > lo and vs == [s for s in allvals if s >= vs[0]]:
            rules.append(("ge", (vs[0],), o))
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


def _bg_candidates(prs):
    """Cheap (BFS-free) pre-filter: a colour is a plausible background iff every
    pair preserves the non-bg footprint and leaves exactly one foreground colour.
    Returns [(bg, fg), ...] ordered by total background area (most likely first)."""
    cand = []
    for bg in range(CHANNELS):
        if not any((a == bg).any() for a, _ in prs):
            continue
        ok = True
        fg = set()
        area = 0
        for a, b in prs:
            if not np.array_equal(a != bg, b != bg):
                ok = False
                break
            fg |= set(np.unique(a[a != bg]).tolist())
            area += int((a == bg).sum())
        if ok and len(fg) == 1:
            cand.append((area, bg, next(iter(fg))))
    cand.sort(reverse=True)
    return [(bg, fg) for _, bg, fg in cand]


def _components_const(b, cs):
    """True iff every component is a single colour in output b."""
    for col, cells in cs:
        r0, c0 = cells[0]
        tc = int(b[r0, c0])
        for r, c in cells:
            if int(b[r, c]) != tc:
                return False
    return True


def _value_map(prs, comps_list, keyfn):
    """value -> colour map (None on conflict).  Components are assumed
    output-constant, so a conflict-free map exactly reproduces every pair."""
    mp = {}
    for (a, b), cs in zip(prs, comps_list):
        sizes = [len(c) for _, c in cs]
        for col, cells in cs:
            key = keyfn(len(cells), sizes, cells, a.shape)
            r0, c0 = cells[0]
            tc = int(b[r0, c0])
            if key in mp and mp[key] != tc:
                return None
            mp[key] = tc
    return mp


def _simulate(prs, comps_list, keyfn, rules, keep):
    """Apply the grouped ge/eq rules exactly as the ONNX will; verify == output."""
    for (a, b), cs in zip(prs, comps_list):
        out = a.copy()
        sizes = [len(c) for _, c in cs]
        for col, cells in cs:
            key = keyfn(len(cells), sizes, cells, a.shape)
            color = keep
            for kind, args, o in rules:
                if (kind == "ge" and key >= args[0]) or (kind == "eq" and key == args[0]):
                    color = o
                    break
            for r, c in cells:
                out[r, c] = color
        if not np.array_equal(out, b):
            return False
    return True


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):
        return []
    if all(np.array_equal(a, b) for a, b in prs):
        return []

    out = []
    for bg, Cfg in _bg_candidates(prs):
        comps_list = [_components(a, bg) for a, _ in prs]
        if not all(_components_const(b, cs) for (_, b), cs in zip(prs, comps_list)):
            continue
        T = min(80, max(25, max(_diameter(a, bg) for a, _ in prs) + 12))

        # ---- BORDER (touch vs interior) takes precedence ----------------- #
        # When the recolouring is perfectly determined by whether a component
        # touches the real grid border (e.g. "fill the ENCLOSED regions"), that
        # is the structural rule and it generalises to component sizes/shapes
        # never seen.  A border map only stays conflict-free when the task IS
        # border-driven, so preferring it can never displace a genuine size/rank
        # rule (those make the border map inconsistent -> skipped).
        bm = _value_map(prs, comps_list, _key_border)
        if bm is not None and len(bm) == 2 and len(set(bm.values())) >= 2:
            try:
                m = _build_border(bm[True], bm[False], Cfg, T)
                onnx.checker.check_model(m, full_check=True)
                out.append((f"border_bg{bg}_fg{Cfg}", m))
            except Exception:
                pass
        if out:
            break

        # ---- SIZE / RANK maps (value -> colour, grouped into ge/eq rules) -- #
        for tag, keyfn, builder in (("size", _key_size, _build_size),
                                    ("rank", _key_rank, _build_rank)):
            mp = _value_map(prs, comps_list, keyfn)
            if mp is None or len(set(mp.values())) < 2:
                continue
            rules = _group_rules(mp, Cfg)
            if not rules or not _simulate(prs, comps_list, keyfn, rules, Cfg):
                continue
            try:
                m = builder(rules, Cfg, T)
                onnx.checker.check_model(m, full_check=True)
                out.append((f"{tag}_bg{bg}_fg{Cfg}", m))
            except Exception:
                pass

        if out:
            break

    return out
