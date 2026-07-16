"""family_golf7_2: cheaper EXACT golf solvers for targets in slice [2::4].

Implemented families
--------------------
* fill_enclosed (tasks 187 "sc_fillenc8_c2_recC", 338 "sc_fillenc8_c3_recC",
  99 "container_fill"-style): the grid holds walls (any single colour) drawn as
  closed loops.  Background cells (colour 0) split into an EXTERIOR region
  (4-connected to the grid border) and ENCLOSED pockets (holes inside the
  walls).  Two output styles are detected:

    - keep_walls (187): walls keep colour, exterior -> Eclr, enclosed -> Nclr.
    - drop_walls (338): everything -> 0 except enclosed pockets -> Nclr.

  The exterior is computed by an iterative geodesic flood from the grid border:
  seeds = background cells adjacent to padding / tensor edge (found with a single
  plus-conv on the "real cell" mask), then N plus-dilations each clipped to the
  background mask.  N is chosen (in numpy) as the convergence depth over the
  given examples plus a small margin, so the static graph reproduces the true
  connected component exactly.  Everything runs on cheap [1,1,30,30] masks.

Each family auto-detects with a numpy reference over train+test+arc-gen and is
proposed only when it reproduces every pair exactly, so wrong guesses cost
nothing (the grader re-validates EXACTness).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT


def _model(nodes, initializers=()):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# --------------------------------------------------------------------------
# numpy reference (geodesic flood from the grid border)
# --------------------------------------------------------------------------

def _exterior(a):
    """Background (==0) cells 4-connected to the grid border."""
    H, W = a.shape
    ext = np.zeros((H, W), bool)
    q = deque()
    for i in range(H):
        for j in range(W):
            if (i == 0 or j == 0 or i == H - 1 or j == W - 1) and a[i, j] == 0:
                ext[i, j] = True
                q.append((i, j))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and a[nr, nc] == 0 and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def _flood_depth(a):
    """Max geodesic distance of any exterior cell from the border seeds."""
    H, W = a.shape
    dist = -np.ones((H, W), int)
    q = deque()
    for i in range(H):
        for j in range(W):
            if (i == 0 or j == 0 or i == H - 1 or j == W - 1) and a[i, j] == 0:
                dist[i, j] = 0
                q.append((i, j))
    md = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and a[nr, nc] == 0 and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                md = max(md, dist[nr, nc])
                q.append((nr, nc))
    return md


def _solve(a, ext_c, enc_c, keep_walls):
    ext = _exterior(a)
    bg = (a == 0)
    enc = bg & ~ext
    out = a.copy()
    if not keep_walls:
        out[a != 0] = 0
    out[ext] = ext_c
    out[enc] = enc_c
    return out


# --------------------------------------------------------------------------
# ONNX builders
# --------------------------------------------------------------------------

_PLUS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float32).reshape(1, 1, 3, 3)


def _flood_nodes(nodes, cst, n_steps):
    """Emit nodes computing exterior mask `F` and enclosed mask `enc`
    (both [1,1,30,30] 0/1) and `REAL` / `ch0`. Returns tensor names."""
    cst("plusk", _PLUS)
    cst("c0", np.array([0], np.int64), INT64)
    cst("c1", np.array([1], np.int64), INT64)
    cst("cax", np.array([1], np.int64), INT64)
    cst("thr45", np.array([4.5], np.float32))

    # REAL = sum over channels (1 at real cells, 0 at padding)
    nodes.append(oh.make_node("ReduceSum", ["input"], ["REAL"], axes=[1], keepdims=1))
    # background channel 0
    nodes.append(oh.make_node("Slice", ["input", "c0", "c1", "cax"], ["ch0"]))
    # border seeds: real bg cells whose plus-sum of REAL < 5 (touch padding/edge)
    nodes.append(oh.make_node("Conv", ["REAL", "plusk"], ["Sreal"],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    nodes.append(oh.make_node("Less", ["Sreal", "thr45"], ["bd"]))
    nodes.append(oh.make_node("Cast", ["bd"], ["bdf"], to=FLOAT))
    nodes.append(oh.make_node("Mul", ["ch0", "bdf"], ["F0"]))

    cur = "F0"
    for k in range(n_steps):
        cv = f"cv{k}"
        nx = f"F{k + 1}"
        nodes.append(oh.make_node("Conv", [cur, "plusk"], [cv],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Min", [cv, "ch0"], [nx]))
        cur = nx
    # cur == exterior
    nodes.append(oh.make_node("Sub", ["ch0", cur], ["enc"]))   # enclosed = bg - exterior
    return cur, "enc", "REAL", "ch0"


def _build_keepwalls(n_steps, ext_c, enc_c):
    """187/251-style: walls keep colour, exterior -> ext_c, enclosed -> enc_c.
    Per output channel c: out_c = (wall channel c) + (F if ext_c==c) + (enc if enc_c==c).
    ext_c / enc_c may be 0 (region keeps background)."""
    inits, nodes = [], []

    def cst(name, arr, dtype=FLOAT):
        arr = np.asarray(arr)
        inits.append(oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist()))
        return name

    F, enc, REAL, ch0 = _flood_nodes(nodes, cst, n_steps)
    nodes.append(oh.make_node("Sub", ["enc", "enc"], ["Z"]))   # zeros

    chans = []
    for c in range(1, 10):
        cst(f"a{c}", np.array([c], np.int64), INT64)
        cst(f"b{c}", np.array([c + 1], np.int64), INT64)
    for c in range(10):
        parts = []
        if c >= 1:                       # wall channel (kept)
            sl = f"sc{c}"
            nodes.append(oh.make_node("Slice", ["input", f"a{c}", f"b{c}", "cax"], [sl]))
            parts.append(sl)
        if c == ext_c:
            parts.append(F)
        if c == enc_c:
            parts.append("enc")
        if not parts:
            chans.append("Z")
        elif len(parts) == 1:
            chans.append(parts[0])
        else:
            acc = parts[0]
            for i in range(1, len(parts)):
                out = f"o{c}_{i}"
                nodes.append(oh.make_node("Add", [acc, parts[i]], [out]))
                acc = out
            chans.append(acc)
    nodes.append(oh.make_node("Concat", chans, ["output"], axis=1))
    return _model(nodes, inits)


_ONES5 = np.ones((1, 1, 5, 5), np.float32)
_N = 30


def _ray_enc_nodes(nodes, cst):
    """Enclosed mask via ray-casting: a background cell is enclosed iff there is
    a wall in all four straight directions (exact for isolated convex boxes).
    Returns (enc, ch0). Uses four [30,30] triangular MatMuls."""
    cst("c0", np.array([0], np.int64), INT64)
    cst("c1", np.array([1], np.int64), INT64)
    cst("cax", np.array([1], np.int64), INT64)
    cst("half", np.array([0.5], np.float32))
    Tup = np.triu(np.ones((_N, _N), np.float32), 1)     # [k,j]=1 if k<j
    Tlo = np.tril(np.ones((_N, _N), np.float32), -1)    # [k,j]=1 if k>j
    cst("Tup", Tup); cst("Tlo", Tlo)

    # wall mask = (sum over all channels) - background channel  (0 at padding)
    nodes.append(oh.make_node("ReduceSum", ["input"], ["REAL"], axes=[1], keepdims=1))
    nodes.append(oh.make_node("Slice", ["input", "c0", "c1", "cax"], ["ch0"]))
    nodes.append(oh.make_node("Sub", ["REAL", "ch0"], ["wall"]))

    nodes.append(oh.make_node("MatMul", ["wall", "Tup"], ["lc"]))   # left:  k<j
    nodes.append(oh.make_node("MatMul", ["wall", "Tlo"], ["rc"]))   # right: k>j
    nodes.append(oh.make_node("MatMul", ["Tlo", "wall"], ["uc"]))   # up:    k<i
    nodes.append(oh.make_node("MatMul", ["Tup", "wall"], ["dc"]))   # down:  k>i
    for nm, src in (("hl", "lc"), ("hr", "rc"), ("hu", "uc"), ("hd", "dc")):
        nodes.append(oh.make_node("Greater", [src, "half"], [nm + "b"]))
        nodes.append(oh.make_node("Cast", [nm + "b"], [nm], to=FLOAT))
    nodes.append(oh.make_node("Mul", ["ch0", "hl"], ["p1"]))
    nodes.append(oh.make_node("Mul", ["p1", "hr"], ["p2"]))
    nodes.append(oh.make_node("Mul", ["p2", "hu"], ["p3"]))
    nodes.append(oh.make_node("Mul", ["p3", "hd"], ["enc"]))
    return "enc", "ch0"


def _build_sizefill(wall_c, ca1, ca4, ca9):
    """302-style: walls (colour wall_c) kept, exterior stays 0, each enclosed
    square hole recoloured by its area: 1->ca1, 4->ca4, 9->ca9 (detected via a
    5x5 box-count that equals the hole area for holes up to 3x3)."""
    inits, nodes = [], []

    def cst(name, arr, dtype=FLOAT):
        arr = np.asarray(arr)
        inits.append(oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist()))
        return name

    enc, ch0 = _ray_enc_nodes(nodes, cst)
    nodes.append(oh.make_node("Sub", [ch0, enc], ["F"]))   # exterior background (kept colour 0)
    F = "F"
    cst("ones5", _ONES5)
    cst("t35", np.array([3.5], np.float32))
    cst("t85", np.array([8.5], np.float32))
    cst("one", np.array([1.0], np.float32))
    cst("wa", np.array([wall_c], np.int64), INT64)
    cst("wb", np.array([wall_c + 1], np.int64), INT64)

    nodes.append(oh.make_node("Conv", ["enc", "ones5"], ["cnt"],
                              kernel_shape=[5, 5], pads=[2, 2, 2, 2]))
    nodes.append(oh.make_node("Greater", ["cnt", "t35"], ["g4"]))
    nodes.append(oh.make_node("Cast", ["g4"], ["f4"], to=FLOAT))
    nodes.append(oh.make_node("Greater", ["cnt", "t85"], ["g9"]))
    nodes.append(oh.make_node("Cast", ["g9"], ["f9"], to=FLOAT))
    nodes.append(oh.make_node("Sub", ["one", "f4"], ["not4"]))
    nodes.append(oh.make_node("Sub", ["one", "f9"], ["not9"]))
    nodes.append(oh.make_node("Mul", ["enc", "not4"], ["m1"]))      # area 1
    nodes.append(oh.make_node("Mul", ["enc", "f4"], ["e4"]))
    nodes.append(oh.make_node("Mul", ["e4", "not9"], ["m4"]))        # area 4
    nodes.append(oh.make_node("Mul", ["enc", "f9"], ["m9"]))         # area 9
    nodes.append(oh.make_node("Sub", ["enc", "enc"], ["Z"]))
    nodes.append(oh.make_node("Slice", ["input", "wa", "wb", "cax"], ["wallc"]))

    # accumulate per output channel
    sel = {ca1: "m1", ca4: "m4", ca9: "m9"}
    chans = []
    for c in range(10):
        parts = []
        if c == 0:
            parts.append(F)
        if c == wall_c:
            parts.append("wallc")
        if c in sel:
            parts.append(sel[c])
        if not parts:
            chans.append("Z")
        elif len(parts) == 1:
            chans.append(parts[0])
        else:
            acc = parts[0]
            for i in range(1, len(parts)):
                out = f"sf{c}_{i}"
                nodes.append(oh.make_node("Add", [acc, parts[i]], [out]))
                acc = out
            chans.append(acc)
    nodes.append(oh.make_node("Concat", chans, ["output"], axis=1))
    return _model(nodes, inits)


def _build_dropwalls(n_steps, enc_c):
    """338-style: everything -> 0 except enclosed pockets -> enc_c."""
    inits, nodes = [], []

    def cst(name, arr, dtype=FLOAT):
        arr = np.asarray(arr)
        inits.append(oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist()))
        return name

    F, enc, REAL, ch0 = _flood_nodes(nodes, cst, n_steps)

    nodes.append(oh.make_node("Sub", ["REAL", "enc"], ["bgrem"]))   # real & not enclosed -> colour 0
    nodes.append(oh.make_node("Sub", ["enc", "enc"], ["Z"]))        # zeros

    chans = []
    for c in range(10):
        if c == 0:
            chans.append("bgrem")
        elif c == enc_c:
            chans.append("enc")
        else:
            chans.append("Z")
    nodes.append(oh.make_node("Concat", chans, ["output"], axis=1))
    return _model(nodes, inits)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def _detect_fill(prs):
    """Return (style, ext_c, enc_c, n_steps) or None."""
    # need same-shape, two-colour (0 + walls) inputs
    for a, b in prs:
        if a.shape != b.shape:
            return None

    # infer mapping from the first pair that has an enclosed region
    cand = []
    # keep_walls style: ext->E, enc->N (walls preserved)
    # try to read E,N from outputs via the flood
    for a, b in prs:
        ext = _exterior(a)
        bg = (a == 0)
        enc = bg & ~ext
        if enc.any():
            e_vals = set(b[ext].tolist()) if ext.any() else set()
            n_vals = set(b[enc].tolist())
            wall = (a != 0)
            w_keep = wall.any() and np.array_equal(b[wall], a[wall])
            cand.append((e_vals, n_vals, w_keep))
    if not cand:
        return None
    # decide style by majority/consistency: try both and validate fully below
    return cand


def _try_fill(prs):
    info = _detect_fill(prs)
    if not info:
        return None
    # enumerate candidate (style, ext_c, enc_c)
    options = set()
    for e_vals, n_vals, w_keep in info:
        for nc in (n_vals or {0}):
            if w_keep:
                for ec in (e_vals or {0}):
                    options.add(("keep", int(ec), int(nc)))
            else:
                options.add(("drop", 0, int(nc)))
    best = None
    for style, ec, nc in options:
        keep = (style == "keep")
        ok = True
        depth = 0
        for a, b in prs:
            pred = _solve(a, ec, nc, keep)
            if pred.shape != b.shape or not np.array_equal(pred, b):
                ok = False
                break
            depth = max(depth, _flood_depth(a))
        if ok:
            best = (style, ec, nc, depth)
            break
    return best


def _comp_areas(enc):
    H, W = enc.shape
    lab = np.zeros((H, W), int)
    cur = 0
    areas = {}
    for i in range(H):
        for j in range(W):
            if enc[i, j] and lab[i, j] == 0:
                cur += 1
                lab[i, j] = cur
                q = deque([(i, j)])
                n = 1
                while q:
                    r, c = q.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and enc[nr, nc] and lab[nr, nc] == 0:
                            lab[nr, nc] = cur
                            q.append((nr, nc))
                            n += 1
                areas[cur] = n
    return lab, areas


def _conv5(m):
    H, W = m.shape
    mp = np.pad(m, 2)
    out = np.zeros((H, W), int)
    for i in range(H):
        for j in range(W):
            out[i, j] = mp[i:i + 5, j:j + 5].sum()
    return out


def _enc_ray(a):
    """Enclosed mask via ray-casting (mirrors the ONNX size-fill graph)."""
    w = (a != 0).astype(int)
    L = np.cumsum(w, axis=1) - w
    R = np.cumsum(w[:, ::-1], axis=1)[:, ::-1] - w
    U = np.cumsum(w, axis=0) - w
    D = np.cumsum(w[::-1], axis=0)[::-1] - w
    return (a == 0) & (L > 0) & (R > 0) & (U > 0) & (D > 0)


def _try_sizefill(prs):
    """302-style: walls of one colour kept, exterior stays 0, each enclosed
    square hole recoloured by its area (1/4/9). Requires the 5x5 box-count to
    equal the area (holes up to 3x3) so the static graph matches exactly."""
    wall_colors = set()
    area2color = {}
    depth = 0
    for a, b in prs:
        if a.shape != b.shape:
            return None
        ic = set(a.ravel().tolist()) - {0}
        if len(ic) != 1:
            return None
        wall_colors |= ic
        enc = _enc_ray(a)
        bg = (a == 0)
        ext = bg & ~enc
        # walls + exterior preserved
        wall = (a != 0)
        if wall.any() and not np.array_equal(b[wall], a[wall]):
            return None
        if ext.any() and not (b[ext] == 0).all():
            return None
        lab, areas = _comp_areas(enc)
        c5 = _conv5(enc.astype(int))
        for k, ar in areas.items():
            cells = (lab == k)
            cols = set(b[cells].tolist())
            if len(cols) != 1:
                return None
            col = int(next(iter(cols)))
            if ar in area2color and area2color[ar] != col:
                return None
            area2color[ar] = col
            if not np.all(c5[cells] == ar):     # 5x5 count must equal area
                return None
        depth = max(depth, _flood_depth(a))
    if len(wall_colors) != 1:
        return None
    if not set(area2color).issubset({1, 4, 9}) or not area2color:
        return None
    wall_c = int(next(iter(wall_colors)))
    ca1 = area2color.get(1, 0)
    ca4 = area2color.get(4, 0)
    ca9 = area2color.get(9, 0)
    return wall_c, ca1, ca4, ca9, depth


# --------------------------------------------------------------------------
# flood_recolor (task 354 "flood_m5"): every S-coloured 4-connected object is
# recoloured to the marker sitting in the top row (row 0) above one of its
# columns.  Implemented as a single colour-value max-flood constrained to the
# S-mask, seeded from the marker columns, then decoded per marker colour.
# --------------------------------------------------------------------------

def _components(mask):
    H, W = mask.shape
    lab = -np.ones((H, W), int)
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] < 0:
                q = deque([(i, j)])
                lab[i, j] = cur
                while q:
                    r, c = q.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and lab[nr, nc] < 0:
                            lab[nr, nc] = cur
                            q.append((nr, nc))
                cur += 1
    return lab, cur


def _flood_recolor_ref(a, S):
    """Recolour each S-object by the single row-0 marker in its columns."""
    lab, n = _components(a == S)
    out = a.copy()
    markers = set()
    for k in range(n):
        cells = np.argwhere(lab == k)
        cols = set(cells[:, 1].tolist())
        found = {int(a[0, c]) for c in cols if a[0, c] != 0}
        if len(found) != 1:
            return None, None
        col = found.pop()
        markers.add(col)
        for (r, c) in cells:
            out[r, c] = col
    return out, markers


def _seed_depth(a, S):
    """4-conn geodesic depth from marker-column S-cells to the whole object."""
    H, W = a.shape
    mask = (a == S)
    colmark = np.zeros((H, W), bool)
    for c in range(W):
        if a[0, c] != 0:
            colmark[:, c] = True
    seed = mask & colmark
    dist = -np.ones((H, W), int)
    q = deque()
    for (r, c) in np.argwhere(seed):
        dist[r, c] = 0
        q.append((r, c))
    md = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                md = max(md, dist[nr, nc])
                q.append((nr, nc))
    if (mask & (dist < 0)).any():
        return None
    return md


def _try_flood_recolor(prs):
    for a, b in prs:
        if a.shape != b.shape:
            return None
    for S in range(1, 10):
        if not any((a == S).any() for a, _ in prs):
            continue
        ok = True
        markers = set()
        depth = 0
        for a, b in prs:
            ref, mk = _flood_recolor_ref(a, S)
            if ref is None or not np.array_equal(ref, b):
                ok = False
                break
            markers |= mk
            d = _seed_depth(a, S)
            if d is None:
                ok = False
                break
            depth = max(depth, d)
        if ok and markers and S not in markers:
            return S, sorted(markers), depth + 1
    return None


def _build_flood_recolor(S, markers, n_steps):
    inits, nodes = [], []

    def cst(name, arr, dtype=FLOAT):
        arr = np.asarray(arr)
        inits.append(oh.make_tensor(name, dtype, list(arr.shape), arr.ravel().tolist()))
        return name

    cst("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    for nm, (a, b) in (("kU", (2, 1)), ("kD", (0, 1)), ("kL", (1, 2)), ("kR", (1, 0))):
        k = np.zeros((1, 1, 3, 3), np.float32)
        k[0, 0, a, b] = 1.0
        cst(nm, k)
    cst("onescol", np.ones((1, 1, 30, 1), np.float32))
    cst("Sa", np.array([S], np.int64), INT64)
    cst("Sb", np.array([S + 1], np.int64), INT64)
    cst("cax", np.array([1], np.int64), INT64)
    cst("r0", np.array([0], np.int64), INT64)
    cst("r1", np.array([1], np.int64), INT64)
    cst("rax", np.array([2], np.int64), INT64)

    nodes.append(oh.make_node("Conv", ["input", "Wcol"], ["colorval"], kernel_shape=[1, 1]))
    nodes.append(oh.make_node("Slice", ["input", "Sa", "Sb", "cax"], ["maskS"]))
    nodes.append(oh.make_node("Slice", ["colorval", "r0", "r1", "rax"], ["row0"]))
    nodes.append(oh.make_node("MatMul", ["onescol", "row0"], ["colbc"]))
    nodes.append(oh.make_node("Mul", ["maskS", "colbc"], ["seed"]))

    cur = "seed"
    for k in range(n_steps):
        for d in "UDLR":
            nodes.append(oh.make_node("Conv", [cur, f"k{d}"], [f"s{d}{k}"],
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Max", [cur, f"sU{k}", f"sD{k}", f"sL{k}", f"sR{k}"],
                                  [f"mx{k}"]))
        nxt = f"cur{k}"
        nodes.append(oh.make_node("Mul", [f"mx{k}", "maskS"], [nxt]))
        cur = nxt
    field = cur

    nodes.append(oh.make_node("Sub", ["maskS", "maskS"], ["Z"]))
    chans = []
    for c in range(10):
        cst(f"a{c}", np.array([c], np.int64), INT64)
        cst(f"b{c}", np.array([c + 1], np.int64), INT64)
        nodes.append(oh.make_node("Slice", ["input", f"a{c}", f"b{c}", "cax"], [f"in{c}"]))
        if c == S:
            chans.append("Z")
        elif c in markers:
            cst(f"lo{c}", np.array([c - 0.5], np.float32))
            cst(f"hi{c}", np.array([c + 0.5], np.float32))
            nodes.append(oh.make_node("Greater", [field, f"lo{c}"], [f"g{c}"]))
            nodes.append(oh.make_node("Less", [field, f"hi{c}"], [f"l{c}"]))
            nodes.append(oh.make_node("And", [f"g{c}", f"l{c}"], [f"and{c}"]))
            nodes.append(oh.make_node("Cast", [f"and{c}"], [f"rc{c}"], to=FLOAT))
            nodes.append(oh.make_node("Add", [f"in{c}", f"rc{c}"], [f"o{c}"]))
            chans.append(f"o{c}")
        else:
            chans.append(f"in{c}")
    nodes.append(oh.make_node("Concat", chans, ["output"], axis=1))
    return _model(nodes, inits)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def candidates(examples):
    prs = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("train", []) + examples.get("test", [])]
    gen = [(np.array(e["input"]), np.array(e["output"]))
           for e in examples.get("arc-gen", [])]
    if not prs:
        return []
    allp = prs + gen if gen else prs
    out = []

    fr = _try_flood_recolor(allp)
    if fr is not None:
        S, markers, n_steps = fr
        m = _build_flood_recolor(S, markers, n_steps)
        out.append((f"floodrecolor_s{S}", m))
        return out

    sf = _try_sizefill(allp)
    if sf is not None:
        wall_c, ca1, ca4, ca9, depth = sf
        m = _build_sizefill(wall_c, ca1, ca4, ca9)
        out.append((f"fillenc_size_w{wall_c}", m))
        return out

    res = _try_fill(allp)
    if res is not None:
        style, ec, nc, depth = res
        n_steps = depth + 1
        if style == "keep":
            m = _build_keepwalls(n_steps, ec, nc)
            out.append((f"fillenc_keep_e{ec}_n{nc}", m))
        else:
            m = _build_dropwalls(n_steps, nc)
            out.append((f"fillenc_drop_n{nc}", m))
    return out
