"""Golf slice 4_0: cheaper EXACT solvers for connected-component-size tasks.

Replaces expensive [900,900] reachability matrices with:
  * label propagation (K iters of 4-neighbour Min on a [1,1,H,W] index field),
  * an [N,N] (N=H*W on the *fixed* small grid) label-equality reduction for the
    per-cell component size.

Targets (all fixed HxW, colour-5 4-connected components on a 0 background):
  169  size_bg0_fg5  -> recolour each component by f(size) (here 5-size).
  374  rank_bg0_fg5  -> exactly 3 distinct-size comps; largest->1, mid->4, small->2.

Each candidate fires ONLY if it reproduces every available train/test pair in
numpy first, then the harness validates EXACTNESS on train+test+arc-gen.
"""
from __future__ import annotations

import onnx
from onnx import helper as oh
import numpy as np

from builders import _model
from ng_utils_shim import DATA_TYPE, HEIGHT, WIDTH, CHANNELS

INT64 = onnx.TensorProto.INT64


def _pairs(ex):
    return [(np.array(e["input"]), np.array(e["output"]))
            for e in ex.get("train", []) + ex.get("test", [])]


def _comps_np(mask):
    """4-connected components of a boolean mask -> list of cell-lists."""
    H, W = mask.shape
    seen = np.zeros((H, W), bool)
    res = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                st = [(i, j)]; seen[i, j] = True; cells = []
                while st:
                    r, c = st.pop(); cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True; st.append((nr, nc))
                res.append(cells)
    return res


def _max_conv_dist(mask):
    """Max over comps of BFS eccentricity from the min-index cell (= iters needed
    for min-index label propagation to converge)."""
    from collections import deque
    H, W = mask.shape
    md = 0
    for cells in _comps_np(mask):
        cs = set(cells)
        mn = min(cells, key=lambda rc: rc[0] * W + rc[1])
        dist = {mn: 0}; q = deque([mn])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in cs and nb not in dist:
                    dist[nb] = dist[(r, c)] + 1
                    md = max(md, dist[nb]); q.append(nb)
    return md


# ---------------------------------------------------------------------------
# shared graph blocks
# ---------------------------------------------------------------------------

def _fg_block(C, H, W):
    """Slice colour-C plane -> FG [1,1,H,W]."""
    sC = oh.make_tensor("sC", INT64, [3], [C, 0, 0])
    eC = oh.make_tensor("eC", INT64, [3], [C + 1, H, W])
    aC = oh.make_tensor("aC", INT64, [3], [1, 2, 3])
    n = [oh.make_node("Slice", ["input", "sC", "eC", "aC"], ["FG"])]
    return n, [sC, eC, aC]


def _label_block(H, W, K, BIG):
    """Min-index label propagation on FG -> converged label field [1,1,H,W]."""
    idxmBIG = [float(r * W + c + 1 - BIG) for r in range(H) for c in range(W)]
    cI = oh.make_tensor("idx", DATA_TYPE, [1, 1, H, W], idxmBIG)
    cBIG = oh.make_tensor("BIG", DATA_TYPE, [1], [float(BIG)])
    cone = oh.make_tensor("one1", DATA_TYPE, [1], [1.0])
    su = oh.make_tensor("su", INT64, [2], [0, 1]); eu = oh.make_tensor("eu", INT64, [2], [H, W + 1])
    sd = oh.make_tensor("sd", INT64, [2], [2, 1]); ed = oh.make_tensor("ed", INT64, [2], [H + 2, W + 1])
    sl = oh.make_tensor("sl", INT64, [2], [1, 0]); el = oh.make_tensor("el", INT64, [2], [H + 1, W])
    sr = oh.make_tensor("sr", INT64, [2], [1, 2]); er = oh.make_tensor("er", INT64, [2], [H + 1, W + 2])
    ax = oh.make_tensor("axl", INT64, [2], [2, 3])
    inits = [cI, cBIG, cone, su, eu, sd, ed, sl, el, sr, er, ax]
    nodes = [
        oh.make_node("Mul", ["FG", "idx"], ["lm"]),
        oh.make_node("Add", ["lm", "BIG"], ["L0"]),
        oh.make_node("Sub", ["one1", "FG"], ["nfg"]),
        oh.make_node("Mul", ["nfg", "BIG"], ["bigm"]),
    ]
    cur = "L0"
    for k in range(K):
        p = f"p{k}"
        nodes += [
            oh.make_node("Pad", [cur], [p + "P"], mode="constant", value=float(BIG),
                         pads=[0, 0, 1, 1, 0, 0, 1, 1]),
            oh.make_node("Slice", [p + "P", "su", "eu", "axl"], [p + "u"]),
            oh.make_node("Slice", [p + "P", "sd", "ed", "axl"], [p + "d"]),
            oh.make_node("Slice", [p + "P", "sl", "el", "axl"], [p + "l"]),
            oh.make_node("Slice", [p + "P", "sr", "er", "axl"], [p + "r"]),
            oh.make_node("Min", [cur, p + "u", p + "d", p + "l", p + "r"], [p + "m"]),
            oh.make_node("Max", [p + "m", "bigm"], [p + "L"]),
        ]
        cur = p + "L"
    return nodes, inits, cur


def _size_block(H, W, label):
    """Per-cell component size [1,1,H,W] from a converged label field."""
    N = H * W
    shA = oh.make_tensor("shA", INT64, [2], [N, 1])
    shB = oh.make_tensor("shB", INT64, [2], [1, N])
    shG = oh.make_tensor("shG", INT64, [4], [1, 1, H, W])
    nh = oh.make_tensor("nh", DATA_TYPE, [1], [-0.5])
    ph = oh.make_tensor("ph", DATA_TYPE, [1], [0.5])
    nodes = [
        oh.make_node("Reshape", [label, "shA"], ["LA"]),
        oh.make_node("Reshape", [label, "shB"], ["LB"]),
        oh.make_node("Sub", ["LA", "LB"], ["diff"]),
        oh.make_node("Greater", ["diff", "nh"], ["gt"]),
        oh.make_node("Less", ["diff", "ph"], ["lt"]),
        oh.make_node("And", ["gt", "lt"], ["eqb"]),
        oh.make_node("Cast", ["eqb"], ["eqf"], to=DATA_TYPE),
        oh.make_node("ReduceSum", ["eqf"], ["sz"], axes=[1], keepdims=1),
        oh.make_node("Reshape", ["sz", "shG"], ["szG"]),
    ]
    inits = [shA, shB, shG, nh, ph]
    return nodes, inits, "szG"


def _band(name, src, lo, hi, inits):
    """indicator(lo < src < hi) cast to float -> name."""
    tlo = oh.make_tensor(name + "lo", DATA_TYPE, [1], [float(lo)])
    thi = oh.make_tensor(name + "hi", DATA_TYPE, [1], [float(hi)])
    inits += [tlo, thi]
    return [
        oh.make_node("Greater", [src, name + "lo"], [name + "g"]),
        oh.make_node("Less", [src, name + "hi"], [name + "l"]),
        oh.make_node("And", [name + "g", name + "l"], [name + "b"]),
        oh.make_node("Cast", [name + "b"], [name], to=DATA_TYPE),
    ]


# ---------------------------------------------------------------------------
# task 169: recolour each colour-C component by f(size).
# ---------------------------------------------------------------------------

def _size_recolor_model(H, W, C, K, size2color, default):
    """All FG cells get `default`; sizes mapping to a non-default colour override
    it.  Unseen sizes therefore fall back to `default` (robust threshold rule)."""
    fgn, fgi = _fg_block(C, H, W)
    ln, li, lab = _label_block(H, W, K, 99999)
    sn, si, szG = _size_block(H, W, lab)
    nodes = fgn + ln + sn
    inits = list(fgi) + list(li) + list(si)
    one = oh.make_tensor("oneZ", DATA_TYPE, [1], [1.0])
    inits.append(one)
    by_color = {}
    for s, col in size2color.items():
        by_color.setdefault(col, []).append(s)
    ch = {}
    override_parts = []
    for col, sizes in by_color.items():
        if col == default:
            continue
        parts = []
        for s in sizes:
            nm = f"sz{s}c{col}"
            nodes += _band(nm, szG, s - 0.5, s + 0.5, inits)
            parts.append(nm)
        acc = parts[0]
        if len(parts) > 1:
            nodes.append(oh.make_node("Sum", parts, [f"col{col}sum"]))
            acc = f"col{col}sum"
        nodes.append(oh.make_node("Mul", ["FG", acc], [f"ch{col}"]))
        ch[col] = f"ch{col}"
        override_parts.append(f"ch{col}")
    # default channel = FG minus all overridden cells
    if override_parts:
        if len(override_parts) > 1:
            nodes.append(oh.make_node("Sum", override_parts, ["ovsum"]))
            nodes.append(oh.make_node("Sub", ["FG", "ovsum"], [f"ch{default}"]))
        else:
            nodes.append(oh.make_node("Sub", ["FG", override_parts[0]], [f"ch{default}"]))
    else:
        nodes.append(oh.make_node("Mul", ["FG", "oneZ"], [f"ch{default}"]))
    ch[default] = f"ch{default}"
    nodes.append(oh.make_node("Sub", ["oneZ", "FG"], ["ch0"]))
    order = ["ch0"]
    need_zero = False
    for c in range(1, CHANNELS):
        if c in ch:
            order.append(ch[c])
        else:
            order.append("z0"); need_zero = True
    if need_zero:
        nodes.append(oh.make_node("Mul", ["FG", "FG"], ["fgsq"]))
        nodes.append(oh.make_node("Sub", ["fgsq", "fgsq"], ["z0"]))
    nodes.append(oh.make_node("Concat", order, ["small"], axis=1))
    nodes.append(oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]))
    return _model(nodes, inits)


def _detect_size_recolor(prs):
    hs = {a.shape[0] for a, b in prs}; ws = {a.shape[1] for a, b in prs}
    if len(hs) != 1 or len(ws) != 1:
        return None
    H = hs.pop(); W = ws.pop()
    if H > HEIGHT or W > WIDTH or H < 1 or W < 1:
        return None
    # single foreground colour
    fgcols = set()
    for a, b in prs:
        if a.shape != b.shape:
            return None
        u = set(np.unique(a).tolist()) - {0}
        fgcols |= u
    if len(fgcols) != 1:
        return None
    C = fgcols.pop()
    if not (1 <= C <= 9):
        return None
    size2color = {}
    cellcount = {}
    maxconv = 0
    for a, b in prs:
        mask = (a == C)
        maxconv = max(maxconv, _max_conv_dist(mask))
        for cells in _comps_np(mask):
            sz = len(cells)
            cols = {int(b[r, c]) for r, c in cells}
            if len(cols) != 1:
                return None
            col = cols.pop()
            if col == 0 or col > 9:
                return None
            if sz in size2color and size2color[sz] != col:
                return None  # size not a function of color
            size2color[sz] = col
            cellcount[col] = cellcount.get(col, 0) + len(cells)
        # background must stay 0
        if (b[a == 0] != 0).any():
            return None
    if not size2color:
        return None
    default = max(cellcount, key=lambda k: cellcount[k])  # majority colour
    K = maxconv + 6
    return (H, W, C, K, size2color, default)


# ---------------------------------------------------------------------------
# task 374: exactly 3 distinct-size comps. largest->cL, mid->cM, smallest->cS.
# ---------------------------------------------------------------------------

def _rank3_model(H, W, C, K, cL, cM, cS):
    fgn, fgi = _fg_block(C, H, W)
    ln, li, lab = _label_block(H, W, K, 99999)
    sn, si, szG = _size_block(H, W, lab)
    nodes = fgn + ln + sn
    inits = list(fgi) + list(li) + list(si)
    one = oh.make_tensor("oneZ", DATA_TYPE, [1], [1.0])
    big = oh.make_tensor("bigZ", DATA_TYPE, [1], [99999.0])
    nh = oh.make_tensor("nhr", DATA_TYPE, [1], [-0.5])
    ph = oh.make_tensor("phr", DATA_TYPE, [1], [0.5])
    inits += [one, big, nh, ph]
    nodes += [
        # bg cells all share one label, so szG at bg = bg-count -> must mask to fg.
        oh.make_node("Mul", [szG, "FG"], ["szF"]),         # fg size, bg->0
        oh.make_node("ReduceMax", ["szF"], ["maxsz"], axes=[2, 3], keepdims=1),
        # min size over fg only: bg cells pushed to +BIG
        oh.make_node("Sub", ["oneZ", "FG"], ["nfgR"]),
        oh.make_node("Mul", ["nfgR", "bigZ"], ["bgbig"]),
        oh.make_node("Add", ["szF", "bgbig"], ["szM"]),    # fg size, bg->BIG
        oh.make_node("ReduceMin", ["szM"], ["minsz"], axes=[2, 3], keepdims=1),
        # eqmax / eqmin bands
        oh.make_node("Sub", [szG, "maxsz"], ["dmax"]),
        oh.make_node("Greater", ["dmax", "nhr"], ["gmax"]),
        oh.make_node("Less", ["dmax", "phr"], ["lmax"]),
        oh.make_node("And", ["gmax", "lmax"], ["emaxb"]),
        oh.make_node("Cast", ["emaxb"], ["emax"], to=DATA_TYPE),
        oh.make_node("Sub", [szG, "minsz"], ["dmin"]),
        oh.make_node("Greater", ["dmin", "nhr"], ["gmin"]),
        oh.make_node("Less", ["dmin", "phr"], ["lmin"]),
        oh.make_node("And", ["gmin", "lmin"], ["eminb"]),
        oh.make_node("Cast", ["eminb"], ["emin"], to=DATA_TYPE),
        # per-cell channels
        oh.make_node("Mul", ["FG", "emax"], ["chL"]),
        oh.make_node("Mul", ["FG", "emin"], ["chS"]),
        oh.make_node("Sub", ["FG", "chL"], ["t1"]),
        oh.make_node("Sub", ["t1", "chS"], ["chM"]),
        oh.make_node("Sub", ["oneZ", "FG"], ["ch0"]),
    ]
    chmap = {cL: "chL", cM: "chM", cS: "chS"}
    nodes.append(oh.make_node("Mul", ["FG", "minsz"], ["zR0"]))  # placeholder zero plane (fg*minsz then sub)
    nodes.append(oh.make_node("Sub", ["zR0", "zR0"], ["z0"]))
    order = ["ch0"]
    for c in range(1, CHANNELS):
        order.append(chmap.get(c, "z0"))
    nodes.append(oh.make_node("Concat", order, ["small"], axis=1))
    nodes.append(oh.make_node("Pad", ["small"], ["output"], mode="constant", value=0.0,
                              pads=[0, 0, 0, 0, 0, 0, HEIGHT - H, WIDTH - W]))
    return _model(nodes, inits)


def _detect_rank3(prs):
    hs = {a.shape[0] for a, b in prs}; ws = {a.shape[1] for a, b in prs}
    if len(hs) != 1 or len(ws) != 1:
        return None
    H = hs.pop(); W = ws.pop()
    if H > HEIGHT or W > WIDTH:
        return None
    fgcols = set()
    for a, b in prs:
        if a.shape != b.shape:
            return None
        u = set(np.unique(a).tolist()) - {0}
        fgcols |= u
    if len(fgcols) != 1:
        return None
    C = fgcols.pop()
    cL = cM = cS = None
    maxconv = 0
    for a, b in prs:
        mask = (a == C)
        maxconv = max(maxconv, _max_conv_dist(mask))
        comps = _comps_np(mask)
        if len(comps) != 3:
            return None
        sizes = [len(c) for c in comps]
        if len(set(sizes)) != 3:
            return None
        order = sorted(range(3), key=lambda k: -sizes[k])  # large, mid, small
        cols = []
        for k in order:
            cc = {int(b[r, c]) for r, c in comps[k]}
            if len(cc) != 1:
                return None
            cols.append(cc.pop())
        if cL is None:
            cL, cM, cS = cols
        elif (cL, cM, cS) != tuple(cols):
            return None
        if (b[a == 0] != 0).any():
            return None
    if cL is None or 0 in (cL, cM, cS):
        return None
    K = maxconv + 8
    return (H, W, C, K, cL, cM, cS)


# ---------------------------------------------------------------------------
# task 346 (fragswatch): exactly one cell is enclosed by a same-colour 3x3 ring
# (8 neighbours all equal to a nonzero colour A, centre colour B != A).  Output
# is 1x1 == B.  Detect ring centres with a depthwise 3x3 ring conv (count of
# same-colour neighbours == 8), keep nonzero ring colours, pick centre colour,
# reduce to origin.
# ---------------------------------------------------------------------------

def _fragswatch_model():
    # one-hot -> scalar colour-value grid via a 1x1 conv with weights [0..9].
    valW = oh.make_tensor("valW", DATA_TYPE, [1, CHANNELS, 1, 1],
                          [float(c) for c in range(CHANNELS)])
    colorvals = oh.make_tensor("colorvals", DATA_TYPE, [1, CHANNELS, 1, 1],
                               [float(c) for c in range(CHANNELS)])
    h5 = oh.make_tensor("h5", DATA_TYPE, [1], [0.5])
    inits = [valW, colorvals, h5]
    # 8-neighbour slice windows on a 1-padded value grid (vp is [1,1,32,32]).
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    slice_inits = []
    ax = oh.make_tensor("nax", INT64, [2], [2, 3]); slice_inits.append(ax)
    nodes = [
        oh.make_node("Conv", ["input", "valW"], ["valg"], kernel_shape=[1, 1],
                     pads=[0, 0, 0, 0]),
        oh.make_node("Pad", ["valg"], ["vp"], mode="constant", value=0.0,
                     pads=[0, 0, 1, 1, 0, 0, 1, 1]),
    ]
    names = []
    for i, (dr, dc) in enumerate(dirs):
        r0, c0 = 1 + dr, 1 + dc
        s = oh.make_tensor(f"ns{i}", INT64, [2], [r0, c0])
        e = oh.make_tensor(f"ne{i}", INT64, [2], [r0 + HEIGHT, c0 + WIDTH])
        slice_inits += [s, e]
        nodes.append(oh.make_node("Slice", ["vp", f"ns{i}", f"ne{i}", "nax"], [f"nb{i}"]))
        names.append(f"nb{i}")
    nodes += [
        oh.make_node("Max", names, ["rmax"]),
        oh.make_node("Min", names, ["rmin"]),
        oh.make_node("Sub", ["rmax", "rmin"], ["rspan"]),
        oh.make_node("Less", ["rspan", "h5"], ["equalb"]),       # all 8 equal
        oh.make_node("Greater", ["rmin", "h5"], ["nzb"]),        # ring colour nonzero
        oh.make_node("Sub", ["valg", "rmax"], ["cdiff"]),
        oh.make_node("Abs", ["cdiff"], ["acdiff"]),
        oh.make_node("Greater", ["acdiff", "h5"], ["ctrb"]),     # centre != ring colour
        oh.make_node("And", ["equalb", "nzb"], ["e1b"]),
        oh.make_node("And", ["e1b", "ctrb"], ["Mb"]),
        oh.make_node("Cast", ["Mb"], ["Mf"], to=DATA_TYPE),
        oh.make_node("Mul", ["valg", "Mf"], ["valM"]),
        oh.make_node("ReduceSum", ["valM"], ["Bsel"], axes=[2, 3], keepdims=1),
        oh.make_node("Sub", ["colorvals", "Bsel"], ["odiff"]),
        oh.make_node("Abs", ["odiff"], ["aodiff"]),
        oh.make_node("Less", ["aodiff", "h5"], ["onehb"]),
        oh.make_node("Cast", ["onehb"], ["oneh"], to=DATA_TYPE),
        oh.make_node("Pad", ["oneh"], ["output"], mode="constant", value=0.0,
                     pads=[0, 0, 0, 0, 0, 0, HEIGHT - 1, WIDTH - 1]),
    ]
    return _model(nodes, inits + slice_inits)


def _detect_fragswatch(prs):
    for a, b in prs:
        if b.shape != (1, 1):
            return False
        H, Wd = a.shape
        centers = []
        for r in range(1, H - 1):
            for c in range(1, Wd - 1):
                ring = [a[r - 1, c - 1], a[r - 1, c], a[r - 1, c + 1],
                        a[r, c - 1], a[r, c + 1],
                        a[r + 1, c - 1], a[r + 1, c], a[r + 1, c + 1]]
                if len(set(int(x) for x in ring)) == 1 and ring[0] != 0 and a[r, c] != ring[0]:
                    centers.append((r, c))
        if len(centers) != 1:
            return False
        r, c = centers[0]
        if int(b[0, 0]) != int(a[r, c]):
            return False
    return True


# ---------------------------------------------------------------------------

def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []

    r3 = _detect_rank3(prs)
    if r3 is not None:
        H, W, C, K, cL, cM, cS = r3
        out.append(("golf4_rank3", _rank3_model(H, W, C, K, cL, cM, cS)))

    sr = _detect_size_recolor(prs)
    if sr is not None:
        H, W, C, K, s2c, default = sr
        out.append(("golf4_sizrecolor", _size_recolor_model(H, W, C, K, s2c, default)))

    if _detect_fragswatch(prs):
        out.append(("golf4_fragswatch", _fragswatch_model()))

    return out
