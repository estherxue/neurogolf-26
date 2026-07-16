"""family_crk7_1 : crack hard unsolved tasks (slice U[1::6]).

Solved here:
  * task 381  -- "bridge color-2 rectangles": for every maximal horizontal run of
    background cells flanked by color-2 on both sides, fill the run with color 9,
    UNLESS any interior column of the run has a color-2 directly above or below the
    run's row (i.e. the run is "blocked" by an intervening rectangle).
    Expressed with only opset-10 ops via two Hillis-Steele prefix-max scans that find,
    for each cell, whether the nearest "special" cell (a 2 or a blocker) on each side
    is a 2 (clean) rather than a blocker (dirty run).

The remaining tasks in the slice (5,25,54,76,90,118,145,165,184,209,264,319,361) were
decoded but require object-level reasoning (centroids/C4-symmetry centres, legends,
flood-fill, rectangle-adjacency graphs, data-dependent crops) that is not expressible
exactly in the static, loop-free opset-10 subset (no Div/NonZero/ScatterND/Loop); for
several (e.g. 90) the top-left padding makes edge-touching answers fundamentally
ambiguous.  See notes.
"""
from __future__ import annotations
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, ng

INT64 = onnx.TensorProto.INT64
NEG = -1.0e9
H = W = 30


def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ----- builder for task 381 --------------------------------------------------

def build_381():
    nodes = []
    inits = []
    ctr = [0]

    def nm(p):
        ctr[0] += 1
        return f"{p}{ctr[0]}"

    def add_init(name, arr, dtype=DATA_TYPE):
        arr = np.asarray(arr)
        t = oh.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist())
        inits.append(t)
        return name

    # slice helpers (opset-10 Slice takes tensor inputs)
    def slice_axis(x, start, end, axis, out=None):
        out = out or nm("sl")
        st = add_init(nm("st"), [start], INT64)
        en = add_init(nm("en"), [end], INT64)
        ax = add_init(nm("ax"), [axis], INT64)
        nodes.append(oh.make_node("Slice", [x, st, en, ax], [out]))
        return out

    def pad(x, pads, value, out=None):
        out = out or nm("pad")
        nodes.append(oh.make_node("Pad", [x], [out], mode="constant",
                                  value=float(value), pads=list(pads)))
        return out

    def shift_w(x, k, fill):
        # k>0 -> content moves right (out[c]=x[c-k]); k<0 -> moves left
        if k > 0:
            p = pad(x, [0, 0, 0, k, 0, 0, 0, 0], fill)
            return slice_axis(p, 0, W, 3)
        else:
            kk = -k
            p = pad(x, [0, 0, 0, 0, 0, 0, 0, kk], fill)
            return slice_axis(p, kk, W + kk, 3)

    def shift_h(x, k, fill):
        if k > 0:
            p = pad(x, [0, 0, k, 0, 0, 0, 0, 0], fill)
            return slice_axis(p, 0, H, 2)
        else:
            kk = -k
            p = pad(x, [0, 0, 0, 0, 0, 0, kk, 0], fill)
            return slice_axis(p, kk, H + kk, 2)

    def binop(op, a, b, out=None):
        out = out or nm(op.lower())
        nodes.append(oh.make_node(op, [a, b], [out]))
        return out

    def prefix_max_w(v, reverse=False):
        inc = v
        k = 1
        while k < W:
            sh = shift_w(inc, (-k if reverse else k), NEG)
            inc = binop("Max", inc, sh)
            k *= 2
        # exclusive: shift by 1
        return shift_w(inc, (-1 if reverse else 1), NEG)

    # ---- channels
    bg = slice_axis("input", 0, 1, 1)      # [1,1,30,30]
    twos = slice_axis("input", 2, 3, 1)
    mid = slice_axis("input", 1, 9, 1)     # channels 1..8 (incl ch2), unchanged

    # blockers: bg cell with a 2 directly above or below
    ta = shift_h(twos, 1, 0.0)             # ta[r]=twos[r-1]
    tb = shift_h(twos, -1, 0.0)            # tb[r]=twos[r+1]
    vert2 = binop("Max", ta, tb)
    blk = binop("Mul", bg, vert2)
    special = binop("Add", twos, blk)
    one = add_init("one", np.ones((1, 1, 1, 1)))
    special = binop("Min", special, one)   # clip to <=1

    # column position encodings (broadcast over height)
    colL = np.arange(1, W + 1).reshape(1, 1, 1, W).astype(np.float32)   # 1..30
    colR = np.arange(W, 0, -1).reshape(1, 1, 1, W).astype(np.float32)   # 30..1
    cL = add_init("colL", colL)
    cR = add_init("colR", colR)
    half = add_init("half", [0.5])

    def side_ok(colconst, reverse):
        twoV = binop("Mul", twos, colconst)
        spV = binop("Mul", special, colconst)
        twoM = prefix_max_w(twoV, reverse=reverse)
        spM = prefix_max_w(spV, reverse=reverse)
        diff = binop("Sub", twoM, spM)
        ad = nm("abs"); nodes.append(oh.make_node("Abs", [diff], [ad]))
        lt = nm("lt"); nodes.append(oh.make_node("Less", [ad, half], [lt]))
        ltf = nm("ltf"); nodes.append(oh.make_node("Cast", [lt], [ltf], to=DATA_TYPE))
        gt = nm("gt"); nodes.append(oh.make_node("Greater", [twoM, half], [gt]))
        gtf = nm("gtf"); nodes.append(oh.make_node("Cast", [gt], [gtf], to=DATA_TYPE))
        return binop("Mul", ltf, gtf)

    leftOK = side_ok(cL, reverse=False)
    rightOK = side_ok(cR, reverse=True)

    notblk = binop("Sub", one, blk)           # broadcasts (1 - blk)
    fill = binop("Mul", bg, notblk)
    fill = binop("Mul", fill, leftOK)
    fill = binop("Mul", fill, rightOK)         # [1,1,30,30]

    ch0 = binop("Sub", bg, fill)               # new background channel
    # output = concat(ch0, mid(ch1..8), fill(=ch9))
    nodes.append(oh.make_node("Concat", [ch0, mid, fill], ["output"], axis=1))

    return _model(nodes, inits)


# ----- builder for task 25 ---------------------------------------------------

def build_25():
    nodes = []
    inits = []
    ctr = [0]

    def nm(p):
        ctr[0] += 1
        return f"{p}{ctr[0]}"

    def add_init(name, arr, dtype=INT64):
        arr = np.asarray(arr)
        inits.append(oh.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist()))
        return name

    def slice_t(x, start, end, axis):
        out = nm("sl")
        st = add_init(nm("st"), [start]); en = add_init(nm("en"), [end]); ax = add_init(nm("ax"), [axis])
        nodes.append(oh.make_node("Slice", [x, st, en, ax], [out]))
        return out

    def pad(x, pads, value):
        out = nm("pad")
        nodes.append(oh.make_node("Pad", [x], [out], mode="constant", value=float(value), pads=list(pads)))
        return out

    def shift_w(x, k, fill):
        # k>0: content moves right (out[c]=x[c-k]); k<0: moves left
        if k > 0:
            p = pad(x, [0, 0, 0, k, 0, 0, 0, 0], fill); return slice_t(p, 0, W, 3)
        kk = -k
        p = pad(x, [0, 0, 0, 0, 0, 0, 0, kk], fill); return slice_t(p, kk, W + kk, 3)

    def op(typ, ins, **attr):
        out = nm(typ.lower())
        nodes.append(oh.make_node(typ, list(ins), [out], **attr))
        return out

    def reduce(typ, x, axes, keep=1):
        return op(typ, [x], axes=list(axes), keepdims=keep)

    def clip01(x):
        return op("Clip", [x], min=0.0, max=1.0)

    def prefix_max_w(v, reverse):
        inc = v; k = 1
        while k < W:
            inc = op("Max", [inc, shift_w(inc, (-k if reverse else k), NEG)]); k *= 2
        return shift_w(inc, (-1 if reverse else 1), NEG)

    def vproc(X9):
        nonzero = reduce("ReduceSum", X9, [1])          # [1,1,30,30]
        row_ne = reduce("ReduceMax", nonzero, [3])      # [1,1,30,1]
        row_ne_c = clip01(row_ne)
        row_empty = op("Sub", [add_init(nm("one"), np.ones((1, 1, 1, 1)), DATA_TYPE), row_ne_c])
        mx = op("Max", [row_empty, X9])                 # broadcast [1,9,30,30]
        vline = reduce("ReduceMin", mx, [2])            # [1,9,1,30]
        lineCells = op("Mul", [vline, row_ne_c])        # [1,9,30,30]
        oneB = add_init(nm("one"), np.ones((1, 1, 1, 1)), DATA_TYPE)
        notv = op("Sub", [oneB, vline])
        dots = op("Mul", [X9, notv])
        leftOf = clip01(prefix_max_w(vline, reverse=True))
        rightOf = clip01(prefix_max_w(vline, reverse=False))
        leftDots = reduce("ReduceMax", op("Mul", [dots, leftOf]), [3])     # [1,9,30,1]
        rightDots = reduce("ReduceMax", op("Mul", [dots, rightOf]), [3])
        leftAdj = shift_w(vline, -1, 0.0)
        rightAdj = shift_w(vline, 1, 0.0)
        leftProj = op("Mul", [leftAdj, leftDots])
        rightProj = op("Mul", [rightAdj, rightDots])
        return clip01(op("Add", [op("Add", [lineCells, leftProj]), rightProj]))

    X9 = slice_t("input", 1, 10, 1)
    contentMask = reduce("ReduceMax", "input", [1])      # [1,1,30,30]
    vert = vproc(X9)
    Xt = op("Transpose", ["input"], perm=[0, 1, 3, 2])
    X9t = slice_t(Xt, 1, 10, 1)
    horizT = vproc(X9t)
    horiz = op("Transpose", [horizT], perm=[0, 1, 3, 2])
    res9 = clip01(op("Add", [vert, horiz]))
    sumc = clip01(reduce("ReduceSum", res9, [1]))
    oneF = add_init("oneF", np.ones((1, 1, 1, 1)), DATA_TYPE)
    ch0 = op("Mul", [contentMask, op("Sub", [oneF, sumc])])
    nodes.append(oh.make_node("Concat", [ch0, res9], ["output"], axis=1))
    return _model(nodes, inits)


def _np_rule_25(a):
    H, W2 = a.shape
    b = np.zeros_like(a)
    row_lines = {}; col_lines = {}
    for r in range(H):
        v = a[r]
        if (v != 0).all() and len(set(v.tolist())) == 1:
            row_lines[r] = v[0]
    for c in range(W2):
        v = a[:, c]
        if (v != 0).all() and len(set(v.tolist())) == 1:
            col_lines[c] = v[0]
    for r, col in row_lines.items(): b[r, :] = col
    for c, col in col_lines.items(): b[:, c] = col
    c2col = {col: c for c, col in col_lines.items()}
    c2row = {col: r for r, col in row_lines.items()}
    for r in range(H):
        for c in range(W2):
            v = a[r, c]
            if v == 0 or r in row_lines or c in col_lines:
                continue
            if v in c2col:
                Lc = c2col[v]; nc = Lc - 1 if c < Lc else Lc + 1
                if 0 <= nc < W2: b[r, nc] = v
            elif v in c2row:
                Lr = c2row[v]; nr = Lr - 1 if r < Lr else Lr + 1
                if 0 <= nr < H: b[nr, c] = v
    return b


# ----- numpy reference (for detection) ---------------------------------------

def _np_rule_381(a):
    b = a.copy()
    Hh, Ww = a.shape
    for r in range(Hh):
        prev = None
        for cc in range(Ww):
            if a[r, cc] == 2:
                if prev is not None and cc - prev > 1:
                    seg = range(prev + 1, cc)
                    if all(a[r, x] == 0 for x in seg):
                        blocked = any(
                            (r - 1 >= 0 and a[r - 1, x] == 2) or
                            (r + 1 < Hh and a[r + 1, x] == 2) for x in seg)
                        if not blocked:
                            for x in seg:
                                b[r, x] = 9
                prev = cc
    return b


def _fits(examples, rule):
    prs = [(np.array(e["input"], int), np.array(e["output"], int))
           for s in ("train", "test") for e in examples.get(s, [])]
    if not prs:
        return False
    for a, b in prs:
        if a.shape != b.shape:
            return False
        try:
            if not np.array_equal(rule(a), b):
                return False
        except Exception:
            return False
    return True


def candidates(examples):
    out = []
    if _fits(examples, _np_rule_381):
        try:
            out.append(("crk7_bridge381", build_381()))
        except Exception:
            pass
    if _fits(examples, _np_rule_25):
        try:
            out.append(("crk7_dot2line25", build_25()))
        except Exception:
            pass
    return out
