"""GRID-SPLIT BY SEPARATOR LINES (origin-anchored, opset 10).

Many ARC grids are divided by constant-colour separator rows/columns into a
REGULAR array of equal sub-cells ("panels").  This family detects the separator
colour and the (constant, fixed-period) split geometry, then realises one of
several panel-combining / panel-reducing rules:

  overlay         Combine all panels into one panel-sized output, colour-
                  preserving, with reading-order priority (top-left panel wins on
                  conflict; reverse order also tried).  A 2-panel split may also
                  fold the second panel onto the first by mirroring it across the
                  split axis (Slice steps=-1).  Realised with per-panel Slice +
                  Where (front-over-back), the half-sized result Pad-anchored to
                  the top-left.

  gate            output[i,j] = C  iff  f(count of "active" panels at (i,j)),
                  else background, with f a count predicate (>=k / <=k / ==k ->
                  AND / OR / NOR / NAND / XOR-of-two / "exactly one").  C is a
                  fixed colour inferred from the targets.  Each panel's "active"
                  plane is just 1-channel0 over its sub-window.

  selred          Per-panel non-bg cell COUNT (a single strided Conv with a ones
                  kernel = one number per panel), then pick the panels matching a
                  count predicate (argmax / >=k / ==k / ...).  The output is the
                  npr x npc grid of {mark-colour | background} cells.

  selbroad        Same selection, but broadcast back over the ORIGINAL grid: the
                  separator lines are kept, every cell of a selected panel is
                  filled (with a fixed mark colour, or with that grid's own
                  content colour), every other real cell becomes background.

Origin safety: the split position depends on the grid size, so the rule is only
well defined when the geometry is CONSTANT across every example.  We therefore
emit only for tasks whose input shape AND separator structure are identical
across all train+test+arc-gen pairs; the panel-sized / grid-sized result is then
zero-padded back to 30x30 so the content stays anchored top-left for any (fixed)
size.  Detection mirrors the ONNX semantics exactly and validates EVERY available
pair (the grader's gate), so wrong hypotheses are dropped before scoring.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL
_NEG = -(1 << 31)            # full-axis reverse Slice sentinel


# --------------------------------------------------------------------------- #
# tiny graph accumulator                                                      #
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._k = 0
        self._one = None

    def name(self, p="t"):
        self._k += 1
        return f"{p}{self._k}"

    def f(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         [float(v) for v in np.asarray(vals).ravel()]))
        return nm

    def i64(self, vals):
        nm = self.name("i")
        self.inits.append(oh.make_tensor(nm, INT64, [len(vals)], [int(v) for v in vals]))
        return nm

    def one(self):
        if self._one is None:
            self._one = self.f([1, 1, 1, 1], [1.0])
        return self._one

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _model(g):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(g.nodes, "g", [x], [y], g.inits)
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _onehot(k):
    return [1.0 if c == k else 0.0 for c in range(CHANNELS)]


def _pad_out(g, win, h, w, name="output"):
    """[1,10,h,w] -> [1,10,30,30], zero-padded, origin anchored."""
    if h == HEIGHT and w == WIDTH:
        g.node("Identity", [win], name)
    else:
        g.node("Pad", [win], name, mode="constant", value=0.0,
               pads=[0, 0, 0, 0, 0, 0, HEIGHT - h, WIDTH - w])


# --------------------------------------------------------------------------- #
# panel windows                                                               #
# --------------------------------------------------------------------------- #
def _panel(g, rect):
    """Full [1,10,h,w] sub-window."""
    r0, r1, c0, c1 = rect
    return g.node("Slice", ["input", g.i64([r0, c0]), g.i64([r1, c1]), g.i64([2, 3])])


def _panel_ch0(g, rect):
    """Background channel-0 plane [1,1,h,w] of the sub-window (1 where bg)."""
    r0, r1, c0, c1 = rect
    return g.node("Slice", ["input", g.i64([0, r0, c0]), g.i64([1, r1, c1]), g.i64([1, 2, 3])])


def _flip(g, t, axis, h, w):
    """Reverse a [1,10,h,w] tensor along axis (2=rows,3=cols)."""
    n = h if axis == 2 else w
    return g.node("Slice", [t, g.i64([n - 1]), g.i64([_NEG]), g.i64([axis]), g.i64([-1])])


def _ch0_of(g, t):
    """Channel-0 plane of an intermediate [1,10,h,w] tensor."""
    return g.node("Slice", [t, g.i64([0]), g.i64([1]), g.i64([1])])


# --------------------------------------------------------------------------- #
# builders                                                                     #
# --------------------------------------------------------------------------- #
def build_overlay(rects, order, h, w, flips):
    """Colour-preserving overlay. `order` lists panel indices high->low priority.
    flips: dict panel_index -> axis (2/3) to mirror that panel, or None."""
    g = _G()
    full = {}
    for k in order:
        t = _panel(g, rects[k])
        if flips and flips.get(k) is not None:
            t = _flip(g, t, flips[k], h, w)
        full[k] = t
    res = full[order[-1]]                                  # lowest priority base
    for k in order[-2::-1]:                                # then higher on top
        front = full[k]
        condbg = g.node("Cast", [_ch0_of(g, front)], to=BOOL)
        res = g.node("Where", [condbg, res, front])
    _pad_out(g, res, h, w)
    return _model(g)


def build_gate(rects, op, k, C, h, w):
    """output = C where predicate(active-count), else background."""
    g = _G()
    one = g.one()
    actives = [g.node("Sub", [one, _panel_ch0(g, r)]) for r in rects]
    cnt = actives[0]
    for a in actives[1:]:
        cnt = g.node("Add", [cnt, a])
    if op == "ge":
        m = g.node("Greater", [cnt, g.f([1, 1, 1, 1], [k - 0.5])])
        mask = g.node("Cast", [m], to=DATA_TYPE)
    elif op == "le":
        m = g.node("Less", [cnt, g.f([1, 1, 1, 1], [k + 0.5])])
        mask = g.node("Cast", [m], to=DATA_TYPE)
    else:  # eq
        gt = g.node("Cast", [g.node("Greater", [cnt, g.f([1, 1, 1, 1], [k - 0.5])])], to=DATA_TYPE)
        lt = g.node("Cast", [g.node("Less", [cnt, g.f([1, 1, 1, 1], [k + 0.5])])], to=DATA_TYPE)
        mask = g.node("Mul", [gt, lt])
    notm = g.node("Sub", [one, mask])
    vecC = g.f([1, CHANNELS, 1, 1], _onehot(C))
    vec0 = g.f([1, CHANNELS, 1, 1], _onehot(0))
    win = g.node("Add", [g.node("Mul", [mask, vecC]), g.node("Mul", [notm, vec0])])
    _pad_out(g, win, h, w)
    return _model(g)


def _nonbg_presence(g):
    """M = non-background presence [1,1,30,30]  (= sum_{c>=1} input_c)."""
    R = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)        # [1,1,30,30]
    ch0 = g.node("Slice", ["input", g.i64([0]), g.i64([1]), g.i64([1])])
    return g.node("Sub", [R, ch0])


def _panel_counts(g, st):
    """Per-panel non-bg count -> [1,1,npr,npc] via a single strided Conv."""
    M = _nonbg_presence(g)
    ch, cw = st["ch"], st["cw"]
    per_r = st["per_r"]; per_c = st["per_c"]
    ker = g.f([1, 1, ch, cw], np.ones((1, 1, ch, cw)))
    conv = g.node("Conv", [M, ker], kernel_shape=[ch, cw],
                  strides=[per_r, per_c], pads=[0, 0, 0, 0])
    counts = g.node("Slice", [conv, g.i64([0, 0]), g.i64([st["npr"], st["npc"]]), g.i64([2, 3])])
    return counts


def _select_mask(g, counts, pred):
    """pred = ('max',) / ('ge',k) / ('le',k) / ('eq',k) -> {0,1} [1,1,npr,npc]."""
    half = g.f([1, 1, 1, 1], [0.5])
    if pred[0] == "max":
        mx = g.node("ReduceMax", [counts], axes=[2, 3], keepdims=1)
        sel = g.node("Cast", [g.node("Greater", [counts, g.node("Sub", [mx, half])])], to=DATA_TYPE)
    elif pred[0] == "ge":
        sel = g.node("Cast", [g.node("Greater", [counts, g.f([1, 1, 1, 1], [pred[1] - 0.5])])], to=DATA_TYPE)
    elif pred[0] == "le":
        sel = g.node("Cast", [g.node("Less", [counts, g.f([1, 1, 1, 1], [pred[1] + 0.5])])], to=DATA_TYPE)
    else:  # eq
        gt = g.node("Cast", [g.node("Greater", [counts, g.f([1, 1, 1, 1], [pred[1] - 0.5])])], to=DATA_TYPE)
        lt = g.node("Cast", [g.node("Less", [counts, g.f([1, 1, 1, 1], [pred[1] + 0.5])])], to=DATA_TYPE)
        sel = g.node("Mul", [gt, lt])
    return sel


def build_selred(st, pred, M):
    """Reduced output: npr x npc grid of {mark M | background}."""
    g = _G()
    counts = _panel_counts(g, st)
    sel = _select_mask(g, counts, pred)                    # [1,1,npr,npc]
    notsel = g.node("Sub", [g.one(), sel])
    vecM = g.f([1, CHANNELS, 1, 1], _onehot(M))
    vec0 = g.f([1, CHANNELS, 1, 1], _onehot(0))
    win = g.node("Add", [g.node("Mul", [sel, vecM]), g.node("Mul", [notsel, vec0])])
    _pad_out(g, win, st["npr"], st["npc"])
    return _model(g)


def _content_onehot(g, S):
    """[1,10,1,1] one-hot of the dominant non-bg, non-separator colour."""
    cnt = g.node("ReduceSum", ["input"], axes=[2, 3], keepdims=1)   # [1,10,1,1]
    off = [(-1.0e6 if (c == 0 or c == S) else 0.0) for c in range(CHANNELS)]
    masked = g.node("Add", [cnt, g.f([1, CHANNELS, 1, 1], off)])
    mx = g.node("ReduceMax", [masked], axes=[1], keepdims=1)
    sel = g.node("Cast", [g.node("Greater", [masked, g.node("Sub", [mx, g.f([1, 1, 1, 1], [0.5])])])],
                 to=DATA_TYPE)
    nbg = g.f([1, CHANNELS, 1, 1], [0.0] + [1.0] * (CHANNELS - 1))
    return g.node("Mul", [sel, nbg])


def build_selbroad(st, pred, mark, S, panelmask, sepmask):
    """Broadcast selection back over the original grid (separators kept).
    mark = ('fixed', colour) or ('content',)."""
    g = _G()
    h, w = st["H"], st["W"]
    counts = _panel_counts(g, st)
    sel = _select_mask(g, counts, pred)                    # [1,1,npr,npc]
    selbig = g.node("Resize", [sel, g.f([4], [1.0, 1.0, float(st["per_r"]), float(st["per_c"])])],
                    mode="nearest")
    selc = g.node("Slice", [selbig, g.i64([0, 0]), g.i64([h, w]), g.i64([2, 3])])  # [1,1,h,w]
    pmask = g.f([1, 1, h, w], panelmask)
    smask = g.f([1, 1, h, w], sepmask)
    fill = g.node("Mul", [selc, pmask])                    # selected panel cells
    unfilled = g.node("Sub", [pmask, fill])                # panel cells not selected -> bg
    if mark[0] == "fixed":
        colvec = g.f([1, CHANNELS, 1, 1], _onehot(mark[1]))
    else:
        colvec = _content_onehot(g, S)
    vec0 = g.f([1, CHANNELS, 1, 1], _onehot(0))
    vecS = g.f([1, CHANNELS, 1, 1], _onehot(S))
    out = g.node("Add", [g.node("Mul", [fill, colvec]), g.node("Mul", [unfilled, vec0])])
    out = g.node("Add", [out, g.node("Mul", [smask, vecS])])
    _pad_out(g, out, h, w)
    return _model(g)


# --------------------------------------------------------------------------- #
# numpy references (mirror ONNX semantics for detection)                      #
# --------------------------------------------------------------------------- #
def _bands(seps, n):
    b = []; cur = []; ss = set(seps)
    for i in range(n):
        if i in ss:
            if cur:
                b.append((cur[0], cur[-1] + 1)); cur = []
        else:
            cur.append(i)
    if cur:
        b.append((cur[0], cur[-1] + 1))
    return b


def _structures(a0, prs):
    """All regular separator-grid structures consistent across every input."""
    h, w = a0.shape
    out = []
    for S in range(CHANNELS):
        rsep = [i for i in range(h) if (a0[i, :] == S).all()]
        csep = [j for j in range(w) if (a0[:, j] == S).all()]
        if not rsep and not csep:
            continue
        rb = _bands(rsep, h); cb = _bands(csep, w)
        if len(rb) == 0 or len(cb) == 0:
            continue
        if len({e - s for s, e in rb}) != 1 or len({e - s for s, e in cb}) != 1:
            continue
        if rb[0][0] != 0 or cb[0][0] != 0:
            continue
        npr, npc = len(rb), len(cb)
        if npr * npc < 2:
            continue
        ch = rb[0][1] - rb[0][0]; cw = cb[0][1] - cb[0][0]
        rstarts = [s for s, _ in rb]; cstarts = [s for s, _ in cb]
        per_r = (rstarts[1] - rstarts[0]) if npr > 1 else ch
        per_c = (cstarts[1] - cstarts[0]) if npc > 1 else cw
        if npr > 1 and any(rstarts[i + 1] - rstarts[i] != per_r for i in range(npr - 1)):
            continue
        if npc > 1 and any(cstarts[i + 1] - cstarts[i] != per_c for i in range(npc - 1)):
            continue
        # geometry must be identical across all inputs
        good = True
        for a, _ in prs:
            if a.shape != (h, w):
                good = False; break
            rs = [i for i in range(h) if (a[i, :] == S).all()]
            cs = [j for j in range(w) if (a[:, j] == S).all()]
            if rs != rsep or cs != csep:
                good = False; break
        if not good:
            continue
        rects = [(rstarts[i], rstarts[i] + ch, cstarts[j], cstarts[j] + cw)
                 for i in range(npr) for j in range(npc)]
        out.append(dict(S=S, ch=ch, cw=cw, npr=npr, npc=npc, per_r=per_r, per_c=per_c,
                        rstarts=rstarts, cstarts=cstarts, rects=rects, H=h, W=w))
    return out


def _panels(a, st):
    return [a[r0:r1, c0:c1] for (r0, r1, c0, c1) in st["rects"]]


def _overlay_np(plist):                       # plist high -> low priority
    res = np.zeros_like(plist[0])
    for p in reversed(plist):
        res = np.where(p != 0, p, res)
    return res


def _flip_np(p, axis):
    return p[:, ::-1] if axis == 3 else p[::-1, :]


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def _pairs(ex):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int); b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            if max(a.shape) > 30 or max(b.shape) > 30:
                continue
            out.append((a, b))
    return out


def _emit(out, seen, name, builder):
    if name in seen:
        return
    try:
        m = builder()
        onnx.checker.check_model(m, full_check=True)
    except Exception:
        return
    seen.add(name)
    out.append((name, m))


def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if len({a.shape for a, _ in prs}) != 1:        # constant geometry only
        return []
    if all(np.array_equal(a, b) for a, b in prs):  # identity -> not our family
        return []

    a0 = prs[0][0]
    structs = _structures(a0, prs)
    if not structs:
        return []
    osh = {b.shape for _, b in prs}
    if len(osh) != 1:
        return []
    oh_, ow = next(iter(osh))

    out, seen = [], set()

    for st in structs:
        ch, cw, npr, npc = st["ch"], st["cw"], st["npr"], st["npc"]
        n = npr * npc
        rects = st["rects"]

        # ---- panel-sized output: overlay & gate ---------------------------- #
        if (oh_, ow) == (ch, cw):
            order_fwd = list(range(n))
            for order, tag in ((order_fwd, "fwd"), (order_fwd[::-1], "rev")):
                if all(np.array_equal(_overlay_np([_panels(a, st)[k] for k in order]), b)
                       for a, b in prs):
                    _emit(out, seen, f"overlay_S{st['S']}_{npr}x{npc}_{tag}",
                          lambda st=st, order=order: build_overlay(st["rects"], order, ch, cw, None))
            # 2-panel fold (mirror the second panel across the split axis)
            if n == 2:
                ax = 3 if npc == 2 else 2
                flips = {1: ax}
                if all(np.array_equal(
                        _overlay_np([_panels(a, st)[0], _flip_np(_panels(a, st)[1], ax)]), b)
                        for a, b in prs):
                    _emit(out, seen, f"overlay_S{st['S']}_fold",
                          lambda st=st, flips=flips: build_overlay(st["rects"], [0, 1], ch, cw, flips))

            # gate: single fixed colour
            Cs = set()
            for _, b in prs:
                Cs |= set(np.unique(b[b != 0]).tolist())
            if len(Cs) == 1:
                C = next(iter(Cs))
                for op in ("ge", "le", "eq"):
                    for k in range(0, n + 1):
                        ok = True
                        for a, b in prs:
                            cnt = sum((p != 0).astype(int) for p in _panels(a, st))
                            if op == "ge":
                                m = cnt >= k
                            elif op == "le":
                                m = cnt <= k
                            else:
                                m = cnt == k
                            if not np.array_equal(np.where(m, C, 0), b):
                                ok = False; break
                        if ok:
                            _emit(out, seen, f"gate_S{st['S']}_{op}{k}_C{C}",
                                  lambda st=st, op=op, k=k, C=C: build_gate(st["rects"], op, k, C, ch, cw))

        # ---- reduced output: one cell per panel ---------------------------- #
        if (oh_, ow) == (npr, npc):
            Ms = set()
            for _, b in prs:
                Ms |= set(np.unique(b[b != 0]).tolist())
            if len(Ms) == 1:
                M = next(iter(Ms))
                counts_list = [np.array([(p != 0).sum() for p in _panels(a, st)]).reshape(npr, npc)
                               for a, _ in prs]
                preds = [("max",)]
                kmax = int(max(c.max() for c in counts_list))
                for k in range(0, kmax + 1):
                    preds += [("ge", k), ("le", k), ("eq", k)]
                for pred in preds:
                    ok = True
                    for (a, b), cnt in zip(prs, counts_list):
                        if pred[0] == "max":
                            sel = cnt == cnt.max()
                        elif pred[0] == "ge":
                            sel = cnt >= pred[1]
                        elif pred[0] == "le":
                            sel = cnt <= pred[1]
                        else:
                            sel = cnt == pred[1]
                        if not np.array_equal(np.where(sel, M, 0), b):
                            ok = False; break
                    if ok:
                        _emit(out, seen, f"selred_S{st['S']}_{pred[0]}{pred[1] if len(pred)>1 else ''}_M{M}",
                              lambda st=st, pred=pred, M=M: build_selred(st, pred, M))
                        break

        # ---- broadcast output: original grid, panels filled ---------------- #
        if (oh_, ow) == (st["H"], st["W"]):
            S = st["S"]
            # fixed panel/separator masks (constant geometry)
            pmask = np.zeros((st["H"], st["W"]), np.float32)
            for (r0, r1, c0, c1) in rects:
                pmask[r0:r1, c0:c1] = 1.0
            a_any = a0
            smask = np.zeros((st["H"], st["W"]), np.float32)
            for i in range(st["H"]):
                if (a_any[i, :] == S).all():
                    smask[i, :] = 1.0
            for j in range(st["W"]):
                if (a_any[:, j] == S).all():
                    smask[:, j] = 1.0
            counts_list = [np.array([(p != 0).sum() for p in _panels(a, st)]).reshape(npr, npc)
                           for a, _ in prs]
            kmax = int(max(c.max() for c in counts_list))
            preds = [("max",)] + [(op, k) for k in range(0, kmax + 1) for op in ("ge", "le", "eq")]
            # determine mark mode from targets
            for pred in preds:
                # numpy reference for both mark modes
                def ref(a, b, cnt, mode):
                    if pred[0] == "max":
                        sel = cnt == cnt.max()
                    elif pred[0] == "ge":
                        sel = cnt >= pred[1]
                    elif pred[0] == "le":
                        sel = cnt <= pred[1]
                    else:
                        sel = cnt == pred[1]
                    exp = np.zeros_like(a)
                    if S != 0:
                        exp[smask.astype(bool)] = S
                    if mode[0] == "content":
                        nz = a[(a != 0) & (a != S)]
                        col = int(np.bincount(nz).argmax()) if nz.size else 0
                    else:
                        col = mode[1]
                    for idx, (r0, r1, c0, c1) in enumerate(rects):
                        if sel[idx // npc, idx % npc]:
                            exp[r0:r1, c0:c1] = col
                    return np.array_equal(exp, b)

                # candidate mark modes
                modes = [("content",)]
                fixedset = set()
                for _, b in prs:
                    fixedset |= set(np.unique(b[(b != 0) & (b != S)]).tolist())
                if len(fixedset) == 1:
                    modes.append(("fixed", next(iter(fixedset))))
                for mode in modes:
                    if all(ref(a, b, cnt, mode) for (a, b), cnt in zip(prs, counts_list)):
                        tagm = "content" if mode[0] == "content" else f"M{mode[1]}"
                        _emit(out, seen,
                              f"selbroad_S{S}_{pred[0]}{pred[1] if len(pred)>1 else ''}_{tagm}",
                              lambda st=st, pred=pred, mode=mode, S=S,
                              pmask=pmask, smask=smask:
                              build_selbroad(st, pred, mode, S, pmask.ravel().tolist(),
                                             smask.ravel().tolist()))
                        break
                else:
                    continue
                break

    return out
