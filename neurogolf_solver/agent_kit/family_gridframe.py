"""Grid-separator & frame family (origin-anchored, fixed structure).

This family implements three closely-related, *origin-safe* mechanisms.  Every
construction keeps grid content anchored at the top-left (0,0) and depends only on
size-independent structure (absolute / periodic positions, pointwise channel maps),
so it survives the top-left zero-padding gotcha and generalises across the variable
grid sizes in arc-gen / the held-out set.

(a) MARGIN add / remove  (origin-safe only on the BOTTOM / RIGHT edge)
    * add  : output = input with a constant-colour full bottom/right margin.
             - constant in/out sizes  -> fill a *static* L-shaped region (Where).
             - variable in / fixed out -> fill every still-empty cell of a fixed
               output box with colour K (Where over a content/absent mask).
    * remove: the trailing margin is dropped.
             - if the margin colour K only ever occurs in the trailing strip, the
               whole operation is just "delete channel K" (a per-channel Mul), which
               is pointwise and therefore valid for *any* size.
             - otherwise (constant out size) a top-left Slice + Pad.

(b) SEPARATOR-GRID recolour
    * plain global per-colour recolour of the cells (a 1x1 Conv, or a cheaper Gather
      when the colour map is a permutation).  Separator lines whose colour maps to
      itself are preserved for free.
    * when the separator colour must be *kept* even though the map would change it,
      the recolour is restricted to the non-separator cells with a STATIC mask
      (periodic in absolute coordinates -> size independent): recolored = Conv1x1,
      output = Where(non_sep_mask, recolored, input).

(c) EXTRACT content ignoring a bottom/right-anchored separator
    * delete the separator colour (per-channel Mul) when it is exclusive to the
      trailing strip, or a fixed top-left Slice + Pad when the content size is
      constant.

Detection fits every parameter from the train+test+arc-gen pairs (raw numpy) and
keeps a hypothesis only if it reproduces *every* pair exactly (the grader's gate),
so wrong guesses are dropped and nothing here memorises individual grids.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from builders import _model, recolor_gather, recolor_conv
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, HEIGHT, WIDTH, CHANNELS

INT64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


# --------------------------------------------------------------------------- #
# pair extraction                                                             #
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


def _onehot(g):
    """grid -> [10,30,30] bool one-hot, top-left anchored (padding == all-zero)."""
    o = np.zeros((CHANNELS, HEIGHT, WIDTH), bool)
    h, w = g.shape
    for c in range(CHANNELS):
        o[c, :h, :w] = (g == c)
    return o


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

    def fconst(self, dims, vals):
        nm = self.name("c")
        self.inits.append(oh.make_tensor(nm, DATA_TYPE, list(dims),
                                         [float(v) for v in vals]))
        return nm

    def bconst(self, dims, vals):
        nm = self.name("m")
        self.inits.append(oh.make_tensor(nm, BOOL, list(dims),
                                         [int(bool(v)) for v in vals]))
        return nm

    def node(self, op, ins, out=None, **attrs):
        out = out or self.name()
        self.nodes.append(oh.make_node(op, list(ins), [out], **attrs))
        return out


def _mask_dims(M):
    """Smallest broadcast representation of a 30x30 bool mask."""
    if (M == M[:, :1]).all():               # row-only
        return [1, 1, HEIGHT, 1], M[:, 0].astype(int).tolist()
    if (M == M[:1, :]).all():               # col-only
        return [1, 1, 1, WIDTH], M[0, :].astype(int).tolist()
    return [1, 1, HEIGHT, WIDTH], M.astype(int).ravel().tolist()


def _chan_onehot(K):
    v = [1.0 if c == K else 0.0 for c in range(CHANNELS)]
    return v


# --------------------------------------------------------------------------- #
# model builders                                                               #
# --------------------------------------------------------------------------- #
def _build_delete_color(K):
    """output = input * mask, mask[K]=0 else 1  (zero out one channel -> remove a colour)."""
    g = _G()
    m = g.fconst([1, CHANNELS, 1, 1], [0.0 if c == K else 1.0 for c in range(CHANNELS)])
    g.node("Mul", ["input", m], "output")
    return _model(g.nodes, g.inits)


def _build_fill_static(M, K):
    """output = Where(M, onehot_K, input).  M is a static bool region; input is all-zero
    (padding) at the fill cells, so this writes colour K there and leaves the rest."""
    g = _G()
    dims, vals = _mask_dims(M)
    mk = g.bconst(dims, vals)
    ok = g.fconst([1, CHANNELS, 1, 1], _chan_onehot(K))
    g.node("Where", [mk, ok, "input"], "output")
    return _model(g.nodes, g.inits)


def _build_fill_box(M, K):
    """Fill every still-empty cell inside static box M with colour K (variable input,
    fixed output box).  present = sum over channels; absent = present < 0.5."""
    g = _G()
    present = g.node("ReduceSum", ["input"], axes=[1], keepdims=1)   # [1,1,30,30]
    half = g.fconst([1, 1, 1, 1], [0.5])
    absent = g.node("Less", [present, half])                        # bool [1,1,30,30]
    dims, vals = _mask_dims(M)
    boxc = g.bconst(dims, vals)
    cond = g.node("And", [boxc, absent])
    ok = g.fconst([1, CHANNELS, 1, 1], _chan_onehot(K))
    g.node("Where", [cond, ok, "input"], "output")
    return _model(g.nodes, g.inits)


def _build_masked_recolor(M, cmap):
    """output = Where(M, Conv1x1(input), input)  (recolour only inside static mask M)."""
    g = _G()
    W = np.zeros((CHANNELS, CHANNELS, 1, 1), np.float32)
    for i, o in enumerate(cmap):
        W[o, i, 0, 0] = 1.0
    wt = g.fconst([CHANNELS, CHANNELS, 1, 1], W.ravel().tolist())
    rec = g.node("Conv", ["input", wt], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    dims, vals = _mask_dims(M)
    mk = g.bconst(dims, vals)
    g.node("Where", [mk, rec, "input"], "output")
    return _model(g.nodes, g.inits)


def _build_crop(R, C):
    """output = Pad(input[:, :, 0:R, 0:C]) back to 30x30 (top-left contiguous crop)."""
    g = _G()
    s = g.name("s"); e = g.name("e"); ax = g.name("a")
    g.inits.append(oh.make_tensor(s, INT64, [2], [0, 0]))
    g.inits.append(oh.make_tensor(e, INT64, [2], [R, C]))
    g.inits.append(oh.make_tensor(ax, INT64, [2], [2, 3]))
    g.node("Slice", ["input", s, e, ax], "small")
    g.node("Pad", ["small"], "output", mode="constant", value=0.0,
           pads=[0, 0, 0, 0, 0, 0, HEIGHT - R, WIDTH - C])
    return _model(g.nodes, g.inits)


# --------------------------------------------------------------------------- #
# detection helpers                                                            #
# --------------------------------------------------------------------------- #
def _global_map(prs):
    """Consistent per-colour map over ALL cells, or None.  Returns length-10 list
    (identity where unconstrained); None if not a single global recolour, or identity."""
    if not all(a.shape == b.shape for a, b in prs):
        return None
    cmap = {}
    for a, b in prs:
        for ic, oc in zip(a.ravel().tolist(), b.ravel().tolist()):
            if ic in cmap and cmap[ic] != oc:
                return None
            cmap[ic] = oc
    full = [cmap.get(i, i) for i in range(CHANNELS)]
    if all(full[i] == i for i in range(CHANNELS)):
        return None                              # identity -> not our job
    return full


def _periodic_lines(idx_sets, lengths):
    """Given, per pair, the sorted set of separator indices and the axis length,
    find (period, phase) with 2<=period, 0<=phase<period such that for every pair
    idx_set == {k in range(L): k % period == phase}.  Return (period, phase) or None.
    Empty everywhere -> return ("none",)."""
    if all(len(s) == 0 for s in idx_sets):
        return ("none",)
    for period in range(2, 16):
        for phase in range(period):
            ok = True
            for s, L in zip(idx_sets, lengths):
                want = set(k for k in range(L) if k % period == phase)
                if set(s) != want:
                    ok = False
                    break
            if ok:
                return (period, phase)
    return None


def _sep_mask(prs):
    """Detect a separator colour S forming periodic full rows and/or cols.  Return a
    30x30 bool NON-separator mask (True where cells may be recoloured) or None."""
    cand = set(range(CHANNELS))
    for a, _ in prs:
        cand &= set(np.unique(a).tolist())
    for S in sorted(cand):
        rsets, csets, Hs, Ws = [], [], [], []
        good = True
        for a, _ in prs:
            H, W = a.shape
            rows = [i for i in range(H) if np.all(a[i, :] == S)]
            cols = [j for j in range(W) if np.all(a[:, j] == S)]
            if len(rows) == H or len(cols) == W:     # degenerate (all one colour)
                good = False
                break
            rsets.append(rows); csets.append(cols); Hs.append(H); Ws.append(W)
        if not good:
            continue
        pr = _periodic_lines(rsets, Hs)
        pc = _periodic_lines(csets, Ws)
        if pr is None or pc is None:
            continue
        if pr == ("none",) and pc == ("none",):
            continue                                  # no separators at all
        rowmask = np.zeros(HEIGHT, bool)
        if pr != ("none",):
            p, ph = pr
            rowmask[[i for i in range(HEIGHT) if i % p == ph]] = True
        colmask = np.zeros(WIDTH, bool)
        if pc != ("none",):
            p, ph = pc
            colmask[[j for j in range(WIDTH) if j % p == ph]] = True
        sep = rowmask[:, None] | colmask[None, :]
        nonsep = ~sep
        if not nonsep.any():
            continue
        return nonsep
    return None


def _verify_onehot(model_pred_fn, prs):
    for a, b in prs:
        if not np.array_equal(model_pred_fn(a), _onehot(b)):
            return False
    return True


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    out = []
    same_shape = all(a.shape == b.shape for a, b in prs)
    ishapes = set(a.shape for a, b in prs)
    oshapes = set(b.shape for a, b in prs)

    # ------------------------------------------------------------------ #
    # (b) recolour cells by an inferred map                              #
    # ------------------------------------------------------------------ #
    if same_shape:
        full = _global_map(prs)
        if full is not None:
            # plain global recolour: permutation -> cheap Gather, else 1x1 Conv
            if len(set(full)) == CHANNELS:           # bijection
                src = [0] * CHANNELS
                for i, o in enumerate(full):
                    src[o] = i                       # output channel o <- input channel i
                try:
                    out.append(("gridframe_recolor_gather", recolor_gather(src)))
                except Exception:
                    pass
            try:
                out.append(("gridframe_recolor_conv", recolor_conv(full)))
            except Exception:
                pass
        else:
            # separator-conditioned recolour: keep the separator, recolour the rest
            nonsep = _sep_mask(prs)
            if nonsep is not None:
                cmap = {}
                ok = True
                for a, b in prs:
                    h, w = a.shape
                    M = nonsep[:h, :w]
                    d = (a != b)
                    if (d & ~M).any():               # change on a separator -> not this rule
                        ok = False
                        break
                    for ic, oc in zip(a[d].tolist(), b[d].tolist()):
                        if ic in cmap and cmap[ic] != oc:
                            ok = False
                            break
                        cmap[ic] = oc
                    if not ok:
                        break
                if ok and cmap:
                    full2 = [cmap.get(i, i) for i in range(CHANNELS)]

                    def pred(a, _f=full2, _ns=nonsep):
                        h, w = a.shape
                        rec = np.array([_f[c] for c in a.ravel()]).reshape(a.shape)
                        res = np.where(_ns[:h, :w], rec, a)
                        return _onehot(res)
                    if _verify_onehot(pred, prs):
                        try:
                            out.append(("gridframe_sep_recolor",
                                        _build_masked_recolor(nonsep, full2)))
                        except Exception:
                            pass

    # ------------------------------------------------------------------ #
    # (a) ADD a constant-colour bottom/right margin                      #
    # ------------------------------------------------------------------ #
    if all(b.shape[0] >= a.shape[0] and b.shape[1] >= a.shape[1] and
           (b.shape != a.shape) for a, b in prs):
        # content preserved top-left and the extra L-region is one constant colour K
        Ks = set()
        ok = True
        for a, b in prs:
            ha, wa = a.shape
            if not np.array_equal(b[:ha, :wa], a):
                ok = False
                break
            extra = b.copy().astype(int)
            keep = np.zeros(b.shape, bool); keep[:ha, :wa] = True
            vals = extra[~keep]
            u = np.unique(vals)
            if len(u) != 1:
                ok = False
                break
            Ks.add(int(u[0]))
        if ok and len(Ks) == 1:
            K = Ks.pop()
            # constant in & out sizes -> static L-shaped margin region
            if len(ishapes) == 1 and len(oshapes) == 1:
                (ha, wa) = next(iter(ishapes)); (hb, wb) = next(iter(oshapes))
                M = np.zeros((HEIGHT, WIDTH), bool)
                M[:hb, :wb] = True
                M[:ha, :wa] = False
                if M.any():
                    def pred(a, _M=M, _K=K):
                        oo = _onehot(a)
                        oo[_K] |= _M
                        # ensure no double-set: clear other channels where filled
                        for c in range(CHANNELS):
                            if c != _K:
                                oo[c] &= ~_M
                        return oo
                    if _verify_onehot(pred, prs):
                        try:
                            out.append((f"gridframe_addmargin_K{K}", _build_fill_static(M, K)))
                        except Exception:
                            pass
            # variable input, single fixed output box -> fill empty cells of the box
            elif len(oshapes) == 1:
                (hb, wb) = next(iter(oshapes))
                M = np.zeros((HEIGHT, WIDTH), bool); M[:hb, :wb] = True

                def predb(a, _M=M, _K=K):
                    oo = _onehot(a)
                    present = oo.any(0)
                    fill = _M & (~present)
                    oo[_K] |= fill
                    return oo
                if _verify_onehot(predb, prs):
                    try:
                        out.append((f"gridframe_fillbox_K{K}", _build_fill_box(M, K)))
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    # (a)/(c) REMOVE a bottom/right margin / extract content             #
    # ------------------------------------------------------------------ #
    if any(b.shape != a.shape for a, b in prs) and \
       all(b.shape[0] <= a.shape[0] and b.shape[1] <= a.shape[1] for a, b in prs):
        # delete-colour: a single colour K, exclusive to the dropped trailing strip
        for K in range(CHANNELS):
            def predk(a, _K=K):
                oo = _onehot(a)
                oo[_K] = False
                return oo
            if any(b.shape != a.shape for a, b in prs) and _verify_onehot(predk, prs):
                out.append((f"gridframe_delcolor_K{K}", _build_delete_color(K)))
                break
        # fixed top-left crop (constant output size, input strictly larger sometimes)
        if len(oshapes) == 1:
            (R, C) = next(iter(oshapes))
            if 0 < R <= HEIGHT and 0 < C <= WIDTH and \
               all(a.shape[0] >= R and a.shape[1] >= C and np.array_equal(a[:R, :C], b)
                   for a, b in prs) and any(a.shape != (R, C) for a, b in prs):
                try:
                    out.append((f"gridframe_crop_{R}x{C}", _build_crop(R, C)))
                except Exception:
                    pass

    return out
