"""family_crk5_1 — symmetry-repair / occlusion-fill solvers.

Targets ARC tasks where the output equals the input except that an
"occlusion" color (a single fixed color, e.g. 9) is removed and the
covered cells are reconstructed from the grid's symmetry group.

Rule (verified on task 074):  the grid is globally symmetric under
  - transpose about the main diagonal (perm [0,1,3,2]),
  - a vertical mirror  c -> 2*ax-1-c,
  - a horizontal mirror r -> 2*ax-1-r,
and the occluded cells (color = occ) are filled by, in sequence,
copying from the (transpose / vmirror / hmirror) partner whenever that
partner is a *known* (non-occluded, in-range) cell.  One sweep suffices.

Everything is done in one-hot FLOAT[1,10,30,30] space:
  occ-mask  = channel[occ]
  known(fK) = sum(fK[:,0:9]) (==1 iff partner carries a real color)
  m         = occ * known
  K'        = m*fK + (1-m)*K          (copies the full 10-vector of fK)
After all sweeps the occlusion channel is 0 everywhere, so the raw K is
emitted directly as the one-hot output.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper as oh

from ng_utils_shim import (
    DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, CHANNELS, HEIGHT, WIDTH,
)

INT64 = onnx.TensorProto.INT64
_NEG = -(1 << 31)


def _model(nodes, initializers):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = oh.make_graph(nodes, "g", [x], [y], list(initializers))
    return oh.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


# ----------------------------------------------------------------------------
# numpy reference that is *bit-faithful* to the ONNX ops we emit
# ----------------------------------------------------------------------------
def _np_transpose(g):
    return g.T


def _np_reverse_shift(g, axis, shift):
    """Reverse `g` fully along axis (0=rows,1=cols) then translate by `shift`
    with zero fill, kept anchored to a HEIGHT/WIDTH window.  Mirrors the
    Slice(step=-1) + Pad + Slice ONNX construction exactly."""
    rev = np.flip(g, axis=axis)
    out = np.zeros_like(g)
    n = g.shape[axis]
    for i in range(n):
        si = i - shift
        if 0 <= si < n:
            if axis == 0:
                out[i, :] = rev[si, :]
            else:
                out[:, i] = rev[:, si]
    return out


def _fill(state, fval, occ):
    """state,fval are int grids; occ is the occlusion color. Fill cells of
    `state`==occ from `fval` where fval is a known color (not occ, not the
    0-padding sentinel -1)."""
    mask = (state == occ) & (fval != occ) & (fval >= 0)
    out = state.copy()
    out[mask] = fval[mask]
    return out


def _apply_chain(grid, occ, transforms):
    """grid: int array (variable size, top-left). transforms: list of specs."""
    g = grid.astype(int).copy()
    for spec in transforms:
        if spec[0] == "T":
            f = g.T.copy()
        elif spec[0] == "V":
            f = _np_reverse_shift(g, 1, spec[1])
        elif spec[0] == "H":
            f = _np_reverse_shift(g, 0, spec[1])
        # mark out-of-range (introduced zeros) as -1 only where genuinely empty?
        # _np_reverse_shift fills with 0; a real color 0 is background. We must
        # distinguish "source was empty" from "source is colour 0". Handle via a
        # validity grid computed the same way on an all-ones map.
        valid = _valid_map(grid.shape, spec)
        f = np.where(valid, f, -1)
        g = _fill(g, f, occ)
    return g


def _valid_map(shape, spec):
    ones = np.ones(shape, int)
    if spec[0] == "T":
        return ones.T.copy().astype(bool)
    elif spec[0] == "V":
        return _np_reverse_shift(ones, 1, spec[1]).astype(bool)
    else:
        return _np_reverse_shift(ones, 0, spec[1]).astype(bool)


# ----------------------------------------------------------------------------
# ONNX builders
# ----------------------------------------------------------------------------
def _slice(data, out, starts, ends, axes, steps, inits, tag):
    s = oh.make_tensor(f"{tag}_s", INT64, [len(starts)], starts)
    e = oh.make_tensor(f"{tag}_e", INT64, [len(ends)], ends)
    a = oh.make_tensor(f"{tag}_a", INT64, [len(axes)], axes)
    ins = [data, f"{tag}_s", f"{tag}_e", f"{tag}_a"]
    new = [s, e, a]
    if steps is not None:
        st = oh.make_tensor(f"{tag}_t", INT64, [len(steps)], steps)
        ins.append(f"{tag}_t")
        new.append(st)
    inits.extend(new)
    return oh.make_node("Slice", ins, [out])


def _transform_nodes(src, dst, spec, inits, tag):
    """Emit nodes producing dst = transform(src) for a [1,10,30,30] tensor."""
    nodes = []
    if spec[0] == "T":
        nodes.append(oh.make_node("Transpose", [src], [dst], perm=[0, 1, 3, 2]))
    else:
        axis = 3 if spec[0] == "V" else 2
        shift = spec[1]
        # full reverse along axis
        rev = f"{tag}_rev"
        start = (HEIGHT - 1) if axis == 2 else (WIDTH - 1)
        nodes.append(_slice(src, rev, [start], [_NEG], [axis], [-1], inits, f"{tag}_rv"))
        # translate by `shift` along axis with zero fill, window back to size
        if axis == 2:
            h0, h1 = max(shift, 0), max(-shift, 0)
            pads = [0, 0, h0, 0, 0, 0, h1, 0]
            cs, ce, ca = [h1], [h1 + HEIGHT], [2]
        else:
            w0, w1 = max(shift, 0), max(-shift, 0)
            pads = [0, 0, 0, w0, 0, 0, 0, w1]
            cs, ce, ca = [w1], [w1 + WIDTH], [3]
        pad = f"{tag}_pad"
        nodes.append(oh.make_node("Pad", [rev], [pad], mode="constant", value=0.0, pads=pads))
        nodes.append(_slice(pad, dst, cs, ce, ca, None, inits, f"{tag}_cp"))
    return nodes


def _translate(src, dst, dy, dx, inits, tag):
    """out[r,c] = src[r-dy, c-dx], zero fill, anchored to HEIGHT x WIDTH.
    Pad (attr) + Slice back to window. Returns list of nodes."""
    h0, w0 = max(dy, 0), max(dx, 0)
    h1, w1 = max(-dy, 0), max(-dx, 0)
    pad = f"{tag}_pd"
    nodes = [oh.make_node("Pad", [src], [pad], mode="constant", value=0.0,
                          pads=[0, 0, h0, w0, 0, 0, h1, w1])]
    nodes.append(_slice(pad, dst, [h1, w1], [h1 + HEIGHT, w1 + WIDTH], [2, 3], None,
                        inits, f"{tag}_cr"))
    return nodes


def build_linecomplete(sp):
    """Fill a cell currently colored 1 with marker colour `m` whenever the two
    cells `sp` away on opposite sides (horizontally or vertically) both carry
    the same marker colour (>1). Single pass. Fixed spacing `sp`."""
    inits = [oh.make_tensor("one", DATA_TYPE, [1], [1.0])]
    nodes = []
    # is1 = X channel 1
    nodes.append(_slice("input", "is1", [1], [2], [1], None, inits, "is1"))

    def end_terms(dst_tag, dy, dx):
        # E1 = X shifted so E1[r,c]=X[r-dy,c-dx]; E2 opposite
        e1 = f"{dst_tag}_e1"
        e2 = f"{dst_tag}_e2"
        nodes.extend(_translate("input", e1, dy, dx, inits, f"{dst_tag}_t1"))
        nodes.extend(_translate("input", e2, -dy, -dx, inits, f"{dst_tag}_t2"))
        # marker(e1)= sum(e1[:,2:10])
        m1s = f"{dst_tag}_m1s"
        nodes.append(_slice(e1, m1s, [2], [CHANNELS], [1], None, inits, f"{dst_tag}_ms"))
        m1 = f"{dst_tag}_m1"
        nodes.append(oh.make_node("ReduceSum", [m1s], [m1], axes=[1], keepdims=1))
        # same = sum(e1*e2)
        prod = f"{dst_tag}_pr"
        nodes.append(oh.make_node("Mul", [e1, e2], [prod]))
        same = f"{dst_tag}_sm"
        nodes.append(oh.make_node("ReduceSum", [prod], [same], axes=[1], keepdims=1))
        cond = f"{dst_tag}_cond"
        nodes.append(oh.make_node("Mul", [m1, same], [f"{dst_tag}_c0"]))
        nodes.append(oh.make_node("Mul", [f"{dst_tag}_c0", "is1"], [cond]))
        return cond, e1

    # horizontal ends are sp columns to each side: E1[r,c]=X[r,c-sp]
    condh, hL = end_terms("h", 0, sp)
    # Y1 = condh*hL + (1-condh)*X
    nodes.append(oh.make_node("Mul", [condh, hL], ["h_fill"]))
    nodes.append(oh.make_node("Sub", ["one", condh], ["h_om"]))
    nodes.append(oh.make_node("Mul", ["h_om", "input"], ["h_keep"]))
    nodes.append(oh.make_node("Add", ["h_fill", "h_keep"], ["Y1"]))

    condv, vU = end_terms("v", sp, 0)
    nodes.append(oh.make_node("Mul", [condv, vU], ["v_fill"]))
    nodes.append(oh.make_node("Sub", ["one", condv], ["v_om"]))
    nodes.append(oh.make_node("Mul", ["v_om", "Y1"], ["v_keep"]))
    nodes.append(oh.make_node("Add", ["v_fill", "v_keep"], ["output"]))
    return _model(nodes, inits)


def _np_linecomplete(g, sp):
    def sh(a, dy, dx):
        o = np.zeros_like(a)
        H, W = a.shape
        for r in range(H):
            for c in range(W):
                sr, sc = r - dy, c - dx
                if 0 <= sr < H and 0 <= sc < W:
                    o[r, c] = a[sr, sc]
        return o
    L = sh(g, 0, sp); R = sh(g, 0, -sp)
    U = sh(g, sp, 0); D = sh(g, -sp, 0)
    condh = (L > 1) & (R > 1) & (L == R)
    condv = (U > 1) & (D > 1) & (U == D)
    out = g.copy()
    out = np.where(condh & (g == 1), L, out)
    out = np.where(condv & (g == 1), U, out)
    return out


def build(occ, transforms):
    inits = []
    nodes = []
    one = oh.make_tensor("one", DATA_TYPE, [1], [1.0])
    inits.append(one)
    cur = "input"
    for i, spec in enumerate(transforms):
        tag = f"s{i}"
        fK = f"{tag}_fK"
        nodes += _transform_nodes(cur, fK, spec, inits, tag)
        # known = sum(fK[:,0:9]) keepdims
        kslice = f"{tag}_ksl"
        nodes.append(_slice(fK, kslice, [0], [CHANNELS - 1], [1], None, inits, f"{tag}_ks"))
        known = f"{tag}_known"
        nodes.append(oh.make_node("ReduceSum", [kslice], [known], axes=[1], keepdims=1))
        # occ mask = cur[:,occ:occ+1]
        occm = f"{tag}_occ"
        nodes.append(_slice(cur, occm, [occ], [occ + 1], [1], None, inits, f"{tag}_oc"))
        m = f"{tag}_m"
        nodes.append(oh.make_node("Mul", [occm, known], [m]))
        mf = f"{tag}_mf"
        nodes.append(oh.make_node("Mul", [m, fK], [mf]))
        onem = f"{tag}_onem"
        nodes.append(oh.make_node("Sub", ["one", m], [onem]))
        keep = f"{tag}_keep"
        nodes.append(oh.make_node("Mul", [onem, cur], [keep]))
        nxt = "output" if i == len(transforms) - 1 else f"{tag}_K"
        nodes.append(oh.make_node("Add", [mf, keep], [nxt]))
        cur = nxt
    return _model(nodes, inits)


# ----------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------
def _all_pairs(examples):
    out = []
    for s in ("train", "test", "arc-gen"):
        for e in examples.get(s, []):
            out.append((np.array(e["input"], int), np.array(e["output"], int)))
    return out


def _detect_occ(pairs):
    cols = set()
    outcols = set()
    for a, b in pairs:
        if a.shape != b.shape:
            return None
        d = a != b
        if not d.any():
            continue
        cols |= set(a[d].tolist())
        outcols |= set(b.ravel().tolist())
    if len(cols) != 1:
        return None
    occ = cols.pop()
    if occ in outcols:
        return None
    return occ


def _vaxis(b):
    H, W = b.shape
    res = []
    for ax in range(1, W):
        ok = True
        for c in range(W):
            cc = 2 * ax - 1 - c
            if 0 <= cc < W and not np.array_equal(b[:, c], b[:, cc]):
                ok = False
                break
        if ok:
            res.append(ax)
    return res


def _haxis(b):
    H, W = b.shape
    res = []
    for ax in range(1, H):
        ok = True
        for r in range(H):
            rr = 2 * ax - 1 - r
            if 0 <= rr < H and not np.array_equal(b[r], b[rr]):
                ok = False
                break
        if ok:
            res.append(ax)
    return res


def _try_linecomplete(pairs):
    if not pairs:
        return []
    # all same shape required
    if any(a.shape != b.shape for a, b in pairs):
        return []
    # must actually change something somewhere (avoid trivial identity claim)
    if not any((a != b).any() for a, b in pairs):
        return []
    for sp in (1, 2, 3, 4, 5, 6):
        if all(np.array_equal(_np_linecomplete(a, sp), b) for a, b in pairs):
            return [(f"linecomplete_{sp}", build_linecomplete(sp))]
    return []


def candidates(examples):
    pairs = _all_pairs(examples)
    if not pairs:
        return []
    out = []
    out += _try_linecomplete(pairs)
    out += _try_symrepair(pairs)
    return out


def _try_symrepair(pairs):
    # require square full-size grids (so transpose is a valid global symmetry
    # and reflections have a fixed, shape-independent axis)
    shapes = set(a.shape for a, _ in pairs)
    if shapes != {(HEIGHT, WIDTH)}:
        return []
    occ = _detect_occ(pairs)
    if occ is None:
        return []

    # gather symmetry axes that hold for every output
    vsets = [set(_vaxis(b)) for _, b in pairs]
    hsets = [set(_haxis(b)) for _, b in pairs]
    Tsym = all(np.array_equal(b, b.T) for _, b in pairs)
    vcommon = set.intersection(*vsets) if vsets else set()
    hcommon = set.intersection(*hsets) if hsets else set()

    # candidate transform pools
    specs = []
    if Tsym:
        specs.append(("T",))
    for ax in sorted(vcommon):
        specs.append(("V", 2 * ax - WIDTH))
    for ax in sorted(hcommon):
        specs.append(("H", 2 * ax - HEIGHT))

    if not specs:
        return []

    # try short ordered chains drawn from the pool; verify EXACT on all pairs
    import itertools
    pool = specs
    best = None
    for r in (1, 2, 3, 4):
        for chain in itertools.permutations(pool, min(r, len(pool))):
            ok = True
            for a, b in pairs:
                g = _apply_chain(a, occ, list(chain))
                if not np.array_equal(g, b):
                    ok = False
                    break
            if ok:
                best = list(chain)
                break
        if best is not None:
            break

    if best is None:
        return []
    model = build(occ, best)
    name = "symrepair_" + "_".join(s[0] + (str(s[1]) if len(s) > 1 else "") for s in best)
    return [(name, model)]
