"""POSITION-CONDITIONED recolor with STATIC, origin-anchored masks (extends
family_maskmap).

family_maskmap recolours ONE fixed region with ONE global colour map and keeps
the identity everywhere else.  This family generalises that: the grid is
partitioned into several POSITION CLASSES by an absolute-position rule, and each
class carries its OWN per-colour recolour map.  So a single input colour can map
to different output colours depending only on WHERE the cell sits:

    output[i,j] = map_{class(i,j)}[ input[i,j] ]

The partition is structural (a property of the absolute, top-left-anchored
coordinates, independent of the grid size wherever possible), so the rule
generalises to grids of any size:

  per{pr}x{pc}   periodic tiling by (i % pr , j % pc)  -> alternating rows
                 (pc==1), alternating / every-k-th columns (pr==1), 2-D motifs.
  diagstripe{p}  diagonal stripes (i + j) % p   (p==2 is the checkerboard).
  cheb/manh/min  distance-from-the-top-left-corner BANDS by parity / mod-p of the
                 Chebyshev max(i,j), Manhattan i+j or min(i,j) distance
                 (concentric square rings, anti-diagonals, ...).
  diagsign       above / on / below the main diagonal (i<j / i==j / i>j).
  quad           the four quadrants (size dependent -> only when the input size is
                 constant across every split; the boundary is then fixed).

Realisation (opset 10, origin-safe, cheap)
------------------------------------------
Classes that share the same colour map are merged into one region; the identity
classes need no work at all (they fall through to the original input).  For each
distinct NON-identity map we emit one 1x1 ``Conv`` recolour gated by its static
region mask:

    cur = input
    for region (mask M, map W):
        cur = Where(M, Conv1x1(input, W), cur)
    output = cur

``W[o,i,0,0] = 1 iff map[i]==o`` (unseen colours default to identity, so they
pass through structurally).  ``M`` is a BOOL initializer collapsed to [1,1,30,1]
(row-only) or [1,1,1,30] (col-only) when possible.  A no-bias Conv maps the
all-zero padding to all-zero, and Where keeps that zero, so padding stays <=0 on
every channel for grids of any size.  With a single recoloured region the cost is
one [1,10,30,30] Conv intermediate (== family_maskmap).

Anti-overfit
------------
Detection fits the per-class maps from the pairs and only emits a partition whose
EVERY class is actually exercised by the data (no ungrounded residue / band /
quadrant is invented), that is genuinely position-DEPENDENT (not reducible to one
global recolour), and that reproduces EVERY train+test+arc-gen pair EXACTLY (the
grader's gate).  The cheapest valid partition (fewest regions, smallest masks) is
preferred.
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


# --------------------------------------------------------------------------- #
# graph helpers                                                               #
# --------------------------------------------------------------------------- #
def _model(nodes, inits):
    x = oh.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = oh.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    g = oh.make_graph(nodes, "g", [x], [y], list(inits))
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def _pairs(ex):
    """All usable (input, output) raw int grids; skip >30 grids (grader ignores)."""
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


def _mask_init(M):
    """Smallest broadcast representation of a 30x30 bool mask (BOOL initializer)."""
    M = np.asarray(M, bool)
    if (M == M[:, :1]).all():                 # depends on row only
        return [1, 1, HEIGHT, 1], M[:, 0].astype(int).tolist()
    if (M == M[:1, :]).all():                 # depends on column only
        return [1, 1, 1, WIDTH], M[0, :].astype(int).tolist()
    return [1, 1, HEIGHT, WIDTH], M.astype(int).ravel().tolist()


# --------------------------------------------------------------------------- #
# partition bank (functions of absolute row i / column j over 0..29)          #
# --------------------------------------------------------------------------- #
def _bank(constshape):
    """List of (complexity, name, cls) where cls is a 30x30 int class-id array."""
    H, W = HEIGHT, WIDTH
    I, J = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = []

    # periodic tiling (i % pr , j % pc) -> pr*pc classes
    for pr in range(1, 5):
        for pc in range(1, 7):
            if pr == 1 and pc == 1:
                continue
            cls = (I % pr) * pc + (J % pc)
            out.append((pr * pc, f"per{pr}x{pc}", cls))

    # diagonal stripes (i + j) % p  (p == 2 is the checkerboard)
    for p in (2, 3, 4):
        out.append((p, f"diagstripe{p}", (I + J) % p))

    # distance-from-origin bands (parity / mod-p of a corner distance metric)
    for nm, arr in (("cheb", np.maximum(I, J)), ("manh", I + J), ("min", np.minimum(I, J))):
        for p in (2, 3):
            out.append((p, f"{nm}{p}", arr % p))

    # above / on / below the main diagonal
    out.append((3, "diagsign", np.where(I < J, 0, np.where(I > J, 2, 1))))

    # the four quadrants (size dependent -> only with a constant input size)
    if constshape is not None:
        h, w = constshape
        for hd, wd, tag in ((h // 2, w // 2, "f"), ((h + 1) // 2, (w + 1) // 2, "c")):
            q = np.zeros((H, W), int)
            q[:hd, :wd] = 0
            q[:hd, wd:] = 1
            q[hd:, :wd] = 2
            q[hd:, wd:] = 3
            out.append((4, f"quad{tag}", q))

    return out


# --------------------------------------------------------------------------- #
# detection helpers                                                           #
# --------------------------------------------------------------------------- #
def _fit(prs, cls):
    """Per-class colour map {class_id -> {in_color: out_color}} or None on conflict."""
    maps = {}
    for a, b in prs:
        h, w = a.shape
        c = cls[:h, :w]
        for ci in np.unique(c):
            m = c == ci
            d = maps.setdefault(int(ci), {})
            for iv, ov in zip(a[m].tolist(), b[m].tolist()):
                if iv in d and d[iv] != ov:
                    return None
                d[iv] = ov
    return maps


def _vec(d):
    """Full length-10 colour map with identity default for unseen colours."""
    v = list(range(CHANNELS))
    for iv, ov in d.items():
        if 0 <= iv < CHANNELS and 0 <= ov < CHANNELS:
            v[iv] = ov
    return v


def _grounded(prs, cls):
    """Every class the partition can ever produce (over the full 30x30) must be
    actually observed in the data -> no ungrounded residue / band / quadrant."""
    producible = set(int(x) for x in np.unique(cls))
    observed = set()
    for a, _ in prs:
        h, w = a.shape
        observed |= set(int(x) for x in np.unique(cls[:h, :w]))
    return producible <= observed and len(observed) >= 2


def _apply(prs, cls, vecs):
    """Exact reproduction check using the per-class full vectors."""
    for a, b in prs:
        h, w = a.shape
        c = cls[:h, :w]
        out = a.copy()
        for ci, v in vecs.items():
            m = c == ci
            sub = a[m]
            out[m] = np.array(v, int)[sub]
        if not np.array_equal(out, b):
            return False
    return True


# --------------------------------------------------------------------------- #
# model construction                                                          #
# --------------------------------------------------------------------------- #
def _conv_w(vec):
    W = np.zeros((CHANNELS, CHANNELS, 1, 1), np.float32)
    for i, o in enumerate(vec):
        W[o, i, 0, 0] = 1.0
    return W


def _build(regions):
    """regions: list of (mask 30x30 bool, vec length-10).  Disjoint masks."""
    nodes, inits = [], []
    cur = "input"
    for k, (M, vec) in enumerate(regions):
        wt = oh.make_tensor(f"W{k}", DATA_TYPE, [CHANNELS, CHANNELS, 1, 1],
                            _conv_w(vec).ravel().tolist())
        inits.append(wt)
        rec = f"rec{k}"
        nodes.append(oh.make_node("Conv", ["input", f"W{k}"], [rec],
                                  kernel_shape=[1, 1], pads=[0, 0, 0, 0]))
        dims, vals = _mask_init(M)
        inits.append(oh.make_tensor(f"M{k}", BOOL, dims, vals))
        out = "output" if k == len(regions) - 1 else f"cur{k}"
        nodes.append(oh.make_node("Where", [f"M{k}", rec, cur], [out]))
        cur = out
    return _model(nodes, inits)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def candidates(ex):
    prs = _pairs(ex)
    if not prs:
        return []
    if any(a.shape != b.shape for a, b in prs):      # recolour preserves shape
        return []
    if all(np.array_equal(a, b) for a, b in prs):    # identity -> not our family
        return []

    constshape = prs[0][0].shape if len({a.shape for a, _ in prs}) == 1 else None
    ident = list(range(CHANNELS))

    found = []                                        # (cost_key, name, regions)
    seen_behaviour = set()
    for complexity, name, cls in _bank(constshape):
        if not _grounded(prs, cls):
            continue
        maps = _fit(prs, cls)
        if maps is None:
            continue
        vecs = {ci: _vec(d) for ci, d in maps.items()}

        # genuinely position dependent (not a single global recolour) ...
        distinct = {tuple(v) for v in vecs.values()}
        if len(distinct) < 2:
            continue
        # ... and something actually changes
        if all(v == ident for v in vecs.values()):
            continue
        # exact reproduction on every available pair (the grader's gate)
        if not _apply(prs, cls, vecs):
            continue

        # merge classes that share a NON-identity map into one region
        groups = {}                                   # vec_tuple -> mask
        for ci, v in vecs.items():
            vt = tuple(v)
            if vt == tuple(ident):
                continue
            M = (cls == ci)
            groups[vt] = (groups[vt] | M) if vt in groups else M
        regions = [(M, list(vt)) for vt, M in groups.items()]
        if not regions:
            continue

        # behaviour signature (over the observed extent) to dedupe partitions
        sig = []
        for M, vt in regions:
            dims, vals = _mask_init(M)
            sig.append((tuple(vt), tuple(dims), tuple(vals)))
        sig = tuple(sorted(sig))
        if sig in seen_behaviour:
            continue
        seen_behaviour.add(sig)

        # cost: prefer fewer regions, then smaller mask params, then partition size
        mask_params = sum(int(np.prod(_mask_init(M)[0])) for M, _ in regions)
        found.append(((len(regions), mask_params, complexity, name), name, regions))

    if not found:
        return []
    found.sort(key=lambda r: r[0])

    out = []
    for _, name, regions in found[:3]:
        try:
            model = _build(regions)
            onnx.checker.check_model(model, full_check=True)
        except Exception:
            continue
        out.append((f"colorbypos_{name}", model))
    return out
