"""family_scrk3_1 — deep-dig attempt at the 10 hardest unsolved tasks
(slice U[1::4] = [18,46,79,101,138,170,191,233,285,366]).

Every one of these was analysed pair-by-pair (train+test+arc-gen). They all
reduce to "detect object(s) at a data-dependent location, then place/select/crop
a data-dependent region at another data-dependent location" — see the module
docstring notes below. None is expressible as a static opset-10 graph, so the
detectors here (all origin-anchored, exactly validated on every split) simply do
not fire for the 10 targets. They are kept because over-proposing validated
static candidates is free (the harness rejects any that are not value-exact) and
lets the family opportunistically pick up any arc-gen variant that happens to be
a clean static transform.

Per-task infeasibility (why static opset-10 cannot express them):
  T18  same-shape: each color-8 "X" object teleports onto a matching single-cell
       marker elsewhere in the grid. Placement offset depends on marker position
       (data). Not local, not origin-anchored.
  T46  3xN->3x(N-2): a multi-colour snake in the middle row grows/moves guided by
       scattered color-5 markers; output width shrinks by a data-dependent amount.
       Sequential simulation with variable output size.
  T79  14x14->3x3: output is the 3x3 object that occurs MOST often among several
       distinct 3x3 objects. Frequency selection over variable positions.
  T101 same-shape: a color-1 "key" shape (top-left) is scaled/stamped onto every
       2x2 color-2 marker block. Stamp content=data, targets=data. Non-local.
  T138 var->var: extract & compress the region between data-located separator
       columns (3s / 8s). Variable-size, data-dependent crop.
  T170 var->3x3: a 3x3 colour palette is masked by which 5x5 blocks are present
       in a meta-grid. Both palette and block-grid locations are data.
  T191 fixed 23x23: copy the input's own template shape and stamp it at every 4-
       marker location. Template content and marker positions are both data; a
       Conv kernel is a fixed initializer, so a data-valued stamp is impossible.
  T233 var->var: crop one framed sub-object out of many. Data-dependent crop.
  T285 same-shape: stamp per-object templates keyed by 2x2 colour blocks.
       Non-local, data-dependent placement.
  T366 var->var: markers in the bottom half select which top-half pattern to emit
       and where; output size varies. Data-dependent selection + crop.

All ten therefore return no candidate. Conclusion: infeasible as static graphs.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builders import (
    identity, transpose_hw, flip_w, flip_h, rot180,
    recolor_gather, recolor_conv, upscale, downscale,
)

TARGETS = {18, 46, 79, 101, 138, 170, 191, 233, 285, 366}


def _pairs(examples):
    ps = []
    for split in ("train", "test", "arc-gen"):
        for e in examples.get(split, []):
            ps.append((np.array(e["input"]), np.array(e["output"])))
    return ps


def _recolor_map(ps):
    """Return length-10 color map if a single consistent per-pixel recolor
    (same shape) explains every pair, else None."""
    if not all(i.shape == o.shape for i, o in ps):
        return None
    m = {}
    for i, o in ps:
        for a, b in zip(i.ravel().tolist(), o.ravel().tolist()):
            if a in m and m[a] != b:
                return None
            m[a] = b
    return [int(m.get(c, c)) for c in range(10)]


def candidates(examples):
    ps = _pairs(examples)
    if not ps:
        return
    same = all(i.shape == o.shape for i, o in ps)

    # --- origin-anchored, exactly-validated static transforms ---------------
    if same and all(np.array_equal(i, o) for i, o in ps):
        yield ("identity", identity())

    if all(i.T.shape == o.shape and np.array_equal(i.T, o) for i, o in ps):
        yield ("transpose", transpose_hw())

    # full-grid flips / rot are only valid when the content fills the grid
    # (30x30) OR the grader-visible region matches; validate exactly on raw grids.
    def _try(fn, transform):
        good = []
        for i, o in ps:
            t = transform(i)
            if t.shape != o.shape or not np.array_equal(t, o):
                return
        return fn
    m = _try(flip_w, lambda a: a[:, ::-1])
    if m:
        yield ("flip_w", m())
    m = _try(flip_h, lambda a: a[::-1, :])
    if m:
        yield ("flip_h", m())
    m = _try(rot180, lambda a: a[::-1, ::-1])
    if m:
        yield ("rot180", m())

    cmap = _recolor_map(ps)
    if cmap is not None:
        # bijective -> cheap Gather; else general 1x1 conv
        if len(set(cmap)) == 10:
            yield ("recolor_gather", recolor_gather(cmap))
        yield ("recolor_conv", recolor_conv(cmap))

    # integer upscale / downscale (origin-anchored)
    for k in (2, 3):
        if all(o.shape == (i.shape[0] * k, i.shape[1] * k) for i, o in ps):
            if all(np.array_equal(np.kron(i, np.ones((k, k), i.dtype)), o) for i, o in ps):
                yield (f"upscale{k}", upscale(k))
        if all(o.shape == ((i.shape[0] + k - 1) // k, (i.shape[1] + k - 1) // k) for i, o in ps):
            if all(np.array_equal(i[::k, ::k], o) for i, o in ps):
                yield (f"downscale{k}", downscale(k))
