"""family_pgolf8_2 -- cheaper EXACT solvers for FIXED-square golf targets in my
slice (golf_targets.json[2::7]) via work-area cropping of a *generalizing*
baseline family, gated by a held-out self-check.

Mechanism (same crop-wrap as family_pgolf7_0): several of my slice's targets are
truly fixed-size (every train+test+arc-gen input AND output is one identical GxG
square) yet their incumbent solver runs on the full 30x30 tensor -> every
intermediate is [1,K,30,30].  Because the grid is always exactly GxG we can run
the byte-identical baseline on a GxG work area: Slice input -> [1,10,G,G], run the
SAME graph on GxG intermediates, Pad the GxG result back to [1,10,30,30].  Same
ops, same numerics, smaller working resolution -> strictly cheaper, still exact.

We reuse family_pgolf7_0's crop-wrap (`_rebuild_cropped`) but point it at the
baseline families that carry *my* slice's fixed-square targets.

ANTI-OVERFIT (critical): a crop-wrap of an OVERFIT baseline (e.g. a per-neighbour
LUT that memorised the seen KxK patches) would still pass the grader's visible
train+test+arc-gen yet fail the private held-out grid.  So before emitting we run
a held-out self-check: refit the baseline family on 70% of arc-gen ONLY, and
require the resulting (full-30) model to be EXACT on the untouched 30% + train +
test.  Only families whose underlying rule genuinely generalises survive; the
shared harness then re-validates EXACTness and keeps the cheapest.
"""
from __future__ import annotations

import importlib

import numpy as np
import onnxruntime as ort

import family_pgolf7_0 as pg7
from ng_utils_shim import ng

# Baseline families that (per held-out discovery) carry a GENUINELY GENERALIZING
# exact rule for one of my slice's fixed-square targets and crop-wrap cheaper than
# the incumbent.  Each is re-validated at runtime by the held-out self-check below,
# so a wrong/overfit entry can never be emitted.
_BASELINE_FAMILIES = [
    "family_golf5_2",   # t314 lattice_complete  12.80 -> ~15.13
    "family_sgolf5_4",  # t34  diagbeam          14.23 -> ~14.75
]

_HELD_FRAC = 0.7        # fit on this fraction of arc-gen, test on the rest
_MIN_HELD = 8           # need enough held-out grids for a meaningful check


def _exact_on(model, exs):
    try:
        sess = ort.InferenceSession(model.SerializeToString())
    except Exception:
        return False
    for e in exs:
        b = ng.convert_to_numpy(e)
        if not b:
            continue
        try:
            out = ng.run_network(sess, b["input"])
        except Exception:
            return False
        if not (out.shape == b["output"].shape and (out == b["output"]).all()):
            return False
    return True


def _generalizes(mod, examples, held):
    """True iff refitting `mod` on train+70%-arc-gen yields a full-30 model that is
    EXACT on the untouched held-out grids (i.e. the rule is not memorised)."""
    ag = examples.get("arc-gen", [])
    k = int(len(ag) * _HELD_FRAC)
    fit = {"train": examples.get("train", []), "test": [], "arc-gen": ag[:k]}
    try:
        c70 = list(mod.candidates(fit))
    except Exception:
        return False
    return any(_exact_on(m, held) for _, m in c70)


def candidates(examples):
    G = pg7._grid_size(examples)
    if G is None:
        return []
    ag = examples.get("arc-gen", [])
    k = int(len(ag) * _HELD_FRAC)
    held = ag[k:]
    if len(held) < _MIN_HELD:
        return []
    out = []
    for fam in _BASELINE_FAMILIES:
        try:
            mod = importlib.import_module(fam)
        except Exception:
            continue
        # cheap gate: does the family even fire (exactly) on this task at full size?
        try:
            full = list(mod.candidates(examples))
        except Exception:
            full = []
        if not full or not any(_exact_on(m, examples.get("train", []) +
                                          examples.get("test", [])) for _, m in full):
            continue
        # anti-overfit gate: the rule must generalise to held-out arc-gen
        if not _generalizes(mod, examples, held):
            continue
        for S in range(G, min(G + 2, 30)):
            out.extend(pg7._rebuild_cropped(mod, examples, S))
    return out
