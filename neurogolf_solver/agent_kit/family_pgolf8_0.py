"""family_pgolf8_0 -- cheaper EXACT solvers for FIXED-SIZE golf targets via
work-area cropping (GOLF slice golf_targets.json[0::7]).

Several of my slice targets are truly FIXED-size (every train+test+arc-gen INPUT
AND OUTPUT is one identical HxW grid), yet their accepted baseline solver runs its
whole computation on the full 30x30 tensor -> every intermediate is [1,K,30,30].
Because the grid is always exactly HxW we can run the byte-identical baseline on a
smaller HxW work area: Slice the input to [1,10,H,W], run the SAME graph on HxW
intermediates, Pad the HxW result back to [1,10,30,30]. Same ops, same numerics,
just a smaller working resolution -> strictly cheaper (when the baseline used
static 30x30 intermediates) and still exact for every grid the generator can
produce at this fixed size.

Mechanism (identical in spirit to family_sgolf_0 / family_pgolf7_0 / _pgolf8_1):
each target's baseline lives in a family module whose builder either (a) is
shape-agnostic (relative Conv/Pad extents, so simply feeding a smaller input
shrinks every intermediate) or (b) derives its spatial extents from module-level
HEIGHT/WIDTH/H/W globals. We temporarily patch those globals to (H,W), rebuild via
that family's own candidates() (whose numpy gate re-fires only on the matching
task), crop-wrap the result, and emit it. The family calls
onnx.checker.check_model on the raw (still 30-declared) model, which can fail once
the interior is HxW, so we no-op the checker for the duration of the rebuild then
run the real checker on the crop-wrapped model before yielding. We emit a couple of
work-area sizes (>= the true size) and let the shared harness keep the cheapest
EXACT one; the grader validates EXACTness on all train+test+arc-gen so any
too-small / wrong size is rejected.

ANTI-OVERFIT: fires ONLY when every train+test+arc-gen input AND output share one
HxW (a truly fixed-size generator -> hidden private grids are the SAME size, so
cropping to that size is value-exact, not a grid-size guess). Variable-size tasks
fail this gate and are skipped.
"""
from __future__ import annotations

import copy
import importlib

import numpy as np
import onnx.checker as _chk
from onnx import helper as oh, TensorProto as TP

# Only family_golf_0's "build*" baselines allocate STATIC 30x30 intermediates, so
# cropping strictly shrinks them -> a genuine win (my slice's fixed-size target t369
# build369: 14.21 -> 15.49; incidentally also t10 build10). The other fixed-size
# baselines behind my slice (family_golf4_0 rank3/sizrecolor, family_golf_5
# holes302, family_golf2_0 dotframe, family_scrk_3 208box) are already shape-agnostic
# (their intermediates are relative, so the UNcropped model is already cheaper than a
# crop-wrapped one -- the Slice/Pad overhead dominates) or already tightly cropped by
# the sibling sgolf* families, so cropping them only loses. We therefore fire crop-wrap
# solely on family_golf_0.
_TARGET_FAMILIES = [
    "family_golf_0",
]
_SPAN = 2  # try work areas (H,W), (H+1,W+1)

_SQUARE_ATTRS = ["GRID", "S", "N", "NH", "NW"]


def _grid_size(examples):
    """Return (H, W) if every train+test+arc-gen input AND output is the same HxW."""
    sizes = set()
    saw = False
    for sec in ("train", "test", "arc-gen"):
        for e in examples.get(sec, []):
            try:
                a = np.array(e["input"], int)
                b = np.array(e["output"], int)
            except Exception:
                return None
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                return None
            sizes.add(a.shape)
            sizes.add(b.shape)
            saw = True
    if not saw or len(sizes) != 1:
        return None
    (h, w), = sizes
    if not (1 <= h <= 28 and 1 <= w <= 28):
        return None
    return (h, w)


def _crop_wrap(model, H, W):
    """Slice input -> [1,10,H,W], run the HxW graph, Pad output -> [1,10,30,30]."""
    m = copy.deepcopy(model)
    g = m.graph
    for nd in g.node:
        nd.input[:] = ["inp_s" if x == "input" else x for x in nd.input]
        nd.output[:] = ["out_s" if x == "output" else x for x in nd.output]
    g.initializer.extend([
        oh.make_tensor("cwS", TP.INT64, [2], [0, 0]),
        oh.make_tensor("cwE", TP.INT64, [2], [H, W]),
        oh.make_tensor("cwA", TP.INT64, [2], [2, 3]),
    ])
    g.node.insert(0, oh.make_node("Slice", ["input", "cwS", "cwE", "cwA"], ["inp_s"], name="cw_s"))
    g.node.append(oh.make_node("Pad", ["out_s"], ["output"], mode="constant", value=0.0,
                               pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W], name="cw_p"))
    return m


def _patch(mod, H, W):
    old = {}
    for a in ("HEIGHT", "H", "NH"):
        if hasattr(mod, a) and isinstance(getattr(mod, a), int):
            old[a] = getattr(mod, a)
            setattr(mod, a, H)
    for a in ("WIDTH", "W", "NW"):
        if hasattr(mod, a) and isinstance(getattr(mod, a), int):
            old[a] = getattr(mod, a)
            setattr(mod, a, W)
    if H == W:
        for a in _SQUARE_ATTRS:
            if hasattr(mod, a) and isinstance(getattr(mod, a), int):
                old[a] = getattr(mod, a)
                setattr(mod, a, H)
        if hasattr(mod, "NC") and isinstance(mod.NC, int):
            old["NC"] = mod.NC
            mod.NC = 2 * H
    return old


def _unpatch(mod, old):
    for a, v in old.items():
        setattr(mod, a, v)


def _rebuild_cropped(mod, examples, H, W):
    out = []
    old = _patch(mod, H, W)
    real_check = _chk.check_model
    _chk.check_model = lambda *a, **k: None  # skip the family's raw 30-declared check
    try:
        cands = list(mod.candidates(examples))
    except Exception:
        cands = []
    finally:
        _chk.check_model = real_check
        _unpatch(mod, old)
    for name, model in cands:
        try:
            wm = _crop_wrap(model, H, W)
            real_check(wm, full_check=True)
        except Exception:
            continue
        out.append((f"{name}_pg8crop{H}x{W}", wm))
    return out


def candidates(examples):
    hw = _grid_size(examples)
    if hw is None:
        return []
    H0, W0 = hw
    res = []
    for fam in _TARGET_FAMILIES:
        try:
            mod = importlib.import_module(fam)
        except Exception:
            continue
        for k in range(_SPAN):
            H, W = min(H0 + k, 30), min(W0 + k, 30)
            res.extend(_rebuild_cropped(mod, examples, H, W))
    return res
