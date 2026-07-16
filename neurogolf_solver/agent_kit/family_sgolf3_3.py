"""family_sgolf3_3 -- cheaper EXACT solvers via FIXED-SIZE work-area cropping.

Same proven mechanism as family_sgolf_0 / family_sgolf3_4: for a task whose every
train+test+arc-gen INPUT and OUTPUT is one fixed square GxG, an existing family's
baseline runs its whole byte-identical computation on the full 30x30 tensor (every
intermediate is [1,K,30,30]). Since the grid is always exactly GxG we Slice the input
to [1,10,S,S] (S >= G, small), rebuild the SAME graph on SxS intermediates (temporarily
patching the family's module-level size globals), then Pad the SxS result back to
[1,10,30,30]. Ops and numerics are unchanged, only the working resolution shrinks ->
value-exact for any grid at this fixed size, and far fewer intermediate bytes -> more
points.

Targets here are FIXED-size baselines whose cropped variant beats the current best:
  - family_crk9_0 : task 392 (crk9_concentric) full-30 baseline 9.9 pts -> crop 10.6 pts
  - family_crk3_3 : task 11  (t11)             full-30 baseline 14.0 pts -> crop 14.2 pts

Each target family's own candidates() re-runs its numpy reference gate, so it only fires
on the matching task; that family's candidates() calls onnx.checker on the raw (still
30x30-declared) model, which fails once the interior is SxS, so we no-op the checker only
for the duration of that rebuild and validate the crop-wrapped model with the real checker
before yielding. We emit several S (>= G) and let the shared harness keep the cheapest
EXACT one; the grader validates EXACTness on all train+test+arc-gen, so any too-small /
wrong S is rejected.

ANTI-OVERFIT: fire ONLY when every train+test+arc-gen input AND output share one square
GxG (a truly fixed-size generator -> the hidden private grids are the SAME size, so
cropping to that size is value-exact, not a grid-size guess). Variable-size tasks fail
this gate and are skipped.
"""
from __future__ import annotations

import copy
import importlib

import numpy as np
import onnx
import onnx.checker as _chk
from onnx import helper as oh, TensorProto as TP

# Each entry: family module carrying the baseline for one FIXED-size target task.
# We try work-area sizes S in [G, G+SPAN); the harness keeps the cheapest EXACT one.
_TARGET_FAMILIES = ["family_crk9_0", "family_crk3_3"]
_SPAN = 3  # try S = G .. G+2

_SIZE_ATTRS = ["H", "W", "HEIGHT", "WIDTH", "GRID", "G", "S", "N", "NH", "NW"]


def _grid_size(examples):
    """Return G if every train+test+arc-gen input AND output is the same square GxG, else None."""
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
    if h != w or not (1 <= h <= 27):
        return None
    return h


def _crop_wrap(model, S):
    """Slice input -> [1,10,S,S], run the SxS graph, Pad output -> [1,10,30,30]."""
    m = copy.deepcopy(model)
    g = m.graph
    for nd in g.node:
        nd.input[:] = ["inp_s" if x == "input" else x for x in nd.input]
        nd.output[:] = ["out_s" if x == "output" else x for x in nd.output]
    g.initializer.extend([
        oh.make_tensor("cwS", TP.INT64, [2], [0, 0]),
        oh.make_tensor("cwE", TP.INT64, [2], [S, S]),
        oh.make_tensor("cwA", TP.INT64, [2], [2, 3]),
    ])
    g.node.insert(0, oh.make_node("Slice", ["input", "cwS", "cwE", "cwA"], ["inp_s"], name="cw_s"))
    g.node.append(oh.make_node("Pad", ["out_s"], ["output"], mode="constant", value=0.0,
                               pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], name="cw_p"))
    return m


def _patch(mod, S):
    old = {}
    for a in _SIZE_ATTRS:
        if hasattr(mod, a) and isinstance(getattr(mod, a), int):
            old[a] = getattr(mod, a)
            setattr(mod, a, S)
    if hasattr(mod, "NC") and isinstance(mod.NC, int):
        old["NC"] = mod.NC
        mod.NC = 2 * S
    return old


def _unpatch(mod, old):
    for a, v in old.items():
        setattr(mod, a, v)


def _rebuild_cropped(mod, examples, S):
    """Rebuild mod's baseline at work-area SxS and crop-wrap. Returns list[(name, model)]."""
    out = []
    old = _patch(mod, S)
    real_check = _chk.check_model
    _chk.check_model = lambda *a, **k: None  # family re-checks the raw 30-declared model; skip it
    try:
        cands = list(mod.candidates(examples))
    except Exception:
        cands = []
    finally:
        _chk.check_model = real_check
        _unpatch(mod, old)
    for name, model in cands:
        try:
            wm = _crop_wrap(model, S)
            real_check(wm, full_check=True)
        except Exception:
            continue
        out.append((f"{name}_crop{S}", wm))
    return out


def candidates(examples):
    G = _grid_size(examples)
    if G is None:
        return []
    res = []
    for fam in _TARGET_FAMILIES:
        try:
            mod = importlib.import_module(fam)
        except Exception:
            continue
        for S in range(G, min(G + _SPAN, 30)):
            res.extend(_rebuild_cropped(mod, examples, S))
    return res
