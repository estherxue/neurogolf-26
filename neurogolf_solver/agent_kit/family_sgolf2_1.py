"""family_sgolf2_1 -- cheaper EXACT solvers via FIXED-SIZE work-area cropping.

Same mechanism as family_sgolf_0, pointed at the float32 size-global baselines that
build the FIXED-SIZE target tasks in this agent's slice (e.g. 354 flood, 212 line
ray-cast). Those baselines run their whole computation on the full 30x30 tensor, so
every intermediate is [1,K,30,30] and any [G,G] matrix initialiser costs G*G params.
When every train+test+arc-gen INPUT *and* OUTPUT is the same square GxG, the byte-
identical graph can run on a smaller SxS work area (S >= G): Slice input -> [1,10,S,S],
run the same graph on SxS intermediates (patching the family's size global G=S so all
interior slice offsets/kernel matrices shrink), Pad the SxS result back to
[1,10,30,30]. Same ops, same numerics -> exact for any grid the generator produces at
this fixed size. Memory shrinks ~ (S/30)^2 and any [G,G] param matrix shrinks the same.

ANTI-OVERFIT: fires ONLY when every train+test+arc-gen input AND output share one
square GxG (a truly fixed-size generator -> the hidden private grids are the same size,
so cropping is value-exact, not a grid-size guess). VAR tasks fail this gate and are
skipped. Each family's own candidates() re-runs its numpy reference gate (so it only
fires on the matching task), and the shared harness validates ONNX EXACTness on all
train+test+arc-gen; too-small / wrong S is rejected.
"""
from __future__ import annotations

import copy
import importlib

import numpy as np
import onnx
import onnx.checker as _chk
from onnx import helper as oh, TensorProto as TP

# float32 size-global baselines carrying a FIXED-size target task for this slice.
_TARGET_FAMILIES = [
    "family_crk2_0",   # task 354 flood-fill legend  (fixed 10x10)
    "family_crk5_2",   # task 212 horizontal line ray-cast + [G,G] matmul params (fixed 10x10)
]
_SPAN = 1  # S = G exactly (all content fits the real GxG region; cheapest exact work area)

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


def _wrap_steps(mod, fname, steps):
    """Force the default step count of mod.<fname>(x, steps=...) to `steps` for the
    duration of a rebuild. Returns the original callable to restore. The chosen step
    count matches the accepted baseline's safety buffer (observed max propagation +2);
    only the spatial work area shrinks, so correctness is unchanged."""
    orig = getattr(mod, fname, None)
    if orig is None:
        return None

    def _forced(*a, **k):
        k.setdefault("steps", steps)
        return orig(*a, **k)

    setattr(mod, fname, _forced)
    return (fname, orig)


# family module -> (builder function name, forced step count). The step count mirrors
# the accepted baseline (observed max propagation depth across all pairs +2 buffer), so
# only the spatial resolution changes vs. that baseline.
_STEP_OVERRIDES = {
    "family_crk2_0": ("_build_flood", 6),  # 354 flood: observed max depth 4, +2 buffer
}


def _rebuild_cropped(mod, examples, S):
    """Rebuild mod's baseline at work-area SxS and crop-wrap. Returns list[(name, model)]."""
    out = []
    old = _patch(mod, S)
    step_saved = None
    ov = _STEP_OVERRIDES.get(mod.__name__)
    if ov:
        step_saved = _wrap_steps(mod, ov[0], ov[1])
    real_check = _chk.check_model
    _chk.check_model = lambda *a, **k: None  # family re-checks the raw 30-declared model; skip it
    try:
        cands = list(mod.candidates(examples))
    except Exception:
        cands = []
    finally:
        _chk.check_model = real_check
        if step_saved:
            setattr(mod, step_saved[0], step_saved[1])
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
