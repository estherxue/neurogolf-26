"""family_pgolf7_3 -- cheaper EXACT solvers via FIXED-SIZE work-area cropping of the
full-30x30 baseline families that were NOT already covered by family_sgolf_0.

Several accepted baselines run their whole computation on the FULL 30x30 tensor because
they read a module-level size global (GRID / G / H / W / N ...). For a truly FIXED-size
target (every train+test+arc-gen INPUT *and* OUTPUT is one identical square GxG), that
30x30 canvas is pure waste: the byte-identical algorithm can run on an SxS work area
(S >= G, S small) -> Slice input -> [1,10,S,S], run the same graph on SxS intermediates,
Pad the result back to [1,10,30,30]. Same ops, same numerics, so it stays value-exact for
any grid the fixed-size generator can produce.

Concretely this beats task 392 (concentric Chebyshev rings): the accepted `golf392_crop`
hard-codes an S=12 work area (NC=24 -> a [576,12,12] search tensor). All 392 grids are a
fixed 10x10, so we crop to S=10 (NC=20 -> [400,10,10]) -> ~2x smaller intermediates ->
9.90 -> 10.62 pts. We emit S = G..G+2 and let the shared harness keep the cheapest EXACT
one; the grader validates EXACTness on all train+test+arc-gen, so any too-small / wrong S
is rejected.

Mechanism mirrors family_sgolf_0: temporarily patch the baseline family's size globals to
S, no-op onnx.checker for the size-mismatched rebuild, crop-wrap, then re-validate the
wrapped model with the real checker before yielding. The family's own candidates() re-runs
its numpy reference gate, so it only fires on the matching task.

ANTI-OVERFIT: fires ONLY when every train+test+arc-gen input AND output share one square
GxG (a truly fixed-size generator -> the hidden private grids are the same size, so cropping
to that size is value-exact, not a grid-size guess). Variable-size tasks fail the gate and
are skipped. Self-checked: build on 70% of arc-gen, confirm EXACT on the untouched 30%.
"""
from __future__ import annotations

import copy
import importlib

import numpy as np
import onnx.checker as _chk
from onnx import helper as oh, TensorProto as TP

# Full-30x30 baseline families carrying a fixed-size target, NOT already handled by
# family_sgolf_0 (which owns crk2_5 / crk9_5 / crk9_4 / golf3_4).
_TARGET_FAMILIES = ["family_crk9_0", "family_crk6_0", "family_crk6_3", "family_crk9_3"]
_SPAN = 3  # try S = G .. G+2

_SIZE_ATTRS = ["H", "W", "HEIGHT", "WIDTH", "GRID", "S", "N", "NH", "NW", "G"]


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
        out.append((f"{name}_pcrop{S}", wm))
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
