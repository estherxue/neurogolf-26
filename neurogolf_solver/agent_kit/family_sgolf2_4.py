"""family_sgolf2_4 -- SAFE fixed-size CROP golf for slice [4::7] FIXED targets.

Technique (anti-overfit safe, identical to family_sgolf_0): a task whose every
train+test+arc-gen INPUT *and* OUTPUT share one identical square GxG is FIXED-size,
so the byte-identical baseline solver can be run on a smaller SxS (S>=G) work area
instead of the padded 30x30 canvas.  We Slice the input -> [1,10,S,S], run the SAME
graph (same ops, same numerics; only the interior geometry constants rescaled from 30
to S via the family's own size globals), then Pad the SxS result back to [1,10,30,30].
The value is identical for every grid the generator can produce at this fixed size;
only the intermediate working resolution shrinks, lowering the memory term of
cost = params + intermediate_memory.

Base families carrying the still-full-30x30 baselines for my FIXED targets:

  * 361  c4_rot_symm      family_crk9_1  (G=10) -> crop10  ~11.65 -> ~13.74
  * 222  keeprect         family_golf2_0 (G=16) -> crop16  ~12.71 -> ~13.70
  * 162  g73_f16_162      family_golf7_3 (G=20) -> crop20  ~14.05 -> ~14.19
  * 265  localrecolor_c2  family_crk9_2  (G=18) -> crop18  ~13.97 -> ~14.16

We also crop-wrap a few sibling baselines (crk2_5/crk2_4/golf8_1) as bonus candidates;
the shared harness/integrator keeps only the cheapest EXACT solver per task, so extra
proposals are pure upside and a wrong / too-small S is simply rejected.

Each family's own candidates() re-runs its numpy reference gate, so it only fires on
the matching task.  The family calls onnx.checker on the raw (still-30-declared) model,
which fails once the interior is SxS -- so we no-op the checker only for the duration of
that rebuild, then validate the crop-wrapped model with the REAL checker before yielding.
The grader re-checks EXACTness on all train+test+arc-gen, so nothing overfits.
"""
from __future__ import annotations

import copy
import importlib

import numpy as np
import onnx.checker as _chk
from onnx import helper as oh, TensorProto as TP

# Base families whose baseline still runs on the full 30x30 canvas for a FIXED target.
_TARGET_FAMILIES = [
    "family_crk9_1",   # 361 c4_rot_symm
    "family_golf2_0",  # 222 keeprect
    "family_golf7_3",  # 162 g73_f16_162 (and 228/390/68 variants)
    "family_crk9_2",   # 265 localrecolor_c2
    "family_crk2_5",   # 342 crk2_5_quad
    "family_crk2_4",   # 68  crk2_4_t68
    "family_golf8_1",  # 41  golf8_span41
]
_SPAN = 3  # try S = G .. G+2 ; harness keeps the cheapest EXACT one

_SIZE_ATTRS = ["H", "W", "HEIGHT", "WIDTH", "GRID", "S", "N", "NH", "NW"]


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
    _chk.check_model = lambda *a, **k: None
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
