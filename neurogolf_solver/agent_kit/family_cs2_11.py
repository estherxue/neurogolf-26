"""family_cs2_11 — FINAL COMPLETE-SWEEP recompile triage.

Swept tasks 31, 181, 100, 203, 290, 49, 95, 360, 121, 375, 21, 11 for a
strictly-cheaper correct graph.  Every incumbent in out_blend6 is already at or
near its structural floor for the rule it implements (verified against the
true DSL verifiers and the ORT-1.23.2 cost model points = 25 - ln(mem+params)):

  * 360 (e3497940): 1 node, 340 params, mem 0 — a single Gather that folds the
    whole lefthalf/mirror-paint rule into one permutation table.  At floor.
  * 011 (09629e4f): 5 nodes, cost 318 — Einsum-factored, four 4-elem tensors.
  * 021 (1190e5a7): 13 nodes, cost 324 — Einsum frontier-count → tiny canvas.
  * 011/181/375/290/100: cost 318-368, single-color / fixed-geometry outputs
    already computed with only a handful of <=40-byte named tensors.
  * 095 (4258a5f9): 3 nodes, cost 492 — dilation via Slice + QuantizeLinear;
    two 196-byte 7x7 tensors are inherent to a content-dependent border.
  * 031/049/121/203: cost 561-751 but genuinely CCL/ordering rules (largest-
    object crop, smallest solid-rect crop, most-colorful-object block, and a
    dynamic concentric-ring colour reversal).  Their bulk (e.g. 203's 10x10
    u8 remap matrix = 400 B) is the minimal size for a *data-dependent*
    10-colour substitution / object argmax; halving it is only +0.69 pts and no
    fundamentally cheaper structure exists (the ring geometry is size-variable,
    so no fixed Gather/Einsum collapse applies).

No task admits a strictly-cheaper correct graph, so this module yields nothing
(candidates() is a no-op) and cannot regress any incumbent.  The machinery
mirrors the sibling cs2_* modules for consistency: were a win found, it would
be registered in _TARGETS and re-gated on train+test exact by evaluate().
"""
from __future__ import annotations

import json
import os

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONNX = os.path.join(_HERE, "out_blend6", "onnx")

# task_num -> (hash, patched-model builder).  Empty: no strictly-cheaper graph
# survived the hard gates for any swept task.
_TARGETS: dict[int, str] = {}


def _sig(example):
    return json.dumps([example.get("train", []), example.get("test", [])], sort_keys=True)


_REGISTRY: dict[str, tuple[int, str]] = {}


def _build_registry():
    if _REGISTRY or not _TARGETS:
        return
    from ng_utils_shim import tasks_dir
    tdir = tasks_dir()
    for tn, h in _TARGETS.items():
        p = tdir / f"task{tn:03d}.json"
        try:
            ex = json.load(open(p))
        except Exception:
            continue
        _REGISTRY[_sig(ex)] = (tn, h)


def candidates(example):
    _build_registry()
    hit = _REGISTRY.get(_sig(example))
    if not hit:
        return
    tn, h = hit
    try:
        raw = onnx.load(os.path.join(_ONNX, f"task{tn:03d}.onnx"))
        yield (f"cs2_{h}", raw)
    except Exception:
        return
