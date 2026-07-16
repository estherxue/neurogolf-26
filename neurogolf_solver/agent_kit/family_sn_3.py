"""SINGLE-NODE COMPILE campaign, batch 3.

Targets: tasks 222, 245, 330, 354, 390, 397, 19, 37, 62, 102.

Goal: emit a single-node (0-memory, 0/low-param) ONNX graph that realises the
task's TRUE rule as a FIXED geometric / recolor / permutation / crop map, for a
strict 25.0 (or near) score that beats the incumbent CNN.

Verdict after reading each verifier (verify_<hash> in _rearc/verifiers.py) AND
probing every train+test+arc-gen pair: NONE of the ten tasks is a fixed map.

  222 91714a58  argmax-largest-object + occurrences/shift stamping   -> CCL, data-dep
  245 a1570a43  find box-shaped object, move other objects inside     -> CCL, data-dep
  330 d2abd087  recolor objects by size (size==6 -> 2 else 1)         -> CCL; NEW colors 1,2
  354 ddf7fa4f  diagonal-frontier line growth / object propagation    -> CCL, data-dep
  390 f8a8fe49  subgrid extract + mirror-paste of interior pattern    -> CCL, data-dep
  397 fcc82909  per-object backdrop fill below-right                  -> CCL; NEW color 3
   19 10fcaaa3  tile 2x2 then underfill 8 at diagonal-neighbours of fg-> local dilation; NEW color 8
   37 1f876c06  connect same-colour dot pairs with a line             -> data-dep geometry
   62 2bcee788  mirror the larger of two objects, dock beside smaller -> CCL; NEW color 3
  102 44d8ac46  fill square "delta" holes of each object with 2       -> CCL; NEW color 2

Empirical confirmation (all train+test+arc-gen pairs):
  * No fixed transform (id/flip*/rot*/transpose/anti-transpose) reproduces the
    output for any task.
  * No global per-colour bijection (recolor Gather) is consistent for any task.
  * 330/397/19/62/102 introduce colours (1,2 / 3 / 8 / 3 / 2) absent from their
    inputs -> a colour-preserving Gather/Transpose/Slice/Pad map is impossible in
    principle.
  * 222/245/354/390/37 keep the palette but relocate/connect objects to
    input-dependent positions, so no single fixed pixel permutation exists.

Every rule is data-dependent connected-component labelling / per-object variable
placement / conditional local fill -- exactly the regime where the incumbent
trained CNN (out_blend6) is already near-optimal. Per the campaign's SKIP-fast
directive, we emit nothing rather than ship a regression or an overfit graph.

This module therefore fingerprints each target's train split and, finding no
fixed-map route, returns no candidates. It is kept as a valid, importable family
so the harness records the definitive "no single-node win" result.
"""
from __future__ import annotations

import numpy as np


def _pairs(ex):
    P = []
    for s in ("train", "test", "arc-gen"):
        for e in ex.get(s, []):
            a = np.array(e["input"], int)
            b = np.array(e["output"], int)
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                continue
            P.append((a, b))
    return P


def candidates(ex):
    """Route each task's train fingerprint. No target admits a fixed single-node
    geometric/recolor/permutation/crop map (see module docstring), so no graph is
    ever emitted -- this guarantees we never regress the incumbent."""
    P = _pairs(ex)
    if not P:
        return []

    # Fingerprint: a fixed single-node win requires, at minimum, that every output
    # colour already appears in the corresponding input (a Gather/Transpose/Slice/
    # Pad map cannot invent colours) AND that some fixed transform reproduces the
    # pairs. Both conditions fail for all ten targets; verify and bail.
    for a, b in P:
        if not set(np.unique(b).tolist()) <= set(np.unique(a).tolist()):
            return []  # new colours -> data-dependent recolor/mark, skip

    tf = {
        "id": lambda a: a,
        "flipud": lambda a: a[::-1],
        "fliplr": lambda a: a[:, ::-1],
        "rot180": lambda a: a[::-1, ::-1],
        "T": lambda a: a.T,
        "rot90": lambda a: np.rot90(a, -1),
        "rot270": lambda a: np.rot90(a, 1),
        "antiT": lambda a: a[::-1, ::-1].T,
    }
    for f in tf.values():
        if all(f(a).shape == b.shape and np.array_equal(f(a), b) for a, b in P):
            # A genuine fixed transform would be emitted by family_symfixed already;
            # none of the targets reach here, but keep the guard honest.
            return []

    return []
