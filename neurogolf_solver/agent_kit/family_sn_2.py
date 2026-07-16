"""SINGLE-NODE COMPILE campaign (batch 2): tasks 201, 268, 270, 284, 378, 8, 71,
91, 94, 154.

Goal: for each task, emit a fixed-map single-node (or tiny, zero-memory) ONNX graph
worth ~25 pts when the TRUE rule (see verify_<hash> in _rearc/verifiers.py) is a fixed
geometric / recolor / permutation / crop map. Otherwise skip and leave the incumbent
pool net (out_blend6 / out_maxev) in place.

RESULT OF THIS BATCH: every one of the ten tasks is a data-dependent, object-based
transform, not a fixed map. Verified two ways:

  1. Reading each verifier: all branch on per-input object properties -
     05f2a901 gravitate one object toward another; aba27056 fill delta + shoot lines
     from box corners; ae3edfdc translate 3/7 pixels relative to per-input 1/2 centers;
     b7249182 connect object endpoints + stamp; ec883f72 shoot diagonals from a
     rectangle's corners; 3345333e symmetry-complete + place an object; 3f7978a0 extend
     a line from a marker; 41e4d17e stamp a plus through each 3x3 object's centre;
     6855a6e4 mirror-place an object; 846bdb03 extract a variable subgrid, mirror & paint.

  2. Empirical battery over ALL train+test+arc-gen pairs (~266 each): no global
     id/flip/rot/transpose/anti-transpose, no global per-color recolor, and 0 examples
     left unchanged - so no fixed pixel-permutation or crop map can be exact. 201 and 91
     even have input-dependent OUTPUT shapes (variable subgrid extraction).

A fixed single-node graph therefore cannot be exact on fresh generator samples for any
of these, and the hard no-overfit gate (exact on >=2000 fresh samples, strictly beat the
incumbent) rules them all out. candidates() returns nothing so the harness keeps the
incumbent pool net for every task.
"""
from __future__ import annotations

# Hashes of the tasks in this batch (json task_hash_map.json), kept for traceability.
_BATCH_HASHES = {
    201: "846bdb03", 268: "aba27056", 270: "ae3edfdc", 284: "b7249182",
    378: "ec883f72", 8: "05f2a901", 71: "3345333e", 91: "3f7978a0",
    94: "41e4d17e", 154: "6855a6e4",
}


def candidates(example):
    """No fixed-map single-node form exists for any task in this batch (all rules are
    data-dependent object transforms). Return no candidates so the incumbent stands."""
    return []
