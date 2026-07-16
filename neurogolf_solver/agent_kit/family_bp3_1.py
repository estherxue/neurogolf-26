"""family_bp3_1 — recompile attempts for tasks 54, 18, 319, 364.

candidates(example) fingerprints the task's train pair and dispatches to the
per-task builder.  A builder returns [] when no graph strictly cheaper than the
current out_blend6 model could be *verified* (exact on train+test and >=1500
fresh ARC-GEN samples) — per the hard gate, we never ship a regression or an
unverified graph.

COST MODEL recap: points = 25 - ln(memory + params);  memory = SUM over NAMED
intermediates (each at its MAX runtime shape, dtype bytes; input/output free);
params = initializer/Constant elements.  Compute/MACs are free.

------------------------------------------------------------------ per-task ---
Current costs (out_blend6, measured with the official neurogolf_utils scorer):
    task 54  (264363fd)  mem 25201  params 193   pts 14.86
    task 18  (0e206a2e)  mem 23656  params 692   pts 14.90
    task 319 (ce602527)  mem 21834  params 269   pts 15.00
    task 364 (e509e548)  mem 20700  params 103   pts 15.06

task 364 (e509e548) — recolor green sprites by letter shape (el->1, you->6,
    aitch->2).  This is a per-COMPONENT classification: junction cell (deg>=3)
    => aitch; else #distinct corner-turn-types (by (vert,horiz) neighbour dir)
    == 2 => you, == 1 => el.  The rule is exact (verified 0/2000 in numpy).
    Realising it in label space needs a masked FLOOD to broadcast each
    component's class to all its pixels.  Distinguishing you(2 corners) from
    el(1 corner) provably needs 4 corner-type channels + 1 junction channel
    (a 2-bit max-fold cannot separate them), and a `you` pixel only sees BOTH
    of its corners after ~12 dilation steps (path length across the U).  A
    [1,5,H,W] uint8 stack costs 2200 B; each masked step (MaxPool + Mul) adds
    ~4400 B, so >=6 steps already exceed the 20700 B we must beat, and ~12 are
    required.  The current ConvInteger+MaxPool template matcher is strictly
    cheaper than any correct label-space flood.  SKIP.

task 319 (ce602527) — magnified-sprite match; output is the tiny (<=5x5) native
    sprite.  Its biggest tensor is `input_u8` = Cast(input,uint8) [1,10,30,30]
    = 9000 B (41% of memory), but it is a SHARED tensor feeding 5 consumers
    (row/col occupancy ReduceMax + 3 per-colour-plane Gathers).  Any label-space
    replacement RAISES per-consumer cost (each Gather-from-float plane is 3600 B
    then 900 B cast, x3 > 9000).  The only true saving is cropping the canvas to
    the <=19x19 board, but that is coupled to ~a dozen size-30 constants
    (rev30, thirty_i64, pad_output 25, ...) whose rewrite cannot be verified
    safely within budget.  SKIP.

task 54 (264363fd) — star/flags reconstruction; 286-node multi-stage geometric
    graph, already fully in uint8/bool label space (its 27x ~900 B [1,1,30,30]
    intermediates over a size-30 board give no downsample headroom).  No
    identified structural redundancy; a full rewrite is high-risk/unverifiable
    in budget.  SKIP.

task 18 (0e206a2e) — clone-restore via rotated reference; 673-node graph on a
    size-30 board.  Same situation as task 54: no cheap structural lever found.
    SKIP.
"""
from __future__ import annotations

import numpy as np


# --- task hashes for the four targets (from task_hash_map.json) ---
_HASH = {54: "264363fd", 18: "0e206a2e", 319: "ce602527", 364: "e509e548"}


def _fingerprint(example):
    """Route a task to one of our four targets by cheap structural signatures of
    its train pairs.  Returns the task id (54/18/319/364) or None.

    Signatures are deliberately loose but disjoint across the four:
      * 319 — output is much smaller than input (native sprite <=5x5).
      * 364 — input uses a single non-bg colour (all green=3) and output recolours
              those same cells to {1,2,6} at identical positions.
      * 18  — input & output are the same square size and both contain two copies
              of a small sprite (output differs from input on a few cells).
      * 54  — input & output same size with a 3x3 star motif (colour appears in a
              plus/box neighbourhood) plus large box rectangles.
    """
    tr = example["train"][0]
    gi = np.array(tr["input"])
    go = np.array(tr["output"])
    ih, iw = gi.shape
    oh, ow = go.shape

    # 319: output strictly smaller (the un-magnified sprite).
    if oh <= 6 and ow <= 6 and (oh < ih or ow < iw) and ih >= 10:
        return 319

    if (ih, iw) == (oh, ow):
        in_colors = set(int(x) for x in gi.ravel()) - {0}
        out_colors = set(int(x) for x in go.ravel()) - {0}
        # 364: single input colour (3=green), recoloured in output at same cells.
        if in_colors == {3} and (gi != 0).sum() == (go != 0).sum() \
                and np.array_equal(gi != 0, go != 0) and out_colors <= {1, 2, 6}:
            return 364
        # 18 vs 54: both same-size transforms; distinguish by the star motif.
        # 54 has a 3x3 "star" (a colour forming a plus/box); 18 does not.
        return _split_18_54(gi, go)
    return None


def _split_18_54(gi, go):
    """Heuristic: task 54 always contains a dense 3x3 star block in the output
    (>=5 cells of a single non-bg colour in some 3x3 window); task 18's sprites
    are thin creatures without such a block."""
    h, w = go.shape
    for r in range(h - 2):
        for c in range(w - 2):
            win = go[r:r + 3, c:c + 3]
            vals = win[win != 0]
            if vals.size >= 7:  # a near-full 3x3 block => star/flag task
                return 54
    return 18


# --- per-task builders: each returns [] until a verified cheaper graph exists ---

def _cands_54(example):
    return []


def _cands_18(example):
    return []


def _cands_319(example):
    return []


def _cands_364(example):
    return []


_DISPATCH = {54: _cands_54, 18: _cands_18, 319: _cands_319, 364: _cands_364}


def candidates(example):
    tid = _fingerprint(example)
    if tid is None:
        return []
    return _DISPATCH[tid](example)
