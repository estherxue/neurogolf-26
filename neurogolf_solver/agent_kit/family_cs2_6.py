"""family_cs2_6 — FINAL COMPLETE-SWEEP recompile triage.

Tasks examined (task -> hash -> incumbent structure -> real score):

  294 bb43febb  1-node Conv 10x10x3x3 +bias   params 910 mem 0   pts 18.187
  331 d364b489  1-node Conv 10x10x3x3 +bias   params 910 mem 0   pts 18.187
  344 d90796e8  1-node Conv 10x10x3x3 +bias   params 910 mem 0   pts 18.187
   15 0ca9ddb6  1-node Conv 10x10x3x3         params 900 mem 0   pts 18.198
   22 137eaa0f  4-node Gather assembly        params 738 mem 162 pts 18.198
   98 4347f46a  1-node Conv 10x10x3x3         params 900 mem 0   pts 18.198
  114 49d1d64f  1-node Conv 10x10x3x3         params 900 mem 0   pts 18.198
  151 67a423a3  1-node Conv 10x10x3x3         params 900 mem 0   pts 18.198
  220 913fb3ed  1-node Conv 10x10x3x3         params 900 mem 0   pts 18.198
  230 95990924  1-node Conv 10x10x3x3         params 900 mem 0   pts 18.198
  259 a740d043  41-node dynamic crop          params  48 mem 844 pts 18.207
   47 23581191  19-node separable bitpack     params  36 mem 826 pts 18.241

No strictly-cheaper *correct* structure was found for any of these:

* The nine 900/910-param single Convs are the structural floor for a
  per-colour 3x3-neighbourhood rule.  Input/output are fixed 10-channel
  float32 one-hot, and every one of these rules mixes colours across
  channels (e.g. colour-1 cells drive channel-7 output), so the 10->10
  3x3 weight (900) is irreducible: a grouped conv can't mix channels, a
  1x1 conv can't see neighbours, and slicing to fewer channels would
  materialise a full-grid float intermediate (>=7200 bytes) that dwarfs
  the 900 params.  The +10 bias on 294/331/344 is a genuine per-channel
  negative threshold (keeps the out-of-grid zero region from firing) and
  cannot be folded away.  A single Conv already has ZERO named
  intermediates, so any alternative needs (named-bytes + params) < 900 —
  impossible when even one full-grid uint8 tensor is 900 bytes and none of
  these rules are row/col separable.

* task 47 is ALREADY the low-rank / separable bitpack the arsenal calls
  for: a per-row code [1,1,30,1] BitwiseAnd a per-channel col code
  [1,10,1,30] straight into the free output (mem 826, cost 862).  Its
  essential channel-differentiated tensor (300) plus the tiny float slice
  windows (168+144) leave little slack; shaving it would need exact
  re-derivation of the bit encoding with new border/collision edge cases,
  a high-risk rebuild for a marginal, uncertain gain — not the
  full-CCL-that-is-really-a-fixed-map profile this sweep targets.

* task 259 is already a tight dynamic bounding-box crop (ArgMax/Slice/
  Gather/Pad, mem 844) producing the small cropped patch directly; it is
  not a wasteful CCL/canvas, and any straightforward Slice+Pad rebuild
  would materialise a full-grid intermediate (>=900) and lose.

* task 22 is a compact 4-node Gather assembly (cost 900) for a complex
  nearest-colour move rule; no cheaper fixed map applies.

Every incumbent is at or near its structural floor, so this module emits
nothing (candidates() yields no graph and therefore can never regress).
"""
from __future__ import annotations


def candidates(example):
    return
    yield  # pragma: no cover  (make this a generator)
