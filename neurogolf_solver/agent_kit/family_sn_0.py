"""SINGLE-NODE COMPILE campaign (tasks 107,234,277,368,396,44,86,93,125,137).

Goal: reduce a task to a single zero-param ONNX node (score 25.0) when its
train->test rule is a FIXED geometric / recolor / permutation / crop map.

Result of analysis (verifier + 5 train examples each):

  107 469497ad  upscale by (numcolors-1) then draw corner shoots  -> variable
                output size (10/15/20 from 5x5), data-dependent. SKIP.
  234 98cf29f8  bordered-frame object gravitate/paint             -> object CCL. SKIP.
  277 b230c067  recolor odd-shaped object (2) vs rest (1)          -> object CCL. SKIP.
  368 e76a88a6  stamp argmax-numcolors shape at other ulcorners    -> object CCL. SKIP.
  396 fcb5c309  find box-object, crop subgrid, swap color          -> variable crop
                (7x7/6x7/7x7), data-dependent. SKIP.
  44  228f6490  recolor bg objects by nearest fg color             -> object CCL. SKIP.
  86  3befdf3e  per-object box outline + diagonals                 -> object CCL. SKIP.
  93  4093f84a  frontier split + column reorder/sort               -> data-dependent. SKIP.
  125 543a7ed5  per-object outbox(3) + delta(4) fills              -> object CCL. SKIP.
  137 5c2c9af4  concentric boxes scaled from leastcolor object     -> object CCL. SKIP.

Checks performed for every task:
  * pure per-cell color map      -> False for all
  * fixed D4 transform (flip/rot/transpose/anti-transpose) -> None matches
  * size relationship            -> 107 & 396 have per-example-variable output size
None reduces to a fixed single-node map, so this family emits nothing.
"""
from __future__ import annotations


def candidates(ex):
    # No task in this batch admits a fixed geometric/recolor/permutation/crop
    # single-node form. Every rule is a data-dependent object/CCL algorithm;
    # the incumbent pool net is already near-optimal for these.
    return []
