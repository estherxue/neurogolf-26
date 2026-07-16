"""family_bp1_2 — local-ORT-legal rebuilds of memory-dominated pool nets.

Targets assigned: task364, task133, task285, task367 (all memory-dominated in
out_blend4).  Dissection results (op dump + per-tensor ORT-profiler memory
breakdown, replicating neurogolf_utils.calculate_memory):

  task367 (e73095fd) — 23-node QLinearConv flood-fill.  The five geodesic-dilation
      steps use `Min(conv_uint8, seed_mask_uint8)` to keep the running mask inside
      the colour-region.  **Min on uint8 is NOT_IMPLEMENTED under local ORT 1.23.2**
      (only int32/fp16), so the shipped net FAILS to load here (scores 0) even though
      its arithmetic is fine.  Because the seed mask `v_z` is 0/1 and the only thing
      that matters downstream is the zero-vs-nonzero support (`Equal(r5c,0)`),
      `Min(conv,v_z)` and `Mul(conv,v_z)` have IDENTICAL support:
          Min(conv,v_z)!=0  <=>  v_z==1 and conv>=1  <=>  Mul(conv,v_z)!=0
      (v_z in {0,1}; QLinearConv saturates to uint8 so values never wrap to 0).
      Swapping Min->Mul makes the graph load under local ORT with bit-identical
      output, at the SAME cost (memory 16100, params 587 -> 15.28 pts).  This is a
      strict win whenever the grader shares the local ORT uint8-Min limitation
      (gate (a) mandates local ORT 1.23.2), and never a regression otherwise.

  task364 (e509e548): loads fine locally (15.06).  Its one fat intermediate is the
      int32 `ConvInteger` code map feeding two `Gather` table look-ups.  Gather
      requires int32/int64 indices (uint8/int8 rejected by ORT), so the int32 code
      map cannot be narrowed without re-adding an equal-size Cast — no strict win
      found.  Skipped.

  task133 (57aa92db) / task285 (b775ac94): also fail to load locally, but via a
      GENUINE multi-way `Max`/`Min` over full [1,10,30,30] grids (stamp overlay /
      candidate selection), not a 0/1 masking.  A load-legal equivalent (fp16/int32
      Max, or Where/Greater) inflates those large intermediates and is not strictly
      dominant; a from-scratch cheaper rebuild of these multi-object programs is out
      of scope.  Skipped.

Only task367 yields.  Detection is by exact self-check: the rebuilt graph is run on
the given train+test pairs and only emitted if every pair matches, so it fires for
task367 alone.
"""
from __future__ import annotations

import pathlib

import numpy as np
import onnx
import onnxruntime as ort

_BASE = pathlib.Path(__file__).resolve().parent / "out_blend4" / "onnx"


def _build_367():
    m = onnx.load(str(_BASE / "task367.onnx"))
    for n in m.graph.node:
        if n.op_type == "Min":          # masked geodesic dilation -> Mul (identical support)
            n.op_type = "Mul"
    onnx.checker.check_model(m, full_check=True)
    return m


def _to_numpy(grid):
    b = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            b[0][int(v)][r][c] = 1.0
    return b


def _from_numpy(out):
    o = out[0] > 0.0
    g = []
    for r in range(30):
        cells = []
        for c in range(30):
            cols = [k for k in range(10) if o[k][r][c]]
            cells.append(cols[0] if len(cols) == 1 else (11 if cols else 10))
        while cells and cells[-1] == 10:
            cells.pop()
        g.append(cells)
    while g and not g[-1]:
        g.pop()
    return g


def _exact_on(model, examples):
    try:
        sess = ort.InferenceSession(model.SerializeToString())
    except Exception:
        return False
    n = 0
    for split in ("train", "test"):
        for e in examples.get(split, []):
            gi, go = e["input"], e["output"]
            if max(len(gi), len(gi[0])) > 30 or max(len(go), len(go[0])) > 30:
                continue
            pred = _from_numpy(sess.run(["output"], {"input": _to_numpy(gi)})[0])
            if pred != [list(r) for r in go]:
                return False
            n += 1
    return n > 0


def candidates(examples):
    try:
        m = _build_367()
    except Exception:
        return
    if _exact_on(m, examples):
        yield ("bp1_e73095fd", m)
