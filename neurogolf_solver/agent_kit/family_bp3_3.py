"""family_bp3_3 — recompile attempt for tasks 349, 173, 76, 138.

RESULT: 4 skips (no verified, strictly-cheaper, locally-loadable win found).
candidates() fingerprints each task's train pair and routes, but every route is
currently empty because the analysis below shows the current models already sit at
(or below) the cost floor reachable by any graph that LOADS under local ORT 1.23.2.

=========================  WHY (measured, not guessed)  =========================
Cost model (from the real grader, ng.score_network):
    points = 25 - ln(memory + params)
    memory = sum over NAMED intermediates of (elements * dtype_itemsize), each
             counted once at its MAX runtime shape; 'input'/'output' are FREE.
    params = initializer/Constant elements.
Targets to beat (real grader, = mem_targets2.json, reproduced locally on the models
that load):  349 -> 14892   173 -> 13383   76 -> 12886   138 -> 12042  (mem+params)

THE WALL — uint8 arithmetic does not run on local ORT 1.23.2.
  The out_blend6 models reach their low cost by doing ALL work in uint8 (1 byte
  per element: a [1,1,30,30] tensor is 900 B, a [1,1,15,15] is 225 B).  But local
  ORT 1.23.2 has NO CPU kernel for elementwise uint8 math — verified by loading:
      Min(13)/Max(13) on uint8  -> NOT_IMPLEMENTED   (this is why 349 & 173 will
                                                       not even load locally today)
      Mul on uint8              -> INVALID_GRAPH  (Type 'tensor(uint8)' invalid)
  A graph that must LOAD locally (HARD GATE a) is therefore forced to carry every
  arithmetic intermediate in fp16 (2 B) or int32 (4 B) — a >=2x byte penalty per
  grid-sized tensor.  Since the current cost is dominated by grid-sized arithmetic
  tensors, a locally-loadable re-expression of the SAME algorithm is >=2x, i.e.
  strictly MORE expensive.  You cannot undercut a uint8 model with an fp16 one.

Per task:
  * 349 (db93a21d, "death stars").  Algorithm fully reverse-engineered and a clean
    label/morphology reconstruction was written and VERIFIED exact on 2000 fresh
    generator samples (0 mismatches) — the logic is recoverable.  BUT the halo is
    intrinsically multi-scale (radius 1..5); the shipped model packs all 5 scales
    into ONE dense [1,5,30,30] uint8 conv output (4500 B) + 1182 conv params.  Any
    MaxPool morphology must materialize erosion+dilation per scale (~15 grid tensors
    ~= 27 KB even in uint8; ~54 KB in the fp16 the local runtime forces) -> pts
    ~14.8 << 15.39.  Regression.  The only sub-14892 route is the conv itself, which
    needs QLinearConv (uint8 out) and thus does not load locally.
  * 173 (72322fa7, sprite in-fill).  Does not load locally (uint8 Max(13)).  98-node
    TopK/ScatterElements pipeline; memory 13307 is already spread over dozens of
    tiny [19]/[3]/[7] tensors plus one irreducible float32 [1,1,30,30] label conv
    (3600).  No 2x-cheaper locally-loadable reformulation exists.
  * 76 (36d67576, rainbow-sprite reveal).  Loads; 12798 B already lean (all tensors
    <=900 B, mostly 225 B uint8/bool at the true 15x15 work size).  Dominant single
    item is the float32 [1,1,15,15] label conv (900).  It cannot shrink: Conv output
    dtype follows its input; feeding fp16/uint8 requires casting the [1,10,H,W] input
    first (>=6000-18000 B), which costs far more than the 450-675 B it would save.
  * 138 (5daaa586, framed ray crop).  Loads; 11840 B lean.  Dominant item float32
    [1,1,25,24] label conv (2400); same irreducibility as 76 (input-cast > saving).
    Everything else is 23x23 (=529 B) cropped work area, already minimal.

CONCLUSION: shipping any locally-loadable rebuild of these four would REGRESS their
leaderboard points.  Per the mandate ("NEVER ship a regression or an unverified
graph; a skip with a clear reason is fine"), all four are skipped.
"""
from __future__ import annotations


def _fp(example):
    """Coarse fingerprint of a task from its first train pair (grid sizes + colors).
    Used only to route; all routes are empty (see module docstring)."""
    tr = example["train"][0]
    gi, go = tr["input"], tr["output"]
    ish = (len(gi), len(gi[0]))
    osh = (len(go), len(go[0]))
    colors = sorted({c for row in gi for c in row} | {c for row in go for c in row})
    return ish, osh, tuple(colors)


def candidates(example):
    # No verified, strictly-cheaper, locally-loadable model was found for any of
    # 349/173/76/138 (see docstring). Emit nothing rather than regress.
    return []
