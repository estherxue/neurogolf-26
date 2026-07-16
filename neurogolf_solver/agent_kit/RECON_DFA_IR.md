# RECON: Trace-Language / DFA notebook → verifier→IR→ONNX lowering compiler

Source notebook: `recon/scottweeden_neurogolf-trace-language-dfa-solvers.ipynb`
(K. Sweeden, "A Trace-Language Framework for Agent Verification", 12 cells).

---

## (a) What the notebook actually does

**Headline: it is NOT an IR compiler. It is a workflow-choreography checker with a small
library of hand-written ONNX solver builders bolted on.** The "trace-language / DFA" part
governs *pipeline phase ordering*, not grid transformations. Two separable halves:

### Half 1 — the DFA "verifier" (cells 2–3, 9–10)
- An `Op` enum of ~40 symbols (the "operation alphabet Σ"): `ANALYZE_TASK`, `BUILD_ONNX`,
  `KRONECKER_SYNTHESIS`, `SYMMETRY_SYNTHESIS`, `GRAVITY_SYNTHESIS`, `LABEL_PROPAGATE`,
  `SCATTERND_HIST`, `FP16_SURGERY`, `CAST_COLLAPSE`, `DIM_SCRUB`, `VERIFY_TRAIN/TEST/ARC_GEN`,
  `COMPUTE_COST`, `BLEND_BUNDLE`, `SHA256_CHECK`, `PACKAGE_SUBMISSION`, `SUBMIT`, etc.
  Many symbols are pure theatre (`SELF_ATTENTION`, `MCTS_SEARCH`, `TRANSFER_LEARNING`,
  `AUTO_ML`, `FEW_SHOT_LEARNING`) — they exist only to be walked in the canonical trace.
- A `DFA` class with states `Q = {INIT, ANALYZED, BUILT, OPTIMIZED, V_TRAIN, V_TEST, V_ARC,
  COSTED, BLENDED, PACKAGED, SUBMITTED, REJECTED, ERROR}` and a hand-written `DELTA` transition
  table. Accept set = `{SUBMITTED, PACKAGED}`.
- Its only real job: reject illegal orderings ("build before analyze", "blend before verify").
  Cell 10 pre-flights a `canonical_trace` list of `Step`s, walking `DELTA` and printing
  `state --[op]--> state'`. It is a lint of the *pipeline*, not of the *math*.
- **Takeaway for us:** the DFA is the least valuable part. We already know our pipeline
  ordering. What we want is the second half plus the *idea* of a verified lowering gate.

### Half 2 — hand-crafted ONNX solver builders (cell 5) — the useful part
Everything is a fixed `[1, 10, 30, 30]` FP32 tensor (one-hot over 10 colors, `encode()` splats
grid pixels into channels). Builders return whole `onnx.ModelProto`s:

- **`make_id`** — single `Identity`. Parameter-free passthrough.
- **`make_recolor(src,dst)`** — `Slice` out src channel → `Greater(0)` mask → `Where(mask,1,·)`
  to write the dst channel, other channels `Slice`d through, one channel zero-filled, then
  `Concat(axis=1)` back to 10 channels. This is the canonical "select a color plane, move it".
- **`make_kronecker(h,w)`** — `Slice` grid → `ReduceMax` over channel → `Greater`→`Cast` to a
  presence mask → `Tile` the grid h×w → `Resize(nearest)` the mask to h·h × w·w → `Mul` to gate,
  `Sub/Add` to restore background, `Pad` back to 30×30, `Concat`. Self-similar fractal / upscale.
- **`make_symmetry(h,w)`** — `Slice` → `Gather` with reversed-index initializers for H-flip and
  V-flip (`idx = arange(w-1,-1,-1)`), `Gather` again for the 180° corner, then 4× `Concat` to
  build the 2h×2w mirror quilt, `Pad` to 30×30.
- **`make_gravity(h,w)`** — separates FG (channels 1–9) from BG (channel 0) by `Slice`, then
  (cell continues) settles mass — the D4/flood family.
- **`make_ca(...)`** — MaxPool-based cellular-automaton / connected-component label propagation
  (referenced in cell 6 solo-probe).

**Three stated "pillars":** (1) rule-based / parameter-free graphs, (2) *static shapes on every
tensor*, (3) *solo-probe validation* — cell 6 feeds a 1-pixel zero tensor through each builder
and asserts output shape `(1,10,30,30)`, no NaN/Inf, deterministic.

### Optimization passes (cell 8)
Real graph rewrites, all structural:
- `cast_elimination` — drop redundant `Cast` chains.
- `dim_scrub` — `squeeze` size-1 dims out of initializers.
- `fp16_surgery` — rewrite FLOAT(1) initializers/inputs/outputs/value_info to FLOAT16(10).
- `validate_model` — `onnx.checker` + shape inference, then reject `BANNED_OPS =
  {Loop, Scan, NonZero, Unique, Compress}` and reject any dynamic (`dim_param`) shape.

### Verification loop (cells 9–10)
`VERIFY_TRAIN → VERIFY_TEST → VERIFY_ARC_GEN → COMPUTE_COST` per task, gated by the DFA so you
cannot `SUBMIT` without having passed all three verifies and a cost computation. `discover_bundles`
+ `load_onnx_bytes` locate a "floor" bundle of prebuilt `.onnx` to blend against. This is exactly
the shape of gate we want — but driven by real oracles (RE-ARC verifiers + ARC-GEN), not a toy DFA.

**Bottom line:** steal the *builder catalogue + static-shape discipline + solo-probe + fp16/cast
optimization + verify-before-emit gate*. Discard the DFA choreography, the 30×30 fixed padding,
and the fake ML ops.

---

## (b) Concrete adaptation: verifier → IR → opset-10 ONNX compiler

### Design shape
```
RE-ARC verify_<id>(I)  ──parse SSA──▶  match to IR template  ──lower──▶  opset-10 subgraph
        │ (oracle)                          │ (6–10 primitives)              │
        └──────────────── ARC-GEN gate: run N generated (in,out) pairs ──────┘
                          emit .onnx only if 100% match AND cost < current
```

We do **not** try to compile arbitrary DSL. We build a **library of parameterized IR templates**,
each backed by a minimal ONNX lowering, and a **matcher** that recognizes which template a
`verify_` one-liner instantiates (by its DSL-call signature + constants). This is tractable
because the short verifiers cluster into a handful of families (see part c).

### The 8 IR primitives (cover the measured families)
Chosen from the actual DSL-call frequency in the ≤8-call verifiers (fill 38, objects 29,
mapply 25, hconcat/vconcat 26, mostcolor/ofcolor 24, hmirror/vmirror/rot 30, canvas 10,
upscale 7, subgrid 6, replace/switch 9). Each primitive lists its opset-10 lowering.

| # | IR primitive | Semantics | Opset-10 lowering (all ≤ opset-10, no banned ops) | Cost notes |
|---|---|---|---|---|
| 1 | **`D4(k)`** | one of the 8 dihedral maps (identity, rot90/180/270, h/v/diag mirror) | `Transpose` (for rotations/transpose-mirrors) + `Gather` with a reversed-index int64 initializer per flipped axis. No float params. | intermediates = one grid; keep FP16 |
| 2 | **`Tile(sh,sw)` / upscale** | Kronecker / block upscale by (sh,sw) | `Tile` with int64 repeats init; for "stamp grid into each nonzero cell" add `ReduceMax→Greater→Cast` presence mask + `Resize(nearest)` + `Mul` (the notebook's kronecker recipe). | biggest tensor is sh·H × sw·W — clamp to true grid size, never 30×30 |
| 3 | **`Recolor(map)`** | per-color remap c→c′ (incl. `replace`, `switch`) | on one-hot planes: `Gather(axis=channel, indices=perm)` with an int64 permutation init — a single node remaps all colors at once. (Cheaper than the notebook's Slice/Where/Concat.) | params = 10-int64 perm ≈ 80 B |
| 4 | **`ColorMask(c)` / select-by-color** | `ofcolor`, `mostcolor`, `leastcolor` → boolean mask | `ReduceSum`/`Equal` on planes → `Greater`/`Equal`; `mostcolor` via `ReduceSum(axis=H,W)`→`ArgMax(channel)`. | mask is BOOL = 1 B/elem — cheapest dtype |
| 5 | **`Fill(mask,c)` / paint / stamp** | write color c where mask true (`fill`, `underfill`, `paint`) | `Where(mask, onehot(c), base)` on the plane stack; `underfill` = `And(mask, bg_mask)` first. | reuses masks; no new params |
| 6 | **`Crop(r0,c0,h,w)` / subgrid** | extract a static sub-rectangle | `Slice` with int64 start/end/axes/step inits. | shrinks all downstream tensors — pure win |
| 7 | **`Flood/CC` (label-propagate)** | connected components / gravity settle | iterated `MaxPool`(3×3, stride1, same-pad) ×K masked by the color plane (notebook's `make_ca`); fixed K unrolled (no `Loop`). Gravity = directional `MaxPool` + shift `Pad`/`Slice`. | K unrolled → K intermediates; cap K to grid diameter |
| 8 | **`Histogram/ScatterHist`** | per-color counts / palette ops | `ReduceSum` over H,W per channel → `(1,10,1,1)`; compare/`ArgMax`/`TopK` for select. | tiny output |

**Concat is the glue** (axis=1 to reassemble color planes, axis=2/3 for quilt layouts — `hconcat`/
`vconcat`). It is not a numbered primitive but every template ends in `Concat`.

These 8 cover the **objects/select+fill (37)**, **D4/tile (21+6)**, **color-mask/fill (11)**,
**crop (7)**, and **recolor (5)** families below — i.e. ~87 of the 90 short tasks in principle.

### Lowering under our cost model (25 − ln(mem + params))
Cost is dominated by `mem` = summed bytes of **intermediate** tensors (counted once; I/O free)
plus `params` = initializer element count. Levers, in order of impact:

1. **Static-shape at the TRUE grid size, not 30×30.** The notebook pads everything to 30×30 →
   9000-elem tensors. A 3×3 task one-hot is 90 elems. This is a ~100× cut in `mem`. Our compiler
   must specialize each `.onnx` to that task's exact input dims (verifiers/ARC-GEN give them).
2. **FP16 everywhere for float planes** (2 B), **BOOL for masks** (1 B). Never FP32 for
   intermediates. Apply the notebook's `fp16_surgery` as a final pass.
3. **Fewer intermediates.** Prefer single-node lowerings (Recolor via one `Gather`, not
   Slice×10+Where+Concat). Fuse `Greater→Cast` only when a later `Mul` needs float.
4. **Small int64 index inits** instead of float weight tensors. A D4 or recolor needs only a
   handful of int64 indices (~tens of bytes), so `params` stays ~O(100) B.
5. **`Crop` early** to shrink every downstream tensor.

Rough per-task budget: a 5×5 recolor with one-hot planes ≈ 250-elem intermediates ×2 B ×~3
tensors ≈ 1.5 KB mem + ~100 B params → cost ≈ 25 − ln(1600) ≈ **17.6**. The existing FP32/30×30
solvers sit far lower (ln(50 KB) ≈ 10.8 → ~14.2), so retargeting the *same* tasks through this IR
is a straight score gain even before we cover new tasks.

### ARC-GEN gating (the verify-before-emit loop)
Replace the notebook's toy DFA with a real 3-stage gate per task, all mandatory before writing
`task###.onnx`:
1. **Train/test gate** — run the candidate ONNX (via `onnxruntime`, `ORT_DISABLE_ALL` so no
   hidden op fusion) on the task's own train+test pairs; require exact grid match after decode
   (`ArgMax` over channel → grid).
2. **ARC-GEN gate** — draw N (e.g. 50–200) fresh `(input, output)` pairs from the official
   ARC-GEN generator for that task id and require **100%** exact match. This is the anti-overfit
   oracle the notebook lacked (its `VERIFY_ARC_GEN` was a stub).
3. **Cost gate** — compute `25 − ln(mem+params)` on the emitted graph; only replace the current
   `out_v5/onnx/task###.onnx` if the new cost is strictly higher. Keep solo-probe (1-pixel +
   random-grid) as a NaN/Inf/shape smoke test, and keep `validate_model` (banned-op + dynamic-shape
   reject, opset-10) as the last hard gate.

Wire it as: `match_template → lower → validate_model → solo_probe → train/test gate →
ARC-GEN gate → cost gate → emit`. A failure at any stage falls back to the existing solver for
that task (never regress).

---

## (c) How many of our 397 tasks look expressible in this IR

Measured directly by parsing `_rearc/verifiers.py` (400 `verify_` fns, SSA form = 1 DSL call per
`xN =` line):

- **≤8 DSL calls: 90 verifiers** ← the primary target set for the IR compiler.
- ≤6 calls: 65 · ≤4 calls: 44 · ≤2 calls: 21. (Long tail: max 157 calls; median ~11.)

Family breakdown of the 90 short verifiers (by dominant DSL signature):

| count | family | maps to IR primitives | example task ids |
|------:|--------|-----------------------|------------------|
| 37 | objects/select + fill | Flood/CC (7) + ColorMask (4) + Fill (5) | 00d62c1b, 05f2a901, 1f876c06, 25ff71a9, 32597951, 39a8645d |
| 21 | D4 mirror + concat (quilt/tile) | D4 (1) + Concat | 3af2c5a8, 3c9b0459, 44f52bb0, 46442a0e, 46f33fce, 4c4377d9 |
| 11 | color-mask / fill | ColorMask (4) + Fill (5) + canvas | 0ca9ddb6, 10fcaaa3, 4258a5f9, 5582e5ca, 6cf79266, 6f8cd79b |
|  7 | crop / subgrid | Crop (6) | 0b148d64, 23b5c85d, 28bf18c6, 72ca375d, a740d043, ae4f1146 |
|  6 | upscale / tile | Tile (2) | 9172f3a0, a8d7556c, ac0a08a4, b91ae062, c59eb873, f25fbde4 |
|  5 | recolor (replace/switch) | Recolor (3) | 0d3d703e, b1948b0a, b60334d2, c8f0f002, d511f180 |
|  3 | other | — | 2dee498d, 7b7f7511, a416b8f3 |

**Realistic first-wave target:** the pure-structural families are the easiest exact wins —
**recolor (5) + upscale/tile (6) + crop (7) + D4 (21) = ~39 tasks** need only primitives
1,2,3,6 + Concat (no connected-components, no float math), so they lower to tiny, cheap graphs
and should pass ARC-GEN cleanly. The **37 objects/select+fill** tasks are higher-value but need
the Flood/CC primitive (unrolled MaxPool) and careful K-capping; treat as second wave. The 11
color-mask/fill are in between.

So: **~39 tasks trivially expressible & cheap now, ~87/90 expressible in principle** with the
8-primitive IR, out of 397 total. The remaining ~307 verifiers (>8 calls) stay on the existing
expensive ONNX solvers until the IR grows more templates.

### Top DSL calls across the 90 short verifiers (drove primitive selection)
`fill 38 · objects 29 · mapply 25 · fork 19 · hconcat 14 · mostcolor 13 · vconcat 12 ·
ofcolor 11 · hmirror 11 · merge 11 · paint 10 · canvas 10 · apply 9 · vmirror 9 · mfilter 8 ·
fgpartition 8 · argmax 8 · upscale 7 · partition 6 · subgrid 6 · rot180 6 · switch 5 ·
replace 4 · colorfilter 3`.
(`mapply/apply/fork/rbind/lbind/compose/merge` are higher-order plumbing, not grid ops — they
disappear once a template is recognized; the matcher pattern-matches the whole SSA graph rather
than compiling these combinators literally.)
