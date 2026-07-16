# NeuroGolf 7250-Level Recon: ONNX Cost-Minimization Techniques

Source: 5 public 7243–7250 LB notebooks + dissection of 400 winning pool nets
(uradkr 7250.18 bundle). Our portfolio avg **14.6/task**; this bundle avg
**18.125/task** (7250.06 local mirror); leaders ~20.15/task.

---

## 0. The scoring reality (measured, not assumed)

From the official verify/score code embedded in every notebook
(`neurogolf_utils.score_network(sanitized, sess.end_profiling())`):

```
points_per_task = max(1.0, 25.0 - ln(max(1, memory + params)))
memory = bytes of every INTERMEDIATE tensor, measured by the ORT profiler
         run with graph_optimization_level = ORT_DISABLE_ALL (NO fusion)
params = element COUNT of all initializers + Constant-node tensors (dtype-blind)
input & output tensors are FREE. node ATTRIBUTES are FREE.
```

Two consequences that reframe everything:

- **Memory is per-NODE-OUTPUT, from a real no-optimization ORT run.** A
  **single-node graph has exactly zero memory** (its only output is the free
  `output`). Cost then = params only. This is why 76/400 winning nets are a
  single node.
- **params counts ELEMENTS, not bytes.** An int64 scalar costs `1`, same as an
  fp32 scalar. Downcasting an *initializer's* dtype does **not** help params —
  it only helps if that tensor is an *intermediate* (memory). Shrinking element
  *count* (rank factorization, broadcast compression, scalar collapse) is what
  moves params.

**The cost cliff:** every **+1 point = cost ÷ e (2.72)**. Halving cost = +0.69.
One fp32 `[1,10,30,30]` canvas = 36 KB ≈ 14.5 pts. Turning that single
intermediate into uint8 = 9 KB = **+1.39 pts**. Polishing 5% off a 20k net =
+0.05 (worthless — redesign instead).

**Pool cost distribution (400 tasks, this bundle):**

| points | cost range | # tasks |
|---|---|---|
| 25.0 | 0–1 | 3 |
| ≥22 | ≤20 | 9 |
| 20–22 | 21–148 | 43 |
| 18–20 | 149–1096 | 134 |
| 16–18 | 1097–8103 | 173 |
| 14.5–16 | 8104–36k | 38 |
| <14.5 | >36k | **0** ← nobody ships a raw fp32 grid anymore |

Median cost 1300, mean 2919, max 34400. The whole game is dragging the 173
mid-tasks from 16–18 up toward 20, and never letting any task exceed one grid.

---

## (a) RANKED reusable lowering techniques

Ranked by points-per-task delivered × how many task-classes they hit.

### T1. Single-node graph — the zero-memory primitive  ⭐ biggest lever
**Replaces:** any multi-node pipeline whose whole computation is one algebraic
transform. **Cost:** memory → **0**; only params remain.
**Verified single-node ops in the pool (76 nets):** Conv ×33, Einsum ×25,
Gather ×12, RoiAlign ×2, MaxRoiPool ×2, Transpose ×2.

Confirmed instances:
- `task067`: `Einsum("nchw,nkwh->nchw", [input,input])` — **self-gating**, one-hot
  input gates itself. 0 params, 0 mem → **25.0**. (Equation found by GPU-sweeping
  the einsum grammar, not by hand.)
- `task179/241`: `Transpose(perm=[0,1,3,2])` — grid mirror/transpose. 0 params → **25.0**.
- `task016`: `Gather(axis=1, idx int64[10])` — **color permutation/remap** (10 params → 21.6).
- `task113`: `Gather(axis=2, idx int64[30])` — **row/col permutation** (30 params → 21.6).
- `task015`: `Conv(3×3, pads=[1,1,1,1], W[10,10,3,3])` — full spatial+color
  morphology in ONE free node (900 params → ~18.2).

**Applies to:** recolor, transpose/mirror/rotate, channel/axis permutation,
crop, single linear spatial filter, low-rank remap. If the task is one linear or
gather operation, it should be one node.

### T2. Rank-factored Einsum sandwich (free initializer reuse)
**Replaces:** a full dense remap weight (e.g. 900-elem conv) with two small
factors. **Cost:** `2·D·k` params for rank-k, and the SAME initializer feeding
N einsum inputs is **counted once**.
Pattern: `Einsum('ra,ai,zcij,bj,sb->zcrs', [U,S,x,S,U])` reuses `U`,`S` twice for
free. **Verified: task108 971 → 300 params.** Pool task313/166/104 are all
low-rank einsum remaps (24–222 params, 19.5–21.8 pts).
**Applies to:** any color↔color or spatial↔spatial linear map that's low rank
(most ARC recolor/scale/tile rules are).

### T3. Comparison/bool terminal — only the SIGN matters
**Replaces:** exact fp32 output rendering. **Cost:** output becomes bool (1
byte, and it's FREE as the output anyway) and the last op is `Equal`/`Greater`/
`>0`. **Terminal-op census of the pool:** Pad 94, **Equal 85**, Einsum 54,
Where 45, Conv 38. So the grader compares — you only need the right sign
pattern, not exact values.
**Corollary (SGD-fitted sign nets, tasks 171/266/278…):** fit a tiny conv with
**hinge loss on ±1 targets**, quantize, emit, verify. Gradient descent finds
smaller sign-correct nets than humans.
**Applies to:** every boolean-output task (a large fraction of the suite).

### T4. uint8 / fp16 morphology architecture (kill the fp32 grid)
**Replaces:** fp32 grid intermediates. **Cost:** 4× cut fp32→uint8/bool, 2× fp32→fp16.
Canonical pipeline (tasks 077/118/187/243/367):
`Conv(1×1, negative pads)` = fused channel-select + **free crop** → `Cast` uint8 →
`QLinearConv`/`MaxPool` morphology → `Where(mask, color, input)` writes the free
output. Verified 25–40% cheaper than hand-golfed equivalents.
Pool evidence: task046/365 keep their unavoidable intermediates in u8/i8/f16/bool
on **downsampled** shapes (`[1,1,3,30]`, `[1,1,10,10]`) not `[1,10,30,30]`.
**Applies to:** flood-fill, CCL, stamp/erode/dilate, any task that genuinely
needs a working canvas — shrink its dtype and its spatial extent.

### T5. Free attributes carry the geometry (negative pads = free crop)
**Replaces:** tensor-encoded geometry. **Cost:** 0 — attributes are free.
Free: `kernel_shape`, `pads` (**including negative = crop**), `strides`,
`Einsum` equation string, `Clip`-6 min/max, `Trilu` upper/lower, and Slice/Pad
geometry when expressible as attributes. Encode logic in attributes, never in
initializers.
**Applies to:** all crop/pad/window/threshold logic.

### T6. Terminal GridSample (gather + mask + zero-pad in one free node)
**Replaces:** multi-node gather/mask/pad chains. **Cost:** one fp16
`[1,30,30,2]` grid (legal alongside an fp32 input), single terminal node.
**Verified: task029 10770 → 5606.**
**Applies to:** warps, shifts-with-boundary, remaps needing out-of-bounds zeroing.

### T7. ConvInteger / QLinearConv renderer
**Replaces:** float rendering + separate background handling. **Cost:** u8 codes
+ `x_zero_point=1` gives signed weights and background-decoding pads for free;
i32 output graded by `>0`. **Verified: task031 1472 → 688.** (QLinearConv ×187,
ConvInteger terminal ×9 in the pool.)
**Applies to:** stamp/paint-by-code, palette rendering, integer morphology.

### T8. RoiAlign / MaxRoiPool as a crop-and-resize primitive
**Replaces:** slice+resize chains. **Cost:** rois `[1,4]`/`[1,5]` = 4–5 params,
single node. Pool task087 (RoiAlign, rois[1,4]→23 pts), task307/223 (MaxRoiPool,
rois[1,5]→23.4 pts).
**Applies to:** "extract the object / bounding box and rescale to canvas".

### T9. value_info-legalized data-dependent Slice/Pad
**Replaces:** static-only crops. **Cost:** attach a static `value_info` to a
dynamic-origin window; the checker + profiler then accept a bbox crop taken
straight off the free `input`. **Verified: task014 7985 → 4171.**
**Applies to:** crop-to-content where the offset depends on the input.
⚠️ Only with a **proven** max window size (generator source or ≥10⁴ draws) — an
observed-from-demos bound can silently fail the hidden suite.

### T10. Canvas-crop surgery (shrink the whole working grid 30→N)
**Replaces:** a full 30×30 working canvas when the generator provably never
exceeds N×N. **Cost:** crop right after input consumers, rewrite every
canvas-encoding constant (30→N, 900→N²), re-inflate once before the terminal.
**Verified: task018 38247 → 33865.**
**Applies to:** any big-intermediate task with a provable small bounding size.

### T11. int64→int32 index narrowing (memory-only, mechanical)
**Replaces:** int64 index *intermediates*. **Cost:** 8→4 bytes/elem on
intermediates (ArgMax outputs, gather indices) where every consumer accepts
int32. Does **nothing** for initializer params (element count unchanged) — use
only on tensors the profiler counts as memory.
**Applies to:** the long bitwise/ArgMax pipelines (BitwiseAnd ×1872, Cast ×1848,
ArgMax ×414 dominate the pool's op census).

### T12. Broadcast / scalar compression of initializers
**Replaces:** uniform or axis-repeated initializers. **Cost:** `[5,5,5,5]`→scalar
`5`; `[1,10,30,30]` all-equal-on-an-axis → `[1,10,1,30]`, when every consumer is
a broadcasting elementwise op (`Add/Sub/Mul/Div/Where/Greater/…`). Pure param
reduction, audited by re-verify.

---

## (b) The batch / automated pipeline (audit · blend · surgery)

Three cooperating loops rebuild the 400-net bundle:

**1. Verified min-merge / blend (uradkr, kojimar, franksunp).**
Collect every top public submission. For each task, keep the **strictly
cheapest** net, then **re-verify each with the official
sanitizer+profiler on ALL examples** (`train + test + arc-gen`, exact
`np.array_equal`). Reject any that fails — this defuses the classic
**local-pass → LB-0 trap** (a net overfit to shipped demos zeroing your whole
submission). kojimar's packager is literally `base_tasks.update(overrides)` then
zip in task order; franksunp does **incremental single-rewire candidates**: one
conservative tensor rewire per version, validated on all examples + strict
400-task archive `check_model` + a direct competition-score row, published only
if cost strictly drops and score holds (e.g. task208 4189→4185, saved 4).

**2. Staged graph surgery (seddiktrk) — the reusable engine.**
`perform_surgery(surgeon, verify_on_data=True)` loops all 400 nets; each surgeon
returns a rewritten graph; it is re-scored with the exact official
`score_task` (`onnx.save → check_network → sanitize → ORT no-opt profiling →
verify train/test/arc-gen → score_network`) and **kept only if
`score_after − score_before > 0` AND verification passes**. Passes, safest→riskiest:
1. **Generic simplification** — `onnxscript.optimizer` (32 iters) + `onnxsim.simplify`.
2. **Lossless cleanup** — prune unused initializers, **dedup identical
   initializers** (rewire N refs to one canonical → free reuse), remove Identity.
3. **Index surgery** — int64→int32 where all consumers accept it (with the
   explicit `INT32_SAFE_INPUTS` / `INT64_REQUIRED` op-position tables),
   insert `Cast(int32)` after ArgMax/ArgMin, dedup tiny shape/index initializers,
   strip default Slice `axes`/`steps` (and hole-punch axes with `input[3]=""`).
4. **Broadcast compression** — scalar + axis collapse (T12), whitelisted ops.
5. **Micro-rewrites** — Constant→initializer rescue, **Conv1×1 channel
   permutation → `Gather(axis=1)`** (T1), other pattern swaps.
6. **FP16 surgery** — fp32→fp16 on value-preserving ops, guarded by
   `|x| ≤ 65504`, audited by re-verify.

Every pass deletes stale `value_info` after edits and runs `onnx.checker`.

**3. Exact local scorer replica.** Calibrated to the LB within ~0.11 pts over
400 tasks. **Environment is load-bearing:** grader uses `onnxruntime==1.24.4`
(uditjain notes onnx 1.21 + ort 1.26 for their replica) — mismatched versions
mis-measure memory. Always score in the grader's exact env.

**Reusable takeaway for us:** build the `score_task` harness first (it IS the
official pipeline), then run each candidate net through
`sanitize → ORT-no-opt-profile → verify-all-examples → cost`, and accept only
strict, verified improvements. This is the safe inner loop for everything above.

---

## (c) Cost-model surprises they exploit

1. **Single-node ⇒ zero memory.** Memory is the ORT profiler's per-node-output
   allocation under `ORT_DISABLE_ALL`. Collapse to one node and memory
   vanishes; only params remain. (We pay 14.5 for a named fp32 canvas that,
   as the sole `output`, would be FREE — our canvas is an *intermediate*, theirs
   is the *output*.)
2. **input & output are free; make your grid the output.** Never leave the final
   canvas as a named intermediate. The terminal op should write directly to
   `output`.
3. **params is dtype-BLIND (element count).** int64 scalars cost the same as
   fp32 — use int64 freely for scalars/indices. Downcasting initializer dtype
   buys **nothing** on params; only element-count reduction (rank/broadcast) does.
4. **Attributes are free — including NEGATIVE pads (free crop) and the entire
   Einsum equation string.** Push all geometry/thresholds/logic into attributes.
5. **The same initializer feeding N inputs is counted once.** Rank-factored
   einsum sandwiches and dedup'd constants exploit this directly.
6. **Grader compares (`>0`, `Equal`) ⇒ only the sign pattern must be right.**
   Enables SGD-fit sign nets and bool terminals (bool = 1 byte and free as output).
7. **value_info can legalize otherwise-illegal data-dependent Slice/Pad**, so
   bbox crops read straight off the free input.
8. **Trap — sparse_initializer:** the official sanitizer leaves dangling refs →
   load failure. Never use it.
9. **Trap — unproven static window sizes:** bounds from a few local demos can
   pass locally and fail the hidden ARC-gen suite (fresh seeds, same generator).
   Prove sizes from arc-gen source or ≥10⁴ draws.

---

## Immediate opportunities for our portfolio (14.6 → ~18)

- **Stop naming the fp32 canvas.** Any task whose answer is one transform: emit a
  single node writing straight to `output` (T1). Instant memory→0.
- **Recolor / permute / transpose / crop tasks → single Gather/Transpose/Conv/
  RoiAlign** (T1, T8): 21–25 pts each.
- **Low-rank linear rules → rank-factored Einsum** (T2): sub-300 params.
- **Boolean-output tasks → comparison terminal + SGD-fit sign net** (T3): drop
  exact-value weights entirely.
- **Genuine canvas tasks → uint8/fp16 + downsampled working shape** (T4): +1.4
  pts just from fp32→uint8 on the one unavoidable intermediate.
- **Port seddiktrk's `perform_surgery` + official `score_task`** as our verified
  inner loop; run min-merge against every public bundle first for free wins.
