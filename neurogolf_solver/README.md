# neurogolf_solver

A faithful, extensible solver pipeline for the **2026 NeuroGolf Championship** (build the
smallest ONNX network that exactly solves each ARC-AGI task). It produces a `submission.zip`
and scores every candidate through the **official `neurogolf_utils.py`**, so local points
equal leaderboard points.

## What it does
Per task, it tries solver tiers from cheapest/safest to most general, validates each candidate
against **all** `train + test + arc-gen` examples (the grader's 0-wrong gate), and keeps the
**highest-scoring** one (`points = max(1, 25 − ln(params + intermediate_memory))`).

| Tier | File | What it solves | Generalizes? |
|------|------|----------------|--------------|
| Symbolic | `builders.py` + `solve.py` | identity, transpose, recolor (Gather/1×1 Conv), flips, rot180, translate | ✅ exact rule |
| Local learner | `learner.py` | any **K-local** rule, compiled to an exact `Conv→ReLU→Conv` lookup | ✅ *with gate* |
| CNN trainer | `trainer_torch.py` | general per-task CNN trained on arc-gen (needs torch) | ✅ *with gate* |

**Anti-overfit gate (the private-set concern).** The local learner and CNN trainer both FIT
on part of arc-gen and require **exact correctness on the held-out remainder** (same generator
as the hidden private set) before a solution is accepted. This rejected memorizing solutions
(e.g. a 1301-pattern "local rule") that would score 0 on the private set.

## Run
```bash
# numpy venv (symbolic + local learner; CNN auto-skipped if torch absent)
python solve_all.py --start 1 --end 400 --out ./out
# -> out/onnx/taskNNN.onnx, out/report.json, out/submission.zip
```
Data + the official utils are located via `ng_utils_shim.py` (set `NG_DATA_DIR`). Download with:
```bash
kaggle competitions download neurogolf-2026 -p $NG_DATA_DIR && unzip -d $NG_DATA_DIR/tasks ...
```

## Current result (honest)
- **Symbolic + gated local learner: 15 / 400 solved, 247.12 pts** — a valid, *generalizing*
  `submission.zip` (max file 61 KB, all ≪ 1.44 MB). See `out_baseline/`.
- The CNN tier works and exports legal opset-10 graphs, but per-task CPU training is slow
  (~80 s/task) and MPS is unavailable here, so it can't scale to 400 in one session. It is
  included for a GPU/where-compute-allows run.

## Why 7000 is a different order of magnitude
Leaders sit at ~7900 ⇒ they solve **nearly all 400** ARC-AGI-1 training tasks with small,
*generalizing* networks. That is essentially "solve ARC-AGI-1 training" — a multi-week effort
that 2500 specialized teams are pushing on. The fixed 30×30 top-left padding also blocks naive
flip/rotate/translate (content lands off-position), so most tasks need either learned CNNs
(positioning learned from arc-gen) or a rich program-synthesis DSL.

## Roadmap to climb toward 7000 (in priority order)
1. **GPU CNN trainer** (the biggest lever): run `trainer_torch.py` per task with architecture
   search (the `LADDER`) on CUDA/MPS; with seconds-per-task it can cover the large "learnable"
   subset. Keep the held-out-arc-gen gate to stay generalizing.
2. **More true-rule symbolic families**: tiling/scaling, crop-to-content, symmetry completion,
   gravity, bordering, connected-component recolor — each compiled to minimal opset-10 graphs.
   These generalize perfectly and golf well.
3. **Local learner upgrades**: minimal-support detection (fewer patterns → less memory → more
   points) and a learned linear/2-layer local classifier (interpolates to unseen neighborhoods).
4. **Program search → ONNX compiler**: an ARC DSL whose programs compile to tensor graphs, with
   the arc-gen oracle as the fitness function.
5. **Cost golfing pass**: quantize/sparsify initializers, prune channels, fuse nodes, shrink
   internal shapes — turns a solved task from ~12 pts toward ~20–25.

## Files
- `ng_utils_shim.py` — locate + import the official `neurogolf_utils` (scoring ground truth).
- `builders.py` — opset-10 ONNX graph builders (one tiny graph per family).
- `learner.py` — local-rule lookup → exact `Conv→ReLU→Conv`, with generalization gate.
- `trainer_torch.py` — per-task CNN trainer (BCE@>0 == grader), held-out gate, ONNX export.
- `evaluate.py` — faithful scoring (mirrors `verify_network`).
- `solve.py` / `solve_all.py` — per-task tiering + the full driver.
