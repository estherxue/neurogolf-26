# Family-builder context (read this first)

Goal: implement a NEW family of exact, generalizing ARC->ONNX solvers for the NeuroGolf 2026
competition. You write ONE python module exposing `candidates(examples)` that yields
`(name, onnx_model)` for tasks your family matches. A shared harness scores it against the
real grader over all 400 tasks. Keep solutions cheap (cost = params + intermediate memory).

## The I/O contract (FIXED — do not change)
- Every ONNX model has one input `"input"` and one output `"output"`, both `FLOAT[1,10,30,30]`
  (one-hot: channel = color 0..9, grid placed TOP-LEFT, zero-padded to 30x30).
- The grader thresholds `(output > 0)` per channel and compares for EXACT equality to the
  one-hot target. At each real cell exactly the right channel must be >0 and all others <=0;
  padding cells must be <=0 on all channels.

## Hard constraints (verified in neurogolf_utils.py)
- opset/IR = 10; operator domain must be "" / "ai.onnx".
- Static shapes only (positive dims). Exactly one input + one output.
- BANNED ops: Loop, Scan, NonZero, Unique, Script, Function, Compress, any Sequence* op,
  subgraphs/functions. File <= 1.44MB.
- cost = (#params in initializers/Constant) + (intermediate-tensor memory bytes). points =
  max(1, 25 - ln(max(1,cost))). input/output tensors are FREE (excluded from memory).
  A full [1,10,30,30] float intermediate = 36000 bytes. Fewer/smaller intermediates = more points.

## The PADDING GOTCHA (critical)
Grids are anchored TOP-LEFT and zero-padded to 30x30, and grid sizes VARY per example. So an
op over the full 30x30 tensor must keep content anchored at the origin (0,0). Pointwise ops
(recolor) and `Transpose` (perm [0,1,3,2]) preserve the origin. A naive horizontal flip /
rotate / translate sends content to the far edge -> WRONG for grids smaller than 30. Anything
whose correct position depends on the (variable, data-dependent) grid size is generally NOT
expressible with a static graph. Design families that are origin-anchored, or rely on the
harness to reject the ones that aren't. Sizes can change (e.g. 3x3->9x9) as long as the
output content stays top-left.

## How to build a model
```python
import onnx
from onnx import helper as oh
from builders import _model            # _model(nodes, initializers=()) -> ModelProto
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, CHANNELS, HEIGHT, WIDTH, ng
INT64 = onnx.TensorProto.INT64
# ng.convert_to_numpy({"input":grid,"output":grid}) -> {"input":[1,10,30,30],"output":...}
```
See ../builders.py for working examples: identity, transpose, recolor_gather/conv, flip_*,
rot180, translate, upscale (Resize+crop), downscale (strided Slice+Pad), constant.

## Detection
Infer your rule's parameters from the train+test pairs (numpy, raw variable-size grids), then
emit the ONNX. The harness validates EXACTNESS on all train+test+arc-gen (the grader's gate),
so over-propose freely — wrong guesses are rejected. To AVOID OVERFITTING, prefer rules
inferred from structure (not memorized); the grader also uses a held-out private set.

## Your module shape
```python
def candidates(examples):
    # examples = {"train":[{"input":grid,"output":grid},...], "test":[...], "arc-gen":[...]}
    prs = [(np.array(e["input"]), np.array(e["output"])) for e in examples["train"]+examples["test"]]
    # ... detect; if matched, yield ("myfamily_param", build_model(param))
```

## Test command
```bash
VENV=/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26--claude-worktrees-kaggle-agent-harness/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad/ngvenv/bin/python
cd <agent_kit dir>
$VENV family_test.py <your_module_name>     # prints SOLVED <n> TOTAL <pts> + per-task list
```
Report back: the module code, and the SOLVED count + TOTAL points + which tasks.
