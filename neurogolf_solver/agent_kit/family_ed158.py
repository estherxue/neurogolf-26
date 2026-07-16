"""family_ed158 -- ENRICHED-ARSENAL rebuild of task158 (hash 6aa20dc0), candidate "ed158".

WHAT THIS TASK IS
-----------------
A reference 3x3 diagonally-symmetric sprite R is drawn once (mag 1, fully rendered);
every other sprite is shown only as its two diagonal corner blocks (solid m x m
squares of R's two corner colors, at diagonal offset 2m). Output renders each sprite
fully: F(R) upscaled by m (nearest-neighbour), F = the flip whose corners match the
two shown markers, placed at the 3m x 3m square. Grids are FLOAT[1,10,30,30] one-hot,
real content <= 26 rows x 25 cols (generator: width 15-25, height=width+/-1).

DISSECTION OF THE INCUMBENT (out_blend10/onnx/task158.onnx, "gw2f16_158|pool70")
--------------------------------------------------------------------------------
70 nodes, opset 18. Static per-tensor bytes (uint8=1, bool=1, int8=1, f32=4, i64=8):
  MEM 26178, PARAMS 2305, COST 28483 -> report.json claims 14.74 pts.
Pipeline (already uses the "enriched arsenal"):
  * color grid  : 1x1 Conv (f32) then Cast->u8, cropped 30x30->26x25 via NEGATIVE
                  Conv pads [0,0,-4,-5].
  * nonbg mask  : Equal(bg) -> Where -> Min(.,1)  [binary clamp].
  * template R  : two 3x3 diagonal QLinearConv detectors (int32 bias) + 4 shifted
                  corner Slices + corner-inequality Where-filters -> anchor map ->
                  ArgMax(row)/Gather/ArgMax(col) -> DYNAMIC Slice of the 3x3 patch.
  * orientation : Gather R's corners, choose base_u8 = R or hflip(R), colors a/b,
                  fill color, base_fill mask (int8).
  * stamping    : ab_u8 (2-ch: ==color_a / ==color_b); per mag in {1,2,3} a
                  QLinearConv corner-pair matched filter (w_pair 5x5/8x8/11x11, int32
                  bias = RUNTIME-BIAS BINARY DETECT) -> 4-flip pair map -> QLinearConv
                  stamp (stamp_w = Gather(base_fill, stamp_idx)) = ADJOINT-CONV PAINT.
  * composite   : Max(fill1,fill2,fill3) -> Greater -> Where(fill, color) -> Pad ->
                  Equal(palette) -> one-hot output.

WHY THE INCUMBENT SCORES 0 ON THE MANDATED STACK (onnx 1.22 / ORT 1.23.2)
-------------------------------------------------------------------------
It uses THREE constructs the local grader cannot execute or measure:
  (1) Min(uint8)  node  -> ORT 1.23.2: NOT_IMPLEMENTED  (session won't load)
  (2) Max(uint8)  node  -> ORT 1.23.2: NOT_IMPLEMENTED  (session won't load)
  (3) negative-pad Conv -> onnx.shape_inference(strict) raises "pads must not
                           contain negative values" -> calculate_memory throws
  (4) DYNAMIC Slice (_fold_patch, runtime start/end) -> strict inference yields
                           dim_param -> calculate_memory returns None (unmeasurable)
So the deployed incumbent fails HARD GATE (a) AND is unscoreable: its real,
reproducible grader score is 0.0.  (report.json's 14.74 is from a legacy stack.)

THE REBUILD (ed158) -- MINIMAL VALID EQUIVALENT
-----------------------------------------------
Byte-identical outputs to the incumbent's logic (verified DIFF=0 over 266 provided
+ 3000 fresh generator samples), with the four illegal constructs replaced by
grader-legal equivalents drawn from the allowed op/dtype set:
  (1) Min(nonbg_values,1)          -> Clip(nonbg_values, 0, 1)              [u8-legal]
  (2) Max(f1,f2,f3)                -> Add(Add(f1,f2),f3)  (fills disjoint)  [u8-legal]
  (3) negative-pad color Conv      -> Conv kernel [5,6] pad 0 (color at [0,0]) so the
                                      valid conv itself crops 30x30->26x25   [+290 params]
  (4) dynamic Slice patch          -> two Gather(axis=2 then 3) with RUNTIME-VALUE,
                                      STATIC-SHAPE index tensors -> static [1,1,3,3]
                                      (enriched-arsenal item 3, kills the dim_param)
Result (real grader): ok=True, agi=(4,0), gen=(262,0), MEM 26919, PARAMS 2598,
COST 29517, POINTS 14.707. Passes: loads ORT 1.23.2; opt-invariant (0/500 across
DISABLE/BASIC/ENABLE_ALL); node_count 72<=600; plain 0.68 ms median<=5 ms;
trace(20) 1.11 MB<=6 MB.

FLOOR NOTE (could NOT strictly beat the stale 14.74)
----------------------------------------------------
The incumbent architecture is already fully arsenal-optimized (runtime-bias
QLinearConv binary detect, adjoint-conv stamping, static Gather corners, int32 bias,
u8/i8/bool throughout, f32 only for the color contraction) and provably tight:
dropping the corner-equality filters changed 465/3266 outputs; shrinking the
pair-conv pads changed 672/3266; the 26x25 crop is exactly the generator's max grid.
Every remaining fat tensor is structural: color_f_crop (f32 2600) is the minimum
color contraction (Conv must be float); the three w_pair kernels are size-forced by
the 2m corner offsets; the 4 pair-map channels encode per-sprite orientation.
Making the incumbent MERELY VALID already costs +1034 bytes over its (fictional)
28483: +650 (Add temp, no u8 Max), +290 params (valid crop Conv, no negative pad),
+94 (static-shape patch Gathers, no dynamic Slice). Those are exactly the ops that
made the incumbent unscoreable, so the +1034 is irreducible. Hence 14.71 is the true
valid floor: it does NOT exceed the unreproducible 14.74, but it strictly beats the
incumbent's actual grader score (0.0) and is the only measurable, loading model.
The single 1/3000-fresh disagreement with the ground truth is an INHERENT
multi-marker pairing ambiguity (a corner block validly pairs two ways); ed158
reproduces the incumbent's detection there exactly, as required.
"""

import base64, gzip, hashlib, json
import onnx

_FP = "c48a662904d59970f46d0d34b63a0424"
_BLOB = (
    "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"
)


def _model():
    return onnx.load_model_from_string(gzip.decompress(base64.b64decode(_BLOB)))


def _fp(train):
    return hashlib.md5(json.dumps(train, sort_keys=True).encode()).hexdigest()


def candidates(example):
    """Return the ed158 model only for task158's train pairs (gated, single-task)."""
    if _fp(example.get("train", [])) != _FP:
        return []
    return [("ed158", _model())]
