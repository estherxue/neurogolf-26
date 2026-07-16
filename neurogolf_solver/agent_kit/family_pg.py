"""PARAMS-GOLF (low-rank factorization of fat initializers) — family_pg.

Goal: for tasks whose cost is params-dominated AND whose fat initializer feeds a
MatMul/Einsum/Gemm, replace a genuinely low-rank W (h*w elems) with factors A,B
(r*(h+w) elems) folded into the SAME contraction, so the factor matrices stay
free initializers and NO new counted intermediate is introduced.

Break-even: factoring saves iff  r*(h+w) < h*w.

Dissection of every candidate in out_blend6/onnx (see verdicts below) found ZERO
admissible wins — candidates() returns []. Each is skipped for a concrete reason:

  t123  canvas_index [30,30] rank11 -> Gather (integer index LUT).
        rank11 WOULD save 240 params on paper (900->660), BUT the consumer is a
        Gather that uses W as int32 INDICES, not a contraction. Factoring means
        materializing A@B into a NAMED [30,30]=900-elem intermediate -> net LOSS.
        This is exactly the Gather-LUT trap. SKIP.

  t398  KEY [30,30] rank26 -> Gather (integer index LUT).
        Both fatal: (a) consumer is a Gather LUT (same trap as t123), and
        (b) rank26 factoring is bigger anyway (26*60=1560 > 900). SKIP.

  t336  B_i8 [9,52] rank9 -> QLinearMatMul.
        Consumer IS a MatMul, but B is already the second factor of a bottleneck
        MLP (A_i8[14,9] @ ... hidden9 ... @ B_i8[9,52]); it is FULL rank 9, so
        factoring is bigger (9*61=549 > 468). The graph is already low-rank. SKIP.

  t264  S0,S1,S2 [14,30] rank14 -> multi-operand Einsum.
        Consumer IS an Einsum, but each selection matrix is FULL rank 14, so
        factoring is bigger (14*44=616 > 420). No low-rank structure. SKIP.

  t305  channel_row_u64 [1,10,30,1] -> BitwiseAnd.
        Consumer is a bitmask AND, not a contraction; a low-rank factor would
        need a [1,r,30,...] intermediate that costs more than saved. SKIP.

  t275  block_route/mod_route/fold [2,30,4] -> Einsum.
        Consumer IS an Einsum, but each [30,4] z-slice is near/full column rank
        (min dim 4; measured ranks 3 and 4). A uniform tensor factor needs r=4 ->
        4*(30+4)*2 = 272 > 240 (WORSE); the one rank-3 slice can't be split out
        of the stacked [2,30,4] operand without differing r per z. No clean, net
        win. SKIP.

Verified against out_blend6/onnx with onnx + numpy.linalg.matrix_rank and the
break-even inequality above.
"""


def candidates(example):
    # No admissible low-rank params-golf win among the dissected candidates.
    return []
