"""family_scrk_3 -- slice U[3::5] = tasks [23,76,96,138,173,208,255,363]

SOLVED: task 208 (box-duplication). Exact numpy rule validated 266/266 and an
opset-10 ONNX construction (below) validated 266/266 through the grader.

  208 (FIXED SxS, S=21 here): the input has one hollow rectangle drawn in the
       rarest non-zero color k, whose interior is solid-0; elsewhere there is a
       second solid-0 rectangle of identical size sitting in noise. Draw a copy
       of the box framing the second rectangle.
       ONNX: crop to SxS; box color = argmin of nonzero channel-sums; box bbox
       -> interior size (ih,iw) as scalars; gated cumulative erosion of the
       0-mask by ih (rows) then iw (cols) yields the two block corners; drop the
       template corner (box_bbox_topleft+1); the remaining corner gives the
       target; translate the box mask by a data-dependent (dr,dc) via shift
       matrices built with Less(|P-Q-d|,0.5); paint box-color one-hot there.

The other 7 tasks in this slice were fully analyzed but have no compact, exact,
generalizing, banned-op-free opset-10 ONNX:
  255 fill eroded interior of large blank occluder RECTANGLES with 3 (needs
      maximal-rectangle / CC detection; local erosion over-fills noise gaps).
  363 replicate an in-grid seed 2-shape at every location whose local 5/0 context
      matches the seed -> stamp differs per grid (diamond / bar / block) => not a
      static function of the input.
  23  recolor each 5 to 2 (length-3 lines) or 8 (2x2 blocks) by a GLOBAL jigsaw
      decomposition of the 5-shape; cells inside a 2x2 square can still be 2 ->
      non-local; a 25-bit 5x5 gather is far too large.
  76  per-object line/sequence extension (data-dependent per object).
  173 stamp a template (read from the grid, keyed by its center color) centered on
      each isolated marker cell -> data-dependent kernels.
  96,138 VAR-size crops (output 7/9/11.. varies) -> forbidden by ANTI-OVERFIT.
"""
import numpy as np
import onnx
from onnx import helper as oh
from ng_utils_shim import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

INT64 = onnx.TensorProto.INT64
FLOAT = onnx.TensorProto.FLOAT


def _rule_208(i):
    """Exact numpy rule (validated 266/266) — also the family detector."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    i = np.asarray(i)
    H, W = i.shape
    vals, cnts = np.unique(i, return_counts=True)
    nz = [(v, c) for v, c in zip(vals, cnts) if v != 0]
    if not nz:
        return None
    box = min(nz, key=lambda x: x[1])[0]
    bc = np.argwhere(i == box)
    br0, br1 = bc[:, 0].min(), bc[:, 0].max()
    bc0, bc1 = bc[:, 1].min(), bc[:, 1].max()
    ih, iw = br1 - br0 - 1, bc1 - bc0 - 1
    if ih < 1 or iw < 1 or ih > H or iw > W:
        return None
    Z = (i == 0).astype(int)
    s = sw(Z, (ih, iw)).sum(axis=(2, 3))
    cands = [(a, b) for a in range(s.shape[0]) for b in range(s.shape[1])
             if s[a, b] == ih * iw]
    tgt = [(a, b) for a, b in cands
           if not (br0 < a and a + ih - 1 < br1 and bc0 < b and b + iw - 1 < bc1)]
    if len(tgt) != 1:
        return None
    a, b = tgt[0]
    r0, r1, c0, c1 = a - 1, a + ih, b - 1, b + iw
    if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
        return None
    o = i.copy()
    o[r0, c0:c1 + 1] = box
    o[r1, c0:c1 + 1] = box
    o[r0:r1 + 1, c0] = box
    o[r0:r1 + 1, c1] = box
    return o


def _build_208(S):
    K = S - 2
    nodes = []
    inits = []

    def C(name, arr, dt=FLOAT):
        arr = np.asarray(arr)
        inits.append(oh.make_tensor(name, dt, list(arr.shape), arr.ravel().tolist()))
        return name

    def N(op, i, o, **kw):
        nodes.append(oh.make_node(op, i if isinstance(i, list) else [i], [o], **kw))
        return o

    C('half', [0.5]); C('one', [1.0]); C('big', [1e9])
    ar = np.arange(S, dtype=np.float32)
    C('ridx', ar); C('cidx', ar)
    C('BIGS', np.full(S, 1e9, np.float32)); C('NEGS', np.full(S, -1.0, np.float32))
    C('ar10', np.arange(10, dtype=np.float32))
    Pg = np.tile(ar.reshape(S, 1), (1, S)); Qg = np.tile(ar.reshape(1, S), (S, 1))
    C('Pg', Pg); C('Qg', Qg); C('Rg', Pg.reshape(1, 1, S, S)); C('Cg', Qg.reshape(1, 1, S, S))
    Su = np.zeros((S, S), np.float32)
    for r in range(S - 1):
        Su[r, r + 1] = 1
    C('Su', Su)
    Sc = np.zeros((S, S), np.float32)
    for c in range(S - 1):
        Sc[c + 1, c] = 1
    C('Sc', Sc)
    C('slc_s', [0, 0], INT64); C('slc_e', [S, S], INT64); C('slc_a', [2, 3], INT64)
    C('c0s', [0], INT64); C('c0e', [1], INT64); C('c0a', [1], INT64)
    C('sh_S1', [S, 1], INT64); C('sh_1S', [1, S], INT64); C('sh_11SS', [1, 1, S, S], INT64)
    C('sh_SS', [S, S], INT64); C('sh_11011', [1, 10, 1, 1], INT64)

    N('Slice', ['input', 'slc_s', 'slc_e', 'slc_a'], 'X')
    N('Slice', ['X', 'c0s', 'c0e', 'c0a'], 'Z')
    N('ReduceSum', 'X', 'cs', axes=[0, 2, 3], keepdims=0)
    N('Less', ['cs', 'half'], 'cs_lt'); N('Where', ['cs_lt', 'big', 'cs'], 'cs2')
    N('ArgMin', 'cs2', 'boxi', axis=0, keepdims=1); N('Cast', 'boxi', 'boxf', to=FLOAT)
    N('Gather', ['X', 'boxi'], 'Bm', axis=1)
    N('ReduceMax', 'Bm', 'rr', axes=[0, 1, 3], keepdims=0)
    N('ReduceMax', 'Bm', 'cc', axes=[0, 1, 2], keepdims=0)
    N('Greater', ['rr', 'half'], 'rrb'); N('Greater', ['cc', 'half'], 'ccb')
    N('Where', ['rrb', 'ridx', 'BIGS'], 'rr_lo'); N('ReduceMin', 'rr_lo', 'br0', keepdims=0)
    N('Where', ['rrb', 'ridx', 'NEGS'], 'rr_hi'); N('ReduceMax', 'rr_hi', 'br1', keepdims=0)
    N('Where', ['ccb', 'cidx', 'BIGS'], 'cc_lo'); N('ReduceMin', 'cc_lo', 'bc0', keepdims=0)
    N('Where', ['ccb', 'cidx', 'NEGS'], 'cc_hi'); N('ReduceMax', 'cc_hi', 'bc1', keepdims=0)
    N('Sub', ['br1', 'br0'], 'ihh'); N('Sub', ['ihh', 'one'], 'ih')
    N('Sub', ['bc1', 'bc0'], 'iww'); N('Sub', ['iww', 'one'], 'iw')

    E = 'Z'; prev = 'Z'
    for j in range(1, K + 1):
        N('MatMul', ['Su', prev], f'Zj{j}'); prev = f'Zj{j}'
        N('Greater', ['ih', C(f'jf{j}', [float(j)])], f'gjb{j}'); N('Cast', f'gjb{j}', f'gj{j}', to=FLOAT)
        N('Sub', ['one', f'Zj{j}'], f'oz{j}'); N('Mul', [f'gj{j}', f'oz{j}'], f'gm{j}'); N('Sub', ['one', f'gm{j}'], f'tj{j}')
        N('Mul', [E, f'tj{j}'], f'erow{j}'); E = f'erow{j}'
    Erow = E
    E = Erow; prev = Erow
    for k in range(1, K + 1):
        N('MatMul', [prev, 'Sc'], f'Ek{k}'); prev = f'Ek{k}'
        N('Greater', ['iw', C(f'kf{k}', [float(k)])], f'hkb{k}'); N('Cast', f'hkb{k}', f'hk{k}', to=FLOAT)
        N('Sub', ['one', f'Ek{k}'], f'oe{k}'); N('Mul', [f'hk{k}', f'oe{k}'], f'hm{k}'); N('Sub', ['one', f'hm{k}'], f'ck{k}')
        N('Mul', [E, f'ck{k}'], f'ecol{k}'); E = f'ecol{k}'
    Efull = E

    N('Add', ['br0', 'one'], 'tr0'); N('Sub', ['ridx', 'tr0'], 'trd'); N('Abs', 'trd', 'tra'); N('Less', ['tra', 'half'], 'trb'); N('Cast', 'trb', 'tr', to=FLOAT)
    N('Add', ['bc0', 'one'], 'tc0'); N('Sub', ['cidx', 'tc0'], 'tcd'); N('Abs', 'tcd', 'tca'); N('Less', ['tca', 'half'], 'tcb'); N('Cast', 'tcb', 'tc', to=FLOAT)
    N('Reshape', ['tr', 'sh_S1'], 'trR'); N('Reshape', ['tc', 'sh_1S'], 'tcR'); N('Mul', ['trR', 'tcR'], 'Tm2d')
    N('Reshape', ['Tm2d', 'sh_11SS'], 'Tm'); N('Sub', ['one', 'Tm'], 'nTm'); N('Mul', [Efull, 'nTm'], 'E2')
    N('Mul', ['E2', 'Rg'], 'E2r'); N('ReduceSum', 'E2r', 'ta', keepdims=0)
    N('Mul', ['E2', 'Cg'], 'E2c'); N('ReduceSum', 'E2c', 'tb', keepdims=0)
    N('Sub', ['ta', 'one'], 'ta1'); N('Sub', ['ta1', 'br0'], 'dr')
    N('Sub', ['tb', 'one'], 'tb1'); N('Sub', ['tb1', 'bc0'], 'dc')
    N('Sub', ['Pg', 'Qg'], 'PQ')
    N('Sub', ['PQ', 'dr'], 'PQr'); N('Abs', 'PQr', 'PQra'); N('Less', ['PQra', 'half'], 'Sdrb'); N('Cast', 'Sdrb', 'Sdr', to=FLOAT)
    N('Add', ['PQ', 'dc'], 'PQc'); N('Abs', 'PQc', 'PQca'); N('Less', ['PQca', 'half'], 'Sdcb'); N('Cast', 'Sdcb', 'Sdc', to=FLOAT)
    N('Reshape', ['Bm', 'sh_SS'], 'Bm2d')
    N('MatMul', ['Sdr', 'Bm2d'], 'TBa'); N('MatMul', ['TBa', 'Sdc'], 'TB2d'); N('Reshape', ['TB2d', 'sh_11SS'], 'TB')
    N('Sub', ['ar10', 'boxf'], 'bcd'); N('Abs', 'bcd', 'bca'); N('Less', ['bca', 'half'], 'bcb'); N('Cast', 'bcb', 'bcoh', to=FLOAT)
    N('Reshape', ['bcoh', 'sh_11011'], 'bcohR')
    N('Sub', ['one', 'TB'], 'nTB'); N('Mul', ['X', 'nTB'], 'keep'); N('Mul', ['bcohR', 'TB'], 'paint'); N('Add', ['keep', 'paint'], 'O')
    N('Pad', 'O', 'output', mode='constant', pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], value=0.0)
    g = oh.make_graph(nodes, 'g208',
                      [oh.make_tensor_value_info('input', DATA_TYPE, GRID_SHAPE)],
                      [oh.make_tensor_value_info('output', DATA_TYPE, GRID_SHAPE)], inits)
    return oh.make_model(g, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def candidates(examples):
    try:
        prs = [(np.array(e['input']), np.array(e['output']))
               for e in examples['train'] + examples['test']]
    except Exception:
        return []
    if not prs:
        return []
    shapes = set(i.shape for i, _ in prs) | set(o.shape for _, o in prs)
    if len(shapes) != 1:
        return []
    H, W = prs[0][0].shape
    if H != W or H < 5 or H > 30:
        return []
    # detector: exact 208 rule on every train+test pair
    for i, o in prs:
        p = _rule_208(i)
        if p is None or not np.array_equal(p, o):
            return []
    return [("scrk3_208box", _build_208(H))]
