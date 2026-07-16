"""family_t2_92 -- task092 "connect same-color endpoint pairs" (flat/tall sticks).

Transform (verified exact on train+test and 3000 fresh generator samples):
  * Each nonzero color occurs exactly twice (a stick's two endpoints).
  * If the pair shares a row  -> horizontal fill between the two columns.
  * If the pair shares a column-> vertical  fill between the two rows.
  * At a crossing the vertical stick wins.

NOTE ON MEMORY (why this does NOT beat the out_blend12 incumbent):
  The incumbent (out_blend12/onnx/task092.onnx, mem+params=6553, 16.21 pts) is the
  reduce-to-per-color-scalar + compact TopK/scatter design: it keeps ALL coordinate
  math in [1,10]/[1,5] space and only ever materialises 3 uint8 [1,1,30,30] scatter
  canvases (dominant tensor: [1,1,30,30] uint8 = 900 B, x3).  That is memory-optimal
  for this task.  This module implements the same transform with an explicit
  per-row / per-col range-fill; every range test needs two full-grid compares, so it
  materialises ~14 full-grid [1,1,30,30] tensors and cannot get below the incumbent.
  It is kept only as a self-contained, gated-exact reference candidate.  Task is
  FLOORED with the enriched arsenal (QLinearConv-binary / adjoint-conv / static-gather
  do not apply to per-color coordinate extraction with vertical-over-horizontal
  precedence).
"""
import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP, numpy_helper as nh


def _build():
    nodes, inits = [], []

    def C(name, arr):
        inits.append(nh.from_array(np.asarray(arr), name))

    def add(op, ins, outs, **kw):
        nodes.append(H.make_node(op, ins, outs, **kw))
        return outs[0]

    C('idx30', np.arange(30, dtype=np.float32))
    C('idx30sq', (np.arange(30, dtype=np.float32) ** 2))
    C('cvec', np.arange(10, dtype=np.float32))
    C('two', np.float32(2)); C('half', np.float32(0.5)); C('zero', np.float32(0))
    C('twoI', np.float32(2))

    # per-color reductions [1,10]
    add('Einsum', ['input'], ['N'], equation='bchw->bc')
    add('Einsum', ['input', 'idx30'], ['Sc'], equation='bchw,w->bc')
    add('Einsum', ['input', 'idx30'], ['Sr'], equation='bchw,h->bc')
    add('Einsum', ['input', 'idx30sq'], ['Sc2'], equation='bchw,w->bc')
    add('Einsum', ['input', 'idx30sq'], ['Sr2'], equation='bchw,h->bc')

    def span(S, S2, p):
        add('Mul', [S2, 'two'], [p + 'a'])
        add('Mul', [S, S], [p + 'b'])
        add('Sub', [p + 'a', p + 'b'], [p + 'c0'])
        add('Relu', [p + 'c0'], [p + 'c'])          # clamp >=0 (avoid nan for non-pairs)
        add('Sqrt', [p + 'c'], [p + 'span'])
        add('Sub', [S, p + 'span'], [p + 'd']); add('Mul', [p + 'd', 'half'], [p + 'min'])
        add('Add', [S, p + 'span'], [p + 'e']); add('Mul', [p + 'e', 'half'], [p + 'max'])
        return p + 'span', p + 'min', p + 'max'

    rspan, rmin, rmax = span('Sr', 'Sr2', 'r')
    cspan, cmin, cmax = span('Sc', 'Sc2', 'c')

    add('Equal', ['N', 'twoI'], ['isN2'])
    add('Equal', [rspan, 'zero'], ['rsp0'])
    add('Equal', [cspan, 'zero'], ['csp0'])
    add('And', ['isN2', 'rsp0'], ['isH'])
    add('And', ['isN2', 'csp0'], ['isV'])

    C('ridx', np.arange(30, dtype=np.float32).reshape(1, 1, 30))
    C('ax2', np.array([2], dtype=np.int64))
    add('Unsqueeze', [rmin, 'ax2'], ['rminU'])
    add('Equal', ['rminU', 'ridx'], ['RowOH']); add('Cast', ['RowOH'], ['RowOHf'], to=TP.FLOAT)
    add('Unsqueeze', [cmin, 'ax2'], ['cminU'])
    add('Equal', ['cminU', 'ridx'], ['ColOH']); add('Cast', ['ColOH'], ['ColOHf'], to=TP.FLOAT)

    add('Cast', ['isH'], ['isHf'], to=TP.FLOAT)
    add('Cast', ['isV'], ['isVf'], to=TP.FLOAT)
    add('Mul', ['isHf', 'cvec'], ['wHc'])
    add('Mul', ['isHf', cmin], ['wHL']); add('Mul', ['isHf', cmax], ['wHR'])
    add('Mul', ['isVf', 'cvec'], ['wVc'])
    add('Mul', ['isVf', rmin], ['wVD']); add('Mul', ['isVf', rmax], ['wVU'])

    def proj(w, oh, out):
        add('Einsum', [w, oh], [out], equation='bc,bci->bi')

    proj('wHc', 'RowOHf', 'Hc'); proj('wHL', 'RowOHf', 'HL'); proj('wHR', 'RowOHf', 'HR')
    proj('wVc', 'ColOHf', 'Vc'); proj('wVD', 'ColOHf', 'VD'); proj('wVU', 'ColOHf', 'VU')

    C('shpRow', np.array([1, 1, 30, 1], dtype=np.int64))
    C('shpCol', np.array([1, 1, 1, 30], dtype=np.int64))
    for t in ['Hc', 'HL', 'HR']:
        add('Reshape', [t, 'shpRow'], [t + 'r'])
    for t in ['Vc', 'VD', 'VU']:
        add('Reshape', [t, 'shpCol'], [t + 'c'])

    C('colI', np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))
    C('rowI', np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))

    add('GreaterOrEqual', ['colI', 'HLr'], ['geH'])
    add('LessOrEqual', ['colI', 'HRr'], ['leH'])
    add('And', ['geH', 'leH'], ['inH']); add('Cast', ['inH'], ['inHf'], to=TP.FLOAT)
    add('Mul', ['Hcr', 'inHf'], ['Hgrid'])
    add('GreaterOrEqual', ['rowI', 'VDc'], ['geV'])
    add('LessOrEqual', ['rowI', 'VUc'], ['leV'])
    add('And', ['geV', 'leV'], ['inV']); add('Cast', ['inV'], ['inVf'], to=TP.FLOAT)
    add('Mul', ['Vcc', 'inVf'], ['Vgrid'])

    add('Greater', ['Vgrid', 'zero'], ['vpos'])
    add('Where', ['vpos', 'Vgrid', 'Hgrid'], ['comb'])
    add('ReduceMax', ['input'], ['pres'], axes=[1], keepdims=1)
    add('Greater', ['pres', 'zero'], ['ingrid'])
    C('sent', np.float32(30))
    add('Where', ['ingrid', 'comb', 'sent'], ['finalf'])
    C('colors', np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    add('Equal', ['finalf', 'colors'], ['output'])

    inp = H.make_tensor_value_info('input', TP.FLOAT, [1, 10, 30, 30])
    out = H.make_tensor_value_info('output', TP.BOOL, [1, 10, 30, 30])
    g = H.make_graph(nodes, 't92', [inp], [out], inits)
    m = H.make_model(g, opset_imports=[H.make_opsetid('', 16)])
    m.ir_version = 10
    return m


def candidates(ex):
    return [("t2_92", _build())]
