import sys, onnx, math
from onnx import numpy_helper as nh
from onnx import shape_inference
TP=onnx.TensorProto
BYT={TP.BOOL:1,TP.FLOAT16:2,TP.FLOAT:4,TP.INT32:4,TP.INT64:8,TP.DOUBLE:8}
def analyze(path):
    m=onnx.load(path)
    m=shape_inference.infer_shapes(m)
    g=m.graph
    io={g.input[0].name, g.output[0].name}
    params=0
    for init in g.initializer:
        params+=int(nh.to_array(init).size)
    # constants
    for n in g.node:
        if n.op_type in ('Constant',):
            for a in n.attribute:
                if a.name=='value':
                    params+=int(nh.to_array(a.t).size)
    # intermediates: value_info + outputs of nodes
    vi={v.name:v for v in list(g.value_info)+list(g.output)+list(g.input)}
    mem=0
    rows=[]
    seen=set()
    for n in g.node:
        for o in n.output:
            if o in io or o in seen: continue
            seen.add(o)
            v=vi.get(o)
            if v is None:
                rows.append((o,'?','?',n.op_type)); continue
            tt=v.type.tensor_type
            dt=tt.elem_type
            dims=[d.dim_value for d in tt.shape.dim]
            el=1
            for d in dims: el*=d
            b=BYT.get(dt,4)*el
            mem+=b
            rows.append((o,dt,dims,n.op_type,b))
    cost=params+mem
    pts=max(1,25-math.log(max(1,cost)))
    print(f'{path}: params={params} mem={mem} cost={cost} pts={pts:.2f} nodes={len(g.node)}')
    for r in rows: print('  ',r)
if __name__=='__main__':
    analyze(sys.argv[1])
