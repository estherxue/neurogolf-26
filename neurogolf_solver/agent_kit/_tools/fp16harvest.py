"""Safe fp16 harvester: per task with f32 intermediates, convert internals to f16 (keep_io_types f32),
then VERIFY exact on all local + N fresh generator samples AND strictly cheaper. Keep only verified wins.
The fresh-sample gate rejects any fp16>2048 precision loss (deterministic -> caught) = no overfit."""
import sys, os, json, random, pathlib, math
import numpy as np, onnx, onnxruntime as ort
from onnx import shape_inference
from onnxconverter_common.float16 import convert_float_to_float16
SC="/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26--claude-worktrees-kaggle-agent-harness/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"
sys.path.insert(0, SC+"/arc-gen")
KIT="/Users/xingyuanxue1122/Documents/coding/neurogolf-26/.claude/worktrees/kaggle-agent-harness/neurogolf_solver/agent_kit"
import task_list
tl=task_list.task_list(); hm=json.load(open(KIT+"/task_hash_map.json"))
BASE=pathlib.Path(KIT)/"out_blend6"; DST=pathlib.Path(KIT)/"out_fp16h"
(DST/"onnx").mkdir(parents=True, exist_ok=True)
DTB={1:4,10:2,7:8,6:4,9:1,2:1,3:1,12:4,5:2,11:8,13:8}
N=int(sys.argv[2]) if len(sys.argv)>2 else 800
TASKS=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 and sys.argv[1]!="ALL" else None
def to_np(g):
    b=np.zeros((1,10,30,30),np.float32)
    for r,row in enumerate(g):
        for c,v in enumerate(row):
            if 0<=v<10 and r<30 and c<30: b[0][v][r][c]=1.0
    return b
def cost(m):
    g=shape_inference.infer_shapes(m,strict_mode=False).graph
    tmap={x.name:x for x in list(g.input)+list(g.value_info)+list(g.output)}
    names=set(tmap.keys())
    for n in g.node:
        for o in n.output:
            if o: names.add(o)
    mem=0
    for name in names:
        if name in("input","output"): continue
        it=tmap.get(name)
        if it is None or not it.type.HasField("tensor_type"): return None
        el=1
        for d in it.type.tensor_type.shape.dim:
            if not d.HasField("dim_value") or d.dim_value<=0: return None
            el*=d.dim_value
        mem+=el*DTB.get(it.type.tensor_type.elem_type,4)
    par=0
    for ini in g.initializer: par+=int(np.prod(ini.dims)) if ini.dims else 1
    for n in g.node:
        if n.op_type=="Constant":
            for a in n.attribute:
                if a.name=="value": par+=int(np.prod(a.t.dims)) if a.t.dims else 1
    return max(1.0,25-math.log(max(1.0,mem+par)))
def sess(m):
    so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(m.SerializeToString() if hasattr(m,'SerializeToString') else m, so)
def exact_on(s, exs):
    for e in exs:
        if max(len(e["input"]),len(e["input"][0]),len(e["output"]),len(e["output"][0]))>30: continue
        try: o=(s.run(["output"],{"input":to_np(e["input"])})[0]>0.0).astype(np.float32)
        except Exception: return False
        if not np.array_equal(o,to_np(e["output"])): return False
    return True
rep=json.load(open(BASE/"report.json"))
for f in (BASE/"onnx").glob("task*.onnx"): (DST/"onnx"/f.name).write_bytes(f.read_bytes())
wins=0; tried=0; gain=0.0; by_task={t["task"]:dict(t) for t in rep["tasks"]}
for t in rep["tasks"]:
    tn=t["task"]
    if TASKS and tn not in TASKS: continue
    if not t.get("name"): continue
    fp=BASE/"onnx"/f"task{tn:03d}.onnx"
    m0=onnx.load(str(fp))
    # skip if not loadable locally OR no f32 intermediates
    try: s0=sess(m0)
    except Exception: continue
    c0=cost(m0)
    if c0 is None: continue
    tried+=1
    try:
        m16=convert_float_to_float16(onnx.load(str(fp)), keep_io_types=True, disable_shape_infer=False)
        s16=sess(m16)
    except Exception: continue
    c16=cost(m16)
    if c16 is None or c16<=c0+0.005: continue
    # verify: local exact + N fresh exact
    loc=json.load(open(f"{SC}/ng_data/tasks/task{tn:03d}.json"))
    localex=loc["train"]+loc["test"]+loc.get("arc-gen",[])
    if not exact_on(s16, localex): continue
    gen=tl[hm[str(tn)]][0]; random.seed(808080+tn); fails=n=tr=0
    while n<N and tr<N*6:
        tr+=1
        try: e=gen()
        except Exception: continue
        if max(len(e["input"]),len(e["input"][0]),len(e["output"]),len(e["output"][0]))>30: continue
        n+=1
        try: o=(s16.run(["output"],{"input":to_np(e["input"])})[0]>0.0).astype(np.float32)
        except Exception: fails+=1; break
        if not np.array_equal(o,to_np(e["output"])): fails+=1; break
    if fails>0: continue
    onnx.save(m16, str(DST/"onnx"/f"task{tn:03d}.onnx"))
    by_task[tn]["points"]=round(c16,2); by_task[tn]["name"]=(t.get("name") or "")+"|f16h"
    wins+=1; gain+=(c16-c0)
    print(f"  t{tn:03d}: {c0:.2f} -> {c16:.2f} (+{c16-c0:.2f}) [{n} fresh exact]",flush=True)
tasks=[by_task[i] for i in sorted(by_task)]
tot=sum(x["points"] for x in tasks if x.get("name"))
json.dump(dict(solved=sum(1 for x in tasks if x.get("name")),total_points=round(tot,3),tasks=tasks),open(DST/"report.json","w"))
import zipfile
with zipfile.ZipFile(DST/"submission.zip","w",zipfile.ZIP_DEFLATED) as z:
    for f in sorted((DST/"onnx").glob("task*.onnx")): z.write(f,f.name)
print(f"FP16HARVEST: {wins}/{tried} converted, +{gain:.2f} pts, total {tot:.2f}")
