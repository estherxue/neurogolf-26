"""For contested tasks: gen-screen blend3 vs blend6 version at N=800, same seed. Decide keep-high-points
(if failure is inherent = rates equal) vs keep-clean (if blend6 strictly cleaner)."""
import sys, os, json, random, pathlib
import numpy as np
SC="/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26--claude-worktrees-kaggle-agent-harness/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"
sys.path.insert(0, SC+"/arc-gen")
KIT="/Users/xingyuanxue1122/Documents/coding/neurogolf-26/.claude/worktrees/kaggle-agent-harness/neurogolf_solver/agent_kit"
import task_list, onnx, onnxruntime as ort
tl=task_list.task_list(); hm=json.load(open(KIT+"/task_hash_map.json"))
b3={t["task"]:t for t in json.load(open(KIT+"/out_blend3/report.json"))["tasks"]}
b6={t["task"]:t for t in json.load(open(KIT+"/out_blend6/report.json"))["tasks"]}
TASKS=[int(x) for x in sys.argv[1].split(",")]
N=800
def to_np(g):
    b=np.zeros((1,10,30,30),np.float32)
    for r,row in enumerate(g):
        for c,v in enumerate(row):
            if 0<=v<10 and r<30 and c<30: b[0][v][r][c]=1.0
    return b
def rate(dirn,tn,samples):
    fp=pathlib.Path(KIT)/dirn/"onnx"/f"task{tn:03d}.onnx"
    try:
        so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        s=ort.InferenceSession(onnx.load(str(fp)).SerializeToString(),so)
    except Exception: return None,"LOADERR"
    f=0
    for e in samples:
        try: o=(s.run(["output"],{"input":to_np(e["input"])})[0]>0.0).astype(np.float32)
        except Exception: f+=1; continue
        if not np.array_equal(o,to_np(e["output"])): f+=1
    return f,len(samples)
res={}
for tn in TASKS:
    gen=tl[hm[str(tn)]][0]; random.seed(31000+tn); samples=[]; tr=0
    while len(samples)<N and tr<N*6:
        tr+=1
        try: e=gen()
        except Exception: continue
        if max(len(e["input"]),len(e["input"][0]),len(e["output"]),len(e["output"][0]))<=30: samples.append(e)
    f3,n3=rate("out_blend3",tn,samples); f6,n6=rate("out_blend6",tn,samples)
    p3=round(b3[tn]["points"],2); p6=round(b6[tn]["points"],2)
    # decision
    if f3 is None: dec="b6 (b3 loaderr)"
    elif f6 is None: dec="b3 (b6 loaderr)"
    elif f3<=f6+2: dec=f"B3 (+{p3-p6:.2f}pts, same/cleaner failure)"   # inherent: take higher points
    else: dec=f"b6 (b3 overfit +{f3-f6} fails)"
    res[tn]=dict(b3_pts=p3,b6_pts=p6,b3_fail=f3,b6_fail=f6,n=n3,dec=dec)
    print(f"t{tn:03d}: b3 {p3}pts {f3}/{n3}f | b6 {p6}pts {f6}/{n6}f -> {dec}",flush=True)
json.dump(res,open(SC+"/contest.json","w"))
print("CONTEST DONE")
