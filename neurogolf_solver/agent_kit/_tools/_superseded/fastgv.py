"""Fast bulk gen-validate (bare ORT, no profiling). N fresh samples per task; report >=1 fail + rate.
usage: python fastgv.py <out_dir> [N] [skip_csv]"""
import sys, os, json, random, pathlib
import numpy as np
SC="/private/tmp/claude-501/-Users-xingyuanxue1122-Documents-coding-neurogolf-26--claude-worktrees-kaggle-agent-harness/f26477d2-2e56-461c-9fe3-1ac499bf563f/scratchpad"
sys.path.insert(0, SC+"/arc-gen")
KIT="/Users/xingyuanxue1122/Documents/coding/neurogolf-26/.claude/worktrees/kaggle-agent-harness/neurogolf_solver/agent_kit"
import task_list, onnx, onnxruntime as ort
tl=task_list.task_list(); hm=json.load(open(KIT+"/task_hash_map.json"))
OUT=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 500
SKIP=set(int(x) for x in sys.argv[3].split(",")) if len(sys.argv)>3 and sys.argv[3] else set()
rep=json.load(open(os.path.join(OUT,"report.json")))
def to_np(g):
    b=np.zeros((1,10,30,30),np.float32)
    for r,row in enumerate(g):
        for c,v in enumerate(row):
            if 0<=v<10 and r<30 and c<30: b[0][v][r][c]=1.0
    return b
def ok_shape(e): return len(e["input"])<=30 and len(e["input"][0])<=30 and len(e["output"])<=30 and len(e["output"][0])<=30
bad={}; loaderr=[]; done=0
for t in rep["tasks"]:
    tn=t["task"]
    if not t.get("name") or tn in SKIP: continue
    fp=pathlib.Path(OUT)/"onnx"/f"task{tn:03d}.onnx"
    if not fp.exists(): continue
    try:
        so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess=ort.InferenceSession(onnx.load(str(fp)).SerializeToString(),so)
    except Exception:
        loaderr.append(tn); print(f"task{tn:03d} LOADERR [{t.get('name')}] {t.get('points'):.2f}",flush=True); continue
    gen=tl[hm[str(tn)]][0]; random.seed(20260708+tn); fails=n=tries=0
    while n<N and tries<N*6:
        tries+=1
        try: e=gen()
        except Exception: continue
        if not ok_shape(e): continue
        n+=1
        try: out=(sess.run(["output"],{"input":to_np(e["input"])})[0]>0.0).astype(np.float32)
        except Exception: fails+=1; continue
        if not np.array_equal(out,to_np(e["output"])): fails+=1
    if fails>0:
        bad[tn]=[round(fails/max(n,1),4),t.get("name"),round(t.get("points",0),2)]
        print(f"task{tn:03d} FAIL {fails}/{n} [{t.get('name')}] {t.get('points'):.2f}",flush=True)
    done+=1
    if done%50==0: print(f"...scanned {done}",flush=True)
json.dump({"bad":bad,"loaderr":loaderr}, open(SC+"/fastgv_"+os.path.basename(OUT)+".json","w"))
riskpts=sum(v[2] for v in bad.values())
print(f"FASTGV {OUT} N={N}: {len(bad)} fail-tasks (riskpts {riskpts:.1f}), {len(loaderr)} loaderr")
