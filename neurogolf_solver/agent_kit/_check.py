import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ng_utils_shim import tasks_dir
tdir = tasks_dir()

def load_all(t):
    ex = json.load(open(tdir / f"task{t:03d}.json"))
    out={}
    for s in ("train","test","arc-gen"):
        out[s]=[(np.array(e["input"],int), np.array(e["output"],int)) for e in ex.get(s,[])]
    return out

def check(t, rule):
    data=load_all(t)
    res={}
    fails=[]
    for s,prs in data.items():
        ok=0; tot=0
        for j,(a,b) in enumerate(prs):
            tot+=1
            try:
                pred=rule(a)
            except Exception as ex:
                pred=None
            if pred is not None and pred.shape==b.shape and np.array_equal(pred,b):
                ok+=1
            else:
                if len(fails)<5: fails.append((s,j))
        res[s]=(ok,tot)
    return res, fails
