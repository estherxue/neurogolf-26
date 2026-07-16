import numpy as np, sys
sys.path.insert(0,'.');sys.path.insert(0,'..')
import _check as C
from ng_utils_shim import ng

NEG=-1e9
def onehot(a):
    return ng.convert_to_numpy({"input":a.tolist(),"output":a.tolist()})["input"]

def shift_w(a,k,fill):
    out=np.full_like(a, fill)
    W=a.shape[-1]
    if k>0: out[...,k:]=a[...,:W-k]
    elif k<0: out[...,:W+k]=a[...,-k:]
    else: out=a.copy()
    return out

def excl_prefix_max_w(v, reverse):
    W=v.shape[-1]; inc=v.copy(); k=1
    while k<W:
        inc=np.maximum(inc, shift_w(inc, (-k if reverse else k), NEG)); k*=2
    return shift_w(inc, (-1 if reverse else 1), NEG)

def vproc(X9, contentMask):
    # X9: [1,9,30,30]; vertical-line processing
    nonzero=X9.sum(axis=1,keepdims=True)              # [1,1,30,30]
    row_ne=nonzero.max(axis=3,keepdims=True)          # [1,1,30,1] (0/1)
    row_empty=1-np.clip(row_ne,0,1)
    vline=np.minimum.reduce([np.maximum(row_empty, X9)],axis=0)  # placeholder
    vline=np.maximum(row_empty, X9).min(axis=2,keepdims=True)    # [1,9,1,30]
    lineCells=vline*np.clip(row_ne,0,1)               # [1,9,30,30] broadcast
    dots=X9*(1-vline)                                  # broadcast vline over rows
    leftOfLine=excl_prefix_max_w(vline, reverse=True)  # exists line col to the right
    rightOfLine=excl_prefix_max_w(vline, reverse=False)
    leftOfLine=np.clip(leftOfLine,0,1); rightOfLine=np.clip(rightOfLine,0,1)
    leftDots=(dots*leftOfLine).max(axis=3,keepdims=True)   # [1,9,30,1]
    rightDots=(dots*rightOfLine).max(axis=3,keepdims=True)
    leftAdj=shift_w(vline,-1,0.0)                      # col Lc-1
    rightAdj=shift_w(vline,1,0.0)                      # col Lc+1
    leftProj=leftAdj*leftDots
    rightProj=rightAdj*rightDots
    res=np.clip(lineCells+leftProj+rightProj,0,1)
    return res

def tensor_rule(inp):
    X=inp
    X9=X[:,1:10,:,:]
    contentMask=X.max(axis=1,keepdims=True)
    vert=vproc(X9, contentMask)
    # horizontal: transpose H<->W
    Xt=np.transpose(X,(0,1,3,2)); X9t=Xt[:,1:10,:,:]
    horiz=np.transpose(vproc(X9t, None),(0,1,3,2))
    res9=np.clip(vert+horiz,0,1)
    sumc=np.clip(res9.sum(axis=1,keepdims=True),0,1)
    ch0=contentMask*(1-sumc)
    out=np.concatenate([ch0,res9],axis=1)
    return out

def rule_grid(a):
    out=tensor_rule(onehot(a))
    g=np.argmax(out[0],axis=0)
    # cells with all-zero -> argmax=0 (bg). fine within content
    H,W=a.shape
    return g[:H,:W]

if __name__=="__main__":
    res,fails=C.check(25, rule_grid)
    print("tensor-ref 25:",res, fails[:8])
