import numpy as np, sys
sys.path.insert(0,'.');sys.path.insert(0,'..')
import _check as C
from ng_utils_shim import ng

NEG=-1e9
def shift_w(a, k):
    # shift along width by k (positive=content moves right; a[...,c]=orig[...,c-k]); zero fill
    out=np.zeros_like(a)
    if k>0: out[...,k:]=a[...,:a.shape[-1]-k]
    elif k<0: out[...,:a.shape[-1]+k]=a[...,-k:]
    else: out=a.copy()
    return out
def shift_h(a, k):
    out=np.zeros_like(a)
    if k>0: out[...,k:,:]=a[...,:a.shape[-2]-k,:]
    elif k<0: out[...,:a.shape[-2]+k,:]=a[...,-k:,:]
    else: out=a.copy()
    return out

def excl_prefix_max_w(v):
    # exclusive prefix max along width: out[...,c]=max(v[...,0..c-1]); fill NEG if none
    # inclusive scan via Hillis-Steele with NEG fill, then shift right by 1
    W=v.shape[-1]
    inc=v.copy()
    k=1
    while k<W:
        sh=np.full_like(inc, NEG)
        sh[...,k:]=inc[...,:W-k]
        inc=np.maximum(inc, sh)
        k*=2
    out=np.full_like(v, NEG)
    out[...,1:]=inc[...,:W-1]
    return out

def excl_revprefix_max_w(v):
    W=v.shape[-1]
    inc=v.copy()
    k=1
    while k<W:
        sh=np.full_like(inc, NEG)
        sh[...,:W-k]=inc[...,k:]
        inc=np.maximum(inc, sh)
        k*=2
    out=np.full_like(v, NEG)
    out[...,:W-1]=inc[...,1:]
    return out

def onehot(a):
    g={"input":a.tolist(),"output":a.tolist()}
    return ng.convert_to_numpy(g)["input"]  # [1,10,30,30]

def tensor_rule_onehot(inp):
    # inp: [1,10,30,30] float
    bg=inp[:,0:1,:,:]
    twos=inp[:,2:3,:,:]
    two_above=shift_h(twos,1)   # two_above[r]=twos[r-1]
    two_below=shift_h(twos,-1)
    vert2=np.maximum(two_above,two_below)
    blk=bg*vert2
    special=np.clip(twos+blk,0,1)
    H,W=30,30
    col=np.arange(W).reshape(1,1,1,W).astype(np.float32)
    colpL=col+1.0
    colpR=(W-col)
    twoVL=twos*colpL; spVL=special*colpL
    twoML=excl_prefix_max_w(twoVL); spML=excl_prefix_max_w(spVL)
    leftOK=((np.abs(twoML-spML)<0.5)&(twoML>0.5)).astype(np.float32)
    twoVR=twos*colpR; spVR=special*colpR
    twoMR=excl_revprefix_max_w(twoVR); spMR=excl_revprefix_max_w(spVR)
    rightOK=((np.abs(twoMR-spMR)<0.5)&(twoMR>0.5)).astype(np.float32)
    fill=bg*(1-blk)*leftOK*rightOK
    out=inp.copy()
    out[:,0:1,:,:]=bg-fill
    out[:,9:10,:,:]=fill
    return out

def rule_grid(a):
    inp=onehot(a)
    out=tensor_rule_onehot(inp)
    # decode to grid of a.shape
    H,W=a.shape
    g=np.argmax(out[0],axis=0)  # but need >0 thresholding; argmax ok since one-hot
    # zero channels -> argmax gives 0; ensure
    return g[:H,:W]

if __name__=="__main__":
    res,fails=C.check(381, rule_grid)
    print("tensor-ref 381:",res, fails[:6])
