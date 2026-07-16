import numpy as np, sys
sys.path.insert(0,'.');sys.path.insert(0,'..')
import _check as C

def rule(a):
    H,W=a.shape
    b=np.zeros_like(a)
    # find full-row lines and full-col lines (all cells same nonzero color)
    row_lines={}  # r -> color
    col_lines={}
    for r in range(H):
        v=a[r]
        nz=set(v[v!=0].tolist())
        if len(nz)==1 and (v!=0).all():
            row_lines[r]=v[0]
    for c in range(W):
        v=a[:,c]
        nz=set(v[v!=0].tolist())
        if len(nz)==1 and (v!=0).all():
            col_lines[c]=v[0]
    # lines persist
    for r,col in row_lines.items(): b[r,:]=col
    for c,col in col_lines.items(): b[:,c]=col
    # orientation: if col_lines exist, dots slide horizontally to matching col line
    color_to_col={col:c for c,col in col_lines.items()}
    color_to_row={col:r for r,col in row_lines.items()}
    for r in range(H):
        for c in range(W):
            v=a[r,c]
            if v==0: continue
            if r in row_lines or c in col_lines: continue  # part of a line
            placed=False
            if v in color_to_col:
                Lc=color_to_col[v]
                nc=Lc-1 if c<Lc else Lc+1
                if 0<=nc<W: b[r,nc]=v
                placed=True
            elif v in color_to_row:
                Lr=color_to_row[v]
                nr=Lr-1 if r<Lr else Lr+1
                if 0<=nr<H: b[nr,c]=v
                placed=True
            # else: deleted (do nothing)
    return b

if __name__=="__main__":
    res,fails=C.check(25, rule)
    print("25 project-to-line:",res, fails[:8])
