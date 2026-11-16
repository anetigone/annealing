import argparse
"""Generate a sparse, untrusted branch polyline. Accepted cells are rechecked."""
import json
from decimal import Decimal
from pathlib import Path
import numpy as np
from scipy.optimize import root
from scipy.special import softmax
from sparse_branch import segment_check
from geometry import Model,D

ROOT=Path(__file__).parent
DATA=ROOT.parents[1]/"results"/"geometry"
DEFAULT_OUT=ROOT.parents[1]/"results"/"geometry_reproduction"
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
OUT=parser.parse_args().out
OUT.mkdir(parents=True,exist_ok=True)
M=np.array([[1.38,-1.03],[-.22,-.19],[-1.16,1.22]])
A=np.array([[-.08,.24],[-.95,-.31]])

def gh(x,j,s):
    v=.15**2+s*s;c=s*s/v;w=softmax(-np.sum((x-M)**2,axis=1)/(2*v));mu=w@M
    cov=(M-mu).T@(w[:,None]*(M-mu))
    return A.T@A@(x-M[j])+c*(x-mu),A.T@A+c*np.eye(2)-c/v*cov

def solve(j,s,x):
    z=root(lambda t:gh(t,j,s)[0],x,jac=lambda t:gh(t,j,s)[1],tol=1e-11)
    assert np.linalg.norm(gh(z.x,j,s)[0])<1e-10
    return z.x

def main():
    model=Model();out=[]
    for j in range(3):
        ss=np.linspace(2,.35,257);pre=[];prev=np.zeros(2)
        for s in ss:prev=solve(j,s,prev);pre.append(prev.copy())
        cache={};cells=[]
        def center(s):
            if s not in cache:
                x=solve(j,float(s),pre[np.argmin(abs(ss-float(s)))])
                cache[s]=[f'{t:.16f}' for t in x]
            return cache[s]
        def visit(a,b,depth=0):
            ca,cb=center(a),center(b)
            for R in ['.03','.04','.02','.06','.08','.015']:
                z=segment_check(model,j,a,b,ca,cb,R)
                if z['pass'] and z['m']>D('.09') and z['rootdist']<D(R)-D('.006'):
                    cells.append({'a':a,'b':b,'ca':ca,'cb':cb,'radius':R});return
            assert depth<12,(j,a,b,z)
            mid=str((Decimal(a)+Decimal(b))/2)
            visit(a,mid,depth+1);visit(mid,b,depth+1)
        # Analytic global convexity handles [1.6,2]. Dominant-template balls
        # handle [.35,.55]. Only the mixed-responsibility band is tabulated.
        visit('1.6','.55');out.append(cells)
        print('target',j,'cells',len(cells),flush=True)
    high_paths=[]
    for j in range(3):
        x=A.T@A@M[j];path=[[f'{t:.16f}' for t in x]]
        for k in range(12):x=x-.35*gh(x,j,2)[0];path.append([f'{t:.16f}' for t in x])
        high_paths.append(path)
    (OUT/'sparse_candidates.json').write_text(json.dumps({'branches':out,'high_paths':high_paths},indent=2))
    print('total',sum(map(len,out)),flush=True)

if __name__=='__main__':main()
