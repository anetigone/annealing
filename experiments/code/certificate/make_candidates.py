import argparse
"""Untrusted floating-point proposal generator. verify.py does not trust solvers."""
import json
from pathlib import Path
import numpy as np
from explore import M,A,gh,solve

def vec(x):return [f'{v:.16f}' for v in x]

ROOT=Path(__file__).parent
DATA=ROOT.parents[1]/"results"/"certificate"
DEFAULT_OUT=ROOT.parents[1]/"results"/"certificate_reproduction"
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
OUT=parser.parse_args().out
OUT.mkdir(parents=True,exist_ok=True)

def main():
    n=3300
    sigmas=np.linspace(2,.35,n+1)
    result={'branch_segments':n,'branches':[],'paths':[],'terminal_centers':[]}
    for j in range(3):
        prev=np.zeros(2);rows=[]
        for k in range(n):
            s=(sigmas[k]+sigmas[k+1])/2;prev=solve(j,s,prev);rows.append(vec(prev))
        result['branches'].append(rows)
        paths={}
        for name,ss in [('constant',[.35]*64),('jump',[2]*12+[.35]*52),('linear',np.linspace(2,.35,64)),('warmup',[2]*32)]:
            x=A.T@A@M[j];hist=[vec(x)]
            for s in ss:
                x=x-.35*gh(x,j,s)[0];hist.append(vec(x))
            paths[name]=hist
        result['paths'].append(paths)
        centers={}
        for name in ['constant','jump','linear']:
            centers[name]=vec(solve(j,.35,np.array(paths[name][-1],float)))
        centers['high']=vec(solve(j,2,np.zeros(2)))
        result['terminal_centers'].append(centers)
    (OUT/'candidates.json').write_text(json.dumps(result,separators=(',',':')))

if __name__=='__main__':main()
