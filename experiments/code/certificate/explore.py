import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from scipy.special import softmax

M = np.array([[1.38,-1.03],[-.22,-.19],[-1.16,1.22]])
A = np.array([[-.08,.24],[-.95,-.31]])

def gh(x,j,sigma):
    v=.15**2+sigma**2;q=sigma**2/v
    w=softmax(-np.sum((x-M)**2,axis=1)/(2*v));mu=w@M
    cov=(M-mu).T@(w[:,None]*(M-mu))
    g=A.T@A@(x-M[j])+q*(x-mu)
    h=A.T@A+q*np.eye(2)-q/v*cov
    return g,h

def solve(j,s,x):
    r=root(lambda z:gh(z,j,s)[0],x,jac=lambda z:gh(z,j,s)[1],tol=1e-11)
    assert np.linalg.norm(gh(r.x,j,s)[0])<1e-10
    return r.x

if __name__=='__main__':
    for j in range(3):
        prev=np.zeros(2);branch=[]
        for s in np.linspace(2,.35,3301):
            prev=solve(j,s,prev);branch.append((s,*prev,np.linalg.eigvalsh(gh(prev,j,s)[1])[0]))
        print('branch',j,'min_curvature',min(r[-1] for r in branch),'high',branch[0], 'low',branch[-1])
        for name,ss in [('const',[.35]*64),('jump',[2]*12+[.35]*52),('linear',np.linspace(2,.35,64))]:
            x=A.T@A@M[j];hist=[];lag=[]
            for s in ss:
                p=solve(j,s,np.array(branch[np.argmin(abs(np.array(branch)[:,0]-s))][1:3]))
                lag.append(np.linalg.norm(x-p))
                hist.append(np.linalg.eigvalsh(gh(x,j,s)[1])[0]);x-=.35*gh(x,j,s)[0]
            print(j,name,'x',x,'err',np.linalg.norm(x-M[j]),'curv',min(hist),'maxlag',max(lag))
