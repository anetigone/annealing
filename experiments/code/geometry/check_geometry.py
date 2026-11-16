import argparse
"""Independent floating-point identity checks; supplementary, not the proof.

The certificate is verify_geometry.py. These tests cross-check completion of
squares, gradients, responsibilities, and the geometric covariance identity.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, softmax
from geometry import I, Model, q_range

ROOT=Path(__file__).parent
DATA=ROOT.parents[1]/"results"/"geometry"
DEFAULT_OUT=ROOT.parents[1]/"results"/"geometry_reproduction"
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
OUT=parser.parse_args().out
OUT.mkdir(parents=True,exist_ok=True)

def main():
    rng=np.random.default_rng(20261030)
    M0=np.array([[1.38,-1.03],[-.22,-.19],[-1.16,1.22]])
    A0=np.array([[-.08,.24],[-.95,-.31]])
    worst=dict(weights=0.,gradient=0.,update=0.,potential=0.,trace=0.,determinant=0.)
    for _ in range(500):
        M=M0+rng.uniform(-1e-4,1e-4,(3,2));A=A0+rng.uniform(-1e-4,1e-4,(2,2))
        tau=.15+rng.uniform(-1e-4,1e-4);lam=1+rng.uniform(-1e-4,1e-4)
        gamma=.35+rng.uniform(-1e-4,1e-4);a=1+rng.uniform(-1e-4,1e-4,3)
        s=rng.uniform(.35,2);y=rng.normal(size=2);x=rng.normal(size=2)
        v=tau*tau+s*s;t=lam*s*s;c=t/v;B=A.T@A;b=A.T@y;Q=B+c*np.eye(2)
        z=np.linalg.solve(Q,(b+c*M).T).T;Sinv=Q/t
        loga=np.log(a)-np.sum(M*M,axis=1)/(2*v)+np.einsum('ij,ij->i',b+c*M,z)/(2*t)
        w=softmax(np.log(a)-np.sum((x-M)**2,axis=1)/(2*v))
        logtrans=loga-.5*np.einsum('ij,jk,ik->i',x-z,Sinv,x-z)
        wt=softmax(logtrans)
        mean=w@M;cov=(M-mean).T@(w[:,None]*(M-mean))
        g=A.T@(A@x-y)+c*(x-mean)
        gt=Q@(x-wt@z)
        unrelaxed=wt@z;relaxed=(np.eye(2)-gamma*Q)@x+gamma*Q@unrelaxed
        # Constants independent of x were omitted in the transformed mixture.
        original=.5*np.sum((A@x-y)**2)/t-logsumexp(np.log(a)-np.sum((x-M)**2,axis=1)/(2*v))
        transformed=-logsumexp(logtrans)+.5*np.sum(y*y)/t
        eig,U=np.linalg.eigh(Q);Qih=(U/np.sqrt(eig))@U.T
        C=c/v*Qih@cov@Qih
        psi=sum(w[i]*w[j]*c/v*(M[i]-M[j])@np.linalg.solve(Q,M[i]-M[j]) for i in range(3) for j in range(i))
        area=np.linalg.det(np.stack([M[1]-M[0],M[2]-M[0]]))
        delta=(c/v)**2*np.prod(w)*area**2/np.linalg.det(Q)
        errors=dict(weights=np.max(abs(w-wt)),gradient=np.max(abs(g-gt)),
          update=np.max(abs(x-gamma*g-relaxed)),potential=abs(original-transformed),trace=abs(np.trace(C)-psi),determinant=abs(np.linalg.det(C)-delta))
        for key,value in errors.items():worst[key]=max(worst[key],float(value))
        assert max(errors.values())<1e-10
        assert np.linalg.eigvalsh(C)[-1]<=psi+1e-12
        kap=max(c/v*(M[i]-M[j])@np.linalg.solve(Q,M[i]-M[j])/4 for i in range(3) for j in range(i))
        assert np.linalg.eigvalsh(C)[-1]<=kap+1e-12
    # Correlated q(s,tau) interval bounds must contain values throughout boxes.
    for _ in range(500):
        slo,shi=np.sort(rng.uniform(.35,2,2));tlo,thi=np.sort(rng.uniform(.1499,.1501,2))
        qi=q_range(I(str(tlo),str(thi)),I(str(slo),str(shi)))
        s=rng.uniform(slo,shi);tau=rng.uniform(tlo,thi);q=s*s/(tau*tau+s*s)
        assert float(qi.lo)-1e-14<=q<=float(qi.hi)+1e-14
    result={'status':'PASS','identity_cases':500,'interval_sampling_cases':500,
      'maximum_absolute_identity_errors':worst,'role':'supplementary floating-point checks, not an interval proof'}
    (OUT/'identity_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
