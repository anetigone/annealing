"""Independent arithmetic checks and model identities; not formal verification."""
import json,random
from fractions import Fraction as F
from decimal import Decimal
from pathlib import Path
from interval import I

def inside(interval,value):
    assert F(interval.lo)<=value<=F(interval.hi),(interval,value)

def exp_bounds(q,n=240):
    if q<0:
        lo,hi=exp_bounds(-q,n);return 1/hi,1/lo
    term=F(1);s=term
    for j in range(1,n+1):term*=q/j;s+=term
    nextterm=term*q/(n+1)
    return s,s+nextterm/(1-q/F(n+2))

def main():
    rng=random.Random(301026);checks=0
    for _ in range(1000):
        a=Decimal(rng.randint(-100000,100000))/1000;b=Decimal(rng.randint(-100000,100000))/1000
        x,y=I(a),I(b)
        for interval,value in [(x+y,F(a)+F(b)),(x-y,F(a)-F(b)),(x*y,F(a)*F(b))]:inside(interval,value);checks+=1
        if b:inside(x/y,F(a)/F(b));checks+=1
        inside(x.sq(),F(a)**2);checks+=1
    for numerator in range(-80,81):
        q=F(numerator,10);x=I(Decimal(numerator)/10)
        lo,hi=exp_bounds(q);inside(x.exp(),lo);inside(x.exp(),hi);checks+=2
    for numerator in range(100):
        q=Decimal(numerator)/10;r=I(q).sqrt()
        assert F(r.lo)**2<=F(q)<=F(r.hi)**2;checks+=1
    inside(I('-2','3').sq(),F(0));inside(I('-2','3').sq(),F(9));checks+=2
    try:I(.1)
    except TypeError:checks+=1
    else:raise AssertionError('Float accepted')
    import numpy as np
    from certificate import Model,P as PI,PINV,O,add,sub,matvec
    from model import A,MU,P,TAU,GAMMA,step
    nr=np.random.default_rng(301026)
    max_hessian_difference=0.
    for _ in range(200):
        st=nr.uniform([-.3,-.2],[1.2,.3]);sig=nr.uniform(.35,2);z=int(nr.integers(3))
        x=MU[1]+P@st;b=A.T@(A@MU[z])
        direct=step(x,b,sig)
        model=Model(z)
        sI=I(format(sig,'.16f'));stI=[I(format(a,'.16f')) for a in st]
        xI=add(O,matvec(PI,stI));bound=model.update(xI,sI)
        for k in range(2):assert float(bound[k].lo)-1e-13<=direct[k]<=float(bound[k].hi)+1e-13
        ds=model.coord_delta(stI,sI);coord=np.linalg.solve(P,direct-MU[1])
        for k in range(2):assert float((stI[k]+ds[k]).lo)-1e-13<=coord[k]<=float((stI[k]+ds[k]).hi)+1e-13
        checks+=4
        v=TAU**2+sig**2;l=-np.sum((x-MU)**2,axis=1)/(2*v)
        w=np.exp(l-l.max());w/=w.sum();center=w@MU
        C=sum(w[j]*np.outer(MU[j]-center,MU[j]-center) for j in range(3))
        H=A.T@A+sig**2/v*(np.eye(2)-C/v)
        def g(q):return (q-step(q,b,sig))/GAMMA
        numerical=np.column_stack([(g(x+np.eye(2)[j]*1e-6)-g(x-np.eye(2)[j]*1e-6))/2e-6 for j in range(2)])
        diff=float(np.max(np.abs(H-numerical)));max_hessian_difference=max(max_hessian_difference,diff)
        assert diff<1e-7
        HI=model.hessian(xI,sI)
        for i in range(2):
            for j in range(2):assert float(HI[i][j].lo)-1e-12<=H[i,j]<=float(HI[i][j].hi)+1e-12
        checks+=5
        # Check the transverse Jacobian identity used by the scalar certificate.
        q=(MU-MU[1])@np.linalg.inv(P).T
        qmean=w@q
        covq=sum(w[j]*np.outer(q[j]-qmean,q[j]-qmean) for j in range(3))
        Bq=np.linalg.solve(P,(A.T@A)@P)
        jq=-GAMMA*Bq[0,1]+GAMMA*(sig**2/v)*(2.8717/v)*covq[0,1]
        transformed=np.linalg.solve(P,(np.eye(2)-GAMMA*H)@P)
        assert abs(jq-transformed[0,1])<1e-12
        checks+=1
    result={'status':'PASS','checks':checks,'maximum_hessian_finite_difference_error':max_hessian_difference,'exp_reference':'Exact rational Taylor lower bound and geometric remainder upper bound','caveat':'Implementation checks do not constitute a formal proof of the verifier.'}
    path=Path(__file__).resolve().parents[1]/'results/implementation_checks.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':main()
