"""Diagnostics for the proved family, separate from interval certification."""
import argparse,json,math
from pathlib import Path
import numpy as np
from family import Family

ROOT=Path(__file__).resolve().parents[1]

def rate_study():
    rows=[]
    for alpha in [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]:
        f=Family(alpha);bounds=f.budgets();b=f.mu@f.B;x=b.copy();paths=[x.copy()]
        for _ in range(bounds['cross_upper']):x=f.step(x,b,f.v0);paths.append(x.copy())
        paths=np.array(paths)
        hgap=int(np.flatnonzero(paths[:,1,0]>=.5+alpha/16)[0])
        trial=paths.copy()
        for _ in range(180):trial=f.step(trial,b,f.vend)
        ok=np.max(np.linalg.norm(trial-f.mu,axis=-1),axis=1)<.1
        hactual=int(np.flatnonzero(ok)[0])
        assert not ok[:bounds['cross_lower']+1].any()
        assert ok[bounds['cross_upper']]
        rows.append(dict(alpha=alpha,**bounds,observed_gap_hitting=hgap,
                         observed_first_terminal_success=hactual,
                         definition='Noiseless floating point; success after 180 common terminal calls, error <0.1.'))
    return rows

def risk_study(n):
    rng=np.random.default_rng(202610311)
    rows=[]
    for alpha in [1e-3,3e-4,1e-4,3e-5,1e-5]:
        f=Family(alpha);bud=f.budgets();z=np.repeat(np.arange(3),n)
        truth=f.mu[z]+alpha**2*rng.standard_normal((3*n,2))
        y=truth@f.A+alpha**2*rng.standard_normal((3*n,2))
        def average(a):return float(sum(f.pi[i]*a[z==i].mean() for i in range(3)))
        out=f.run(y,[(f.v0,bud['cross_upper']),(f.vend,bud['refine'])])
        restricted=f.run(y,[(f.vend,bud['total'])])
        oracle=f.oracle(y)[np.arange(3*n),z]
        loss=np.sum((out-truth)**2,axis=1)
        oloss=np.sum((oracle-truth)**2,axis=1)
        rloss=np.sum((restricted-truth)**2,axis=1)
        excess=np.sum((out-oracle)**2,axis=1)
        rows.append(dict(alpha=alpha,n_per_class=n,budget=bud,prior=f.pi.tolist(),
                         mid_risk=average(loss),restricted_risk=average(rloss),
                         oracle_sample_risk=average(oloss),oracle_exact_risk=f.oracle_risk(),
                         least_squares_exact_risk=f.ls_risk(),oracle_deviation=average(excess),
                         mid_to_oracle=average(loss)/f.oracle_risk(),
                         restricted_class1_error_min=float(np.linalg.norm(restricted[z==1]-truth[z==1],axis=1).min())))
        print('risk',rows[-1],flush=True)
    return rows

def perturbation_study(n=200):
    rng=np.random.default_rng(202610312);maxerror=0.;minbad=10.
    for _ in range(n):
        alpha=10**rng.uniform(-7,-3)
        f=Family(alpha,a=rng.uniform(.99,1.01),d=rng.uniform(1.99,2.01),
                 omega=rng.uniform(.039,.041),v0=rng.uniform(.99,1.01),theta=rng.uniform(-1,1))
        # Bound the observation vector directly by the deterministic theorem.
        b=f.mu@f.B+rng.uniform(-alpha/500,alpha/500,(3,2))
        x=b.copy();K=f.budgets()
        for _ in range(K['cross_upper']):
            v=f.v0+rng.uniform(-alpha/1000,alpha/1000)
            x=f.step(x,b,v)
        assert x[0,0]<.5-alpha/16 and x[1,0]>.5+alpha/16
        assert -1.1<x[2,0]<-.5 and 1.5<x[2,1]<2.1
        for _ in range(K['refine']):x=f.step(x,b,f.vend)
        mc=(b+f.cend*f.mu)/(np.array([alpha,1.])+f.cend)
        maxerror=max(maxerror,float(np.linalg.norm(x-mc,axis=1).max()))
        xbad=b.copy()
        for _ in range(200):
            v=f.vend if rng.random()<.5 else rng.uniform(4,20)
            xbad=f.step(xbad,b,v)
            assert xbad[1,0]<=.3+1e-12
        minbad=min(minbad,float(np.linalg.norm(xbad[1]-f.mu[1])))
    return dict(status='PASS_FLOATING_POINT',n=n,max_terminal_oracle_distance=maxerror,
                min_restricted_class1_template_error=minbad,
                caveat='Random diagnostics are not the uniform proof.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=2000);args=p.parse_args()
    result={'status':'FLOATING_POINT_ONLY','rate':rate_study(),
            'perturbations':perturbation_study(),'risk':risk_study(args.n)}
    (ROOT/'results/experiments.json').write_text(json.dumps(result,indent=2)+'\n')
    print('rate',result['rate'],flush=True)

if __name__=='__main__':main()
