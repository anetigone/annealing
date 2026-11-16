"""Floating-point diagnostics, kept separate from the uniform certificate."""
import argparse,json,math
from pathlib import Path
import numpy as np
from coarse_control import Family,run,run_from_b

ROOT=Path(__file__).resolve().parents[1]

def gate_study():
    rows=[]
    for alpha in [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]:
        f=Family(alpha);refine=f.budgets()['refine']
        out,meta=run_from_b(f,f.mu@f.B,refine)
        assert (meta['target']==np.arange(3)).all() and not meta['forced'].any()
        assert meta['gate_time'][0]==0 and meta['gate_time'][1]<=25 and meta['gate_time'][2]<=5
        rows.append(dict(alpha=alpha,gate_times=meta['gate_time'].tolist(),
                         total_calls=meta['total_calls'],refine_minimum=refine,
                         max_template_error=float(np.linalg.norm(out-f.mu,axis=1).max())))
    return rows

def perturbation_study(n=240):
    rng=np.random.default_rng(202610314);maximum=[0,0,0];worst=0.
    for _ in range(n):
        alpha=10**rng.uniform(-8,-3)
        f=Family(alpha,a=rng.uniform(.99,1.01),d=rng.uniform(1.99,2.01),
                 omega=rng.uniform(.039,.041),v0=rng.uniform(.99,1.01),theta=rng.uniform(-1,1))
        b=f.mu@f.B+rng.uniform(-alpha/500,alpha/500,(3,2))
        refine=f.budgets()['refine']
        out,meta=run_from_b(f,b,refine,rng=rng)
        assert (meta['target']==np.arange(3)).all() and not meta['forced'].any()
        assert meta['gate_time'][0]==0 and meta['gate_time'][1]<=25 and meta['gate_time'][2]<=5
        maximum=[max(m,int(h)) for m,h in zip(maximum,meta['gate_time'])]
        mc=(b+f.cend*f.mu)/(np.array([alpha,1])+f.cend)
        error=float(np.linalg.norm(out-mc,axis=1).max());worst=max(worst,error)
        bound=6*.75**refine+12*math.exp(-1/(64*alpha))
        assert error<=bound+3e-15
    return dict(status='PASS_FLOATING_POINT',n=n,max_gate_times=maximum,
                max_terminal_distance_to_matched_coefficient_mean=worst)

def sensitivity_study():
    rows=[]
    for alpha in [1e-5,1e-6,1e-7]:
        f=Family(alpha);h=math.ceil(math.log(20/alpha)/math.log(25/23))
        for offset in [-20,0,20]:
            v=f.v0+offset*alpha;b=f.mu@f.B;x=b.copy()
            for _ in range(h):x=f.step(x,b,v)
            plateau=x.copy()
            for _ in range(180):x=f.step(x,b,f.vend)
            errors=np.linalg.norm(x-f.mu,axis=1)
            if offset<0:assert errors[0]>.5
            if offset>0:assert errors[1]>.5
            if offset==0:assert errors.max()<.01
            rows.append(dict(alpha=alpha,offset_in_alpha=offset,h=h,
                             plateau_s=plateau[:,0].tolist(),terminal_errors=errors.tolist()))
    return rows

def risk_study(n):
    rng=np.random.default_rng(202610315);rows=[]
    for alpha in [1e-3,1e-4,1e-5]:
        f=Family(alpha);z=np.repeat(np.arange(3),n)
        truth=f.mu[z]+alpha**2*rng.standard_normal((3*n,2))
        y=truth@f.A+alpha**2*rng.standard_normal((3*n,2))
        out,meta=run(f,y,f.budgets()['refine'],rng=rng)
        oracle=f.oracle(y)[np.arange(3*n),z]
        def average(arr):return float(sum(f.pi[i]*arr[z==i].mean() for i in range(3)))
        rows.append(dict(alpha=alpha,n_per_class=n,total_calls=meta['total_calls'],
            gate_min=[int(meta['gate_time'][z==i].min()) for i in range(3)],
            gate_max=[int(meta['gate_time'][z==i].max()) for i in range(3)],
            wrong_targets=int((meta['target']!=z).sum()),forced=int(meta['forced'].sum()),
            red_risk=average(np.sum((out-truth)**2,axis=1)),
            oracle_sample_risk=average(np.sum((oracle-truth)**2,axis=1)),
            oracle_exact_risk=f.oracle_risk(),
            red_oracle_distance=average(np.sum((out-oracle)**2,axis=1))))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=3000);args=p.parse_args()
    result={'status':'PASS_FLOATING_POINT','gate_study':gate_study(),
        'perturbations':perturbation_study(),'sensitivity':sensitivity_study(),
        'risk':risk_study(args.n),
        'scope':'These diagnostics are not a uniform proof or an exact phase diagram.'}
    (ROOT/'results/diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
