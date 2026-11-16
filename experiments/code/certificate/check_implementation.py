import argparse
"""Independent implementation checks; these supplement, not replace, verify.py.

Fraction checks validate arithmetic containment without binary floating point.
Taylor series brackets validate exp independently of Decimal.exp. Floating point
finite differences check the derived Hessian and scale derivative formulas.
"""
from fractions import Fraction as F
import random,json
from pathlib import Path
from interval import I,D,norm,Model

def contains(a,x):return F(a.lo)<=x<=F(a.hi)
ROOT=Path(__file__).parent
DATA=ROOT.parents[1]/"results"/"certificate"
DEFAULT_OUT=ROOT.parents[1]/"results"/"certificate_reproduction"
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
OUT=parser.parse_args().out
OUT.mkdir(parents=True,exist_ok=True)

def main():
    rng=random.Random(20261030);count=0
    for _ in range(1000):
        x=str(rng.randint(-100000,100000)/1000)
        y=str(rng.randint(1,100000)/1000)
        a,b=I(x),I(y);u,v=F(x),F(y)
        for val,exact in [(a+b,u+v),(a-b,u-v),(a*b,u*v),(a/b,u/v),(a.sq(),u*u)]:
            assert contains(val,exact);count+=1
        sq=a.sq().sqrt();assert F(sq.lo)<=abs(u)<=F(sq.hi);count+=1
    for n in range(41):
        x=F(n,10);term=F(1);s=term
        for k in range(1,180):term*=x/k;s+=term
        nxt=term*x/180;upper=s+nxt/(1-x/181)
        val=I(str(n/10)).exp()
        # The rational series enclosure is far tighter than the Decimal one.
        assert F(val.lo)<=s<=upper<=F(val.hi);count+=1
    import numpy as np
    from scipy.special import softmax
    from explore import M,A,gh
    model=Model('0','0');max_herr=0.;max_serr=0.;max_gerr=0.
    for _ in range(100):
        x=np.array([rng.uniform(-2,2),rng.uniform(-2,2)]);s=rng.uniform(.35,2);j=rng.randrange(3)
        xx=[I(str(t)) for t in x];ss=I(str(s))
        g=model.g(xx,j,ss);h=model.h(xx,ss);gs=model.gs(xx,ss)
        gp,hp=gh(x,j,s)
        gi=np.array([float((t.lo+t.hi)/2) for t in g]);hi=np.array([[float((t.lo+t.hi)/2) for t in row] for row in h])
        max_gerr=max(max_gerr,float(np.max(abs(gi-gp))));max_herr=max(max_herr,float(np.max(abs(hi-hp))))
        ds=1e-5;fd=(gh(x,j,s+ds)[0]-gh(x,j,s-ds)[0])/(2*ds)
        max_serr=max(max_serr,float(np.max(abs(np.array([float((t.lo+t.hi)/2) for t in gs])-fd))))
    assert max_gerr<1e-12 and max_herr<1e-12 and max_serr<1e-7
    result={'exact_arithmetic_and_exp_brackets':count,'random_derivative_checks':100,
      'max_gradient_formula_difference':max_gerr,'max_hessian_formula_difference':max_herr,
      'max_scale_derivative_finite_difference_error':max_serr,'status':'PASS'}
    (OUT/'implementation_checks.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
