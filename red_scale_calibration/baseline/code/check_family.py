"""Independent numerical identity checks. Does not certify the theorem."""
import json
from pathlib import Path
import numpy as np
from family import Family

def main():
    rng=np.random.default_rng(202610313);checks=0;worst=0.
    for _ in range(600):
        alpha=10**rng.uniform(-6,-3)
        f=Family(alpha,a=rng.uniform(.99,1.01),d=rng.uniform(1.99,2.01),
                 omega=rng.uniform(.039,.041),v0=rng.uniform(.99,1.01),theta=rng.uniform(-1,1))
        assert np.max(np.abs(f.weights(f.anchor,f.v0)-f.desired))<1e-12
        virtual=np.array([alpha/2,0.])
        assert np.max(np.abs(f.step(f.anchor,virtual,f.v0)-f.anchor))<1e-12
        ratio=f.pi[1]/f.pi[0]
        recovered=.5*(ratio-1)/(f.a+ratio*(f.a+1))
        assert abs(recovered-f.omega)<1e-12
        left=np.log((f.pi[2]/f.pi[0])*f.desired[0]/f.omega)
        right=(f.a**2+f.a+f.d**2-2*f.d*f.anchor[1])/(2*f.v0)
        assert abs(left-right)<1e-12
        assert abs(f.c(f.vend)-f.cend)<1e-12
        checks+=5
        x=rng.uniform([-.1,-.1],[.75,.15]);v=f.v0+rng.uniform(-alpha/100,alpha/100)
        w=f.weights(x,v);m=w@f.mu
        C=sum(w[i]*np.outer(f.mu[i]-m,f.mu[i]-m) for i in range(3))
        H=f.B+f.c(v)*(np.eye(2)-C/v)
        def g(z):return (z-f.step(z,virtual,v))/.25
        numerical=np.column_stack([(g(x+np.eye(2)[j]*1e-6)-g(x-np.eye(2)[j]*1e-6))/2e-6 for j in range(2)])
        err=float(np.abs(H-numerical).max());worst=max(worst,err)
        assert err<1e-7
        assert np.linalg.eigvalsh(H).min()>.32
        assert ((np.eye(2)-.25*H)[0,1]<0)
        # Exact mismatch resolvent identity for arbitrary observations.
        b=rng.normal(size=2)
        z=int(rng.integers(3));c=f.cend
        m1=(b+f.mu[z])/(np.array([alpha,1])+1)
        mc=(b+c*f.mu[z])/(np.array([alpha,1])+c)
        rhs=(c-1)*(f.mu[z]-m1)/(np.array([alpha,1])+c)
        assert np.max(np.abs(mc-m1-rhs))<1e-12
        checks+=4
    result=dict(status='PASS',checks=checks,max_hessian_finite_difference_error=worst,
                note='Numerical identity checks supplement, not replace, directed-rounding constants and written proofs.')
    (Path(__file__).resolve().parents[1]/'results/identity_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':main()
