"""Floating-point version of the analytically parameterized family."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class Family:
    alpha: float
    a: float=1.
    d: float=2.
    omega: float=.04
    v0: float=1.
    theta: float=0.

    def __post_init__(self):
        self.mu=np.array([[0.,0.],[1.,0.],[-self.a,self.d]])
        self.A=np.diag([np.sqrt(self.alpha),1.])
        self.B=np.diag([self.alpha,1.])
        self.tau=self.alpha**2
        self.lam=(1+self.alpha**2)*(1+self.theta*self.alpha)
        self.cend=1+self.theta*self.alpha
        self.c0=self.c(self.v0)
        self.anchor=np.array([.5,self.c0*self.d*self.omega/(1+self.c0)])
        self.desired=np.array([.5-(self.a+1)*self.omega,.5+self.a*self.omega,self.omega])
        logpi=np.log(self.desired)+np.sum((self.anchor-self.mu)**2,axis=1)/(2*self.v0)
        self.pi=np.exp(logpi-logpi.max());self.pi/=self.pi.sum()

    def c(self,v):return self.lam*(1-self.tau**2/v)
    @property
    def vend(self):return self.alpha**2+self.alpha**4

    def weights(self,x,v):
        log=np.log(self.pi)-np.sum((x[...,None,:]-self.mu)**2,axis=-1)/(2*v)
        w=np.exp(log-log.max(axis=-1,keepdims=True))
        return w/w.sum(axis=-1,keepdims=True)

    def step(self,x,b,v):
        return x-.25*(x@self.B-b+self.c(v)*(x-self.weights(x,v)@self.mu))

    def run(self,y,stages):
        b=y@self.A;x=b.copy()
        for v,n in stages:
            for _ in range(n):x=self.step(x,b,v)
        return x

    def oracle(self,y,c=1.):
        # Return all label-conditioned means, not an estimator using hidden Z.
        return (y[:,None,:]@self.A+c*self.mu)/(np.array([self.alpha,1.])+c)

    def budgets(self):
        upper=math.ceil(math.log(64/self.alpha)/math.log(25/23))
        lower=math.floor(2+math.log(1/(10*self.alpha))/math.log(1/.74))
        refine=math.ceil(math.log(12/self.alpha**3)/math.log(4/3))
        return dict(cross_lower=lower,cross_upper=upper,refine=refine,total=upper+refine)

    def oracle_risk(self):return self.alpha**4*(1/(1+self.alpha)+.5)
    def ls_risk(self):return self.alpha**4*(1/self.alpha+1)

    def phase_bounds(self,v,accuracy=.01):
        scale=abs(v-self.v0)
        delta=(np.sqrt(2)*self.alpha/500+4*scale)/.32
        envelope=(.5*self.alpha+np.sqrt(2)*self.alpha/500+4*scale)/.32
        margin=3*self.alpha/32-delta
        hplus=math.ceil(math.log(2/margin)/math.log(25/23)) if margin>0 else None
        hminus=math.floor(2+math.log(.30/(envelope+self.alpha/100))/math.log(1/.74))
        soft=12*math.exp(-1/(64*self.alpha))
        rplus=math.ceil(math.log(6/(accuracy-soft))/math.log(4/3)) if accuracy>soft else None
        q=(3-(1+self.theta)*self.alpha)/4
        rminus=max(0,math.ceil(math.log(.25/accuracy)/math.log(1/q)))
        return dict(cross_lower=hminus,cross_upper=hplus,refine_lower=rminus,
                    refine_upper=rplus,certified_margin=margin,
                    scope='The formula applies inside |v-v0|<=alpha/100; undefined upper bounds are not failure claims.')

