"""Floating-point exploration; never used as the interval arithmetic kernel."""
import numpy as np

MU=np.array([[1.38,-1.03],[-.22,-.19],[-1.16,1.22]])
A=np.array([[-.08,.24],[-.95,-.31]])
TAU=.15
GAMMA=.35
LAMBDA=1.
EPS=.35
P=np.column_stack((MU[2]-MU[1],[-1.41,-.94]))

def step(x,b,sigma,lam=LAMBDA,A_=A,mu=MU):
    v=TAU**2+np.asarray(sigma)**2
    while np.ndim(v)<x.ndim-1: v=np.expand_dims(v,-1)
    logits=-np.sum((x[...,None,:]-mu)**2,axis=-1)/(2*np.asarray(v)[...,None])
    w=np.exp(logits-logits.max(axis=-1,keepdims=True))
    w/=w.sum(axis=-1,keepdims=True)
    c=lam*np.asarray(sigma)**2/(TAU**2+np.asarray(sigma)**2)
    while np.ndim(c)<x.ndim-1: c=np.expand_dims(c,-1)
    return x-GAMMA*(x@(A_.T@A_)-b+np.asarray(c)[...,None]*(x-w@mu))

def run(y,schedule,lam=LAMBDA,trajectory=False,A_=A,mu=MU):
    b=y@A_;x=b.copy();path=[x.copy()]
    for s in schedule:
        x=step(x,b,s,lam,A_,mu)
        if trajectory:path.append(x.copy())
    return (x,np.array(path)) if trajectory else x

def schedules():
    return {'constant':[EPS]*64,'jump12':[2.]*12+[EPS]*52,
            'linear':np.linspace(2.,EPS,64).tolist(),
            'middle16':[2.]*12+[.9]*16+[EPS]*36,
            'middle32':[2.]*12+[.9]*32+[EPS]*20}
