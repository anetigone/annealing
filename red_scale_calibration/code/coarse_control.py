"""Observable scale control; every output update remains the original RED step."""
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'baseline/code'))
from family import Family

def observable_target(b,alpha):
    return np.where(b[:,1]>1,2,np.where(b[:,0]>alpha/2,1,0))

def in_gate(x,target):
    s,t=x[:,0],x[:,1]
    g0=(s>=-.1)&(s<=.4)&(t>=-.1)&(t<=.15)
    g1=(s>=.53)&(s<=1.1)&(t>=-.1)&(t<=.15)
    g2=(s>=-1.1)&(s<=-.5)&(t>=1.5)&(t<=2.1)
    return np.where(target==0,g0,np.where(target==1,g1,g2))

def run_from_b(f,b,refine,middle=.7,rng=None):
    """Exactly 25+refine RED updates per sample, including early terminal calls.

    The cap makes the estimator defined on all Gaussian observations. A random
    band policy is a numerical diagnostic; the certificate covers every choice
    in the band, including history-dependent choices.
    """
    b=np.asarray(b,dtype=float).reshape(-1,2)
    x=b.copy();target=observable_target(b,f.alpha)
    gate_time=np.where(in_gate(x,target),0,-1)
    for k in range(25):
        active=gate_time<0;terminal=~active
        v=float(rng.uniform(.69,.71)) if rng is not None else middle
        if not .69<=v<=.71:raise ValueError('Middle scale is outside the certified band')
        if active.any():x[active]=f.step(x[active],b[active],v)
        if terminal.any():x[terminal]=f.step(x[terminal],b[terminal],f.vend)
        newly=active&in_gate(x,target)
        gate_time[newly]=k+1
    forced=gate_time<0
    gate_time[forced]=25
    for _ in range(refine):x=f.step(x,b,f.vend)
    return x,dict(target=target,gate_time=gate_time,forced=forced,total_calls=25+refine)

def run(f,y,refine,middle=.7,rng=None):
    return run_from_b(f,np.asarray(y)@f.A,refine,middle,rng)
