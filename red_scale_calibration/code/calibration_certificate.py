"""Uniform auxiliary inequalities for coarse observable scale control.

Uses the original outward-rounded Decimal kernel. No parameter sampling or
stationary-point candidates enter this certificate. See the written proof for
the comparison, stopping, implicit-function, and probability arguments.
"""
import json,math,sys
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'baseline/code'))
import structural_certificate as C
from interval import I,dot,norm,sub,eigbounds

def power(x,n):
    result=I(1)
    for _ in range(n):result*=x
    return result

def main():
    if not __debug__:raise RuntimeError('Do not use python -O')
    result={'status':'INCOMPLETE','family':'Identical to baseline/theory_chain.md',
            'alpha':['0','.001'],'coarse_variance_band':['.69','.71']}
    vc=I('.69','.71');cv=C.c(vc)
    assert cv.lo>Decimal('.99899') and cv.hi<Decimal('1.001002')
    # The sharper c(v)>=1-alpha>=.999 follows exactly from v>=v_L.
    # Uniform class-1 channel: s in [-.01,.53], t in [-.01,.05].
    wleft=C.weights([I('-.01'),I('.05')],vc)
    wright=C.weights([I('.53'),I('.05')],vc)
    gap=wright[1]-C.a*wright[2]-I('.53')
    assert wleft[2].hi<Decimal('.05') and gap.lo>Decimal('.018')
    Cxx=I('.25')+I('1.01').sq()*I('.05')*I('.95')+2*I('1.01')*I('.05')
    assert (Cxx/vc).hi<Decimal('.6')
    t_drift=I('-.000002','.000002')-I('.05')+cv*(C.d*wleft[2]-I('.05'))
    assert t_drift.hi<0
    lower25=I('.575')-I('.575002')*power(I('.9001'),25)
    assert lower25.lo>Decimal('.53')
    overshoot=I('.75025')*I('.53')+I('.25')*(I('.001002')+I('1.001002'))
    assert overshoot.hi<Decimal('.65')
    result['class1_channel']={'w2_upper':str(wleft[2].hi),
        'boundary_mean_minus_s_lower':str(gap.lo),
        'mean_s_derivative_upper':str((Cxx/vc).hi),
        't_upper_boundary_drift':str(t_drift.hi),
        'comparison_q':'.9001','comparison_intercept':'.0574425',
        'hypothetical_s25_lower':str(lower25.lo),'gate_overshoot_upper':str(overshoot.hi)}
    # Class 2: D2=[-1.1,.01] x [1.75,2.1]. All four boundary signs follow
    # from the two responsibility bounds below and convex coordinate updates.
    wthird=C.weights([I('.01'),I('1.75')],vc)
    third_mean=wthird[1]-C.a*wthird[2]
    assert wthird[2].lo>Decimal('.78') and third_mean.hi<Decimal('-.68')
    third_t_drift=I('1.99')-I('.000002')-I('1.75')+I('1.001002')*(I('1.99')*I('.78')-I('1.75'))
    assert third_t_drift.lo>0
    upper5=-I('.68')+I('.680002')*power(I('.75025'),5)
    assert upper5.hi<Decimal('-.5')
    result['class2_channel']={'w2_lower':str(wthird[2].lo),
        'mean_s_upper':str(third_mean.hi),
        't_lower_boundary_drift':str(third_t_drift.lo),
        'hypothetical_s5_upper':str(upper5.hi)}
    # exp(-z) >= 1-z+z^2/2-z^3/6 for z>=0, and pi<22/7.
    # Thus the probability that four standard Gaussian coordinates have
    # magnitude at most 1.5 is strictly greater than .5, without a CDF lookup.
    lower_integral=sum(((-1)**k*power(I('1.5'),2*k+1)/(2**k*math.factorial(k)*(2*k+1))
                        for k in range(4)),I(0))
    lower_probability=power((I(7)/11).sqrt()*lower_integral,4)
    assert lower_probability.lo>Decimal('.5')
    result['small_noise_event_probability_lower']=str(lower_probability.lo)
    # A fixed, alpha-independent neighborhood around the old calibrated v0.
    vl=C.v0+I('-.001','.001')
    R=['-.1','.75','-.1','.15']
    images={str(z):C.invariant(R,vl,z) for z in (0,1)}
    w1=C.weights([I('.75'),I('-.1')],vl)[1]
    w2=C.weights([I('-.1'),I('.15')],vl)[2]
    assert w1.hi<Decimal('.64') and w2.hi<Decimal('.13')
    wt=I('.13');wr=I('.64');aa=I('1.01');dd=I('2.01')
    xx=I('.25')+aa.sq()*wt*(1-wt)+2*aa*wr*wt
    yy=dd.sq()*wt*(1-wt);xy=dd*wt*(aa*(1-wt)+wr)
    cl=C.c(vl);hxx=cl*(1-xx/vl);hyy=1+cl*(1-yy/vl)
    off=(cl*xy/vl).hi
    hlo,_=eigbounds([[I(hxx.lo),I(-off)],[I(-off),I(hyy.lo)]])
    assert hlo>Decimal('.32') and (1+cl).hi<Decimal('2.01')
    box=[I('-.1','.75'),I('-.1','.15')]
    md=I(0)
    for (i,j),p in zip([(0,1),(0,2),(1,2)],[I('.25'),I('.13'),I('.13')]):
        diff=sub(C.mu[i],C.mu[j])
        gapij=dot(C.mu[i],C.mu[i])-dot(C.mu[j],C.mu[j])-2*dot(box,diff)
        md+=p*norm(diff)*I(gapij.absmax())/(2*vl.sq())
    gv=C.lam*C.tau2/vl.sq()*(norm(box)+3)+cl*md
    assert gv.hi<4
    # Consequently the virtual fixed point lies within .0125 of q(v0).
    local=[I('.48','.52'),I('.02','.06')]
    assert C.anchor[1].lo-Decimal('.0125')>Decimal('.02')
    assert C.anchor[1].hi+Decimal('.0125')<Decimal('.06')
    w=C.weights(local,vl);mean=[w[1]-C.a*w[2],C.d*w[2]]
    mv=[I(0),I(0)]
    for i,j in [(0,1),(0,2),(1,2)]:
        diff=sub(C.mu[i],C.mu[j])
        gapij=dot(C.mu[i],C.mu[i])-dot(C.mu[j],C.mu[j])-2*dot(local,diff)
        for k in range(2):mv[k]+=w[i]*w[j]*diff[k]*gapij/(2*vl.sq())
    partial=[C.lam*C.tau2/vl.sq()*(local[k]-mean[k])-cl*mv[k] for k in range(2)]
    assert partial[0].lo>Decimal('.12') and partial[1].hi<Decimal('-.16')
    derivative_min=I('.12')/(C.alpha+cl)
    assert derivative_min.lo>Decimal('.1')
    perturbation=(I('.5')+I(2).sqrt()/500)/I('.32')
    remaining=2-perturbation-I('.1')
    assert remaining.lo>Decimal('.01')
    # For class 0, a premature choice of component 1 traps the terminal
    # trajectory in [.5+alpha/100,1.1] x [-.1,.15].
    endpoint=I('.001')
    odds01=(-1/(100*endpoint*(1+endpoint.sq()))).exp()
    odds21=2*(-I('5.32')/(2*endpoint.sq()*(1+endpoint.sq()))).exp()
    wrong_mean_lower=1-odds01-I('2.01')*odds21
    assert wrong_mean_lower.lo>Decimal('.9999')
    right_boundary_drift=-endpoint*(I('.502')+endpoint/100)+I('.999')*(I('.9999')-I('.50001'))
    assert right_boundary_drift.lo>0
    result['local_plateau_necessity']={'variance_radius':'.001','rectangle_images':images,
        'hessian_lower':str(hlo),'gradient_v_norm_upper':str(gv.hi),
        'partial_v_g':[p.json() for p in partial],
        'minus_virtual_root_s_derivative_lower':str(derivative_min.lo),
        'observation_root_shift_coefficient':str(perturbation.hi),
        'wrong_side_margin_at_20alpha':str(remaining.lo),
        'class0_wrong_mean_s_lower':str(wrong_mean_lower.lo),
        'class0_wrong_left_boundary_drift_lower':str(right_boundary_drift.lo)}
    result['status']='PASS'
    (ROOT/'results/calibration_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
