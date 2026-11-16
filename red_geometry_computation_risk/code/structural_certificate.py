"""Uniform geometric constants for a normalized three-template family.

No stationary-point candidates or sampled parameter grids are used. The
coordinates are monotone, so rectangle images are bounded on corner faces.
All decimal strings are exact rationals; interval.py rounds outward.
"""
import json
from pathlib import Path
from decimal import Decimal, getcontext
from interval import I, norm, dot, sub, add, eigbounds

getcontext().prec=60
ROOT=Path(__file__).resolve().parents[1]
alpha=I('0','.001');theta=I('-1','1')
a=I('.99','1.01');d=I('1.99','2.01')
omega=I('.039','.041');v0=I('.99','1.01')
lam=(1+alpha.sq())*(1+theta*alpha);tau2=alpha.sq().sq();gamma=I('.25')
c0=lam*(1-tau2/v0)
anchor=[I('.5'),c0*d*omega/(1+c0)]
desired=[I('.5')-(a+1)*omega,I('.5')+a*omega,omega]
ratios=[I(1),desired[1]/desired[0],
        desired[2]/desired[0]*((a.sq()+a+d.sq()-2*d*anchor[1])/(2*v0)).exp()]
mu=[[I(0),I(0)],[I(1),I(0)],[-a,d]]
vgood=v0+I('-.00001','.00001')  # |v-v0| <= alpha/100, alpha <= .001.

def c(v):return lam*(1-tau2/v)

def weights(x,v):
    result=[]
    for i in range(3):
        den=I(1)
        for j in range(3):
            if i==j:continue
            gap=2*dot(x,sub(mu[j],mu[i]))+dot(mu[i],mu[i])-dot(mu[j],mu[j])
            den+=ratios[j]/ratios[i]*(gap/(2*v)).exp()
        result.append(1/den)
    return result

def update(x,v,label=None,virtual=False):
    cv=c(v);w=weights(x,v)
    ds=alpha*I('.5') if virtual else alpha*mu[label][0]
    dt=I(0) if virtual else mu[label][1]
    err=I('-.000002','.000002') if not virtual else I(0)
    ds+=err;dt+=err
    mean=[w[1]-a*w[2],d*w[2]]
    return [(1-gamma*(alpha+cv))*x[0]+gamma*(ds+cv*mean[0]),
            (1-gamma*(1+cv))*x[1]+gamma*(dt+cv*mean[1])]

def image_rectangle(rect,v,label=None,virtual=False):
    # Both own derivatives are positive; both cross derivatives are negative.
    sl,sh,tl,th=rect
    upper_s_lower_t=update([I(sh),I(tl)],v,label,virtual)
    lower_s_upper_t=update([I(sl),I(th)],v,label,virtual)
    return [lower_s_upper_t[0].lo,upper_s_lower_t[0].hi,
            upper_s_lower_t[1].lo,lower_s_upper_t[1].hi]

def invariant(rect,v,label):
    out=image_rectangle(rect,v,label)
    assert out[0]>=Decimal(rect[0]) and out[1]<=Decimal(rect[1]),(label,rect,out)
    assert out[2]>=Decimal(rect[2]) and out[3]<=Decimal(rect[3]),(label,rect,out)
    return [str(x) for x in out]

def main():
    if not __debug__:raise RuntimeError('Do not use python -O')
    result={'status':'INCOMPLETE','parameter_ranges':{
        'alpha':['0','.001'],'theta':theta.json(),'a':a.json(),'d':d.json(),
        'omega':omega.json(),'v0':v0.json()},'anchor_y':anchor[1].json(),
        'prior_ratios':[r.json() for r in ratios]}
    pi=[r/sum(ratios,I(0)) for r in ratios]
    assert all(p.lo>Decimal('.23') for p in pi)
    assert pi[1].lo>Decimal('.30')
    result['prior_probabilities']=[r.json() for r in pi]
    # R contains every backprojection for labels 0,1 and the explicit anchor.
    R=['-.1','.75','-.1','.15']
    result['binary_rectangle_images']={str(z):invariant(R,vgood,z) for z in (0,1)}
    # Actual class-1 initial states and first iterates also lie strictly in the
    # terminal wrong halfspace; this covers h=0,1 in the crossing lower bound.
    initial1=[I('-.000002','.001002'),I('-.000002','.000002')]
    first1=image_rectangle([str(initial1[0].lo),str(initial1[0].hi),
                            str(initial1[1].lo),str(initial1[1].hi)],vgood,1)
    assert first1[1]<Decimal('.20')
    result['class1_first_step_box']=[str(t) for t in first1]
    wRmax=weights([I('.75'),I('-.1')],vgood)[1]
    wTmax=weights([I('-.1'),I('.15')],vgood)[2]
    assert wRmax.hi<Decimal('.64') and wTmax.hi<Decimal('.13'),(wRmax,wTmax)
    # These rational covariance bounds follow from w_R<=.64, w_T<=.13.
    wT=I('.13');wR=I('.64');amax=I('1.01');dmax=I('2.01')
    Cxx=I('.25')+amax.sq()*wT*(1-wT)+2*amax*wR*wT
    Cyy=dmax.sq()*wT*(1-wT)
    Cxy=dmax*wT*(amax*(1-wT)+wR)
    cv=c(vgood)
    assert (1+cv).hi<Decimal('2.01')
    assert (1-gamma*(alpha+cv)).lo>Decimal('.74')
    hxx=cv*(1-Cxx/vgood);hyy=1+cv*(1-Cyy/vgood)
    # Compare with a constant symmetric matrix with the smallest diagonals.
    off=(cv*Cxy/vgood).hi
    lower,upper=eigbounds([[I(hxx.lo),I(-off)],[I(-off),I(hyy.lo)]])
    assert lower>Decimal('.32'),lower
    result['contraction']={'wR_upper':str(wRmax.hi),'wT_upper':str(wTmax.hi),
                          'hessian_lower':str(lower),'q':'.92'}
    # Uniform bound for d/dv g_v; pairwise covariance formula.
    box=[I('-.1','.75'),I('-.1','.15')]
    distances=[norm(sub(mu[i],mu[j])) for i,j in [(0,1),(0,2),(1,2)]]
    gaps=[(dot(mu[i],mu[i])-dot(mu[j],mu[j])-2*dot(box,sub(mu[i],mu[j]))).absmax()
          for i,j in [(0,1),(0,2),(1,2)]]
    pairbounds=[I('.25'),I('.13'),I('.13')]
    mean_derivative=sum((p*dist*I(gap) for p,dist,gap in zip(pairbounds,distances,gaps)),I(0))/(2*vgood.sq())
    mean_norm_bound=I(3)
    gradient_v=lam*tau2/vgood.sq()*(norm(box)+mean_norm_bound)+cv*mean_derivative
    assert gradient_v.hi<4,gradient_v
    result['gradient_scale_derivative_upper']=str(gradient_v.hi)
    # The reference trajectory uses data (alpha/2,0), with explicit anchor.
    ref=[Decimal(0),Decimal('.0005'),Decimal(0),Decimal(0)]
    refrows=[]
    for _ in range(2):
        ref=image_rectangle([str(x) for x in ref],v0,virtual=True)
        refrows.append([str(x) for x in ref])
    assert ref[1]<Decimal('.20') and ref[2]>anchor[1].hi,(ref,anchor)
    result['reference_first_two_steps']=refrows
    R2=['-1.1','-.5','1.5','2.1']
    result['third_rectangle_image']=invariant(R2,vgood,2)
    box2=[-alpha.hi*a.hi-Decimal('.000002'),Decimal('.000002'),
          d.lo-Decimal('.000002'),d.hi+Decimal('.000002')]
    rows=[]
    for k in range(12):
        box2=image_rectangle([str(x) for x in box2],vgood,2)
        rows.append([str(x) for x in box2])
        if box2[0]>=Decimal(R2[0]) and box2[1]<=Decimal(R2[1]) and box2[2]>=Decimal(R2[2]) and box2[3]<=Decimal(R2[3]):break
    else:raise AssertionError(('third entrance failed',rows))
    result['third_entrance_steps']=len(rows);result['third_entrance_boxes']=rows
    # Arbitrarily high v >= 4: use independent ratio envelopes.
    s=I('.3');t=I('-.000002')
    hmin=ratios[2]*(-(a.sq()+d.sq()+2*a*s-2*d*t)/8).exp()
    rmax=ratios[1]
    high_mean=(rmax-a*hmin)/(1+rmax+hmin)
    assert high_mean.hi<Decimal('.20'),high_mean
    low_p=1/(1+(I('.2')/I('.05')).exp()/ratios[1])
    assert low_p.hi<Decimal('.025'),low_p
    result['forbidden_scales']={'high_mean_upper':str(high_mean.hi),
                                'low_binary_responsibility_upper':str(low_p.hi),
                                'cutoff':'.3','low_v_upper':'.05','high_v_lower':'4'}
    coeff=I('.999','1.001002')
    for name,uppermean in [('low',I('.025')),('high',I('.20'))]:
        drift=I('.001')*I('.702')+coeff*(uppermean-I('.3'))
        assert drift.hi<0
        result['forbidden_scales'][name+'_boundary_drift_upper']=str(drift.hi)
    sqrt2=I(2).sqrt()
    tracking=(sqrt2/500+I('.004'))/I('.32')
    comparison=(I('.5')+sqrt2/500+I('.04'))/I('.32')
    assert tracking.hi<(I(1)/32).lo and comparison.hi<2
    result['tracking_noise_scale_coefficient']=str(tracking.hi)
    result['reference_comparison_coefficient']=str(comparison.hi)
    # Terminal responsibility odds. In Q0,Q1 the binary squared-distance gap
    # is alpha/16. Every other gap is bounded below by these corner formulas.
    terminal_gaps={
        'Q0_target0_competitor2':a.sq()+d.sq()-I('.2')*a-I('.3')*d,
        'Q1_target1_competitor2':a.sq()+a+d.sq()-I('.3')*d,
        'Q2_target2_competitor0':-a.sq()-d.sq()+a+3*d,
        'Q2_target2_competitor1':-a.sq()-d.sq()+a+3*d+2,
    }
    assert all(gap.lo>Decimal('1.89') for gap in terminal_gaps.values())
    assert max((ratios[i]/ratios[j]).hi for i in range(3) for j in range(3) if i!=j)<2
    endpoint=I('.001')
    assert (32*(1+endpoint.sq())).hi<33
    assert (2*endpoint*(1+endpoint.sq())).hi<(33*I('1.89')).lo
    leakage=4*(-1/(33*endpoint)).exp()
    terminal_forcing=endpoint/500+I('1.001')*3*leakage
    assert terminal_forcing.hi<Decimal('.09')
    wrong_binary=I('1.297')*(-1/(100*endpoint*(1+endpoint.sq()))).exp()
    assert wrong_binary.hi<Decimal('.00006')
    wrong_drift=endpoint*(I('.502')+endpoint/100)+I('.999')*(I('.00006')-I('.49999'))
    assert wrong_drift.hi<0
    result['terminal_scalar_checks']={
        'nonbinary_squared_distance_gaps':{key:gap.json() for key,gap in terminal_gaps.items()},
        'forcing_displacement_upper':str(terminal_forcing.hi),
        'wrong_halfspace_binary_upper':str(wrong_binary.hi),
        'wrong_halfspace_drift_upper':str(wrong_drift.hi),
    }
    # Elementary endpoint inequalities; their ratios are increasing on the
    # stated alpha interval, as shown in the manuscript.
    small=I('.00001')
    soft=12*(-1/(64*small)).exp()
    tail=256*I(8).sqrt()*(-1/(I(4000000)*small.sq())).exp()
    assert soft.hi<(small*small*small/2).lo
    assert tail.hi<(small*small*small*small*small*small).lo
    assert (4*(-1/(64*I('.001'))).exp()).hi<Decimal('.000001')
    sharp_soft=4*(-1/(33*I('.001'))).exp()
    terminal_cov=9*sharp_soft/I('.001').sq()
    terminal_q=I('.75')+I('.25')*I('1.001')*terminal_cov
    assert terminal_q.hi<Decimal('.751')
    result['terminal_contraction_upper']=str(terminal_q.hi)
    result['risk_endpoint_inequalities']={'alpha':'.00001','soft_upper':str(soft.hi),
                                         'selection_tail_upper':str(tail.hi)}
    result['status']='PASS'
    (ROOT/'results/structural_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
