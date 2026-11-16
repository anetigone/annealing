"""Root-free geometric certificates for the three-template RED map.

All certificate model inputs are exact decimal rationals. Numerical candidates
for warm-up/tubes are proposals only, and are checked by interval residuals.
The kernel and this verifier still require independent review.
"""
import argparse,json,math,sys
from pathlib import Path
from decimal import Decimal, getcontext
from interval import I,DOWN,UP,NEAR,dot,norm,matvec,matmul,transpose,add,sub,eigbounds

getcontext().prec=60
ROOT=Path(__file__).resolve().parents[1]
MU0=[['1.38','-1.03'],['-.22','-.19'],['-1.16','1.22']]
A0=[['-.08','.24'],['-.95','-.31']]
O=[I('-.22'),I('-.19')]
P=[[I('-.94'),I('-1.41')],[I('1.41'),I('-.94')]]
D2=I('2.8717')
PINV=[[x/D2 for x in row] for row in transpose(P)]

class Model:
    def __init__(self,label,noise='.001',delta='.0001'):
        d=I(-Decimal(delta),Decimal(delta))
        self.mu=[[I(x)+d for x in row] for row in MU0]
        self.A=[[I(x)+d for x in row] for row in A0]
        self.B=matmul(transpose(self.A),self.A)
        self.tau=I('.15')+d;self.lam=I(1)+d;self.gamma=I('.35')+d
        self.a=[I(1)+d for _ in range(3)]
        eta=[I(-Decimal(noise),Decimal(noise)) for _ in range(2)]
        self.y=add(matvec(self.A,self.mu[label]),eta)
        self.b=matvec(transpose(self.A),self.y)
        self.coord_mu=[matvec(PINV,sub(m,O)) for m in self.mu]
        self.coord_B=matmul(matmul(PINV,self.B),P)
        self.coord_b=matvec(PINV,sub(self.b,matvec(self.B,O)))
    def weights(self,x,sigma):
        v=self.tau.sq()+sigma.sq();w=[]
        for i in range(3):
            den=I(1)
            for j in range(3):
                if i==j:continue
                diff=2*dot(x,sub(self.mu[j],self.mu[i]))+dot(self.mu[i],self.mu[i])-dot(self.mu[j],self.mu[j])
                den+=self.a[j]/self.a[i]*(diff/(2*v)).exp()
            w.append(1/den)
        return w
    def field(self,x,sigma):
        v=self.tau.sq()+sigma.sq();c=self.lam*sigma.sq()/v
        w=self.weights(x,sigma)
        mean=add(self.mu[1],add([w[0]*a for a in sub(self.mu[0],self.mu[1])],
                              [w[2]*a for a in sub(self.mu[2],self.mu[1])]))
        return add(sub(matvec(self.B,x),self.b),[c*a for a in sub(x,mean)])
    def update(self,x,sigma):
        v=self.tau.sq()+sigma.sq();c=self.lam*sigma.sq()/v;w=self.weights(x,sigma)
        M=[[(I(int(i==j))-self.gamma*(self.B[i][j]+(c if i==j else 0))) for j in range(2)] for i in range(2)]
        mean=add(self.mu[1],add([w[0]*a for a in sub(self.mu[0],self.mu[1])],
                              [w[2]*a for a in sub(self.mu[2],self.mu[1])]))
        return add(matvec(M,x),[self.gamma*(self.b[i]+c*mean[i]) for i in range(2)])
    def coord_delta(self,st,sigma):
        x=add(O,matvec(P,st));w=self.weights(x,sigma)
        c=self.lam*sigma.sq()/(self.tau.sq()+sigma.sq())
        mean=add(self.coord_mu[1],add([w[0]*a for a in sub(self.coord_mu[0],self.coord_mu[1])],
                                    [w[2]*a for a in sub(self.coord_mu[2],self.coord_mu[1])]))
        f=add(sub(matvec(self.coord_B,st),self.coord_b),[c*a for a in sub(st,mean)])
        return [-self.gamma*a for a in f]
    def hessian(self,x,sigma):
        v=self.tau.sq()+sigma.sq();c=self.lam*sigma.sq()/v;w=self.weights(x,sigma)
        C=[[I(0) for _ in range(2)] for _ in range(2)]
        for i in range(3):
            for j in range(i):
                d=sub(self.mu[i],self.mu[j]);p=w[i]*w[j]
                for k in range(2):
                    for l in range(2):C[k][l]+=p*d[k]*d[l]
        return [[self.B[k][l]+(c if k==l else 0)-c/v*C[k][l] for l in range(2)] for k in range(2)]

def grid(a,b,n):
    a=I(a);b=I(b)
    points=[a+(b-a)*i/n for i in range(n+1)]
    return [I(points[i].lo,points[i+1].hi) for i in range(n)]

def endpoint_trap(model,R='.35',n=70):
    center=[I('-.491'),I('-.1574')];radius=I(R)
    initial=norm(sub(model.b,center)).hi
    assert initial<radius.lo
    maxima={s:Decimal(0) for s in ['2','.35']};cells=0
    for dx in grid('-'+R,R,n):
        for dy in grid('-'+R,R,n):
            if (dx.sq()+dy.sq()).lo>radius.sq().hi:continue
            x=add(center,[dx,dy]);cells+=1
            for s in maxima:
                r=norm(sub(model.update(x,I(s)),center)).hi
                maxima[s]=max(maxima[s],r)
    for val in maxima.values():assert val<radius.lo,(R,n,maxima)
    separation=DOWN.subtract(norm(sub(model.mu[2],center)).lo,radius.hi)
    return dict(center=[str(x.lo) for x in center],radius=R,cells=cells,
                initial_distance_upper=str(initial),image_radius_upper={k:str(v) for k,v in maxima.items()},
                error_lower=str(separation))

def rectangle_invariance(model,rect,sigma,n=100):
    sl,sh,tl,th=rect
    # Own-coordinate monotonicity follows from H <= B+cI and gamma Q_ii<1.
    c=model.lam*sigma.sq()/(model.tau.sq()+sigma.sq())
    diag=[(model.gamma*(model.coord_B[i][i]+c)).hi for i in range(2)]
    assert max(diag)<1
    out=[]
    for axis,(low,high,otherlow,otherhigh) in enumerate([(sl,sh,tl,th),(tl,th,sl,sh)]):
        vals=[]
        for fixed in [low,high]:
            bounds=[]
            for z in grid(otherlow,otherhigh,n):
                st=[I(fixed),z] if axis==0 else [z,I(fixed)]
                a=st[axis]+model.coord_delta(st,sigma)[axis];bounds.append(a)
            vals.append([min(a.lo for a in bounds),max(a.hi for a in bounds)])
        assert vals[0][0]>=Decimal(low),(rect,axis,'lower',vals)
        assert vals[1][1]<=Decimal(high),(rect,axis,'upper',vals)
        out.append([[str(x) for x in v] for v in vals])
    return dict(rectangle=rect,face_bounds=out,own_diagonal_upper=[str(v) for v in diag])

def affine_progress_grid(model,rect,sigma,alpha,beta,ns=128,nt=40):
    # Certify s_next >= alpha*s + beta on the whole rectangle.
    sl,sh,tl,th=rect;minimum=Decimal('Infinity');cells=0
    sigmas=[sigma] if sigma.lo==sigma.hi else grid(str(sigma.lo),str(sigma.hi),5)
    for sig in sigmas:
        for s in grid(sl,sh,ns):
            for t in grid(tl,th,nt):
                ds=model.coord_delta([s,t],sig)[0]
                margin=(I(1)-I(alpha))*s+ds-I(beta)
                minimum=min(minimum,margin.lo);cells+=1
    assert minimum>0,(rect,alpha,beta,str(minimum))
    return dict(rectangle=rect,alpha=alpha,beta=beta,margin_lower=str(minimum),cells=cells)

def affine_progress(model,rect,sigma,alpha,beta,ns=128):
    """Reduce the transverse direction analytically before interval checking.

    P is a scaled orthogonal matrix, P^T P = D2 I. Thus the off-diagonal
    Jacobian of the coordinate map is
        J_01 = gamma*(-Bcoord_01 + c*D2/v*Cov(q)_01).
    We certify J_01 < 0 on each full transverse strip. The smallest next-s
    is then attained at its upper t face, without assuming contraction or
    knowing any stationary point. Strict interval bounds cover every point.
    """
    sl,sh,tl,th=rect
    minimum=Decimal('Infinity');maximum_cross=Decimal('-Infinity');cells=0
    sigmas=[sigma] if sigma.lo==sigma.hi else grid(str(sigma.lo),str(sigma.hi),5)
    for sig in sigmas:
        v=model.tau.sq()+sig.sq();c=model.lam*sig.sq()/v
        for s in grid(sl,sh,ns):
            x=add(O,matvec(P,[s,I(tl,th)]));w=model.weights(x,sig)
            cov=I(0)
            for i in range(3):
                for j in range(i):
                    d=sub(model.coord_mu[i],model.coord_mu[j])
                    cov+=w[i]*w[j]*d[0]*d[1]
            cross=model.gamma*(-model.coord_B[0][1]+c*D2/v*cov)
            maximum_cross=max(maximum_cross,cross.hi)
            assert cross.hi<0,('transverse monotonicity',rect,str(cross))
            ds=model.coord_delta([s,I(th)],sig)[0]
            margin=(I(1)-I(alpha))*s+ds-I(beta)
            minimum=min(minimum,margin.lo);cells+=1
    assert minimum>0,(rect,alpha,beta,str(minimum))
    return dict(rectangle=rect,alpha=alpha,beta=beta,margin_lower=str(minimum),
                transverse_derivative_upper=str(maximum_cross),cells=cells,
                method='Full-strip transverse monotonicity, then upper-face scalar envelope')

def tube(model,schedule,label,candidate_path=None):
    if candidate_path is None:
        raise ValueError('Provide fixed rational candidate centers; run make_candidates.py once to propose them')
    centers=[[I(v) for v in row] for row in candidate_path]
    assert len(centers)==len(schedule)+1
    R=norm(sub(model.b,centers[0])).hi;rows=[]
    for k,sig in enumerate(schedule):
        x=[a+I(-R,R) for a in centers[k]]
        m,L=eigbounds(model.hessian(x,sig))
        rh=max((1-model.gamma*I(m)).absmax(),(1-model.gamma*I(L)).absmax())
        residual=norm(sub(model.update(centers[k],sig),centers[k+1])).hi
        R=UP.add(UP.multiply(rh,R),residual)
        assert R<10,('tube divergence',label,k,str(R))
        rows.append(dict(k=k,sigma=sig.json(),rho=str(rh),residual=str(residual),radius=str(R)))
    final=[a+I(-R,R) for a in centers[-1]]
    return dict(candidate_centers=[[str(x.lo) for x in row] for row in centers],
                rows=rows,final_radius=str(R),final_center=[str(x.lo) for x in centers[-1]],
                final_coordinate_box=[a.json() for a in matvec(PINV,sub(final,O))],
                error_upper=str(UP.add(norm(sub(centers[-1],model.mu[label])).hi,R)))

def terminal_ball(model,label,R='.3'):
    center=[I(v) for v in MU0[label]];r=Decimal(R)
    box=[x+I(-r,r) for x in center]
    m,L=eigbounds(model.hessian(box,I('.35')))
    rho=max((1-model.gamma*I(m)).absmax(),(1-model.gamma*I(L)).absmax())
    residual=norm([model.gamma*x for x in model.field(center,I('.35'))]).hi
    assert m>0 and rho<1 and UP.add(UP.multiply(rho,r),residual)<r,(m,L,rho,residual)
    return dict(center=[str(v.lo) for v in center],radius=R,m=str(m),L=str(L),rho=str(rho),residual=str(residual))

def hitting_bound(alpha,beta,start,target,max_steps=100):
    x=Decimal(start);target=Decimal(target)
    for n in range(max_steps+1):
        if x>=target:return dict(steps=n,last_lower=str(x),target=str(target))
        x=DOWN.add(DOWN.multiply(Decimal(alpha),x),Decimal(beta))
    raise AssertionError('Insufficient hitting time')

def join_proof(result,delta):
    warm=result['warmup']['final_coordinate_box']
    assert Decimal(warm[0][0])>=Decimal('.20') and Decimal(warm[0][1])<=Decimal('.72')
    assert Decimal(warm[1][0])>=Decimal('.025') and Decimal(warm[1][1])<=Decimal('.10')
    middle=hitting_bound('.93','.040','.20','.52')
    assert middle['steps']<=32
    terminal1=hitting_bound('1.35','-.1715','.52','.70')
    terminal2=hitting_bound('.75','.245','.70','.90')
    transfer_steps=terminal1['steps']+terminal2['steps']
    assert transfer_steps<=20
    # P^T P = D2 I exactly; the terminal target rectangle is inside the ball.
    radius=(D2*(I('.10').sq()+I('.10').sq())).sqrt().hi
    ball=result['terminal_ball'];assert radius<Decimal(ball['radius'])
    refinement_steps=20-transfer_steps
    for _ in range(refinement_steps):
        radius=UP.add(UP.multiply(Decimal(ball['rho']),radius),Decimal(ball['residual']))
    class2_error=UP.add(radius,(2*I(delta).sq()).sqrt().hi)
    errors=[Decimal(result['label0_tube']['error_upper']),Decimal(result['label1_tube']['error_upper']),class2_error]
    assert max(errors)<Decimal('.2')
    for label in [0,1]:
        # Tube error is relative to the actual template; adjust for the fixed ball center.
        center_offset=(2*I(delta).sq()).sqrt().hi
        assert UP.add(errors[label],center_offset)<Decimal(result['label'+str(label)+'_terminal_ball']['radius'])
    failure=Decimal(result['endpoint_trap']['error_lower'])
    # Conditional on the bounded noise event, for independent uniform labels.
    squared_sum=Decimal(0)
    for e in errors:squared_sum=UP.add(squared_sum,UP.multiply(e,e))
    mse_success_upper=UP.divide(squared_sum,Decimal(3))
    mse_failure_lower=DOWN.divide(DOWN.multiply(failure,failure),Decimal(3))
    return dict(middle_hitting=middle,terminal_hitting1=terminal1,terminal_hitting2=terminal2,
                terminal_transfer_steps=transfer_steps,terminal_refinement_steps=refinement_steps,
                errors64_upper=[str(v) for v in errors],
                conditional_mse_upper=str(mse_success_upper),
                endpoint_only_conditional_mse_lower=str(mse_failure_lower),
                conditional_mse_gain_lower=str(DOWN.subtract(mse_failure_lower,mse_success_upper)))

def gaussian_risk_certificate(model,theorem,noise):
    """Uniform growth bound, plus Gaussian tails for nu <= noise/5."""
    bmin,bmax=eigbounds(model.B)
    mu_max=max(norm(mu).hi for mu in model.mu)
    amax=I(bmax).sqrt().hi
    tau_min=I(model.tau.lo);tau_max=I(model.tau.hi)
    cmin=(I(model.lam.lo)*I('.35').sq()/(tau_max.sq()+I('.35').sq())).lo
    cmax=(I(model.lam.hi)*I(2).sq()/(tau_min.sq()+I(2).sq())).hi
    assert (model.gamma*I(bmax+cmax)).hi<1
    rho=(1-model.gamma*I(bmin+cmin)).hi
    denom=1-I(rho)
    driven_const=(I(model.gamma.hi)*(I(bmax)+I(cmax))*I(mu_max)/denom).hi
    driven_noise=(I(model.gamma.hi)*I(amax)/denom).hi
    state_const=max((I(bmax)*I(mu_max)).hi,driven_const)
    state_noise=max(amax,driven_noise)
    err_const=(I(state_const)+I(mu_max)).hi
    assert err_const<6 and state_noise<Decimal('1.2')
    # pi > 3.14 gives a rigorous upper bound on the standard normal density at 5.
    phi_upper=(I('-12.5').exp()/I('6.28').sqrt()).hi
    probability_upper=(4*I(phi_upper)/5).hi
    assert probability_upper<Decimal('.0000012')
    nu=I(noise)/5
    second_tail=(4*nu.sq()*(5*I(phi_upper)+2*I(phi_upper)/5)).hi
    p_upper=Decimal('.0000012')
    mse_a=(I(theorem['conditional_mse_upper'])+72*I(p_upper)+I('2.88')*I(second_tail)).hi
    mse_b=((1-I(p_upper))*I(theorem['endpoint_only_conditional_mse_lower'])).lo
    return dict(gaussian_std_max=str(nu.hi),global_error_bound='6 + 1.2 * ||eta||_2',
                verified_global_constant=str(err_const),verified_global_noise_coefficient=str(state_noise),
                affine_contraction_upper=str(rho),outside_event_probability_upper=str(p_upper),
                outside_event_noise_second_moment_upper=str(second_tail),
                annealing_unconditional_mse_upper=str(mse_a),endpoint_only_unconditional_mse_lower=str(mse_b),
                unconditional_mse_gain_lower=str(DOWN.subtract(mse_b,mse_a)),
                success_probability_gain_lower=str((I(4)/3*(1-I(p_upper))-1).lo))

def main():
    if not __debug__:raise RuntimeError('Run without -O')
    ap=argparse.ArgumentParser();ap.add_argument('--noise',default='.005');ap.add_argument('--delta',default='.0001');ap.add_argument('--out',default='certificate_noise005.json');ap.add_argument('--stage',choices=['all','trap','geometry','tubes'],default='all');ap.add_argument('--progress-method',choices=['scalar','grid'],default='scalar');args=ap.parse_args()
    candidates=json.loads((ROOT/'results/candidates.json').read_text())
    m=Model(2,args.noise,args.delta);result={'status':'INCOMPLETE','scope':args.stage,'noise_box':args.noise,'parameter_radius':args.delta,'precision':50}
    if args.stage in ['all','trap']:
        result['endpoint_trap']=endpoint_trap(m)
        print('endpoint trap PASS',result['endpoint_trap'],flush=True)
    if args.stage in ['all','geometry']:
        sigma=I('.89','.91')
        for name,rect,ss in [('middle_outer',['.20','.72','.025','.10'],sigma),('middle_target',['.52','.72','.025','.10'],sigma),('terminal_outer',['.52','1.03','-.03','.10'],I('.35')),('terminal_target',['.90','1.03','-.03','.10'],I('.35'))]:
            result[name]=rectangle_invariance(m,rect,ss)
            print(name,'PASS',flush=True)
        progress=affine_progress if args.progress_method=='scalar' else affine_progress_grid
        result['middle_progress']=progress(m,['.20','.52','.025','.10'],sigma,'.93','.040')
        result['terminal_progress1']=progress(m,['.52','.70','-.03','.10'],I('.35'),'1.35','-.1715')
        result['terminal_progress2']=progress(m,['.70','.90','-.03','.10'],I('.35'),'.75','.245')
        print('progress PASS',flush=True)
    if args.stage in ['all','tubes']:
        result['warmup']=tube(m,[I(2)]*12,2,candidates['2'][:13])
        print('warmup',result['warmup']['final_coordinate_box'],flush=True)
        result['terminal_ball']=terminal_ball(m,2)
        for label in [0,1]:
            model=Model(label,args.noise,args.delta)
            result['label'+str(label)+'_tube']=tube(model,[I(2)]*12+[I('.899','.901')]*32+[I('.35')]*20,label,candidates[str(label)])
            result['label'+str(label)+'_terminal_ball']=terminal_ball(model,label)
            print('label',label,'final error',result['label'+str(label)+'_tube']['error_upper'],flush=True)
    if args.stage=='all':
        result['theorem']=join_proof(result,args.delta)
        result['gaussian_risk']=gaussian_risk_certificate(m,result['theorem'],args.noise)
    result['status']='PASS'
    (ROOT/'results'/args.out).write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',args.out,flush=True)

if __name__=='__main__':main()
