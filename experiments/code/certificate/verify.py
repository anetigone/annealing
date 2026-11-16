"""Directed-rounding certificate verifier; no NumPy/SciPy dependency.

Candidates are proposal coordinates, not assumed roots or trajectories. Every
existence, spectral, residual and inclusion assertion is checked independently.
"""
import argparse,json,time,sys,decimal
from pathlib import Path
from interval import I,D,DN,UP,norm,box,eigbounds,Model

ROOT=Path(__file__).parent
DATA=ROOT.parents[1]/"results"/"certificate"
DEFAULT_OUT=ROOT.parents[1]/"results"/"certificate_reproduction"

def diffnorm(a,b):return norm([I(x)-I(y) for x,y in zip(a,b)])
def sig(k,n=64):
    s=I(2)-I('1.65')*I(k)/I(n-1)
    # Same exact endpoints, with independent internal schedule perturbations.
    return s+(I('-0.0001','0.0001') if 0<k<n-1 else I(0))

def branch_checks(data,model):
    output=[]
    n=data['branch_segments'];assert n==3300
    for j in range(3):
        min_m=D(100);max_L=D(0);max_d=D(0);max_speed=D(0);max_glue=D(0)
        rows=data['branches'][j];assert len(rows)==n
        for i,c in enumerate(rows):
            # Every sigma in this closed slab, not just its endpoints.
            s=I(2)-I('0.0005')*I(i,i+1)
            m,L=eigbounds(model.h(box(c,'0.03'),s))
            assert m>D('0.16') and L<D('2.01'),('branch curvature',j,i,m,L)
            residual=norm(model.g([I(t) for t in c],j,s)).hi
            d=(I(residual)/I(m)).hi
            assert d<D('0.014'),('root inclusion',j,i,d)
            speed=(norm(model.gs(box(c,'0.014'),s))/I(m)).hi
            assert speed<D('4.5'),('branch speed',j,i,speed)
            if i:
                glue=(diffnorm(c,rows[i-1])+I('0.014')).hi
                assert glue<D('0.03'),('branch gluing',j,i,glue)
                max_glue=max(max_glue,glue)
            min_m=min(min_m,m);max_L=max(max_L,L);max_d=max(max_d,d);max_speed=max(max_speed,speed)
        # Glue the branch endpoints to the separately certified high and
        # correct-terminal stationary points.
        for center,name in [(rows[0],'high'),(rows[-1],'linear')]:
            assert (diffnorm(center,data['terminal_centers'][j][name])+I('0.003')).hi<D('0.03')
        output.append({'target':j,'min_hessian':str(min_m),'max_hessian':str(max_L),
          'max_root_distance':str(max_d),'max_speed':str(max_speed),'max_glue_distance':str(max_glue)})
        print('branch',j,output[-1],flush=True)
    return output

def finite_checks(data,model):
    out=[];allrows=[]
    for j in range(3):
        for name in ['constant','jump','linear','warmup']:
            # No claim about the constant method on target 0: that case is
            # close to a basin boundary and unnecessary for the strict gain.
            if j==0 and name=='constant':continue
            hist=data['paths'][j][name];n=len(hist)-1
            assert n==(32 if name=='warmup' else 64)
            r=diffnorm(model.init(j),hist[0]).hi
            max_r=r;max_rho=D(0);min_m=D(100);rows=[]
            for k in range(n):
                s=I(2) if name=='warmup' or (name=='jump' and k<12) else sig(k) if name=='linear' else I('0.35')
                m,L,rho=model.contraction(box(hist[k],r),s)
                residual=diffnorm(model.step([I(t) for t in hist[k]],j,s),hist[k+1]).hi
                rn=(I(rho)*I(r)+I(residual)).hi
                assert rn<D('0.2'),('finite radius too wide',j,name,k,rn)
                rows.append({'k':k,'sigma':[str(s.lo),str(s.hi)],'input_radius':str(r),'min_hessian':str(m),'max_hessian':str(L),'jacobian_norm':str(rho),'step_residual':str(residual),'output_radius':str(rn)})
                r=rn;max_r=max(max_r,r);max_rho=max(max_rho,rho);min_m=min(min_m,m)
            if name=='warmup':
                # Warmup ends close to the high-sigma branch. Center has a
                # certified stationary point in radius 0.003 below.
                distance=(diffnorm(hist[-1],data['terminal_centers'][j]['high'])+I(r)+I('0.003')).hi
                assert distance<D('0.006'),('warmup branch entrance',j,distance)
            else:
                target_error=(diffnorm(hist[-1],model.M[j])+I(r)).hi
                if name=='linear':assert target_error<D('0.015'),('recovery',j,target_error)
                center=data['terminal_centers'][j][name]
                trap_distance=(diffnorm(hist[-1],center)+I(r)).hi
                assert trap_distance<D('0.02'),('terminal trap entry',j,name,trap_distance)
                distance=target_error
            if j==2 and name=='linear':
                k=50;H=model.h(box(hist[k],rows[k]['input_radius']),sig(k));v=[I('.454'),I('-.891')]
                rayleigh=sum((v[a]*H[a][b]*v[b] for a in range(2) for b in range(2)),I(0))/sum((t.sq() for t in v),I(0))
                assert rayleigh.hi<D('-0.08'),('negative curvature witness',rayleigh)
                # sigma_50 and its jitter remain inside this fixed slab.
                slab=I(2)-I('0.0005')*I(2618,2620)
                assert sig(k).lo>slab.lo and sig(k).hi<slab.hi
                # Both possible cells are checked for a lower separation.
                lag=min((diffnorm(hist[k],data['branches'][j][i])-I(rows[k]['input_radius'])-I('0.014')).lo for i in [2618,2619])
                assert lag>D('0.50'),('large branch lag',lag)
                print('nonadiabatic witness',str(rayleigh.hi),'lag',str(lag),flush=True)
            item={'target':j,'method':name,'max_radius':str(max_r),'final_radius':str(r),
                'max_jacobian_norm':str(max_rho),'min_path_hessian':str(min_m),'error_or_warmup_distance':str(distance)}
            if j==2 and name=='linear':
                item['negative_rayleigh_upper_at_step_50']=str(rayleigh.hi)
                item['branch_distance_lower_at_step_50']=str(lag)
            out.append(item)
            allrows.append({'target':j,'method':name,'rows':rows})
            print('finite',j,name,out[-1],flush=True)
    return out,allrows

def traps(data,model):
    out=[]
    for j in range(3):
        for name in ['constant','jump','linear','high']:
            c=data['terminal_centers'][j][name];s=I(2) if name=='high' else I('0.35')
            m,L,rho=model.contraction(box(c,'0.02'),s)
            res=norm(model.g([I(x) for x in c],j,s)).hi
            d=(I(res)/I(m)).hi
            assert m>D('0.6') and L<D('2.01') and d<D('0.003') and rho<D('0.8'),('trap',j,name,m,L,d,rho)
            # T(B(c,R)) subset B(c,R) by residual + contraction.
            image=(I(rho)*I('0.02')+model.gamma*I(res)).hi
            assert image<D('0.02')
            errlow=(diffnorm(c,model.M[j])-I('0.02')).lo
            errhigh=(diffnorm(c,model.M[j])+I('0.02')).hi
            if name=='linear':
                assert errhigh<D('0.021')
                assert (diffnorm(c,model.M[j])+I('0.003')+I('0.0135')).hi<D('0.017')
            if name in ['constant','jump'] and j==2:assert errlow>D('1.50')
            if name=='jump' and j==0:assert errlow>D('1.29')
            out.append({'target':j,'method':name,'min_hessian':str(m),'root_offset':str(d),'contraction':str(rho),'trap_image_radius':str(image),'error_lower_bound':str(errlow),'error_upper_bound':str(errhigh)})
    return out

def global_checks(model):
    # Uniform coarse norm bounds used for the Gaussian tail argument.
    bmin,bmax=eigbounds(model.B)
    assert bmin>D('0.06') and bmax<D('1.01')
    assert max(norm(t).hi for t in model.M)<D('1.73')
    diam2=max(sum(((model.M[i][k]-model.M[j][k]).sq() for k in range(2)),I(0)).hi for i in range(3) for j in range(i))
    v=model.tau.sq()+4;q=4/v
    high_m=(I(bmin)+model.lam*q*(1-I(diam2)/(4*v))).lo
    assert high_m>D('0.3')
    # Full rank observation and noncollinear, separated templates.
    u=[model.M[1][k]-model.M[0][k] for k in range(2)]
    w=[model.M[2][k]-model.M[0][k] for k in range(2)]
    area=u[0]*w[1]-u[1]*w[0]
    assert area.hi<D('-1.46')
    assert min(diffnorm(model.M[i],model.M[j]).lo for i in range(3) for j in range(i))>D('1.69')
    # q >= .84 at every sigma >= .35; lambda*q >= .83.
    qmin=(I('0.35').sq()/(model.tau.sq()+I('0.35').sq())).lo
    assert qmin>D('0.84') and (model.lam*I(qmin)).lo>D('0.83')
    assert (model.gamma*(I(bmax)+model.lam)).hi<D(1)
    assert model.lam.hi<D('1.01') and model.gamma.lo>D('0.349')
    # Contraction of affine part I-gamma(B+lambda*q I), q<=1.
    assert (1-model.gamma*(I(bmin)+model.lam*I(qmin))).hi<D('0.69')
    # A priori |x0|<1.76+1.01|eta|, forcing <3.51+1.01|eta|.
    assert (I(bmax)*I('1.73')).hi<D('1.76')
    assert (I(bmax).sqrt()).hi<D('1.01')
    assert (I(bmax)*I('1.73')+model.lam*I('1.73')).hi<D('3.51')
    # |xK-mu| <= 7.5 + 2.2 |eta| for every K>=0 and schedule in [.35,2].
    assert (I('1.76')+model.gamma*I('3.51')/I('0.31')+I('1.73')).hi<D('7.5')
    assert (I('1.01')+model.gamma*I('1.01')/I('0.31')).hi<D('2.2')
    # Classical slow-tracking corollary: 32 high-noise calls, then 10001
    # linearly spaced calls from 2 to .35.
    rho=(1-model.gamma*I('0.16')).hi
    assert rho<D('0.945') and (model.gamma*I('2.01')).hi<D(1)
    assert (I('0.945')*I('0.015')+I('4.5')*I('0.000165')).hi<D('0.015')
    assert (I('4.5')*I('0.000165')).hi<=(I('0.055')*I('0.0135')).lo
    # Gaussian tail: for nu<=.0002, P(Ec)<1.2e-6 and second moment
    # E[|eta|^2 1_Ec]<1.296e-12. Mills' bound at t=5 suffices.
    denom=(2*I('3.14159')).sqrt()
    phi5=(I('-12.5').exp()/denom).hi
    assert phi5<D('0.0000015')
    tail=I('0.0000012');p=1-tail
    tail_m2=4*I('0.0002').sq()*(5*I('0.0000015')+2*I('0.0000003'))
    mse=I('0.015').sq()+I('112.5')*tail+I('9.68')*tail_m2
    assert mse.hi<D('0.000361')
    cmse=p*I('1.50').sq()/3;jmse=p*(I('1.50').sq()+I('1.29').sq())/3
    assert (cmse-mse).lo>D('0.7496') and (jmse-mse).lo>D('1.3043')
    assert (4*p/3-1).lo>D('0.3333317') and (5*p/3-1).lo>D('0.6666646')
    return {'B_min':str(bmin),'B_max':str(bmax),'q_min':str(qmin),'global_high_curvature':str(high_m),
       'ann_mse_upper':str(mse.hi),'constant_mse_lower':str(cmse.lo),'jump_mse_lower':str(jmse.lo),
       'constant_probability_gain_lower':str((4*p/3-1).lo),'jump_probability_gain_lower':str((5*p/3-1).lo)}

def main():
    if not __debug__:raise RuntimeError('Run without -O: assertions are the certificate checks.')
    ap=argparse.ArgumentParser();ap.add_argument('--skip-branch',action='store_true');ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument("--data",type=Path,default=DATA);args=ap.parse_args();DATA_DIR=args.data;OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
    start=time.time();data=json.loads((DATA_DIR/'candidates.json').read_text());model=Model()
    result={'status':'FINITE_ONLY_PASS' if args.skip_branch else 'PASS','parameter_radius':'0.0001','noise_event_radius':'0.001','schedule_jitter':'0.0001','precision':42,
       'python':sys.version,'decimal_version':decimal.__libmpdec_version__,
       'candidate_source':str(DATA_DIR/'candidates.json')}
    result['global']=global_checks(model)
    result['traps']=traps(data,model)
    result['finite'],rows=finite_checks(data,model)
    if not args.skip_branch:result['branches']=branch_checks(data,model)
    result['seconds']=time.time()-start
    (OUT/('quick_result.json' if args.skip_branch else 'certificate_result.json')).write_text(json.dumps(result,indent=2))
    (OUT/'finite_tubes.json').write_text(json.dumps(rows,indent=2))
    print('PASS',result['seconds'],flush=True)

if __name__=='__main__':main()
