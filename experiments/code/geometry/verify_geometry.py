import argparse
"""Verify the geometric compression and its exact coverage. Standard library only."""
import sys,json,time
from pathlib import Path
from geometry import *
from sparse_branch import segment_check

ROOT=Path(__file__).parent
DATA=ROOT.parents[1]/"results"/"geometry"
DEFAULT_OUT=ROOT.parents[1]/"results"/"geometry_reproduction"
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
parser.add_argument("--data",type=Path,default=DATA)
args=parser.parse_args()
DATA=args.data
OUT=args.out
OUT.mkdir(parents=True,exist_ok=True)

def read_as_str(z):
    if isinstance(z,dict):return {str(k):read_as_str(v) for k,v in z.items()}
    if isinstance(z,list):return [read_as_str(v) for v in z]
    if isinstance(z,I):return [str(z.lo),str(z.hi)]
    return str(z) if hasattr(z,'as_tuple') else z

def convexity_checks(model):
    amin,amax=eigbounds(model.B)
    cross=dot(sub(model.M[1],model.M[0]),[sub(model.M[2],model.M[0])[1],-sub(model.M[2],model.M[0])[0]])
    assert amin>D('.06') and amax<D('1.01')
    assert cross.hi<D('-1.46')
    assert min(norm(sub(model.M[i],model.M[j])).lo for i in range(3) for j in range(i))>D('1.69')
    # Q increases with sigma, so its upper bound at 2 covers every scale.
    _,_,Qmax,_=matrices(model,I(2));_,global_L=eigbounds(Qmax)
    assert global_L<D('2.01') and (model.gamma*I(global_L)).hi<1
    low=lengths(model,I('1.53'))[(2,0)]/4
    high=max((x/4).hi for x in lengths(model,I('1.54')).values())
    assert low.lo>1 and high<1
    kap=max((x/4).hi for x in lengths(model,I('1.6')).values())
    _,_,Q,_=matrices(model,I('1.6'));alpha,L=eigbounds(Q)
    m=(I(alpha)*(1-I(kap))).lo
    assert m>D('.08')
    return {'transition_bracket':['1.53','1.54'],'global_L_upper':global_L,'noncollinearity_determinant':cross,'B_eigenvalue_bounds':[amin,amax],'kappa_lower_at_1_53':low.lo,
       'kappa_upper_at_1_54':high,'kappa_upper_at_1_6':kap,'global_m_from_1_6':m}

def conservative_tracking(model):
    """An intentionally coarse finite schedule derived from the compressed bounds."""
    _,_,Q,_=matrices(model,I(2));alpha,L=eigbounds(Q)
    kappa=max((x/4).hi for x in lengths(model,I(2)).values())
    m=(I(alpha)*(1-I(kappa))).lo
    rho=(1-model.gamma*I(m)).hi
    warm=[]
    for j in range(3):
        # The unique root is in the convex hull of these three algebraic centers.
        distance=max(norm(sub(model.init(j),single_center(model,j,l,I(2)))).hi for l in range(3))
        error=I(distance)
        for _ in range(40):error*=I(rho)
        assert error.hi<D('.003')
        warm.append({'initial_distance_upper':distance,'after_40_upper':error.hi})
    # 1,400,001 values, endpoints included. One final update at sigma=.35.
    step=I('1.65')/1400000
    drift=(70*step).hi
    contraction_margin=(model.gamma*I('.08')*I('.003')).lo
    assert drift<contraction_margin
    # The correct low-scale ball lies within .102 of the true template.
    for j in range(3):
        z=dominant_ball(model,j,j,I('.35'),'.1')
        assert z['error_upper']<D('.102')
    return {'warmup_calls':40,'ramp_calls':1400001,'total_calls':1400041,
       'high_m':m,'high_rho':rho,'warmup':warm,'ramp_step':step,
       'drift_upper':drift,'contraction_margin_lower':contraction_margin,
       'tracking_radius':'.003','reconstruction_error_upper':'.105'}

def branch_checks(model,data,convex):
    summary=[];records=[]
    for j,cells in enumerate(data['branches']):
        assert cells[0]['a']=='1.6' and cells[-1]['b']=='.55'
        worst_m=D(100);worst_margin=D(100);max_v=D(0)
        prev=None;rows=[]
        for cell in cells:
            if prev:
                assert prev['b']==cell['a'] and prev['cb']==cell['ca']
            z=segment_check(model,j,cell['a'],cell['b'],cell['ca'],cell['cb'],cell['radius'])
            assert z['pass'] and z['m']>D('.09')
            margin=(I(cell['radius'])-I(z['rootdist'])).lo
            assert margin>D('.006')
            # On the whole Euclidean tube: trace criterion gives m; H<=Q.
            assert (model.gamma*I(z['L'])).hi<1
            x=[I(min(D(cell['ca'][k]),D(cell['cb'][k])),max(D(cell['ca'][k]),D(cell['cb'][k])))+I(D(cell['radius']).copy_negate(),D(cell['radius'])) for k in range(2)]
            vel=(norm(model.gs(x,I(cell['b'],cell['a'])))/I(z['m'])).hi
            max_v=max(max_v,vel);worst_m=min(worst_m,z['m']);worst_margin=min(worst_margin,margin)
            rows.append({**cell,**z,'velocity_bound':vel});prev=cell
        terminal=dominant_ball(model,j,j,I('.35','.55'),'.1')
        assert terminal['invariance']<1 and terminal['competition']<1 and terminal['m']>D('.3')
        # A root lies within inv_ratio*R of the algebraic center; this leaves
        # a uniform .006 neighborhood for the root's stable tube.
        assert (I('.1')*(1-I(terminal['invariance']))).lo>D('.006')
        end_center=single_center(model,j,j,I('.55'))
        join=(norm(sub([I(x) for x in cells[-1]['cb']],end_center))+I(rows[-1]['rootdist'])).hi
        assert join<D('.1')
        xlow=[x+I('-.1','.1') for x in terminal['center']]
        vlow=(norm(model.gs(xlow,I('.35','.55')))/I(terminal['m'])).hi
        # A stationary point is a convex combination of the completed-square
        # component centers. This bounds the high-scale unique root.
        centers=[single_center(model,j,l,I('1.6','2')) for l in range(3)]
        xhigh=[I(min(z[k].lo for z in centers),max(z[k].hi for z in centers)) for k in range(2)]
        vhigh=(norm(model.gs(xhigh,I('1.6','2')))/I(convex['global_m_from_1_6'])).hi
        assert max(vhigh,vlow,max_v)<D(70)
        summary.append({'target':j,'middle_cells':len(cells),'middle_m_lower':worst_m,'stable_radius_lower':worst_margin,
            'middle_speed_upper':max_v,'high_speed_upper':vhigh,'low_speed_upper':vlow,'low_ball':terminal,'join_distance':join})
        records.append(rows)
    return summary,records

def warmup(model,j,hist):
    assert len(hist)==13
    r=norm(sub(model.init(j),[I(t) for t in hist[0]])).hi
    for k in range(12):
        _,_,rho=model.contraction(box(hist[k],r),I(2))
        residual=norm(sub(model.step([I(t) for t in hist[k]],j,I(2)),[I(t) for t in hist[k+1]])).hi
        r=(I(rho)*I(r)+I(residual)).hi
    return r

def trap_checks(model,data):
    output=[]
    for target,component,R in [(0,1,'.14'),(2,1,'.35')]:
        z=dominant_ball(model,target,component,I('.35'),R)
        assert z['invariance']<1 and z['competition']<1 and z['error_lower']>1
        assert (model.gamma*I(z['L'])).hi<1
        r=warmup(model,target,data['high_paths'][target])
        entry=(norm(sub([I(t) for t in data['high_paths'][target][-1]],z['center']))+I(r)).hi
        assert entry<D(R)
        output.append({'target':target,'component':component,'radius':R,'warmup_endpoint_radius':r,'jump_entry_distance':entry,**z})
    big=Model(delta='.0001',noise='.1')
    z=dominant_ball(big,2,1,I('.35'),'.35')
    assert z['invariance']<D('.24') and z['competition']<D('.89')
    assert z['error_lower']>D('1.11') and (big.gamma*I(z['L'])).hi<1
    entry=init_distance_to_center(big,2,1,I('.35'))
    assert entry<D('.28') and entry<D('.35')
    output.append({'target':2,'component':1,'noise_radius':'.1','constant_entry_distance':entry,**z})
    correct=[]
    for j in range(3):
        # Existence/stability of correct wells at larger noise, not trajectory
        # capture: this assertion alone never claims annealing succeeds.
        z=dominant_ball(big,j,j,I('.35'),'.1')
        assert z['invariance']<1 and z['competition']<1
        assert z['error_upper']<D('.175')
        correct.append({'target':j,**z})
    return output,correct

def pull_ratios(model):
    _,_,Q,Qi=matrices(model,I('.35'));out={}
    for i in range(3):
        for j in range(i):
            d=sub(model.M[i],model.M[j]);u=dot(d,matvec(matmul(Qi,model.B),d))/dot(d,d)
            assert u.hi<D('.5')
            out[str((i,j))]={'chi':u,'template_distance':norm(d),'measurement_distance':norm(matvec(model.A,d))}
    return out

def main():
    if not __debug__:raise RuntimeError('Assertions must be enabled.')
    start=time.time();model=Model();data=json.loads((DATA/'sparse_candidates.json').read_text())
    assert len(data['branches'])==3 and len(data['high_paths'])==3
    result={'status':'PASS','model_radius':'0.0001','branch_noise_radius':'0.001','expanded_constant_failure_noise_radius':'0.1'}
    result['convexity']=convexity_checks(model)
    result['branches'],rows=branch_checks(model,data,result['convexity'])
    result['coarse_tracking']=conservative_tracking(model)
    result['wrong_traps'],result['correct_traps_at_larger_noise']=trap_checks(model,data)
    result['pull_ratios']=pull_ratios(model)
    result['mixed_cells']=sum(len(x) for x in data['branches'])
    assert result['mixed_cells']==101
    result['seconds']=time.time()-start
    result['candidate_source']=str(DATA/'sparse_candidates.json')
    (OUT/'geometry_certificate.json').write_text(json.dumps(read_as_str(result),indent=2))
    (OUT/'sparse_cell_bounds.json').write_text(json.dumps(read_as_str(rows),indent=2))
    print(json.dumps({'status':'PASS','cells':result['mixed_cells'],'seconds':result['seconds'],
      'branch_counts':[r['middle_cells'] for r in result['branches']],
      'extended_failure':read_as_str(result['wrong_traps'][-1])},indent=2))

if __name__=='__main__':main()
