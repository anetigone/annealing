"""Reproducible floating-point diagnostics, separate from the proof verifier."""
import argparse,json
from pathlib import Path
import numpy as np
from model import MU,A,TAU,GAMMA,EPS,P,step,run,schedules

ROOT=Path(__file__).resolve().parents[1]

def draw(rng,n_per_class,within,noise):
    z=np.repeat(np.arange(3),n_per_class)
    x=MU[z]+within*rng.standard_normal((len(z),2))
    y=x@A.T+noise*rng.standard_normal((len(z),2))
    return z,x,y

def posterior_mean(y,within,noise):
    S=within**2*(A@A.T)+noise**2*np.eye(2)
    residual=y[:,None,:]-MU@A.T
    Sinv=np.linalg.inv(S)
    logits=-.5*np.einsum('nki,ij,nkj->nk',residual,Sinv,residual)
    w=np.exp(logits-logits.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    gain=within**2*A.T@Sinv
    conditional=MU+residual@gain.T
    return np.sum(w[:,:,None]*conditional,axis=1)

def metrics(x,truth,z):
    loss=np.sum((x-truth)**2,axis=1)
    predicted=np.argmin(np.sum((x[:,None,:]-MU)**2,axis=2),axis=1)
    good=predicted==z
    return dict(mse=float(loss.mean()),success_at_point2=float(np.mean(loss<=.04)),
                label_accuracy=float(good.mean()),wrong_count=int((~good).sum()),n=len(z),
                correct_conditional_mse=float(loss[good].mean()) if good.any() else None,
                wrong_conditional_mse=float(loss[~good].mean()) if (~good).any() else None,
                correct_risk_contribution=float(np.mean(loss*good)),
                wrong_risk_contribution=float(np.mean(loss*(~good)))),loss

def paired_ci(a,b,z):
    diff=a-b;means=[diff[z==i].mean() for i in range(3)]
    se=np.sqrt(sum(diff[z==i].var(ddof=1)/np.sum(z==i) for i in range(3))/9)
    avg=float(np.mean(means))
    return dict(difference=avg,normal_95_interval=[avg-1.96*se,avg+1.96*se],
                description='Stratified paired normal approximation; not a certified risk bound')

def risk_study(n=10000,nval=1000):
    lam_grid=[.001,.003,.01,.03,.1,.13,.3,1.,3.]
    ss=schedules();out={'seed_validation':202610301,'seed_test':202610302,
                       'lambda_grid':lam_grid,'n_per_class_test':n,'n_per_class_validation':nval,'settings':[]}
    rv=np.random.default_rng(out['seed_validation']);rt=np.random.default_rng(out['seed_test'])
    B=A.T@A;evals=np.linalg.eigvalsh(B);c=EPS**2/(TAU**2+EPS**2)
    for within,noise in [(0.,.05),(.15,.01),(.15,.05),(.15,.1)]:
        zv,xv,yv=draw(rv,nval,within,noise)
        chosen={};validation={}
        for method in ['constant','linear','middle32']:
            vals=[float(np.mean(np.sum((run(yv,ss[method],lam)-xv)**2,axis=1))) for lam in lam_grid]
            idx=int(np.argmin(vals));chosen[method]=lam_grid[idx];validation[method]=vals
        z,truth,y=draw(rt,n,within,noise)
        rec=dict(within_std=within,observation_std=noise,selected_lambda=chosen,
                 validation_mse=validation,controlled={},tuned={},comparisons={})
        losses={}
        for method,sch in ss.items():
            rec['controlled'][method],losses[method]=metrics(run(y,sch),truth,z)
        for method,lam in chosen.items():
            rec['tuned'][method],losses[method+'_tuned']=metrics(run(y,ss[method],lam),truth,z)
        rec['least_squares'],losses['least_squares']=metrics(y@np.linalg.inv(A).T,truth,z)
        rec['posterior_mmse'],losses['posterior_mmse']=metrics(posterior_mean(y,within,noise),truth,z)
        for method in ['linear','middle32','constant_tuned','linear_tuned','middle32_tuned']:
            rec['comparisons'][method+'_minus_LS']=paired_ci(losses[method],losses['least_squares'],z)
        rec['oracle_known_label']={
            'mse_at_lambda1':float(np.sum((within**2*c*c+noise**2*evals)/(evals+c)**2)),
            'optimal_c':float(noise**2/within**2) if within>0 else None,
            'optimal_lambda_at_fixed_epsilon':float((noise**2/within**2)*(TAU**2+EPS**2)/EPS**2) if within>0 else None,
            'least_squares_exact_risk':float(noise**2*np.sum(1/evals))}
        out['settings'].append(rec)
        print('risk',within,noise,'linear',rec['controlled']['linear']['mse'],'LS',rec['least_squares']['mse'],
              'tuned',chosen,flush=True)
    (ROOT/'results/risk.json').write_text(json.dumps(out,indent=2)+'\n')

def phase_study():
    levels=np.linspace(.55,1.4,86);holds=np.arange(1,52)
    warm=run(MU@A.T,[2.]*12);b=MU@(A.T@A)
    snapshots=[];x=np.broadcast_to(warm,(len(levels),3,2)).copy()
    for h in holds:
        x=step(x,b,levels[:,None]);snapshots.append(x.copy())
    # S,H,Z,2. Every path has exactly 64 calls before the primary measurement.
    x=np.stack(snapshots,axis=1);tail=52-holds
    for k in range(51):
        nxt=step(x,b,EPS);x=np.where((k<tail)[None,:,None,None],nxt,x)
    errors=np.linalg.norm(x-MU,axis=-1);acc=np.mean(errors<=.2,axis=-1)
    for k in range(256):x=step(x,b,EPS)
    tail_errors=np.linalg.norm(x-MU,axis=-1)
    np.savez_compressed(ROOT/'results/phase.npz',levels=levels,holds=holds,
                        errors64=errors,success64=acc,errors320=tail_errors)
    singles=np.array([np.linalg.norm(run(MU@A.T,[2.]*h+[EPS]*(64-h))-MU,axis=1) for h in range(64)])
    nominal={}
    for name,sch in schedules().items():
        y=MU@A.T;x,path=run(y,sch,trajectory=True)
        nominal[name]={'errors64':np.linalg.norm(x-MU,axis=1).tolist(),
                       'errors320':np.linalg.norm(run(y,list(sch)+[EPS]*256)-MU,axis=1).tolist(),
                       'schedule':sch,'trajectory':path.tolist()}
    out={'nominal':nominal,'single_jump_errors64':singles.tolist(),
         'single_jump_all_class_success_count':int(np.all(singles<=.2,axis=1).sum()),
         'phase_all_class_success_grid_count':int(np.all(errors<=.2,axis=-1).sum()),
         'phase_total_grid_count':int(acc.size),
         'diagnostic_tail_calls':256,'note':'Phase grid is empirical; terminal tails vary with hold length.'}
    (ROOT/'results/nominal.json').write_text(json.dumps(out,indent=2)+'\n')
    print('phase done',out['phase_all_class_success_grid_count'],flush=True)

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    p=np.load(ROOT/'results/phase.npz');nom=json.loads((ROOT/'results/nominal.json').read_text())
    risk=json.loads((ROOT/'results/risk.json').read_text());rec=next(s for s in risk['settings'] if s['within_std']==.15 and s['observation_std']==.05)
    fig,axes=plt.subplots(1,3,figsize=(16.3,4.8),constrained_layout=True)
    ax=axes[0];colors=['#e5e7eb','#f7c873','#8ac2c9','#246d8b'];cmap=ListedColormap(colors)
    im=ax.imshow((p['success64']*3).T,origin='lower',aspect='auto',extent=[.545,1.405,.5,51.5],cmap=cmap,vmin=-.5,vmax=3.5,interpolation='nearest')
    ax.scatter([.9],[32],s=70,facecolors='none',edgecolors='#ce3b44',linewidths=2)
    ax.set(xlabel='Intermediate noise level',ylabel='Intermediate calls',title='A  Empirical 64-call success region')
    fig.colorbar(im,ax=ax,ticks=[0,1,2,3],label='Successful templates (out of 3)',shrink=.83)
    ax=axes[1]
    for name,label,col in [('jump12','Direct jump','#989da3'),('linear','Linear annealing','#d78729'),('middle32','Intermediate plateau','#246d8b')]:
        path=np.array(nom['nominal'][name]['trajectory'])[:,2,:]
        st=(path-MU[1])@np.linalg.inv(P).T
        ax.plot(np.arange(65),st[:,0],label=label,color=col,lw=2)
    ax.axhline(.5,color='#ce3b44',ls='--',lw=1,label='Equal responsibilities')
    ax.axvspan(12,44,color='#246d8b',alpha=.07)
    ax.set(xlabel='Denoiser calls',ylabel='Progress toward template 2, s',title='B  Finite-time transfer of the hard class')
    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax=axes[2]
    entries=[rec['controlled']['linear'],rec['controlled']['middle32'],rec['tuned']['linear'],rec['tuned']['middle32']]
    labels=['Linear\n$\\lambda=1$','Plateau\n$\\lambda=1$','Linear\n$\\lambda=0.3$','Plateau\n$\\lambda=0.3$']
    good=[s['correct_risk_contribution'] for s in entries]
    bad=[s['wrong_risk_contribution'] for s in entries]
    ax.bar(labels,good,color='#246d8b',label='Correct-label contribution')
    ax.bar(labels,bad,bottom=good,color='#ce6670',label='Wrong-label contribution')
    ax.axhline(rec['least_squares']['mse'],color='#333333',ls='--',lw=1,label='Least squares')
    ax.axhline(rec['posterior_mmse']['mse'],color='#66955c',ls=':',lw=1.3,label='Matched posterior MMSE')
    ax.set(ylabel='Sample squared-error risk',title='C  Continuous GMM: risk decomposition',ylim=(0,.12))
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    fig.savefig(ROOT/'figures/annealing_progress.png',dpi=190)
    fig.savefig(ROOT/'figures/annealing_progress.pdf')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','risk','phase','plot'],default='all');ap.add_argument('--n',type=int,default=10000);a=ap.parse_args()
    if a.stage in ['all','phase']:phase_study()
    if a.stage in ['all','risk']:risk_study(a.n)
    if a.stage in ['all','plot']:plot()

if __name__=='__main__':main()
