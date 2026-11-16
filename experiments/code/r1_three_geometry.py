#!/usr/bin/env python3
"""Geometry diagnostics for the selected three-template case; numerical only."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from scipy.special import softmax,logsumexp
from r1_three_templates import restore


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results"/"reproduction")
    out=parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    case=json.loads((out/'three_template_selected.json').read_text())
    means=np.array(case['means']);A=np.array(case['A']);tau=.15;eps=.35;K=64
    schedules={'constant':np.full(K,eps),'two_stage_H12':np.r_[np.full(12,2.),np.full(52,eps)],'linear_anneal':np.linspace(2.,eps,K)}
    def derivatives(x,y,sigma):
        v=tau*tau+sigma*sigma;q=sigma*sigma/v
        weights=softmax(-np.sum((x-means)**2,axis=1)/(2*v));mu=weights@means
        centered=means-mu;cov=(centered.T*weights)@centered
        grad=A.T@(A@x-y)+q*(x-mu)
        hess=A.T@A+q*np.eye(2)-q/v*cov
        return grad,hess
    def potential(x,y,sigma):
        v=tau*tau+sigma*sigma
        lp=logsumexp(-np.sum((x[:,None,:]-means[None,:,:])**2,axis=2)/(2*v),axis=1)-np.log(3)
        return .5*np.sum((x@A.T-y)**2,axis=1)-sigma*sigma*lp
    trajectories={};polish=[];entries=[];branches=[]
    for j in range(3):
        y=A@means[j];x0=A.T@y
        for name,ss in schedules.items():
            x=x0[None,:].copy();hist=[x[0].copy()]
            for sigma in ss:
                x=restore(x,y[None,:],means,A,[sigma]);hist.append(x[0].copy())
            history=np.array(hist);trajectories[(j,name)]=history
            all_polished=restore(history,np.tile(y,(K+1,1)),means,A,np.full(512,eps))
            labels=np.argmin(np.sum((all_polished[:,None,:]-means[None,:,:])**2,axis=2),axis=1)
            wrong=np.flatnonzero(labels!=j)
            permanent_entry=int(wrong[-1]+1) if len(wrong) else 0
            entries.append({'target':j,'method':name,'permanent_correct_terminal_basin_entry_step':permanent_entry if permanent_entry<=K else None,'terminal_basin_labels_by_step':labels.tolist()})
            p=all_polished[-1];g,h=derivatives(p,y,eps)
            polish.append({'target':j,'method':name,'error_K':float(np.linalg.norm(history[-1]-means[j])),'error_after_common_512_terminal_steps':float(np.linalg.norm(p-means[j])),'terminal_gradient_norm':float(np.linalg.norm(g)),'terminal_hessian_min_eigenvalue':float(np.linalg.eigvalsh(h)[0]),'terminal_objective_after_polish':float(potential(p[None,:],y,eps)[0])})
        prev=np.zeros(2)
        for sigma in np.linspace(2.,eps,201):
            sol=root(lambda x:derivatives(x,y,sigma)[0],prev,jac=lambda x:derivatives(x,y,sigma)[1],tol=1e-10)
            g,h=derivatives(sol.x,y,sigma)
            branches.append({'target':j,'sigma':float(sigma),'x1':float(sol.x[0]),'x2':float(sol.x[1]),'gradient_norm':float(np.linalg.norm(g)),'hessian_min_eigenvalue':float(np.linalg.eigvalsh(h)[0]),'solver_success':bool(sol.success)})
            prev=sol.x
    with (out/'three_template_polish.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(polish[0]));w.writeheader();w.writerows(polish)
    with (out/'three_template_stationary_branch.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(branches[0]));w.writeheader();w.writerows(branches)
    (out/'three_template_basin_entry.json').write_text(json.dumps(entries,indent=2))
    np.savez_compressed(out/'three_template_trajectories.npz',**{f'target{j}_{name}':v for (j,name),v in trajectories.items()})
    # Finite-budget fixed-objective basin comparison, for target 0.
    xx=np.linspace(-2.1,2.1,221);yy=np.linspace(-1.7,1.7,201);X,Y=np.meshgrid(xx,yy);grid=np.c_[X.ravel(),Y.ravel()]
    observed=A@means[0];data=np.tile(observed,(len(grid),1));basin_counts={}
    for name in ['constant','linear_anneal']:
        e=restore(grid,data,means,A,schedules[name]);e=restore(e,data,means,A,np.full(256,eps))
        labels=np.argmin(np.sum((e[:,None,:]-means[None,:,:])**2,axis=2),axis=1)
        basin_counts[name]={str(j):int(np.sum(labels==j)) for j in range(3)}
    # These grid proportions are descriptive, not probabilities for the main observation initializer.
    (out/'three_template_basin_grid.json').write_text(json.dumps({'target':0,'grid_size':len(grid),'domain':[-2.1,2.1,-1.7,1.7],'post_budget_common_polish_steps':256,'counts':basin_counts},indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),constrained_layout=True)
    F=potential(grid,observed,eps).reshape(X.shape);axes[0].contour(X,Y,F,levels=22,colors='#D8DEE4',linewidths=.65)
    colors={'constant':'#CD633C','two_stage_H12':'#697684','linear_anneal':'#126A97'}
    labels={'constant':'Constant terminal noise','two_stage_H12':'High noise, then jump','linear_anneal':'Linear annealing'}
    for name in schedules:
        path=trajectories[(0,name)]
        axes[0].plot(path[:,0],path[:,1],color=colors[name],lw=2.2,label=labels[name],ls='--' if name=='two_stage_H12' else '-')
        axes[0].scatter(*path[-1],color=colors[name],s=45,zorder=6)
    axes[0].scatter(means[:,0],means[:,1],marker='*',s=150,color='#253E52',zorder=7)
    axes[0].scatter(*trajectories[(0,'constant')][0],marker='x',s=65,color='black',zorder=8)
    for j,m in enumerate(means):axes[0].annotate('True template' if j==0 else f'Template {j}',m+np.array([.05,.06]),fontsize=9)
    axes[0].set(title='A. A gradual path reaches a different final basin',xlabel='x1',ylabel='x2',xlim=(-1.5,1.7),ylim=(-1.45,1.55),aspect='equal')
    axes[0].legend(frameon=False,loc='upper right',fontsize=8)
    with (out/'three_template_validation.csv').open() as f:rows=list(csv.DictReader(f))
    subset=[r for r in rows if float(r['noise_std'])==.05 and r['initialization']=='backprojection']
    values=[float(r['template_success'])*100 for r in subset]
    names=[r['method'] for r in subset]
    bars=axes[1].bar(range(3),values,color=[colors[n] for n in names],width=.6)
    for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value+1.4,f'{value:.3f}%',ha='center',fontweight='bold')
    axes[1].set_xticks(range(3),['Constant','High then jump','Linear annealing'])
    axes[1].set(title='B. Paired noisy observations, same 64 calls',ylabel='Within radius 0.2 of the true template (%)',ylim=(0,108))
    fig.suptitle('R1 three-template case: gradual annealing survives measurement initialisation',fontsize=13,fontweight='bold')
    fig.text(.5,-.045,'x0 = A^T y; noise std = 0.05; n = 100,000. Three equally likely templates; exact MMSE denoiser.\nExploratory synthetic construction selected from 300 draws. Same final objective. Numerical evidence; no three-template theorem yet.',ha='center',fontsize=9,color='#697684')
    fig.savefig(out/'r1_three_template_geometry.png',dpi=180,bbox_inches='tight');plt.close(fig)
    branch_summary=[{'target':j,'min_sampled_hessian_eigenvalue':min(r['hessian_min_eigenvalue'] for r in branches if r['target']==j),'max_gradient_norm':max(r['gradient_norm'] for r in branches if r['target']==j)} for j in range(3)]
    print(json.dumps({'basin_entries':[{k:v for k,v in row.items() if k!='terminal_basin_labels_by_step'} for row in entries],'polish':polish,'branch':branch_summary,'basin_grid':basin_counts},indent=2))


if __name__=='__main__':main()
