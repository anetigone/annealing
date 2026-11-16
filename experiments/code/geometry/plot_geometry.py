import argparse
"""Nominal illustration. Certified bounds are in geometry_certificate.json."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.special import softmax
from make_sparse_candidates import M,A,solve,gh

ROOT=Path(__file__).parent
DATA=ROOT.parents[1]/"results"/"geometry"
DEFAULT_OUT=ROOT.parents[1]/"results"/"geometry_reproduction"
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
OUT=parser.parse_args().out
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#d17821','#286e9e','#2d8b65'];B=A.T@A

def matrix(s):
    v=.15**2+s*s;c=s*s/v;Q=B+c*np.eye(2)
    return v,c,Q

def centers(j,s):
    v,c,Q=matrix(s)
    return np.linalg.solve(Q,(B@M[j]+c*M).T).T

def values(s,x):
    v,c,Q=matrix(s);w=softmax(-np.sum((x-M)**2,axis=1)/(2*v))
    ell={(i,j):c/v*(M[i]-M[j])@np.linalg.solve(Q,M[i]-M[j]) for i in range(3) for j in range(i)}
    return max(ell.values())/4,sum(w[i]*w[j]*v for (i,j),v in ell.items())

ss=np.linspace(2,.35,350);curves=[];kap=[]
for j in range(3):
    prev=np.zeros(2);curve=[]
    for s in ss:
        prev=solve(j,s,prev);k,p=values(s,prev);curve.append(p)
        if j==0:kap.append(k)
    curves.append(curve)

fig,axes=plt.subplots(1,2,figsize=(12.5,5.8),gridspec_kw={'width_ratios':[1.06,1]})
ax=axes[0]
ax.axvspan(.35,.55,color='#d9eee1',alpha=.85)
ax.axvspan(.55,1.6,color='#f4ead4',alpha=.6)
ax.axvspan(1.6,2,color='#dae7f4',alpha=.8)
ax.plot(ss,kap,color='#273744',lw=2,label=r'Global diameter $\kappa$')
for j in range(3):ax.plot(ss,curves[j],color=colors[j],lw=1.7,label=fr'Branch competition $\Psi$, class {j}')
ax.axhline(1,color='#555',ls='--',lw=1)
ax.axvspan(1.53,1.54,color='#485765',alpha=.3)
ax.annotate('Certified convexity threshold\ninside [1.53, 1.54]',xy=(1.535,1),xytext=(.9,3.2),arrowprops={'arrowstyle':'->','lw':.8},fontsize=9)
ax.text(.45,14,'3 dominant\nballs',ha='center',va='top',fontsize=8.5,bbox={'facecolor':'#d9eee1','edgecolor':'none','alpha':.9,'pad':1})
ax.text(1.07,14,'101 continuous cells',ha='center',va='top',fontsize=9)
ax.text(1.8,14,'Global\nconvexity',ha='center',va='top',fontsize=9)
ax.set_yscale('log');ax.set_ylim(1e-4,18);ax.set_xlim(.35,2)
ax.set_xlabel(r'Denoiser noise $\sigma$');ax.set_ylabel('Dimensionless competition')
ax.set_title('A. Where each geometric certificate applies',loc='left',pad=12,fontweight='bold')
ax.legend(loc='lower right',frameon=True,facecolor='white',fontsize=8.3)
ax.grid(axis='y',alpha=.13)

ax=axes[1];j=2;z=centers(j,.35)[1];x0=B@M[j]
ax.add_patch(Circle(z,.35,fc='#f4d9d8',ec='#b35554',ls='--',lw=1.3,alpha=.8))
ax.plot(M[:,0],M[:,1],color='#a8afb4',lw=.8,ls=':')
for i,mu in enumerate(M):
    ax.scatter(*mu,color=colors[i],marker='*',s=140,zorder=5)
    ax.annotate(fr'$\mu_{i}$',mu,xytext=(7,5),textcoords='offset points')
ax.scatter(*z,c='#b35554',marker='+',s=90,zorder=6)
ax.annotate(r'$z_1$: wrong-component center',z,xytext=(-25,-38),textcoords='offset points',fontsize=8.7)
x=x0.copy();high=[x.copy()]
for _ in range(12):x=x-.35*gh(x,j,2)[0];high.append(x.copy())
high=np.array(high)
x=x0.copy();ann=[x.copy()]
for s in np.linspace(2,.35,64):x=x-.35*gh(x,j,s)[0];ann.append(x.copy())
ann=np.array(ann)
ax.plot(ann[:,0],ann[:,1],c=colors[2],lw=2,label='64-call gradual path')
ax.plot(high[:,0],high[:,1],c='#6a6f78',lw=1.5,ls=':',label='12 high-noise calls')
ax.scatter(*x0,c='#202a32',marker='o',s=38,zorder=7)
ax.annotate(r'$x_0=A^Ty$',x0,xytext=(-52,5),textcoords='offset points',fontsize=9)
ax.scatter(*high[-1],c='#6a6f78',marker='s',s=30,zorder=7)
ax.annotate(r'$x_{12}^{\rm high}$',high[-1],xytext=(12,-6),textcoords='offset points',fontsize=9)
ax.text(-1.48,-1.06,'Wrong ball: radius 0.35\nRobust to '+r'$\|\eta\|_\infty\leq0.1$'+' for constant-noise entry\n'+r'$\chi_{21}\approx0.1047<1/2$',fontsize=8.7,
        bbox={'boxstyle':'round,pad=.5','facecolor':'white','edgecolor':'#ddd'})
ax.set_aspect('equal');ax.set_xlim(-1.6,1.65);ax.set_ylim(-1.3,1.5)
ax.set_xlabel(r'$x_1$');ax.set_ylabel(r'$x_2$')
ax.set_title('B. Weak observation pull leaves a wrong trap',loc='left',pad=12,fontweight='bold')
ax.legend(loc='upper right',frameon=False,fontsize=8.6)
fig.subplots_adjust(bottom=.19,top=.9,wspace=.27)
fig.text(.5,.04,'Curves and paths show the nominal model at zero observation noise.\nThe certificate covers all stated parameter and noise boxes; nominal drawings are not the proof.',ha='center',fontsize=9,color='#555')
fig.savefig(OUT/'R1_geometric_conditions.png',dpi=180)
fig.savefig(OUT/'R1_geometric_conditions.svg')
