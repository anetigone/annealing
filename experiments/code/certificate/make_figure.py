import argparse
"""Illustration only. Proof bounds are supplied by verify.py, not this plot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from explore import M,A,gh,solve

ROOT=Path(__file__).parent
DATA=ROOT.parents[1]/"results"/"certificate"
DEFAULT_OUT=ROOT.parents[1]/"results"/"certificate_reproduction"
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
parser.add_argument("--data",type=Path,default=DATA)
args=parser.parse_args()
DATA=args.data
OUT=args.out
OUT.mkdir(parents=True,exist_ok=True)
data=json.loads((DATA/'candidates.json').read_text());j=2
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11.8,4.9),layout='constrained')
branch=np.array(data['branches'][j],float)
axs[0].plot(branch[:,0],branch[:,1],ls='--',color='#1B8B72',lw=2,label='Stable stationary branch')
colors={'linear':'#155FAD','constant':'#D66C39','jump':'#68737E'}
labels={'linear':'Gradual path (64 calls)','constant':'Constant terminal noise','jump':'12 high calls, then jump'}
for name in ['constant','jump','linear']:
 h=np.array(data['paths'][j][name],float)
 axs[0].plot(h[:,0],h[:,1],color=colors[name],lw=2,label=labels[name])
 axs[0].scatter(*h[-1],color=colors[name],s=35,zorder=5)
axs[0].scatter(*M[2],marker='*',s=180,color='#1E3146',zorder=6)
axs[0].annotate('Correct template',M[2],xytext=(-1.47,1.34),fontsize=9)
path=np.array(data['paths'][j]['linear'],float);axs[0].scatter(*path[50],s=60,color='#9B3A76',zorder=6)
axs[0].annotate('Step 50',path[50],xytext=(-.73,.87),arrowprops={'arrowstyle':'-','color':'#9B3A76'},color='#9B3A76')
r50=solve(j,2-1.65*50/63,branch[2619])
axs[0].plot([path[50,0],r50[0]],[path[50,1],r50[1]],ls=':',lw=2,color='#9B3A76',zorder=5)
axs[0].scatter(*r50,s=65,facecolor='white',edgecolor='#1B8B72',lw=2,zorder=7)
axs[0].annotate('Branch at the same noise level',r50,xytext=(-1.49,.99),fontsize=8,color='#1B8B72',arrowprops={'arrowstyle':'-','color':'#1B8B72'})
axs[0].scatter(*path[0],marker='x',s=60,color='black',zorder=6)
axs[0].set(xlabel='x1',ylabel='x2',title='A. Same start and final objective; different basins',xlim=(-1.55,-.23),ylim=(-.35,1.5))
axs[0].legend(loc='lower left',fontsize=8,frameon=False)
minpath=[];minbranch=[]
for k,s in enumerate(np.linspace(2,.35,64)):
 minpath.append(np.linalg.eigvalsh(gh(path[k],j,s)[1])[0])
 root=solve(j,s,branch[min(3299,int((2-s)/.0005))]);minbranch.append(np.linalg.eigvalsh(gh(root,j,s)[1])[0])
axs[1].axhspan(-.3,0,color='#F7E9ED');axs[1].axhline(0,color='#ABB3BB',lw=.9)
axs[1].plot(range(64),minbranch,color='#1B8B72',ls='--',lw=2,label='At stationary branch')
axs[1].plot(range(64),minpath,color='#155FAD',lw=2,label='At the 64-call trajectory')
axs[1].scatter(50,minpath[50],s=45,color='#9B3A76',zorder=5)
axs[1].annotate('Verified at step 50:\nminimum eigenvalue < -0.08',xy=(50,minpath[50]),xytext=(14,-.23),fontsize=9,color='#9B3A76',arrowprops={'arrowstyle':'-','color':'#9B3A76'})
axs[1].set(xlabel='Iteration k (before call k + 1)',ylabel='Smallest Hessian eigenvalue',title='B. Fast success includes a negative-curvature phase',ylim=(-.32,1.02))
axs[1].legend(loc='upper left',fontsize=8,frameon=False)
fig.suptitle('R1: stable-branch existence and finite-budget success are different claims',fontsize=13,fontweight='bold')
fig.savefig(OUT/'R1_path_certificate.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'R1_path_certificate.pdf',bbox_inches='tight')
