"""Publication-style figures from saved empirical results and exact model."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from model import A, MU, P, step

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})
phase=np.load(ROOT/'results/fixed_tail_phase.npz')
data=json.loads((ROOT/'results/fair_controls.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(16.6,4.5),layout='constrained')
ax=axs[0]
score=np.sum(phase['errors64']<=.2,axis=-1)
im=ax.imshow(score.T,origin='lower',aspect='auto',extent=[.545,1.405,-.5,44.5],
             cmap=ListedColormap(['#e5e7eb','#efc36d','#8ac2c9','#216b89']),vmin=-.5,vmax=3.5)
ax.scatter([.9],[32],s=75,facecolor='none',edgecolor='#d54a50',linewidth=2,
           label='Certified M32 center')
ax.set(xlabel='Intermediate noise level',ylabel='Intermediate calls, h',
       title='A  Empirical success with matched budgets')
ax.legend(loc='upper left',fontsize=8,framealpha=.9)
ax.text(.57,3,'64 total calls; 20 terminal calls',fontsize=9,color='#333333',
        bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':3})
fig.colorbar(im,ax=ax,ticks=[0,1,2,3],shrink=.85,label='Templates with error ≤ 0.2')
ax=axs[1]
s=np.linspace(.2,.52,350)
x=MU[1]+np.stack((s,np.full_like(s,.1)),axis=1)@P.T
b=MU[2]@(A.T@A)
for sig,col in [(2.,'#90979d'),(.9,'#216b89'),(.35,'#b46580')]:
    ds=(step(x,b,sig)-x)@np.linalg.inv(P).T
    ax.plot(s,ds[:,0],label=rf'$\sigma={sig:g}$',color=col,lw=2)
ax.plot(s,.04-.07*s,'--',color='#e4a33c',lw=1.7,label='Certified lower bound (middle)')
ax.axhline(0,color='#444444',lw=.8)
ax.set(xlabel='Progress coordinate, s',ylabel='One-step progress in s',
       title='B  A window of positive directional drift',ylim=(-.1,.045))
ax.legend(frameon=False,fontsize=8,loc='lower left')
ax.text(.205,.040,'Nominal field at t = 0.10',fontsize=8,color='#555555')
ax=axs[2]
entries=[data['risk_common_lambda'][l][m] for l,m in
         [('1.0','linear'),('1.0','middle32'),('0.3','linear'),('0.3','middle32')]]
labels=['Linear\nλ = 1','Plateau\nλ = 1',
        'Linear\nλ = 0.3','Plateau\nλ = 0.3']
good=np.array([r['correct_risk_contribution'] for r in entries])
bad=np.array([r['wrong_risk_contribution'] for r in entries])
ax.bar(labels,good,color='#216b89',label='Correct-label contribution')
ax.bar(labels,bad,bottom=good,color='#c56879',label='Wrong-label contribution')
ax.axhline(data['least_squares']['mse'],color='#30343b',ls='--',lw=1.3,label='Least squares')
ax.axhline(data['posterior_mmse']['mse'],color='#66955c',ls=':',lw=1.6,label='Matched posterior MMSE')
ax.set(ylabel='Sample squared-error risk',title='C  Selection and within-component recovery',ylim=(0,.115))
ax.legend(frameon=False,fontsize=8,loc='upper right')
fig.savefig(ROOT/'figures/mechanism_and_risk.png',dpi=190)
fig.savefig(ROOT/'figures/mechanism_and_risk.pdf')
plt.close(fig)
print(ROOT/'figures/mechanism_and_risk.png')
