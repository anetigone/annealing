import io,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from family import Family

ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/experiments.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                    'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(16.6,4.6),layout='constrained')
x=np.array([-np.log10(a['alpha']) for a in r['rate']]);ax=axs[0]
ax.plot(x,[a['cross_upper'] for a in r['rate']],color='#26718b',label='Sufficient middle calls',lw=2)
ax.plot(x,[a['observed_first_terminal_success'] for a in r['rate']],color='#d89737',marker='o',label='Observed first success',lw=2)
ax.plot(x,[a['cross_lower'] for a in r['rate']],color='#b36179',label='Permanent failure through this count',lw=2)
ax.set(xlabel=r'$\log_{10}(1/\alpha)$',ylabel='Middle-scale calls',title='A  Matching logarithmic computation orders')
ax.legend(frameon=False,fontsize=8)
ax=axs[1];f=Family(1e-4);offset=np.linspace(-.0067,.0067,151)
b=[f.phase_bounds(f.v0+f.alpha*z) for z in offset]
ax.plot(offset,[a['cross_upper'] for a in b],color='#26718b',lw=2,label='Sufficient count')
ax.plot(offset,[a['cross_lower'] for a in b],color='#b36179',lw=2,label='Proved insufficient count')
ax.set(xlabel=r'$(v-v_0)/\alpha$',ylabel='Middle-scale calls',title='B  Scale-dependent sufficient / failure bounds')
ax.text(.5,.50,'An upper-bound divergence is not\na claim about a physical transition.',
        transform=ax.transAxes,ha='center',fontsize=9,color='#555555')
ax.legend(frameon=False,fontsize=8)
ax=axs[2];alpha=np.array([a['alpha'] for a in r['risk']])
ax.loglog(alpha,[a['restricted_risk'] for a in r['risk']],'s-',color='#b36179',label='Restricted-scale RED (empirical)')
ax.loglog(alpha,[a['least_squares_exact_risk'] for a in r['risk']],'--',color='#777777',label='Least squares (exact)')
ax.loglog(alpha,[a['oracle_exact_risk'] for a in r['risk']],color='#26718b',label='Known-component risk (exact)',lw=2)
ax.loglog(alpha,[a['mid_risk'] for a in r['risk']],'o',mfc='none',mec='#d89737',ms=7,label='Middle-scale RED (empirical)')
ax.set(xlabel=r'Weak observation eigenvalue, $\alpha$',ylabel='Unconditional squared-error risk',
       title='C  Statistical separation in the same family')
ax.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,.80))
for extension in ('png','pdf'):
    buffer=io.BytesIO()
    fig.savefig(buffer,format=extension,dpi=190)
    target=ROOT/f'figures/theory_chain.{extension}'
    staging=target.with_suffix('.tmp')
    staging.write_bytes(buffer.getvalue())
    staging.replace(target)
print(ROOT/'figures/theory_chain.png')
