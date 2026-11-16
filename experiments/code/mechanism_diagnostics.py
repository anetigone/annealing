"""Small, prespecified diagnostics for the structured three-template family.

No tuning or search. Float64 plots do not replace the interval certificates.
Run from the repository root; use a fresh --out for a new diagnostic batch.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'red_geometry_computation_risk/code'))
sys.path.insert(0, str(ROOT / 'red_scale_calibration/code'))
from family import Family
from coarse_control import observable_target, in_gate, run_from_b

COLORS = ['#2366a8', '#ce6c21', '#29835b', '#8b5ba5']


def trajectory(f, b, policy, calls, hold=0):
    x = b.copy().reshape(1, 2)
    target = observable_target(x, f.alpha)
    switched = bool(in_gate(x, target)[0]) if policy == 'coarse' else False
    h = 0 if switched else None
    rows = [x[0].copy()]
    for k in range(calls):
        if policy == 'coarse':
            v = f.vend if switched else .70
        else:
            v = f.vend if k >= hold else {'low': .05, 'high': 4., 'common': f.v0}[policy]
        x = f.step(x, b.reshape(1, 2), v)
        rows.append(x[0].copy())
        if policy == 'coarse' and not switched and (in_gate(x, target)[0] or k == 24):
            h, switched = k + 1, True
    return np.asarray(rows), h


def mechanism_figure(out):
    f = Family(1e-3)
    b = f.mu[1] @ f.B
    paths = {p: trajectory(f, b, p, 80, 25) for p in ['low', 'high', 'coarse']}
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)
    ax = axes[0]
    ax.add_patch(Rectangle((-1.18, -f.alpha/500), 1.48, 2.35+f.alpha/500,
                           facecolor='#d8e5f1', edgecolor='none', alpha=.65))
    ax.axvline(.3, color=COLORS[0], lw=.8, ls=':')
    gates = [(-.1, -.1, .5, .25), (.53, -.1, .57, .25), (-1.1, 1.5, .6, .6)]
    for z, (s, t, w, h) in enumerate(gates):
        ax.add_patch(Rectangle((s, t), w, h, facecolor='none', edgecolor=COLORS[2], lw=.9))
        ax.text(s + .03, t + h - .12, '$G_%d$' % z, color=COLORS[2], fontsize=8)
    ax.scatter(f.mu[:, 0], f.mu[:, 1], marker='*', s=65, c='black', zorder=5)
    for z, (s, t) in enumerate(f.mu):
        ax.annotate(r'$\mu_%d$' % z, (s, t), xytext=(5, 5), textcoords='offset points', fontsize=8)
    init = f.mu @ f.B
    ax.scatter(init[:, 0], init[:, 1], facecolors='none', edgecolors=COLORS[1], s=45, zorder=6)
    ax.annotate('$x_0^{(0)},x_0^{(1)}$', (0., 0.), xytext=(-.85, .45),
                arrowprops={'arrowstyle': '-', 'lw': .6}, fontsize=8)
    ax.annotate('$x_0^{(2)}$', init[2], xytext=(.25, 1.8),
                arrowprops={'arrowstyle': '-', 'lw': .6}, fontsize=8)
    ax.text(-1.12, 1.03, r'$\mathcal{W}_\alpha$'+'\n(clipped view)', fontsize=8, color=COLORS[0])
    ax.set(xlim=(-1.18, 1.22), ylim=(-.2, 2.35), xlabel='$s$', ylabel='$t$', title='(a) Geometry and entrances')
    ax = axes[1]
    s = np.linspace(-.01, .53, 240)
    for v, name, color in [(.05, '$v=.05$', COLORS[0]), (4., '$v=4$', COLORS[1]), (.70, '$v=.70$', COLORS[2])]:
        x = np.column_stack([s, np.full_like(s, .05)])
        drift = (f.step(x, np.broadcast_to(b, x.shape), v) - x)[:, 0] / .25
        ax.plot(s, drift, label=name, color=color, lw=1.5)
    ax.axhline(0, color='black', lw=.6)
    for s0 in [.3, .53]: ax.axvline(s0, ls=':', color='gray', lw=.6)
    ax.set(xlabel='$s$ at $t=.05$', ylabel=r'Horizontal drift $(T_v(x)-x)_1/\gamma$',
           title='(b) A nominal drift slice', xlim=(-.01, .54))
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    for p, name, color in [('low', 'Low then terminal', COLORS[0]), ('high', 'High then terminal', COLORS[1])]:
        path, _ = paths[p]
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.3, label=name)
    path, h = paths['coarse']
    ax.plot(path[:h+1, 0], path[:h+1, 1], color=COLORS[2], lw=1.7, label='Intermediate band')
    ax.plot(path[h:, 0], path[h:, 1], color=COLORS[2], ls='--', lw=1.7, label='Terminal refinement')
    ax.scatter(*path[h], s=42, facecolors='white', edgecolors='black', zorder=5)
    ax.annotate('$h=%d$' % h, path[h], xytext=(.67, .10),
                arrowprops={'arrowstyle': '-', 'lw': .6}, fontsize=8)
    ax.axvline(.3, ls=':', lw=.6, color=COLORS[0])
    ax.axvline(.53, ls=':', lw=.6, color=COLORS[2])
    ax.set(xlabel='$s$', ylabel='$t$', xlim=(-.13, 1.04), ylim=(-.008, .38), title='(c) Raw RED trajectories, class 1')
    ax.legend(fontsize=7.2, frameon=False, loc='upper right')
    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(8)
        ax.title.set_size(10)
    fig.savefig(out / 'mechanism.pdf')
    fig.savefig(out / 'mechanism.png', dpi=200)
    plt.close(fig)
    return {p: {'trajectory': x.tolist(), 'switch': h} for p, (x, h) in paths.items()}


def paired_risk(alpha, n, rng):
    f = Family(alpha)
    z = np.repeat(np.arange(3), n)
    truth = f.mu[z] + alpha**2 * rng.standard_normal((3*n, 2))
    y = truth @ f.A + alpha**2 * rng.standard_normal((3*n, 2))
    b = y @ f.A
    budget = f.budgets()
    k, h, r = budget['total'], budget['cross_upper'], budget['refine']
    coarse, meta = run_from_b(f, b, k-25)
    outputs = {'coarse': coarse}
    for name, stages in [('terminal_only', [(f.vend, k)]),
                         ('high_then_terminal', [(4., 25), (f.vend, k-25)]),
                         ('calibrated_common', [(f.v0, h), (f.vend, r)])]:
        outputs[name] = f.run(y, stages)
    oracle = f.oracle(y)[np.arange(3*n), z]
    def weighted(x): return float(sum(f.pi[j]*x[z == j].mean() for j in range(3)))
    def se(x): return float(np.sqrt(sum(f.pi[j]**2*x[z == j].var(ddof=1)/n for j in range(3))))
    oracle_loss = np.sum((oracle-truth)**2, axis=1)
    metrics = {}
    for name, x in outputs.items():
        loss = np.sum((x-truth)**2, axis=1)
        error2 = np.sum((x[:, None] - f.mu)**2, axis=2)
        distance = np.sum((x-oracle)**2, axis=1)
        metrics[name] = {'squared_euclidean_risk': weighted(loss), 'risk_standard_error': se(loss),
            'risk_over_exact_oracle': weighted(loss)/f.oracle_risk(),
            'paired_loss_difference_to_oracle': weighted(loss-oracle_loss),
            'paired_difference_standard_error': se(loss-oracle_loss),
            'distance_to_oracle_mean': weighted(distance),
            'mode_correct_rate': weighted((error2.argmin(axis=1) == z).astype(float)),
            'template_neighborhood_success_rate': weighted((error2[np.arange(3*n), z] < .1**2).astype(float))}
        assert np.isfinite(x).all() and x.shape == truth.shape
    return {'alpha': alpha, 'n_per_class': n, 'total_calls': k, 'common_hold': h,
        'terminal_minimum': r, 'coarse_gate_min': [int(meta['gate_time'][z == j].min()) for j in range(3)],
        'coarse_gate_max': [int(meta['gate_time'][z == j].max()) for j in range(3)],
        'wrong_observable_targets': int((meta['target'] != z).sum()), 'forced': int(meta['forced'].sum()),
        'exact_oracle_risk': f.oracle_risk(), 'empirical_oracle_risk': weighted(oracle_loss),
        'metrics': metrics}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n', type=int, default=1024)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    paths = mechanism_figure(args.out)
    rng = np.random.default_rng(20261102)
    rows = [paired_risk(a, args.n, rng) for a in [1e-3, 1e-4, 1e-5]]
    nominal = []
    for a in [1e-3, 1e-4, 1e-5, 1e-6]:
        f = Family(a)
        _, meta = run_from_b(f, f.mu @ f.B, f.budgets()['refine'])
        nominal.append({'alpha': a, 'gate_times': meta['gate_time'].tolist()})
    result = {'seed': 20261102, 'parameters': {'a': 1, 'd': 2, 'omega': .04, 'v0': 1, 'theta': 0},
        'selection': 'Prespecified nominal parameters; no search or tuning; fresh stratified Gaussian noise.',
        'scope': 'Float64 diagnostics only, not uniform proofs. Noise/truth/initialization/call budget/terminal scale shared across methods.',
        'metrics': 'pi-weighted class means; mode = nearest template; neighborhood radius = .1; squared Euclidean risk not dimension-normalized.',
        'nominal_entrance': nominal, 'paired_risk': rows, 'figure_class1_paths': paths}
    (args.out/'diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'nominal_entrance': nominal, 'paired_risk': rows}, indent=2))


if __name__ == '__main__':
    main()
