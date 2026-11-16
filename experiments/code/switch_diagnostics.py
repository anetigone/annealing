"""Compare one shared switch, an observation preset, and entrance feedback."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from mechanism_diagnostics import Family, observable_target, run_from_b


def metrics(f, x, truth, z):
    assert x.shape == truth.shape and np.isfinite(x).all()
    distances = np.sum((x[:, None] - f.mu)**2, axis=2)
    mode = distances.argmin(axis=1) == z
    near = distances[np.arange(len(z)), z] < .1**2
    loss = np.sum((x-truth)**2, axis=1)
    rows = [[float(quantity[z == j].mean()) for j in range(3)]
            for quantity in (mode, near, loss)]
    return dict(zip(('mode_correct', 'within_radius_.1', 'squared_error'),
                    ({'per_class': row, 'weighted': float(f.pi @ row)} for row in rows)))


def compare(alpha, rng):
    f = Family(alpha)
    calls = 180
    z_nominal = np.arange(3)
    b_nominal = f.mu @ f.B
    scan = []
    x = b_nominal.copy()
    for h in range(26):
        terminal = x.copy()
        for _ in range(calls-h):
            terminal = f.step(terminal, b_nominal, f.vend)
        scan.append({'h': h, 'switch_state': x.tolist(),
                     'metrics': metrics(f, terminal, f.mu, z_nominal)})
        x = f.step(x, b_nominal, .70)
    # Select a single shared h on the deterministic design cases, never per sample.
    h = min(scan, key=lambda row: row['metrics']['squared_error']['weighted'])['h']
    z = np.repeat(z_nominal, 1024)
    truth = f.mu[z] + alpha**2 * rng.standard_normal((len(z), 2))
    eta = alpha**2 * rng.standard_normal(truth.shape)
    b = (truth @ f.A + eta) @ f.A
    target = observable_target(b, alpha)
    preset = np.array([0, 25, 5])[target]
    outputs = {}
    for name, hold in [('shared', np.full(len(z), h)), ('preset', preset)]:
        x = b.copy()
        for k in range(calls):
            active = k < hold
            x[active] = f.step(x[active], b[active], .70)
            x[~active] = f.step(x[~active], b[~active], f.vend)
        outputs[name] = metrics(f, x, truth, z)
    x, meta = run_from_b(f, b, calls-25)
    outputs['feedback'] = metrics(f, x, truth, z)
    return {'alpha': alpha, 'calls': calls, 'theorem_budget': 25+f.budgets()['refine'],
            'nominal_scan': scan, 'selected_shared_h': h,
            'validation': outputs, 'exact_oracle_risk': f.oracle_risk(),
            'target_errors': int((target != z).sum()), 'forced': int(meta['forced'].sum()),
            'mean_feedback_middle_calls': float(f.pi @ [
                meta['gate_time'][z == j].mean() for j in range(3)]),
            'mean_preset_middle_calls': float(f.pi @ [
                preset[z == j].mean() for j in range(3)])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    seed = 20261103
    rng = np.random.default_rng(seed)
    data = {'seed': seed, 'n_per_class': 1024, 'middle_variance': .70,
            'rows': [compare(alpha, rng) for alpha in (1e-3, 1e-4, 1e-5, 1e-6)]}
    (args.out/'diagnostics.json').write_text(json.dumps(data, indent=2)+'\n')
