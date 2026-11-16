"""Floating-point controls; none of these experiments is an interval proof."""
import argparse
import json
from pathlib import Path
import numpy as np
from model import A, MU, TAU, EPS, P, step, run, schedules
from experiments import draw, metrics, paired_ci, posterior_mean

ROOT = Path(__file__).resolve().parents[1]

def controlled_run(y, schedule, terminal_lambda=1., fixed_c=False):
    b = y @ A
    x = b.copy()
    c_end = terminal_lambda * EPS**2 / (TAU**2 + EPS**2)
    for sigma in schedule:
        lam = c_end * (TAU**2 + sigma**2) / sigma**2 if fixed_c else terminal_lambda
        x = step(x, b, sigma, lam)
    return x

def fixed_tail_phase():
    levels = np.linspace(.55, 1.4, 86)
    holds = np.arange(45)
    b = MU @ (A.T @ A)
    x = np.broadcast_to(b, (len(levels), len(holds), 3, 2)).copy()
    for k in range(44):
        sig = np.where(k < (44-holds)[None, :], 2., levels[:, None])
        x = step(x, b, sig[:, :, None])
    for _ in range(20):
        x = step(x, b, EPS)
    e64 = np.linalg.norm(x-MU, axis=-1)
    for _ in range(256):
        x = step(x, b, EPS)
    e320 = np.linalg.norm(x-MU, axis=-1)
    np.savez_compressed(ROOT/'results/fixed_tail_phase.npz', levels=levels,
                        holds=holds, errors64=e64, errors320=e320)
    i = int(np.argmin(np.abs(levels-.9)))
    return dict(protocol='H^(44-h), M^h, L^20; 64 calls and 20 terminal calls for every point',
                levels_count=len(levels), holds_count=len(holds),
                all_class_success_count=int(np.all(e64 <= .2, axis=-1).sum()),
                sigma09_success_holds=holds[np.all(e64[i] <= .2, axis=-1)].tolist(),
                no_intermediate_success_count=int(np.all(e64[:, 0] <= .2, axis=-1).sum()),
                diagnostic_tail_calls=256)

def study(n):
    sch = schedules()
    result = {'status':'FLOATING_POINT_ONLY', 'seed':202610303,
              'n_test_per_class':n, 'fixed_tail':fixed_tail_phase(),
              'nominal_coefficient_ablation':{}, 'risk_common_lambda':{},
              'postselection_refinement':{}}
    for fixed_c in (False, True):
        name = 'fixed_c' if fixed_c else 'fixed_lambda'
        result['nominal_coefficient_ablation'][name] = {
            method: np.linalg.norm(controlled_run(MU@A.T, schedule, 1., fixed_c)-MU,
                                   axis=1).tolist() for method, schedule in sch.items()}
    rng = np.random.default_rng(result['seed'])
    z, truth, y = draw(rng, n, .15, .05)
    ls = y @ np.linalg.inv(A).T
    ls_metrics, ls_loss = metrics(ls, truth, z)
    result['least_squares'] = ls_metrics
    result['posterior_mmse'], _ = metrics(posterior_mean(y, .15, .05), truth, z)
    # lambda=0.3 was selected on the independent validation set in experiments.py.
    # This study evaluates every schedule at the same lambda and terminal target.
    outputs = {}
    for lam in (1., .3):
        result['risk_common_lambda'][str(lam)] = {}
        for method, schedule in sch.items():
            x = controlled_run(y, schedule, lam)
            rec, loss = metrics(x, truth, z)
            rec['paired_minus_LS'] = paired_ci(loss, ls_loss, z)
            result['risk_common_lambda'][str(lam)][method] = rec
            if lam == .3:
                outputs[method] = x
    result['fixed_c_risk_lambda03'] = {}
    for method in ('constant', 'linear', 'middle32'):
        x = controlled_run(y, sch[method], .3, True)
        rec, loss = metrics(x, truth, z)
        rec['paired_minus_LS'] = paired_ci(loss, ls_loss, z)
        result['fixed_c_risk_lambda03'][method] = rec
    # Extra diagnostic estimator: select a label using 64 RED calls, then one
    # closed-form conditional Gaussian solve. This is not the original output.
    B = A.T@A
    rho, nu = .15, .05
    c_star = nu**2/rho**2
    S = np.linalg.inv(B+c_star*np.eye(2))
    oracle_risk = float(np.trace(nu**2*S))
    costs = np.sum(((MU[:, None, :]-MU[None, :, :]) @ (c_star*S))**2, axis=-1)
    result['postselection_refinement'].update(
        caveat='Modified estimator: 64 denoiser calls plus one linear solve; not a same-terminal-RED comparison.',
        c_star=c_star, known_label_oracle_risk=oracle_risk, confusion_costs=costs.tolist(), methods={})
    for method, x in outputs.items():
        predicted = np.argmin(np.sum((x[:, None, :]-MU)**2, axis=-1), axis=1)
        refined = (y@A+c_star*MU[predicted])@S
        rec, loss = metrics(refined, truth, z)
        rec['selection_accuracy_before_refinement'] = float(np.mean(predicted==z))
        rec['oracle_plus_empirical_confusion_cost'] = oracle_risk+float(costs[z, predicted].mean())
        # Exact expectation is zero by conditional Gaussian orthogonality.
        difference = loss-costs[z, predicted]-oracle_risk
        se = np.sqrt(sum(difference[z==i].var(ddof=1)/n for i in range(3))/9)
        rec['identity_sample_residual'] = float(difference.mean())
        rec['identity_residual_normal95'] = [float(difference.mean()-1.96*se),
                                            float(difference.mean()+1.96*se)]
        confusion = np.zeros((3, 3), dtype=int)
        np.add.at(confusion, (z, predicted), 1)
        rec['confusion_counts'] = confusion.tolist()
        result['postselection_refinement']['methods'][method] = rec
    # The same location, with three scales, isolates the field rather than paths.
    loc = MU[1]+P@np.array([.3, .06])
    result['same_location_drift'] = {str(s):float(np.linalg.solve(P, step(loc, MU[2]@(A.T@A), s)-loc)[0])
                                   for s in (.35, .9, 2.)}
    (ROOT/'results/fair_controls.json').write_text(json.dumps(result, indent=2)+'\n')
    print('fixed tail', result['fixed_tail'], flush=True)
    print('coefficient ablation', result['nominal_coefficient_ablation'], flush=True)
    for lam, rows in result['risk_common_lambda'].items():
        print('risk common lambda',lam,{k:round(v['mse'],8) for k,v in rows.items()},flush=True)
    print('LS',ls_metrics['mse'],'posterior',result['posterior_mmse']['mse'],flush=True)
    print('refinement',{k:round(v['mse'],8) for k,v in result['postselection_refinement']['methods'].items()},flush=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=10000)
    study(ap.parse_args().n)
