"""Diagnose rare-sample influence and frozen-control decision probabilities."""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtr, roots_hermitenorm, roots_legendre

from budget_control_validation import load, model_for, write
from budget_mechanism import config, risk, trace
from scale_selection_followup import gate


ROOT = Path(__file__).resolve().parents[1] / "results"
BUDGETS = (22, 106)


def losses(model, truth, b, variances):
    """Individual calibration errors; same candidates as the original search."""
    x = np.broadcast_to(b, (len(variances), *b.shape)).copy()
    active = gate(b)[None, ..., None]
    result = np.full((2, 26, len(variances), *b.shape[:-1]), np.nan)
    for m in range(26):
        if m:
            x = np.where(active, model.step(x, b, variances[:, None, None, None]),
                         model.step(x, b, model.vend))
        final = x.copy()
        for k in range(m, 107):
            if k in BUDGETS and m < k:
                result[BUDGETS.index(k), m] = np.sum((final-truth)**2, -1)
            if k < 106:
                final = model.step(final, b, model.vend)
    return result


def select(scores, variances, budget, vend):
    ranked = [(float(scores[0, 0]), 0, float(vend))]
    ranked.extend((float(scores[m, j]), m, float(v))
                  for m in range(1, min(25, budget-1)+1) for j, v in enumerate(variances))
    value, calls, variance = min(ranked)
    return {"v": variance, "calls": calls, "calibration_risk": value}


def influence(out, spec, choices, calibration):
    rows, arrays = [], {}
    vs = np.array(spec["variance_candidates"])
    for mi in (5, 6):
        case, rec = spec["models"][mi], choices[2, mi]
        model = model_for(case)
        truth, b = calibration[f"truth_2_{mi}"], calibration[f"b_2_{mi}"]
        predicted = np.where(b[..., 1] > 1, 2, np.where(b[..., 0] > .0005, 1, 0))
        misclassified = np.argwhere(predicted != np.arange(3)[:, None])
        assert len(misclassified) == 1
        z, j = map(int, misclassified[0])
        errors = losses(model, truth, b, vs)
        arrays[f"errors_{mi}"] = errors
        full_weights = model.pi[:, None]*np.full(b.shape[:-1], 1/b.shape[1])
        full_scores = np.sum(errors*full_weights, axis=(-2, -1))
        saved = calibration[f"scores_2_{mi}"][[0, 4], 1]
        assert np.allclose(full_scores, saved, rtol=2e-14, atol=1e-18, equal_nan=True)
        keep = np.ones(b.shape[:-1], bool)
        keep[z, j] = False
        weights = model.pi[:, None]*keep/keep.sum(1)[:, None]
        dropped_scores = np.sum(errors*weights, axis=(-2, -1))
        selected = {}
        for bi, budget in enumerate(BUDGETS):
            full = select(full_scores[bi], vs, budget, model.vend)
            assert full["calls"] == rec["choices"][str(budget)]["noise_gated"]["calls"]
            assert full["v"] == rec["choices"][str(budget)]["noise_gated"]["v"]
            selected[str(budget)] = {"full": full,
                "without_one_sample": select(dropped_scores[bi], vs, budget, model.vend)}
        comparisons = {}
        for name, cfg in (("ready", config(case, rec, 22, "ready")),
                          ("nominal_tuned", config(case, choices[0, mi], 22, "noise_gated")),
                          ("noisy_tuned", config(case, rec, 22, "noise_gated"))):
            state = trace(model, b, cfg, True, 22)[-1]
            sq = np.sum((state-truth)**2, -1)
            comparisons[name] = {"config": cfg, "risk": risk(model, truth, state),
                "rare_sample_squared_error": float(sq[z, j]),
                "rare_sample_risk_contribution": float(model.pi[z]*sq[z, j]/b.shape[1]),
                "remaining_risk_contribution": float(risk(model, truth, state)-model.pi[z]*sq[z, j]/b.shape[1])}
        sigma = np.sqrt(model.tau**2*model.alpha**2+.005**2*model.alpha)
        gate_error = float(ndtr(-model.alpha/(2*sigma)))
        rows.append({"model_index": mi, "d": case["d"], "h": case["h"],
                     "removed_class": z, "removed_draw": j, "b": b[z, j].tolist(),
                     "class0_false_gate_probability": gate_error,
                     "probability_at_least_one_in_16": 1-(1-gate_error)**16,
                     "empirical_frequency_over_true_probability": (1/16)/gate_error,
                     "selections": selected, "original_config_comparisons": comparisons})
        print("influence", mi, selected, flush=True)
    np.savez_compressed(out / "calibration_losses.npz", **arrays)
    write(out / "influence.json", {"scope": "Post hoc leave-one-out influence, not a new selection method or evaluation claim.",
                                    "records": rows})


def endpoint(model, b, cfg, gated):
    x = b.copy()
    active = gate(b) if gated else np.ones(b.shape[:-1], bool)
    for k in range(106):
        if k < cfg["calls"]:
            x[active] = model.step(x[active], b[active], cfg["v"])
            x[~active] = model.step(x[~active], b[~active], model.vend)
        else:
            x = model.step(x, b, model.vend)
    return x


def decision_boundary(model, transverse, cfg, gated):
    """Finite-106-call 0-versus-1 decision, not an infinite-horizon basin claim."""
    lower, upper = np.full(len(transverse), -.01), np.full(len(transverse), .011)
    for values, expected in ((lower, 0), (upper, 1)):
        x = endpoint(model, np.stack([values, transverse], -1), cfg, gated)
        nearest = np.sum((x[:, None]-model.mu)**2, -1).argmin(-1)
        assert np.all(nearest == expected)
    for _ in range(43):
        middle = (lower+upper)/2
        x = endpoint(model, np.stack([middle, transverse], -1), cfg, gated)
        nearest = np.sum((x[:, None]-model.mu)**2, -1).argmin(-1)
        assert np.all(nearest != 2)
        lower = np.where(x[:, 0] < .5, middle, lower)
        upper = np.where(x[:, 0] >= .5, middle, upper)
    return (lower+upper)/2


def probability(out, plan, spec, choices, calibration, evaluation):
    rows, arrays = [], {}
    for ti, target in enumerate(plan["deterministic_probability_check"]["targets"]):
        ni, mi, policy, budget = (target[n] for n in ("noise_index", "model_index", "policy", "choice_budget"))
        case, rec = spec["models"][mi], choices[ni, mi]
        model, nu = model_for(case), spec["measurement_noise_sd"][ni]
        sigma1 = np.sqrt(model.tau**2*model.alpha**2+nu**2*model.alpha)
        sigma2 = np.sqrt(model.tau**2+nu**2)
        configurations = [("original", config(case, rec, budget, policy), policy == "noise_gated"),
                          ("short_choice", config(case, rec, 22, policy), policy == "noise_gated"),
                          ("ready", config(case, rec, 22, "ready"), True)]
        for label, cfg, gated in configurations:
            quadrature = []
            for nodes in (32, 64, 128):
                locations, weights = roots_hermitenorm(nodes)
                weights /= np.sqrt(2*np.pi)
                threshold = decision_boundary(model, sigma2*locations, cfg, gated)
                errors = np.array([weights @ ndtr(-threshold/sigma1),
                                   weights @ ndtr((threshold-model.alpha)/sigma1)])
                quadrature.append({"nodes": nodes, "class0_failure_probability": float(errors[0]),
                                   "class1_failure_probability": float(errors[1])})
            observed = {}
            for phase, data in (("calibration", calibration), ("evaluation", evaluation)):
                b = data[f"b_{ni}_{mi}"]
                x = endpoint(model, b, cfg, gated)
                nearest = np.sum((x[..., None, :]-model.mu)**2, -1).argmin(-1)
                observed[phase] = (nearest != np.arange(3)[:, None]).sum(1).tolist()
            standard_y = np.linspace(-4, 4, 121)
            boundary = decision_boundary(model, sigma2*standard_y, cfg, gated)
            arrays[f"y_{ti}_{label}"] = standard_y
            arrays[f"boundary_{ti}_{label}"] = boundary/model.alpha
            p0, p1 = (quadrature[-1][key] for key in ("class0_failure_probability", "class1_failure_probability"))
            rows.append({"target": ti, "noise_index": ni, "model_index": mi,
                         "policy": policy, "choice_budget": budget, "label": label,
                         "config": cfg, "quadrature": quadrature,
                         "miss_failure_in_16_by_class": [(1-p0)**16, (1-p1)**16],
                         "expected_failure_in_32_by_class": [32*p0, 32*p1],
                         "observed_wrong_by_class": observed})
            print("probability", ti, label, quadrature[-1], flush=True)
    np.savez_compressed(out / "decision_boundaries.npz", **arrays)
    write(out / "probabilities.json", {
        "scope": "Deterministic Gaussian quadrature and monotone bisection of the frozen 106-call map for classes 0 and 1. Not independent random validation or an interval-certified probability. Class-2 and b2>1 tails are outside this calculation; nearest class-2 is excluded on every root evaluation. Use node agreement as a numerical check, not a rigorous error bound.",
        "records": rows})


def refine_probability(out, plan, spec, choices):
    """Resolve the gate kink with split quadrature; no parameters are selected."""
    rows = []
    for ti, target in enumerate(plan["deterministic_probability_check"]["targets"]):
        ni, mi, policy, budget = (target[n] for n in ("noise_index", "model_index", "policy", "choice_budget"))
        model, nu = model_for(spec["models"][mi]), spec["measurement_noise_sd"][ni]
        cfg = config(spec["models"][mi], choices[ni, mi], budget, policy)
        gated = policy == "noise_gated"
        sigma1 = np.sqrt(model.tau**2*model.alpha**2+nu**2*model.alpha)
        sigma2 = np.sqrt(model.tau**2+nu**2)
        limits, kink = [-9., 9.], None
        if gated:
            # The ungated map evaluated at the gate threshold locates the kink.
            lo, hi = -9., 9.
            for _ in range(43):
                mid = (lo+hi)/2
                b = np.array([[model.alpha/2, mid*sigma2]])
                if endpoint(model, b, cfg, False)[0, 0] >= .5:
                    lo = mid
                else:
                    hi = mid
            kink = (lo+hi)/2
            limits = [-9., kink, 9.]
        values = []
        for nodes in (32, 64, 128):
            locations, weights = roots_legendre(nodes)
            total = np.zeros(2)
            for left, right in zip(limits[:-1], limits[1:]):
                y = (left+right)/2+(right-left)*locations/2
                threshold = decision_boundary(model, sigma2*y, cfg, False)
                if gated:
                    threshold = np.maximum(threshold, model.alpha/2)
                density = np.exp(-y*y/2)/np.sqrt(2*np.pi)
                errors = np.array([ndtr(-threshold/sigma1), ndtr((threshold-model.alpha)/sigma1)])
                total += (right-left)/2*(errors @ (weights*density))
            values.append({"nodes_per_segment": nodes, "class0_failure_probability": float(total[0]),
                           "class1_failure_probability": float(total[1])})
        # Compare clipped raw boundaries against the actual gated map.
        y = np.linspace(-4, 4, 17)*sigma2
        actual = decision_boundary(model, y, cfg, gated)
        raw = decision_boundary(model, y, cfg, False)
        predicted = np.maximum(raw, model.alpha/2) if gated else raw
        assert np.max(np.abs(actual-predicted)) < 3e-15
        rows.append({"target": ti, "gate_kink_in_b2_sd": kink, "quadrature": values,
                     "boundary_identity_max_error": float(np.max(np.abs(actual-predicted)))})
        print("refined probability", ti, kink, values[-1], flush=True)
    write(out / "probability_refinement.json", {
        "reason": "The max(gate threshold, raw platform decision boundary) creates a kink and slows Hermite convergence. Split Gauss-Legendre integration at that kink on [-9,9] standard deviations; omitted Gaussian b2 mass is below 2.3e-19. The same frozen controllers and finite-106-call decision are retained. This is numerical quadrature, not an interval certificate.",
        "records": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("influence", "probability", "refine"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = load(args.out / "plan.json")
    filename = {"influence": "influence.json", "probability": "probabilities.json",
                "refine": "probability_refinement.json"}[args.phase]
    if (args.out / filename).exists():
        raise FileExistsError("Outputs already exist")
    source = ROOT / "budget_control_validation"
    spec = load(source / "plan.json")
    choices = {(r["noise_index"], r["model_index"]): r for r in load(source / "choices.json")["records"]}
    calibration, evaluation = np.load(source / "calibration.npz"), np.load(source / "evaluation.npz")
    if args.phase == "influence":
        influence(args.out, spec, choices, calibration)
    elif args.phase == "probability":
        probability(args.out, plan, spec, choices, calibration, evaluation)
    else:
        refine_probability(args.out, plan, spec, choices)


if __name__ == "__main__":
    main()
