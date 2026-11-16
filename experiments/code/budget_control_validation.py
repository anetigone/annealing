"""Freeze noisy calibration and independently validate budget-matched controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from scale_selection_diagnostic import ALPHA, E1, E2, rectangles
from scale_selection_followup import V_GRID, gate, model_for, write
from switch_ready_short_budget import PRACTICAL_DIFF, load


ROOT = Path(__file__).resolve().parents[1] / "results"
BUDGETS = (22, 25, 32, 40, 106)
POLICIES = ("terminal", "ready", "same_center_direct", "proxy_shared",
            "noise_shared", "noise_gated")


def plan(out):
    source = load(ROOT / "budget_fair_controls/plan.json")
    cases = [c for c in source["models"] if c["included"]]
    assert len(cases) == 12
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", {
        "source": "experiments/results/budget_fair_controls/plan.json",
        "models": cases, "budgets": list(BUDGETS),
        "measurement_noise_sd": [ALPHA**2, .001, .005],
        "calibration_seed": 2026111601, "evaluation_seed": 2026111602,
        "calibration_draws_per_class": 16, "evaluation_draws_per_class": 32,
        "variance_candidates": V_GRID.tolist(), "maximum_prefix": 25,
        "comparison_threshold": PRACTICAL_DIFF,
        "selection": "Minimize calibration prior-weighted squared Euclidean error at each "
                     "budget; ties: shorter prefix then smaller variance. Include terminal-only; "
                     "m <= min(25,K-1), so every candidate ends with a terminal call.",
        "first_budget": "22 replaces the historical 21-call readout: a frozen direct-entry "
                        "prefix can use 21 calls, so 22 leaves at least one terminal call for all policies.",
        "gate": "b2 <= 1 and b1 > alpha/2; no true label or evaluation result is an input",
        "random_streams": "SeedSequence([phase_seed, noise_index, model_index]); "
                          "independent calibration/evaluation; paired across policies",
        "scope": "Previously selected 12 geometries, new noise only. All observations retained; "
                 "event E and classifier correctness are evaluation strata. No new certificate "
                 "outside E, and no claim of global control optimality.",
    })


def observations(model, spec, phase, noise_index, model_index):
    seed = np.random.SeedSequence([spec[f"{phase}_seed"], noise_index, model_index])
    rng = np.random.default_rng(seed)
    shape = (3, spec[f"{phase}_draws_per_class"], 2)
    truth = model.mu[:, None, :] + model.tau * rng.standard_normal(shape)
    noise = spec["measurement_noise_sd"][noise_index] * rng.standard_normal(shape)
    b = (truth @ model.A + noise) @ model.A
    return truth, b


def tune(model, truth, b, spec):
    variances = np.array(spec["variance_candidates"])
    budgets = spec["budgets"]
    # Policy, variance, class, draw, coordinate. Both controls see the same data.
    x = np.broadcast_to(b, (2, len(variances), *b.shape)).copy()
    active = gate(b, model.alpha)[None, ..., None]
    scores = np.full((len(budgets), 2, spec["maximum_prefix"] + 1, len(variances)), np.nan)
    for m in range(spec["maximum_prefix"] + 1):
        if m:
            stepped = model.step(x, b, variances[None, :, None, None, None])
            stepped[1] = np.where(active, stepped[1], model.step(x[1], b, model.vend))
            x = stepped
        final = x.copy()
        for k in range(m, max(budgets) + 1):
            if k in budgets and m < k:
                errors = np.sum((final - truth)**2, axis=-1).mean(axis=-1)
                scores[budgets.index(k), :, m] = errors @ model.pi
            if k < max(budgets):
                final = model.step(final, b, model.vend)
    choices = {}
    for j, budget in enumerate(budgets):
        choices[budget] = {}
        cap = min(spec["maximum_prefix"], budget - 1)
        for p, name in enumerate(("noise_shared", "noise_gated")):
            ranked = [(float(scores[j, p, 0, 0]), 0, float(model.vend))]
            ranked.extend((float(scores[j, p, m, v]), m, float(variance))
                          for m in range(1, cap + 1) for v, variance in enumerate(variances))
            error, m, variance = min(ranked)
            if not np.isfinite(error):
                raise ValueError("Nonfinite calibration error")
            choices[budget][name] = {"v": variance, "calls": m,
                                     "calibration_risk": error, "candidates": len(ranked)}
    return choices, scores


def calibrate(out):
    spec = load(out / "plan.json")
    if (out / "choices.json").exists():
        raise FileExistsError("Calibration choices already exist")
    records, arrays, proxies = [], {}, {}
    for mi, case in enumerate(spec["models"]):
        model = model_for(case)
        truth = model.mu[:, None, :]
        proxies[mi] = tune(model, truth, truth @ model.B, spec)[0]
    for ni, nu in enumerate(spec["measurement_noise_sd"]):
        for mi, case in enumerate(spec["models"]):
            model = model_for(case)
            truth, b = observations(model, spec, "calibration", ni, mi)
            choices, scores = tune(model, truth, b, spec)
            for budget, value in choices.items():
                value["proxy_shared"] = proxies[mi][budget]["noise_shared"]
            arrays[f"scores_{ni}_{mi}"] = scores
            arrays[f"truth_{ni}_{mi}"] = truth
            arrays[f"b_{ni}_{mi}"] = b
            records.append({"noise_index": ni, "model_index": mi, "nu": nu,
                            "d": case["d"], "h": case["h"], "choices": choices})
            print(f"calibrated nu={nu:g}, d={case['d']}, h={case['h']}", flush=True)
    np.savez_compressed(out / "calibration.npz", **arrays)
    write(out / "choices.json", {"records": records})


def simulate(model, b, case, config, gated, budget):
    bounds = np.array(rectangles(case["d"], float))
    low, high = bounds[..., 0][:, None, :], bounds[..., 1][:, None, :]
    active = gate(b, model.alpha) if gated else None
    x, first = b.copy(), np.full(b.shape[:-1], -1, dtype=int)
    for k in range(budget + 1):
        inside = np.all((low <= x) & (x <= high), axis=-1)
        first[(first < 0) & inside] = k
        if k == config["calls"]:
            at_switch = inside.copy()
        if k < budget:
            if k < config["calls"]:
                if gated:
                    x[active] = model.step(x[active], b[active], config["v"])
                    x[~active] = model.step(x[~active], b[~active], model.vend)
                else:
                    x = model.step(x, b, config["v"])
            else:
                x = model.step(x, b, model.vend)
    return x, first, at_switch


def metrics(model, truth, x):
    squared = np.sum((x - truth)**2, axis=-1)
    nearest = np.sum((x[..., None, :] - model.mu)**2, axis=-1).argmin(axis=-1)
    near = np.sum((x - model.mu[:, None, :])**2, axis=-1) <= .1**2
    if not np.isfinite(squared).all():
        raise ValueError("Nonfinite evaluation error")
    return {"risk": float(model.pi @ squared.mean(axis=1)),
            "class_risk": squared.mean(axis=1).tolist(),
            "mode_counts": (nearest == np.arange(3)[:, None]).sum(axis=1).tolist(),
            "neighborhood_counts": near.sum(axis=1).tolist()}


def summarize(records, spec):
    result = {}
    for ni, nu in enumerate(spec["measurement_noise_sd"]):
        rows = [r for r in records if r["noise_index"] == ni]
        by_budget = {}
        for k in spec["budgets"]:
            by_policy = {}
            for policy in POLICIES:
                values = [r["policies"][str(k)][policy] for r in rows]
                by_policy[policy] = {
                    "risk": float(np.mean([v["risk"] for v in values])),
                    "hard_risk": float(np.mean([v["class_risk"][1] for v in values])),
                    "mode_count": sum(sum(v["mode_counts"]) for v in values),
                    "neighborhood_count": sum(sum(v["neighborhood_counts"]) for v in values)}
            pairs = {}
            for control in POLICIES:
                if control == "ready":
                    continue
                diffs = [r["policies"][str(k)]["ready"]["risk"] -
                         r["policies"][str(k)][control]["risk"] for r in rows]
                hard = [r["policies"][str(k)]["ready"]["class_risk"][1] -
                        r["policies"][str(k)][control]["class_risk"][1] for r in rows]
                eps = spec["comparison_threshold"]
                pairs[control] = {"differences": diffs, "hard_differences": hard,
                                  "better": sum(d < -eps for d in diffs),
                                  "worse": sum(d > eps for d in diffs),
                                  "hard_better": sum(d < -eps for d in hard),
                                  "hard_worse": sum(d > eps for d in hard)}
            by_budget[k] = {"metrics": by_policy, "ready_minus": pairs}
        result[ni] = {"nu": nu, "event_count": sum(sum(r["event_counts"]) for r in rows),
                      "draws": sum(r["draws"] for r in rows),
                      "classifier_correct": sum(sum(r["classifier_counts"]) for r in rows),
                      "budgets": by_budget}
    return result


def evaluate(out):
    spec = load(out / "plan.json")
    choices = load(out / "choices.json")["records"]
    if (out / "results.json").exists():
        raise FileExistsError("Evaluation results already exist")
    lookup = {(r["noise_index"], r["model_index"]): r for r in choices}
    records, arrays = [], {}
    for ni, nu in enumerate(spec["measurement_noise_sd"]):
        for mi, case in enumerate(spec["models"]):
            model = model_for(case)
            truth, b = observations(model, spec, "evaluation", ni, mi)
            event = np.all(np.abs(b - model.mu[:, None, :] @ model.B) <= [E1, E2], axis=-1)
            predicted = np.where(b[..., 1] > 1, 2, np.where(b[..., 0] > ALPHA/2, 1, 0))
            classifier = predicted == np.arange(3)[:, None]
            assert classifier[event].all()
            policies = {}
            arrays[f"truth_{ni}_{mi}"], arrays[f"b_{ni}_{mi}"] = truth, b
            for k in spec["budgets"]:
                configs = {"terminal": {"v": model.vend, "calls": 0},
                           "ready": {"v": case["ready"]["v"], "calls": case["ready"]["prefix"]},
                           "same_center_direct": {"v": case["same_center_direct"]["v"],
                                                  "calls": case["same_center_direct"]["prefix"]},
                           **lookup[ni, mi]["choices"][str(k)]}
                values = {}
                for name, cfg in configs.items():
                    gated = name in ("ready", "same_center_direct", "noise_gated")
                    assert cfg["calls"] < k
                    x, first, inside = simulate(model, b, case, cfg, gated, k)
                    values[name] = {**metrics(model, truth, x), "switch_counts": inside.sum(axis=1).tolist()}
                    arrays[f"x_{ni}_{mi}_{k}_{name}"] = x
                    arrays[f"first_{ni}_{mi}_{k}_{name}"] = first
                    if name in ("ready", "same_center_direct"):
                        cap = case[name]["entry_bound"]
                        assert np.all((first[1, event[1]] >= 0) & (first[1, event[1]] <= cap))
                    # Decompose paired error by event without dropping its complement.
                    squared = np.sum((x - truth)**2, axis=-1)
                    values[name]["event_risk_contribution"] = float(model.pi @ (squared * event).mean(axis=1))
                    values[name]["outside_risk_contribution"] = float(model.pi @ (squared * ~event).mean(axis=1))
                policies[str(k)] = values
            records.append({"noise_index": ni, "model_index": mi, "nu": nu,
                            "d": case["d"], "h": case["h"], "draws": event.size,
                            "event_counts": event.sum(axis=1).tolist(),
                            "classifier_counts": classifier.sum(axis=1).tolist(), "policies": policies})
    np.savez_compressed(out / "evaluation.npz", **arrays)
    write(out / "results.json", {"summary": summarize(records, spec), "records": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "calibrate", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"plan": plan, "calibrate": calibrate, "evaluate": evaluate}[args.phase](args.out)


if __name__ == "__main__":
    main()
