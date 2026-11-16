"""Diagnose saved histories and recompute the main theorem's scalar margins."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from common import read_json, write_csv, write_json
from schedules import FAMILIES, schedule


def scalar_margins():
    """Re-evaluate analytic worst-case bounds; not an interval certificate."""
    gamma, c_min, c_max, delta = .25, .999, 1.002, .0001
    alpha, e1, e2 = .001, .00025, .005
    q_barrier = .3 - (alpha * .7 + e1 + delta / gamma) / c_min
    odds = 1.31 * math.exp(-.2 / .17)
    low = odds / (1 + odds)
    high_odds = 1.60 * math.exp(-(1.02**2 + 2.02**2 + .6*1.02 + .02*2.02) / 5)
    high = (1.31 - .98*high_odds) / (2.31 + high_odds)
    dc = .98**2 + 1.98**2 - 2*1.02*.01 - 2*2.02*.05
    rc, hc = 1.27*math.exp(-.51/.69), 1.96*math.exp(-dc/1.42)
    wc = hc / (1 + rc + hc)
    derivative = (.25 + 1.02*3.02*wc) / .69
    de = .98**2 + 1.98**2 + 2*.98*.53 - 2*2.02*.05
    re, he = 1.27*math.exp(.03/.71), 1.96*math.exp(-de/1.42)
    surplus = (re - 1.02*he) / (1 + re + he) - .53
    q = 1 - gamma*c_min*.44
    target = .53 + (.016 - delta/(gamma*c_min))/.44
    lower25 = target - (target + .01)*q**25
    leakage = 4*math.exp(-30)
    values = {
        "scope": "Float reevaluation of analytic worst-case bounds in S4/S5, not a new interval certificate or a proof of learned-network applicability.",
        "barrier_q": q_barrier, "low_mean_upper": low,
        "high_mean_upper": high,
        "low_barrier_slack": q_barrier-low, "high_barrier_slack": q_barrier-high,
        "bottom_displacement_slack": gamma*((1+c_min)*.01-e2)-delta,
        "competitor_responsibility_upper": wc,
        "mean_derivative_upper": derivative, "entrance_surplus_lower": surplus,
        "comparison_at_25": lower25,
        "comparison_entry_slack": lower25-.53,
        "top_drift_upper_divided_by_step": e2-.05+c_max*(2.02*.044-.05)+delta/gamma,
        "crossing_upper": .53+gamma*((alpha+c_max)*.47+e1)+delta,
        "terminal_leakage_upper": leakage,
        "terminal_face_slack": c_min*.09-(e2+c_max*3*leakage+delta/gamma),
        "terminal_linear_factor_upper": 1-gamma*c_min,
        "minimum_class1_mass": 1.27/(1+1.27+1.96),
        "coefficient_condition_implies_sigma_over_tau_at_least": math.sqrt(333),
        "field_error_epsilon_gives_update_error": "gamma*lambda*epsilon; requires a uniform denoiser error bound on every set tested, not average AWGN denoising quality.",
    }
    assert min(values[k] for k in ("low_barrier_slack", "high_barrier_slack",
                                  "bottom_displacement_slack", "comparison_entry_slack",
                                  "terminal_face_slack")) > 0
    assert derivative < .56 and surplus > .016 and values["crossing_upper"] < .649
    return values


def saved_decomposition(source):
    with (source / "evaluation/records.csv").open(encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    references = {(r["image"], r["kernel"], r["noise_255"], r["seed"],
                   r["budget"], r["extra_calls"]): r for r in records if r["family"] == "terminal"}
    rows = []
    for r in records:
        if r["family"] == "terminal" or r["status"] != "ok":
            continue
        key = tuple(r[k] for k in ("image", "kernel", "noise_255", "seed", "budget", "extra_calls"))
        ref = references[key]
        gap = float(r["mse"])-float(ref["mse"])
        distance2 = float(r["output_rms_vs_terminal"])**2
        cross = gap-distance2
        weak_gap = float(r["weak_mse"])-float(ref["weak_mse"])
        cosine = cross/(2*math.sqrt(float(ref["mse"])*distance2)) if distance2 else 0
        rows.append({"image": r["image"], "family": r["family"],
                     "total_calls": int(r["total_calls"]), "mse_gap": gap,
                     "squared_displacement_per_pixel": distance2,
                     "twice_error_displacement_inner_product_per_pixel": cross,
                     "cosine_with_terminal_error": cosine,
                     "weak_mse_gap": weak_gap, "other_mse_gap": gap-weak_gap,
                     "terminal_weak_error_fraction": float(ref["weak_mse"])/float(ref["mse"])})
    aggregates = []
    for k in sorted({r["total_calls"] for r in rows}):
        for family in FAMILIES[1:]:
            group = [r for r in rows if r["total_calls"] == k and r["family"] == family]
            mean = lambda name: float(np.mean([r[name] for r in group]))
            gap = mean("mse_gap")
            aggregates.append({"total_calls": k, "family": family, "images": len(group),
                               "mean_mse_gap": gap,
                               "mean_squared_displacement": mean("squared_displacement_per_pixel"),
                               "mean_cross_term": mean("twice_error_displacement_inner_product_per_pixel"),
                               "cross_term_fraction_of_mean_gap": mean("twice_error_displacement_inner_product_per_pixel")/gap if gap else None,
                               "mean_cosine_with_terminal_error": mean("cosine_with_terminal_error"),
                               "mean_weak_mse_gap": mean("weak_mse_gap"),
                               "weak_fraction_of_mean_gap": mean("weak_mse_gap")/gap if gap else None,
                               "mean_other_mse_gap": mean("other_mse_gap"),
                               "positive_error_inner_products": sum(r["twice_error_displacement_inner_product_per_pixel"] > 0 for r in group),
                               "terminal_weak_error_fraction": mean("terminal_weak_error_fraction")})
    return rows, aggregates


def candidate_coverage(source):
    spec = read_json(source / "calibration/plan.json")
    settings = spec["settings"]
    terminal = settings["solver"]["terminal_sigma_255"] / 255
    rows = []
    for budget, groups in spec["candidates"].items():
        k = int(budget)
        for family in FAMILIES[1:]:
            items = groups[family][1:]
            actual_nonterminal = [int(np.count_nonzero(schedule(c, k, terminal) != terminal)) for c in items]
            rows.append({"budget": k, "family": family, "nonzero_candidates": len(items),
                         "sigma_255": sorted({c["sigma_255"] for c in items}),
                         "prefix_lengths": sorted({c["prefix"] for c in items}),
                         "actual_nonterminal_calls": sorted(set(actual_nonterminal))})
    return rows


def preflight_noise_contrast(path):
    spec = read_json(path)
    rows = [r for r in spec["rows"] if r["network"] == "drunet" and r["calls"] == 64]
    references = {(r["kernel"], r["noise_255"]): r for r in rows if r["family"] == "terminal"}
    return [{"kernel": r["kernel"], "noise_255": r["noise_255"], "family": r["family"],
             "mse_gap": r["mse"]-references[(r["kernel"], r["noise_255"])]["mse"],
             "relative_mse_gap": r["mse"]/references[(r["kernel"], r["noise_255"])]["mse"]-1}
            for r in rows if r["family"] != "terminal"]


def gaussian_reference():
    """A matched one-Gaussian MMSE counterexample to universal annealing gain."""
    gamma, lam, tau, low, high, prefix = .25, .1, .1, .02, .1, 16
    c_low, c_high = (lam*s*s/(tau*tau+s*s) for s in (low, high))
    rows = []
    for beta in (.001, .05, .5):
        q_low = 1-gamma*(beta+c_low)
        terminal = history = beta
        difference_at_switch = None
        for k in range(1, 129):
            terminal = q_low*terminal+gamma*beta
            c = c_high if k <= prefix else c_low
            history = (1-gamma*(beta+c))*history+gamma*beta
            assert -1e-15 <= history <= terminal+1e-15 <= 1+1e-15
            if k == prefix:
                difference_at_switch = terminal-history
            if k in (16, 64, 128):
                expected = difference_at_switch*q_low**(k-prefix)
                assert abs((terminal-history)-expected) < 1e-14
                rows.append({"beta": beta, "calls": k,
                             "terminal_risk": tau*tau*(1-terminal)**2,
                             "history_risk": tau*tau*(1-history)**2,
                             "output_difference_per_unit_truth": terminal-history,
                             "terminal_distance_factor_per_step": q_low})
    return {"scope": "Exact scalar Gaussian prior, noiseless observation, backprojection, identical RED. This verifies an analytic alternative mechanism; it is not a fit to DRUNet.",
            "gamma": gamma, "lambda": lam, "tau": tau,
            "sigma_terminal": low, "sigma_prefix": high, "prefix": prefix, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, aggregates = saved_decomposition(args.source)
    write_csv(args.out / "error_decomposition.csv", rows)
    write_csv(args.out / "error_summary.csv", aggregates)
    write_json(args.out / "theorem_margins.json", scalar_margins())
    write_json(args.out / "candidate_coverage.json", candidate_coverage(args.source))
    write_json(args.out / "gaussian_reference.json", gaussian_reference())
    if args.preflight:
        write_csv(args.out / "preflight_noise_contrast.csv", preflight_noise_contrast(args.preflight))
    print(json.dumps({"observations": len(rows), "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
