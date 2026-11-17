"""Infer same-state RED strength interventions from saved force scalars; no network."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def strength_bounds(row, step, old_lam, new_lam):
    """Bound MSE(new_lam)-MSE(old_lam), keeping x, sigma, y and step fixed."""
    data = float(row["data_field_rms"])
    reg = float(row["regularization_field_rms"])
    error_reg = float(row["regularization_error_inner_product"])
    ratio = new_lam / old_lam
    center = (-2 * step * (ratio - 1) * error_reg
              + step**2 * (ratio**2 - 1) * reg**2)
    radius = 2 * step**2 * abs(ratio - 1) * data * reg
    return center - radius, center + radius


def optimum_bounds(row, step, old_lam):
    """Interval for the unconstrained one-step optimum, before clipping at zero."""
    data = float(row["data_field_rms"])
    reg = float(row["regularization_field_rms"])
    error_reg = float(row["regularization_error_inner_product"])
    if reg <= 0:
        raise ValueError("A positive residual norm is needed for the strength diagnostic")
    center = old_lam * error_reg / (step * reg**2)
    radius = old_lam * data / reg
    return center - radius, center + radius


def save_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def transport_thresholds(source, out):
    """Reweight the saved denoiser secant at the same pair, not on a new trajectory."""
    with (source / "terminal_transport.csv").open(encoding="utf-8-sig") as stream:
        transport = list(csv.DictReader(stream))
    rows, groups = [], defaultdict(list)
    for row in transport:
        beta = float(row["effective_data_gain"])
        gain = float(row["denoiser_secant_projection_gain"])
        critical = beta / (gain - 1) if gain > 1 else None
        result = {k: row[k] for k in ("image", "noise_255", "family", "from_call")}
        result.update({"effective_data_gain": beta, "denoiser_projection_gain": gain,
                       "critical_lambda_if_expansive": critical,
                       "directional_curvature_at_lambda_1": beta + 1 - gain})
        rows.append(result)
        if int(row["from_call"]) >= 64:
            groups[(float(row["noise_255"]), row["family"])].append(result)
    summary = []
    for (noise, family), group in sorted(groups.items()):
        critical = [r["critical_lambda_if_expansive"] for r in group
                    if r["critical_lambda_if_expansive"] is not None]
        summary.append({"noise_255": noise, "family": family, "paired_steps": len(group),
                        "denoiser_expansive_steps": len(critical),
                        "minimum_critical_lambda": min(critical) if critical else None,
                        "negative_curvature_steps_at_lambda_1": sum(r["directional_curvature_at_lambda_1"] < 0 for r in group)})
    save_csv(out / "secant_thresholds.csv", rows)
    (out / "transport_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def analyze(source, out):
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8-sig"))
    solver = plan["settings"]["solver"]
    step, lam = solver["step"], solver["regularization"]
    if lam <= 0:
        raise ValueError("Source force measurement needs positive regularization")
    with (source / "forces.csv").open(encoding="utf-8-sig") as stream:
        forces = list(csv.DictReader(stream))
    with (source / "endpoints.csv").open(encoding="utf-8-sig") as stream:
        endpoints = list(csv.DictReader(stream))
    optima, interventions, endpoint_groups = [], [], defaultdict(list)
    for row in forces:
        key = {k: row[k] for k in ("image", "noise_255", "state_call", "sigma_255")}
        low, high = optimum_bounds(row, step, lam)
        optima.append({**key, "unconstrained_lambda_lower": low,
                       "unconstrained_lambda_upper": high,
                       "nonnegative_optimum_is_zero": high < 0})
        for new_lam in (0., .01, .03, .1, .3, 1.):
            low, high = strength_bounds(row, step, lam, new_lam)
            interventions.append({**key, "new_lambda": new_lam,
                                  "one_step_mse_change_lower": low,
                                  "one_step_mse_change_upper": high})
    for row in endpoints:
        if row["family"] == "terminal":
            endpoint_groups[(float(row["noise_255"]), int(row["total_calls"]))].append(row)
    endpoint_summary = []
    for (noise, k), rows in sorted(endpoint_groups.items()):
        mse = float(np.mean([float(r["mse"]) for r in rows]))
        data = float(np.mean([float(r["data_only_mse"]) for r in rows]))
        endpoint_summary.append({"noise_255": noise, "total_calls": k,
                                 "terminal_mse": mse, "data_only_mse": data,
                                 "terminal_excess_over_data_percent": 100 * (mse / data - 1)})
    summaries = []
    for noise, new_lam in ((2.55, .01), (7.65, 1.)):
        for k in (0, 16, 64, 128):
            rows = [r for r in interventions if float(r["noise_255"]) == noise
                    and int(r["state_call"]) == k
                    and float(r["sigma_255"]) == solver["terminal_sigma_255"]
                    and r["new_lambda"] == new_lam]
            if rows:
                summaries.append({"noise_255": noise, "state_call": k,
                                  "sigma_255": solver["terminal_sigma_255"],
                                  "old_lambda": lam, "new_lambda": new_lam,
                                  "images": len(rows),
                                  "mean_mse_change_lower": float(np.mean([r["one_step_mse_change_lower"] for r in rows])),
                                  "mean_mse_change_upper": float(np.mean([r["one_step_mse_change_upper"] for r in rows]))})
    out.mkdir(parents=True, exist_ok=False)
    save_csv(out / "local_optima.csv", optima)
    save_csv(out / "strength_interventions.csv", interventions)
    save_csv(out / "endpoint_comparison.csv", endpoint_summary)
    summary = {"source": str(source.resolve()), "solver": solver,
               "network_forwards": 0, "same_state_interventions": summaries,
               "scope": "One-step interventions at saved terminal-reference states. "
                        "Truth is used only to diagnose directions. Cauchy-Schwarz bounds "
                        "on saved floating-point scalars, not outward-rounded certificates "
                        "or tuned trajectory optima. No prescriptions selected here."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    transport_thresholds(source, out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source, args.out)
    print(json.dumps(result["same_state_interventions"], indent=2))


if __name__ == "__main__":
    main()
