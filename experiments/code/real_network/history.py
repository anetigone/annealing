"""Freeze best nonzero prefixes from saved development scores; no new tuning."""
import copy
import csv
import math
from pathlib import Path

import numpy as np

from common import condition_key, image_list, read_json, write_csv, write_json
from evaluate import check_split, evaluate
from schedules import FAMILIES, schedule, tie_key
from summarize import summarize


def select_choices(spec, original, score_rows, budgets):
    settings = spec["settings"]
    terminal = settings["solver"]["terminal_sigma_255"] / 255
    tolerance = settings["search"]["mse_tolerance"]
    indexed = {}
    for row in score_rows:
        key = (row["kernel"], float(row["noise_255"]), int(row["budget"]),
               row["family"], int(row["candidate_index"]))
        if key in indexed:
            raise ValueError(f"Duplicate source development score: {key}")
        indexed[key] = row
    conditions, selected_rows = {}, []
    for kernel in settings["kernels"]:
        for noise in settings["noise_sigma_255"]:
            for budget in budgets:
                name = condition_key(kernel, noise, budget)
                source = original["conditions"][name]
                conditions[name] = {}
                for family in FAMILIES:
                    candidates = spec["candidates"][str(budget)][family]
                    eligible = []
                    for index, candidate in enumerate(candidates):
                        sigmas = schedule(candidate, budget, terminal)
                        nonzero = candidate.get("prefix", 0) > 0 and np.any(sigmas != terminal)
                        if (family == "terminal" and nonzero) or (family != "terminal" and not nonzero):
                            continue
                        row = indexed[(kernel["name"], float(noise), budget, family, index)]
                        if row["mse"] and int(row["failures"]) == 0 and math.isfinite(float(row["mse"])):
                            eligible.append((float(row["mse"]), index, candidate))
                    if not eligible:
                        raise ValueError(f"No finite {'terminal' if family == 'terminal' else 'nonzero-prefix'} development candidate: {name}/{family}")
                    best = min(item[0] for item in eligible)
                    tied = [item for item in eligible if item[0] <= best + tolerance]
                    mse, index, candidate = min(tied, key=lambda item: tie_key(item[2]))
                    conditions[name][family] = {
                        "candidate": copy.deepcopy(candidate), "candidate_index": index,
                        "candidate_count": len(eligible), "development_mse": mse,
                        "selection_constraint": "terminal" if family == "terminal" else "nonzero_prefix",
                        "source_unrestricted_choice": copy.deepcopy(source[family]),
                        "development_mse_minus_unrestricted": mse-source[family]["development_mse"],
                    }
                    selected_rows.append({
                        "kernel": kernel["name"], "noise_255": noise, "budget": budget,
                        "family": family, "candidate_index": index, "kind": candidate["kind"],
                        "prefix": candidate.get("prefix", 0), "sigma_255": candidate.get("sigma_255", ""),
                        "pattern": candidate.get("pattern", ""), "block": candidate.get("block", ""),
                        "development_mse": mse,
                        "source_unrestricted_mse": source[family]["development_mse"],
                        "development_mse_minus_unrestricted": mse-source[family]["development_mse"],
                    })
    return conditions, selected_rows


def prepare_history(source_calibration, out, budgets=None, anchor=64, extra_calls=(32, 64),
                    evaluation_count=68, milliseconds_per_call=28):
    source, out = Path(source_calibration), Path(out)
    spec = read_json(source / "plan.json")
    original = read_json(source / "choices.json")
    with (source / "calibration/scores.csv").open(encoding="utf-8", newline="") as stream:
        scores = list(csv.DictReader(stream))
    budgets = list(spec["settings"]["budgets"] if budgets is None else budgets)
    if not budgets or len(set(budgets)) != len(budgets) or any(k not in spec["settings"]["budgets"] for k in budgets):
        raise ValueError("Use distinct budgets that were scored in the source calibration")
    if anchor not in budgets or not extra_calls or any(not isinstance(k, int) or k <= 0 for k in extra_calls):
        raise ValueError("Tail anchor must be a selected budget; extra calls must be positive integers")
    if evaluation_count <= 0 or not math.isfinite(milliseconds_per_call) or milliseconds_per_call <= 0:
        raise ValueError("Positive evaluation count and per-call estimate required")
    conditions, selected_rows = select_choices(spec, original, scores, budgets)
    settings = copy.deepcopy(spec["settings"])
    settings["budgets"] = budgets
    settings["tail"] = {"anchor": anchor, "extra_calls": sorted(set(extra_calls))}
    terminal = settings["solver"]["terminal_sigma_255"] / 255
    forwards_per_image, logical_per_image, histories = 0, 0, []
    for kernel in settings["kernels"]:
        for noise in settings["noise_sigma_255"]:
            for budget in budgets:
                choices = conditions[condition_key(kernel, noise, budget)]
                unique = {tuple(schedule(choices[family]["candidate"], budget, terminal)) for family in FAMILIES}
                length = budget + (max(extra_calls) if budget == anchor else 0)
                forwards_per_image += len(unique) * length
                logical_per_image += len(FAMILIES) * length
                histories.append({"kernel": kernel["name"], "noise_255": noise,
                                  "budget": budget, "distinct_histories": len(unique)})
    factor = evaluation_count * len(settings["evaluation_seeds"])
    costs = {
        "additional_calibration_forwards": 0, "evaluation_images": evaluation_count,
        "logical_evaluation_forwards": logical_per_image * factor,
        "actual_evaluation_forwards_if_finite": forwards_per_image * factor,
        "assumed_milliseconds_per_call": milliseconds_per_call,
        "inference_minutes": forwards_per_image * factor * milliseconds_per_call / 60000,
        "histories": histories,
        "scope": "Exact frozen prefixes; extra calls are terminal-only extensions of the "
                 "anchor choice, not newly tuned optima at the longer budgets. No new calibration.",
    }
    frozen = copy.deepcopy(spec)
    frozen["settings"] = settings
    frozen["candidates"] = {str(k): copy.deepcopy(spec["candidates"][str(k)]) for k in budgets}
    frozen["source_calibration"] = str(source.resolve())
    frozen["selection"] = (
        "For terminal retain the baseline. For each other family, select "
        "the best finite nonzero-prefix candidate from the saved development "
        "scores, using the original tolerance and tie rule. No evaluation scores read.")
    frozen["scope"] = (
        "Follow-up history diagnostic after the source benchmark was inspected. "
        "This restriction is not an error-optimality claim or a new blind test; "
        "the unrestricted source result remains the performance comparison.")
    out.mkdir(parents=True, exist_ok=False)
    derived = out / "calibration"
    derived.mkdir()
    write_json(out / "settings.json", settings)
    write_json(out / "cost_estimate.json", costs)
    write_csv(out / "selected_prefixes.csv", selected_rows)
    write_json(derived / "plan.json", frozen)
    write_json(derived / "choices.json", {
        "conditions": conditions, "network_layout": original["network_layout"],
        "source_calibration": str(source.resolve()), "additional_development_network_forwards": 0,
        "selection_constraint": "Best nonzero prefix per family; terminal reference retained",
    })
    print(f"Nonzero-prefix choices frozen: {out / 'selected_prefixes.csv'}", flush=True)
    print(f"New calibration: 0 calls; evaluation: {costs['actual_evaluation_forwards_if_finite']:,} "
          f"forwards; inference estimate: {costs['inference_minutes']:.1f} minutes before I/O/metrics", flush=True)
    for row in selected_rows:
        print(f"K={row['budget']} {row['family']}: prefix={row['prefix']}, sigma_255={row['sigma_255']}, "
              f"dev MSE={row['development_mse']:.9g}", flush=True)
    return costs


def history_workflow(source_calibration, images, out, dataset="bsd68", device="cuda",
                     save_states=False, denoiser=None, **options):
    files = image_list(images)
    source_spec = read_json(Path(source_calibration) / "plan.json")
    check_split(source_spec["development_images"], files, dataset, source_spec["settings"]["gray_policy"])
    costs = prepare_history(source_calibration, out, evaluation_count=len(files), **options)
    out = Path(out)
    evaluate(out / "calibration", images, out / "evaluation", dataset, device, save_states,
             denoiser=denoiser)
    summarize(out / "evaluation")
    write_json(out / "workflow_complete.json", {
        "summary": str((out / "evaluation/summary.json").resolve()),
        "scope": "Frozen nonzero-prefix diagnostic, no new calibration or per-image selection",
        "estimated_evaluation_forwards": costs["actual_evaluation_forwards_if_finite"],
    })
    print(f"History diagnostic complete: {out / 'evaluation/summary.json'}", flush=True)
