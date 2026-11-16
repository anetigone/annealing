"""Freeze a local development split, then choose each budget's schedules."""
from pathlib import Path

import numpy as np

from common import (condition_key, configure_device, image_list, normalize_settings,
                    read_json, red_trajectory, write_csv, write_json)
from denoisers import load_denoiser
from problem import CircularBlur, load_image, load_kernel, observation
from schedules import FAMILIES, candidate_families, schedule, tie_key


def plan(config, images, out):
    settings = normalize_settings(read_json(config))
    files = image_list(images)
    for kernel in settings["kernels"]:
        kernel["values"] = load_kernel(kernel).tolist()
    # Store the complete candidate sets before any network or truth evaluation.
    candidates = {str(k): candidate_families(settings, k) for k in settings["budgets"]}
    target = Path(out)
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / "plan.json", {
        "settings": settings, "development_images": files, "candidates": candidates,
        "selection": "Equal image/seed mean MSE at each kernel/noise/budget; candidates "
                     "within the predeclared MSE tolerance tie by shorter prefix, lower sigma, "
                     "then pattern/block. Evaluation labels never choose a prescription.",
        "scope": "Empirical fixed RED schedule comparison. No GMM certificates transferred.",
        "computation": "Exactly identical complete scale sequences share development scores; "
                       "candidate sets, selection rules and logical budgets are unchanged.",
    })
    print(f"Plan saved: {target / 'plan.json'} ({len(files)} development images)", flush=True)


def calibrate(out, device_name, denoiser=None):
    out = Path(out)
    spec = read_json(out / "plan.json")
    if (out / "choices.json").exists():
        raise FileExistsError("Choices already frozen; use a new experiment directory")
    score_dir = out / "calibration"
    score_dir.mkdir(exist_ok=False)
    settings = spec["settings"]
    device = configure_device(device_name)
    denoiser = denoiser or load_denoiser(settings["network"], device)
    rows, choices, total_calls, reused_candidates = [], {}, 0, 0
    terminal = settings["solver"]["terminal_sigma_255"] / 255
    for ki, kernel in enumerate(settings["kernels"]):
        for ni, noise in enumerate(settings["noise_sigma_255"]):
            for budget in settings["budgets"]:
                groups = spec["candidates"][str(budget)]
                choices_for_condition = {}
                score_cache = {}
                for family in FAMILIES:
                    scores = []
                    for ci, candidate in enumerate(groups[family]):
                        sigmas = schedule(candidate, budget, terminal)
                        cache_key = tuple(sigmas)
                        reused_from = ""
                        if cache_key in score_cache:
                            mean, failures, reused_from = score_cache[cache_key]
                            calls = 0
                            reused_candidates += 1
                        else:
                            errors, failures = [], 0
                            calls_before = denoiser.calls
                            for ii, image in enumerate(spec["development_images"]):
                                truth = load_image(image["path"], device, settings["gray_policy"])
                                blur = CircularBlur(np.asarray(kernel["values"]), truth)
                                for seed in settings["calibration_seeds"]:
                                    y = observation(truth, blur, noise, seed, ii, ki, ni)
                                    try:
                                        snapshots = list(red_trajectory(denoiser, blur, y, sigmas,
                                                                       settings["solver"], {budget}))
                                        x = snapshots[0][1]
                                        errors.append(float((x.double()-truth.double()).square().mean().item()))
                                    except FloatingPointError:
                                        failures += 1
                            calls = denoiser.calls - calls_before
                            total_calls += calls
                            mean = float(np.mean(errors)) if not failures else None
                            score_cache[cache_key] = (mean, failures, f"{family}:{ci}")
                        row = {"kernel": kernel["name"], "noise_255": noise, "budget": budget,
                               "family": family, "candidate_index": ci, "mse": mean,
                               "failures": failures, "network_forwards": calls,
                               "reused_from": reused_from}
                        rows.append(row)
                        scores.append(mean)
                    eligible = [i for i, value in enumerate(scores) if value is not None]
                    if not eligible:
                        write_csv(score_dir / "scores.csv", rows)
                        raise RuntimeError(f"All {family} candidates failed; scores retained, no choices frozen")
                    best = min(scores[i] for i in eligible)
                    tolerance = settings["search"]["mse_tolerance"]
                    tied = [i for i in eligible if scores[i] <= best + tolerance]
                    selected = min(tied, key=lambda i: tie_key(groups[family][i]))
                    choices_for_condition[family] = {
                        "candidate": groups[family][selected], "development_mse": scores[selected],
                        "candidate_index": selected, "candidate_count": len(groups[family]),
                    }
                    print(f"{kernel['name']} noise={noise:g} K={budget} {family}: "
                          f"dev MSE={scores[selected]:.7g}", flush=True)
                choices[condition_key(kernel, noise, budget)] = choices_for_condition
    write_csv(score_dir / "scores.csv", rows)
    write_json(out / "choices.json", {
        "conditions": choices, "network_layout": denoiser.layout, "device": str(device),
        "development_network_forwards": total_calls, "reused_candidates": reused_candidates,
    })
    print(f"Choices frozen: {out / 'choices.json'}", flush=True)
