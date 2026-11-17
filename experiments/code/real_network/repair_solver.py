"""Bounded development experiment: calibrate RED, then test early/late scale pulses."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from common import (configure_device, read_json, red_trajectory, write_csv, write_json)
from denoisers import load_denoiser
from problem import CircularBlur, load_image, observation


BUDGET = 64
TOLERANCE = 1e-8


def constant_candidates(base):
    solvers = [dict(base)]
    for step, lam, terminal in product((.125, .25, .75), (.01, .1, 1.), (2.5, 5., 10.)):
        solver = {**base, "step": step, "regularization": lam,
                  "terminal_sigma_255": terminal}
        if solver not in solvers:
            solvers.append(solver)
    return [{"id": f"constant_{i:02d}", "solver": s, "history": {"shape": "terminal"}}
            for i, s in enumerate(solvers)]


def history_specs(terminal):
    levels = sorted({terminal + 2.5, max(10., 2 * terminal), 25.})
    specs = []
    for level, length, placement in product(levels, (4, 16), ("early", "delayed")):
        specs.append({"shape": "pulse", "sigma_255": level, "length": length,
                      "placement": placement, "start": 0 if placement == "early" else 32})
    for level, placement in product((levels[0], levels[-1]), ("early", "delayed")):
        specs.append({"shape": "geometric", "sigma_255": level, "length": 16,
                      "placement": placement, "start": 0 if placement == "early" else 32})
    return specs


def scale_sequence(candidate):
    terminal = candidate["solver"]["terminal_sigma_255"]
    sigmas = np.full(BUDGET, terminal / 255, dtype=np.float64)
    h = candidate["history"]
    if h["shape"] == "terminal":
        return sigmas
    start, length = h["start"], h["length"]
    if not 0 <= start < start + length <= BUDGET - 16:
        raise ValueError("Keep at least 16 common terminal updates")
    high = h["sigma_255"] / 255
    sigmas[start:start + length] = (high if h["shape"] == "pulse" else
                                  np.geomspace(high, terminal / 255, length))
    return sigmas


def prepare(source, out, fit_count, check_count, check_seeds):
    source_plan = read_json(source / "plan.json")
    settings = source_plan["settings"]
    files = [{**f, "source_index": i} for i, f in enumerate(source_plan["development_images"])]
    if len(settings["kernels"]) != 1 or len(settings["calibration_seeds"]) != 1:
        raise ValueError("Use the existing single-kernel, single-seed development plan")
    if min(fit_count, check_count) < 1 or fit_count + check_count > len(files):
        raise ValueError("Use disjoint nonempty subsets of the source development images")
    used_seeds = set(settings["calibration_seeds"] + settings["evaluation_seeds"])
    if (not check_seeds or len(set(check_seeds)) != len(check_seeds)
            or any(s < 0 or s in used_seeds for s in check_seeds)):
        raise ValueError("Confirmation needs distinct new nonnegative noise seeds")
    constants = constant_candidates(settings["solver"])
    histories = {str(c["solver"]["terminal_sigma_255"]): history_specs(c["solver"]["terminal_sigma_255"])
                 for c in constants}
    if any(s["sigma_255"] > 50 for specs in histories.values() for s in specs):
        raise ValueError("Use noise levels within the existing 0--50 network interface")
    spec = {
        "source_calibration": str(source.resolve()), "settings": settings,
        "fit_images": files[:fit_count], "check_images": files[fit_count:fit_count + check_count],
        "fit_seeds": settings["calibration_seeds"], "check_seeds": check_seeds,
        "noise_sigma_255": [2.55, 7.65], "budget": BUDGET, "readouts": [16, 32, 64],
        "constant_candidates": constants, "history_candidates_by_terminal": histories,
        "selection": "For each noise amplitude, select one global constant RED solver by "
                     "equal-image fit MSE at 64 calls. Hold it fixed for all scale histories. "
                     "Select best nonzero early and delayed histories on fit data only; also "
                     "freeze the exact early counterpart of the selected delayed history. "
                     "Ties within 1e-8 use the saved order. A performance selector retains "
                     "terminal when no history beats it by more than 1e-8. Freeze before confirmation.",
        "scope": "Directed repair on existing development images; new noise confirmation "
                 "uses the next source images. Same standard-normal field across amplitudes "
                 "and methods. All methods have 64 network calls, no label-dependent control. "
                 "Data loss and RED residual remain unnormalized. Initializer unchanged. "
                 "Solver repair and within-solver schedule gains are reported separately.",
    }
    fit_calls = 2 * fit_count * len(spec["fit_seeds"]) * (len(constants) + len(next(iter(histories.values())))) * BUDGET
    check_calls = 2 * check_count * len(check_seeds) * 5 * BUDGET
    spec["cost"] = {"fit_forwards": fit_calls, "check_forwards": check_calls,
                    "total_forwards_if_finite": fit_calls + check_calls,
                    "inference_minutes_at_28_ms": (fit_calls + check_calls) * .028 / 60}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", spec)
    return spec


def measure(spec, candidates, noise, images, seeds, denoiser, device):
    rows = []
    settings, kernel = spec["settings"], spec["settings"]["kernels"][0]
    for image in images:
        truth = load_image(image["path"], device, settings["gray_policy"])
        blur = CircularBlur(np.asarray(kernel["values"]), truth)
        for seed in seeds:
            # Noise index zero deliberately couples both amplitudes to the same field.
            y = observation(truth, blur, noise, seed, image["source_index"], 0, 0)
            for candidate in candidates:
                key = {"noise_255": noise, "image": image["id"], "seed": seed,
                       "candidate": candidate["id"]}
                try:
                    for k, x, info in red_trajectory(denoiser, blur, y, scale_sequence(candidate),
                                                    candidate["solver"], set(spec["readouts"])):
                        mse = float((x.double() - truth.double()).square().mean().item())
                        rows.append({**key, "total_calls": k, "mse": mse,
                                     "psnr": 10 * np.log10(1 / max(mse, 1e-30)),
                                     "field_rms_before_last_update": info["field_rms_before_last_update"],
                                     "failure": ""})
                except FloatingPointError as error:
                    rows.append({**key, "total_calls": BUDGET, "mse": None, "psnr": None,
                                 "field_rms_before_last_update": None, "failure": str(error)})
        print(f"Measured {image['id']} noise={noise:g}; forwards={denoiser.calls}", flush=True)
    return rows


def scores(rows, candidates, expected):
    result = []
    for candidate in candidates:
        final = [r for r in rows if r["candidate"] == candidate["id"] and r["total_calls"] == BUDGET]
        mean = (float(np.mean([r["mse"] for r in final]))
                if len(final) == expected and not any(r["failure"] for r in final) else None)
        result.append({"candidate": candidate, "mse": mean})
    return result


def choose(scored):
    eligible = [s for s in scored if s["mse"] is not None]
    if not eligible:
        raise RuntimeError("All candidates failed; saved records retain failures")
    best = min(s["mse"] for s in eligible)
    return next(s for s in eligible if s["mse"] <= best + TOLERANCE)


def check_summary(rows):
    result = []
    for noise, k in product(sorted({r["noise_255"] for r in rows}), (16, 32, 64)):
        selected = [r for r in rows if r["noise_255"] == noise and r["total_calls"] == k]
        if any(r["failure"] for r in rows if r["noise_255"] == noise):
            result.append({"noise_255": noise, "total_calls": k, "candidate": "failed_group",
                           "mse": None, "mean_psnr": None, "mse_change_vs_original_percent": None,
                           "mse_change_vs_tuned_percent": None, "images_better_than_tuned": None,
                           "images": len({r["image"] for r in selected})})
            continue
        means = {name: float(np.mean([r["mse"] for r in selected if r["candidate"] == name]))
                 for name in dict.fromkeys(r["candidate"] for r in selected)}
        for name, mse in means.items():
            arm = [r for r in selected if r["candidate"] == name]
            wins = 0
            for image in {r["image"] for r in arm}:
                a = np.mean([r["mse"] for r in arm if r["image"] == image])
                b = np.mean([r["mse"] for r in selected if r["image"] == image
                             and r["candidate"] == "tuned_terminal"])
                wins += bool(a < b - TOLERANCE)
            result.append({"noise_255": noise, "total_calls": k, "candidate": name,
                           "mse": mse, "mean_psnr": float(np.mean([r["psnr"] for r in arm])),
                           "mse_change_vs_original_percent": 100 * (mse / means["original_terminal"] - 1),
                           "mse_change_vs_tuned_percent": 100 * (mse / means["tuned_terminal"] - 1),
                           "images_better_than_tuned": wins, "images": len({r["image"] for r in arm})})
    return result


def run(spec, out, device_name, denoiser=None):
    device = configure_device(device_name)
    denoiser = denoiser or load_denoiser(spec["settings"]["network"], device)
    start_calls = denoiser.calls
    fit_size = len(spec["fit_images"]) * len(spec["fit_seeds"])
    constant_rows, solvers = [], {}
    for noise in spec["noise_sigma_255"]:
        rows = measure(spec, spec["constant_candidates"], noise, spec["fit_images"],
                       spec["fit_seeds"], denoiser, device)
        constant_rows.extend(rows)
        try:
            solvers[str(noise)] = choose(scores(rows, spec["constant_candidates"], fit_size))
        except RuntimeError:
            write_csv(out / "constant_fit.csv", constant_rows)
            raise
    write_csv(out / "constant_fit.csv", constant_rows)
    write_json(out / "solver_choices.json", solvers)
    history_rows, choices = [], {}
    for noise in spec["noise_sigma_255"]:
        fixed = solvers[str(noise)]
        solver = fixed["candidate"]["solver"]
        histories = [{"id": f"history_{i:02d}", "solver": solver, "history": h}
                     for i, h in enumerate(spec["history_candidates_by_terminal"][str(solver["terminal_sigma_255"])])]
        rows = measure(spec, histories, noise, spec["fit_images"], spec["fit_seeds"], denoiser, device)
        history_rows.extend(rows)
        scored = scores(rows, histories, fit_size)
        try:
            early = choose([s for s in scored if s["candidate"]["history"]["placement"] == "early"])
            delayed = choose([s for s in scored if s["candidate"]["history"]["placement"] == "delayed"])
        except RuntimeError:
            write_csv(out / "history_fit.csv", history_rows)
            raise
        paired_spec = {**delayed["candidate"]["history"], "placement": "early", "start": 0}
        counterpart = next(c for c in histories if c["history"] == paired_spec)
        best = choose([fixed, *scored])
        arms = {"original_terminal": spec["constant_candidates"][0],
                "tuned_terminal": fixed["candidate"], "best_early": early["candidate"],
                "best_delayed": delayed["candidate"], "paired_early": counterpart}
        choices[str(noise)] = {
            "solver": solver, "terminal_fit_mse": fixed["mse"],
            "early_fit_mse": early["mse"], "delayed_fit_mse": delayed["mse"],
            "performance_selection": best, "arms": [{**c, "id": name} for name, c in arms.items()],
            "paired_comparison": "best_delayed versus paired_early has identical scale multiset, "
                                 "solver and call budget; only placement differs. "
                                 "Nonzero diagnostic arms remain even if performance selection is terminal.",
        }
    write_csv(out / "history_fit.csv", history_rows)
    write_json(out / "history_choices.json", choices)
    fit_calls = denoiser.calls - start_calls
    # No confirmation labels have been loaded before the solver/history files are frozen.
    check_rows = []
    for noise in spec["noise_sigma_255"]:
        check_rows.extend(measure(spec, choices[str(noise)]["arms"], noise, spec["check_images"],
                                  spec["check_seeds"], denoiser, device))
    write_csv(out / "check_records.csv", check_rows)
    write_csv(out / "check_summary.csv", check_summary(check_rows))
    calls = denoiser.calls - start_calls
    failures = sum(bool(r["failure"]) for r in constant_rows + history_rows + check_rows)
    if not failures and calls != spec["cost"]["total_forwards_if_finite"]:
        raise AssertionError("Actual forward count differs from the frozen plan")
    write_json(out / "completion.json", {"network_forwards": calls, "fit_forwards": fit_calls,
                                        "check_forwards": calls - fit_calls, "failed_trajectories": failures,
                                        "confirmation_choices_frozen_before_loading_check_images": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-count", type=int, default=2)
    parser.add_argument("--check-count", type=int, default=2)
    parser.add_argument("--check-seeds", type=int, nargs="+", default=[2026111711, 2026111712])
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    spec = prepare(args.source_calibration, args.out, args.fit_count, args.check_count, args.check_seeds)
    print(spec["cost"], flush=True)
    if not args.plan_only:
        run(spec, args.out, args.device)


if __name__ == "__main__":
    main()
