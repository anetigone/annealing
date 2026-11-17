"""Repeat frozen BSD68 comparisons with three new seeds and a matched sigma49 pulse."""
import argparse
import copy
import csv
import json
from pathlib import Path
import time

import numpy as np


DEFAULT_SEEDS = (2026111716, 2026111717, 2026111718)
METHODS = ("tuned_terminal", "history_25", "history_49")
COMPARISONS = (("history_25", "tuned_terminal"), ("history_49", "tuned_terminal"),
               ("history_25", "history_49"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(source, out, seeds):
    original = json.loads((source / "plan.json").read_text(encoding="utf-8-sig"))
    images, noises = original["check_images"], original["noise_sigma_255"]
    if (original["dataset"] != "bsd68" or len(images) != 68
            or len({i["id"] for i in images}) != 68 or sorted(noises) != [2.55, 7.65]):
        raise ValueError("Use the saved complete two-noise BSD68 frozen plan")
    old_settings = original["settings"]
    used_seeds = set(original["check_seeds"] + old_settings.get("calibration_seeds", [])
                     + old_settings.get("evaluation_seeds", []))
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(s < 0 or s in used_seeds for s in seeds)):
        raise ValueError("Use distinct new nonnegative noise seeds")
    choices = {}
    for noise in noises:
        old = original["choices"][str(noise)]
        terminal = copy.deepcopy(next(c for c in old if c["id"] == "tuned_terminal"))
        middle = copy.deepcopy(next(c for c in old if c["id"] == "frozen_history"))
        history = middle["history"]
        if (history["shape"] != "pulse" or history["sigma_255"] != 25
                or history["length"] != 16 or history["start"] not in (0, 32)
                or middle["solver"] != terminal["solver"]):
            raise ValueError("Expected the frozen 16-call sigma25 pulse and common solver")
        middle["id"] = "history_25"
        high = copy.deepcopy(middle)
        high["id"], high["history"]["sigma_255"] = "history_49", 49.
        choices[str(noise)] = [terminal, middle, high]
    expected = len(images) * len(noises) * len(seeds) * len(METHODS) * 64
    spec = {
        "source_frozen_plan": str((source / "plan.json").resolve()),
        "settings": {key: copy.deepcopy(old_settings[key]) for key in ("network", "kernels", "gray_policy")},
        "dataset": "bsd68", "check_images": images, "check_seeds": seeds,
        "noise_sigma_255": noises, "readouts": [64], "choices": choices,
        "solver_per_noise": {str(n): choices[str(n)][0]["solver"] for n in noises},
        "comparisons": [list(pair) for pair in COMPARISONS],
        "expected_network_forwards": expected,
        "expected_records": len(images) * len(noises) * len(seeds) * len(METHODS),
        "inference_minutes_at_28_ms": expected * .028 / 60,
        "statistics": "Report each noise seed separately. For pooled results first average "
                      "the seeds within each image, then weight images equally. Bootstrap "
                      "2000 paired images after seed averaging; do not count 204 image/seed "
                      "observations as independent images. Win/tie tolerance is MSE 1e-8.",
        "scope": "Read prescriptions and image order from the saved plan only, not its "
                 "evaluation scores. Sigma49 changes only pulse level; start, duration, solver, "
                 "initialization and terminal noise match sigma25. All methods use 64 calls. "
                 "No refitting, per-image selection or label-dependent stopping. Standard-normal "
                 "fields are shared across methods and noise amplitudes, with source image indices "
                 "preserved. Data loss and RED residual remain unnormalized. Previously viewed "
                 "BSD68 is not declared a new blind or pretraining-unseen test set.",
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", spec)
    return spec


def comparison(left, right, values, scores, draws):
    a, b = values[left], values[right]
    samples = 100 * (1 - a[draws].mean(axis=1) / b[draws].mean(axis=1))
    low, high = np.quantile(samples, [.025, .975])
    return {"left": left, "right": right,
            "mse_gain_percent": float(100 * (1 - a.mean() / b.mean())),
            "mean_mse_difference": float(np.mean(a - b)),
            "mean_psnr_difference_db": float(np.mean(scores[left] - scores[right])),
            "paired_image_bootstrap_low_percent": float(low),
            "paired_image_bootstrap_high_percent": float(high),
            "image_wins": int(np.sum(a < b - 1e-8)),
            "image_ties": int(np.sum(np.abs(a - b) <= 1e-8)),
            "image_losses": int(np.sum(a > b + 1e-8)), "images": len(a)}


def summarize(rows, spec, out):
    images, seeds = [i["id"] for i in spec["check_images"]], spec["check_seeds"]
    lookup = {(r["noise_255"], r["image"], r["seed"], r["candidate"]): r for r in rows}
    if len(lookup) != spec["expected_records"]:
        raise ValueError("Incomplete or duplicate result keys; raw records retained")
    pooled_rows, image_rows, pooled_pairs, seed_rows, seed_pairs = [], [], [], [], []
    for ni, noise in enumerate(spec["noise_sigma_255"]):
        values = {m: np.array([[lookup[(noise, image, s, m)]["mse"] for s in seeds]
                              for image in images], dtype=float) for m in METHODS}
        scores = {m: np.array([[lookup[(noise, image, s, m)]["psnr"] for s in seeds]
                              for image in images], dtype=float) for m in METHODS}
        draws = np.random.default_rng(2026111730 + ni).integers(0, len(images), size=(2000, len(images)))
        for si, seed in enumerate(seeds):
            per_seed = {m: a[:, si] for m, a in values.items()}
            per_scores = {m: a[:, si] for m, a in scores.items()}
            for method in METHODS:
                seed_rows.append({"noise_255": noise, "seed": seed, "candidate": method,
                                  "mse": float(per_seed[method].mean()),
                                  "mean_psnr": float(per_scores[method].mean()), "images": len(images)})
            for left, right in COMPARISONS:
                seed_pairs.append({"noise_255": noise, "seed": seed,
                                   **comparison(left, right, per_seed, per_scores, draws)})
        means = {m: a.mean(axis=1) for m, a in values.items()}
        mean_scores = {m: a.mean(axis=1) for m, a in scores.items()}
        for method in METHODS:
            pooled_rows.append({"noise_255": noise, "candidate": method,
                                "mse": float(means[method].mean()),
                                "mean_psnr": float(mean_scores[method].mean()),
                                "images": len(images), "noise_seeds": len(seeds)})
            image_rows.extend({"noise_255": noise, "image": image, "candidate": method,
                               "mse": float(means[method][i]), "mean_psnr": float(mean_scores[method][i]),
                               "noise_seeds": len(seeds)} for i, image in enumerate(images))
        for left, right in COMPARISONS:
            gains = 100 * (1 - values[left].mean(axis=0) / values[right].mean(axis=0))
            pooled_pairs.append({"noise_255": noise, "noise_seeds": len(seeds),
                                 **comparison(left, right, means, mean_scores, draws),
                                 "minimum_seed_gain_percent": float(gains.min()),
                                 "maximum_seed_gain_percent": float(gains.max())})
    write_csv(out / "summary.csv", pooled_rows)
    write_csv(out / "paired_images.csv", image_rows)
    write_csv(out / "paired_comparisons.csv", pooled_pairs)
    write_csv(out / "seed_summary.csv", seed_rows)
    write_csv(out / "seed_comparisons.csv", seed_pairs)
    for result in pooled_pairs:
        print(result, flush=True)


def run(spec, out, device_name):
    from common import configure_device
    from denoisers import load_denoiser
    from repair_solver import measure

    device = configure_device(device_name)
    denoiser = load_denoiser(spec["settings"]["network"], device)
    start = time.perf_counter()
    rows = []
    with (out / "records.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = None
        for noise in spec["noise_sigma_255"]:
            for image in spec["check_images"]:
                measured = measure(spec, spec["choices"][str(noise)], noise, [image],
                                   spec["check_seeds"], denoiser, device)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(measured[0])); writer.writeheader()
                writer.writerows(measured)
                rows.extend(measured)
                stream.flush()
    failures = sum(bool(r["failure"]) for r in rows)
    if not failures:
        if denoiser.calls != spec["expected_network_forwards"] or len(rows) != spec["expected_records"]:
            raise AssertionError("Call or record count differs from the frozen plan")
        summarize(rows, spec, out)
    else:
        print("Nonfinite trajectories retained in records.csv; complete-group summaries skipped", flush=True)
    write_json(out / "completion.json", {"network_forwards": denoiser.calls,
                                         "expected_network_forwards": spec["expected_network_forwards"],
                                         "records": len(rows), "failed_trajectories": failures,
                                         "images": len(spec["check_images"]), "noise_seeds": len(spec["check_seeds"]),
                                         "seconds_including_metrics": time.perf_counter() - start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/results/drunet_directed_run/bsd68"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    spec = prepare(args.source, args.out, args.seeds)
    print(f"Frozen scale validation: {spec['expected_network_forwards']} forwards; "
          f"about {spec['inference_minutes_at_28_ms']:.2f} inference minutes", flush=True)
    if not args.plan_only:
        run(spec, args.out, args.device)


if __name__ == "__main__":
    main()
