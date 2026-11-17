"""Validate frozen repair prescriptions on the remaining development images."""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from common import configure_device, image_list, read_json, write_csv, write_json
from denoisers import load_denoiser
from evaluate import check_split
from repair_solver import check_summary, measure


def prepare(source, development_plan, out, seeds, images_path=None, dataset="custom"):
    original = read_json(source / "plan.json")
    development = read_json(development_plan)
    frozen = read_json(source / "history_choices.json")
    seen = {i["id"] for i in original["fit_images"] + original["check_images"]}
    if images_path:
        files = image_list(images_path)
        check_split(development["development_images"], files, dataset, original["settings"]["gray_policy"])
        images = [{**image, "source_index": i} for i, image in enumerate(files)]
    else:
        if dataset != "custom":
            raise ValueError("A named benchmark needs --images")
        images = [{**image, "source_index": i} for i, image in enumerate(development["development_images"])
                  if image["id"] not in seen]
    if not images:
        raise ValueError("No remaining development images")
    used_seeds = set(original["fit_seeds"] + original["check_seeds"]
                     + development["settings"]["evaluation_seeds"])
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s in used_seeds for s in seeds):
        raise ValueError("Use distinct new nonnegative noise seeds")
    choices = {}
    for noise in original["noise_sigma_255"]:
        group = frozen[str(noise)]
        selected = copy.deepcopy(group["performance_selection"]["candidate"])
        if selected["history"]["shape"] == "terminal":
            raise ValueError("Expected the already frozen nonzero-history experiment")
        matched = copy.deepcopy(selected)
        early = matched["history"]["placement"] == "early"
        matched["history"].update({"placement": "delayed" if early else "early", "start": 32 if early else 0})
        arms = [next(a for a in group["arms"] if a["id"] == name)
                for name in ("original_terminal", "tuned_terminal")]
        choices[str(noise)] = [*arms, {**selected, "id": "frozen_history"},
                               {**matched, "id": "matched_order"}]
    spec = {"source_repair": str(source.resolve()), "development_plan": str(development_plan.resolve()),
            "settings": original["settings"], "check_images": images, "check_seeds": seeds,
            "dataset": dataset, "cohort": "external_frozen_evaluation" if images_path else "remaining_development",
            "noise_sigma_255": original["noise_sigma_255"], "readouts": [64], "choices": choices,
            "expected_network_forwards": len(images) * len(seeds) * len(choices) * 4 * 64,
            "scope": "Frozen development prescriptions; original image order and supplied new noise seeds. "
                     "Preserve source image indices and couple standard-normal fields across "
                     "noise amplitudes. Four fixed 64-call strategies; no tuning, label-dependent "
                     "stopping or reselection. The matched-order control changes placement only. "
                     "Solver repair and within-solver history effects remain separate. "
                     "Pretraining independence is not inferred. BSD68/Set12 were viewed in "
                     "earlier experiments; an external evaluation is not declared a new blind test."}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", spec)
    return spec


def paired_summary(rows, spec):
    images = [i["id"] for i in spec["check_images"]]
    arms = ("original_terminal", "tuned_terminal", "frozen_history", "matched_order")
    comparisons = (("tuned_terminal", "original_terminal"), ("frozen_history", "tuned_terminal"),
                   ("matched_order", "frozen_history"))
    image_rows, paired = [], []
    for ni, noise in enumerate(spec["noise_sigma_255"]):
        selected = [r for r in rows if r["noise_255"] == noise]
        if any(r["failure"] for r in selected):
            raise RuntimeError("Failed trajectories retained; do not summarize incomplete groups")
        values = {}
        for image in images:
            values[image] = {}
            for arm in arms:
                observations = [r for r in selected if r["image"] == image and r["candidate"] == arm]
                mse = float(np.mean([r["mse"] for r in observations]))
                values[image][arm] = mse
                image_rows.append({"noise_255": noise, "image": image, "candidate": arm,
                                   "mse": mse, "mean_psnr": float(np.mean([r["psnr"] for r in observations]))})
        rng = np.random.default_rng(2026111720 + ni)
        draws = rng.integers(0, len(images), size=(2000, len(images)))
        for left, right in comparisons:
            a = np.array([values[image][left] for image in images])
            b = np.array([values[image][right] for image in images])
            distribution = 100 * (1 - a[draws].mean(axis=1) / b[draws].mean(axis=1))
            lower, upper = np.quantile(distribution, [.025, .975])
            paired.append({"noise_255": noise, "left": left, "right": right,
                           "mse_gain_percent": float(100 * (1 - a.mean() / b.mean())),
                           "paired_image_bootstrap_low_percent": float(lower),
                           "paired_image_bootstrap_high_percent": float(upper),
                           "mean_mse_difference": float(np.mean(a - b)),
                           "image_wins": int(np.sum(a < b - 1e-8)),
                           "image_ties": int(np.sum(np.abs(a - b) <= 1e-8)),
                           "image_losses": int(np.sum(a > b + 1e-8)), "images": len(images)})
    return image_rows, paired


def run(spec, out, device_name):
    device = configure_device(device_name)
    denoiser = load_denoiser(spec["settings"]["network"], device)
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
    write_csv(out / "summary.csv", [r for r in check_summary(rows) if r["total_calls"] in spec["readouts"]])
    failures = sum(bool(r["failure"]) for r in rows)
    if not failures:
        if denoiser.calls != spec["expected_network_forwards"]:
            raise AssertionError("Actual forward count differs from the frozen plan")
        image_rows, paired = paired_summary(rows, spec)
        write_csv(out / "paired_images.csv", image_rows)
        write_csv(out / "paired_comparisons.csv", paired)
        for result in paired:
            print(result, flush=True)
    write_json(out / "completion.json", {"network_forwards": denoiser.calls,
                                         "expected_network_forwards": spec["expected_network_forwards"],
                                         "failed_trajectories": failures, "images": len(spec["check_images"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--development-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026111713, 2026111714])
    parser.add_argument("--images", type=Path, help="Optional separate image folder; otherwise remaining development images")
    parser.add_argument("--dataset", choices=("custom", "bsd68", "set12"), default="custom")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    spec = prepare(args.source, args.development_plan, args.out, args.seeds, args.images, args.dataset)
    print(f"Frozen validation: {len(spec['check_images'])} images; "
          f"{spec['expected_network_forwards']} forwards", flush=True)
    if not args.plan_only:
        run(spec, args.out, args.device)


if __name__ == "__main__":
    main()
