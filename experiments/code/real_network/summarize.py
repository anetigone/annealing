"""Paired image summaries; repeated noise stays clustered within each image."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from itertools import combinations

import numpy as np

from common import read_json, write_csv, write_json
from schedules import FAMILIES


def summarize(out):
    out = Path(out)
    if not (out / "completion.json").is_file():
        raise RuntimeError("Evaluation is incomplete; completion.json is missing")
    spec = read_json(out / "plan.json")
    with (out / "records.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    baseline, groups, indexed = {}, defaultdict(list), {}
    for row in rows:
        key = (row["image"], row["kernel"], row["noise_255"], row["seed"], row["budget"], row["extra_calls"])
        if row["family"] == "terminal":
            baseline[key] = row
        indexed[(key, row["family"])] = row
        groups[(row["kernel"], row["noise_255"], row["budget"], row["extra_calls"], row["family"])].append(row)
    summaries, paired = [], []
    tolerance = spec["settings"]["search"]["mse_tolerance"]
    for key, records in sorted(groups.items()):
        image_groups = defaultdict(list)
        for row in records:
            image_groups[row["image"]].append(row)
        image_scores, differences, psnr_differences = [], [], []
        for image, repeated in image_groups.items():
            refs = [baseline[(r["image"], r["kernel"], r["noise_255"], r["seed"], r["budget"], r["extra_calls"])] for r in repeated]
            success = all(r["status"] == "ok" for r in repeated)
            pair_success = success and all(r["status"] == "ok" for r in refs)
            mse = float(np.mean([float(r["mse"]) for r in repeated])) if success else None
            delta = float(np.mean([float(a["mse"])-float(b["mse"]) for a, b in zip(repeated, refs)])) if pair_success else None
            psnr_delta = float(np.mean([float(a["psnr"])-float(b["psnr"]) for a, b in zip(repeated, refs)])) if pair_success else None
            paired.append({"image": image, "kernel": key[0], "noise_255": key[1], "budget": key[2],
                           "extra_calls": key[3], "family": key[4], "mse": mse,
                           "mse_minus_terminal": delta, "psnr_minus_terminal": psnr_delta,
                           "all_noise_replicates_finite": success})
            if success:
                image_scores.append({field: float(np.mean([float(r[field]) for r in repeated]))
                                     for field in ("mse", "psnr", "ssim", "clipped_psnr", "weak_mse", "seconds")})
            if pair_success:
                differences.append(delta)
                psnr_differences.append(psnr_delta)
        n = len(image_groups)
        complete = len(image_scores) == n
        paired_complete = len(differences) == n
        values = np.asarray(differences)
        interval = None
        if paired_complete and n >= 2:
            rng = np.random.default_rng(20261116)
            boot = rng.choice(values, size=(2000, n), replace=True).mean(axis=1)
            interval = np.quantile(boot, [.025, .975]).tolist()
        summaries.append({
            "kernel": key[0], "noise_255": float(key[1]), "budget": int(key[2]),
            "extra_calls": int(key[3]), "family": key[4], "images": n,
            "finite_images": len(image_scores), "failed_records": sum(r["status"] != "ok" for r in records),
            "means": {field: float(np.mean([r[field] for r in image_scores]))
                      for field in image_scores[0]} if complete else None,
            "mean_mse_minus_terminal": float(values.mean()) if paired_complete else None,
            "median_mse_minus_terminal": float(np.median(values)) if paired_complete else None,
            "mean_psnr_minus_terminal": float(np.mean(psnr_differences)) if paired_complete else None,
            "image_bootstrap_mse_ci95": interval,
            "wins_ties_losses_on_finite_pairs": [int(np.sum(values < -tolerance)),
                                                 int(np.sum(np.abs(values) <= tolerance)),
                                                 int(np.sum(values > tolerance))],
            "finite_paired_images": len(values),
        })
    comparisons, all_pairs = [], []
    conditions = sorted({key[:4] for key in groups})
    for condition in conditions:
        for left, right in combinations(FAMILIES, 2):
            repeated = defaultdict(list)
            for row in groups[(*condition, left)]:
                key = (row["image"], row["kernel"], row["noise_255"], row["seed"], row["budget"], row["extra_calls"])
                ref = indexed[(key, right)]
                difference = float(row["mse"])-float(ref["mse"]) if row["status"] == ref["status"] == "ok" else None
                repeated[row["image"]].append(difference)
            values = []
            for image, draws in repeated.items():
                value = float(np.mean(draws)) if all(v is not None for v in draws) else None
                all_pairs.append({"image": image, "kernel": condition[0], "noise_255": condition[1],
                                  "budget": condition[2], "extra_calls": condition[3], "left": left,
                                  "right": right, "mse_left_minus_right": value})
                if value is not None:
                    values.append(value)
            values = np.asarray(values)
            complete = len(values) == len(repeated)
            interval = None
            if complete and len(values) >= 2:
                rng = np.random.default_rng(20261116)
                interval = np.quantile(rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1), [.025, .975]).tolist()
            comparisons.append({"kernel": condition[0], "noise_255": float(condition[1]),
                                "budget": int(condition[2]), "extra_calls": int(condition[3]),
                                "left": left, "right": right, "finite_paired_images": len(values),
                                "mean_mse_left_minus_right": float(values.mean()) if complete else None,
                                "median_mse_left_minus_right": float(np.median(values)) if complete else None,
                                "image_bootstrap_mse_ci95": interval,
                                "wins_ties_losses_on_finite_pairs": [int(np.sum(values < -tolerance)),
                                                                     int(np.sum(np.abs(values) <= tolerance)),
                                                                     int(np.sum(values > tolerance))]})
    write_csv(out / "paired_images.csv", paired)
    write_csv(out / "paired_comparisons.csv", all_pairs)
    write_json(out / "summary.json", {
        "rows": summaries, "pairwise_comparisons": comparisons, "pair_threshold_mse": tolerance,
        "scope": "Descriptive results for these images and frozen development choices. "
                 "2000 paired image-bootstrap draws per condition; seeds remain within image. "
                 "Intervals do not account for development selection or multiple comparisons. "
                 "A group containing failed images has no all-image mean; failures are retained.",
    })
    print(f"Summary saved: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    summarize(parser.parse_args().out)
