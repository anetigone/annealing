"""Summarize saved common-terminal trajectories without importing the network."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, out):
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8-sig"))
    records, pairs = read_csv(source / "records.csv"), read_csv(source / "pairs.csv")
    lookup = {(r["image"], r["seed"], r["noise_255"], r["total_calls"], r["family"]): r
              for r in records}
    groups = defaultdict(list)
    details = []
    for pair in pairs:
        key = tuple(pair[k] for k in ("image", "seed", "noise_255", "total_calls"))
        terminal, history = (lookup[(*key, name)] for name in ("terminal", "history"))
        row = {k: pair[k] for k in ("image", "seed", "noise_255", "total_calls")}
        row.update({"terminal_mse": float(terminal["mse"]), "history_mse": float(history["mse"]),
                    "mse_gain_percent": 100 * (1 - float(history["mse"]) / float(terminal["mse"])),
                    "output_rms": float(pair["output_rms"]),
                    "terminal_field_rms": float(terminal["field_rms_at_output"]),
                    "history_field_rms": float(history["field_rms_at_output"]),
                    "distance_over_joint_next_motion": float(pair["distance_over_joint_next_motion"]),
                    "field_secant_projection": float(pair["directional_curvature"]) if pair["directional_curvature"] else None,
                    "effective_data_gain": float(pair["effective_data_gain"]) if pair["effective_data_gain"] else None,
                    "denoiser_projection_gain": float(pair["denoiser_projection_gain"]) if pair["denoiser_projection_gain"] else None,
                    "next_distance_ratio": float(pair["next_distance_ratio"]) if pair["next_distance_ratio"] else None,
                    "bitwise_equal": pair["bitwise_equal"] == "True"})
        details.append(row)
        groups[(float(pair["noise_255"]), int(pair["total_calls"]))].append(row)
    summary = []
    for (noise, k), rows in sorted(groups.items()):
        mean = lambda field: float(np.mean([r[field] for r in rows]))
        pooled = lambda field: float(np.mean([r[field]**2 for r in rows])**.5)
        image_gaps = defaultdict(list)
        for r in rows:
            image_gaps[r["image"]].append(r["history_mse"] - r["terminal_mse"])
        summary.append({"noise_255": noise, "total_calls": k, "observations": len(rows),
                        "terminal_mse": mean("terminal_mse"), "history_mse": mean("history_mse"),
                        "mse_gain_percent": 100 * (1 - mean("history_mse") / mean("terminal_mse")),
                        "pooled_output_rms": pooled("output_rms"),
                        "pooled_terminal_field_rms": pooled("terminal_field_rms"),
                        "pooled_history_field_rms": pooled("history_field_rms"),
                        "negative_secant_projection_count": sum(r["field_secant_projection"] is not None
                                                                 and r["field_secant_projection"] < 0 for r in rows),
                        "next_distance_increase_count": sum(r["next_distance_ratio"] is not None
                                                             and r["next_distance_ratio"] > 1 for r in rows),
                        "bitwise_equal_count": sum(r["bitwise_equal"] for r in rows),
                        "images_better": int(sum(np.mean(gaps) < 0 for gaps in image_gaps.values())),
                        "images": len(image_gaps)})
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "observations.csv", details)
    (out / "summary.json").write_text(json.dumps({
        "source": str(source.resolve()), "plan_max_calls": plan["max_calls"],
        "rows": summary,
        "interpretation": "Squared errors and field norms at actual output states. Equal "
                          "image/seed weights. Pooled RMS is sqrt(mean(per-observation RMS^2)). "
                          "Field secant projections describe observed pairs, without assuming "
                          "a scalar potential or uniform neighborhood stability.",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for row in analyze(args.source, args.out):
        print(row)


if __name__ == "__main__":
    main()
