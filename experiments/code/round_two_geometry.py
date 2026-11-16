"""Freeze and test the second-round geometry and competition slice."""
import argparse
import itertools
import json
from pathlib import Path

import mpmath
import numpy as np

from round_one_evidence import plt, run
from scale_selection_diagnostic import ALPHA, BUDGET, E1, E2, rectangles, select
from scale_selection_followup import ERROR_LIMIT, V_GRID, model_for, search, write


D = (1.35, 1.55, 1.7, 2.1, 2.5, 2.75)
H = (.8, 1.35, 1.9, 2.25, 2.65, 3.2)
READOUTS = (25, 40, 64, BUDGET)
SEED = 20261117
DRAWS = 32


def plan(out):
    mpmath.iv.dps = 45
    rows = []
    for d, h in itertools.product(D, H):
        row = select(d, h, "budget", details=True)
        m = model_for(row)
        row["shared"] = search(m, False)
        if row["band"] is None:
            row["fallback"] = search(m, True)
        row["restricted_pair"] = [row["low"], row["high"]] if row["checks"]["obstacle"] else [.18, 2.2]
        rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", {"seed": SEED, "draws_per_class": DRAWS,
                             "alpha": ALPHA, "tau": ALPHA**2, "nu": ALPHA**2,
                             "event_backprojection_tolerance": [E1, E2],
                             "budgets": READOUTS, "grid": {"d": D, "h": H},
                             "template_radius": .1, "mechanism_high_mse": .25,
                             "mechanism_low_mse": 1e-8,
                             "tuning": {"candidate_v": V_GRID.tolist(),
                                        "prefix_calls": [1, 25], "proxy_worst_mse_limit": ERROR_LIMIT,
                                        "rule": "shortest passing prefix, lower median passing variance; "
                                                "otherwise minimize worst proxy error then calls then variance"},
                             "models": rows})


def evaluate(out):
    spec = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(spec["seed"])
    n = spec["draws_per_class"]
    rows = []
    for row in spec["models"]:
        model = model_for(row)
        shape = (3, n, 2)
        truth = model.mu[:, None, :] + model.tau*rng.standard_normal(shape)
        b = (truth @ model.A + spec["nu"]*rng.standard_normal(shape)) @ model.A
        event = np.all(np.abs(b-model.mu[:, None, :] @ model.B) <= [E1, E2], axis=-1)
        predicted = np.where(b[..., 1] > 1, 2, np.where(b[..., 0] > ALPHA/2, 1, 0))
        lo, hi = row["restricted_pair"]
        shared = row["shared"]
        policies = {"terminal": ([], False), "restricted": ([lo, hi]*12+[lo], False),
                    "shared": ([shared["v"]]*shared["calls"], False)}
        certified = row["band"] is not None
        gated = "preset" if certified else "fallback"
        choice = {"v": row["selected"], "calls": row["certified_calls"]} if certified else row[gated]
        policies[gated] = ([choice["v"]]*choice["calls"], True)
        results = {}
        for key, (prefix, gate) in policies.items():
            record, _ = run(model, truth, b, np.array(rectangles(row["d"], float)), prefix, gate)
            record["budgets"] = {k: record["budgets"][k] for k in spec["budgets"]}
            results[key] = record
        if certified and event.all():
            first = np.array(results["preset"]["first_entry"])[1]
            assert np.all((0 <= first) & (first <= row["certified_calls"]))
            assert results["preset"]["at_switch"] == [n]*3
        final = {k: r["budgets"][BUDGET] for k, r in results.items()}
        separated = {}
        for key in ("shared", gated):
            separated[key] = (
                min(final[k]["class_mse"][1] for k in ("terminal", "restricted")) >= spec["mechanism_high_mse"]
                and max(final[key]["class_mse"]) <= spec["mechanism_low_mse"]
                and final[key]["mode_counts"] == final[key]["neighborhood_counts"] == [n]*3
            )
        rows.append({"d": row["d"], "h": row["h"], "certified_calls": row.get("certified_calls"),
                     "obstacle": row["checks"]["obstacle"], "channel": row["checks"]["channel"],
                     "terminal_pass": row["checks"]["terminal_pass"],
                     "event_counts": event.sum(axis=1).tolist(),
                     "classifier_correct": (predicted == np.arange(3)[:, None]).sum(axis=1).tolist(),
                     "separated": separated, "policies": results})
    write(out/"results.json", rows)
    figure(out, spec, rows)


def figure(out, spec, rows):
    grid = spec["grid"]
    coords = {(r["d"], r["h"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4), layout="constrained")
    for panel, ax in enumerate(axes):
        for i, h in enumerate(grid["h"]):
            for j, d in enumerate(grid["d"]):
                r = coords[d, h]
                cap = r["certified_calls"]
                kind = "preset" if cap else "fallback"
                if panel == 0:
                    obstacle = "+" if r["obstacle"] else "-"
                    channel = "+" if r["channel"] else "-"
                    terminal = sum(r["terminal_pass"])
                    label = f"{obstacle} / {channel} / {terminal}\n" + (f"cap {cap}" if cap else "uncert.")
                    color = "#dceee5" if cap else "#faebdd"
                else:
                    key = kind if panel == 1 else "shared"
                    row = r["policies"][key]["budgets"][BUDGET]
                    label = f"{row['mse']:.1e}\n{sum(row['neighborhood_counts'])}/96"
                    color = "#dceee5" if r["separated"][key] else "#faebdd"
                ax.fill_between([j-.5, j+.5], i-.5, i+.5, color=color, linewidth=0)
                ax.text(j, i, label, ha="center", va="center", fontsize=7.3)
        ax.set(xlim=(-.5, 5.5), ylim=(-.5, 5.5), xticks=range(6), yticks=range(6),
               xticklabels=grid["d"], yticklabels=grid["h"], xlabel="Height d")
    axes[0].set(ylabel="Relative mass h", title="Obstacle / channel / terminal classes")
    axes[1].set(title="Certified preset or gated fallback")
    axes[2].set(title="Independently tuned shared")
    fig.supxlabel("Recovery panels: prior-weighted squared error and neighborhood count at 106 calls. "
                  "Green: numerical separation from both controls.", fontsize=8)
    fig.savefig(out/"geometry_slice.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (plan if args.phase == "plan" else evaluate)(args.out)


if __name__ == "__main__":
    main()
