"""Compare frozen band centers at matched budgets on the existing observations."""
import argparse
import json
from pathlib import Path

import numpy as np

from round_one_evidence import ROOT, plt, run
from scale_selection_diagnostic import ALPHA, BUDGET, E1, E2, rectangles
from scale_selection_followup import ERROR_LIMIT, V_GRID, model_for, search, write


BUDGETS = (21, 25, 32, 40, BUDGET)
PRACTICAL_DIFF = 1e-10


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plan(out):
    base = load(ROOT/"switch_ready_diagnostic/plan.json")
    bands = load(ROOT/"switch_ready_band/results.json")
    lookup = {(m["d"], m["h"]): m for m in bands["models"]}
    models = []
    for model in base["models"]:
        row = lookup[model["d"], model["h"]]
        # Keep the original draw order, including the four excluded points.
        case = {"d": row["d"], "h": row["h"], "included": row["original_terminal_pass"]}
        if case["included"]:
            case.update(row["choices"], shared=search(model_for(row), False))
            assert max(case["direct"]["prefix"], case["ready"]["prefix"], case["shared"]["calls"]) <= min(BUDGETS)
        models.append(case)
    assert sum(m["included"] for m in models) == 12
    out.mkdir(parents=True, exist_ok=False)
    write(out/"plan.json", {"seed": base["seed"], "draws_per_class": base["draws_per_class"],
                            "budgets": BUDGETS, "comparison_threshold": PRACTICAL_DIFF,
                            "shared_tuning": {"variance_candidates": V_GRID.tolist(),
                                              "prefix_calls": [1, 25],
                                              "proxy_total_calls": BUDGET,
                                              "proxy_worst_class_error_limit": ERROR_LIMIT,
                                              "selection": "shortest passing prefix then lower median passing variance; "
                                                           "otherwise minimize worst proxy error then calls then variance"},
                            "models": models})


def evaluate(out):
    plan = load(out/"plan.json")
    source = load(ROOT/"switch_ready_diagnostic/results.json")
    old = {(r["d"], r["h"]): r for r in source}
    rng = np.random.default_rng(plan["seed"])
    n = plan["draws_per_class"]
    records = []
    for case in plan["models"]:
        model = model_for(case)
        truth = model.mu[:, None, :] + model.tau*rng.standard_normal((3, n, 2))
        b = (truth @ model.A + ALPHA**2*rng.standard_normal((3, n, 2))) @ model.A
        if not case["included"]:
            continue
        event = np.all(np.abs(b-model.mu[:, None, :]@model.B) <= [E1, E2], axis=-1)
        bounds = np.array(rectangles(case["d"], float))
        shared = case["shared"]
        schedules = {"terminal": ([], False), "shared": ([shared["v"]]*shared["calls"], False)}
        for key in ("direct", "ready"):
            cfg = case[key]
            schedules[key] = ([cfg["v"]]*cfg["prefix"], True)
        results = {key: run(model, truth, b, bounds, schedule, gate,
                            checkpoints=tuple(plan["budgets"]))[0]
                   for key, (schedule, gate) in schedules.items()}
        prior = old[case["d"], case["h"]]
        assert event.sum(axis=1).tolist() == prior["event_counts"]
        assert results["terminal"]["budgets"][BUDGET] == prior["policies"]["terminal"]["final"]
        for key in ("direct", "ready"):
            first = np.array(results[key]["first_entry"])[1, event[1]]
            assert np.all((0 <= first) & (first <= case[key]["entry_bound"]))
        records.append({"d": case["d"], "h": case["h"], "event_counts": event.sum(axis=1).tolist(),
                        "policies": results})
    data = summarize(records, plan)
    write(out/"results.json", {"summary": data, "records": records})
    figure(out, data, plan)


def summarize(rows, plan):
    result = {}
    for budget in plan["budgets"]:
        m = {key: [r["policies"][key]["budgets"][budget] for r in rows]
             for key in rows[0]["policies"]}
        stats = {key: {"mse": float(np.mean([x["mse"] for x in values])),
                       "hard_class_mse": float(np.mean([x["class_mse"][1] for x in values])),
                       "mode_count": sum(sum(x["mode_counts"]) for x in values),
                       "neighborhood_count": sum(sum(x["neighborhood_counts"]) for x in values)}
                 for key, values in m.items()}
        pairs = {}
        for control in ("direct", "shared"):
            d = [x["mse"]-y["mse"] for x, y in zip(m["ready"], m[control], strict=True)]
            hard = [x["class_mse"][1]-y["class_mse"][1]
                    for x, y in zip(m["ready"], m[control], strict=True)]
            eps = plan["comparison_threshold"]
            pairs[control] = {"per_model": d, "per_model_hard": hard,
                              "mean": float(np.mean(d)), "hard_mean": float(np.mean(hard)),
                              "below": sum(x < -eps for x in d), "above": sum(x > eps for x in d),
                              "hard_below": sum(x < -eps for x in hard),
                              "hard_above": sum(x > eps for x in hard)}
        result[budget] = {"metrics": stats, "ready_minus": pairs}
    return result


def figure(out, results, plan):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", sharey=True)
    line = plan["comparison_threshold"]
    for panel, field, avg, title in zip(ax, ("per_model", "per_model_hard"), ("mean", "hard_mean"),
                                      ("All classes (prior-weighted)", "Difficult class")):
        for key, offset, label, color in (("direct", -.16, "Ready − direct", "#176f55"),
                                          ("shared", .16, "Ready − tuned shared", "#985329")):
            for j, k in enumerate(plan["budgets"]):
                comp = results[k]["ready_minus"][key]
                x = j + offset + np.linspace(-.07, .07, len(comp[field]))
                panel.scatter(x, comp[field], color=color, s=13, alpha=.45, linewidths=0)
                panel.scatter(j+offset, comp[avg], label=label if j == 0 else None,
                              color=color, marker="D", s=26, zorder=4)
        panel.axhspan(-line, line, color=".93", zorder=0)
        panel.axhline(0, color=".45", lw=.8, zorder=1)
        panel.set(title=title, xlabel="Total denoiser calls",
                  xticks=range(len(plan["budgets"])), xticklabels=plan["budgets"])
        panel.set_yscale("symlog", linthresh=line)
    ax[0].set_ylabel("Paired difference in squared error")
    ax[1].legend(frameon=False, fontsize=8)
    fig.supxlabel("Lower favors ready. Dots: models; diamonds: model means.", fontsize=8)
    fig.savefig(out/"paired_error.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (plan if args.phase == "plan" else evaluate)(args.out)


if __name__ == "__main__":
    main()
