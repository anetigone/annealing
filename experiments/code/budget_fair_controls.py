"""Budget-specific shared tuning and a same-center switching ablation."""
import argparse
import json
from pathlib import Path

import numpy as np

from round_one_evidence import ROOT, run
from scale_selection_diagnostic import ALPHA, BUDGET, E1, E2, rectangles
from scale_selection_followup import V_GRID, model_for, write
from switch_ready_band import certify, IV
from switch_ready_short_budget import BUDGETS, PRACTICAL_DIFF, load


def tune(model, budget):
    b = model.mu@model.B
    x = np.broadcast_to(b, (len(V_GRID), *b.shape)).copy()
    scored = []
    for prefix in range(min(25, budget)+1):
        if prefix:
            x = model.step(x, b, V_GRID[:, None, None])
        final = x.copy()
        for _ in range(budget-prefix):
            final = model.step(final, b, model.vend)
        squared = np.sum((final-model.mu)**2, axis=-1)
        risks = squared@model.pi
        if prefix == 0:
            scored.append((float(risks[0]), 0, float(model.vend)))
        else:
            scored.extend((float(e), prefix, float(v)) for e, v in zip(risks, V_GRID))
    best, calls, v = min(scored)
    return {"v": v, "calls": calls, "proxy_risk": best, "candidate_count": len(scored)}


def plan(out):
    old = load(ROOT/"switch_ready_short_budget/plan.json")
    source = load(ROOT/"switch_ready_diagnostic/plan.json")
    lookup = {(c["d"], c["h"]): c for c in source["models"]}
    IV.dps = 45
    rows = []
    for case in old["models"]:
        row = dict(case)
        if case["included"]:
            v = case["ready"]["v"]
            paired = certify(lookup[case["d"], case["h"]], [v], 5, ".01")["choices"]
            assert paired["ready"] == case["ready"] and paired["direct"] is not None
            row["same_center_direct"] = paired["direct"]
            row["budget_shared"] = {k: tune(model_for(case), k) for k in BUDGETS}
        rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    write(out/"plan.json", {"seed": old["seed"], "draws_per_class": old["draws_per_class"],
                            "budgets": BUDGETS, "comparison_threshold": PRACTICAL_DIFF,
                            "tuning": "For each budget minimize prior-weighted noiseless proxy MSE; "
                                      "ties: shorter prefix then lower variance; m=0 included; "
                                      "v=.20:.01:1.10, m=1:min(25,budget)",
                            "same_center": "Use previously frozen ready center; certify direct entry "
                                           "on the identical .02 band and error event",
                            "models": rows})


def evaluate(out):
    spec = load(out/"plan.json")
    original = load(ROOT/"switch_ready_short_budget/results.json")
    lookup = {(r["d"], r["h"]): r for r in original["records"]}
    rng = np.random.default_rng(spec["seed"])
    n = spec["draws_per_class"]
    rows = []
    for case in spec["models"]:
        model = model_for(case)
        truth = model.mu[:, None, :]+model.tau*rng.standard_normal((3, n, 2))
        b = (truth@model.A+ALPHA**2*rng.standard_normal((3, n, 2)))@model.A
        if not case["included"]:
            continue
        event = np.all(np.abs(b-model.mu[:, None, :]@model.B) <= [E1, E2], axis=-1)
        assert event.all()
        bounds = np.array(rectangles(case["d"], float))
        policies = {}
        for key in ("ready", "same_center_direct"):
            cfg = case[key]
            assert cfg["prefix"] <= min(BUDGETS)
            result, _ = run(model, truth, b, bounds, [cfg["v"]]*cfg["prefix"], True,
                            checkpoints=BUDGETS)
            first = np.array(result["first_entry"])[1]
            assert np.all((0 <= first) & (first <= cfg["entry_bound"]))
            policies[key] = result
        prior = lookup[case["d"], case["h"]]
        assert {str(k): v for k, v in policies["ready"]["budgets"].items()} == prior["policies"]["ready"]["budgets"]
        policies["budget_shared"] = {"budgets": {}}
        for budget in BUDGETS:
            cfg = case["budget_shared"][str(budget)]
            result, _ = run(model, truth, b, bounds, [cfg["v"]]*cfg["calls"], False,
                            checkpoints=(budget,))
            policies["budget_shared"]["budgets"][budget] = result["budgets"][budget]
        rows.append({"d": case["d"], "h": case["h"], "policies": policies})
    summary = {}
    for k in BUDGETS:
        metrics = {}
        for key in rows[0]["policies"]:
            m = [r["policies"][key]["budgets"][k] for r in rows]
            metrics[key] = {"mse": float(np.mean([v["mse"] for v in m])),
                            "hard_mse": float(np.mean([v["class_mse"][1] for v in m])),
                            "mode_count": sum(sum(v["mode_counts"]) for v in m),
                            "neighborhood_count": sum(sum(v["neighborhood_counts"]) for v in m)}
        pairs = {}
        for control in ("same_center_direct", "budget_shared"):
            differences = [r["policies"]["ready"]["budgets"][k]["mse"]
                           -r["policies"][control]["budgets"][k]["mse"] for r in rows]
            hard = [r["policies"]["ready"]["budgets"][k]["class_mse"][1]
                    -r["policies"][control]["budgets"][k]["class_mse"][1] for r in rows]
            eps = spec["comparison_threshold"]
            pairs[control] = {"all_class_differences": differences, "hard_differences": hard,
                              "ready_better": sum(v < -eps for v in differences),
                              "ready_worse": sum(v > eps for v in differences),
                              "hard_better": sum(v < -eps for v in hard),
                              "hard_worse": sum(v > eps for v in hard)}
        summary[k] = {"metrics": metrics, "pairs": pairs}
    write(out/"results.json", {"summary": summary, "records": rows})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", "evaluate"))
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    (plan if args.phase == "plan" else evaluate)(args.out)


if __name__ == "__main__":
    main()
