"""Freeze shared and gated platform searches, then evaluate on new noise."""
import argparse
import json
from pathlib import Path

import numpy as np

from scale_selection_diagnostic import ALPHA, BUDGET, E1, E2, FIXTURE, PREFIX, Family, rectangles


V_GRID = np.arange(20, 111) / 100
NOISE_SEED = 20261114
ERROR_LIMIT = 1e-8
REFINEMENT = BUDGET - PREFIX


def write(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def model_for(case):
    model = Family(ALPHA, d=case["d"])
    model.pi = np.array([1, 1.29, case["h"]])
    model.pi /= model.pi.sum()
    return model


def gate(b, alpha=ALPHA):
    return (b[..., 1] <= 1) & (b[..., 0] > alpha/2)


def search(model, gated, *, initial=None):
    b = model.mu @ model.B
    use = gate(b, model.alpha)[:, None] if gated else True
    x = np.broadcast_to(b if initial is None else initial, (len(V_GRID), *b.shape)).copy()
    variance = V_GRID[:, None, None]
    scored, feasible = [], []
    for calls in range(1, PREFIX+1):
        x = np.where(use, model.step(x, b, variance), model.step(x, b, model.vend))
        final = x
        for _ in range(BUDGET-calls):
            final = model.step(final, b, model.vend)
        worst = np.max(np.sum((final-model.mu)**2, axis=-1), axis=-1)
        for v, error in zip(V_GRID, worst):
            scored.append((float(error), calls, float(v)))
            if error <= ERROR_LIMIT:
                feasible.append((calls, float(v)))
    if feasible:
        # Minimize calls, then take the median passing variance at that length.
        # Using the grid midpoint among passing values avoids sub-precision
        # ranking by the nearly zero final error of successful proxies.
        calls = min(pair[0] for pair in feasible)
        passing = [v for m, v in feasible if m == calls]
        v = passing[(len(passing)-1)//2]
    else:
        _, calls, v = min(scored)
    return {"v": v, "calls": calls, "passing_pairs": len(feasible),
            "passing_variances_at_25": [v for m, v in feasible if m == 25]}


def make_plan(source, out):
    prior = json.loads(source.read_text(encoding="utf-8"))
    assert prior["rule"] == "budget" and prior["noise_seed"] != NOISE_SEED
    cases = []
    for case in prior["models"]:
        if case["band"] is None:
            continue
        model = model_for(case)
        cases.append({**case, "draws": 512 if (case["d"], case["h"]) == FIXTURE else 64,
                      "tuned_shared": search(model, False), "tuned_gated": search(model, True)})
    out.mkdir(parents=True, exist_ok=False)
    write(out/"plan.json", {"seed": NOISE_SEED, "candidate_v": V_GRID.tolist(),
                            "candidate_calls": list(range(1, PREFIX+1)),
                            "noiseless_max_class_squared_error_limit": ERROR_LIMIT,
                            "budget": BUDGET, "models": cases})


def run(model, case, truth, b, schedule, gated, budget=BUDGET):
    mask = gate(b) if gated else np.ones(b.shape[:-1], dtype=bool)
    x = b.copy()
    bounds = np.array(rectangles(case["d"], float))
    lo, hi = bounds[..., 0][:, None, :], bounds[..., 1][:, None, :]
    for v in schedule:
        x[mask] = model.step(x[mask], b[mask], v)
        x[~mask] = model.step(x[~mask], b[~mask], model.vend)
    entry = np.all((lo <= x) & (x <= hi), axis=-1)
    switch_s = x[..., 0].copy()
    for _ in range(budget-len(schedule)):
        x = model.step(x, b, model.vend)
    squared = np.sum((x-truth)**2, axis=-1)
    nearest = np.argmin(np.sum((x[..., None, :]-model.mu)**2, axis=-1), axis=-1)
    correct = nearest == np.arange(3)[:, None]
    assert np.isfinite(squared).all()
    return {"entry_counts": entry.sum(axis=1).tolist(),
            "nearest_counts": correct.sum(axis=1).tolist(),
            "class1_model_at_switch": bool(entry[1].all()),
            "all_class_model_at_switch": bool(entry.all()),
            "all_class_model_final_nearest": bool(correct.all()),
            "class_squared_errors": squared.mean(axis=1).tolist(),
            "risk": float(model.pi @ squared.mean(axis=1)),
            "prior_weighted_prefix_calls": float(model.pi @ mask.mean(axis=1)*len(schedule)),
            "switch_s_ranges": np.stack([switch_s.min(axis=1), switch_s.max(axis=1)], -1).tolist(),
            "budget": budget}


def evaluate(out):
    plan = json.loads((out/"plan.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(plan["seed"])
    records = []
    for case in plan["models"]:
        model = model_for(case)
        shape = (3, case["draws"], 2)
        truth = model.mu[:, None, :] + model.tau*rng.standard_normal(shape)
        b = (truth @ model.A + ALPHA**2*rng.standard_normal(shape)) @ model.A
        event = np.all(np.abs(b-model.mu[:, None, :] @ model.B) <= [E1, E2], axis=-1)
        mid, m = case["selected"], case["certified_calls"]
        protocols = {"same_center_shared": ([mid]*PREFIX, False, BUDGET),
                     "preset_25": ([mid]*PREFIX, True, BUDGET),
                     "preset_certified": ([mid]*m, True, BUDGET),
                     "preset_short_budget": ([mid]*m, True, m+REFINEMENT)}
        for key in ("tuned_shared", "tuned_gated"):
            choice = case[key]
            protocols[key] = ([choice["v"]]*choice["calls"], key == "tuned_gated", BUDGET)
        if (case["d"], case["h"]) == FIXTURE:
            protocols["nominal_shared_025"] = ([.25]*25, False, BUDGET)
        results = {k: run(model, case, truth, b, *args) for k, args in protocols.items()}
        if event.all():
            assert results["preset_certified"]["all_class_model_at_switch"]
        records.append({"d": case["d"], "h": case["h"], "draws": case["draws"],
                        "event_count": int(event.sum()), "results": results})
    summary = {}
    for key in records[0]["results"]:
        rows = [r["results"][key] for r in records]
        summary[key] = {k: sum(r[k] for r in rows) for k in
                        ("class1_model_at_switch", "all_class_model_at_switch",
                         "all_class_model_final_nearest")}
        summary[key].update({k: float(np.mean([r[k] for r in rows])) for k in
                             ("risk", "prior_weighted_prefix_calls", "budget")})
    write(out/"results.json", {"summary": summary, "records": records})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=Path("experiments/results/scale_selection_budget/plan.json"))
    args = parser.parse_args()
    if args.phase == "plan":
        make_plan(args.source, args.out)
    else:
        evaluate(args.out)


if __name__ == "__main__":
    main()
