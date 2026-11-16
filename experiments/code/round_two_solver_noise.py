"""Probe solver initialization and noise outside the nominal certificate."""
import argparse
import itertools
import json
from pathlib import Path

import mpmath
import numpy as np

from round_one_evidence import run
from round_two_geometry import DRAWS, READOUTS
from scale_selection_diagnostic import ALPHA, BUDGET, E1, E2, Family, rectangles, select
from scale_selection_followup import ERROR_LIMIT, V_GRID, search, write


TAU = 1e-6
END_SD = .001
NOMINAL_LAMBDA = 1+1e-6


class FixedTerminal(Family):
    @property
    def vend(self):
        return END_SD**2+self.tau**2


def model_for(row):
    model = FixedTerminal(row["alpha"])
    model.tau, model.lam = TAU, NOMINAL_LAMBDA*row["lambda_ratio"]
    model.pi = np.array([1., 1.29, 1.78])
    model.pi /= model.pi.sum()
    return model


def initialize(b, alpha, beta):
    return b*np.array([1-beta+beta/alpha, 1.])


def plan(out):
    mpmath.iv.dps = 45
    nominal = select(2., 1.78, "budget", details=True)
    settings = [("init_lambda", ALPHA, 1e-6, beta, ratio)
                for beta, ratio in itertools.product((0, .25, .5, .75, 1), (.5, 1, 1.5))]
    settings += [("noise_gain", alpha, nu, 0, 1)
                 for alpha, nu in itertools.product((.0003, .001, .003, .01), (1e-6, 1e-4, .001, .005, .01))]
    rows = []
    for family, alpha, nu, beta, ratio in settings:
        factor = 1-beta+beta/alpha
        # The nominal face checks depend on b, not the initial state.
        # Reuse the barrier only if all event-E starts still fit its region.
        same_map = alpha == ALPHA and ratio == 1
        obstacle = same_map and factor*(alpha+E1) < .3
        full = same_map and beta == 0
        row = dict(slice=family, alpha=alpha, nu=nu, beta=beta, lambda_ratio=ratio,
                   nominal_obstacle_on_e=obstacle, nominal_full_on_e=full,
                   e_start_s_upper_class1=factor*(alpha+E1))
        model = model_for(row)
        b = model.mu @ model.B
        initial = initialize(b, alpha, beta)
        row["shared"] = search(model, False, initial=initial)
        if not full:
            row["fallback"] = search(model, True, initial=initial)
        rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    write(out/"plan.json", {"seeds": {"init_lambda": 20261118, "noise_gain": 20261119},
                            "draws_per_class": DRAWS, "budgets": READOUTS, "tau": TAU,
                            "terminal_sd": END_SD, "nominal_lambda": NOMINAL_LAMBDA,
                            "event_tolerance": [E1, E2], "nominal": nominal,
                            "tuning": {"candidate_v": V_GRID.tolist(), "prefix_calls": [1, 25],
                                       "proxy_worst_mse_limit": ERROR_LIMIT,
                                       "rule": "shortest passing prefix, lower median passing variance; "
                                               "otherwise minimize worst proxy error then calls then variance"},
                            "models": rows})


def evaluate(out):
    spec = json.loads((out/"plan.json").read_text(encoding="utf-8"))
    n = spec["draws_per_class"]
    # Pair standard normals across levels within each slice.
    normals = {}
    for key, seed in spec["seeds"].items():
        rng = np.random.default_rng(seed)
        normals[key] = rng.standard_normal((3, n, 2)), rng.standard_normal((3, n, 2))
    rows = []
    fixed = spec["nominal"]
    for setting in spec["models"]:
        model = model_for(setting)
        alpha, beta = setting["alpha"], setting["beta"]
        u, eps = normals[setting["slice"]]
        truth = model.mu[:, None, :] + model.tau*u
        b = (truth @ model.A + setting["nu"]*eps) @ model.A
        initial = initialize(b, alpha, beta)
        ls = initialize(b, alpha, 1)
        ls_mse = np.sum((ls-truth)**2, axis=-1).mean(axis=1)
        event = np.all(np.abs(b-model.mu[:, None, :] @ model.B) <= [E1, E2], axis=-1)
        prediction = np.where(b[..., 1] > 1, 2, np.where(b[..., 0] > alpha/2, 1, 0))
        shared = setting["shared"]
        gated_key = "preset" if setting["nominal_full_on_e"] else "fallback"
        gated = ({"v": fixed["selected"], "calls": fixed["certified_calls"]}
                 if gated_key == "preset" else setting["fallback"])
        policies = {"terminal": ([], False),
                    "restricted": ([fixed["low"], fixed["high"]]*12+[fixed["low"]], False),
                    "shared": ([shared["v"]]*shared["calls"], False),
                    gated_key: ([gated["v"]]*gated["calls"], True)}
        results = {}
        for key, (prefix, gate) in policies.items():
            record, _ = run(model, truth, b, np.array(rectangles(2., float)), prefix, gate,
                            initial=initial, obstacle_diagnostic=key == "restricted")
            record["budgets"] = {k: record["budgets"][k] for k in spec["budgets"]}
            record["first_budget_all_near"] = next((k for k, m in record["budgets"].items()
                                                    if m["neighborhood_counts"] == [n]*3), None)
            results[key] = record
        if setting["nominal_full_on_e"]:
            first = np.array(results["preset"]["first_entry"])[1, event[1]]
            assert np.all((0 <= first) & (first <= fixed["certified_calls"]))
        if setting["nominal_obstacle_on_e"] and event[1].all():
            obs = results["restricted"]["obstacle"]
            assert obs == {"starts_inside": n, "ever_crossed_given_start_inside": 0}
        rows.append({**setting, "event_counts": event.sum(axis=1).tolist(),
                     "classifier_correct": (prediction == np.arange(3)[:, None]).sum(axis=1).tolist(),
                     "ls_mse": float(model.pi @ ls_mse), "ls_class_mse": ls_mse.tolist(),
                     "policies": results})
    assert len(rows) == 35 and all(len(row["policies"]["shared"]["first_entry"]) == 3 for row in rows)
    write(out/"results.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (plan if args.phase == "plan" else evaluate)(args.out)


if __name__ == "__main__":
    main()
