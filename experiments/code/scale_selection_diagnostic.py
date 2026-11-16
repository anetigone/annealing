"""Select bands from interval inequalities, then test frozen RED schedules."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import mpmath
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "red_geometry_computation_risk/code"))
from family import Family


ALPHA = .001
E1, E2 = .00025, .005
PREFIX, BUDGET = 25, 106
SEED, DRAWS = 20261112, 32
GRID = {"d": [1.6, 1.8, 2., 2.2, 2.4],
        "h": [1.2, 1.5, 1.78, 2.1, 2.4]}
FIXTURE = (2., 1.78)


def rectangles(d, number):
    return [
        ((number(-.1), number(.4)), (number(-.1), number(.15))),
        ((number(.53), number(1.1)), (number(-.1), number(.15))),
        ((number(-1.1), number(.01)), (d-number(.27), d+number(.27))),
    ]


def select(d_value, h_value, rule, details=False):
    # Point intervals formed from exact decimal inputs. Require strict
    # inequalities so uncertain comparisons cannot certify a model.
    iv = mpmath.iv
    num = lambda value: iv.mpf(str(value))
    d, h, r = map(num, (d_value, h_value, 1.29))
    alpha, e1, e2, gamma, delta = map(num, (ALPHA, E1, E2, .25, .0001))
    c_min, c_max, xi, eta = map(num, (.999, 1.002, .3, .01))
    left, exit_s, right, top = map(num, (-.01, .53, 1.1, .05))
    q_b = xi - (alpha*(1-xi)+e1+delta/gamma)/c_min
    gap_b = 1+d*d+2*xi+2*d*eta
    obstacle_checks = {"right_face_threshold_positive": q_b,
                       "right_face_threshold_below_boundary": xi-q_b,
                       "initialization_right": xi-alpha-e1,
                       "transverse_nonovershoot": 1-gamma*(1+c_max),
                       "transverse_face": gamma*((1+c_min)*eta-e2)-delta}

    def low_margin(v):
        odds = r * iv.exp(-(num(.5)-xi)/v)
        return q_b-odds/(1+odds)

    def high_margin(v):
        odds = h * iv.exp(-gap_b/(2*v))
        return q_b-(r-odds)/(1+r+odds)

    low = next((k/100 for k in range(30, 7, -1)
                if low_margin(num(k)/100) > 0), None)
    high = next((k/10 for k in range(15, 81)
                 if high_margin(num(k)/10) > 0), None)
    obstacle_checks.update(low_right_face=low_margin(num(low or .08)),
                           high_right_face=high_margin(num(high or 8)))
    obstacle = all(v > 0 for v in obstacle_checks.values())
    result = {"d": d_value, "h": h_value, "low": low, "high": high,
              "band": None, "reason": "no certified restricted cutoffs"}
    if not obstacle and not details:
        return result

    mu = [(num(0), num(0)), (num(1), num(0)), (-num(1), d)]
    prior = [num(1), r, h]
    boxes = rectangles(d, num)
    channel_checks = {"initial_left": alpha-e1-left, "initial_right": exit_s-alpha-e1,
                      "initial_top": top-e2, "drift_data_sign": alpha*(1-exit_s)-e1,
                      "prepared_right_face": gamma*((alpha+c_min)*(right-1)-e1)-delta,
                      "prepared_overshoot": right-exit_s-gamma*((alpha+c_max)*(1-exit_s)+e1)-delta,
                      "vertical_bound_scope": 3-d}
    classifier = min(alpha/2-e1, 1-e2, d-e2-1)
    terminal = [{}, {}, {}]
    for z in (0, 2):
        for j, (lo, hi) in enumerate(boxes[z]):
            center = mu[z][j]*[alpha, num(1)][j]
            tol = [e1, e2][j]
            terminal[z][f"initial_{j}"] = min(center-tol-lo, hi-center-tol)
    v_l = alpha**2 + alpha**4
    for z, box in enumerate(boxes):
        gaps_by_competitor = {}
        for j in range(3):
            if j == z:
                continue
            gaps = [sum((x-y)**2-(x-w)**2
                        for x, y, w in zip(corner, mu[j], mu[z]))
                    for corner in itertools.product(*box)]
            gap = min(gap.a for gap in gaps)
            gaps_by_competitor[j] = gap
            terminal[z][f"gap_vs_{j}"] = gap
        if not all(gap > 0 for gap in gaps_by_competitor.values()):
            continue
        leakage = sum(prior[j]/prior[z]*iv.exp(-gap/(2*v_l))
                      for j, gap in gaps_by_competitor.items())
        for j, (lo, hi) in enumerate(box):
            clearance = min(mu[z][j]-lo, hi-mu[z][j])
            terminal[z][f"face_{j}"] = (([alpha, num(1)][j]+c_min)*clearance
                                        -([e1, e2][j]+c_max*3*leakage+delta/gamma))
    channel_static = all(v > 0 for v in channel_checks.values())
    terminal_ok = [all(v > 0 for v in row.values()) for row in terminal]
    terminal_ready = classifier > 0 and all(terminal_ok) and c_min <= 1 <= c_max and d < 3
    if not details:
        assert channel_static and terminal_ready

    candidates = []
    channel_candidates = []
    best_budget_margin = None
    stages = dict(geometry_pass=0, drift_pass=0, entry_pass=0)
    start = 40 if rule == "first" else 24
    for k in range(start, 111, 2):
        v0, v1 = num(k-1)/100, num(k+1)/100
        within_cutoffs = obstacle and num(low) < v0 < v1 < num(high)
        if not channel_static or (not details and not within_cutoffs):
            continue
        stages["geometry_pass"] += 1
        rc = r*iv.exp((left-num(.5))/v0)
        hc = h*iv.exp(-(1+d*d+2*left-2*d*top)/(2*v1))
        w = hc/(1+rc+hc)
        slope = (num(.25)+3*w)/v0
        if not (slope < 1 and all(e2-top+c*(d*w-top)+delta/gamma < 0
                                  for c in (c_min, c_max))):
            continue
        re = r*iv.exp((exit_s-num(.5))/v1)
        he = h*iv.exp(-(1+d*d+2*exit_s-2*d*top)/(2*v1))
        surplus = (re-he)/(1+re+he)-exit_s
        if not gamma*c_min*surplus > delta:
            continue
        stages["drift_pass"] += 1
        rate = 1-gamma*c_min*(1-slope)
        lower = alpha-e1
        for call in range(1, PREFIX+1):
            lower = rate*lower + (1-rate)*exit_s + gamma*c_min*surplus-delta
            if rule == "budget" and lower > exit_s:
                stages["entry_pass"] += 1
                entry = (call, k, float(lower.a))
                channel_candidates.append(entry)
                if within_cutoffs:
                    candidates.append(entry)
                break
        if call == PREFIX:
            margin = float((lower-exit_s).a)
            best_budget_margin = margin if best_budget_margin is None else max(best_budget_margin, margin)
        if rule == "first" and lower > exit_s and within_cutoffs:
            result.update(band=[(k-1)/100, (k+1)/100], selected=k/100,
                          s25_lower=float(lower.a), reason="certified")
            return result
    if details:
        lower_ends = lambda row: {key: float(value.a) for key, value in row.items()}
        best = min(channel_candidates, default=None)
        result["checks"] = {
            "obstacle": bool(obstacle), "obstacle_margins": lower_ends(obstacle_checks),
            "channel": best is not None, "channel_margins": lower_ends(channel_checks),
            "channel_candidates": stages,
            "best_entry_margin_at_cap_among_full_runs": best_budget_margin,
            "channel_only_choice": None if best is None else {"calls": best[0], "v": best[1]/100},
            "compatible_bands": len(candidates),
            "classifier_margin": float(classifier.a),
            "terminal_pass": terminal_ok, "terminal_margins": [lower_ends(row) for row in terminal],
        }
    if candidates and terminal_ready:
        call, k, lower = min(candidates)
        result.update(band=[(k-1)/100, (k+1)/100], selected=k/100,
                      certified_calls=call, s_lower=float(lower), reason="certified")
        return result
    if not obstacle:
        result["reason"] = "no certified restricted cutoffs"
    elif not terminal_ready:
        result["reason"] = "classifier or terminal condition unverified"
    else:
        result["reason"] = "no certified band on the prespecified grid"
    return result


def make_plan(out, rule):
    mpmath.iv.dps = 45
    out.mkdir(parents=True, exist_ok=False)
    models = [select(d, h, rule) for d, h in itertools.product(GRID["d"], GRID["h"])]
    plan = {"grid": GRID, "alpha": ALPHA, "rule": rule,
            "noise_seed": SEED if rule == "first" else SEED+1,
            "draws_per_class": DRAWS, "prefix": PREFIX, "budget": BUDGET,
            "noise_sd": ALPHA**2, "candidate_midpoints": [.40 if rule == "first" else .24, 1.10, .02],
            "candidate_half_width": .01, "update_error_allowance": .0001,
            "fixture": list(FIXTURE), "models": models}
    (out/"plan.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")


def protocols(case):
    lo, hi, mid = [case[key] for key in ("low", "high", "selected")]
    schedules = {
        "restricted": ([lo, hi]*12+[lo], False),
        "geometric": (np.geomspace(hi, lo, PREFIX).tolist(), False),
        "shared_platform": ([mid]*PREFIX, False),
        "observation_preset": ([mid]*PREFIX, True),
    }
    if (case["d"], case["h"]) == FIXTURE:
        schedules["skip_band"] = ([hi]*12+[lo]*13, False)
        schedules["one_band_call"] = ([hi]*12+[mid]+[lo]*12, False)
    return schedules


def run(model, case, truth, b, prefix, gated):
    x = b.copy()
    prediction = np.where(b[..., 1] > 1, 2, np.where(b[..., 0] > ALPHA/2, 1, 0))
    select_band = prediction == 1 if gated else np.ones(b.shape[:-1], dtype=bool)
    bounds = np.array(rectangles(case["d"], float))
    low, high = bounds[..., 0][:, None, :], bounds[..., 1][:, None, :]
    first = np.where(np.all((low <= x) & (x <= high), axis=-1), 0, -1)
    lo_band, hi_band = case["band"]
    calls_in_band = sum(lo_band <= v <= hi_band for v in prefix)*select_band
    for k, v in enumerate(prefix + [model.vend]*(BUDGET-PREFIX), start=1):
        active = select_band if k <= PREFIX else np.ones_like(select_band)
        x[active] = model.step(x[active], b[active], v)
        x[~active] = model.step(x[~active], b[~active], model.vend)
        in_region = np.all((low <= x) & (x <= high), axis=-1)
        first[(first < 0) & in_region] = k
        if k == PREFIX:
            at_switch = in_region.copy()
            s_switch = x[1, :, 0].copy()
    squared = np.sum((x-truth)**2, axis=-1)
    assert np.isfinite(squared).all()
    return {"entry_by_switch": at_switch.sum(axis=1).tolist(),
            "first_entry_calls": first.tolist(),
            "all_classes_at_switch": bool(at_switch.all()),
            "class1_at_switch": bool(at_switch[1].all()),
            "class1_s_at_switch": [float(s_switch.min()), float(s_switch.max())],
            "class1_band_calls": int(calls_in_band[1].max()),
            "class_mean_squared_error": squared.mean(axis=1).tolist(),
            "prior_weighted_squared_error": float(model.pi @ squared.mean(axis=1)),
            "budget": BUDGET}


def evaluate(out):
    plan = json.loads((out/"plan.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(plan["noise_seed"])
    records = []
    for case in plan["models"]:
        if case["band"] is None:
            continue
        model = Family(ALPHA, d=case["d"])
        model.pi = np.array([1, 1.29, case["h"]])
        model.pi /= model.pi.sum()
        shape = (3, plan["draws_per_class"], 2)
        truth = model.mu[:, None, :] + model.tau*rng.standard_normal(shape)
        b = (truth @ model.A + plan["noise_sd"]*rng.standard_normal(shape)) @ model.A
        event = np.all(np.abs(b-model.mu[:, None, :] @ model.B) <= [E1, E2], axis=-1)
        results = {name: run(model, case, truth, b, prefix, gated)
                   for name, (prefix, gated) in protocols(case).items()}
        assert not results["restricted"]["class1_at_switch"]
        if event.all():
            assert results["observation_preset"]["all_classes_at_switch"]
        records.append({"d": case["d"], "h": case["h"], "event_count": int(event.sum()),
                        "results": results})

    summary = {}
    for name in ("restricted", "geometric", "shared_platform", "observation_preset"):
        runs = [rec["results"][name] for rec in records]
        summary[name] = {"class1_models_at_switch": sum(r["class1_at_switch"] for r in runs),
                         "all_class_models_at_switch": sum(r["all_classes_at_switch"] for r in runs),
                         "mean_risk": float(np.mean([r["prior_weighted_squared_error"] for r in runs]))}
    result = {"evaluated": len(records), "uncertified": len(plan["models"])-len(records),
              "summary": summary, "records": records}
    with (out/"results.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"evaluated": result["evaluated"], "summary": summary}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rule", choices=("first", "budget"), default="first",
                        help="band selection rule used when writing the plan")
    args = parser.parse_args()
    if args.phase == "plan":
        make_plan(args.out, args.rule)
    else:
        evaluate(args.out)


if __name__ == "__main__":
    main()
