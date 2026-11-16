"""Post hoc trajectory and certificate diagnosis of frozen budget controls."""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from budget_control_validation import load, model_for, write
from scale_selection_followup import gate
from switch_ready_band import advance, band_segments
from switch_ready_diagnostic import IV, N, READY, describe, inside


ROOT = Path(__file__).resolve().parents[1] / "results"


def config(case, record, budget, name):
    if name in ("ready", "same_center_direct"):
        return {"v": case[name]["v"], "calls": case[name]["prefix"]}
    return record["choices"][str(budget)][name]


def trace(model, b, cfg, gated, budget):
    x = b.copy()
    active = gate(b, model.alpha) if gated else np.ones(b.shape[:-1], bool)
    states = [x.copy()]
    for k in range(budget):
        if k < cfg["calls"]:
            x[active] = model.step(x[active], b[active], cfg["v"])
            x[~active] = model.step(x[~active], b[~active], model.vend)
        else:
            x = model.step(x, b, model.vend)
        states.append(x.copy())
    return np.array(states)


def risk(model, truth, x):
    return float(model.pi @ np.sum((x-truth)**2, -1).mean(1))


def terminal_parts(model, truth, b, states, start):
    """Identity uses label-conditioned fixed points, not a label-based controller."""
    q = 1-.25*(np.diag(model.B)+model.c(model.vend))
    fixed = (b+model.c(model.vend)*model.mu[:, None, :]) / (
        np.diag(model.B)+model.c(model.vend))
    transient = (states[start]-fixed)*q**(len(states)-1-start)
    base = fixed-truth
    predicted = fixed+transient
    return {"risk": risk(model, truth, states[-1]),
            "fixed_point_risk": float(model.pi @ np.sum(base**2, -1).mean(1)),
            "transient_squared": float(model.pi @ np.sum(transient**2, -1).mean(1)),
            "cross_term": float(2*model.pi @ np.sum(base*transient, -1).mean(1)),
            "affine_error_max": float(np.max(np.abs(predicted-states[-1]))),
            "switch_hard_mean": states[start, 1].mean(0).tolist(),
            "switch_hard_range": np.stack([states[start, 1].min(0),
                                            states[start, 1].max(0)], 1).tolist()}


def certificate(case, cfg, half_width):
    IV.dps = 45
    vs = band_segments(N(cfg["v"]), N(half_width), 5 if half_width else 1)
    tau, lam = N(.000001), 1+N(.001)**2
    segments = [(v, lam*(1-tau**2/v)) for v in vs]
    lower, upper = (N(.00075), N(.005)), (N(.00125), N(-.005))
    for _ in range(cfg["calls"]):
        lower, upper = advance(lower, upper, (N(.00075), N(.005)),
                               (N(.00125), N(-.005)), segments,
                               N(case["d"]), N(case["h"]), N(.0001))
    return {"ready_sufficient_check": bool(inside(lower, upper, READY)),
            "enclosure": describe(lower, upper)}


def nominal(source, spec, choices, data):
    rows = []
    for mi, case in enumerate(spec["models"]):
        model, rec = model_for(case), choices[0, mi]
        truth, b = data[f"truth_0_{mi}"], data[f"b_0_{mi}"]
        ready, tuned = config(case, rec, 22, "ready"), config(case, rec, 22, "noise_gated")
        counter = {"v": tuned["v"], "calls": ready["calls"]}
        values = {}
        for name, cfg in (("ready", ready), ("tuned", tuned), ("variance_only", counter)):
            states = trace(model, b, cfg, True, 22)
            if name != "variance_only":
                saved = data[f"x_0_{mi}_22_{'ready' if name == 'ready' else 'noise_gated'}"]
                assert np.array_equal(states[-1], saved)
            values[name] = {"config": cfg, **terminal_parts(model, truth, b, states, cfg["calls"])}
        rr, tr, vr = (values[n]["risk"] for n in ("ready", "tuned", "variance_only"))
        rows.append({"model_index": mi, "d": case["d"], "h": case["h"],
                     "values": values, "ready_minus_tuned": rr-tr,
                     "variance_effect": rr-vr, "prefix_effect_after_variance": vr-tr,
                     "tuned_fixed_certificate": certificate(case, tuned, 0),
                     "tuned_band_certificate": certificate(case, tuned, .01)})
    return rows


def switch_margin(model, x, z):
    # Signed distance to each equal-posterior plane at the terminal variance.
    other = [j for j in range(3) if j != z]
    gaps = [.5*(np.sum((x-model.mu[j])**2)-np.sum((x-model.mu[z])**2)) +
            model.vend*np.log(model.pi[z]/model.pi[j]) for j in other]
    distances = [g/np.linalg.norm(model.mu[z]-model.mu[j]) for g, j in zip(gaps, other)]
    return float(min(distances))


def failures(spec, choices, data):
    rows, prefixes = [], []
    for ni, nu in enumerate(spec["measurement_noise_sd"]):
        if ni == 0:
            continue
        for mi, case in enumerate(spec["models"]):
            model, rec = model_for(case), choices[ni, mi]
            truth, b = data[f"truth_{ni}_{mi}"], data[f"b_{ni}_{mi}"]
            for name in ("noise_shared", "noise_gated"):
                early = config(case, rec, 22, name)
                continued = trace(model, b, early, name == "noise_gated", 106)[-1]
                nearest = np.sum((continued[..., None, :]-model.mu)**2, -1).argmin(-1)
                prefixes.append({"noise_index": ni, "model_index": mi, "policy": name,
                                 "config_22": early, "risk_at_106": risk(model, truth, continued),
                                 "wrong_by_class": (nearest != np.arange(3)[:, None]).sum(1).tolist()})
                for budget in (22, 106):
                    cfg = config(case, rec, budget, name)
                    states = trace(model, b, cfg, name == "noise_gated", budget)
                    saved = data[f"x_{ni}_{mi}_{budget}_{name}"]
                    assert np.array_equal(states[-1], saved)
                    nearest = np.sum((saved[..., None, :]-model.mu)**2, -1).argmin(-1)
                    for z, j in np.argwhere(nearest != np.arange(3)[:, None]):
                        bj = b[z, j]
                        mean = model.mu[z] @ model.B
                        replacements = {}
                        for coordinate in (0, 1):
                            replaced = bj.copy()
                            replaced[coordinate] = mean[coordinate]
                            state = trace(model, replaced[None, None], cfg,
                                          name == "noise_gated", budget)
                            end = state[-1, 0, 0]
                            mode = int(np.sum((end-model.mu)**2, -1).argmin())
                            replacements[str(coordinate)] = {"mode": mode,
                                "switch_margin": switch_margin(model, state[cfg["calls"], 0, 0], z)}
                        pred = 2 if bj[1] > 1 else int(bj[0] > model.alpha/2)
                        rows.append({"noise_index": ni, "model_index": mi, "policy": name,
                                     "budget": budget, "class": int(z), "draw": int(j),
                                     "chosen_mode": int(nearest[z, j]), "config": cfg,
                                     "b": bj.tolist(), "classifier_correct": bool(pred == z),
                                     "inside_event": bool(np.all(np.abs(bj-mean) <= [.00025, .005])),
                                     "switch": states[cfg["calls"], z, j].tolist(),
                                     "switch_margin": switch_margin(model, states[cfg["calls"], z, j], z),
                                     "coordinate_replacements": replacements})
    return rows, prefixes


def flatness(spec, choices, calibration):
    rows = []
    for ni in (1, 2):
        for mi, case in enumerate(spec["models"]):
            model, rec = model_for(case), choices[ni, mi]
            truth, b = calibration[f"truth_{ni}_{mi}"], calibration[f"b_{ni}_{mi}"]
            scores = calibration[f"scores_{ni}_{mi}"][4]
            for p, name in enumerate(("noise_shared", "noise_gated")):
                values = {}
                for key, cfg in (("long_choice", config(case, rec, 106, name)),
                                 ("short_choice", config(case, rec, 22, name))):
                    states = trace(model, b, cfg, name == "noise_gated", 106)
                    values[key] = {"config": cfg, **terminal_parts(model, truth, b, states, cfg["calls"])}
                best = rec["choices"]["106"][name]["calibration_risk"]
                rows.append({"noise_index": ni, "model_index": mi, "policy": name,
                             "near_minimum_counts": {str(t): int(np.sum(scores[p, 1:] <= best+t))
                                                      for t in (1e-18, 1e-15, 1e-12, 1e-10)},
                             "values": values})
    return rows


def sensitivities(out, spec, choices):
    targets = load(out / "plan.json")["deterministic_probability_check"]["targets"]
    rows = []
    for ti, target in enumerate(targets):
        ni, mi, policy, budget = (target[n] for n in ("noise_index", "model_index", "policy", "choice_budget"))
        case, rec = spec["models"][mi], choices[ni, mi]
        model, nu = model_for(case), spec["measurement_noise_sd"][ni]
        z = 0 if policy == "noise_shared" else 1
        b = model.mu[z] @ model.B
        sigma = np.sqrt(model.tau**2*np.diag(model.B)**2+nu**2*np.diag(model.B))
        for name, cfg in (("original", config(case, rec, budget, policy)),
                          ("ready", config(case, rec, 22, "ready"))):
            x, jac = b.copy(), np.eye(2)
            for _ in range(cfg["calls"]):
                w = model.weights(x, cfg["v"])
                centered = model.mu-w @ model.mu
                cov = (centered.T*w) @ centered
                state_jac = np.eye(2)-.25*(model.B+model.c(cfg["v"])*np.eye(2))
                state_jac += .25*model.c(cfg["v"])/cfg["v"]*cov
                jac = state_jac @ jac+.25*np.eye(2)
                x = model.step(x, b, cfg["v"])
            numerical = np.zeros((2, 2))
            for coordinate in range(2):
                step = np.zeros(2)
                step[coordinate] = 1e-8 if coordinate == 0 else 1e-6
                endpoints = []
                for point in (b-step, b+step):
                    state = point.copy()
                    for _ in range(cfg["calls"]):
                        state = model.step(state, point, cfg["v"])
                    endpoints.append(state)
                numerical[:, coordinate] = (endpoints[1]-endpoints[0])/(2*step[coordinate])
            assert np.max(np.abs(jac-numerical)) < 1e-7
            sign = 1 if z == 1 else -1
            margin = sign*(x[0]-.5+model.vend*np.log(model.pi[1]/model.pi[0]))
            sd = np.linalg.norm(jac[0]*sigma)
            rows.append({"target": ti, "label": name, "class": z, "config": cfg,
                         "switch_at_mean_b": x.tolist(), "platform_jacobian_wrt_b": jac.tolist(),
                         "finite_difference_max_error": float(np.max(np.abs(jac-numerical))),
                         "signed_margin_at_mean": float(margin), "linearized_margin_sd": float(sd),
                         "margin_in_sd": float(margin/sd),
                         "raw_branch_linearized_failure_probability": float(ndtr(-margin/sd))})
    write(out / "sensitivity.json", {
        "scope": "Local derivative of the constant-platform prefix at class-mean b, checked by central differences. For the ready comparator this diagnoses the raw platform branch, including the hypothetical class-0 branch; gate discontinuities are excluded. Linearized probabilities are explanatory approximations, not risk guarantees. Full frozen-policy probabilities use separate quadrature.",
        "records": rows})
    print("local sensitivities", [(r["target"], r["label"], r["margin_in_sd"]) for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--certificates-only", action="store_true")
    parser.add_argument("--sensitivity-only", action="store_true")
    args = parser.parse_args()
    load(args.out / "plan.json")  # Diagnosis is specified before producing new outputs.
    source = ROOT / "budget_control_validation"
    spec = load(source / "plan.json")
    choices = {(r["noise_index"], r["model_index"]): r for r in load(source / "choices.json")["records"]}
    if args.sensitivity_only:
        sensitivities(args.out, spec, choices)
        return
    if args.certificates_only:
        rows = []
        for mi, case in enumerate(spec["models"]):
            tuned = config(case, choices[0, mi], 22, "noise_gated")
            counter = {"v": tuned["v"], "calls": case["ready"]["prefix"]}
            rows.append({"model_index": mi, "variance_only_config": counter,
                         "variance_only_band": certificate(case, counter, .01),
                         "ready_band_recheck": certificate(case, config(case, choices[0, mi], 22, "ready"), .01)})
        write(args.out / "certificate_controls.json", {"records": rows})
        print("variance-only band checks", sum(r["variance_only_band"]["ready_sufficient_check"] for r in rows))
        return
    if (args.out / "trajectories.json").exists():
        raise FileExistsError("Diagnosis outputs already exist")
    data, calibration = np.load(source / "evaluation.npz"), np.load(source / "calibration.npz")
    first = nominal(source, spec, choices, data)
    wrong, prefixes = failures(spec, choices, data)
    write(args.out / "trajectories.json", {"nominal": first, "failures": wrong,
                                           "short_config_continued": prefixes,
                                           "flatness": flatness(spec, choices, calibration)})
    print("nominal cases", len(first), "failure records", len(wrong), flush=True)
    print("mean variance effect", np.mean([r["variance_effect"] for r in first]),
          "mean prefix effect", np.mean([r["prefix_effect_after_variance"] for r in first]))
    print("tuned sufficient checks fixed/band", sum(r["tuned_fixed_certificate"]["ready_sufficient_check"] for r in first),
          sum(r["tuned_band_certificate"]["ready_sufficient_check"] for r in first))


if __name__ == "__main__":
    main()
