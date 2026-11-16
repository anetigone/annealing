"""Check whether early terminal geometry predicts later visual patch outcomes."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from centered_patch_development import weights
from scale_selection_followup import write
from terminal_ball import terminal_ball_check


SOURCE = Path(__file__).resolve().parents[1]/"results"/"centered_patch_development"
MODELS = ("centered8_tau", "raw8_002")
COMPETITORS = ("terminal", "long_low_high", "tuned_shared")
PROBE, BUDGETS = 1250, (1500, 5000)
BRANCH_DISTANCE = .001


def inputs(kind):
    data = np.load(SOURCE/"data.npz")
    centered = kind == MODELS[0]
    path = "model_8.npz" if centered else "raw_reference_detail/model.npz"
    model = np.load(SOURCE/path)
    operator = data["shape_operator"] if centered else data["operator"]
    sigma = float(np.sqrt(model["tau2"])) if centered else .02
    return data, (operator, operator@operator.T, model["mu"], model["pi"], float(model["tau2"]), sigma)


def update(x, b, model, sigma):
    _, gram, means, pi, tau2, _ = model
    v = tau2+sigma**2
    c = .1*sigma**2/v
    return x-.25*(x@gram-b+c*(x-weights(x, means, pi, v)@means))


def iterate(x, b, model, prefix, last, start=0):
    assert last >= start
    for k in range(start, last):
        x = update(x, b, model, prefix[k] if k < len(prefix) else model[-1])
    return x


def tune(data, model, kind, grid, seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(data["train"]), 48, replace=False)
    pixel_truth = data["train"][ids]
    y = pixel_truth@data["operator"]+.01*rng.standard_normal(pixel_truth.shape)
    operator = model[0]
    centered = kind == MODELS[0]
    b = (y@data["basis"] if centered else y)@operator.T
    truth = pixel_truth@data["basis"] if centered else pixel_truth
    scored = []
    for sigma, length in grid:
        end = iterate(b, b, model, [sigma]*length, BUDGETS[0])
        scored.append({"sigma": sigma, "prefix": length, "mse": float(np.mean(np.sum((end-truth)**2, axis=1)))})
    choice = min(scored, key=lambda q: (q["mse"], q["prefix"], q["sigma"]))
    return [choice["sigma"]]*choice["prefix"], {"choice": choice, "all_candidates": scored,
                                               "train_indices": ids.tolist(), "seed": seed}


def plan(out):
    records = {}
    for kind in MODELS:
        data, model = inputs(kind)
        sigma = model[-1]
        low = .022 if kind == MODELS[1] else 1.1*sigma
        grid = list(itertools.product(sorted({sigma, .035, .06, .10, .16, .26, .4}), (25, 100, 500)))
        prefix, tuning = tune(data, model, kind, grid, 20261127)
        records[kind] = {
            "model": kind, "terminal_sigma": sigma, "tuning": tuning,
            "schedules": {"terminal": [], "middle": [.1]*25, "tuned_shared": prefix,
                          "long_low_high": ([low]*120+[1.]*5)*8},
        }
    out.mkdir(parents=True, exist_ok=False)
    write(out/"plan.json", {
        "source": str(SOURCE.name), "probe_calls": PROBE, "evaluation_calls": BUDGETS,
        "competition_steps": [0, 5, 10, 15, 20], "runner_up_responsibility_min": .10,
        "drift_sign_margin": 1e-6, "branch_distance": BRANCH_DISTANCE,
        "minimum_distinct_per_half": 2, "development_halves": [[0, 128], [128, 256]],
        "admission": "For both long low-high and tuned shared: each existing half has at least two "
                     "competition-linked predicted distinct patches, no contradicted distinct predictions, "
                     "and lower middle mean MSE at both readouts.",
        "models": records,
    })


def competition(b, model, spec):
    x = b.copy()
    _, _, means, pi, tau2, sigma = model
    competition_seen = np.zeros(len(x), dtype=bool)
    sign_change = competition_seen.copy()
    max_runner_up = np.zeros(len(x))
    for k in range(max(spec["competition_steps"])+1):
        middle = update(x, b, model, .1)
        if k in spec["competition_steps"]:
            w = weights(x, means, pi, tau2+.1**2)
            ranked = np.argsort(w, axis=1)[:, -2:]
            runner = w[np.arange(len(w)), ranked[:, 0]]
            max_runner_up = np.maximum(max_runner_up, runner)
            contested = runner >= spec["runner_up_responsibility_min"]
            u = means[ranked[:, 1]]-means[ranked[:, 0]]
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            m = np.sum((middle-x)*u, axis=1)
            competition_seen |= contested
            for endpoint in (sigma, 1.):
                d = np.sum((update(x, b, model, endpoint)-x)*u, axis=1)
                eps = spec["drift_sign_margin"]
                sign_change |= contested & (((m > eps) & (d < -eps)) | ((m < -eps) & (d > eps)))
        x = middle
    return competition_seen, sign_change, max_runner_up


def ball(center, points, b, model):
    _, gram, means, pi, tau2, sigma = model
    v = tau2+sigma**2
    c = .1*sigma**2/v
    reference = int(np.argmax(weights(center[None], means, pi, v)[0]))
    # Use the residual to size a candidate ball, then check it with actual curvature.
    distance = np.linalg.norm(points-center, axis=1).max()
    residual = np.linalg.norm(update(center[None], b[None], model, sigma)[0]-center)
    radius = max(1.01*distance, 1e-4)
    for size in (radius, max(radius, 1500*residual)):
        check = terminal_ball_check(gram, b, means, pi, v, c, .25, center, size, reference, points)
        if check["passed"]:
            return size
    return None


def predict(out):
    spec = json.loads((out/"plan.json").read_text())
    predictions = {}
    saved = {}
    for kind, row in spec["models"].items():
        data, model = inputs(kind)
        y = data["shape_y"] if kind == MODELS[0] else data["observation"]
        b = y@model[0].T
        states = {k: iterate(b, b, model, p, spec["probe_calls"]) for k, p in row["schedules"].items()}
        seen, flips, runner = competition(b, model, spec)
        decisions = {}
        for control in COMPETITORS:
            records = []
            for i, (x, y) in enumerate(zip(states["middle"], states[control], strict=True)):
                d = np.linalg.norm(x-y)
                result = "unresolved"
                if d >= spec["branch_distance"]:
                    a, c = ball(x, x[None], b[i], model), ball(y, y[None], b[i], model)
                    if a is not None and c is not None and d-a-c > spec["branch_distance"]:
                        result = "distinct"
                if result != "distinct" and ball(x, np.array([x, y]), b[i], model) is not None:
                    result = "common"
                records.append(result)
            decisions[control] = records
        predictions[kind] = {"competition": seen.tolist(), "drift_flip": flips.tolist(),
                             "max_runner_up": runner.tolist(), "predictions": decisions}
        saved.update({f"{kind}_{key}": value for key, value in states.items()})
    write(out/"predictions.json", predictions)
    with (out/"probe_states.npz").open("xb") as f:
        np.savez_compressed(f, **saved)


def evaluate(out):
    spec = json.loads((out/"plan.json").read_text())
    pred = json.loads((out/"predictions.json").read_text())
    start = np.load(out/"probe_states.npz")
    records, saved = {}, {}
    for kind, row in spec["models"].items():
        data, model = inputs(kind)
        y = data["shape_y"] if kind == MODELS[0] else data["observation"]
        b = y@model[0].T
        outcomes, errors = {}, {}
        states = {key: start[f"{kind}_{key}"] for key in row["schedules"]}
        previous = spec["probe_calls"]
        for last in spec["evaluation_calls"]:
            states = {k: iterate(x, b, model, row["schedules"][k], last, previous) for k, x in states.items()}
            previous = last
            full = {k: (x@data["basis"].T+data["dc_y"][:, None]*data["dc"] if kind == MODELS[0] else x)
                    for k, x in states.items()}
            errors[last] = {k: np.sum((x-data["truth"])**2, axis=1) for k, x in full.items()}
            assert all(np.isfinite(v).all() for v in errors[last].values())
            outcomes[last] = {k: float(np.mean(y)) for k, y in errors[last].items()}
            saved.update({f"{kind}_{last}_{k}": y for k, y in errors[last].items()})
        scores = {}
        for control in COMPETITORS:
            p = np.array(pred[kind]["predictions"][control])
            d = np.linalg.norm(states["middle"]-states[control], axis=1)
            distinct = d > spec["branch_distance"]
            flips = np.array(pred[kind]["drift_flip"])
            halves = []
            for a, b in spec["development_halves"]:
                sl = slice(a, b)
                halves.append({"competition_linked_distinct": int(((p=="distinct") & distinct & flips)[sl].sum()),
                               "half_mean_mse_diff": [float((x["middle"]-x[control])[sl].mean())
                                                      for x in errors.values()]})
            scores[control] = {
                "predicted_common": int((p=="common").sum()),
                "predicted_distinct": int((p=="distinct").sum()),
                "unresolved": int((p=="unresolved").sum()),
                "later_separated": int(distinct.sum()),
                "distinct_predictions_contradicted": int(((p=="distinct") & ~distinct).sum()),
                "common_predictions_with_later_separation": int(((p=="common") & distinct).sum()),
                "distinct_prediction_indices": np.flatnonzero(p=="distinct").tolist(),
                "halves": halves,
            }
        passes = all(not scores[k]["distinct_predictions_contradicted"] and
                     all(h["competition_linked_distinct"] >= spec["minimum_distinct_per_half"]
                         and max(h["half_mean_mse_diff"]) < 0 for h in scores[k]["halves"])
                     for k in ("long_low_high", "tuned_shared"))
        records[kind] = {"risk": outcomes, "comparisons": scores, "admission_passed": bool(passes),
                         "competition_count": sum(pred[kind]["competition"]), "drift_flip_count": sum(pred[kind]["drift_flip"])}
    write(out/"results.json", records)
    with (out/"errors.npz").open("xb") as f:
        np.savez_compressed(f, **saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "predict", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"plan": plan, "predict": predict, "evaluate": evaluate}[args.phase](args.out)


if __name__ == "__main__":
    main()
