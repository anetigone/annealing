"""Select a fixed platform by certifying entry to a terminal preimage box."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from round_one_evidence import run
from round_two_failure_check import IV, N, step
from scale_selection_diagnostic import ALPHA, BUDGET, E1, E2, rectangles, select
from scale_selection_followup import model_for, write


D = (1.30, 1.43, 1.62, 1.85)
H = (1.55, 2.05, 2.45, 3.0)
V = tuple(k/100 for k in range(24, 111, 2))
CAP, DRAWS, SEED = 25, 16, 20261120
READY = ((.50001, .65), (-.02, .05))
TARGET = ((.53, 1.1), (-.1, .15))


def inside(lower, upper, box):
    # Lower and upper for the order (s,-t), with strict interval comparisons.
    return (lower[0].a > N(box[0][0]) and upper[0].b < N(box[0][1])
            and upper[1].a > N(box[1][0]) and lower[1].b < N(box[1][1]))


def describe(lower, upper):
    return [[float(lower[0].a), float(upper[0].b)], [float(upper[1].a), float(lower[1].b)]]


def certify(d, h, obstacle):
    b_minus = (N(.00075), N(.005))
    b_plus = (N(.00125), N(-.005))
    delta = N(.0001)
    neg, pos = (-delta, delta), (delta, -delta)
    d, h = N(d), N(h)
    vend = N(.001)**2 + N(.000001)**2
    first = step((N(READY[0][0]), N(READY[1][1])), b_minus, vend, d, h, neg)
    last = step((N(READY[0][1]), N(READY[1][0])), b_plus, vend, d, h, pos)
    ready_valid = inside(first, last, TARGET)
    choices = {"direct": None, "ready": None}
    n_pass = {k: 0 for k in choices}
    if obstacle["checks"]["obstacle"]:
        for variance in V:
            if not obstacle["low"] < variance < obstacle["high"]:
                continue
            v = N(variance)
            lower, upper = b_minus, b_plus
            passed = set()
            for k in range(1, CAP+1):
                lower = step(lower, b_minus, v, d, h, neg)
                upper = step(upper, b_plus, v, d, h, pos)
                for key, box, offset in (("direct", TARGET, 0), ("ready", READY, 1)):
                    if key in passed or (key == "ready" and (not ready_valid or k == CAP)):
                        continue
                    if inside(lower, upper, box):
                        passed.add(key)
                        n_pass[key] += 1
                        choice = dict(v=variance, prefix=k, entry_bound=k+offset,
                                      switch_enclosure=describe(lower, upper))
                        saved = choices[key]
                        if saved is None or (k, variance) < (saved["prefix"], saved["v"]):
                            choices[key] = choice
                if len(passed) == 2:
                    break
    return {"preimage_verified": bool(ready_valid), "terminal_image": describe(first, last),
            "choices": choices, "passing_variances": n_pass}


def plan(out):
    IV.dps = 45
    cases = []
    for d, h in itertools.product(D, H):
        base = select(d, h, "budget", details=True)
        cert = certify(d, h, base)
        # Both policies still need the shared gate and all-class terminal tests.
        checks = base["checks"]
        if not (checks["classifier_margin"] > 0 and all(checks["terminal_pass"])):
            cert["choices"] = {"direct": None, "ready": None}
        cases.append({"d": d, "h": h, "low": base["low"], "high": base["high"],
                      "old_band": base["band"], "old_center": base.get("selected"),
                      "old_entry_bound": base.get("certified_calls"),
                      "checks": checks, **cert})
    out.mkdir(parents=True, exist_ok=False)
    write(out/"plan.json", {"grid": {"d": D, "h": H}, "ready_box": READY, "g1": TARGET,
                            "seed": SEED, "draws_per_class": DRAWS, "budget": BUDGET,
                            "entry_cap": CAP, "update_error_bound_per_coordinate": .0001,
                            "candidate_fixed_variances": V,
                            "selection": "shortest qualifying prefix then lowest variance",
                            "event_tolerance": [E1, E2], "models": cases})


def evaluate(out):
    spec = json.loads((out/"plan.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(spec["seed"])
    n = spec["draws_per_class"]
    rows = []
    for case in spec["models"]:
        model = model_for(case)
        truth = model.mu[:, None, :] + model.tau*rng.standard_normal((3, n, 2))
        b = (truth@model.A + ALPHA**2*rng.standard_normal((3, n, 2)))@model.A
        event = np.all(np.abs(b-model.mu[:, None, :]@model.B) <= [E1, E2], axis=-1)
        rec = {}
        schedules = {"terminal": ([], False)}
        if case["low"] is not None and case["high"] is not None:
            schedules["restricted"] = ([case["low"], case["high"]]*12+[case["low"]], False)
        if case["old_band"]:
            schedules["old_band"] = ([case["old_center"]]*case["old_entry_bound"], True)
        for key, cfg in case["choices"].items():
            if cfg:
                schedules[key] = ([cfg["v"]]*cfg["prefix"], True)
        for key, (prefix, gated) in schedules.items():
            r, _ = run(model, truth, b, np.array(rectangles(case["d"], float)), prefix, gated)
            cfg = case["choices"].get(key)
            if cfg:
                first = np.array(r["first_entry"])[1, event[1]]
                assert np.all((0 <= first) & (first <= cfg["entry_bound"]))
            rec[key] = {"at_switch": r["at_switch"], "first_entry": r["first_entry"],
                        "prefix_calls": len(prefix), "final": r["budgets"][BUDGET]}
        rows.append({"d": case["d"], "h": case["h"], "event_counts": event.sum(axis=1).tolist(),
                     "policies": rec})
    write(out/"results.json", rows)


def terminal_followup(out):
    """Diagnose the failed class-2 box separately from frozen validation."""
    IV.dps = 45
    spec = json.loads((out/"plan.json").read_text(encoding="utf-8"))
    records = []
    for case in spec["models"]:
        failures = [z for z, ok in enumerate(case["checks"]["terminal_pass"]) if not ok]
        if failures != [2]:
            continue
        d, h = N(case["d"]), N(case["h"])
        vend = N(.001)**2+N(.000001)**2
        old_corner = (N(.01), d-N(.27))
        escaped = step(old_corner, (N(-.001), d), vend, d, h)[1]
        assert escaped.b < old_corner[1].a
        lo, hi = d-N(.2), d+N(.2)
        dl = N(.0001)
        lower = step((N(-1.1), hi), (N(-.00125), d+N(.005)), vend, d, h, (-dl, dl))
        upper = step((N(.01), lo), (N(-.00075), d-N(.005)), vend, d, h, (dl, -dl))
        assert lower[0].a > N(-1.1) and upper[0].b < N(.01)
        assert upper[1].a > lo.b and lower[1].b < hi.a
        assert N(-1.1) < N(-.00125) < N(-.00075) < N(.01) and N(.005) < N(.2)
        # Minimum of the affine distance gaps on the new rectangle.
        gap0 = 2*d*lo-1-d*d-2*N(.01)
        gap1 = gap0+1-2*N(.01)
        assert gap0 > 0 and gap1 > 0
        rec = certify(case["d"], case["h"], case)
        assert rec["preimage_verified"] and all(rec["choices"].values())
        records.append({"d": case["d"], "h": case["h"],
                        "old_corner_true_terminal_t": str(escaped),
                        "new_half_height": .2, "new_gap_vs_0": str(gap0),
                        "new_gap_vs_1": str(gap1),
                        "new_one_step_image": describe(lower, upper), **rec})
    assert len(records) == 4
    write(out/"terminal_followup.json", records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", "evaluate", "terminal-followup"))
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    {"plan": plan, "evaluate": evaluate, "terminal-followup": terminal_followup}[args.phase](args.out)


if __name__ == "__main__":
    main()
