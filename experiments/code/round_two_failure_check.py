"""Check which second-round failures come from sufficient bounds or the path."""
import argparse
import json
from pathlib import Path

import mpmath
import numpy as np

from round_two_solver_noise import initialize, model_for, rectangles
from scale_selection_followup import write


ROOT = Path(__file__).resolve().parents[1]/"results"
IV = mpmath.iv
N = lambda x: IV.mpf(str(x))


def step(x, b, v, d, h, disturbance=(0, 0)):
    s, t = x
    beta, gamma, tau = N(.001), N(.25), N(.000001)
    c = (1+beta**2)*(1-tau**2/v)
    r = N(1.29)*IV.exp((s-N(.5))/v)
    w = h*IV.exp((-1-d*d-2*s+2*d*t)/(2*v))
    denominator = 1+r+w
    return (s+gamma*(b[0]-beta*s+c*((r-w)/denominator-s))+disturbance[0],
            t+gamma*(b[1]-t+c*(d*w/denominator-t))+disturbance[1])


def channel(row, delta, extra_calls=0):
    # Order (s,-t): self derivatives are nonnegative and posterior C12 <= 0.
    # The same two observation corners bound b and the backprojected start.
    d, h = N(row["d"]), N(row["h"])
    choice = row["fallback"]
    v, m = N(choice["v"]), choice["calls"] + extra_calls
    terminal = N(.001)**2+N(.000001)**2
    b_minus, b_plus = (N(.00075), N(.005)), (N(.00125), N(-.005))
    low, high = b_minus, b_plus
    out = []
    for k in range(m+2):
        inside = (low[0].a > N(.53) and high[0].b < N(1.1)
                  and high[1].a > N(-.1) and low[1].b < N(.15))
        out.append({"call": k, "s": [float(low[0].a), float(high[0].b)],
                    "t": [float(high[1].a), float(low[1].b)], "inside": bool(inside),
                    "all_left": bool(high[0].b < N(.53))})
        if k == m+1:
            break
        current = v if k < m else terminal
        low = step(low, b_minus, current, d, h, (-delta, delta))
        high = step(high, b_plus, current, d, h, (delta, -delta))
    # The lower order corner is itself a realizable trajectory, not a relaxation.
    wrong_box = (low[0].a > N(-.1) and low[0].b < N(.4)
                 and low[1].a > N(-.1) and low[1].b < N(.15))
    if wrong_box:
        # Bound deviations from mu_0 using the largest odds in G_0,
        # with the class-1 data force and arbitrary allowed future errors.
        r = N(1.29)*IV.exp(-N(.1)/terminal)
        h_odds = h*IV.exp(-(1+d*d-N(.2)-N(.3)*d)/(2*terminal))
        assert b_minus[0]+r+h_odds+4*delta < (1+N(.001))*N(.1)
        assert b_minus[1]+d*h_odds+4*delta < 2*N(.1)
    return {"switch": out[m], "after_terminal": out[m+1],
            "all_prefix_left": all(x["all_left"] for x in out[:m+1]),
            "lower_corner_absorbed_in_g0": bool(wrong_box), "trajectory": out}


def high_barrier():
    d, h = N(2.75), N(.8)
    x, b = (N(.3), N(-.01)), (N(.00125), N(.005))
    q = N(.3)-(N(.001)*(1-N(.3))+N(.00025)+N(.0001)/N(.25))/N(.999)
    gap = (1+d*d+2*x[0]-2*d*x[1])/2
    h_min = (N(1.29)-q*(1+N(1.29)))/(1+q)
    old_cutoff = gap/IV.ln(h/h_min)

    def face_margin(lo, hi):
        r = N(1.29) if hi is None else N(1.29)*IV.exp(-N(.2)/hi)
        w = h*IV.exp(-gap/lo)
        return q-(r-w)/(1+r+w)

    # On the face (s=.3,t=-.01), pair the largest R and smallest H
    # over each interval. The same proof then covers the unbounded region.
    margins = [face_margin(N(k)/100, N(k+1)/100) for k in range(800, 900)]
    tail = face_margin(N(9), None)
    assert all(q > 0 for q in [*margins, tail])
    v = N(2.2)
    outward_at_old_example = step(x, b, v, d, h)[0]-x[0]
    assert outward_at_old_example > 0
    point = (N(.001), N(0))
    states = [point]
    for k in range(26):
        variance = N(.18 if k%2 == 0 else 2.2) if k < 25 else N(.000001000001)
        states.append(step(states[-1], point, variance, d, h))
    first_exit = next(k for k, u in enumerate(states) if u[0].a > N(.3))
    first_entry = next(k for k, u in enumerate(states)
                       if u[0].a > N(.53) and u[0].b < N(1.1)
                       and u[1].a > N(-.1) and u[1].b < N(.15))
    return {"old_scalar_cutoff_infimum": str(old_cutoff), "sharpened_high_cutoff": 8,
            "min_segment_margin": min(float(q.a) for q in margins), "tail_margin": float(tail.a),
            "face_drift_at_2_2": str(outward_at_old_example),
            "nominal_example_first_barrier_exit": first_exit,
            "nominal_example_first_g1_entry": first_entry}


def noise_check():
    folder = ROOT/"round_two_solver_noise"
    spec = json.loads((folder/"plan.json").read_text())
    original = json.loads((folder/"results.json").read_text())
    rng = np.random.default_rng(spec["seeds"]["noise_gain"])
    n = spec["draws_per_class"]
    u, eps = (rng.standard_normal((3, n, 2)) for _ in range(2))
    rec = []
    for row in (r for r in original if r["slice"] == "noise_gain"):
        model = model_for(row)
        truth = model.mu[:, None, :]+model.tau*u
        b = (truth@model.A+row["nu"]*eps)@model.A
        event = np.all(np.abs(b-model.mu[:, None, :]@model.B) <= spec["event_tolerance"], axis=-1)
        z = np.arange(3)[:, None]
        pred = np.where(b[..., 1] > 1, 2, np.where(b[..., 0] > row["alpha"]/2, 1, 0))
        correct_gate = pred == z
        assert event.sum(axis=1).tolist() == row["event_counts"]
        assert correct_gate.sum(axis=1).tolist() == row["classifier_correct"]
        gate_key = "preset" if row["nominal_full_on_e"] else "fallback"
        fixed = spec["nominal"]
        gated = {"v": fixed["selected"], "calls": fixed["certified_calls"]} if gate_key == "preset" else row["fallback"]
        record = {}
        for key, cfg, mask in ((gate_key, gated, pred == 1), ("shared", row["shared"], np.ones_like(event))):
            x = initialize(b, row["alpha"], 0)
            for _ in range(cfg["calls"]):
                x[mask] = model.step(x[mask], b[mask], cfg["v"])
                x[~mask] = model.step(x[~mask], b[~mask], model.vend)
            bounds = np.array(rectangles(2., float))
            at_switch = np.all((bounds[..., 0][:, None] <= x) & (x <= bounds[..., 1][:, None]), axis=-1)
            for _ in range(106-cfg["calls"]):
                x = model.step(x, b, model.vend)
            dist = np.sum((x[..., None, :]-model.mu)**2, axis=-1)
            mode = dist.argmin(axis=-1) == z
            near = np.sum((x-model.mu[:, None, :])**2, axis=-1) <= .1**2
            squared = np.sum((x-truth)**2, axis=-1)
            old = row["policies"][key]
            assert at_switch.sum(axis=1).tolist() == old["at_switch"]
            assert mode.sum(axis=1).tolist() == old["budgets"]["106"]["mode_counts"]
            assert near.sum(axis=1).tolist() == old["budgets"]["106"]["neighborhood_counts"]
            assert squared.mean(axis=1).tolist() == old["budgets"]["106"]["class_mse"]
            groups = {}
            for name, group in (("in_e", event), ("outside_e", ~event),
                                ("classifier_right", correct_gate), ("classifier_wrong", ~correct_gate)):
                groups[name] = {"count": int(group.sum()),
                                "switch_failures": int((~at_switch & group).sum()),
                                "mode_failures": int((~mode & group).sum()),
                                "neighborhood_failures": int((~near & group).sum()),
                                "contribution_to_risk": float(model.pi@(np.where(group, squared, 0).sum(axis=1)/n))}
            record[key] = groups
        rec.append({"alpha": row["alpha"], "nu": row["nu"], "outcomes": record})
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=45)
    args = parser.parse_args()
    IV.dps = args.digits
    plan = json.loads((ROOT/"round_two_geometry/plan.json").read_text())
    cases = []
    for row in plan["models"]:
        if row["checks"]["obstacle"] and not row["checks"]["channel"]:
            exact = channel(row, N(0))
            bounded = channel(row, N(.0001))
            assert exact["all_prefix_left"] and exact["after_terminal"]["inside"]
            case = {"d": row["d"], "h": row["h"], "frozen": row["fallback"],
                    "exact": exact, "bounded_update_error": bounded}
            if not bounded["after_terminal"]["inside"]:
                assert bounded["lower_corner_absorbed_in_g0"]
                case["one_extra_middle_call"] = channel(row, N(.0001), extra_calls=1)
            cases.append(case)
    assert len(cases) == 10
    results = {"digits": args.digits, "channels": cases, "high_barrier": high_barrier(),
               "noise_replay": noise_check()}
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/"results.json", results)


if __name__ == "__main__":
    main()
