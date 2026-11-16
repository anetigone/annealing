"""Certify a positive-width platform with arbitrary scale choices at each call."""
import argparse
import json
from pathlib import Path
from decimal import Decimal

from scale_selection_followup import write
from switch_ready_diagnostic import CAP, IV, N, READY, TARGET, describe, inside


ROOT = Path(__file__).resolve().parents[1]/"results"/"switch_ready_diagnostic"
HALF_WIDTH = ".01"
SUBDIVISIONS = 5


def band_segments(v, half_width, subdivisions):
    left, right = v-half_width, v+half_width
    points = [left+(right-left)*j/subdivisions for j in range(subdivisions+1)]
    return [IV.mpf([a.a, b.b]) for a, b in zip(points[:-1], points[1:])]


def mean_bounds(s, t, variance, d, h):
    r = N(1.29)*IV.exp((s-N(.5))/variance)
    w = h*IV.exp((-1-d*d-2*s+2*d*t)/(2*variance))
    # The horizontal posterior mean increases with r and decreases with w;
    # the vertical mean has opposite monotonicity.
    horizontal_lo = (r.a-w.b)/(1+r.a+w.b)
    horizontal_hi = (r.b-w.a)/(1+r.b+w.a)
    vertical_lo = d*w.a/(1+r.b+w.a)
    vertical_hi = d*w.b/(1+r.a+w.b)
    horizontal = IV.mpf([horizontal_lo.a, horizontal_hi.b])
    vertical = IV.mpf([vertical_lo.a, vertical_hi.b])
    return horizontal, vertical


def advance(lower, upper, b_minus, b_plus, segments, d, h, delta):
    gamma, alpha = N(.25), N(.001)
    def coord_ranges(point, b, v, c):
        s, t = point
        m_s, m_t = mean_bounds(s, t, v, d, h)
        return (s + gamma*(b[0]-alpha*s+c*(m_s-s)),
                t + gamma*(b[1]-t+c*(m_t-t)))

    lo_s, lo_t, hi_s, hi_t = [], [], [], []
    for v, c in segments:
        # Use state monotonicity to evaluate only the two order corners.
        # Interval scales enclose all v in each segment, with no assumption
        # that successive calls choose the same v or are independent.
        s, t = coord_ranges(lower, b_minus, v, c)
        lo_s.append((s-delta).a)
        lo_t.append((t+delta).b)
        s, t = coord_ranges(upper, b_plus, v, c)
        hi_s.append((s+delta).b)
        hi_t.append((t-delta).a)
    return (min(lo_s), max(lo_t)), (max(hi_s), min(hi_t))


def certify(case, centers, subdivisions, half_width):
    d, h = N(case["d"]), N(case["h"])
    b_minus = (N(.00075), N(.005))
    b_plus = (N(.00125), N(-.005))
    delta, tau, lam = N(.0001), N(.000001), 1+N(.001)**2
    choices = {"direct": None, "ready": None}
    passing = {k: [] for k in choices}
    for center in centers:
        v, width = N(center), N(half_width)
        left, right = v-width, v+width
        if not N(case["low"]) < left < right < N(case["high"]):
            continue
        vs = band_segments(v, width, subdivisions)
        segments = [(w, lam*(1-tau**2/w)) for w in vs]
        lower, upper = b_minus, b_plus
        passed = set()
        for k in range(1, CAP+1):
            lower, upper = advance(lower, upper, b_minus, b_plus, segments, d, h, delta)
            for mode, box, offset in (("direct", TARGET, 0), ("ready", READY, 1)):
                if mode in passed or (mode == "ready" and (not case["preimage_verified"] or k == CAP)):
                    continue
                if inside(lower, upper, box):
                    passed.add(mode)
                    passing[mode].append(center)
                    cv, hw = Decimal(str(center)), Decimal(half_width)
                    current = {"v": center, "band": [str(cv-hw), str(cv+hw)],
                               "prefix": k, "entry_bound": k+offset,
                               "switch_enclosure": describe(lower, upper)}
                    previous = choices[mode]
                    if previous is None or (k, center) < (previous["prefix"], previous["v"]):
                        choices[mode] = current
            if len(passed) == 2:
                break
    return {"choices": choices, "passing_centers": passing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subdivisions", type=int, default=SUBDIVISIONS)
    parser.add_argument("--half-width", default=HALF_WIDTH)
    args = parser.parse_args()
    assert args.subdivisions > 0 and Decimal(args.half_width) > 0
    IV.dps = 45
    source = json.loads((ROOT/"plan.json").read_text())
    repaired = {(r["d"], r["h"]) for r in json.loads((ROOT/"terminal_followup.json").read_text())}
    rows = []
    for case in source["models"]:
        key = case["d"], case["h"]
        terminal = (all(case["checks"]["terminal_pass"]) and case["checks"]["classifier_margin"] > 0)
        assert case["preimage_verified"] and case["checks"]["obstacle"]
        rec = certify(case, source["candidate_fixed_variances"], args.subdivisions, args.half_width)
        rows.append({"d": key[0], "h": key[1], "old_band": case["old_band"],
                     "old_entry_bound": case["old_entry_bound"],
                     "fixed_choices": case["choices"], "original_terminal_pass": terminal,
                     "terminal_followup_pass": key in repaired, **rec})
        print(key, *(None if a is None else a["entry_bound"] for a in rec["choices"].values()), flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/"results.json", {"half_width": args.half_width, "subdivisions": args.subdivisions,
                                    "digits": 45, "entry_cap": CAP,
                                    "update_error_bound_per_coordinate": .0001, "models": rows})


if __name__ == "__main__":
    main()
