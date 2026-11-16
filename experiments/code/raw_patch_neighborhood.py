"""Prove basin entry for the selected raw patch over an observation neighborhood."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

from mpmath import iv
import numpy as np

from centered_patch_model import structure_basis
from patch_terminal_interval import exact


def norm(vector):
    return iv.sqrt(sum(x**2 for x in vector))


class Flow:
    def __init__(self, means, pi, diagonal, tau2, b):
        self.means = means
        self.d = diagonal
        self.d_float = np.array([float(x.mid) for x in diagonal])
        self.tau2 = tau2
        self.b = b
        self.gamma = iv.mpf(".25")
        self.pairs = [(i, j) for i in range(len(pi)) for j in range(i)]
        self.dm = [[a-b for a, b in zip(means[i], means[j])] for i, j in self.pairs]
        self.squares = [[a**2 for a in row] for row in self.dm]
        self.squares_float = np.array([[float(x.mid) for x in row] for row in self.squares])
        self.distance = [norm(row) for row in self.dm]
        self.offset = [(sum(a**2 for a in means[j])-sum(a**2 for a in means[i]))/2
                       for i, j in self.pairs]
        self.log_prior_odds = [iv.ln(pi[i]/pi[j]) for i, j in self.pairs]

    @lru_cache(maxsize=None)
    def coefficients(self, stage):
        sigma = iv.mpf(stage)
        variance = self.tau2 + sigma**2
        return variance, iv.mpf(".1")*(1-self.tau2/variance)

    def evaluate(self, point, radius, stage):
        """Image of the center and covariance pair caps throughout the input ball."""
        v, c = self.coefficients(stage)
        count = len(self.means)
        odds = [[iv.mpf(0) for _ in range(count)] for _ in range(count)]
        lower_sums = [iv.mpf(1) for _ in range(count)]
        logistic = []
        for (i, j), dm, offset, prior, distance in zip(
                self.pairs, self.dm, self.offset, self.log_prior_odds, self.distance):
            center_odds = prior+(offset+sum(x*d for x, d in zip(point, dm)))/v
            odds[i][j], odds[j][i] = center_odds, -center_odds
            spread = radius*distance/v
            lo = (center_odds-spread).a
            hi = (center_odds+spread).b
            lower_sums[j] += iv.exp(lo).a
            lower_sums[i] += iv.exp(-hi).a
            nearest = lo if lo > 0 else (-hi if hi < 0 else iv.mpf(0))
            e = iv.exp(-nearest)
            logistic.append(e/(1+e)**2)
        cap = [(1/s).b for s in lower_sums]
        pair_upper = [min(bound.b, (cap[i]*cap[j]).b)
                      for (i, j), bound in zip(self.pairs, logistic)]
        # Use a dominant component as a reference to reduce cancellation
        # when enclosing a range of smoothing variances at the center.
        ref = min(range(count), key=lambda i: float(lower_sums[i].a))
        unnormalized = [iv.exp(odds[i][ref]) for i in range(count)]
        total = sum(unnormalized)
        w = [value/total for value in unnormalized]
        base = self.means[ref]
        image = [x+self.gamma*(b-d*x+c*(base[k]-x+sum(
            w[i]*(self.means[i][k]-base[k]) for i in range(count) if i != ref)))
                 for k, (x, b, d) in enumerate(zip(point, self.b, self.d))]
        return image, pair_upper

    @lru_cache(maxsize=8192)
    def normalized(self, stage, eta_grid):
        v, c = self.coefficients(stage)
        eta = iv.mpf(eta_grid)/100000
        assert all((d+c-eta).a > 0 for d in self.d)
        return [(c/v)*sum(s/(d+c-eta) for s, d in zip(squares, self.d))
                for squares in self.squares]

    def factor(self, stage, pair_upper):
        """Propose a curvature lower bound, then prove it with a scalar trace test."""
        v, c = self.coefficients(stage)
        c_float = float(c.a)
        diagonal = self.d_float+c_float
        trace_diagonal = ((np.array([float(x.b) for x in pair_upper])
                           @self.squares_float)*float((c/v).b))
        high = diagonal.min()
        low = high-trace_diagonal.sum()
        for _ in range(16):
            mid = (high+low)/2
            if np.sum(trace_diagonal/(diagonal-mid)) <= 1:
                low = mid
            else:
                high = mid
        eta_grid = int(np.floor(low*100000))-1
        assert (1-self.gamma*(max(self.d)+c)).a > 0
        assert sum(p*t for p, t in zip(pair_upper, self.normalized(stage, eta_grid))).b < 1
        return (1-self.gamma*iv.mpf(eta_grid)/100000).b

    def tube(self, stage, point, radius, delta):
        image, pair_upper = self.evaluate(point, radius, stage)
        center = [x.mid for x in image]
        factor = self.factor(stage, pair_upper)
        step_error = norm([a-b for a, b in zip(image, center)])
        return center, (factor*radius+step_error+self.gamma*delta).b

    def invariant(self, stage, center, radius, delta):
        image, pair_upper = self.evaluate(center, radius, stage)
        q = self.factor(stage, pair_upper)
        margin = (1-q)*radius-norm([x-y for x, y in zip(image, center)])-self.gamma*delta
        assert q < 1 and margin.a > 0, (stage, str(q), str(margin))
        return {"q_upper": str(q), "margin": str(margin)}


def distance(point, center):
    return norm([x-y for x, y in zip(point, center)])


def certify(root, mode="base"):
    raw = np.load(root/"raw_reference_detail/absolute_.02_states.npz")
    observed = np.load(root/"data.npz")
    fixed = np.load(root/"raw_certification/candidates.npz")["centers"]
    sample = 37
    transform = np.column_stack(structure_basis())
    q = [list(map(exact, row)) for row in transform]
    operator = [list(map(exact, row)) for row in raw["operator"]]
    assert np.array_equal(raw["operator"], observed["operator"])

    def project(row):
        return [sum(exact(x)*a for x, a in zip(row, axis)) for axis in zip(*q)]

    # Check A is symmetric and diagonalized by the exact dyadic orthogonal Q.
    assert np.array_equal(raw["operator"], raw["operator"].T)
    axes = list(zip(*q))
    gains = [iv.mpf(1), iv.mpf(".5"), iv.mpf(".25"), iv.mpf(0)]
    spectrum = []
    for j, axis in enumerate(axes):
        assert all(sum(x*y for x, y in zip(axis, a)) == int(j == i)
                   for i, a in enumerate(axes))
        image = [sum(a*x for a, x in zip(row, axis)) for row in operator]
        gain, = [g for g in gains if all(x == g*y for x, y in zip(image, axis))]
        spectrum.append(gain)
    y = project(observed["observation"][sample])
    b = [g*x for g, x in zip(spectrum, y)]
    assert distance(b, project(raw["b"][sample])).b < iv.mpf("1e-14")
    truth = project(raw["truth"][sample])
    balls = [project(row) for row in fixed]
    means = [project(row) for row in raw["means"]]
    pi = list(map(exact, raw["pi"]))
    assert sum(pi) == 1 and all(p > 0 for p in pi)
    flow = Flow(means, pi, [g**2 for g in spectrum], exact(raw["tau2"]), b)
    delta = iv.mpf("0.000002").b
    radius = iv.mpf("0.005").a
    terminal = (".02", ".02")
    low = (".02", ".020001")
    middle = (".0999999", ".1000001")
    substitute = (".0229999", ".0230001")
    high = (".3", ".3")
    terminal_checks = [flow.invariant(terminal, center, radius, delta) for center in balls]
    low_check = flow.invariant(low, balls[0], radius, delta)
    separation = distance(*balls)-2*radius
    low_error = (distance(balls[0], truth)-radius)**2
    good_error = (distance(balls[1], truth)+radius)**2
    assert (distance(balls[0], truth)-radius).a > 0
    assert separation.a > 0 and (low_error-good_error).a > 0
    records = []
    protocols = [
        ("any_narrow_low", low, 5000, low, 0),
        ("middle", middle, 25, terminal, 1),
        ("low_substitute", substitute, 500, terminal, 1),
        ("high_comparator", high, 25, terminal, 0),
    ]
    switch_low, switch_high = (".022", ".022"), ("1", "1")
    if mode == "switching":
        protocols = [("low_high_substitute", switch_low, 1000, terminal, 1)]
    elif mode == "holds":
        protocols = [("low_hold", switch_low, 1000, terminal, 0),
                     ("high_hold", switch_high, 1000, terminal, 0)]
    for name, stage, hold, tail, branch in protocols:
        state, uncertainty = b.copy(), delta
        max_radius = delta
        for k in range(1, 5001):
            active = stage if k <= hold else tail
            if mode == "switching" and k <= hold and (k-1) % 125 >= 120:
                active = switch_high
            state, uncertainty = flow.tube(active, state, uncertainty, delta)
            max_radius = max(uncertainty, max_radius)
            assert uncertainty < iv.mpf(".03"), (name, k, str(uncertainty))
            if (k >= hold or name == "any_narrow_low") and k % 25 == 0:
                margin = radius-distance(state, balls[branch])-uncertainty
                if margin.a > 0:
                    break
        assert margin.a > 0, (name, str(margin))
        records.append({
            "policy": name, "stage_sigma": stage,
            "stage_calls": None if name == "any_narrow_low" else hold,
            "tail_sigma": tail, "target_ball": branch, "entry_call": k,
            "max_tube_radius": str(max_radius), "entry_margin": str(margin),
        })
        if mode == "switching":
            records[-1]["stage_blocks"] = [
                {"sigma": switch_low, "calls": 120},
                {"sigma": switch_high, "calls": 5},
            ]
            records[-1]["stage_repeats"] = 8
        print(f"PASS {name} enters ball {branch} at {k}, margin {margin}", flush=True)
    return {
        "status": "PASS", "digits": iv.dps, "source_sample": sample,
        "backprojected_observation_radius": "0.000002", "ball_radius": "0.005",
        "lambda": ".1", "gamma": ".25", "terminal_sigma": ".02",
        "ball_separation": str(separation), "terminal_checks": terminal_checks,
        "narrow_low_invariance": low_check, "records": records,
        "bad_ball_squared_error_lower": str(low_error.a),
        "good_ball_squared_error_upper": str(good_error.b),
        "squared_error_gap_lower": str((low_error-good_error).a),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["base", "switching", "holds"], default="base")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    iv.dps = 60
    results = certify(args.data, args.mode)
    (args.out/"certificate.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
