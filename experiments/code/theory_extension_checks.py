"""Recheck threshold and attribution identities from the theory extension."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def coefficients(s, r, tau2, a, gaps, composition):
    p = .5 - s
    mean_a = composition @ a
    mean_gap = composition @ gaps
    mean_product = composition @ (a * gaps)
    cs = (1 - s) * r - s
    quadratic = mean_product - mean_a * mean_gap + tau2 * (mean_a + s)
    linear = ((1 + r) * mean_product + r * mean_gap
              - p * r * (1 + mean_a)
              - tau2 * (cs - (mean_a + s) * (1 + r)))
    constant = -p * r - tau2 * cs * (1 + r)
    return np.array([quadratic, linear, constant])


def prior_drift(z, s, r, tau2, h, a, gaps, composition):
    target = r * np.exp(-(.5 - s) * z)
    competitors = h * composition * np.exp(-gaps * z)
    mean = (target - competitors @ a) / (1 + target + competitors.sum())
    return (1 - tau2 * z) * (mean - s)


def convex_value(z, s, r, tau2, h, a, gaps, composition):
    u = prior_drift(0., s, r, tau2, h, a, gaps, composition)
    scale = 1 - tau2 * z
    denominator = (1 - s) * scale - u
    first = (s * scale + u) / denominator * np.exp((.5 - s) * z)
    others = (((a + s) * scale + u) / denominator
              * np.exp(-(gaps - (.5 - s)) * z))
    return first + h * (composition @ others)


def threshold_checks(rng):
    max_derivative_error = 0.
    relaxed_cases = 0
    threshold_comparisons = 0
    for _ in range(128):
        s = rng.uniform(.1, .45)
        r = s / (1 - s) * rng.uniform(1.1, 12)
        p = .5 - s
        a = rng.uniform(.1, 3, size=3)
        gaps = p + rng.uniform(.1, 8, size=3)
        composition = rng.dirichlet(np.ones(3))
        # The target-only terminal weight is below s.
        terminal = p / (2 * np.log(r * (1 - s) / s))
        tau2 = terminal * rng.uniform(.05, .98)
        cs = (1 - s) * r - s
        hmax = cs / (composition @ a + s)
        h = hmax * rng.uniform(.1, .95)
        polynomial = coefficients(s, r, tau2, a, gaps, composition)
        value = np.polyval(polynomial, h)
        # Complex differentiation evaluates the original Gaussian-odds formula.
        derivative = prior_drift(1e-25j, s, r, tau2, h,
                                 a, gaps, composition).imag / 1e-25
        expected = value / (1 + r + h)**2
        max_derivative_error = max(max_derivative_error, abs(derivative - expected))
        assert np.isclose(derivative, expected, rtol=1e-11, atol=1e-12)
        mean_a = composition @ a
        h0 = r / mean_a
        roots = np.roots(np.trim_zeros(polynomial, "f"))
        admissible = [x.real for x in roots if abs(x.imag) < 1e-10 and 0 < x.real < h0]
        assert len(admissible) == 1
        u = prior_drift(0., s, r, tau2, h, a, gaps, composition)
        boundary = min(1 / terminal, (1 - u / (1 - s)) / tau2)
        result = minimize_scalar(
            lambda z: convex_value(z, s, r, tau2, h, a, gaps, composition),
            bounds=(0, boundary * (1 - 1e-9)), method="bounded",
            options={"xatol": 1e-11})
        numerical_improvement = result.fun < r - 1e-10
        # Keep a clear sign margin for this finite numerical cross-check.
        assert abs(value) > 1e-8
        assert numerical_improvement == (value > 0)
        threshold_comparisons += 1
        relaxed_cases += int(cs / (1 + r) >= (1 - s) * (1 - tau2 / terminal))
    return {
        "cases": 128, "convex_minimum_comparisons": threshold_comparisons,
        "cases_failing_removed_denominator_condition": relaxed_cases,
        "max_derivative_identity_error": max_derivative_error,
        "scope": "Finite consistency checks; proofs establish the continuum claims.",
    }


def attribution_checks(rng):
    max_error = 0.
    for _ in range(64):
        means = rng.normal(size=(5, 4))
        weights = rng.dirichlet(np.ones(5))
        x = rng.normal(size=4)
        c0, c1 = rng.uniform(.01, 2, size=2)
        v0, v1 = rng.uniform(.1, 3, size=2)
        gamma = .25

        def mean(v):
            logits = np.log(weights) - np.sum((x - means)**2, axis=1) / (2 * v)
            w = np.exp(logits - logits.max())
            return w @ means / w.sum()

        m0, m1 = mean(v0), mean(v1)
        # Shared data terms cancel in all four update differences.
        def update(c, m):
            return x - gamma * c * (x - m)

        total = update(c1, m1) - update(c0, m0)
        coefficient = update(c1, m0) - update(c0, m0)
        smoothing = update(c0, m1) - update(c0, m0)
        interaction = gamma * (c1 - c0) * (m1 - m0)
        error = np.linalg.norm(total - coefficient - smoothing - interaction)
        max_error = max(max_error, error)
        assert error < 1e-12
        anchored = -gamma * (c1 - c0) * (x - m1) + gamma * c0 * (m1 - m0)
        assert np.linalg.norm(total - anchored) < 1e-12
    return {"cases": 64, "max_four_corner_error": max_error}


def explicit_examples():
    a = np.array([.1, 3.])
    gaps = np.array([10., 2.])
    composition = np.array([.5, .5])
    s, r, tau2 = .3, 1.29, 1e-4
    point = np.array([s, 3.])
    templates = np.array([[-.1, 3 + np.sqrt(28.93)], [-3, 3 + np.sqrt(2.2)]])
    actual = (np.sum((point - templates)**2, axis=1) - point @ point) / 2
    assert np.allclose(actual, gaps, atol=1e-13, rtol=0)
    polynomial = coefficients(s, r, tau2, a, gaps, composition)
    roots = np.sort(np.roots(polynomial))
    h0 = r / (composition @ a)
    hmax = ((1 - s) * r - s) / (composition @ a + s)
    assert 0 < roots[0] < hmax < h0 < roots[1]
    multiple_roots = {"coefficients": polynomial.tolist(), "positive_roots": roots.tolist(),
                      "hmax": hmax, "h0": h0, "geometric_gaps": actual.tolist()}
    # This single-competitor geometry violates the removed condition.
    s, r, tau2, terminal, h = .3, 10., .009, .01, .1
    a, gaps, composition = np.array([1.]), np.array([2.82]), np.array([1.])
    u = prior_drift(0., s, r, tau2, h, a, gaps, composition)
    boundary = (1 - u / (1 - s)) / tau2
    result = minimize_scalar(
        lambda z: convex_value(z, s, r, tau2, h, a, gaps, composition),
        bounds=(0, boundary * (1 - 1e-9)), method="bounded")
    # Minimize the convex certificate; direct drift need not be unimodal.
    improved = prior_drift(result.x, s, r, tau2, h, a, gaps, composition)
    assert improved > u
    lhs = ((1 - s) * r - s) / (1 + r)
    rhs = (1 - s) * (1 - tau2 / terminal)
    assert lhs > rhs
    relaxed = {"removed_condition_lhs": lhs, "removed_condition_rhs": rhs,
               "D_at_h": float(np.polyval(coefficients(s, r, tau2, a, gaps, composition), h)),
               "exhibited_inverse_variance": float(result.x),
               "endpoint_drift": float(u), "intermediate_drift": float(improved)}
    # Single-competitor polynomial reduction.
    for h in [.01, .1, .3]:
        full = np.polyval(coefficients(s, r, tau2, a, gaps, composition), h)
        cs = (1 - s) * r - s
        single = (-(.5 - s) * r
                  + h * (a[0] * gaps[0] + (1 + a[0]) * (gaps[0] - (.5 - s)) * r)
                  - tau2 * (cs - (a[0] + s) * h) * (1 + r + h))
        assert np.isclose(full, single, rtol=1e-13, atol=1e-13)
    return {"double_positive_root": multiple_roots, "relaxed_condition": relaxed,
            "single_competitor_reduction_checks": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    seed = 20261110
    rng = np.random.default_rng(seed)
    output = {"status": "PASS", "seed": seed,
              "threshold": threshold_checks(rng),
              "attribution": attribution_checks(rng),
              "examples": explicit_examples()}
    (args.out / "threshold_checks.json").write_text(json.dumps(output, indent=2) + "\n")
    print("PASS: 128 threshold cases, 64 attribution cases, and explicit examples.")


if __name__ == "__main__":
    main()
