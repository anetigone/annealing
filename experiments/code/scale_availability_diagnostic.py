"""Small paired diagnostic for the 0/25/0 preset and restricted comparators."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "red_geometry_computation_risk" / "code"))
from family import Family


def run_case(alpha, tau, nu, v_terminal, lam, n_terminal, n_per_class, rng,
             fine=False):
    model = Family(alpha)
    # Use only the common MMSE update from Family. Set the new model inputs
    # explicitly, with prior ratios independent of its balanced construction.
    model.tau = tau
    model.lam = lam
    model.pi = np.array([1.0, 1.29, 1.78]) / 4.07

    labels = np.repeat(np.arange(3), n_per_class)
    truth = model.mu[labels] + tau * rng.standard_normal((len(labels), 2))
    y = truth @ model.A + nu * rng.standard_normal(truth.shape)
    b = y @ model.A
    target = np.where(b[:, 1] > 1, 2, (b[:, 0] > alpha / 2).astype(int))
    budget = 25 + n_terminal

    def weighted(values):
        return float(sum(w * values[labels == z].mean() for z, w in enumerate(model.pi)))

    def metrics(x):
        assert x.shape == truth.shape and np.isfinite(x).all()
        squared_error = np.sum((x - truth)**2, axis=1)
        template_distance = np.sum((x[:, None] - model.mu)**2, axis=-1)
        return {
            "squared_euclidean_risk": weighted(squared_error),
            "mode_correct_rate": weighted(template_distance.argmin(axis=1) == labels),
            "template_neighborhood_success_rate": weighted(
                template_distance[np.arange(len(labels)), labels] < 0.1**2
            ),
        }

    x_preset = b.copy()
    x_restricted = b.copy()
    x_terminal = b.copy()
    use_band = target == 1
    for k in range(budget):
        if k < 25:
            x_preset[use_band] = model.step(x_preset[use_band], b[use_band], 0.70)
            x_preset[~use_band] = model.step(
                x_preset[~use_band], b[~use_band], v_terminal
            )
        else:
            x_preset = model.step(x_preset, b, v_terminal)
        v_restricted = (0.17 if k % 2 == 0 else 2.5) if k < 25 else v_terminal
        x_restricted = model.step(x_restricted, b, v_restricted)
        x_terminal = model.step(x_terminal, b, v_terminal)

    e = b - model.mu[labels] @ model.B
    good_event = (np.abs(e[:, 0]) <= alpha / 4) & (np.abs(e[:, 1]) <= 0.005)
    result = {
        "alpha": alpha,
        "tau": tau,
        "nu": nu,
        "v_terminal": v_terminal,
        "lambda": lam,
        "c_terminal": model.c(v_terminal),
        "n_terminal": n_terminal,
        "total_calls": budget,
        "wrong_observation_targets": int((target != labels).sum()),
        "outside_noise_event": int((~good_event).sum()),
        "preset": metrics(x_preset),
        "restricted_alternation": metrics(x_restricted),
        "terminal_only": metrics(x_terminal),
    }
    if fine:
        beta = np.diag(model.B)
        means = model.mu[labels]
        # The oracle is used only for evaluation of the preset's excess risk.
        oracle_mean = means + e / (beta + 1)
        result["oracle_risk_analytic"] = float(np.sum(tau**2 / (beta + 1)))
        result["preset_excess_risk"] = weighted(
            np.sum((x_preset - oracle_mean)**2, axis=1)
        )
        least_squares = b / beta
        x_ls_init = least_squares.copy()
        x_weak = b.copy()
        for _ in range(budget):
            x_ls_init = model.step(x_ls_init, b, v_terminal)
        model.lam = alpha / 4
        for _ in range(budget):
            x_weak = model.step(x_weak, b, v_terminal)
        result.update({
            "least_squares_reference": metrics(least_squares),
            "classifier_template_reference": metrics(model.mu[target]),
            "terminal_ls_initialization": metrics(x_ls_init),
            "terminal_weak_lambda": {"lambda": model.lam, **metrics(x_weak)},
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--fine", action="store_true", help="Include the fine-risk setting")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    seed = 20261109
    n_per_class = 512
    rng = np.random.default_rng(seed)

    alpha = 1e-3
    n_cal = math.ceil(math.log(12 / alpha**3) / math.log(4 / 3))
    alpha_ind, tau_ind, nu_ind = 1e-6, 1e-4, 1e-5
    s_ind = math.hypot(tau_ind, nu_ind)
    n_ind = math.ceil(math.log(1 / s_ind) / math.log(1 / 0.751))
    output = {
        "seed": seed,
        "n_per_class": n_per_class,
        "geometry": {"a": 1, "d": 2, "pi1_over_pi0": 1.29, "pi2_over_pi0": 1.78},
        "selection": "Two prespecified parameter settings, no search or tuning.",
        "comparison": "Shared draws, initializations, total budget and terminal variance.",
        "rows": [
            run_case(alpha, alpha**2, alpha**2, alpha**2 + alpha**4,
                     1 + alpha**2, n_cal, n_per_class, rng),
            run_case(alpha_ind, tau_ind, nu_ind, s_ind,
                     1.0, n_ind, n_per_class, rng),
        ],
    }
    if args.fine:
        alpha = 1e-5
        n_cal = math.ceil(math.log(12 / alpha**3) / math.log(4 / 3))
        output["selection"] = "Three prespecified settings, no search or tuning."
        output["rows"].append(run_case(
            alpha, alpha**2, alpha**2, alpha**2 + alpha**4, 1 + alpha**2,
            n_cal, n_per_class, rng, fine=True,
        ))
    (args.out / "diagnostic.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
