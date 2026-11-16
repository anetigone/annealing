"""Independent-noise microvalidation of one selected raw-GMM development patch."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from centered_patch_development import weights


POLICIES = ["terminal", "intermediate", "alternating", "high1", "infinity_limit"]
BUDGETS = [154, 1000, 5000]


def update(x, b, means, pi, gram, tau2, sigma):
    variance = tau2 + sigma**2
    if np.isinf(sigma):
        posterior_mean = np.broadcast_to(pi @ means, x.shape)
        coefficient = .1
    else:
        posterior_mean = weights(x, means, pi, variance) @ means
        coefficient = .1 * sigma**2 / variance
    return x - .25 * (x @ gram - b + coefficient * (x - posterior_mean))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Fresh output directory")
    args = parser.parse_args()
    folder = args.out or args.data / "raw_probe"
    folder.mkdir(exist_ok=False)
    plan = {
        "role": "Post-hoc development microvalidation of selected patch 37",
        "selection": "Selected by persistent endpoint distance above 1e-3, not error",
        "source": "raw_reference_detail/absolute_.02_states.npz",
        "sample": 37, "new_noise_seeds": [20261123, 20261124],
        "observations_per_seed": 128, "noise_std": .01,
        "lambda": .1, "gamma": .25, "terminal_sigma": .02,
        "intermediate_calls": 25, "policies": POLICIES, "budgets": BUDGETS,
        "intermediate_sigma": .1, "alternating_sigmas": [.02, .3],
        "high_sigma": 1., "infinite_scale": "Analytic limit map, not a finite-scale implementation",
        "scale_map_sigmas": [.02, .035, .05, .075, .1, .15, .3, .6, 1., "infinity_limit"],
        "scale_map_holds": [5, 25, 50], "scale_map_budget": 5000,
        "scope": "Same one truth patch, new independent observation noise. "
                 "Not a population image result, not independent of patch/model selection. "
                 "New noise does not remove post-selection bias."
    }
    (folder / "plan.json").write_text(json.dumps(plan, indent=2))
    source = np.load(args.data / plan["source"])
    sample = plan["sample"]
    means, pi, operator = [source[key] for key in ["means", "pi", "operator"]]
    tau2 = float(source["tau2"])
    gram = operator @ operator.T
    truth_one = source["truth"][sample]
    truth = np.repeat(truth_one[None], 256, axis=0)
    noise = np.concatenate([np.random.default_rng(seed).standard_normal((128, 16))
                            for seed in plan["new_noise_seeds"]])
    observation = truth @ operator + .01 * noise
    b = observation @ operator.T
    outputs = []
    errors = {}
    metrics = {}
    for policy in POLICIES:
        x = b.copy()
        metrics[policy] = {}
        errors[policy] = {}
        for k in range(5000):
            sigma = .02
            if k < 25:
                if policy == "intermediate":
                    sigma = .1
                elif policy == "alternating":
                    sigma = .02 if k % 2 == 0 else .3
                elif policy == "high1":
                    sigma = 1.
                elif policy == "infinity_limit":
                    sigma = np.inf
            x = update(x, b, means, pi, gram, tau2, sigma)
            if k + 1 in BUDGETS:
                assert np.isfinite(x).all()
                outputs.append(x.copy())
                error = np.sum((x - truth)**2, axis=1).reshape(2, 128)
                errors[policy][str(k + 1)] = error
                metrics[policy][str(k + 1)] = {
                    "risk": float(error.mean()),
                    "per_seed_risk": error.mean(axis=1).tolist()
                }
        print(f"Completed microvalidation policy {policy}", flush=True)
    for policy in POLICIES:
        for budget in BUDGETS:
            delta = errors[policy][str(budget)] - errors["terminal"][str(budget)]
            metrics[policy][str(budget)].update({
                "paired_difference": float(delta.mean()),
                "per_seed_paired_difference": delta.mean(axis=1).tolist(),
                "per_seed_paired_standard_error": (
                    delta.std(axis=1, ddof=1) / np.sqrt(128)).tolist()
            })
    separation = {}
    baseline = outputs[2]
    for index, policy in enumerate(POLICIES):
        distant = np.linalg.norm(outputs[index * 3 + 2] - baseline, axis=1) > 1e-3
        separation[policy] = distant.reshape(2, 128).sum(axis=1).tolist()
    np.savez_compressed(folder / "noise_outputs.npz", means=means, pi=pi, tau2=tau2,
                        operator=operator, truth=truth, observation=observation, b=b,
                        noise=noise, states=np.array(outputs),
                        budgets=np.array(BUDGETS), policies=np.array(POLICIES))
    (folder / "summary.json").write_text(json.dumps({
        "selection": plan["selection"], "sample": sample,
        "new_noise_seeds": plan["new_noise_seeds"], "metrics": metrics,
        "separated_endpoints5000_per_seed": separation,
        "separation_distance_threshold": 1e-3,
        "scope": plan["scope"]
    }, indent=2))

    sigmas = [.02, .035, .05, .075, .1, .15, .3, .6, 1., np.inf]
    holds = [5, 25, 50]
    original_b = source["b"][sample:sample + 1]
    original_baseline = source["states"][3, sample]
    original_middle = source["states"][7, sample]
    map_rows = []
    map_values = np.zeros((len(holds), len(sigmas)))
    map_outputs = []
    for h, hold in enumerate(holds):
        for j, sigma in enumerate(sigmas):
            x = original_b.copy()
            for k in range(5000):
                x = update(x, original_b, means, pi, gram, tau2,
                           sigma if k < hold else .02)
            distance = float(np.linalg.norm(x[0] - original_baseline))
            map_values[h, j] = distance
            map_outputs.append(x[0].copy())
            map_rows.append({
                "hold": hold, "sigma": "infinity_limit" if np.isinf(sigma) else sigma,
                "distance_to_terminal": distance,
                "distance_to_original_middle": float(np.linalg.norm(x[0] - original_middle)),
                "squared_euclidean_error": float(np.sum((x[0] - truth_one)**2))
            })
    (folder / "scale_map.json").write_text(json.dumps({
        "selection": plan["selection"], "observation": "Original saved sample 37",
        "rows": map_rows,
        "scope": "Small discrete exploration, not a continuous-scale or arbitrary-controller certificate"
    }, indent=2))
    np.savez_compressed(folder / "scale_outputs.npz", states=np.array(map_outputs),
                        original_b=original_b, sigmas=np.array(sigmas), holds=np.array(holds))
    fig, ax = plt.subplots(figsize=(9.5, 2.8), layout="constrained")
    picture = ax.imshow(map_values, cmap="viridis", aspect="auto", vmin=0)
    ax.set_xticks(range(len(sigmas)),
                  [".02", ".035", ".05", ".075", ".1", ".15", ".3", ".6", "1", "infinity"])
    ax.set_yticks(range(len(holds)), holds)
    ax.set(xlabel="Intermediate sigma", ylabel="Holding calls",
           title="Selected development patch 37, distance from terminal branch at 5000 calls")
    fig.colorbar(picture, ax=ax, label="Endpoint distance")
    fig.savefig(folder / "scale_map.png", dpi=160)
    fig.savefig(folder / "scale_map.svg")
    plt.close(fig)
    print(json.dumps({p: metrics[p]["5000"] for p in POLICIES}), flush=True)


if __name__ == "__main__":
    main()
