"""Small attribution controls for the fixed patch mechanism study."""
import argparse
import json
from pathlib import Path
import numpy as np
from patch_diagnostic import build_case


def plot_controls(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    calls = [25, 154, 1000, 5000]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    for ax, kind, title in zip(axes, ["photo", "matched_gmm"],
                               ["Held-out photo patches", "Matched GMM truths"]):
        for policy, label in [("terminal", "Terminal"),
                              ("intermediate", "Intermediate"),
                              ("low_high", "Low/high"),
                              ("intermediate_frozen_c", "Smoothing only"),
                              ("data_only", "Data only")]:
            ax.plot(calls, [result[kind][policy][str(n)]["risk"] for n in calls],
                    marker="o", markersize=4, label=label)
        ax.set(xscale="log", xlabel="Denoiser calls", ylabel="Squared Euclidean error",
               title=title)
        ax.spines[["right", "top"]].set_visible(False)
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(out/"budget_attribution.pdf")
    fig.savefig(out/"budget_attribution.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    _, _, truth, mu, pi, tau2, a, y, b = build_case()
    rng = np.random.default_rng(20261111)
    labels = rng.choice(8, size=256, p=pi)
    matched = mu[labels]+np.sqrt(tau2)*rng.standard_normal(truth.shape)
    matched_b = (matched@a+.01*rng.standard_normal(truth.shape))@a.T
    beta, basis = np.linalg.eigh(a@a.T)
    masks = [beta < 1e-12, np.isclose(beta, .0625),
             np.isclose(beta, .25), np.isclose(beta, 1)]
    c_l = .1*.02**2/(tau2+.02**2)
    def weights(x, sigma):
        v = tau2+sigma*sigma
        logits = np.log(pi)-np.sum((x[:, None]-mu)**2, axis=2)/(2*v)
        w = np.exp(logits-logits.max(axis=1, keepdims=True))
        return w/w.sum(axis=1, keepdims=True)
    def run(truth, b):
        output = {}
        final = {}
        for policy in ["terminal", "intermediate", "low_high",
                       "intermediate_frozen_c", "intermediate_frozen_weights", "data_only"]:
            x = b.copy()
            output[policy] = {}
            for k in range(5000):
                sigma = .02
                if k < 25:
                    if policy.startswith("intermediate"):
                        sigma = .1
                    elif policy == "low_high":
                        sigma = .02 if k % 2 == 0 else .3
                c = .1*sigma*sigma/(tau2+sigma*sigma)
                if policy == "intermediate_frozen_c":
                    c = c_l
                if policy == "data_only":
                    c = 0
                weight_sigma = .02 if policy == "intermediate_frozen_weights" else sigma
                x -= .25*(x@a@a.T-b+c*(x-weights(x, weight_sigma)@mu))
                if k+1 in [25, 154, 1000, 5000]:
                    error = (x-truth)@basis
                    output[policy][str(k+1)] = {
                        "risk": float(np.mean(np.sum(error**2, axis=1))),
                        "spectral_error": [float(np.mean(np.sum(error[:, mask]**2, axis=1)))
                                           for mask in masks]}
                if k+1 == 154:
                    final[policy] = x.copy()
            output[policy]["output_difference_from_terminal_at154"] = float(
                np.mean(np.sum((final[policy]-final["terminal"])**2, axis=1)))
        return output
    result = {"photo": run(truth, b), "matched_gmm": run(matched, matched_b),
              "matched_seed": 20261111,
              "spectral_order": ["null", ".0625", ".25", "1"],
              "scope": "Frozen coefficient and frozen weight controls alter the update for attribution only."}
    (args.out/"controls.json").write_text(json.dumps(result, indent=2))
    plot_controls(result, args.out)


if __name__ == "__main__":
    main()
