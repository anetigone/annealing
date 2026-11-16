"""Compare frozen raw-patch schedules on two new independent noise batches."""
import argparse
import json
from pathlib import Path

import numpy as np

from centered_raw_probe import update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    sample, seeds, count, budget = 37, [20261125, 20261126], 256, 5000
    schedules = {
        "terminal": [],
        "middle": [(.1, 25)],
        "low_substitute": [(.023, 500)],
        "low_high_substitute": [(.022, 120), (1., 5)] * 8,
    }
    plan = {
        "sample": sample, "noise_seeds": seeds, "per_seed": count,
        "noise_std": .01, "budget": budget, "prefix_blocks_sigma_calls": schedules,
        "terminal_sigma": .02, "lambda": .1, "gamma": .25,
        "initialization": "Backprojection of each observation",
        "selection": "Schedules selected on the original development observation by branch "
                     "distance; no schedule selection using these new noises.",
        "scope": "Conditional on a selected fixed truth patch and fitted GMM, not across patches.",
    }
    (args.out/"plan.json").write_text(json.dumps(plan, indent=2))
    raw = np.load(args.data/"raw_reference_detail/absolute_.02_states.npz")
    a, means, pi = (raw[key] for key in ["operator", "means", "pi"])
    tau2 = float(raw["tau2"])
    gram = a @ a.T
    truth = raw["truth"][sample]
    noise = .01*np.concatenate([np.random.default_rng(seed).standard_normal((count, 16))
                                for seed in seeds])
    y = truth @ a + noise
    b = y @ a.T
    outputs = {}
    for label, blocks in schedules.items():
        scales = [s for s, n in blocks for _ in range(n)]
        scales += [.02]*(budget-len(scales))
        assert len(scales) == budget
        x = b.copy()
        for sigma in scales:
            x = update(x, b, means, pi, gram, tau2, sigma)
        assert x.shape == b.shape and np.isfinite(x).all()
        outputs[label] = x
    errors = {label: np.sum((x-truth)**2, axis=1).reshape(len(seeds), count)
              for label, x in outputs.items()}

    def summarize(values):
        return {"mean": float(values.mean()), "seed_means": values.mean(axis=1).tolist(),
                "standard_error": float(values.std(ddof=1)/np.sqrt(values.size))}

    summary = {label: {"risk": summarize(e), "difference_vs_terminal": summarize(e-errors["terminal"]),
                       "difference_vs_middle": summarize(e-errors["middle"]),
                       "distant_from_terminal": (np.linalg.norm(
                           outputs[label]-outputs["terminal"], axis=1) > 1e-3
                       ).reshape(len(seeds), count).sum(axis=1).tolist()}
               for label, e in errors.items()}
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(args.out/"outputs.npz", truth=truth, noise=noise, observation=y, b=b,
                        **outputs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
