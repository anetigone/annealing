"""Diagnose geometry, terminal dynamics and error sources of the fixed patch case."""
import argparse
import json
from pathlib import Path
import numpy as np
from patch_diagnostic import build_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    photos, train, truth, mu, pi, tau2, a, y, b = build_case()
    beta, basis = np.linalg.eigh(a@a.T)
    null = basis[:, beta < 1e-12]
    observed = basis[:, beta >= 1e-12]
    lam, gamma = .1, .25
    def weights(x, sigma):
        v = tau2+sigma*sigma
        logits = np.log(pi)-np.sum((x[:, None]-mu)**2, axis=2)/(2*v)
        w = np.exp(logits-logits.max(axis=1, keepdims=True))
        return w/w.sum(axis=1, keepdims=True)
    def field(x, sigma):
        c = lam*sigma*sigma/(tau2+sigma*sigma)
        return x@a@a.T-b+c*(x-weights(x, sigma)@mu)
    def metrics(x):
        error = x-truth
        return {"risk": float(np.mean(np.sum(error**2, axis=1))),
                "null_error": float(np.mean(np.sum((error@null)**2, axis=1))),
                "observed_error": float(np.mean(np.sum((error@observed)**2, axis=1))),
                "gradient_rms": float(np.sqrt(np.mean(np.sum(field(x, .02)**2, axis=1))))}
    snapshots = {}
    outputs = {}
    curvature = {}
    for policy in ["terminal", "intermediate", "low_high"]:
        x = b.copy()
        snapshots[policy] = {}
        hessian_mins = []
        responsibilities = []
        for k in range(5000):
            sigma = .02
            if k < 25:
                if policy == "intermediate":
                    sigma = .1
                elif policy == "low_high":
                    sigma = .02 if k % 2 == 0 else .3
            x -= gamma*field(x, sigma)
            if k+1 in [25, 154, 1000, 5000]:
                snapshots[policy][str(k+1)] = metrics(x)
                outputs[policy, k+1] = x.copy()
                w = weights(x, .02)
                mean = w@mu
                centered = mu[None]-mean[:, None]
                cov = np.einsum("nk,nki,nkj->nij", w, centered, centered)
                c = lam*.02**2/(tau2+.02**2)
                hess = a@a.T+c*(np.eye(16)-cov/(tau2+.02**2))
                hessian_mins.extend(np.linalg.eigvalsh(hess)[:, 0].tolist())
                responsibilities.append(float(w.max(axis=1).mean()))
        curvature[policy] = {"min_sampled_terminal_hessian_eigenvalue": min(hessian_mins),
                             "mean_max_responsibilities": responsibilities}
    differences = {}
    for n in [25, 154, 1000, 5000]:
        ref = outputs["terminal", n]
        differences[str(n)] = {}
        for policy in ["intermediate", "low_high"]:
            x = outputs[policy, n]
            paired = np.sum((x-truth)**2, axis=1)-np.sum((ref-truth)**2, axis=1)
            differences[str(n)][policy] = {
                "mean_squared_output_difference": float(np.mean(np.sum((x-ref)**2, axis=1))),
                "different_dominant_prior_components": int(np.sum(
                    weights(x, .02).argmax(axis=1) != weights(ref, .02).argmax(axis=1))),
                "paired_risk_difference": float(paired.mean()),
                "paired_standard_error": float(paired.std(ddof=1)/np.sqrt(len(paired)))}
    secants = np.array([mu[i]-mu[j] for i in range(8) for j in range(i)])
    gains = np.sum((secants@a)**2, axis=1)/np.sum(secants**2, axis=1)
    # Exact fitted-model Gaussian-mixture conditional mean for the same observations.
    cy = tau2*a.T@a + .01**2*np.eye(16)
    inverse = np.linalg.inv(cy)
    residual = y[:, None]-mu@a
    logits = np.log(pi)-.5*np.einsum("nki,ij,nkj->nk", residual, inverse, residual)
    w = np.exp(logits-logits.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    component_means = mu[None]+tau2*residual@inverse@a.T
    bayes = np.sum(w[:, :, None]*component_means, axis=1)
    # Null-direction estimate from the prior means; compare what is predictable.
    center = pi@mu
    input_stats = {
        "truth_null_energy": float(np.mean(np.sum((truth@null)**2, axis=1))),
        "prior_mean_null_energy": float(np.sum((center@null)**2)),
        "true_nearest_mean_residual_per_coordinate": float(np.min(
            np.sum((truth[:, None]-mu)**2, axis=2), axis=1).mean()/16),
        "train_residual_per_coordinate": float(tau2)}
    output = {"seed": 20261110, "selection": "Same fixed case; no parameter search",
              "data_gram_eigenvalues": beta.tolist(),
              "pairwise_template_sensing_gain": {
                  "min": float(gains.min()), "median": float(np.median(gains)),
                  "max": float(gains.max())},
              "scales": {str(s): {"v": float(tau2+s*s), "c": float(lam*s*s/(tau2+s*s))}
                         for s in [.02, .1, .3]},
              "terminal_null_contraction_if_single_component": float(1-gamma*lam*.02**2/(tau2+.02**2)),
              "snapshots": snapshots, "differences": differences,
              "sampled_curvature": curvature, "input_statistics": input_stats,
              "fitted_model_posterior_mean": metrics(bayes)}
    (args.out/"mechanism.json").write_text(json.dumps(output, indent=2))
    np.savez_compressed(args.out/"outputs.npz", truth=truth, mu=mu, pi=pi, a=a,
                        bayes=bayes, terminal=outputs["terminal",154],
                        intermediate=outputs["intermediate",154],
                        low_high=outputs["low_high",154],
                        terminal_long=outputs["terminal",5000])


if __name__ == "__main__":
    main()
