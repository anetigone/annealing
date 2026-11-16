"""Apply common-ball sufficient conditions to the fixed photo-patch experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from patch_diagnostic import build_case
from terminal_ball import terminal_ball_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    _, _, truth, mu, pi, tau2, a, y, b = build_case()
    gram = a@a.T
    v = tau2+.02**2
    c = .1*.02**2/v
    gamma = .25
    def posterior(x):
        log = np.log(pi)-np.sum((x[:, None]-mu)**2, axis=2)/(2*v)
        w = np.exp(log-log.max(axis=1, keepdims=True))
        return w/w.sum(axis=1, keepdims=True)
    def field(x):
        return x@gram-b+c*(x-posterior(x)@mu)
    checkpoints = [154, 1000, 5000]
    states = {}
    for policy in ["terminal", "intermediate", "low_high"]:
        x = b.copy()
        for k in range(5000):
            sigma = .02
            if k < 25:
                if policy == "intermediate":
                    sigma = .1
                elif policy == "low_high":
                    sigma = .02 if k % 2 == 0 else .3
            vk = tau2+sigma**2
            log = np.log(pi)-np.sum((x[:, None]-mu)**2, axis=2)/(2*vk)
            w = np.exp(log-log.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            x -= gamma*(x@gram-b+.1*sigma**2/vk*(x-w@mu))
            if k+1 in checkpoints:
                states[policy, k+1] = x.copy()
    # Newton iterates are only candidates. Residual and ball inequalities
    # subsequently determine whether each candidate yields a valid region.
    center = states["terminal", 5000].copy()
    for _ in range(10):
        w = posterior(center)
        centered = mu[None]-(w@mu)[:, None]
        cov = np.einsum("nk,nki,nkj->nij", w, centered, centered)
        hessian = gram+c*(np.eye(16)-cov/v)
        center -= np.linalg.solve(hessian, field(center)[..., None])[..., 0]
    centered_mu = mu-mu[0]
    _, singular, right = np.linalg.svd(centered_mu[1:], full_matrices=False)
    span = right[singular > 1e-12].T
    # A subspace restricted Rayleigh bound is different from pairwise secant gains.
    span_gain = np.linalg.eigvalsh(span.T@gram@span)
    inverse = np.linalg.inv(gram+c*np.eye(16))
    schur = np.linalg.inv(span.T@inverse@span)
    rows = []
    for k in checkpoints:
        for i in range(len(b)):
            points = np.stack([states[policy, k][i]
                               for policy in ["terminal", "intermediate", "low_high"]])
            distance = np.linalg.norm(points-center[i], axis=1).max()
            radius = max(1.01*distance, 1e-9)
            # All reference templates are tried. Selection uses only the model,
            # observation and candidate states, never the truth patch.
            checks = [terminal_ball_check(
                gram, b[i], mu, pi, v, c, gamma, center[i], radius, j, points)
                      for j in range(len(mu))]
            best = min(checks, key=lambda row: row["rho_upper"])
            best.update({"sample": i, "time": k, "radius": float(radius),
                         "reference": int(np.argmin([r["rho_upper"] for r in checks]))})
            rows.append(best)
    summary = {}
    for k in checkpoints:
        group = [r for r in rows if r["time"] == k]
        passed = [r for r in group if r["passed"]]
        summary[str(k)] = {
            "passed": len(passed), "total": len(group),
            "max_rho": max(r["rho_upper"] for r in group),
            "max_q_passed": max((r["q_upper"] for r in passed), default=None),
            "min_containment_margin_passed": min(
                (r["containment_margin"] for r in passed), default=None),
            "failed_curvature": sum(r["rho_upper"] >= 1 for r in group),
            "failed_containment": sum(r["containment_margin"] < 0 for r in group)}
    (args.out/"checks.json").write_text(json.dumps({
        "scope": "Binary64 sufficient-condition check, no outward rounding",
        "selection": "Three fixed times, Newton center, radius 1.01 times covering distance; all 8 references",
        "template_span_dimension": span.shape[1],
        "span_sensing_eigenvalues": span_gain.tolist(),
        "schur_eigenvalues": np.linalg.eigvalsh(schur).tolist(),
        "summary": summary, "rows": rows}, indent=2))
    np.savez_compressed(args.out/"candidates.npz", gram=gram, b=b, means=mu,
                        pi=pi, v=v, c=c, gamma=gamma, centers=center,
                        states=np.array([states[policy, k] for k in checkpoints
                                         for policy in ["terminal", "intermediate", "low_high"]]))


if __name__ == "__main__":
    main()
