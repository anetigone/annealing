"""Frozen-model independent-noise and held-out-photo patch validation."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from terminal_ball import terminal_ball_check


POLICIES = ["terminal", "intermediate", "low_high",
            "intermediate_frozen_c", "intermediate_frozen_weights"]
TIMES = [25, 154, 1000, 5000]


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    old = np.load(root/"experiments/results/patch_mechanism/outputs.npz")
    original = json.loads((root/"experiments/results/patch_revision/patch.json").read_text())
    np.savez_compressed(out/"frozen_model.npz", mu=old["mu"], pi=old["pi"],
                        a=old["a"], tau2=original["tau2"], flower=old["truth"])
    plan = {
        "model": "Exported original china-trained eight-component common-isotropic GMM",
        "photos": ["flower", "set12_01", "set12_02", "set12_10"],
        "patches_per_photo": 256, "noise_seed_bases": [20261112, 20261113, 20261114],
        "noise_seeds": "base + 1000 * photo_index",
        "patch_seed": "20261115 + photo_index, frozen across noise repeats",
        "flower_patches": "Original saved 256 truth patches, independent observation noise",
        "new_patches": "Nonoverlapping grayscale 4x4 patches, uniform subset without replacement",
        "sigma_noise": .01, "lambda": .1, "gamma": .25,
        "terminal_sigma": .02, "intermediate_sigma": .1, "high_sigma": .3,
        "intermediate_calls": 25, "policies": POLICIES, "times": TIMES,
        "ball": "154-call three original outputs, center from terminal5000 + 10 Newton; "
                "radius=max(1.01*covering_distance,1e-9); try all 8 references",
        "certificate": "Report fixed rho=.45 and general rho<1 separately",
        "selection": "Frozen before independent outcomes; no schedule or model tuning",
        "scope": "Finite independent-noise repeats and three new test photographs, no training"
    }
    (out/"plan.json").write_text(json.dumps(plan, indent=2))
    print("Frozen plan and model written.")


def weights(x, mu, pi, v):
    logits = np.log(pi)-np.sum((x[:, None]-mu)**2, axis=2)/(2*v)
    w = np.exp(logits-logits.max(axis=1, keepdims=True))
    return w/w.sum(axis=1, keepdims=True)


def photo_truth(name, index, model, data_dir):
    if name == "flower":
        return model["flower"].copy()
    gray = np.asarray(Image.open(data_dir/(name+".png")).convert("L"), dtype=float)/255
    h, w = [n//4*4 for n in gray.shape]
    patches = gray[:h, :w].reshape(h//4, 4, w//4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)
    rng = np.random.default_rng(20261115+index)
    return patches[rng.choice(len(patches), 256, replace=False)]


def run_photo(out, name, index, data_dir):
    model = np.load(out/"frozen_model.npz")
    mu, pi, a, tau2 = [model[key] for key in ["mu", "pi", "a", "tau2"]]
    gram = a@a.T
    truth_one = photo_truth(name, index, model, data_dir)
    truth = np.tile(truth_one, (3, 1))
    seeds = [base+1000*index for base in [20261112, 20261113, 20261114]]
    noise = np.concatenate([np.random.default_rng(seed).standard_normal(truth_one.shape)
                            for seed in seeds])
    b = (truth@a+.01*noise)@a.T
    folder = out/name
    folder.mkdir(exist_ok=False)
    states = {}
    risk = {}
    terminal_v = float(tau2+.02**2)
    terminal_c = .1*.02**2/terminal_v
    for policy in POLICIES:
        x = b.copy()
        risk[policy] = {}
        for k in range(5000):
            sigma = .02
            if k < 25:
                if policy.startswith("intermediate"):
                    sigma = .1
                elif policy == "low_high":
                    sigma = .02 if k % 2 == 0 else .3
            v = tau2+sigma*sigma
            c = .1*sigma*sigma/v
            if policy == "intermediate_frozen_c":
                c = terminal_c
            weight_v = terminal_v if policy == "intermediate_frozen_weights" else v
            x -= .25*(x@a@a.T-b+c*(x-weights(x, mu, pi, weight_v)@mu))
            if k+1 in TIMES:
                assert np.isfinite(x).all()
                states[policy, k+1] = x.copy()
                errors = np.sum((x-truth)**2, axis=1).reshape(3, 256)
                risk[policy][str(k+1)] = {"mean": float(errors.mean()),
                                         "per_noise_seed": errors.mean(axis=1).tolist()}
    center = states["terminal", 5000].copy()
    for _ in range(10):
        w = weights(center, mu, pi, terminal_v)
        centered = mu[None]-(w@mu)[:, None]
        covariance = np.einsum("nk,nki,nkj->nij", w, centered, centered)
        hessian = gram+terminal_c*(np.eye(16)-covariance/terminal_v)
        field = center@gram-b+terminal_c*(center-w@mu)
        center -= np.linalg.solve(hessian, field[..., None])[..., 0]
    assert np.isfinite(center).all()
    rows = []
    for sample in range(len(truth)):
        points = np.array([states[policy, 154][sample] for policy in POLICIES[:3]])
        radius = max(1.01*np.linalg.norm(points-center[sample], axis=1).max(), 1e-9)
        checks = [terminal_ball_check(gram, b[sample], mu, pi, terminal_v, terminal_c,
                                     .25, center[sample], radius, j, points) for j in range(8)]
        best_index = int(np.argmin([row["rho_upper"] for row in checks]))
        best = checks[best_index]
        fixed_q = 1-.25*.55*terminal_c
        fixed_margin = (1-fixed_q)*radius-best["residual"]
        fixed = best["rho_pairwise"] < .45 and fixed_margin > 0 and best["starts_inside"]
        best.update({"sample": sample, "time": 154, "radius": float(radius),
                     "reference": best_index, "fixed_passed": bool(fixed),
                     "fixed_containment_margin": float(fixed_margin)})
        rows.append(best)
    distance_checks = {}
    for policy in POLICIES[1:3]:
        initial = np.sum((states[policy,154]-states["terminal",154])**2, axis=1)
        item = {}
        for n in [154, 1000, 5000]:
            current = np.sum((states[policy,n]-states["terminal",n])**2, axis=1)
            fixed = np.array([r["fixed_passed"] for r in rows])
            general = np.array([r["passed"] for r in rows])
            q_values = np.array([r["q_upper"] for r in rows])
            bound = q_values**(2*(n-154))*initial
            item[str(n)] = {
                "mean_output_squared_distance": float(current.mean()),
                "per_noise_seed": current.reshape(3,256).mean(axis=1).tolist(),
                "different_dominant_components": int(np.sum(
                    weights(states[policy,n],mu,pi,terminal_v).argmax(axis=1) !=
                    weights(states["terminal",n],mu,pi,terminal_v).argmax(axis=1))),
                "mean_squared_bound_on_general_pass": float(bound[general].mean()),
                "mean_squared_observed_on_general_pass": float(current[general].mean()),
                "fixed_rho_pass_count": int(fixed.sum()),
                "float_observed_exceeds_general_bound": int(np.sum(
                    current[general] > bound[general]*(1+1e-8)+1e-25))}
        distance_checks[policy] = item
    summary = {
        "photo": name, "noise_seeds": seeds, "patches": 256, "observations": len(truth),
        "risk": risk, "distances": distance_checks,
        "ball": {"general_passed": sum(r["passed"] for r in rows),
                 "fixed_passed": sum(r["fixed_passed"] for r in rows),
                 "max_rho": max(r["rho_upper"] for r in rows),
                 "failed_general_curvature": sum(r["rho_upper"] >= 1 for r in rows),
                 "failed_general_containment": sum(r["containment_margin"] < 0 for r in rows),
                 "failed_coverage": sum(not r["starts_inside"] for r in rows),
                 "failed_step": sum(r["step_margin"] < 0 for r in rows)},
        "scope": "No tuning. Float decay is an empirical check. Certificate uses exported starts."
    }
    (folder/"summary.json").write_text(json.dumps(summary, indent=2))
    (folder/"checks.json").write_text(json.dumps({"rows": rows}, indent=2))
    np.savez_compressed(folder/"candidates.npz", gram=gram, b=b, means=mu, pi=pi,
                        v=terminal_v, c=terminal_c, gamma=.25, centers=center,
                        states=np.array([states[policy,154] for policy in POLICIES[:3]]))
    np.savez_compressed(folder/"outputs.npz", truth=truth,
                        states=np.array([states[policy,n] for policy in POLICIES for n in TIMES]))
    print(json.dumps({"photo": name, "ball": summary["ball"],
                      "risks154": {p: risk[p]["154"]["mean"] for p in POLICIES[:3]}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--photo", choices=["flower", "set12_01", "set12_02", "set12_10"])
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.out)
    else:
        plan = json.loads((args.out/"plan.json").read_text())
        assert args.photo in plan["photos"] and args.data is not None
        run_photo(args.out, args.photo, plan["photos"].index(args.photo), args.data)


if __name__ == "__main__":
    main()
