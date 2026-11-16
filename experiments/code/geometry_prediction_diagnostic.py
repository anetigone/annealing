"""Compare four distinct tests on a fixed distance/mass grid."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "red_geometry_computation_risk/code"))
from family import Family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    distances = np.linspace(.5, 3, 41)
    masses = np.geomspace(.001, 6, 41)
    d, h = np.meshgrid(distances, masses)
    r, alpha, delta, gamma = 1.29, .001, 1e-4, .25
    v_l = alpha**2 + alpha**4
    c_minus, c_plus = .999, 1.002
    v = np.geomspace(v_l, 100, 1200)
    # Same corner and exact class-1 noiseless observation for panels 1 and 2.
    odds1 = r * np.exp(-.2 / v)
    odds2 = h[..., None] * np.exp(-(.8 + d[..., None]**2 / 2 +
                                                  .01*d[..., None]) / v)
    mean = (odds1 - odds2) / (1 + odds1 + odds2)
    drift = alpha * .7 + (1 + alpha**2) * (1-alpha**4/v) * (mean-.3)
    infinity = alpha*.7 + (1+alpha**2)*((r-h)/(1+r+h)-.3)
    improved = drift.max(axis=-1) > np.maximum(drift[..., 0], infinity) + 1e-10
    broken = (drift + delta/gamma).max(axis=-1) > 0
    # Analytic full-channel sufficient tests for e=0, band [.69,.71].
    rc = r*np.exp(-.51/.69)
    hc = h*np.exp(-(1+d*d-.02-.1*d)/(2*.71))
    wc = hc/(1+rc+hc)
    kappa = (.25+3*wc)/.69
    re = r*np.exp(.03/.71)
    he = h*np.exp(-(1+d*d+1.06-.1*d)/(2*.71))
    surplus = (re-he)/(1+re+he)-.53
    zeta = 1-kappa
    qc = 1-gamma*c_minus*zeta
    # Evaluate comparison without dividing by zeta near zero.
    comparison = np.full(d.shape, -.01)
    for _ in range(25):
        comparison = qc*comparison + gamma*c_minus*(zeta*.53+surplus)-delta
    top = -.05 + np.maximum(c_minus*(d*wc-.05),
                             c_plus*(d*wc-.05))+delta/gamma
    channel = (kappa < 1) & (surplus > delta/(gamma*c_minus)) & (top <= 0)
    channel &= comparison >= .53
    recovered = np.zeros(d.shape, dtype=bool)
    entered = np.zeros(d.shape, dtype=bool)
    for i in range(len(masses)):
        for j in range(len(distances)):
            model = Family(alpha, d=float(d[i, j]))
            model.pi = np.array([1, r, h[i, j]])/(1+r+h[i, j])
            b = model.mu @ model.B
            x = b.copy()
            for _ in range(25):
                x[1:2] = model.step(x[1:2], b[1:2], .7)
                x[[0, 2]] = model.step(x[[0, 2]], b[[0, 2]], v_l)
            entered[i, j] = (.53 <= x[1, 0] <= 1.1 and -.01 <= x[1, 1] <= .05)
            for _ in range(81):
                x = model.step(x, b, v_l)
            recovered[i, j] = np.all(np.linalg.norm(x-model.mu, axis=1) < .1)
    assert np.all(entered[channel])
    panels = [improved, broken, channel, recovered]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), layout="constrained")
    titles = ["One-step endpoint improvement", "Barrier can be violated",
              "Analytic channel certified", "All-class recovery observed"]
    for ax, values, title in zip(axes, panels, titles):
        ax.pcolormesh(d, h, values, shading="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set(xlabel="Third-template height d", title=title)
    axes[0].set_ylabel("Prior ratio h")
    for ax in axes:
        ax.set_yscale("log")
    fig.savefig(args.out/"geometry_prediction.pdf", bbox_inches="tight")
    np.savez_compressed(args.out/"grid.npz", d=d, h=h, improved=improved,
                        barrier_broken=broken, channel=channel, recovered=recovered)
    (args.out/"summary.json").write_text(json.dumps({
        "grid": [41, 41], "v_samples": 1200, "observation": "Noiseless, all classes",
        "protocol": "0/25/0 at v=.70 then terminal, 106 calls",
        "counts": dict(zip(titles, [int(x.sum()) for x in panels])),
        "channel_and_recovery": int((channel & recovered).sum()),
        "scope": "Sampled grid diagnostics; negative channel mask means uncertified."
    }, indent=2))


if __name__ == "__main__":
    main()
