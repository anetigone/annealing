"""Plot the analytic three-template thresholds at the manuscript's nominal corner."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a, d, r = 1.0, 2.0, 1.29
    s, t, alpha = 0.3, -0.01, 1e-3
    tau = alpha**2
    p_gap = 0.5 - s
    q_nominal = (a*a + d*d + 2*a*s - 2*d*t) / 2
    delta, gamma = 1e-4, 0.25
    lam = 1 + alpha**2

    # Positive root of the quadratic in the competing prior mass.
    q = np.linspace(0.801, 4, 400)  # Q = .8 + d**2/2 + .01*d for d > 0
    c_s = (1-s)*r - s
    h_limit = c_s / (a+s)
    quad = tau**2 * (a+s)
    linear = a*q + (1+a)*(q-p_gap)*r - tau**2*(c_s - (a+s)*(1+r))
    constant = -p_gap*r - tau**2*c_s*(1+r)
    h_crit = -2*constant / (linear + np.sqrt(linear**2 - 4*quad*constant))
    assert np.isfinite(h_crit).all() and (h_crit > 0).all()

    def clearance(v):
        odds_1 = r * np.exp(-p_gap/v)
        odds_2 = 1.78 * np.exp(-q_nominal/v)
        mean = (odds_1 - a*odds_2)/(1 + odds_1 + odds_2)
        return alpha*(1-s) + lam*(1 - tau**2/v)*(mean-s) + delta/gamma

    lower = brentq(clearance, 0.1, 0.7)
    upper = brentq(clearance, 0.7, 2.5)
    assert 0.17 < lower < 0.69 < 0.71 < upper < 2.5
    v = np.geomspace(0.08, 6, 500)

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), layout="constrained")
    blue, orange, green = "#286488", "#b85c29", "#187750"

    ax = axes[0]
    clipped = np.minimum(h_crit, h_limit)
    ax.fill_between(q, 0, clipped, color=blue, alpha=0.13)
    ax.fill_between(q, clipped, h_limit, color=orange, alpha=0.17)
    ax.plot(q, np.where(h_crit <= h_limit, h_crit, np.nan), color=orange, lw=1.7)
    ax.axvline(q_nominal, color="0.5", lw=0.9, ls=":")
    ax.text(2.18, 0.28, "Interior scale improves\non both endpoints", color="#913f1a")
    ax.annotate("Endpoints suffice", xy=(1.23, 0.042), xytext=(0.94, 0.15),
                color=blue, arrowprops={"arrowstyle": "-", "color": blue, "lw": 0.8})
    ax.set(xlim=(q.min(), q.max()), ylim=(0, h_limit), xlabel=r"Competitor distance gap $Q$",
           ylabel=r"Competitor prior ratio $h$", title="(a) Endpoint dominance threshold")

    ax = axes[1]
    ax.axvspan(v.min(), 0.17, color=blue, alpha=0.13, label="Prescribed restricted set")
    ax.axvspan(2.5, v.max(), color=blue, alpha=0.13)
    for cutoff in (0.17, 2.5):
        ax.axvline(cutoff, color=blue, lw=0.8, ls=":")
    ax.axvspan(lower, upper, color=orange, alpha=0.16, label="Barrier can be violated")
    ax.axvspan(0.69, 0.71, color=green, alpha=0.9, label="Certified band")
    ax.axhline(0, color="0.45", lw=0.8)
    ax.plot(v, clearance(v), color="0.15", lw=1.5)
    ax.scatter([lower, upper], [0, 0], color=orange, s=14, zorder=3)
    ax.set(xscale="log", xlim=(v.min(), v.max()), xlabel=r"Total variance $v$",
           ylabel="Worst horizontal drift", title="(b) Corner drift and scale sets")
    ax.set_xticks([0.1, 0.17, 0.4, 0.7, 1.5, 2.5, 5],
                  [".1", ".17", ".4", ".7", "1.5", "2.5", "5"])
    ax.legend(loc="lower center", fontsize=7.6, frameon=False)
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
