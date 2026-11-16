"""Plot frozen-control decision curves and nominal error decomposition."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from budget_control_validation import load, model_for


ROOT = Path(__file__).resolve().parents[1] / "results"
COLORS = ("#3379a5", "#d97e36")


def decision_panel(ax, ti, ni, mi, spec, calibration, evaluation, curves):
    model = model_for(spec["models"][mi])
    nu = spec["measurement_noise_sd"][ni]
    for phase, data, marker, alpha, size in (("Evaluation", evaluation, "o", .5, 19),
                                           ("Calibration", calibration, "+", 1., 42)):
        b = data[f"b_{ni}_{mi}"]
        for z in (0, 1):
            ax.scatter(b[z, :, 0]/model.alpha, b[z, :, 1]/nu,
                       color=COLORS[z], marker=marker, alpha=alpha, s=size,
                       linewidths=.8 if marker == "+" else 0)
    for label, style, color in (("original", "-", "#614f99"),
                                 ("ready", "--", "#555555")):
        ax.plot(curves[f"boundary_{ti}_{label}"], curves[f"y_{ti}_{label}"],
                style, color=color, lw=1.6)
    if ti == 0:
        ax.plot(curves["boundary_0_short_choice"], curves["y_0_short_choice"],
                ":", color="#267455", lw=1.6)
    ax.set(xlabel=r"Weak observation $b_1/\alpha$", ylabel=r"Transverse observation $b_2/\nu$",
           xlim=(-.2, 1.4), ylim=(-4, 4))
    ax.grid(alpha=.16)
    chosen = "noise_gated" if ti == 1 else "noise_shared"
    x = evaluation[f"x_{ni}_{mi}_106_{chosen}"]
    nearest = np.sum((x[..., None, :]-model.mu)**2, -1).argmin(-1)
    for z, j in np.argwhere(nearest[:2] != np.arange(2)[:, None]):
        b = evaluation[f"b_{ni}_{mi}"][z, j]
        ax.scatter(b[0]/model.alpha, b[1]/nu, s=68, facecolor="none", edgecolor="#b52c37", lw=1.2)
    if ti == 1:
        b = calibration[f"b_{ni}_{mi}"][0, 15]
        ax.scatter(b[0]/model.alpha, b[1]/nu, s=85, facecolor="none", edgecolor="#242424", lw=1.4)
        ax.annotate("Influential calibration sample", (b[0]/model.alpha, b[1]/nu),
                    xytext=(.06, 3.2), fontsize=7, arrowprops={"arrowstyle": "-", "lw": .7})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    target = args.out / "mechanism.png"
    if target.exists():
        raise FileExistsError(target)
    source = ROOT / "budget_control_validation"
    spec = load(source / "plan.json")
    replay = load(args.out / "trajectories.json")
    calibration = np.load(source / "calibration.npz")
    evaluation = np.load(source / "evaluation.npz")
    curves = np.load(args.out / "decision_boundaries.npz")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    rows = replay["nominal"]
    variance = np.array([r["variance_effect"] for r in rows])
    prefix = np.array([r["prefix_effect_after_variance"] for r in rows])
    scale = 12*np.mean([r["values"]["ready"]["risk"] for r in rows])/100
    axes[0].bar(range(12), variance/scale, color=COLORS[0], label="Variance change")
    axes[0].bar(range(12), prefix/scale, bottom=variance/scale,
                color=COLORS[1], label="One fewer prefix call")
    axes[0].set(title="(a) Nominal 22-call gain: where it comes from",
                xlabel="Fixed geometry (d / h)", ylabel="Contribution to pooled error reduction (pp)",
                xticks=range(12), xticklabels=[f"{r['d']}/{r['h']}" for r in rows])
    axes[0].tick_params(axis="x", labelrotation=75, labelsize=7)
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    axes[0].grid(axis="y", alpha=.16)
    decision_panel(axes[1], 1, 2, 5, spec, calibration, evaluation, curves)
    axes[1].set_title(r"(b) Gated: $d=1.62,h=2.05,\nu=.005$")
    decision_panel(axes[2], 0, 1, 7, spec, calibration, evaluation, curves)
    axes[2].set_title(r"(c) Shared: $d=1.62,h=3,\nu=.001$")
    axes[1].legend(handles=[Line2D([], [], color=COLORS[0], marker="o", lw=0, label="Class 0"),
                           Line2D([], [], color=COLORS[1], marker="o", lw=0, label="Class 1"),
                           Line2D([], [], color=".4", marker="+", lw=0, label="Calibration"),
                           Line2D([], [], color="#b52c37", marker="o", markerfacecolor="none", lw=0,
                                  label="Evaluation failure")], loc="lower left", fontsize=6.7, frameon=False)
    axes[2].legend(handles=[Line2D([], [], color="#614f99", label="Original selected control"),
                           Line2D([], [], color="#555555", ls="--", label="Ready control"),
                           Line2D([], [], color="#267455", ls=":", label="22-call choice, continued")],
                   loc="lower left", fontsize=6.7, frameon=False)
    fig.supxlabel("Curves in (b,c): 106-call decisions between templates 0 and 1; right side selects 1.\n"
                  "All original observations and choices are retained. Post hoc diagnosis; no new policy is fitted.", fontsize=8)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
