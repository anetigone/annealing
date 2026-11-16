"""Plot matched nominal trajectories for the early manuscript figure."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "red_geometry_computation_risk/code"))
from family import Family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = Family(.001)
    model.pi = np.array([1., 1.29, 1.78])
    model.pi /= model.pi.sum()
    b = model.B @ model.mu[1]
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    fig, (geometry, history) = plt.subplots(1, 2, figsize=(7.2, 2.13),
                                          layout="constrained")
    geometry.add_patch(Rectangle((-.12, -.01), .42, .23, color="#6d91ac", alpha=.18))
    geometry.add_patch(Rectangle((.53, -.1), .57, .25, color="#63a583", alpha=.18))
    geometry.axvline(.3, color="#406889", lw=.8, ls=":")
    geometry.axvline(.53, color="#187750", lw=.8, ls=":")
    geometry.text(.09, .188, r"Restricted region", ha="center", color="#385d77")
    geometry.text(.82, .127, r"Terminal region $G_1$", ha="center", color="#126343")
    for label, color, prefix in [
            ("Restricted switching", "#406889", [.17, 2.5] * 12 + [.17]),
            ("Intermediate preset", "#187750", [.7] * 25)]:
        x = b.copy()
        states = [x]
        for v in prefix + [model.vend] * 81:
            x = model.step(x, b, v)
            states.append(x)
        states = np.asarray(states)
        assert np.isfinite(states).all()
        geometry.plot(*states[:26].T, ".-", color=color, ms=2.2, lw=1.15)
        geometry.plot(*states[25:].T, "--", color=color, lw=1.1)
        geometry.plot(*states[25], "o", color=color, ms=4, mfc="white")
        history.plot(states[:, 0], color=color, lw=1.3, label=label)
        history.plot(25, states[25, 0], "o", color=color, ms=4, mfc="white")
    geometry.scatter(*b, marker="s", color="black", s=16, zorder=5)
    geometry.scatter(*model.mu[1], marker="*", color="black", s=60, zorder=5)
    geometry.text(.01, -.077, r"Start $A^\top y$", ha="left")
    geometry.text(.99, -.073, r"Truth $\mu_1$", ha="right")
    geometry.set(xlim=(-.12, 1.10), ylim=(-.10, .22), xlabel="Weak coordinate $s$",
                 ylabel="Transverse coordinate $t$", title="(a) Barrier and terminal entry")
    history.axvline(25, color=".5", lw=.8, ls=":")
    history.axhline(.3, color="#406889", lw=.65, ls=":")
    history.axhline(.53, color="#187750", lw=.65, ls=":")
    history.set(xlim=(0, 106), ylim=(-.30, 1.05), xlabel="Denoiser calls",
                ylabel="Weak coordinate $s$", title="(b) Matched budget")
    history.legend(loc="center right", bbox_to_anchor=(1, .70),
                   frameon=False, fontsize=8)
    for ax in (geometry, history):
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
