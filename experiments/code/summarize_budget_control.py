"""Plot independent paired controls and report conditional Monte Carlo error."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from budget_control_validation import ALPHA, POLICIES, load, model_for, write


CONTROLS = ("same_center_direct", "noise_shared", "noise_gated")
COLORS = ("#167450", "#a45d2b", "#65539d")
LABELS = ("Same-center direct", "Noisy tuned shared", "Noisy tuned gated")


def summarize(out):
    spec, results = load(out / "plan.json"), load(out / "results.json")
    data = np.load(out / "evaluation.npz")
    budgets, count = spec["budgets"], len(spec["models"])
    pairs, summary_rows = [], []
    for ni, nu in enumerate(spec["measurement_noise_sd"]):
        summary = results["summary"][str(ni)]
        for budget in budgets:
            for name in POLICIES:
                value = summary["budgets"][str(budget)]["metrics"][name]
                summary_rows.append({"noise_sd": nu, "calls": budget, "policy": name, **value})
            for control in POLICIES:
                if control == "ready":
                    continue
                mean, variance, distance, residual = 0., 0., 0., [0., 0.]
                for mi, case in enumerate(spec["models"]):
                    model = model_for(case)
                    truth = data[f"truth_{ni}_{mi}"]
                    b = data[f"b_{ni}_{mi}"]
                    ready = data[f"x_{ni}_{mi}_{budget}_ready"]
                    other = data[f"x_{ni}_{mi}_{budget}_{control}"]
                    diff = np.sum((ready-truth)**2, -1) - np.sum((other-truth)**2, -1)
                    mean += float(model.pi @ diff.mean(axis=1)) / count
                    variance += float((model.pi**2) @ diff.var(axis=1, ddof=1)) / diff.shape[1] / count**2
                    distance += float(model.pi @ np.sum((ready-other)**2, -1).mean(axis=1)) / count
                    for j, x in enumerate((ready, other)):
                        field = (x-model.step(x, b, model.vend))/.25
                        residual[j] += float(model.pi @ np.sum(field**2, -1).mean(axis=1)) / count
                se = float(np.sqrt(variance))
                pairs.append({"noise_sd": nu, "calls": budget, "control": control,
                              "ready_minus_control": mean, "paired_mc_se": se,
                              "normal_95_interval": [mean-1.96*se, mean+1.96*se],
                              "output_rms_distance": float(np.sqrt(distance)),
                              "terminal_field_rms": np.sqrt(residual).tolist()})
    write(out / "paired_summary.json", {
        "scope": "Monte Carlo uncertainty conditional on the frozen models and calibration choices; "
                 "independent stratified draws within class/model. Normal intervals are descriptive, "
                 "not simultaneous bounds or uncertainty across natural-image populations. "
                 "Terminal field norms at intermediate budgets are counterfactual terminal diagnostics.",
        "records": pairs})
    with (out / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot(out, spec, results)
    failures(out)
    print(json.dumps({nu: {k: v for k, v in row.items() if k != "budgets"}
                      for nu, row in results["summary"].items()}, indent=2))
    for ni, nu in enumerate(spec["measurement_noise_sd"]):
        print(f"noise sd {nu:g}")
        for budget in (budgets[0], 40, 106):
            values = results["summary"][str(ni)]["budgets"][str(budget)]
            print(budget, {name: f"{value['risk']:.8g}" for name, value in values["metrics"].items()})
            print("ready better/worse:", {name: [value["better"], value["worse"]]
                                         for name, value in values["ready_minus"].items()})


def plot(out, spec, results):
    if (out / "paired_error.pdf").exists():
        raise FileExistsError("Plot already exists")
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), layout="constrained", sharey=True)
    for ni, (ax, nu) in enumerate(zip(axes, spec["measurement_noise_sd"], strict=True)):
        for control, color, label, offset in zip(CONTROLS, COLORS, LABELS, (-.22, 0, .22), strict=True):
            for j, budget in enumerate(spec["budgets"]):
                comp = results["summary"][str(ni)]["budgets"][str(budget)]["ready_minus"][control]
                values = comp["differences"]
                x = j+offset+np.linspace(-.045, .045, len(values))
                ax.scatter(x, values, color=color, alpha=.42, s=12, linewidths=0)
                ax.scatter(j+offset, np.mean(values), color=color, marker="D", s=25,
                           label=label if ni == 0 and j == 0 else None, zorder=4)
        eps = spec["comparison_threshold"]
        ax.axhspan(-eps, eps, color=".94", zorder=0)
        ax.axhline(0, color=".45", lw=.7)
        ax.set_yscale("symlog", linthresh=eps)
        ax.set(title=rf"Measurement noise $\nu={nu:g}$", xlabel="Total denoiser calls",
               xticks=range(len(spec["budgets"])), xticklabels=spec["budgets"])
    axes[0].set_ylabel("Ready minus control: squared error")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
    fig.supxlabel("Lower favors ready. Dots: 12 fixed models; diamonds: model means.\n"
                  "Evaluation noise is independent of calibration.", fontsize=7)
    fig.savefig(out / "paired_error.pdf")
    plt.close(fig)


def failures(out):
    """Post hoc classification/event stratification; no policy is changed."""
    spec = load(out / "plan.json")
    data = np.load(out / "evaluation.npz")
    rows = []
    for ni, nu in enumerate(spec["measurement_noise_sd"]):
        for mi, case in enumerate(spec["models"]):
            model = model_for(case)
            b = data[f"b_{ni}_{mi}"]
            event = np.all(np.abs(b-model.mu[:, None, :] @ model.B) <= [.00025, .005], -1)
            predicted = np.where(b[..., 1] > 1, 2, np.where(b[..., 0] > ALPHA/2, 1, 0))
            correct = predicted == np.arange(3)[:, None]
            for name in POLICIES:
                x = data[f"x_{ni}_{mi}_106_{name}"]
                nearest = np.sum((x[..., None, :]-model.mu)**2, -1).argmin(axis=-1)
                wrong = nearest != np.arange(3)[:, None]
                if wrong.any():
                    rows.append({"nu": nu, "d": case["d"], "h": case["h"], "policy": name,
                                 "wrong_by_class": wrong.sum(axis=1).tolist(),
                                 "wrong_inside_event": int((wrong & event).sum()),
                                 "wrong_despite_correct_classifier": int((wrong & correct).sum())})
    write(out / "failures.json", {"scope": "Post hoc 106-call description on saved evaluation states", "records": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    summarize(parser.parse_args().out)
