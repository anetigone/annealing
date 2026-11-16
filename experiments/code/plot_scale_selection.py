"""Plot the frozen scale choices and first-entry bounds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw_panel(ax, values, labels, heights, masses, title, limits, cmap):
    colors = matplotlib.colormaps[cmap].copy()
    colors.set_bad("#e4e6e8")
    heatmap = ax.imshow(values, origin="lower", cmap=colors,
              vmin=limits[0], vmax=limits[1], aspect="auto")
    ax.set_xticks(range(len(heights)), [f"{d:g}" for d in heights])
    ax.set_yticks(range(len(masses)), [f"{h:g}" for h in masses])
    ax.set_xticks(np.arange(len(heights)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(masses)+1)-.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, column), label in np.ndenumerate(labels):
        value = values[row, column]
        rgb = np.array(heatmap.cmap(heatmap.norm(value))[:3])
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        luminance = linear @ np.array([.2126, .7152, .0722])
        color = "black" if luminance > .179 else "white"
        ax.text(column, row, label, ha="center", va="center", color=color)
    ax.set(xlabel="Height $d$", ylabel="Prior ratio $h$", title=title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.data/"plan.json").read_text(encoding="utf-8"))
    results = json.loads((args.data/"results.json").read_text(encoding="utf-8"))
    models = {(case["d"], case["h"]): case for case in plan["models"]}
    records = {(record["d"], record["h"]): record for record in results["records"]}
    heights, masses = plan["grid"]["d"], plan["grid"]["h"]
    shape = (len(masses), len(heights))
    centers = np.full(shape, np.nan)
    calls = np.full(shape, np.nan)
    center_labels = np.full(shape, "--", dtype=object)
    call_labels = center_labels.copy()
    for row, h in enumerate(masses):
        for column, d in enumerate(heights):
            case = models[d, h]
            if case["band"] is None:
                assert (d, h) not in records
                continue
            entry = records[d, h]["results"]["observation_preset"]["first_entry_calls"][1]
            observed, bound = max(entry), case["certified_calls"]
            assert min(entry) > 0 and observed <= bound
            centers[row, column] = case["selected"]
            calls[row, column] = bound
            center_labels[row, column] = f"{case['selected']:.2f}"
            call_labels[row, column] = f"{observed} / {bound}"
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, (ax_centers, ax_calls) = plt.subplots(1, 2, figsize=(6.9, 2.40),
                                               layout="constrained")
    draw_panel(ax_centers, centers, center_labels, heights, masses,
               "(a) Selected variance center", (.34, .62), "GnBu")
    draw_panel(ax_calls, calls, call_labels, heights, masses,
               "(b) Observed / certified entry calls", (13, 25), "PuBu")
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
