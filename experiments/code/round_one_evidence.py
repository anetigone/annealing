"""Reuse frozen schedules and noise to connect drift, entry, and recovery."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from scale_selection_diagnostic import ALPHA, BUDGET, E1, E2, FIXTURE, GRID, rectangles
from scale_selection_followup import gate, model_for, search, write


ROOT = Path(__file__).resolve().parents[1] / "results"
CHECKPOINTS = (25, 32, 40, 50, 64, 81, 95, BUDGET)
RADIUS = .1
EXTRA_SEED = 20261116
POLICIES = {
    "terminal": ("Terminal only", "#555555"),
    "restricted": ("Restricted alternation", "#406889"),
    "preset": ("Certified preset", "#187750"),
    "shared": ("Tuned shared", "#a25725"),
    "fallback": ("Tuned gated fallback", "#845199"),
}


def plan(out):
    base = json.loads((ROOT / "scale_selection_budget/plan.json").read_text())
    follow = json.loads((ROOT / "scale_selection_followup/plan.json").read_text())
    frozen = {(row["d"], row["h"]): row for row in follow["models"]}
    cases = []
    for row in base["models"]:
        key = row["d"], row["h"]
        if key in frozen:
            cases.append(frozen[key])
        else:
            model = model_for(row)
            cases.append({**row, "draws": 32, "tuned_shared": search(model, False),
                          "tuned_gated": search(model, True)})
    assert len(cases) == 25 and sum(row["band"] is not None for row in cases) == 23
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", {"models": cases, "frozen_seed": follow["seed"],
                            "extra_seed": EXTRA_SEED, "budgets": CHECKPOINTS,
                            "template_radius": RADIUS, "restricted_prefix_calls": 25,
                            "fixture": [*FIXTURE, 1, 0]})


def run(model, truth, b, bounds, prefix, gated, *, initial=None, obstacle_diagnostic=False,
        checkpoints=CHECKPOINTS):
    mask = gate(b, model.alpha) if gated else np.ones(b.shape[:-1], dtype=bool)
    x = (b if initial is None else initial).copy()
    lo, hi = bounds[..., 0][:, None, :], bounds[..., 1][:, None, :]
    first = np.full(b.shape[:-1], -1, dtype=int)
    states, metrics = [], {}
    switch_inside = None
    outside = np.zeros(x.shape[1], dtype=bool)
    starts_outside = None
    for k in range(BUDGET + 1):
        if obstacle_diagnostic:
            outside |= (x[1, :, 0] > .3) | (x[1, :, 1] < -.01)
            if k == 0:
                starts_outside = outside.copy()
        inside = np.all((lo <= x) & (x <= hi), axis=-1)
        first[(first < 0) & inside] = k
        if k == len(prefix):
            switch_inside = inside.copy()
        if k in checkpoints:
            squared = np.sum((x - truth)**2, axis=-1)
            distances = np.sum((x[..., None, :] - model.mu)**2, axis=-1)
            correct = distances.argmin(axis=-1) == np.arange(3)[:, None]
            near = np.sum((x - model.mu[:, None, :])**2, axis=-1) <= RADIUS**2
            assert np.isfinite(squared).all()
            metrics[k] = {"mse": float(model.pi @ squared.mean(axis=1)),
                          "class_mse": squared.mean(axis=1).tolist(),
                          "mode_counts": correct.sum(axis=1).tolist(),
                          "neighborhood_counts": near.sum(axis=1).tolist()}
        states.append(x[1, 0].copy())
        if k < len(prefix):
            x[mask] = model.step(x[mask], b[mask], prefix[k])
            x[~mask] = model.step(x[~mask], b[~mask], model.vend)
        elif k < BUDGET:
            x = model.step(x, b, model.vend)
    summary = {"prefix": len(prefix), "first_entry": first.tolist(),
               "at_switch": switch_inside.sum(axis=1).tolist(), "budgets": metrics}
    if obstacle_diagnostic:
        summary["obstacle"] = {"starts_inside": int((~starts_outside).sum()),
                               "ever_crossed_given_start_inside": int((outside & ~starts_outside).sum())}
    return summary, np.array(states)


def evaluate(out):
    spec = json.loads((out / "plan.json").read_text())
    old_rng = np.random.default_rng(spec["frozen_seed"])
    extra_rng = np.random.default_rng(spec["extra_seed"])
    records = []
    for row in spec["models"]:
        certified = row["band"] is not None
        rng = old_rng if certified else extra_rng
        model = model_for(row)
        n = row["draws"]
        truth = model.mu[:, None, :] + model.tau * rng.standard_normal((3, n, 2))
        b = (truth @ model.A + ALPHA**2 * rng.standard_normal((3, n, 2))) @ model.A
        event = np.all(np.abs(b - model.mu[:, None, :] @ model.B) <= [E1, E2], axis=-1)
        schedules = {"terminal": ([], False),
                     "restricted": ([row["low"], row["high"]] * 12 + [row["low"]], False)}
        shared = row["tuned_shared"]
        schedules["shared"] = ([shared["v"]] * shared["calls"], False)
        if certified:
            schedules["preset"] = ([row["selected"]] * row["certified_calls"], True)
        else:
            tuned = row["tuned_gated"]
            schedules["fallback"] = ([tuned["v"]] * tuned["calls"], True)
        results, sample_paths = {}, {}
        for key, (prefix, gated) in schedules.items():
            results[key], sample_paths[key] = run(
                model, truth, b, np.array(rectangles(row["d"], float)), prefix, gated)
        if certified and event.all():
            first = np.array(results["preset"]["first_entry"])
            assert np.all((0 <= first[1]) & (first[1] <= row["certified_calls"]))
            assert results["preset"]["at_switch"] == [n] * 3
        prediction = np.where(b[..., 1] > 1, 2, np.where(b[..., 0] > ALPHA/2, 1, 0))
        records.append({"d": row["d"], "h": row["h"], "draws": n,
                        "event_counts": event.sum(axis=1).tolist(),
                        "classifier_correct": (prediction == np.arange(3)[:, None]).sum(axis=1).tolist(),
                        "certified_calls": row.get("certified_calls"), "policies": results})
        if (row["d"], row["h"]) == FIXTURE:
            mechanism(out, model, row, b[1, 0], sample_paths)
    archived = json.loads((ROOT / "scale_selection_followup/results.json").read_text())
    for new, old in zip((r for r in records if r["certified_calls"]), archived["records"], strict=True):
        assert (new["d"], new["h"]) == (old["d"], old["h"])
        for key, source in (("shared", "tuned_shared"), ("preset", "preset_certified")):
            now, before = new["policies"][key]["budgets"][BUDGET], old["results"][source]
            assert now["class_mse"] == before["class_squared_errors"]
            assert now["mode_counts"] == before["nearest_counts"]
    write(out / "results.json", records)
    figures(out, records)


def mechanism(out, model, row, b, paths):
    x = paths["preset"]
    first = np.flatnonzero(np.all((x >= [.53, -.1]) & (x <= [1.1, .15]), axis=1))[0]
    selected = (0, first // 2, first - 1)
    write(out / "fixture.json", {"backprojection": b.tolist(), "first_entry": int(first),
                                "sampled_calls": [int(k) for k in selected],
                                "paths": {key: path.tolist() for key, path in paths.items()}})
    lo, hi = row["low"], row["high"]
    v = np.geomspace(model.vend, 20, 500)
    fig, (ax, weight, drift) = plt.subplots(1, 3, figsize=(11.5, 3.15), layout="constrained")
    ax.axvspan(-.35, .3, color="#406889", alpha=.12, label=r"Restricted bound $s\leq .3$")
    ax.add_patch(Rectangle((.53, -.1), .57, .25, color="#187750", alpha=.13))
    ax.text(.85, .135, r"$G_1$", ha="center", color="#187750")
    for key in ("restricted", "preset"):
        label, color = POLICIES[key]
        path = paths[key]
        m = row["certified_calls"] if key == "preset" else 25
        ax.plot(*path[:m+1].T, color=color, label=label)
        ax.plot(*path[m:].T, "--", color=color, lw=1)
    ax.plot(*x[first], '*', color="#187750", label=f"First entry (call {first})")
    for k, color in zip(selected, ("#343434", "#845199", "#a25725")):
        ax.scatter(*x[k], color=color, s=16, zorder=5)
        ax.annotate(str(k), x[k], xytext=(0, 5), textcoords="offset points", color=color)
        w = model.weights(x[k], v[:, None])
        step = model.step(x[k], b, v[:, None])
        for z, ls in enumerate((":", "-", "--")):
            weight.plot(v, w[:, z], color=color, ls=ls, lw=1.1)
        drift.plot(v, step[:, 0] - x[k, 0], color=color, label=f"Call {k}")
    ax.set(xlim=(-.35, 1.11), ylim=(-.1, .17), xlabel="Weak coordinate s",
           ylabel="Transverse coordinate t", title="(a) Matched trajectories")
    ax.legend(fontsize=6.8, loc="lower right", frameon=False)
    for panel in (weight, drift):
        panel.set(xscale="log", xlim=(.03, 20), xlabel="Total variance v")
        panel.axvspan(.03, lo, color="#406889", alpha=.12)
        panel.axvspan(hi, 20, color="#406889", alpha=.12)
        panel.axvspan(*row["band"], color="#187750", alpha=.30)
    for z, ls in enumerate((":", "-", "--")):
        weight.plot([], [], ls=ls, color=".3", label=f"Template {z}")
    weight.set(ylabel="Responsibility", title="(b) Template competition")
    drift.set(ylabel=r"Horizontal update $s_{k+1}-s_k$", title="(c) Local drift")
    drift.axhline(0, color=".5", lw=.7)
    weight.legend(fontsize=7, frameon=False)
    drift.legend(fontsize=7, frameon=False)
    fig.savefig(out / "mechanism.pdf")
    plt.close(fig)


def figures(out, records):
    certified = [r for r in records if r["certified_calls"]]
    fig, ax = plt.subplots(figsize=(5.9, 3.4), layout="constrained")
    for key in ("terminal", "restricted", "preset", "shared"):
        label, color = POLICIES[key]
        y = [np.mean([r["policies"][key]["budgets"][k]["mse"] for r in certified])
             for k in CHECKPOINTS]
        ax.semilogy(CHECKPOINTS, y, ".--" if key == "terminal" else ".-",
                    color=color, label=label, zorder=4 if key == "terminal" else 3)
    ax.set(xlabel="Total denoiser calls", ylabel="Mean squared Euclidean error")
    ax.legend(frameon=False)
    fig.savefig(out / "budget_error.pdf")
    plt.close(fig)

    lookup = {(r["d"], r["h"]): r for r in records}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.15), layout="constrained")
    for ax, kind, title in zip(axes, ("preset", "shared"),
                               ("Certified preset / numerical fallback*", "Independently tuned shared")):
        for i, h in enumerate(GRID["h"]):
            for j, d in enumerate(GRID["d"]):
                r = lookup[(d, h)]
                cap, n = r["certified_calls"], r["draws"]
                key = ("preset" if cap else "fallback") if kind == "preset" else kind
                policy = r["policies"][key]
                last = policy["budgets"][BUDGET]
                first = np.array(policy["first_entry"])[1]
                observed = str(first.max()) if (first >= 0).all() else "--"
                mode, near = last["mode_counts"], last["neighborhood_counts"]
                passed = near == [n] * 3
                bg = ("#e0efe6" if cap and key == "preset" else "#e9e1f2") if passed else "#f8e7da"
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, color=bg, ec=None))
                star = "*" if key == "fallback" else ""
                guarantee = str(cap) if key == "preset" else "--"
                ax.text(j, i, f"{guarantee} / {observed}{star}\n"
                        f"G: {sum(v == n for v in policy['at_switch'])}/3\n"
                        f"M/N: {sum(mode)}/{sum(near)}\nn={3*n}",
                        ha="center", va="center", fontsize=7.2)
        ax.set(xlim=(-.5, 4.5), ylim=(-.5, 4.5), xticks=range(5), yticks=range(5),
               xticklabels=GRID["d"], yticklabels=GRID["h"], xlabel="Third-template height d",
               ylabel="Third-template relative mass h", title=title)
    fig.supxlabel("Cell: certified bound / latest observed class-1 first entry; "
                 "G: all draws in rectangle at switch by class; M/N: mode / neighborhood counts",
                 fontsize=8)
    fig.savefig(out / "entry_recovery.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "evaluate"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    (plan if args.phase == "plan" else evaluate)(args.out)


if __name__ == "__main__":
    main()
