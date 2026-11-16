"""Analyze saved development force probes without executing a denoiser."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

FAMILIES = ("endpoints", "platform", "geometric")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path, keys):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for r in rows:
        for key, value in r.items():
            if key not in ("image", "family"):
                r[key] = float(value) if value else None
                if r[key] is not None and not np.isfinite(r[key]):
                    raise ValueError(f"Nonfinite {path.name}/{key}")
        identity = tuple(r[k] for k in keys)
        if identity in seen:
            raise ValueError(f"Duplicate row in {path.name}: {identity}")
        seen.add(identity)
    return rows


def force_effects(rows, gamma):
    result = []
    for r in rows:
        reg = r["regularization_field_rms"]
        central = -2*gamma*r["regularization_error_inner_product"]+gamma**2*reg**2
        half_width = 2*gamma**2*r["data_field_rms"]*reg
        result.append({**r, "reg_vs_data_only_mse_gap_lower": central-half_width,
                       "reg_vs_data_only_mse_gap_upper": central+half_width})
    return result


def paired_endpoints(rows):
    refs = {(r["image"], r["noise_255"], r["total_calls"]): r
            for r in rows if r["family"] == "terminal"}
    fractions = {}
    for r in refs.values():
        if r["image"] not in fractions:
            fractions[r["image"]] = ((r["mse"]-r["border_mse"])
                                      /(r["interior_mse"]-r["border_mse"]))
    assert all(0 < value < 1 for value in fractions.values())
    pairs, reconstruction_errors = [], []
    for r in rows:
        w = fractions[r["image"]]
        reconstruction_errors.append(abs(r["mse"]-w*r["interior_mse"]-(1-w)*r["border_mse"]))
        if r["family"] == "terminal":
            continue
        ref = refs[(r["image"], r["noise_255"], r["total_calls"])]
        interior = r["interior_mse"]-ref["interior_mse"]
        border = r["border_mse"]-ref["border_mse"]
        gap = r["mse"]-ref["mse"]
        assert abs(gap-r["mse_gap_vs_terminal"]) < 1e-15
        pairs.append({"image": r["image"], "noise_255": r["noise_255"],
                      "family": r["family"], "total_calls": r["total_calls"],
                      "mse_gap": gap, "relative_mse_gap": gap/ref["mse"],
                      "interior_mse_gap": interior, "border_mse_gap": border,
                      "interior_contribution": w*interior,
                      "border_contribution": (1-w)*border,
                      "output_rms_vs_terminal": r["output_rms_vs_terminal"]})
    assert max(reconstruction_errors) < 1e-14
    return pairs, fractions, max(reconstruction_errors)


def transport_contributions(rows, endpoints):
    saved = {(r["image"], r["noise_255"], r["family"], int(r["total_calls"])): r
             for r in endpoints}
    result = []
    cases = sorted({(r["image"], r["noise_255"], r["family"]) for r in rows})
    for image, noise, family in cases:
        for start, end in ((32, 64), (64, 128)):
            group = sorted((r for r in rows if (r["image"], r["noise_255"], r["family"]) == (image, noise, family)
                            and start <= r["from_call"] < end), key=lambda r: r["from_call"])
            assert [int(r["from_call"]) for r in group] == list(range(start, end))
            initial, final = group[0]["before_squared_distance"], group[-1]["after_squared_distance"]
            assert np.isclose(initial, saved[(image, noise, family, start)]["output_rms_vs_terminal"]**2,
                              rtol=1e-12, atol=1e-25)
            assert np.isclose(final, saved[(image, noise, family, end)]["output_rms_vs_terminal"]**2,
                              rtol=1e-12, atol=1e-25)
            values = {"data_reduction": -sum(r["data_distance_change_term"] for r in group),
                      "denoiser_reduction": -sum(r["denoiser_distance_change_term"] for r in group),
                      "step_reduction": -sum(r["squared_step_term"] for r in group),
                      "roundoff_reduction": -sum(r["identity_roundoff_residual"] for r in group)}
            reduction = initial-final
            assert abs(reduction-sum(values.values())) < 1e-19
            result.append({"image": image, "noise_255": noise, "family": family,
                           "start": start, "end": end, "initial_distance2": initial,
                           "final_distance2": final, "net_reduction": reduction, **values,
                           "max_distance_ratio": max(r["distance_ratio"] for r in group),
                           "secant_projection_above_one_steps": sum(r["denoiser_secant_projection_gain"] > 1 for r in group),
                           "steps": len(group)})
    pooled = []
    for noise, family, start in sorted({(r["noise_255"], r["family"], r["start"]) for r in result}):
        group = [r for r in result if (r["noise_255"], r["family"], r["start"]) == (noise, family, start)]
        mean = lambda key: float(np.mean([r[key] for r in group]))
        net = mean("net_reduction")
        pooled.append({"noise_255": noise, "family": family, "start": start, "end": group[0]["end"],
                       "images": len(group), "mean_initial_distance2": mean("initial_distance2"),
                       "mean_final_distance2": mean("final_distance2"), "mean_net_reduction": net,
                       **{k+"_share": mean(k)/net if net else None for k in
                          ("data_reduction", "denoiser_reduction", "step_reduction", "roundoff_reduction")},
                       "secant_projection_above_one_steps": sum(r["secant_projection_above_one_steps"] for r in group),
                       "steps": sum(r["steps"] for r in group)})
    return result, pooled


def means(rows, keys, values):
    groups = sorted({tuple(r[k] for k in keys) for r in rows})
    result = []
    for identity in groups:
        group = [r for r in rows if tuple(r[k] for k in keys) == identity]
        result.append({**dict(zip(keys, identity)), "images": len(group),
                       **{v: float(np.mean([r[v] for r in group])) for v in values}})
    return result


def plot(out, forces, pairs, endpoints, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), layout="constrained")
    low, high = sorted({r["noise_255"] for r in forces})
    colors = {low: "#2864a0", high: "#b44135"}
    ax = axes[0, 0]
    for noise in (low, high):
        for image in sorted({r["image"] for r in forces}):
            group = sorted((r for r in forces if r["image"] == image and r["noise_255"] == noise
                            and r["state_call"] == 64 and r["sigma_255"] != 5), key=lambda r: r["sigma_255"])
            ax.scatter([r["sigma_255"] for r in group], [1e6*r["one_step_mse_gap_vs_terminal"] for r in group],
                       color=colors[noise], alpha=.45, s=20)
        group = means([r for r in forces if r["noise_255"] == noise and r["state_call"] == 64
                       and r["sigma_255"] != 5], ["sigma_255"], ["one_step_mse_gap_vs_terminal"])
        ax.plot([r["sigma_255"] for r in group], [1e6*r["one_step_mse_gap_vs_terminal"] for r in group],
                "o-", color=colors[noise], label=f"Observation noise {noise:g}/255")
    ax.axhline(0, color=".4", linewidth=.8)
    ax.set(title="(a) Replacing one terminal call at state 64", xlabel="Alternative denoiser sigma (8-bit units)",
           ylabel="MSE change vs sigma 5 (x 1e-6)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    group = [next(r for r in pooled if r["noise_255"] == n and r["family"] == f and r["start"] == 64)
             for n in (low, high) for f in FAMILIES]
    positive, negative = np.zeros(6), np.zeros(6)
    for key, label, color in (("data_reduction_share", "Data term", "#4b83af"),
                              ("denoiser_reduction_share", "Denoiser term", "#d28b41"),
                              ("step_reduction_share", "Squared step", "#93999f")):
        values = np.array([100*r[key] for r in group])
        ax.bar(range(6), values, bottom=np.where(values >= 0, positive, negative), label=label, color=color)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    ax.axhline(100, color=".25", linestyle="--", linewidth=.8)
    ax.axhline(0, color=".4", linewidth=.8)
    ax.set_xticks(range(6), [f"{n:g}\n{f[:3]}" for n in (low, high) for f in FAMILIES])
    ax.set(title="(b) Distance-reduction terms, calls 64 to 128", xlabel="Observation noise (8-bit units) / history",
           ylabel="Share of net squared-distance reduction (%)", ylim=(-9, 116))
    ax.legend(fontsize=8, loc="center right")
    ax = axes[1, 0]
    group = [means([r for r in pairs if r["family"] == "platform" and r["noise_255"] == n
                    and r["total_calls"] == k], ["noise_255", "total_calls"],
                   ["mse_gap", "interior_contribution", "border_contribution"])[0]
             for n in (low, high) for k in (64, 128)]
    positive, negative = np.zeros(4), np.zeros(4)
    for key, label, color in (("interior_contribution", "Interior contribution", "#548f75"),
                              ("border_contribution", "32-pixel border contribution", "#bf8b59")):
        values = np.array([1e6*r[key] for r in group])
        ax.bar(range(4), values, bottom=np.where(values >= 0, positive, negative), label=label, color=color)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    ax.scatter(range(4), [1e6*r["mse_gap"] for r in group], color="black", marker="D", s=22, label="Whole image")
    ax.axhline(0, color=".4", linewidth=.8)
    ax.set_xticks(range(4), [f"{n:g} / {k}" for n in (low, high) for k in (64, 128)])
    ax.set(title="(c) Platform error gap: interior and border", xlabel="Observation noise / total calls",
           ylabel="Contribution to whole-image MSE gap (x 1e-6)")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for noise in (low, high):
        group = means([r for r in endpoints if r["noise_255"] == noise and r["family"] == "terminal"
                       and r["total_calls"] > 0], ["total_calls"], ["mse", "data_only_mse"])
        for key, label, style in (("mse", "terminal RED", "o-"), ("data_only_mse", "data only", "s--")):
            ax.plot([r["total_calls"] for r in group], [1e3*r[key] for r in group], style,
                    color=colors[noise], markersize=4, label=f"{noise:g}/255, {label}")
    ax.set(title="(d) Fixed-solver progress on the same observations", xlabel="Total gradient updates",
           ylabel="Mean per-image MSE (x 1e-3)")
    ax.legend(fontsize=8)
    fig.suptitle("DRUNet diagnostics: two fixed development images, frozen histories", fontsize=12)
    fig.savefig(out/"cause_diagnostics.png", dpi=180)
    fig.savefig(out/"cause_diagnostics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec, completion = read_json(args.source/"plan.json"), read_json(args.source/"completion.json")
    forces = read_rows(args.source/"forces.csv", ("image", "noise_255", "state_call", "sigma_255"))
    endpoints = read_rows(args.source/"endpoints.csv", ("image", "noise_255", "family", "total_calls"))
    transports = read_rows(args.source/"terminal_transport.csv", ("image", "noise_255", "family", "from_call"))
    sanity = read_rows(args.source/"denoising_sanity.csv", ("image",))
    images, noises = len(spec["development_images"]), len(spec["noise_sigma_255"])
    if noises != 2:
        raise ValueError("This comparison expects the two planned noise amplitudes")
    assert len(forces) == images*noises*16 and len(endpoints) == images*noises*24
    prefixes = [spec["frozen_choices"][f]["candidate"]["prefix"] for f in FAMILIES]
    assert len(transports) == images*noises*sum(128-m for m in prefixes)
    assert len(sanity) == images
    assert completion["network_forwards"] == spec["expected_network_forwards"]
    assert completion["all_checks_bitwise_equal"] and completion["production_integrator_checks"] == images*noises
    effects = force_effects(forces, spec["settings"]["solver"]["step"])
    pairs, fractions, region_error = paired_endpoints(endpoints)
    individual, pooled = transport_contributions(transports, endpoints)
    force_means = means(effects, ["noise_255", "state_call", "sigma_255"],
                        ["one_step_mse_gap_vs_terminal", "regularization_over_data_rms",
                         "residual_orthogonal_energy_fraction", "reg_vs_data_only_mse_gap_lower",
                         "reg_vs_data_only_mse_gap_upper"])
    pair_means = means(pairs, ["noise_255", "family", "total_calls"],
                       ["mse_gap", "interior_mse_gap", "border_mse_gap", "interior_contribution",
                        "border_contribution", "output_rms_vs_terminal"])
    times = [(args.source/name).stat().st_mtime for name in ("plan.json", "completion.json")]
    china = timezone(timedelta(hours=8))
    summary = {"scope": spec["scope"], "images": images, "noise_amplitudes": spec["noise_sigma_255"],
               "counts": {"endpoints": len(endpoints), "forces": len(forces), "transport": len(transports)},
               "completion": completion, "interior_area_fractions": fractions,
               "region_reconstruction_max_error": region_error,
               "transport_max_absolute_roundoff": max(abs(r["identity_roundoff_residual"]) for r in transports),
               "source_file_times_Asia_Shanghai": [datetime.fromtimestamp(t, china).isoformat() for t in times],
               "source_file_interval_seconds": times[1]-times[0],
               "sanity": sanity, "force_means": force_means, "pair_means": pair_means,
               "transport_pooled": pooled}
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out/"force_effects.csv", effects)
    write_csv(args.out/"paired_endpoints.csv", pairs)
    write_csv(args.out/"transport_contributions.csv", individual)
    write_csv(args.out/"transport_pooled.csv", pooled)
    write_json(args.out/"summary.json", summary)
    plot(args.out, effects, pairs, endpoints, pooled)
    print(f"Saved {args.out}; {images} development images, {completion['network_forwards']} forwards, all checks passed")


if __name__ == "__main__":
    main()
