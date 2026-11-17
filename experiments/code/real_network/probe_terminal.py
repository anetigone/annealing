"""Extend the frozen repaired histories under their common terminal RED field."""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from common import configure_device, read_json, red_trajectory, write_json
from denoisers import load_denoiser
from problem import CircularBlur, load_image, observation
from repair_solver import scale_sequence


def inner(a, b):
    return float((a.double() * b.double()).mean().item())


def rms(a):
    return inner(a, a)**.5


def terminal_field(denoiser, blur, y, x, solver):
    denoised = denoiser(x, solver["terminal_sigma_255"] / 255)
    data = blur.adjoint(blur.forward(x) - y)
    reg = solver["regularization"] * (x - denoised)
    field = data + reg
    if not torch.isfinite(field).all():
        raise FloatingPointError("Nonfinite terminal RED field")
    return field, data, reg, denoised


def endpoint(x, truth, parts, step):
    field, data, reg, _ = parts
    error = x.double() - truth.double()
    mse = inner(error, error)
    fr = rms(field)
    return {"mse": mse, "sse": mse * x.numel(),
            "psnr": 10 * np.log10(1 / max(mse, 1e-30)),
            "state_rms": rms(x), "field_rms_at_output": fr,
            "field_max_abs_at_output": float(field.abs().max().item()),
            "data_field_rms": rms(data), "regularization_field_rms": rms(reg),
            "next_step_rms": step * fr,
            "relative_field_rms": fr / max(rms(data) + rms(reg), 1e-30)}


def pair_row(states, parts, step, truth):
    left, right = states
    delta = right.double() - left.double()
    dd = parts[1][1].double() - parts[0][1].double()
    dr = parts[1][2].double() - parts[0][2].double()
    df = parts[1][0].double() - parts[0][0].double()
    denoised_delta = parts[1][3].double() - parts[0][3].double()
    distance2 = inner(delta, delta)
    next_states = [x - step * p[0] for x, p in zip(states, parts)]
    next_delta = next_states[1].double() - next_states[0].double()
    after2 = inner(next_delta, next_delta)
    data_term = -2 * step * inner(delta, dd)
    reg_term = -2 * step * inner(delta, dr)
    square_term = step**2 * inner(df, df)
    motion = step * (rms(parts[0][0]) + rms(parts[1][0]))
    return {"output_rms": distance2**.5, "bitwise_equal": torch.equal(left, right),
            "mse_gap": inner(right.double() - truth.double(), right.double() - truth.double())
                       - inner(left.double() - truth.double(), left.double() - truth.double()),
            "distance_over_joint_next_motion": distance2**.5 / max(motion, 1e-30),
            "effective_data_gain": inner(delta, dd) / distance2 if distance2 else None,
            "denoiser_projection_gain": inner(delta, denoised_delta) / distance2 if distance2 else None,
            "directional_curvature": inner(delta, df) / distance2 if distance2 else None,
            "next_distance_ratio": (after2 / distance2)**.5 if distance2 else None,
            "data_distance_change_term": data_term,
            "regularization_distance_change_term": reg_term,
            "squared_step_term": square_term,
            "distance_identity_roundoff": after2 - distance2 - data_term - reg_term - square_term}


@torch.inference_mode()
def run(source, out, max_calls=1024, device_name="cuda", resume_from=None,
        image_count=None, seed_count=None):
    source_spec = read_json(source / "plan.json")
    choices = read_json(source / "history_choices.json")
    resumed = read_json(resume_from / "plan.json") if resume_from else None
    images = resumed["images"] if resumed else source_spec["check_images"]
    seeds = resumed["seeds"] if resumed else source_spec["check_seeds"]
    if image_count is not None:
        if not 1 <= image_count <= len(images):
            raise ValueError("Use a nonempty prefix of the saved image order")
        images = images[:image_count]
    if seed_count is not None:
        if not 1 <= seed_count <= len(seeds):
            raise ValueError("Use a nonempty prefix of the saved seed order")
        seeds = seeds[:seed_count]
    base = resumed["max_calls"] if resumed else 64
    if max_calls <= base:
        raise ValueError("Final call count must exceed the saved starting point")
    if resumed and resumed["repair_source"] != str(source.resolve()):
        raise ValueError("Resume from the same frozen repair source")
    readouts = sorted({base, max_calls, *(k for k in (128, 256, 512, 1024, 2048, 4096, 8192)
                                        if base < k < max_calls)})
    settings = source_spec["settings"]
    kernel = settings["kernels"][0]
    selected = {}
    for noise in source_spec["noise_sigma_255"]:
        group = choices[str(noise)]
        terminal = next(a for a in group["arms"] if a["id"] == "tuned_terminal")
        history = group["performance_selection"]["candidate"]
        if history["history"]["shape"] == "terminal":
            raise ValueError("This probe needs a previously frozen nonzero history")
        selected[str(noise)] = [terminal, {**history, "id": "frozen_history"}]
    observations = len(images) * len(seeds) * len(source_spec["noise_sigma_255"])
    expected = observations * 2 * (max_calls - base + 1 + (0 if resumed else 64))
    out.mkdir(parents=True, exist_ok=False)
    (out / "states").mkdir()
    write_json(out / "plan.json", {
        "repair_source": str(source.resolve()), "settings": settings,
        "images": images, "seeds": seeds,
        "noise_sigma_255": source_spec["noise_sigma_255"], "choices": selected,
        "base_calls": base, "max_calls": max_calls, "readouts": readouts,
        "resume_from": str(resume_from.resolve()) if resumed else None,
        "expected_network_forwards": expected,
        "scope": "Same observations and frozen solver/history as repair confirmation. "
                 "No refitting or label-dependent stopping. Both trajectories use the same "
                 "terminal field after call 64. True endpoint fields are measured, and reused "
                 "for the next update; only the final field costs an extra call. Residual and "
                 "data loss normalization unchanged. Secants are along measured pairs, not "
                 "regional stability certificates. Checkpoints permit continuation without replay.",
    })
    original_mse = {}
    if not resumed:
        with (source / "check_records.csv").open(encoding="utf-8-sig") as f:
            original_mse = {(float(r["noise_255"]), r["image"], int(r["seed"]), r["candidate"]): float(r["mse"])
                            for r in csv.DictReader(f) if int(r["total_calls"]) == 64}
    device = configure_device(device_name)
    denoiser = load_denoiser(settings["network"], device)
    start = time.perf_counter()
    print(f"Frozen terminal extension: {expected} forwards, readouts={readouts}", flush=True)
    with (out / "records.csv").open("x", encoding="utf-8", newline="") as sf, (out / "pairs.csv").open("x", encoding="utf-8", newline="") as pf:
        sw, pw = None, None
        for image in images:
            truth = load_image(image["path"], device, settings["gray_policy"])
            blur = CircularBlur(np.asarray(kernel["values"]), truth)
            for seed in seeds:
                for noise in source_spec["noise_sigma_255"]:
                    candidates = selected[str(noise)]
                    solver = candidates[0]["solver"]
                    step = solver["step"]
                    y = observation(truth, blur, noise, seed, image["source_index"], 0, 0)
                    prefix = f"im{image['source_index']:02d}_s{seed}_n{noise:g}"
                    if resumed:
                        with np.load(resume_from / "states" / f"{prefix}_k{base:05d}.npz") as saved:
                            states = [torch.as_tensor(saved[name], device=device) for name in ("terminal", "history")]
                    else:
                        states = [next(red_trajectory(denoiser, blur, y, scale_sequence(c), solver, {64}))[1]
                                  for c in candidates]
                    parts = [terminal_field(denoiser, blur, y, x, solver) for x in states]
                    key = {"image": image["id"], "seed": seed, "noise_255": noise}
                    k = base
                    while True:
                        if k in readouts:
                            ends = []
                            for name, x, p in zip(("terminal", "history"), states, parts):
                                row = {**key, "family": name, "total_calls": k, **endpoint(x, truth, p, step),
                                       "source_mse_difference_at_64": None}
                                if k == 64 and not resumed:
                                    source_arm = ("tuned_terminal" if name == "terminal" else
                                                  next(a["id"] for a in choices[str(noise)]["arms"]
                                                       if a["solver"] == candidates[1]["solver"]
                                                       and a["history"] == candidates[1]["history"]))
                                    row["source_mse_difference_at_64"] = row["mse"] - original_mse[(noise, image["id"], seed, source_arm)]
                                    if abs(row["source_mse_difference_at_64"]) > 1e-10:
                                        raise ValueError("Replayed 64-call state differs from the saved source MSE")
                                if sw is None:
                                    sw = csv.DictWriter(sf, fieldnames=list(row)); sw.writeheader()
                                sw.writerow(row)
                                ends.append(row)
                            pair = {**key, "total_calls": k, **pair_row(states, parts, step, truth)}
                            if pw is None:
                                pw = csv.DictWriter(pf, fieldnames=list(pair)); pw.writeheader()
                            pw.writerow(pair)
                            sf.flush(); pf.flush()
                            if k in (base, max_calls):
                                np.savez(out / "states" / f"{prefix}_k{k:05d}.npz",
                                         terminal=states[0].cpu().numpy(), history=states[1].cpu().numpy())
                            print(f"{image['id']} seed={seed} noise={noise:g} K={k}: "
                                  f"delta={pair['output_rms']:.6g}, gap={pair['mse_gap']:.6g}, "
                                  f"fields={ends[0]['field_rms_at_output']:.6g}/{ends[1]['field_rms_at_output']:.6g}, "
                                  f"forwards={denoiser.calls}", flush=True)
                        if k == max_calls:
                            break
                        states = [x - step * p[0] for x, p in zip(states, parts)]
                        if not all(torch.isfinite(x).all() for x in states):
                            raise FloatingPointError(f"Nonfinite state at call {k + 1}")
                        parts = [terminal_field(denoiser, blur, y, x, solver) for x in states]
                        k += 1
    write_json(out / "completion.json", {"network_forwards": denoiser.calls,
                                        "expected_network_forwards": expected,
                                        "observations": observations, "max_calls": max_calls,
                                        "seconds_including_metrics_and_checkpoints": time.perf_counter() - start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--image-count", type=int, help="Keep the first saved images; diagnostic subset")
    parser.add_argument("--seed-count", type=int, help="Keep the first saved noise seeds; diagnostic subset")
    args = parser.parse_args()
    run(args.source, args.out, args.max_calls, args.device, args.resume_from,
        args.image_count, args.seed_count)


if __name__ == "__main__":
    main()
