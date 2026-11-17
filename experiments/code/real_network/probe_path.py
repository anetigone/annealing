"""Three fixed scale interventions from one actual state, then a common terminal tail."""
import argparse
import csv
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import torch

from common import configure_device, read_json, red_trajectory, write_csv, write_json
from denoisers import load_denoiser
from problem import CircularBlur, load_image, observation
from probe_terminal import endpoint, inner, pair_row, rms, terminal_field


START = 32
LENGTH = 16
LEVELS = (10., 25., 49.)


def action_sigma(level, k):
    return level if START <= k < START + LENGTH else 10.


def prepare(source, out, max_calls=None):
    saved = read_json(source / "plan.json")
    noise, image, seed = 7.65, saved["images"][0], saved["seeds"][0]
    candidates = saved["choices"][str(noise)]
    solver = candidates[0]["solver"]
    history = candidates[1]["history"]
    if (solver["terminal_sigma_255"] != 10 or history["shape"] != "pulse"
            or history["start"] != START or history["length"] != LENGTH
            or history["sigma_255"] != 25):
        raise ValueError("Use the saved sigma10 terminal / sigma25 pulse experiment")
    final = saved["max_calls"] if max_calls is None else max_calls
    if final < 64:
        raise ValueError("Keep at least 16 common terminal updates; max-calls must be >=64")
    reference_calls = saved["max_calls"]
    prefix = f"im{image['source_index']:02d}_s{seed}_n{noise:g}"
    checkpoint = source / "states" / f"{prefix}_k{reference_calls:05d}.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Saved reference endpoints missing: {checkpoint}")
    readouts = sorted({*range(START, START + LENGTH + 1), final,
                       *(k for k in (64, 128, 256, 512, 1024, 2048, 4096, 8192) if k <= final)})
    restoration = START + len(LEVELS) * (final - START)
    extra_terminal = (len(LEVELS) - 1) * sum(START < k < START + LENGTH for k in readouts)
    diagnostics = len(LEVELS) + extra_terminal
    spec = {"terminal_source": str(source.resolve()), "settings": saved["settings"],
            "image": image, "seed": seed, "noise_255": noise, "solver": solver,
            "reference_checkpoint": str(checkpoint.resolve()), "reference_calls": reference_calls,
            "shared_prefix_calls": START, "intervention_calls": LENGTH,
            "intervention_sigma_255": list(LEVELS), "max_calls": final, "readouts": readouts,
            "logical_restoration_calls_per_branch": final,
            "restoration_forwards_with_shared_prefix": restoration,
            "extra_diagnostic_forwards": diagnostics,
            "expected_network_forwards": restoration + diagnostics,
            "scope": "Same first saved image, noise seed, observation and state after 32 sigma10 "
                     "updates. Each branch then receives 16 fixed calls at sigma10/25/49, followed "
                     "by sigma10 to the same total budget. No tuning or label-dependent control. "
                     "Diagnostic terminal fields during the pulse do not drive the updates. "
                     "Coordinates and nearest-reference distances describe the full trajectories; "
                     "they do not certify a one-dimensional channel, a basin, or arbitrary switching "
                     "barriers. References are finite-time saved near-stationary states."}
    out.mkdir(parents=True, exist_ok=False)
    (out / "states").mkdir()
    write_json(out / "plan.json", spec)
    return spec


def projection(x, action, terminal, references, step):
    u, v = references
    delta = v.double() - u.double()
    norm2 = inner(delta, delta)
    offset = x.double() - u.double()
    coordinate = inner(offset, delta) / norm2
    dist_u, dist_v = rms(offset), rms(x.double() - v.double())
    action_move = -step * inner(action[0], delta) / norm2
    terminal_move = -step * inner(terminal[0], delta) / norm2
    data_move = -step * inner(action[1], delta) / norm2
    return {"coordinate_between_references": coordinate,
            "orthogonal_rms_to_reference_line": rms(offset - coordinate * delta),
            "rms_to_terminal_reference": dist_u, "rms_to_history_reference": dist_v,
            "closest_reference": "terminal" if dist_u <= dist_v else "history",
            "closest_distance_over_reference_separation": min(dist_u, dist_v) / norm2**.5,
            "action_field_rms": rms(action[0]), "action_coordinate_change": action_move,
            "terminal_coordinate_change": terminal_move,
            "data_coordinate_change": data_move,
            "action_regularization_coordinate_change": action_move - data_move,
            "action_orthogonal_step_rms": rms(-step * action[0].double() - action_move * delta),
            "terminal_orthogonal_step_rms": rms(-step * terminal[0].double() - terminal_move * delta)}


@torch.inference_mode()
def run(spec, out, device_name="cuda"):
    settings, solver = spec["settings"], spec["solver"]
    device = configure_device(device_name)
    denoiser = load_denoiser(settings["network"], device)
    image, seed, noise = spec["image"], spec["seed"], spec["noise_255"]
    truth = load_image(image["path"], device, settings["gray_policy"])
    blur = CircularBlur(np.asarray(settings["kernels"][0]["values"]), truth)
    y = observation(truth, blur, noise, seed, image["source_index"], 0, 0)
    with np.load(spec["reference_checkpoint"]) as saved:
        references = [torch.as_tensor(saved[name], device=device) for name in ("terminal", "history")]
    separation = rms(references[1] - references[0])
    if not separation:
        raise ValueError("Saved reference endpoints coincide")
    start_time = time.perf_counter()
    common = next(red_trajectory(denoiser, blur, y, np.full(START, 10 / 255), solver, {START}))[1]
    np.savez(out / "states/common_k00032.npz", state=common.cpu().numpy())
    states = {level: common.clone() for level in LEVELS}
    final_rows, final_pairs = [], []
    print(f"Directed path probe: {spec['expected_network_forwards']} forwards; "
          f"shared prefix={START}, pulse={LENGTH}, total={spec['max_calls']}", flush=True)
    with (out / "records.csv").open("x", encoding="utf-8", newline="") as sf, (out / "pairs.csv").open("x", encoding="utf-8", newline="") as pf:
        sw, pw = None, None
        for k in range(START, spec["max_calls"] + 1):
            parts = {level: terminal_field(denoiser, blur, y, x,
                                          {**solver, "terminal_sigma_255": action_sigma(level, k)})
                     for level, x in states.items()}
            if k in spec["readouts"]:
                terminal_parts = {}
                descriptions = []
                for level, x in states.items():
                    if action_sigma(level, k) == 10:
                        terminal_parts[level] = parts[level]
                    elif k == START:
                        terminal_parts[level] = parts[10.]
                    else:
                        terminal_parts[level] = terminal_field(denoiser, blur, y, x, solver)
                    row = {"image": image["id"], "seed": seed, "noise_255": noise,
                           "intervention_sigma_255": level, "total_calls": k,
                           "completed_intervention_calls": min(LENGTH, k - START),
                           "next_action_sigma_255": action_sigma(level, k),
                           **endpoint(x, truth, terminal_parts[level], solver["step"]),
                           **projection(x, parts[level], terminal_parts[level], references, solver["step"])}
                    if sw is None:
                        sw = csv.DictWriter(sf, fieldnames=list(row)); sw.writeheader()
                    sw.writerow(row)
                    if k in (START + LENGTH, 64, spec["max_calls"]):
                        np.savez(out / "states" / f"sigma{level:g}_k{k:05d}.npz", state=x.cpu().numpy())
                    if k == spec["max_calls"]:
                        final_rows.append(row)
                    descriptions.append(f"s{level:g}:dT={row['rms_to_terminal_reference']:.5g},"
                                        f"dH={row['rms_to_history_reference']:.5g},"
                                        f"F={row['field_rms_at_output']:.5g}")
                for left, right in combinations(LEVELS, 2):
                    pair = pair_row([states[left], states[right]],
                                    [terminal_parts[left], terminal_parts[right]], solver["step"], truth)
                    pair["common_terminal_next_distance_ratio"] = pair.pop("next_distance_ratio")
                    pair["terminal_field_secant_projection"] = pair.pop("directional_curvature")
                    row = {"total_calls": k, "left_sigma_255": left, "right_sigma_255": right,
                           "actual_next_updates_both_terminal": k >= START + LENGTH, **pair}
                    if pw is None:
                        pw = csv.DictWriter(pf, fieldnames=list(row)); pw.writeheader()
                    pw.writerow(row)
                    if k == spec["max_calls"]:
                        final_pairs.append(row)
                sf.flush(); pf.flush()
                if k % 4 == 0:
                    print(f"K={k} | " + " | ".join(descriptions) + f" | forwards={denoiser.calls}", flush=True)
            elif k % 256 == 0:
                print(f"K={k}; forwards={denoiser.calls}", flush=True)
            if k == spec["max_calls"]:
                break
            states = {level: x - solver["step"] * parts[level][0] for level, x in states.items()}
            if not all(torch.isfinite(x).all() for x in states.values()):
                raise FloatingPointError(f"Nonfinite state after call {k + 1}")
    write_csv(out / "summary.csv", final_rows)
    write_csv(out / "final_pairs.csv", final_pairs)
    if denoiser.calls != spec["expected_network_forwards"]:
        raise AssertionError("Actual forward count differs from the frozen plan")
    write_json(out / "completion.json", {"network_forwards": denoiser.calls,
                                         "expected_network_forwards": spec["expected_network_forwards"],
                                         "logical_calls_per_branch": spec["max_calls"],
                                         "reference_separation_rms": separation,
                                         "seconds": time.perf_counter() - start_time})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/results/drunet_terminal_long"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    spec = prepare(args.source, args.out, args.max_calls)
    print(f"Planned forwards: {spec['expected_network_forwards']}", flush=True)
    if not args.plan_only:
        run(spec, args.out, args.device)


if __name__ == "__main__":
    main()
