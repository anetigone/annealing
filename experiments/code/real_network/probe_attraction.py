"""Probe local return around a saved pair of terminal near-stationary states."""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from common import configure_device, read_json, write_csv, write_json
from denoisers import load_denoiser
from problem import CircularBlur, load_image, observation
from probe_terminal import inner, rms, terminal_field


@torch.inference_mode()
def run(source, out, steps=1024, noise=7.65):
    spec = read_json(source / "plan.json")
    image, seed = spec["images"][0], spec["seeds"][0]
    solver = spec["choices"][str(noise)][0]["solver"]
    settings, base = spec["settings"], spec["max_calls"]
    device = configure_device("cuda")
    denoiser = load_denoiser(settings["network"], device)
    truth = load_image(image["path"], device, settings["gray_policy"])
    blur = CircularBlur(np.asarray(settings["kernels"][0]["values"]), truth)
    y = observation(truth, blur, noise, seed, image["source_index"], 0, 0)
    prefix = f"im{image['source_index']:02d}_s{seed}_n{noise:g}"
    checkpoint = source / "states" / f"{prefix}_k{base:05d}.npz"
    with np.load(checkpoint) as saved:
        terminal, history = (torch.as_tensor(saved[n], device=device) for n in ("terminal", "history"))
    delta = history - terminal
    if rms(delta) == 0:
        raise ValueError("The saved endpoints coincide")
    fraction = .25
    states = {"terminal": terminal, "history": history,
              "terminal_toward": terminal + fraction * delta,
              "terminal_away": terminal - fraction * delta,
              "history_toward": history - fraction * delta,
              "history_away": history + fraction * delta,
              "midpoint": terminal + .5 * delta}
    readouts = sorted({0, steps, *(k for k in (128, 512) if k < steps)})
    levels = sorted({2.5, 5., solver["terminal_sigma_255"], 25., 49.})
    expected = len(states) * (steps + len(levels))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", {
        "terminal_source": str(source.resolve()), "checkpoint": str(checkpoint.resolve()),
        "image": image, "seed": seed, "noise_255": noise, "solver": solver,
        "base_calls": base, "additional_steps": steps, "readouts": readouts,
        "perturbation_fraction_of_endpoint_difference": fraction,
        "initial_separation_rms": rms(delta), "families": list(states),
        "force_probe_sigma_255": levels, "expected_network_forwards": expected,
        "scope": "Exploratory mechanism probe at the first saved image/seed and specified "
                 "noise. Both saved endpoints, plus/minus quarter of their difference, and "
                 "their midpoint. No truth-dependent perturbations, refitting or stopping. "
                 "Return distances use the simultaneously continued unperturbed endpoints. "
                 "These sampled directions do not certify entire attraction basins.",
    })
    start = time.perf_counter()
    parts = {name: terminal_field(denoiser, blur, y, x, solver) for name, x in states.items()}
    forces = []
    for name, x in states.items():
        for level in levels:
            field = (parts[name][0] if level == solver["terminal_sigma_255"] else
                     terminal_field(denoiser, blur, y, x, {**solver, "terminal_sigma_255": level})[0])
            coordinate_change = -solver["step"] * inner(field, delta) / inner(delta, delta)
            data_change = -solver["step"] * inner(parts[name][1], delta) / inner(delta, delta)
            displacement = -solver["step"] * field.double()
            forces.append({"family": name, "sigma_255": level,
                           "projected_next_coordinate_change": coordinate_change,
                           "data_coordinate_change": data_change,
                           "regularization_coordinate_change": coordinate_change - data_change,
                           "orthogonal_step_rms": rms(displacement - coordinate_change * delta.double()),
                           "field_rms": rms(field)})
    write_csv(out / "scale_forces.csv", forces)
    print(f"Attraction probe: {expected} forwards, starting separation={rms(delta):.6g}", flush=True)
    with (out / "records.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = None
        for k in range(steps + 1):
            if k in readouts:
                live_delta = states["history"].double() - states["terminal"].double()
                denom = inner(live_delta, live_delta)
                for name, x in states.items():
                    offset = x.double() - states["terminal"].double()
                    coordinate = inner(offset, live_delta) / denom if denom else None
                    orthogonal = offset - coordinate * live_delta if coordinate is not None else offset
                    error = x.double() - truth.double()
                    row = {"family": name, "additional_calls": k,
                           "mse": inner(error, error), "field_rms_at_output": rms(parts[name][0]),
                           "rms_to_live_terminal": rms(offset),
                           "rms_to_live_history": rms(x.double() - states["history"].double()),
                           "rms_to_start": rms(x.double() - (terminal if name == "terminal" else history).double())
                                           if name in ("terminal", "history") else None,
                           "live_endpoint_separation_rms": denom**.5,
                           "coordinate_between_live_endpoints": coordinate,
                           "orthogonal_rms": rms(orthogonal)}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row)
                stream.flush()
                print(f"additional={k}, separation={rms(live_delta):.6g}, "
                      f"terminal_toward_return={rms(states['terminal_toward'] - states['terminal']):.6g}, "
                      f"history_toward_return={rms(states['history_toward'] - states['history']):.6g}, "
                      f"base_fields={rms(parts['terminal'][0]):.6g}/{rms(parts['history'][0]):.6g}, "
                      f"forwards={denoiser.calls}", flush=True)
            if k == steps:
                break
            states = {name: x - solver["step"] * parts[name][0] for name, x in states.items()}
            parts = {name: terminal_field(denoiser, blur, y, x, solver) for name, x in states.items()}
    np.savez(out / "final_states.npz", **{name: x.cpu().numpy() for name, x in states.items()})
    write_json(out / "completion.json", {
        "network_forwards": denoiser.calls, "expected_network_forwards": expected,
        "seconds": time.perf_counter() - start,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--noise-255", type=float, default=7.65)
    args = parser.parse_args()
    run(args.source, args.out, args.steps, args.noise_255)


if __name__ == "__main__":
    main()
