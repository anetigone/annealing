"""Independent local-image evaluation; frozen choices only, no tuning."""
from pathlib import Path
from itertools import combinations

import numpy as np
import torch

from common import (condition_key, configure_device, image_list, read_json,
                    red_trajectory, write_json)
from denoisers import load_denoiser
from problem import CircularBlur, load_image, metrics, observation
from schedules import FAMILIES, schedule


def check_split(development, evaluation, dataset, gray_policy):
    old_paths = {str(Path(v["path"]).resolve()).casefold() for v in development}
    old_names = {Path(v["id"]).stem.casefold() for v in development}
    overlap = [v["id"] for v in evaluation if str(Path(v["path"]).resolve()).casefold() in old_paths
               or Path(v["id"]).stem.casefold() in old_names]
    if overlap:
        raise ValueError(f"Development/evaluation overlap by path or image name: {overlap[:5]}")
    expected = {"bsd68": 68, "set12": 12}.get(dataset)
    if expected is not None and len(evaluation) != expected:
        raise ValueError(f"{dataset} requires all {expected} images; use dataset=custom for a subset")
    if dataset in ("bsd68", "set12") and gray_policy != "require_grayscale":
        raise ValueError("Standard gray benchmark labels require original grayscale files")


def evaluate(calibration, images, out, dataset, device_name, save_states=False, denoiser=None):
    import csv

    calibration, out = Path(calibration), Path(out)
    spec = read_json(calibration / "plan.json")
    frozen = read_json(calibration / "choices.json")
    settings = spec["settings"]
    files = image_list(images)
    check_split(spec["development_images"], files, dataset, settings["gray_policy"])
    device = configure_device(device_name)
    denoiser = denoiser or load_denoiser(settings["network"], device)
    if denoiser.layout != frozen["network_layout"]:
        raise ValueError("Evaluation network layout differs from calibration")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", {
        "settings": settings, "evaluation_images": files, "dataset": dataset,
        "calibration_directory": str(calibration.resolve()), "frozen_choices": frozen,
        "device": str(device), "torch_version": torch.__version__, "save_states": save_states,
        "split_check": "Reject equal paths/image stems; renamed duplicates are not detected. "
                       "Pretraining independence must be checked separately.",
        "metric_scope": "Float images, data_range=1, no border crop; 11x11 sigma=1.5 "
                        "population SSIM on valid centers. Clipped quality is separate. "
                        "Field norm is at the state before the last update, not at its output.",
    })
    if save_states:
        (out / "states").mkdir()
    terminal = settings["solver"]["terminal_sigma_255"] / 255
    total_forwards, failures = 0, 0
    writer = None
    with (out / "records.csv").open("x", encoding="utf-8", newline="") as stream, (out / "distances.csv").open("x", encoding="utf-8", newline="") as distance_stream:
        distance_writer = csv.DictWriter(distance_stream, fieldnames=[
            "image", "kernel", "noise_255", "seed", "budget", "extra_calls",
            "left", "right", "output_rms", "both_finite",
        ])
        distance_writer.writeheader()
        for ii, image in enumerate(files):
            truth = load_image(image["path"], device, settings["gray_policy"])
            for ki, kernel in enumerate(settings["kernels"]):
                blur = CircularBlur(np.asarray(kernel["values"]), truth)
                for ni, noise in enumerate(settings["noise_sigma_255"]):
                    for si, seed in enumerate(settings["evaluation_seeds"]):
                        y = observation(truth, blur, noise, seed, ii, ki, ni)
                        for budget in settings["budgets"]:
                            extra = sorted({0, *settings["tail"]["extra_calls"]}) if budget == settings["tail"]["anchor"] else [0]
                            readouts = {budget + k for k in extra}
                            choices = frozen["conditions"][condition_key(kernel, noise, budget)]
                            references = {}
                            paired_states = {k: {} for k in readouts}
                            for family in FAMILIES:
                                choice = choices[family]
                                sigmas = schedule(choice["candidate"], budget, terminal)
                                sigmas = np.concatenate((sigmas, np.full(max(extra), terminal)))
                                snapshots, reason = {}, ""
                                before = denoiser.calls
                                try:
                                    for k, x, info in red_trajectory(denoiser, blur, y, sigmas,
                                                                   settings["solver"], readouts):
                                        snapshots[k] = (x.detach().cpu(), info)
                                except FloatingPointError as error:
                                    reason = str(error)
                                total_forwards += denoiser.calls - before
                                for additional in extra:
                                    k = budget + additional
                                    valid = k in snapshots
                                    row = {
                                        "image": image["id"], "kernel": kernel["name"],
                                        "noise_255": noise, "seed": seed, "budget": budget,
                                        "extra_calls": additional, "total_calls": k, "family": family,
                                        "status": "ok" if valid else "nonfinite", "failure": "" if valid else reason,
                                        "development_mse": choice["development_mse"],
                                        "old_set12_exposure": dataset == "set12" and Path(image["id"]).stem in ("01", "02", "10"),
                                        "sse": None, "mse": None, "psnr": None, "ssim": None,
                                        "weak_sse": None, "weak_mse": None, "clipped_mse": None,
                                        "clipped_psnr": None, "output_rms_vs_terminal": None,
                                        "seconds": None, "logical_calls": None, "network_forwards": None,
                                        "field_rms_before_last_update": None, "peak_cuda_bytes": None,
                                    }
                                    if valid:
                                        cpu_x, info = snapshots[k]
                                        paired_states[k][family] = cpu_x
                                        x = cpu_x.to(device)
                                        row.update(metrics(x, truth, blur, settings["weak_gain"]))
                                        row.update(info)
                                        if family == "terminal":
                                            references[k] = cpu_x
                                        if k in references:
                                            row["output_rms_vs_terminal"] = float((cpu_x.double()-references[k].double()).square().mean().sqrt().item())
                                        if save_states:
                                            name = f"im{ii:04d}_k{ki}_n{ni}_s{si}_K{budget}_{family}_L{additional}.npy"
                                            np.save(out / "states" / name, cpu_x.numpy())
                                    else:
                                        failures += 1
                                    if writer is None:
                                        writer = csv.DictWriter(stream, fieldnames=list(row))
                                        writer.writeheader()
                                    writer.writerow(row)
                                stream.flush()
                            for additional in extra:
                                for left, right in combinations(FAMILIES, 2):
                                    states = paired_states[budget+additional]
                                    both = left in states and right in states
                                    distance = float((states[left].double()-states[right].double()).square().mean().sqrt().item()) if both else None
                                    distance_writer.writerow({
                                        "image": image["id"], "kernel": kernel["name"], "noise_255": noise,
                                        "seed": seed, "budget": budget, "extra_calls": additional,
                                        "left": left, "right": right, "output_rms": distance, "both_finite": both,
                                    })
                            distance_stream.flush()
            print(f"Evaluated {ii+1}/{len(files)}: {image['id']}", flush=True)
    write_json(out / "completion.json", {"images": len(files), "nonfinite_records": failures,
                                         "total_network_forwards": total_forwards})
    print(f"Evaluation saved: {out / 'records.csv'}", flush=True)
