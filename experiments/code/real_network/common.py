"""Small local experiment helpers; no model, image, or dataset downloads."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_list(folder):
    root = Path(folder).resolve()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    if not root.is_dir() or not paths:
        raise FileNotFoundError(f"No local images found: {root}")
    return [{"id": p.relative_to(root).as_posix(), "path": str(p)} for p in paths]


def normalize_settings(settings):
    if settings["network"]["name"] not in ("drunet", "ffdnet"):
        raise ValueError("network.name must be drunet or ffdnet")
    settings["network"]["weights"] = str(Path(settings["network"]["weights"]).resolve())
    budgets = settings["budgets"]
    if not budgets or any(not isinstance(k, int) or k < 4 for k in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be distinct integers >= 4")
    solver, search = settings["solver"], settings["search"]
    numbers = [solver["step"], solver["regularization"], solver["terminal_sigma_255"],
               settings["weak_gain"], search["mse_tolerance"], settings["minimum_terminal_fraction"]]
    if not all(np.isfinite(numbers)) or solver["step"] <= 0 or solver["regularization"] < 0:
        raise ValueError("Invalid finite solver parameters")
    terminal = solver["terminal_sigma_255"]
    if not 0 < terminal <= 50 or not 0 < settings["minimum_terminal_fraction"] < 1:
        raise ValueError("Invalid terminal sigma or terminal fraction")
    if not 0 <= settings["weak_gain"] <= 1 or search["mse_tolerance"] < 0:
        raise ValueError("Invalid weak spectral threshold or MSE tolerance")
    if not search["sigma_255"] or any(not terminal < v <= 50 for v in search["sigma_255"]):
        raise ValueError("Candidate sigmas must be above terminal and at most 50 (8-bit units)")
    if search["max_candidates"] < 5 or any(not 0 < f < 1 for f in search["prefix_fractions"]):
        raise ValueError("Use max_candidates >= 5 and prefix fractions in (0,1)")
    if not search["prefix_fractions"] or not search["block_lengths"] or any(b < 1 for b in search["block_lengths"]):
        raise ValueError("Nonempty positive prefix/block candidates required")
    if solver["initialization"] not in ("backprojection", "ridge") or solver["ridge"] <= 0:
        raise ValueError("Invalid initialization or positive ridge parameter")
    if settings["gray_policy"] not in ("require_grayscale", "pillow_luma"):
        raise ValueError("Unknown grayscale policy")
    if settings["tail"]["anchor"] not in budgets or any(not isinstance(k, int) or k <= 0 for k in settings["tail"]["extra_calls"]):
        raise ValueError("Tail anchor must be a budget, with positive integer extra calls")
    names = [spec["name"] for spec in settings["kernels"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Need uniquely named kernels")
    for spec in settings["kernels"]:
        if spec["type"] != "gaussian":
            spec["path"] = str(Path(spec["path"]).resolve())
    for key in ("calibration_seeds", "evaluation_seeds"):
        if not settings[key] or any(not isinstance(s, int) or s < 0 for s in settings[key]):
            raise ValueError("Need nonnegative integer seeds")
    if set(settings["calibration_seeds"]) & set(settings["evaluation_seeds"]):
        raise ValueError("Calibration and evaluation noise seeds must be separate")
    if not settings["noise_sigma_255"] or any(not np.isfinite(v) or v < 0 for v in settings["noise_sigma_255"]):
        raise ValueError("Invalid observation noise")
    return settings


def configure_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set --device cpu explicitly")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def condition_key(kernel, noise, budget):
    return f"{kernel['name']}|{noise:g}|{budget}"


@torch.inference_mode()
def red_trajectory(denoiser, blur, y, sigmas, solver, readouts):
    """Snapshot field norm refers to x_(k-1); no extra denoiser calls."""
    x = blur.initialize(y, solver)
    synchronize(x.device)
    if x.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(x.device)
    start, calls_before, peak_so_far = time.perf_counter(), denoiser.calls, 0
    for k, sigma in enumerate(sigmas, 1):
        field = blur.adjoint(blur.forward(x) - y) + solver["regularization"] * (x - denoiser(x, float(sigma)))
        updated = x - solver["step"] * field
        if not torch.isfinite(updated).all():
            raise FloatingPointError(f"Nonfinite RED state at call {k}")
        x = updated
        if k in readouts:
            synchronize(x.device)
            peak = torch.cuda.max_memory_allocated(x.device) if x.device.type == "cuda" else 0
            peak_so_far = max(peak_so_far, peak)
            snapshot = x.clone()
            synchronize(x.device)
            paused_at = time.perf_counter()
            yield k, snapshot, {
                "seconds": paused_at-start,
                "logical_calls": k, "network_forwards": denoiser.calls-calls_before,
                "field_rms_before_last_update": float(field.double().square().mean().sqrt().item()),
                "peak_cuda_bytes": peak_so_far,
            }
            synchronize(x.device)
            start += time.perf_counter() - paused_at
            if x.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(x.device)
