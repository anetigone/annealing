"""Work estimates and a foreground local-resource workflow; never downloads."""
from pathlib import Path
import math

from PIL import Image

from calibrate import plan, calibrate
from common import image_list, normalize_settings, read_json, write_json
from evaluate import check_split, evaluate
from schedules import FAMILIES, candidate_families, schedule
from summarize import summarize


def estimate(settings, development_count, evaluation_count, milliseconds_per_call=28):
    if min(development_count, evaluation_count) <= 0 or not math.isfinite(milliseconds_per_call) or milliseconds_per_call <= 0:
        raise ValueError("Positive image counts and per-call time required")
    terminal = settings["solver"]["terminal_sigma_255"] / 255
    logical, unique, evaluation_per_family = 0, 0, 0
    for budget in settings["budgets"]:
        groups = candidate_families(settings, budget)
        sequences = [tuple(schedule(item, budget, terminal)) for items in groups.values() for item in items]
        logical += budget * len(sequences)
        unique += budget * len(set(sequences))
        tail = max([0, *settings["tail"]["extra_calls"]]) if budget == settings["tail"]["anchor"] else 0
        evaluation_per_family += budget + tail
    conditions = len(settings["kernels"]) * len(settings["noise_sigma_255"])
    development_factor = development_count * conditions * len(settings["calibration_seeds"])
    evaluation_factor = evaluation_count * conditions * len(settings["evaluation_seeds"])
    calibration_calls = unique * development_factor
    evaluation_min = evaluation_per_family * evaluation_factor
    evaluation_max = evaluation_min * len(FAMILIES)
    return {
        "development_images": development_count, "evaluation_images": evaluation_count,
        "conditions": conditions, "calibration_calls_without_reuse": logical * development_factor,
        "calibration_calls_with_reuse": calibration_calls,
        "evaluation_forward_range": [evaluation_min, evaluation_max],
        "total_forward_range": [calibration_calls + evaluation_min, calibration_calls + evaluation_max],
        "total_calls_without_reuse": logical * development_factor + evaluation_max,
        "assumed_milliseconds_per_call": milliseconds_per_call,
        "inference_minutes_range": [(calibration_calls + v) * milliseconds_per_call / 60000
                                    for v in (evaluation_min, evaluation_max)],
        "scope": "Identical complete scale sequences reuse exact scores/trajectories; all "
                 "candidate sets and logical per-method budgets are retained. Assumes finite "
                 "trajectories. Timing excludes I/O/metrics/setup and varies with image shape. "
                 "Evaluation reuse depends on the choices, which are not yet known.",
    }


def workflow(config, development_images, evaluation_images, out, dataset, device,
             milliseconds_per_call=28, save_states=False):
    import json

    settings = normalize_settings(read_json(config))
    development, assessment = image_list(development_images), image_list(evaluation_images)
    check_split(development, assessment, dataset, settings["gray_policy"])
    if not Path(settings["network"]["weights"]).is_file():
        raise FileNotFoundError("Local weights missing; see experiments/real_network.md")
    # Readability checks use no restoration score and do not select parameters.
    for entry in development + assessment:
        with Image.open(entry["path"]) as image:
            if settings["gray_policy"] == "require_grayscale" and image.mode != "L":
                raise ValueError(f"Expected original grayscale: {entry['path']}")
            if min(image.size) < 11:
                raise ValueError(f"Image smaller than SSIM window: {entry['path']}")
            image.verify()
    costs = estimate(settings, len(development), len(assessment), milliseconds_per_call)
    print("Work estimate before execution:", json.dumps(costs, indent=2), flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "settings.json", settings)
    write_json(out / "cost_estimate.json", costs)
    plan(out / "settings.json", development_images, out / "calibration")
    calibrate(out / "calibration", device)
    evaluate(out / "calibration", evaluation_images, out / "evaluation", dataset, device, save_states)
    summarize(out / "evaluation")
    write_json(out / "workflow_complete.json", {"summary": str((out / "evaluation/summary.json").resolve())})
    print(f"Workflow complete: {out / 'evaluation/summary.json'}", flush=True)
