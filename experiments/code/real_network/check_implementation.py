"""Small CPU checks using synthetic images and a linear denoiser, no downloads."""
import argparse
import copy
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from calibrate import calibrate, plan
from common import read_json, red_trajectory, write_json
from denoisers import Denoiser, DRUNet, FFDNet, load_denoiser
from evaluate import check_split, evaluate
from problem import CircularBlur, metrics
from schedules import FAMILIES, candidate_families, schedule
from summarize import summarize
from workflow import estimate, workflow
from history import history_workflow, select_choices


class LinearDenoiser:
    layout = "synthetic_linear_stub"

    def __init__(self):
        self.calls = 0

    def __call__(self, x, sigma):
        self.calls += 1
        return x / (1 + sigma)


class NonfiniteDenoiser(LinearDenoiser):
    def __call__(self, x, sigma):
        self.calls += 1
        return torch.full_like(x, float("nan"))


def expect_error(kind, function):
    try:
        function()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def check(out):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    checks = []
    reference = torch.randn(1, 1, 17, 19, dtype=torch.float64)
    kernel = np.asarray([[1, 2, 0], [0, 3, 1], [1, 0, 2]], dtype=np.float64)
    kernel /= kernel.sum()
    blur = CircularBlur(kernel, reference)
    other = torch.randn_like(reference)
    lhs = (blur.forward(reference)*other).sum()
    rhs = (reference*blur.adjoint(other)).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=1e-12)
    direct = sum(kernel[i, j] * torch.roll(reference, (i-1, j-1), (-2, -1))
                 for i in range(3) for j in range(3))
    torch.testing.assert_close(blur.forward(reference), direct, atol=1e-12, rtol=1e-12)
    checks.append("periodic convolution orientation and true adjoint")

    x = torch.rand(1, 1, 17, 19)
    for name, model in (("drunet", DRUNet()), ("ffdnet", FFDNet("kair")),
                        ("ffdnet", FFDNet("deepinv"))):
        wrapper = Denoiser(model, name)
        result = wrapper(x, 15/255)
        assert result.shape == x.shape and torch.isfinite(result).all() and wrapper.calls == 1
        if name == "ffdnet":
            weights = out / f"random_ffdnet_{model.layout}.pth"
            torch.save(model.state_dict(), weights)
            loaded = load_denoiser({"name": name, "weights": str(weights)}, "cpu")
            torch.testing.assert_close(loaded(x, 15/255), result, atol=0, rtol=0)
    expect_error(FileNotFoundError, lambda: load_denoiser({"name": "drunet", "weights": str(out / "missing.pth")}, "cpu"))
    checks.append("three untrained network layouts, padding, strict local load and missing-file behavior")

    identity = CircularBlur(np.ones((1, 1)), x)
    solver = {"initialization": "backprojection", "ridge": .01, "step": .25, "regularization": .1}
    fake = LinearDenoiser()
    trajectory = list(red_trajectory(fake, identity, x, [.1], solver, {1}))
    expected = x - .25*.1*(x-x/1.1)
    torch.testing.assert_close(trajectory[0][1], expected, atol=1e-7, rtol=1e-6)
    assert fake.calls == 1 and trajectory[0][2]["network_forwards"] == 1
    measured = metrics(x, x, identity, .05)
    assert measured["sse"] == 0 and abs(measured["ssim"] - 1) < 1e-12
    checks.append("unnormalized RED step, single-call accounting, float MSE/SSIM")

    root = Path(__file__).resolve().parents[3]
    settings = read_json(root / "experiments/configs/drunet_deblur.json")
    settings["kernels"] = [{"name": "small_gaussian", "type": "gaussian", "size": 5, "sigma": 1.0}]
    settings["budgets"] = [8, 12]
    settings["tail"] = {"anchor": 12, "extra_calls": [4, 16]}
    settings["noise_sigma_255"] = [2.55]
    settings["evaluation_seeds"] = [2026111612]
    config = out / "synthetic_config.json"
    write_json(config, settings)
    for budget in settings["budgets"]:
        groups = candidate_families(settings, budget)
        assert len({len(groups[f]) for f in FAMILIES[1:]}) == 1
        for items in groups.values():
            for candidate in items:
                sigmas = schedule(candidate, budget, 5/255)
                assert len(sigmas) == budget and sigmas[-1] == 5/255
                assert np.all((sigmas > 0) & (sigmas <= 50/255))
    checks.append("equal candidate allowances, endpoint histories and common terminal tails")

    for folder in ("development", "evaluation"):
        (out / folder).mkdir()
        for index in range(2):
            rng = np.random.default_rng(index + (0 if folder == "development" else 20))
            array = rng.integers(0, 256, size=(24, 28), dtype=np.uint8)
            Image.fromarray(array).save(out / folder / f"{folder}_{index}.png")
    calibration, evaluation = out / "calibration_run", out / "evaluation_run"
    plan(config, out / "development", calibration)
    frozen = read_json(calibration / "plan.json")
    expect_error(ValueError, lambda: check_split(frozen["development_images"], frozen["development_images"], "custom", "require_grayscale"))
    calibrate(calibration, "cpu", LinearDenoiser())
    evaluate(calibration, out / "evaluation", evaluation, "custom", "cpu", denoiser=LinearDenoiser())
    summarize(evaluation)
    complete = read_json(evaluation / "completion.json")
    assert complete["nonfinite_records"] == 0
    assert complete["total_network_forwards"] == 2*(8+12+16)
    assert complete["reused_trajectories"] == 12
    assert len(read_json(evaluation / "summary.json")["rows"]) == 16
    assert len(read_json(evaluation / "summary.json")["pairwise_comparisons"]) == 24
    assert all(row["mean_mse_left_minus_right"] == 0
               for row in read_json(evaluation / "summary.json")["pairwise_comparisons"])
    expect_error(FileExistsError, lambda: calibrate(calibration, "cpu", LinearDenoiser()))
    checks.append("plan -> calibration -> independent evaluation -> paired summary, no overwrite")
    failed = out / "nonfinite_run"
    evaluate(calibration, out / "evaluation", failed, "custom", "cpu", denoiser=NonfiniteDenoiser())
    summarize(failed)
    assert read_json(failed / "completion.json")["nonfinite_records"] == 32
    assert all(row["means"] is None for row in read_json(failed / "summary.json")["rows"])
    checks.append("nonfinite trajectories retained with no misleading finite-only aggregate")
    fast = read_json(root / "experiments/configs/drunet_fast.json")
    assert estimate(fast, 12, 68)["total_calls_without_reuse"] == 56000
    settings["network"] = {"name": "ffdnet", "weights": str((out / "random_ffdnet_kair.pth").resolve())}
    flow_config = out / "workflow_config.json"
    write_json(flow_config, settings)
    with patch("workflow.calibrate", side_effect=lambda *args: calibrate(*args, denoiser=LinearDenoiser())), \
            patch("workflow.evaluate", side_effect=lambda *args: evaluate(*args, denoiser=LinearDenoiser())):
        workflow(flow_config, out / "development", out / "evaluation", out / "single_command",
                 "custom", "cpu")
    assert (out / "single_command/workflow_complete.json").is_file()
    checks.append("exact trajectory reuse preserves paired errors; fast cost estimate and single-command workflow")
    original_plan = (calibration / "plan.json").read_text(encoding="utf-8")
    original_choices = (calibration / "choices.json").read_text(encoding="utf-8")
    history_out = out / "history_diagnostic"
    history_workflow(calibration, out / "evaluation", history_out, "custom", "cpu",
                     denoiser=LinearDenoiser(), anchor=12, extra_calls=[4, 8])
    restricted = read_json(history_out / "calibration/choices.json")
    for condition in restricted["conditions"].values():
        for family in FAMILIES[1:]:
            assert condition[family]["candidate"]["prefix"] > 0
            assert condition[family]["source_unrestricted_choice"]["candidate"]["prefix"] == 0
    assert (calibration / "plan.json").read_text(encoding="utf-8") == original_plan
    assert (calibration / "choices.json").read_text(encoding="utf-8") == original_choices
    with (evaluation / "records.csv").open(newline="") as stream:
        old_records = {(r["image"], r["budget"], r["extra_calls"]): r for r in csv.DictReader(stream)
                       if r["family"] == "terminal"}
    with (history_out / "evaluation/records.csv").open(newline="") as stream:
        new_records = list(csv.DictReader(stream))
    for row in new_records:
        key = (row["image"], row["budget"], row["extra_calls"])
        if row["family"] == "terminal" and key in old_records:
            assert row["mse"] == old_records[key]["mse"]
    sample = {int(r["extra_calls"]): float(r["output_rms_vs_terminal"]) for r in new_records
              if r["image"] == "evaluation_0.png" and r["family"] == "geometric" and r["budget"] == "12"}
    assert sample[0] > sample[4] > sample[8] > 0
    costs = read_json(history_out / "cost_estimate.json")
    assert costs["additional_calibration_forwards"] == 0
    assert costs["actual_evaluation_forwards_if_finite"] == read_json(history_out / "evaluation/completion.json")["total_network_forwards"]
    with (calibration / "calibration/scores.csv").open(newline="") as stream:
        saved_scores = list(csv.DictReader(stream))
    terminal_only = copy.deepcopy(frozen)
    for groups in terminal_only["candidates"].values():
        for family in FAMILIES[1:]:
            groups[family] = groups[family][:1]
    expect_error(ValueError, lambda: select_choices(terminal_only, read_json(calibration / "choices.json"),
                                                   saved_scores, [8, 12]))
    checks.append("saved-score nonzero selection, source immutability, shared observations/tail and no zero-prefix fallback")
    write_json(out / "checks.json", {"passed": checks, "scope": "Synthetic CPU plumbing and untrained layouts only; no pretrained weights or real datasets used."})
    print("PASS: " + "; ".join(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New synthetic-check directory")
    check(parser.parse_args().out)
