"""Small development-only force and trajectory diagnostics, with local assets."""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import (condition_key, configure_device, read_json, red_trajectory,
                    write_csv, write_json)
from denoisers import load_denoiser
from problem import CircularBlur, load_image, observation
from schedules import FAMILIES, schedule


def inner(a, b):
    return float((a.double()*b.double()).mean().item())


def rms(a):
    return inner(a, a)**.5


def weak_part(x, blur, threshold):
    spectrum = torch.fft.fft2(x.double(), norm="ortho")
    return torch.fft.ifft2(spectrum*(blur.gain <= threshold), norm="ortho").real


def region_means(values):
    margin = min(32, min(values.shape[-2:])//4)
    interior = values[..., margin:-margin, margin:-margin]
    border_sum = values.sum()-interior.sum()
    return {"border_width": margin, "interior_mean": float(interior.mean().item()),
            "border_mean": float((border_sum/(values.numel()-interior.numel())).item())}


def state_metrics(x, truth, blur, threshold):
    error = x.double()-truth.double()
    weak = weak_part(error, blur, threshold)
    regions = region_means(error.square())
    return {"mse": inner(error, error),
            "weak_mse": inner(weak, weak),
            "border_width": regions["border_width"],
            "interior_mse": regions["interior_mean"], "border_mse": regions["border_mean"],
            "rms_outside_0_1": rms(x.double()-x.double().clamp(0, 1))}


@torch.inference_mode()
def trajectory(denoiser, blur, y, sigmas, solver):
    states, denoised = [blur.initialize(y, solver)], []
    for sigma in sigmas:
        x = states[-1]
        d = denoiser(x, float(sigma))
        field = blur.adjoint(blur.forward(x)-y)+solver["regularization"]*(x-d)
        updated = x-solver["step"]*field
        if not torch.isfinite(updated).all():
            raise FloatingPointError("Nonfinite diagnostic state")
        denoised.append(d)
        states.append(updated)
    return states, denoised


@torch.inference_mode()
def force_rows(denoiser, states, denoised, truth, blur, y, solver, threshold):
    rows = []
    gamma, lam = solver["step"], solver["regularization"]
    terminal = solver["terminal_sigma_255"]
    levels = sorted({terminal, 10., 25., 49.})
    for k in (0, 16, 64, 128):
        x = states[k]
        d_low = denoised[k] if k < len(denoised) else denoiser(x, terminal/255)
        low_residual = x.double()-d_low.double()
        data = blur.adjoint(blur.forward(x)-y)
        low_next = x-gamma*(data+lam*(x-d_low))
        error_next = low_next.double()-truth.double()
        for level in levels:
            d = d_low if level == terminal else denoiser(x, level/255)
            if not torch.isfinite(d).all():
                raise FloatingPointError(f"Nonfinite scale probe at state {k}, sigma={level}")
            residual = x.double()-d.double()
            displacement = gamma*lam*(d.double()-d_low.double())
            scale_fit = inner(residual, low_residual)/max(inner(low_residual, low_residual), 1e-30)
            orthogonal = residual-scale_fit*low_residual
            cross, squared = 2*inner(error_next, displacement), inner(displacement, displacement)
            regions = region_means(2*error_next*displacement+displacement.square())
            w_error, w_displacement = weak_part(error_next, blur, threshold), weak_part(displacement, blur, threshold)
            # Both candidates start at the very same state. This is a local counterfactual.
            rows.append({"state_call": k, "sigma_255": level, "data_field_rms": rms(data),
                         "regularization_field_rms": lam*rms(residual),
                         "regularization_over_data_rms": lam*rms(residual)/max(rms(data), 1e-30),
                         "residual_multiple_of_terminal_fit": scale_fit,
                         "residual_orthogonal_energy_fraction": inner(orthogonal, orthogonal)/max(inner(residual, residual), 1e-30),
                         "scale_displacement_rms": rms(displacement),
                         "one_step_mse_gap_vs_terminal": cross+squared,
                         "one_step_cross_term": cross, "one_step_squared_displacement": squared,
                         "one_step_weak_mse_gap": 2*inner(w_error, w_displacement)+inner(w_displacement, w_displacement),
                         "border_width": regions["border_width"],
                         "one_step_interior_mse_gap": regions["interior_mean"],
                         "one_step_border_mse_gap": regions["border_mean"],
                         "regularization_error_inner_product": inner(x.double()-truth.double(), lam*residual)})
    return rows


def transport_row(k, states, denoised, reference, ref_denoised, blur, solver, threshold):
    delta = states[k].double()-reference[k].double()
    delta_d = denoised[k].double()-ref_denoised[k].double()
    b_delta = torch.fft.ifft2(torch.fft.fft2(delta)*blur.gain.double()).real
    g_delta = b_delta+solver["regularization"]*(delta-delta_d)
    gamma, lam = solver["step"], solver["regularization"]
    before = inner(delta, delta)
    after_delta = states[k+1].double()-reference[k+1].double()
    after = inner(after_delta, after_delta)
    data_term = -2*gamma*inner(delta, b_delta)
    denoiser_term = -2*gamma*lam*inner(delta, delta-delta_d)
    step_term = gamma**2*inner(g_delta, g_delta)
    denominator = before if before else 1
    weak = weak_part(delta, blur, threshold)
    return {"from_call": k, "before_squared_distance": before,
            "after_squared_distance": after, "distance_ratio": (after/before)**.5 if before else None,
            "data_distance_change_term": data_term,
            "denoiser_distance_change_term": denoiser_term,
            "squared_step_term": step_term,
            "identity_roundoff_residual": after-before-data_term-denoiser_term-step_term,
            "effective_data_gain": inner(delta, b_delta)/denominator if before else None,
            "denoiser_secant_projection_gain": inner(delta, delta_d)/denominator if before else None,
            "weak_displacement_energy_fraction": inner(weak, weak)/denominator if before else None}


@torch.inference_mode()
def probe(source, out, device_name, image_count, noises):
    spec, frozen = read_json(source/"plan.json"), read_json(source/"choices.json")
    settings, solver = spec["settings"], spec["settings"]["solver"]
    if len(settings["kernels"]) != 1 or len(settings["calibration_seeds"]) != 1:
        raise ValueError("This bounded diagnostic expects one source kernel and development noise seed")
    if solver["terminal_sigma_255"] != 5 or 64 not in settings["budgets"]:
        raise ValueError("This diagnostic addresses the frozen sigma=5, K=64 experiment")
    if not 1 <= image_count <= 12:
        raise ValueError("Use 1--12 source development images")
    files = spec["development_images"][:image_count]
    if len(files) != image_count:
        raise ValueError("Not enough source development images")
    source_noise = settings["noise_sigma_255"][0]
    kernel = settings["kernels"][0]
    choices = frozen["conditions"][condition_key(kernel, source_noise, 64)]
    terminal = solver["terminal_sigma_255"]/255
    sequences = {f: np.r_[schedule(choices[f]["candidate"], 64, terminal),
                         np.full(64, terminal)] for f in FAMILIES}
    if len({tuple(seq) for seq in sequences.values()}) != 4:
        raise ValueError("Use the frozen nonzero-history calibration, not the terminal-only choices")
    device = configure_device(device_name)
    denoiser = load_denoiser(settings["network"], device)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"plan.json", {
        "source_calibration": str(source.resolve()), "development_images": files,
        "settings": settings, "frozen_choices": choices, "noise_sigma_255": noises,
        "scope": "Posthoc causal diagnostics only. First images in the existing development order, no BSD68 or parameter selection. Across noise amplitudes reuse the same standard-normal field. All histories retain source choices, including at the changed noise amplitude.",
        "counterfactuals": "Same-state scale probes, data-only trajectory, and common-terminal secant decomposition. Residual and data loss normalization unchanged. No model or data downloads.",
        "expected_network_forwards": image_count*(len(noises)*(4*128+13+16)+1),
    })
    endpoint_rows, forces, transports, sanity = [], [], [], []
    seed = settings["calibration_seeds"][0]
    threshold = settings["weak_gain"]
    for ii, image in enumerate(files):
        truth = load_image(image["path"], device, settings["gray_policy"])
        blur = CircularBlur(np.asarray(kernel["values"]), truth)
        noisy = observation(truth, CircularBlur(np.ones((1, 1)), truth), 25., seed+1, ii, 0, 0)
        restored = denoiser(noisy, 25/255)
        if not torch.isfinite(restored).all():
            raise FloatingPointError("Nonfinite ordinary denoising check")
        sanity.append({"image": image["id"],
                       "noisy_mse": inner(noisy.double()-truth.double(), noisy.double()-truth.double()),
                       "denoised_mse": inner(restored.double()-truth.double(), restored.double()-truth.double())})
        for noise in noises:
            # Keep ni=0 so the standard-normal field stays fixed across amplitudes.
            y = observation(truth, blur, noise, seed, ii, 0, 0)
            common = {"image": image["id"], "noise_255": noise}
            ref, ref_d = trajectory(denoiser, blur, y, sequences["terminal"], solver)
            # Independently use the production integrator on the same 16-step case.
            production = list(red_trajectory(denoiser, blur, y, sequences["terminal"][:16], solver, {16}))[0][1]
            if not torch.equal(production, ref[16]):
                raise AssertionError("Diagnostic integrator differs from production")
            # This check costs 16 extra calls, recorded separately in completion.
            forces.extend({**common, **row} for row in force_rows(denoiser, ref, ref_d, truth, blur, y, solver, threshold))
            data_only = blur.initialize(y, solver)
            data_states = {0: data_only}
            for k in range(1, 129):
                data_only = data_only-solver["step"]*blur.adjoint(blur.forward(data_only)-y)
                if k in (16, 32, 64, 96, 128):
                    data_states[k] = data_only
            for family in FAMILIES:
                states, ds = (ref, ref_d) if family == "terminal" else trajectory(denoiser, blur, y, sequences[family], solver)
                for k in (0, 16, 32, 64, 96, 128):
                    error = states[k].double()-truth.double()
                    displacement = states[k].double()-ref[k].double()
                    endpoint_rows.append({**common, "family": family, "total_calls": k,
                                          **state_metrics(states[k], truth, blur, threshold),
                                          "output_rms_vs_terminal": rms(displacement),
                                          "mse_gap_vs_terminal": inner(error, error)-inner(ref[k].double()-truth.double(), ref[k].double()-truth.double()),
                                          "data_only_mse": inner(data_states[k].double()-truth.double(), data_states[k].double()-truth.double())})
                if family != "terminal":
                    prefix = choices[family]["candidate"]["prefix"]
                    for k in range(prefix, 128):
                        if sequences[family][k] != terminal:
                            raise AssertionError("Expected a common terminal tail")
                        transports.append({**common, "family": family,
                                           **transport_row(k, states, ds, ref, ref_d, blur, solver, threshold)})
            print(f"Diagnosed {image['id']} noise={noise:g}; forwards={denoiser.calls}", flush=True)
    write_csv(out/"endpoints.csv", endpoint_rows)
    write_csv(out/"forces.csv", forces)
    write_csv(out/"terminal_transport.csv", transports)
    write_csv(out/"denoising_sanity.csv", sanity)
    expected = image_count*(len(noises)*(4*128+13+16)+1)
    if denoiser.calls != expected:
        raise AssertionError(f"Unexpected call count: {denoiser.calls} != {expected}")
    write_json(out/"completion.json", {"network_forwards": denoiser.calls,
                                       "production_integrator_checks": image_count*len(noises),
                                       "all_checks_bitwise_equal": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-count", type=int, default=2)
    parser.add_argument("--noise-255", type=float, nargs="+", default=[2.55, 7.65])
    args = parser.parse_args()
    if not all(np.isfinite(n) and n >= 0 for n in args.noise_255):
        raise ValueError("Use finite nonnegative noise amplitudes")
    probe(args.calibration, args.out, args.device, args.image_count, args.noise_255)


if __name__ == "__main__":
    main()
