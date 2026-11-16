"""Decompose the data-only reference on the saved development observations."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def decompose(source):
    plan = json.loads((source/"plan.json").read_text(encoding="utf-8-sig"))
    settings = plan["settings"]
    if len(settings["kernels"]) != 1:
        raise ValueError("Use the single-kernel development probe")
    if settings["solver"]["initialization"] != "backprojection":
        raise ValueError("The saved data-only decomposition assumes backprojection")
    with (source/"endpoints.csv").open(encoding="utf-8") as stream:
        saved = {(r["image"], float(r["noise_255"]), int(r["total_calls"])): float(r["data_only_mse"])
                 for r in csv.DictReader(stream) if r["family"] == "terminal"}
    kernel = np.asarray(settings["kernels"][0]["values"], dtype=np.float64)
    gamma, seed = settings["solver"]["step"], settings["calibration_seeds"][0]
    rows = []
    for ii, image in enumerate(plan["development_images"]):
        with Image.open(image["path"]) as opened:
            truth = (np.asarray(opened.convert("L"), dtype=np.float32)/255).astype(np.float64)
        psf = np.zeros_like(truth)
        kh, kw = kernel.shape
        psf[:kh, :kw] = kernel
        psf = np.roll(psf, (-(kh//2), -(kw//2)), (0, 1))
        otf = np.fft.fft2(psf)
        gain = abs(otf)**2
        rng = np.random.default_rng(np.random.SeedSequence([seed, ii, 0, 0]))
        epsilon = rng.standard_normal((1, 1, *truth.shape)).astype(np.float32)[0, 0].astype(np.float64)
        clean_b = gain*np.fft.fft2(truth)
        noise_b = otf.conjugate()*np.fft.fft2(epsilon)
        transfer = np.ones_like(gain)
        for k in range(129):
            if k in (0, 16, 32, 64, 96, 128):
                bias = np.fft.ifft2(transfer*clean_b).real-truth
                unit_response = np.fft.ifft2(transfer*noise_b).real
                for noise in plan["noise_sigma_255"]:
                    response = (noise/255)*unit_response
                    bias2 = float(np.mean(bias**2))
                    noise2 = float(np.mean(response**2))
                    cross = float(2*np.mean(bias*response))
                    mse = float(np.mean((bias+response)**2))
                    assert abs(mse-bias2-noise2-cross) < 1e-16
                    original = saved[(image["id"], noise, k)]
                    rows.append({"image": image["id"], "noise_255": noise, "total_calls": k,
                                 "bias_mse": bias2, "noise_response_mse": noise2,
                                 "cross_term": cross, "reconstructed_mse": mse,
                                 "saved_float32_data_mse": original,
                                 "float64_minus_saved_mse": mse-original})
            transfer = (1-gamma*gain)*transfer+gamma
    discrepancy = max(abs(r["float64_minus_saved_mse"]) for r in rows)
    if discrepancy > 1e-8:
        raise ValueError(f"Double-precision reference is too far from the saved run: {discrepancy}")
    return {"scope": "No denoiser calls. Same two saved development images, seed and standard-normal fields. Double-precision periodic operator; comparison with original float32 GPU data-only endpoints is reported explicitly. Cross terms for the actual noise realization are retained, not replaced by their expectation.",
            "source_probe": str(source), "maximum_absolute_mse_discrepancy": discrepancy,
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = decompose(args.source)
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
