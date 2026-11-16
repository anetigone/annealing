"""Periodic known blur, local image input, and float-image recovery metrics."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import convolve1d
import torch


def load_image(path, device, gray_policy="require_grayscale"):
    with Image.open(path) as image:
        if gray_policy == "require_grayscale" and image.mode != "L":
            raise ValueError(f"Expected original 8-bit grayscale image, got {image.mode}: {path}")
        if gray_policy not in ("require_grayscale", "pillow_luma"):
            raise ValueError(f"Unknown grayscale policy: {gray_policy}")
        array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255
    if min(array.shape) < 11:
        raise ValueError("Images must be at least 11x11 for the fixed SSIM window")
    return torch.from_numpy(array)[None, None].to(device)


def load_kernel(spec):
    kind = spec["type"]
    if kind == "gaussian":
        size, sigma = int(spec["size"]), float(spec["sigma"])
        if size <= 0 or size % 2 != 1 or sigma <= 0:
            raise ValueError("Gaussian kernel needs positive odd size and positive sigma")
        grid = np.arange(size, dtype=np.float64) - size // 2
        line = np.exp(-grid**2 / (2 * sigma**2))
        kernel = np.outer(line, line)
    else:
        path = Path(spec["path"])
        if kind == "npy":
            # The documented, manually acquired Levin09 mirror can be an object array.
            array = np.load(path, allow_pickle=True)
        elif kind == "mat":
            array = loadmat(path)[spec.get("key", "kernels")]
        else:
            raise ValueError(f"Unsupported kernel type: {kind}")
        if array.dtype == object:
            kernel = array.reshape(-1)[int(spec.get("index", 0))]
        elif array.ndim == 3:
            kernel = np.take(array, int(spec.get("index", 0)), axis=int(spec.get("axis", 0)))
        else:
            kernel = array
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or not np.isfinite(kernel).all() or np.any(kernel < 0) or kernel.sum() <= 0:
        raise ValueError("Expected a finite nonnegative 2D blur kernel with positive mass")
    return kernel / kernel.sum()


class CircularBlur:
    def __init__(self, kernel, reference):
        h, w = reference.shape[-2:]
        kh, kw = kernel.shape
        if kh > h or kw > w:
            raise ValueError("Kernel must fit the image; images are never resized or cropped")
        psf = reference.new_zeros((h, w))
        psf[:kh, :kw] = torch.as_tensor(kernel, dtype=reference.dtype, device=reference.device)
        psf = torch.roll(psf, (-(kh // 2), -(kw // 2)), (-2, -1))
        self.otf = torch.fft.fft2(psf)
        self.gain = self.otf.abs().square()

    def forward(self, x):
        return torch.fft.ifft2(torch.fft.fft2(x) * self.otf).real

    def adjoint(self, y):
        return torch.fft.ifft2(torch.fft.fft2(y) * self.otf.conj()).real

    def initialize(self, y, solver):
        b = self.adjoint(y)
        if solver["initialization"] == "backprojection":
            return b
        rho = float(solver["ridge"])
        if solver["initialization"] != "ridge" or rho <= 0:
            raise ValueError("Initialization must be backprojection or ridge with rho > 0")
        return torch.fft.ifft2(torch.fft.fft2(b) / (self.gain + rho)).real


def observation(truth, blur, noise_255, seed, image_index, kernel_index, noise_index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, image_index, kernel_index, noise_index]))
    noise = torch.as_tensor(rng.standard_normal(tuple(truth.shape)), device=truth.device,
                            dtype=truth.dtype) * (noise_255 / 255)
    return blur.forward(truth) + noise


def ssim(x, truth):
    grid = np.arange(11, dtype=np.float64) - 5
    weights = np.exp(-grid**2 / (2 * 1.5**2))
    weights /= weights.sum()

    def smooth(array):
        result = convolve1d(convolve1d(array, weights, axis=0), weights, axis=1)
        return result[5:-5, 5:-5]

    ux, uy = smooth(x), smooth(truth)
    vx, vy = smooth(x*x) - ux*ux, smooth(truth*truth) - uy*uy
    cov = smooth(x*truth) - ux*uy
    score = ((2*ux*uy + .01**2) * (2*cov + .03**2)
             / ((ux*ux + uy*uy + .01**2) * (vx + vy + .03**2)))
    return float(score.mean())


def metrics(x, truth, blur, weak_gain):
    error = (x.double() - truth.double())
    sse = float(error.square().sum().item())
    mse = sse / error.numel()
    spectrum = torch.fft.fft2(error, norm="ortho")
    weak_sse = float((spectrum.abs().square() * (blur.gain <= weak_gain)).sum().item())
    a = x.detach().cpu().numpy().squeeze().astype(np.float64)
    b = truth.detach().cpu().numpy().squeeze().astype(np.float64)
    clipped_mse = float(np.mean((np.clip(a, 0, 1) - b)**2))
    return {
        "sse": sse, "mse": mse, "psnr": 10*np.log10(1/max(mse, 1e-30)),
        "ssim": ssim(a, b), "weak_sse": weak_sse, "weak_mse": weak_sse/error.numel(),
        "clipped_mse": clipped_mse,
        "clipped_psnr": 10*np.log10(1/max(clipped_mse, 1e-30)),
    }
