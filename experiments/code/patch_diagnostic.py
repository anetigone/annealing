"""Local natural-photo patch GMM diagnostic with fixed blur and protocols."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_images


def build_case():
    photos = load_sample_images()
    patches = []
    for image in photos.images:
        gray = np.asarray(Image.fromarray(image).convert("L"), dtype=float)/255
        blocks = gray[:gray.shape[0]//4*4, :gray.shape[1]//4*4]
        patches.append(blocks.reshape(blocks.shape[0]//4, 4, blocks.shape[1]//4, 4)
                       .transpose(0, 2, 1, 3).reshape(-1, 16))
    rng = np.random.default_rng(20261110)
    train = patches[0][rng.choice(len(patches[0]), 2048, replace=False)]
    truth = patches[1][rng.choice(len(patches[1]), 256, replace=False)]
    fit = KMeans(n_clusters=8, random_state=20261110, n_init=5).fit(train)
    mu = fit.cluster_centers_
    pi = np.bincount(fit.labels_, minlength=8)/len(train)
    tau2 = np.mean((train-mu[fit.labels_])**2)
    # Periodic 4x4 box-like separable blur, weights [.25,.5,.25].
    basis = np.eye(16).reshape(16, 4, 4)
    blurred = .5*basis + .25*np.roll(basis, 1, 1)+.25*np.roll(basis, -1, 1)
    blurred = .5*blurred + .25*np.roll(blurred, 1, 2)+.25*np.roll(blurred, -1, 2)
    a = blurred.reshape(16, 16)
    y = truth@a + .01*rng.standard_normal(truth.shape)
    b = y@a.T
    return photos, train, truth, mu, pi, tau2, a, y, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    photos, train, truth, mu, pi, tau2, a, y, b = build_case()
    lam, gamma, terminal = .1, .25, .02
    def step(x, sigma):
        v = tau2 + sigma**2
        logw = np.log(pi)-np.sum((x[:, None]-mu)**2, axis=2)/(2*v)
        w = np.exp(logw-logw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        return x-gamma*(x@a@a.T-b+lam*sigma**2/v*(x-w@mu))
    results = {}
    for policy in ["terminal", "intermediate", "low_high"]:
        x = b.copy()
        for k in range(154):
            sigma = terminal
            if k < 25:
                if policy == "intermediate":
                    sigma = .1
                elif policy == "low_high":
                    sigma = terminal if k % 2 == 0 else .3
            x = step(x, sigma)
        error = np.sum((x-truth)**2, axis=1)
        assert np.isfinite(x).all()
        results[policy] = {"squared_euclidean_risk": float(error.mean()),
                           "pixel_mse": float(error.mean()/16)}
    output = {"seed": 20261110, "training_photo": photos.filenames[0],
              "test_photo": photos.filenames[1], "train_patches": 2048,
              "test_patches": 256, "components": 8, "tau2": float(tau2),
              "noise_std": .01, "lambda": lam, "gamma": gamma,
              "terminal_sigma": terminal, "calls": 154,
              "selection": "Fixed protocols and parameters, no schedule search.",
              "scope": "Empirical photo patches, fitted common isotropic GMM; no template labels.",
              "results": results}
    (args.out/"patch.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
