"""Evaluate the fine preset at two precisions using affine terminal recursion."""
import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261109)
    # Replay the original two rows' draw consumption exactly.
    for _ in range(4):
        rng.standard_normal((1536, 2))
    normals_u = rng.standard_normal((1536, 2))
    normals_eta = rng.standard_normal((1536, 2))
    rows = []
    for digits in (50, 80):
        mp.mp.dps = digits
        alpha = mp.mpf("1e-5")
        tau = alpha**2
        beta = [alpha, mp.mpf(1)]
        mu = [(0, 0), (1, 0), (-1, 2)]
        masses = [mp.mpf(1), mp.mpf("1.29"), mp.mpf("1.78")]
        sums = [mp.mpf(0)] * 3
        entry_count = 0
        for i in range(1536):
            z = i // 512
            truth = [mp.mpf(mu[z][j]) + tau * mp.mpf(float(normals_u[i, j]))
                     for j in range(2)]
            b = [beta[j] * truth[j] + mp.sqrt(beta[j]) * tau *
                 mp.mpf(float(normals_eta[i, j])) for j in range(2)]
            x = b.copy()
            target = 2 if b[1] > 1 else int(b[0] > alpha / 2)
            if target == 1:
                v = mp.mpf(".70")
                c = (1 + alpha**2) * (1 - tau**2 / v)
                for _ in range(25):
                    w = [masses[k] * mp.exp(-sum((x[j] - mu[k][j])**2
                         for j in range(2)) / (2 * v)) for k in range(3)]
                    total = sum(w)
                    mean = [sum(w[k] * mu[k][j] for k in range(3)) / total
                            for j in range(2)]
                    x = [x[j] - (beta[j] * x[j] - b[j] +
                         c * (x[j] - mean[j])) / 4 for j in range(2)]
            rectangles = [(-.1, .4, -.1, .15), (.53, 1.1, -.1, .15),
                          (-1.12, .01, 1.73, 2.27)]
            lo, hi, bottom, top = rectangles[z]
            assert target == z and lo <= x[0] <= hi and bottom <= x[1] <= top
            entry_count += 1
            center = [(b[j] + mu[z][j]) / (beta[j] + 1) for j in range(2)]
            n = 129 if target == 1 else 154
            residual = [(1 - (beta[j] + 1) / 4)**n * (x[j] - center[j])
                        for j in range(2)]
            sums[z] += sum(t**2 for t in residual)
        excess = sum(masses[z] / sum(masses) * sums[z] / 512 for z in range(3))
        # G_z is invariant and all competing squared-distance gaps are >= .06.
        # Accumulated deviation from the affine recursion is <= 12 exp(-.03/v_L).
        v_l = alpha**2 + alpha**4
        log_leak = mp.log10(12) - mp.mpf(".03") / v_l / mp.log(10)
        rows.append({"digits": digits, "excess_affine_reference": str(excess),
                     "entry_checks": entry_count,
                     "terminal_output_error_log10_upper": str(log_leak)})
    output = {"seed": 20261109, "samples_per_class": 512,
              "input": "Exact decimal parameters and binary64 standard-normal draws",
              "terminal": "Affine recursion after verified entry, c_L=1",
              "rows": rows}
    (args.out / "precision.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
