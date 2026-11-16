"""Noiseless classwise weak-regularization budget curve."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "red_geometry_computation_risk/code"))
from family import Family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = Family(1e-5)
    model.pi = np.array([1, 1.29, 1.78])/4.07
    model.lam = model.alpha/4
    b = model.mu@model.B
    x = b.copy()
    records = []
    checkpoints = {154, 1000, 10000, 100000, 1000000}
    first = [None]*3
    for k in range(1, max(checkpoints)+1):
        x = model.step(x, b, model.vend)
        for z in range(3):
            if first[z] is None and np.linalg.norm(x[z]-model.mu[z]) < .1:
                first[z] = k
        if k in checkpoints:
            error = np.sum((x-model.mu)**2, axis=1)
            records.append({"calls": k, "risk": float(model.pi@error),
                            "classwise_squared_error": error.tolist()})
    (args.out/"budget.json").write_text(json.dumps({
        "observation": "Noiseless template observations, prior-weighted evaluation",
        "alpha": model.alpha, "lambda": model.lam,
        "first_radius_point1_entry": first, "records": records,
        "scope": "Fixed geometry and weak-lambda terminal-only protocol; sampled trajectory."
    }, indent=2))


if __name__ == "__main__":
    main()
