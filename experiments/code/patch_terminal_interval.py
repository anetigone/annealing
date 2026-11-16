"""Outward-rounded common-ball checks for exported patch states at H=154."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import numpy as np
from mpmath import iv


def exact(value):
    numerator, denominator = float(value).as_integer_ratio()
    return iv.mpf(numerator)/denominator


def lower_positive(value):
    return value.a > 0


def norm(values):
    return iv.sqrt(sum(x*x for x in values))


def positive_pivots(matrix):
    """Interval LDL without pivoting proves positive definiteness."""
    size = len(matrix)
    lower = [[iv.mpf(int(i == j)) for j in range(size)] for i in range(size)]
    pivots = []
    for i in range(size):
        pivot = matrix[i][i]-sum(lower[i][k]**2*pivots[k] for k in range(i))
        assert lower_positive(pivot), ("nonpositive LDL pivot", i, str(pivot))
        pivots.append(pivot)
        for j in range(i+1, size):
            lower[j][i] = (matrix[j][i]-sum(
                lower[j][k]*lower[i][k]*pivots[k] for k in range(i)))/pivot
    return min(np.nextafter(float(p.a), -np.inf) for p in pivots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adaptive-rho", action="store_true",
                        help="Verify sample-specific rho upper bounds, retaining .45 when possible")
    parser.add_argument("--time",type=int,default=154,
                        help="Common time identifying the exported starts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    iv.dps = 60
    data = np.load(args.data/"candidates.npz")
    checks = json.loads((args.data/"checks.json").read_text())["rows"]
    gram = data["gram"]
    size = gram.shape[0]
    # An exported rational factor permits structure-coordinate certificates.
    # Older 16D archives use the original periodic separable blur.
    if "operator" in data:
        a = data["operator"]
    else:
        basis = np.eye(size).reshape(size, 4, 4)
        blurred = .5*basis+.25*np.roll(basis, 1, 1)+.25*np.roll(basis, -1, 1)
        blurred = .5*blurred+.25*np.roll(blurred, 1, 2)+.25*np.roll(blurred, -1, 2)
        a = blurred.reshape(size, size)
    for i in range(size):
        for j in range(size):
            assert Fraction(float(gram[i, j])) == sum(
                Fraction(float(a[i, k]))*Fraction(float(a[j, k])) for k in range(size))
    row_bound = max(sum(abs(Fraction(float(x))) for x in row) for row in gram)
    norm_upper = iv.mpf(row_bound.numerator)/row_bound.denominator
    B = [[exact(x) for x in row] for row in gram]
    means = [[exact(x) for x in row] for row in data["means"]]
    pi = [exact(x) for x in data["pi"]]
    v, c, gamma = [exact(data[key]) for key in ["v", "c", "gamma"]]
    rho = iv.mpf("0.45")
    m = (1-rho)*c
    q = 1-gamma*m
    assert lower_positive(1-gamma*(norm_upper+c))
    count = len(means)
    difference = [[[means[i][k]-means[j][k] for k in range(size)]
                   for j in range(count)] for i in range(count)]
    distances = [[norm(difference[i][j]) for j in range(count)] for i in range(count)]
    records = []
    rho_values = []
    for sample in range(len(data["b"])):
        row = next(r for r in checks if r["sample"] == sample and r["time"] == args.time)
        rho_label = ".45"
        if args.adaptive_rho and row["rho_pairwise"] >= .45:
            # This only chooses a proposed proof bound. Interval LDL must still
            # verify it. It never changes the estimator or the evaluation set.
            rho_label = str((math.ceil(1000*row["rho_pairwise"])+1)/1000)
        rho = iv.mpf(rho_label)
        assert rho.b < 1
        q = 1-gamma*(1-rho)*c
        rho_values.append(rho)
        radius = exact(row["radius"])
        center = [exact(x) for x in data["centers"][sample]]
        energy = [sum((center[k]-mu[k])**2 for k in range(size)) for mu in means]
        log_odds = [[iv.ln(pi[i]/pi[j])-(energy[i]-energy[j])/(2*v)
                     for j in range(count)] for i in range(count)]
        lo = [[log_odds[i][j]-radius*distances[i][j]/v
               for j in range(count)] for i in range(count)]
        hi = [[log_odds[i][j]+radius*distances[i][j]/v
               for j in range(count)] for i in range(count)]
        cap = [1/sum(iv.exp(lo[j][i]) for j in range(count)) for i in range(count)]
        covariance = [[iv.mpf(0) for _ in range(size)] for _ in range(size)]
        for i in range(count):
            for j in range(i):
                if lo[i][j].a > 0:
                    nearest = lo[i][j]
                elif hi[i][j].b < 0:
                    nearest = hi[i][j]
                else:
                    nearest = iv.mpf(0)
                pair = 1/((1+iv.exp(nearest))*(1+iv.exp(-nearest)))
                product = cap[i]*cap[j]
                bound = pair if pair.b < product.b else product
                for k in range(size):
                    for ell in range(size):
                        covariance[k][ell] += bound*difference[i][j][k]*difference[i][j][ell]
        matrix = [[rho*(B[i][j]+(c if i == j else 0))-(c/v)*covariance[i][j]
                   for j in range(size)] for i in range(size)]
        min_pivot = positive_pivots(matrix)
        raw = [pi[i]*iv.exp(-energy[i]/(2*v)) for i in range(count)]
        weights = [w/sum(raw) for w in raw]
        posterior = [sum(weights[i]*means[i][k] for i in range(count)) for k in range(size)]
        field = [sum(B[i][j]*center[j] for j in range(size))-exact(data["b"][sample, i])+
                 c*(center[i]-posterior[i]) for i in range(size)]
        residual = gamma*norm(field)
        invariant_margin = (1-q)*radius-residual
        assert lower_positive(invariant_margin), (sample, "invariance")
        start_margins = []
        for policy in range(3):
            point = data["states"][policy, sample]
            margin = radius-norm([exact(point[k])-center[k] for k in range(size)])
            assert lower_positive(margin), (sample, "coverage")
            start_margins.append(str(margin))
        records.append({"sample": sample, "min_ldl_pivot_lower": min_pivot,
                        "containment_margin": str(invariant_margin),
                        "start_margins": start_margins})
        if args.adaptive_rho:
            records[-1].update({"rho_upper": rho_label, "q": str(q)})
    rho = max(rho_values, key=lambda value: float(value.b))
    q = 1-gamma*(1-rho)*c
    rho_label = ".45" if not args.adaptive_rho else str(rho)
    (args.out/"certificate.json").write_text(json.dumps({
        "status": "PASS", "precision_digits": iv.dps, "rho_upper": rho_label,
        "q": str(q), "samples": len(records), "common_time": args.time,
        "minimum_ldl_pivot_lower": min(r["min_ldl_pivot_lower"] for r in records),
        "scope": "Exact terminal map with exported binary64 parameters and states interpreted as exact inputs. "
                 "Earlier floating-point propagation and subsequent floating-point errors are not enclosed. "
                 "No probability claim over new observations.",
        "records": records}, indent=2))
    print(f"PASS: {len(records)} common terminal balls, q={q}")


if __name__ == "__main__":
    main()
