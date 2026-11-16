"""Diagnose the fitted prior's geometry and its terminal-scale limit."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from mpmath import iv
from patch_terminal_interval import exact


def rational_row_rank(matrix):
    rows = [[Fraction(x) for x in row] for row in matrix]
    rank = 0
    for column in range(matrix.shape[1]):
        pivot = next((i for i in range(rank,len(rows)) if rows[i][column]),None)
        if pivot is None:
            continue
        rows[rank],rows[pivot] = rows[pivot],rows[rank]
        scale = rows[rank][column]
        rows[rank] = [x/scale for x in rows[rank]]
        for i in range(rank+1,len(rows)):
            multiplier = rows[i][column]
            rows[i] = [a-multiplier*b for a,b in zip(rows[i],rows[rank])]
        rank += 1
        if rank == len(rows):
            break
    return rank


def verify_negative_rayleigh(mu, pi, gram, state, direction, v, c):
    iv.dps = 60
    x = [exact(z) for z in state]
    u = [exact(z) for z in direction]
    means = [[exact(z) for z in row] for row in mu]
    raw = [exact(pi[i])*iv.exp(-sum((x[k]-mean[k])**2 for k in range(16))/(2*exact(v)))
           for i,mean in enumerate(means)]
    w = [z/sum(raw) for z in raw]
    projection = [sum(u[k]*mean[k] for k in range(16)) for mean in means]
    variance = sum(wi*t*t for wi,t in zip(w,projection))-sum(
        wi*t for wi,t in zip(w,projection))**2
    squared_norm = sum(z*z for z in u)
    data = sum(u[i]*exact(gram[i,j])*u[j] for i in range(16) for j in range(16))
    rayleigh = (data+exact(c)*squared_norm-exact(c)/exact(v)*variance)/squared_norm
    assert rayleigh.b < 0
    return str(rayleigh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = np.load(args.validation/"frozen_model.npz")
    mu, pi, a, tau2 = [model[k] for k in ["mu", "pi", "a", "tau2"]]
    tau2 = float(tau2)
    centered = mu-pi@mu
    covariance = (centered.T*pi)@centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    dc = np.ones(16)/4
    gram = a@a.T
    v = tau2+.02**2
    c = .1*.02**2/v
    inverse = np.linalg.inv(gram+c*np.eye(16))
    differences = mu[:,None]-mu[None]
    squared = np.einsum("ijk,kl,ijl->ij", differences, inverse, differences)
    first, second = np.unravel_index(squared.argmax(), squared.shape)
    rows = mu[1:]-mu[0]
    witnesses = []
    for epsilon in [.01, 1e-12]:
        desired = np.full(len(mu), epsilon)
        desired[[first, second]] = (1-(len(mu)-2)*epsilon)/2
        rhs = v*(np.log(desired[1:]/desired[0])-np.log(pi[1:]/pi[0]))
        rhs += (np.sum(mu[1:]**2,axis=1)-np.sum(mu[0]**2))/2
        state = np.linalg.lstsq(rows,rhs,rcond=None)[0]
        log = np.log(pi)-np.sum((state-mu)**2,axis=1)/(2*v)
        weights = np.exp(log-log.max())
        weights /= weights.sum()
        shifted = mu-weights@mu
        posterior_covariance = (shifted.T*weights)@shifted
        hessian = gram+c*np.eye(16)-c/v*posterior_covariance
        assert np.max(np.abs(weights-desired)) < 1e-8
        values, vectors = np.linalg.eigh(hessian)
        minimum = float(values[0])
        assert minimum < -.01
        interval_rayleigh = verify_negative_rayleigh(mu,pi,gram,state,vectors[:,0],v,c)
        witnesses.append({
            "other_component_weight": epsilon, "state": state.tolist(),
            "state_norm": float(np.linalg.norm(state)),
            "state_range": [float(state.min()),float(state.max())],
            "posterior": weights.tolist(), "minimum_hessian_eigenvalue": minimum,
            "negative_rayleigh_interval": interval_rayleigh})
    plan = json.loads((args.validation/"plan.json").read_text())
    ranges = {}
    for photo in plan["photos"]:
        outputs = np.load(args.validation/photo/"outputs.npz")
        states = outputs["states"]
        ranges[photo] = {"state_min": float(states.min()),
                         "state_max": float(states.max()),
                         "max_state_norm": float(np.linalg.norm(states,axis=2).max())}
    result = {
        "mean_covariance_top_fraction": float(eigenvalues[-1]/eigenvalues.sum()),
        "top_mean_direction_dc_alignment_squared": float((eigenvectors[:,-1]@dc)**2),
        "dc_sensing_gain": float(dc@gram@dc),
        "tau_standard_deviation": float(np.sqrt(tau2)),
        "terminal_sigma_over_tau": float(.02/np.sqrt(tau2)),
        "terminal_c": c, "tau2": tau2,
        "coefficient_formula_evaluations": {
            str(sigma): {"v": tau2+sigma*sigma,
                         "c": .1*sigma*sigma/(tau2+sigma*sigma)}
            for sigma in [.2,.1,.02,.01,.001]},
        "template_difference_singular_values": np.linalg.svd(rows,compute_uv=False).tolist(),
        "exact_binary64_mean_difference_rank": rational_row_rank(
            np.array([[Fraction(float(x))-Fraction(float(y)) for x,y in zip(row,mu[0])]
                      for row in mu[1:]],dtype=object)),
        "global_whitened_diameter_rho_supremum_for_affinely_independent_means": float(c/(4*v)*squared.max()),
        "extremal_pair": [int(first),int(second)], "negative_curvature_witnesses": witnesses,
        "recorded_state_ranges": ranges,
        "scope": "Fixed fitted model, analytic coefficient evaluations and two constructed states. "
                 "No recovery parameter search. Large witnesses do not prove absence of all "
                 "negative curvature in the natural-pixel domain."
    }
    (args.out/"geometry_limits.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in [
        "mean_covariance_top_fraction","top_mean_direction_dc_alignment_squared",
        "tau_standard_deviation","terminal_c"]}))


if __name__ == "__main__":
    main()
