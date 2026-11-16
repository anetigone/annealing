"""Model-input sufficient checks for a common terminal contraction ball."""
import numpy as np

def pairwise_covariance_bound(means, pi, v, center, radius):
    """Bound C on a ball using both pair odds and all-component competition."""
    energy = np.sum((center-means)**2, axis=1)
    diff = means[:, None]-means[None]
    distances = np.linalg.norm(diff, axis=-1)
    log_odds = np.log(pi)[:, None]-np.log(pi)[None]
    log_odds -= (energy[:, None]-energy[None])/(2*v)
    lower = log_odds-radius*distances/v
    upper = log_odds+radius*distances/v
    # w_i = 1 / sum_j exp(log(w_j/w_i)).
    cap = np.exp(-np.logaddexp.reduce(lower, axis=0))
    bound = np.zeros((means.shape[1], means.shape[1]))
    for i in range(len(means)):
        for j in range(i):
            nearest_zero = np.clip(0, lower[i, j], upper[i, j])
            log_pair = -np.logaddexp(0, nearest_zero)-np.logaddexp(0, -nearest_zero)
            product = min(cap[i]*cap[j], np.exp(log_pair))
            bound += product*np.outer(diff[i, j], diff[i, j])
    return bound

def terminal_ball_check(B, b, means, pi, v, c, gamma,
                        center, radius, reference, points=None):
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    means = np.asarray(means, dtype=float)
    pi = np.asarray(pi, dtype=float)
    center = np.asarray(center, dtype=float)
    arrays = [B, b, means, pi, center]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("model inputs must be finite")
    if not np.isfinite([v, c, gamma, radius]).all():
        raise ValueError("v, c, gamma, radius must be finite")
    d = len(center)
    if B.shape != (d, d) or b.shape != (d,):
        raise ValueError("B and b have incompatible dimensions")
    if means.ndim != 2 or means.shape[1] != d:
        raise ValueError("means must have shape (K, d)")
    if pi.shape != (len(means),) or np.any(pi <= 0):
        raise ValueError("pi must contain positive mixture weights")
    if not (v > 0 and c > 0 and gamma > 0 and radius > 0):
        raise ValueError("v, c, gamma, radius must be positive")
    if not 0 <= reference < len(means):
        raise ValueError("reference is out of range")
    if not np.allclose(B, B.T, atol=1e-12, rtol=1e-12):
        raise ValueError("B must be symmetric")
    B = (B + B.T) / 2
    eigen_B = np.linalg.eigvalsh(B)
    if eigen_B[0] < -1e-12:
        raise ValueError("B must be positive semidefinite")

    delta = means - means[reference]
    energy = np.sum((center - means)**2, axis=1)
    gaps = energy - energy[reference]
    log_odds = (np.log(pi) - np.log(pi[reference])
                - (gaps - 2*radius*np.linalg.norm(delta, axis=1))/(2*v))
    beta = np.exp(-np.logaddexp(0, -log_odds))
    beta[reference] = 0
    Cbar = (delta.T * beta) @ delta

    chol = np.linalg.cholesky(B + c*np.eye(d))
    pair_bound = pairwise_covariance_bound(means, pi, v, center, radius)
    rho_bounds = []
    for covariance in [Cbar, pair_bound]:
        left = np.linalg.solve(chol, covariance)
        normalized = np.linalg.solve(chol, left.T).T
        normalized = (normalized + normalized.T) / 2
        rho_bounds.append((c/v)*np.linalg.eigvalsh(normalized)[-1])
    rho = min(rho_bounds)
    lower_curvature = (1-rho)*(eigen_B[0]+c)
    upper_curvature = eigen_B[-1]+c
    q = 1-gamma*lower_curvature

    log_w = np.log(pi) - energy/(2*v)
    weights = np.exp(log_w-log_w.max())
    weights /= weights.sum()
    m = weights @ means
    residual = gamma*np.linalg.norm(B@center-b+c*(center-m))
    containment_margin = (1-q)*radius-residual
    starts_inside = True
    max_start_distance = None
    if points is not None:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != d:
            raise ValueError("points must have shape (N, d)")
        max_start_distance = float(np.linalg.norm(points-center, axis=1).max())
        starts_inside = max_start_distance <= radius
    passed = (rho < 1 and gamma*upper_curvature <= 1
              and containment_margin >= 0 and starts_inside)
    return dict(
        passed=bool(passed), rho_upper=float(rho), q_upper=float(q),
        curvature_margin=float(1-rho),
        step_margin=float(1-gamma*upper_curvature),
        containment_margin=float(containment_margin),
        residual=float(residual), starts_inside=bool(starts_inside),
        max_start_distance=max_start_distance,
        covariance_bound=("reference" if rho_bounds[0] <= rho_bounds[1] else "pairwise"),
        rho_reference=float(rho_bounds[0]), rho_pairwise=float(rho_bounds[1]),
        note="Floating-point diagnostic; not a rounded interval certificate."
    )
