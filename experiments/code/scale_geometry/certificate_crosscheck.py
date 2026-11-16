"""Cross-check the analytic scalar bounds using mpmath intervals and exact rationals."""
import argparse
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

import mpmath
from mpmath import iv


def box_checks(check):
    I = iv.mpf
    a0, a1, d0, d1 = map(I, (".98", "1.02", "1.98", "2.02"))
    r0, r1, h0, h1 = map(I, ("1.27", "1.31", "1.60", "1.96"))
    alpha, eps, step, error = map(I, (".001", ".000002", ".25", ".0001"))
    c0, c1, xi, eta = map(I, (".999", "1.002", ".3", ".01"))
    low, high, v0, v1 = map(I, (".17", "2.5", ".69", ".71"))
    left, entry, top, right = map(I, ("-.01", ".53", ".05", "1.1"))

    check("coefficient upper envelope", (1+alpha**2)*(1+alpha), c1)
    check("nonnegative base update coefficients", step*(1+c1), 1)
    check("class 1 starts below barrier", alpha+eps, xi)
    check("barrier lower transverse face", error, step*((1+c0)*eta-eps))
    check("class 1 prior mass", ".30", r0/(1+r0+h1))
    check("all prior pair ratios below 2: h", h1, 2)
    check("all prior pair ratios below 2: r", r1, 2)
    check("all prior pair ratios below 2: h/r", h1/r0, 2)

    b_cap = xi - (alpha*(1-xi)+eps+error/step)/c0
    low_log_odds = iv.ln(r1) + (xi-I(".5"))/low
    low_mean = 1/(1+iv.exp(-low_log_odds))
    barrier_gap = (a1*a1+d1*d1)/2+a1*xi+d1*eta
    third = h0*iv.exp(-barrier_gap/high)
    high_mean = (r1-a0*third)/(1+r1+third)
    check("low-scale barrier", low_mean, b_cap)
    check("high-scale barrier", high_mean, b_cap)
    check("chosen low cutoff", low, (I(".5")-xi)/(iv.ln(r1)+iv.ln((1-b_cap)/b_cap)))
    required = ((1-b_cap)*r1-b_cap)/(a0+b_cap)
    check("chosen high cutoff", barrier_gap/iv.ln(h0/required), high)

    base_gap = (a0*a0+d0*d0)/2-d1*top
    channel_gap, entrance_gap = base_gap+a1*left, base_gap+a0*entry
    check("positive channel distance", 0, channel_gap)
    rc, hc = r0*iv.exp((left-I(".5"))/v0), h1*iv.exp(-channel_gap/v1)
    w2_cap = 1/(1+(1+rc)/hc)
    check("rounded posterior leakage", w2_cap, ".044")
    check("rounded derivative bound", (I(".25")+(a1*a1+2*a1)*w2_cap)/v0, ".56")
    check("positive entry distance", 0, entrance_gap)
    re, he = r0*iv.exp((entry-I(".5"))/v1), h1*iv.exp(-entrance_gap/v1)
    check("rounded entry surplus", ".016", (re-a1*he)/(1+re+he)-entry)
    check("observation weak-coordinate term is nonnegative", I(".002"), 1-entry)
    check("progress survives update perturbation", error/step, c0*I(".016"))
    check("transverse bottom", error, step*((1+c0)*eta-eps))
    top_prior = d1*I(".044")-top
    check("transverse top", eps-top+c1*top_prior+error/step, 0)
    # Direct affine iteration instead of the fixed-point closed form.
    progress = left
    for _ in range(25):
        progress += step*c0*(I(".016")+I(".44")*(entry-progress))-error
    check("25-call entry", entry, progress)
    check("first crossing does not overshoot", entry+step*((alpha+c1)*(1-entry)+eps)+error, right)
    check("prepared right face", error, step*(c0*(right-1)-eps))

    # At each extremal state corner the minimizing template coordinates
    # are endpoints, by the monotonic derivatives of the squared-distance gaps.
    corners = [
        (("-.1", ".4"), ("-.1", ".15")),
        ((".53", "1.1"), ("-.1", ".15")),
        (("-1.12", ".01"), ("1.73", "2.27")),
    ]
    gaps = {(z,j): [] for z,j in product(range(3), repeat=2) if z != j}
    for a, d in product(map(F, (".98","1.02")), map(F, ("1.98","2.02"))):
        means = ((0,0), (1,0), (-a,d))
        for z,j in gaps:
            for state in product(*(tuple(map(F, x)) for x in corners[z])):
                squared = [sum((x-m)**2 for x,m in zip(state, mu)) for mu in means]
                gaps[z,j].append(squared[j]-squared[z])
    for (z,j), values in gaps.items():
        check(f"G{z}_to_{j} gap at least .06", F(".06"), min(values), allow_equal=True)
    check("template diameter below 3", (a1+1)**2+d1**2, 9)
    check("G2 left face distance", F(".1"), F("1.12")-F("1.02"), allow_equal=True)
    check("G2 lower face distance", ".1", d0-I("1.73"))
    check("G2 upper face distance", ".1", I("2.27")-d1)
    check("class 2 initial horizontal lower bound", "-1.12", -a1*alpha-eps)
    check("class 2 initial horizontal upper bound", eps, ".01")
    check("class 2 initial vertical lower bound", "1.73", d0-eps)
    check("class 2 initial vertical upper bound", d1+eps, "2.27")
    vl = alpha**2*(1+alpha**2)
    leakage = 4*iv.exp(-I(".03")/vl)
    check("terminal gap to exponential envelope", vl/alpha, ".99")
    check("terminal invariant-face slack", eps+(1+alpha)*3*leakage+error/step, c0*I(".09"))
    check("terminal center initially within 6", I("1.32")+2*eps/c0, 6)
    check("terminal contraction", I(".75")+step*(1+alpha)*9*leakage/vl, ".751")

    # Earlier balanced family bounds preserved in the same certificate.
    def weights(a,d,w,v,c):
        mass0 = I(".5")-(a+1)*w
        r = (I(".5")+a*w)/mass0
        exponent = (a*a+a+d*d)/(2*v)-d*d*c*w/(v*(1+c))
        return r, w*iv.exp(exponent)/mass0

    r_low, h_low = weights(*map(I, (".99","1.99",".039","1.01")), c1)
    r_high, h_high = weights(*map(I, ("1.01","2.01",".041",".99")), c0)
    check("original r lower contained", r0, r_low)
    check("original r upper contained", r_high, r1)
    check("original h lower contained", h0, h_low)
    check("original h upper contained", h_high, h1)
    check("original h increases with omega", 0, 1/I(".041")-I("2.01")**2*c1/(I(".99")*(1+c0)))
    check("original h positive d derivative factor", 0, 1-2*c1*I(".041")/(1+c0))

    tiny = I("1e-5")
    check("A_ralpha <= alpha^3 exponential term", 12*iv.exp(-1/(64*tiny))/tiny**3, ".5")
    check("2 T_alpha <= alpha^6", 256*iv.sqrt(8)*iv.exp(-1/(4000000*tiny**2))/tiny**6, 1)
    check("coefficient mismatch L2 <= alpha^3", alpha/c0**2+1/(2*(1+c0)**2), 1)
    check("sqrt 2 rational bound", 2, I("1.415")**2)
    check("fourth root 3 rational bound", 3, I("1.317")**4)
    check("alpha power bound", alpha**3, I(".000032")**2)
    check("global L4 envelope below 9", I("1.415")*(6+4*error+2*I("1.317")*(alpha**2+I(".000032"))), 9)
    check("fourth moment implies original tail constant", 9**4, 128**2)

    # Wider coordinate tolerances used in the fixed-geometry theorem.
    x_tol, y_tol = alpha/4, I(".005")
    wide_cap = xi-(alpha*(1-xi)+x_tol+error/step)/c0
    check("wide noise: low-scale barrier", low_mean, wide_cap)
    check("wide noise: high-scale barrier", high_mean, wide_cap)
    check("wide noise: transverse bottom", error, step*((1+c0)*eta-y_tol))
    check("wide noise: transverse top", y_tol-top+c1*top_prior+error/step, 0)
    check("wide noise: horizontal data sign", I(".25"), 1-entry)
    check("wide noise: crossing upper bound", entry+step*((alpha+c1)*(1-entry)+x_tol)+error, ".649")
    check("wide noise: prepared right face", error, step*(c0*(right-1)-x_tol))
    check("wide noise: class 2 initial s lower bound", "-1.12", -a1*alpha-x_tol)
    check("wide noise: class 2 initial s upper bound", x_tol, ".01")
    check("wide noise: class 2 initial t lower bound", "1.73", d0-y_tol)
    check("wide noise: class 2 initial t upper bound", d1+y_tol, "2.27")
    check("wide noise: terminal invariant-face slack", y_tol+12*c1*iv.exp(-30)+error/step, c0*I(".09"))


def nominal_checks(check):
    I = iv.mpf
    # Compute displacement directly from Gaussian weights, not J_u.
    s, t, a, d, r, h = map(I, (".3","-.01","1","2","1.29","1.78"))
    alpha, tau2, vl, step = map(I, (".001","1e-12",".000001000001",".25"))
    lam = 1+alpha**2
    distances = (s*s+t*t, (s-1)**2+t*t, (s+a)**2+(t-d)**2)
    P, Q = ((distances[j]-distances[0])/2 for j in (1,2))
    cap = (1-s)*r-s

    def poly(mass):
        return -P*r+mass*(a*Q+(1+a)*(Q-P)*r)-tau2*(cap-(a+s)*mass)*(1+r+mass)

    check("nominal: critical mass lower bracket", poly(I(".02693223098")), 0)
    check("nominal: critical mass upper bracket", 0, poly(I(".02693223100")))
    check("nominal: mass ceiling lower bracket", ".46384615384", cap/(a+s))
    check("nominal: mass ceiling upper bracket", cap/(a+s), ".46384615385")

    for delta, low, high, name in [
        ("0", (".18095055387",".18095055389"), ("1.87272136332","1.87272136335"), "exact"),
        (".0001", (".18063871120",".18063871122"), ("1.87551308613","1.87551308616"), "robust"),
    ]:
        target = (-I(delta)/step-alpha*(1-s))/lam
        fraction = 1-tau2/vl
        denominator = (1-s)*fraction-target
        u_min, w_min = (s*fraction+target)/denominator, ((a+s)*fraction+target)/denominator
        d_first = -target*tau2/denominator**2
        d_second = 2*(1-s)*tau2*d_first/denominator
        check(f"nominal {name}: terminal denominator", 0, denominator)
        check(f"nominal {name}: terminal numerator", 0, s*fraction+target)
        check(f"nominal {name}: first convexity margin", 2*P*d_first+d_second, P**2*u_min)
        check(f"nominal {name}: second convexity margin", (a+1)*d_second, (Q-P)**2*w_min)

        def drift(v):
            v = I(v)
            weights = [mass*iv.exp(-gap/(2*v)) for mass,gap in zip((1,r,h),distances)]
            mean = (weights[1]-a*weights[2])/sum(weights)
            return alpha*(1-s)+lam*(1-tau2/v)*(mean-s)+I(delta)/step

        check(f"nominal {name}: just below lower root", drift(low[0]), 0)
        check(f"nominal {name}: just above lower root", 0, drift(low[1]))
        check(f"nominal {name}: just below upper root", 0, drift(high[0]))
        check(f"nominal {name}: just above upper root", drift(high[1]), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    iv.dps = 70
    checks = []

    def check(name, left, right, allow_equal=False):
        if isinstance(left, F) and isinstance(right, F):
            margin = right-left
            passed = margin >= 0 if allow_equal else margin > 0
        else:
            margin = iv.mpf(right)-iv.mpf(left)
            passed = margin.a >= 0 if allow_equal else margin.a > 0
        if not passed:
            raise ArithmeticError(f"{name}: {margin}")
        checks.append({"name": name, "margin": str(margin)})

    box_checks(check)
    nominal_checks(check)
    data = {
        "status": "PASS",
        "backend": f"mpmath {mpmath.__version__} iv, exact Fraction for rational comparisons",
        "interval_precision_digits": iv.dps,
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    print(f"PASS: {len(checks)} inequalities")


if __name__ == "__main__":
    main()
