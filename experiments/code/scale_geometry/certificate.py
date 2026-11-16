#!/usr/bin/env python3
"""Outward-rounded checks for the geometric RED sufficient conditions.

Python 3.10+; standard library only. No sampled trajectories or parameter grid.
The analytic reductions and monotonicity arguments are in paper/latex/scale_geometry_proof.tex.
This is a numerical certificate, not a formally verified proof assistant.
"""
import argparse
import json
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.precision < 40:
        raise ValueError("Use at least 40 decimal digits")
    down = Context(prec=args.precision, rounding=ROUND_FLOOR)
    up = Context(prec=args.precision, rounding=ROUND_CEILING)
    near = Context(prec=args.precision, rounding=ROUND_HALF_EVEN)

    def dec(x):
        if isinstance(x, float):
            raise TypeError("Binary floating-point inputs are forbidden")
        return Decimal(x)

    class I:
        def __init__(self, x, hi=None):
            if isinstance(x, I):
                if hi is not None:
                    raise TypeError("Ambiguous interval")
                self.lo, self.hi = x.lo, x.hi
            else:
                self.lo = dec(x)
                self.hi = self.lo if hi is None else dec(hi)
            if not self.lo.is_finite() or not self.hi.is_finite() or self.lo > self.hi:
                raise ValueError("Invalid interval")

        def __add__(self, x):
            x = I(x)
            return I(down.add(self.lo, x.lo), up.add(self.hi, x.hi))

        __radd__ = __add__

        def __neg__(self):
            return I(self.hi.copy_negate(), self.lo.copy_negate())

        def __sub__(self, x):
            return self + -I(x)

        def __rsub__(self, x):
            return I(x) + -self

        def __mul__(self, x):
            x = I(x)
            pairs = [(a, b) for a in (self.lo, self.hi) for b in (x.lo, x.hi)]
            return I(min(down.multiply(a, b) for a, b in pairs),
                     max(up.multiply(a, b) for a, b in pairs))

        __rmul__ = __mul__

        def __truediv__(self, x):
            x = I(x)
            if x.lo <= 0 <= x.hi:
                raise ZeroDivisionError("Interval denominator contains zero")
            return self * I(down.divide(Decimal(1), x.hi),
                            up.divide(Decimal(1), x.lo))

        def __rtruediv__(self, x):
            return I(x) / self

        def __pow__(self, n):
            if not isinstance(n, int) or n < 0:
                raise ValueError("Nonnegative integer powers only")
            if n == 0:
                return I(1)
            # Repeated interval products are enclosing even across zero.
            out = I(1)
            for _ in range(n):
                out = out * self
            return out

        def exp(self):
            # Decimal exp is correctly rounded, ROUND_HALF_EVEN. Adjacent
            # representables enclose the exact real value at each endpoint.
            return I(near.next_minus(near.exp(self.lo)),
                     near.next_plus(near.exp(self.hi)))

        def log(self):
            if self.lo <= 0:
                raise ValueError("Log requires a positive interval")
            return I(near.next_minus(near.ln(self.lo)),
                     near.next_plus(near.ln(self.hi)))

        def sqrt(self):
            if self.lo < 0:
                raise ValueError("Sqrt requires a nonnegative interval")
            low = Decimal(0) if self.lo == 0 else near.next_minus(near.sqrt(self.lo))
            return I(low, near.next_plus(near.sqrt(self.hi)))

        def serial(self):
            return {"lower": str(self.lo), "upper": str(self.hi)}

    checks = []
    values = {}

    def record(name, x):
        values[name] = I(x).serial()
        return x

    def lt(name, x, y):
        x, y = I(x), I(y)
        ok = x.hi < y.lo
        checks.append({"name": name, "relation": "<", "left": x.serial(),
                       "right": y.serial(), "pass": ok})
        if not ok:
            raise ArithmeticError("Certificate failure: " + name)

    def le(name, x, y):
        x, y = I(x), I(y)
        ok = x.hi <= y.lo
        checks.append({"name": name, "relation": "<=", "left": x.serial(),
                       "right": y.serial(), "pass": ok})
        if not ok:
            raise ArithmeticError("Certificate failure: " + name)

    # Basic checks of the implementation; these are not a formal kernel proof.
    if (I(1) / 3).lo * 3 > 1 or (I(1) / 3).hi * 3 < 1:
        raise ArithmeticError("Division enclosure sanity check failed")
    try:
        I(0.1)
        raise ArithmeticError("Float rejection failed")
    except TypeError:
        pass
    try:
        I(1) / I(-1, 1)
        raise ArithmeticError("Zero denominator rejection failed")
    except ZeroDivisionError:
        pass

    amin, amax, dmin, dmax = map(I, (".98", "1.02", "1.98", "2.02"))
    rmin, rmax, hmin, hmax = map(I, ("1.27", "1.31", "1.60", "1.96"))
    alpha, eps, gamma, delta = map(I, (".001", ".000002", ".25", ".0001"))
    cmin, cmax, xi, eta = map(I, (".999", "1.002", ".3", ".01"))
    vlo, vhi = I(".17"), I("2.5")
    vminus, vplus = I(".69"), I(".71")

    lt("coefficient upper envelope", (1 + alpha**2)*(1 + alpha), cmax)
    lt("nonnegative base update coefficients", gamma*(1 + cmax), 1)
    lt("class 1 starts below barrier", alpha + eps, xi)
    lt("barrier lower transverse face", delta, gamma*((1+cmin)*eta-eps))
    pi1 = record("pi1_lower", rmin/(1+rmin+hmax))
    lt("class 1 prior mass", ".30", pi1)
    lt("all prior pair ratios below 2: h", hmax, 2)
    lt("all prior pair ratios below 2: r", rmax, 2)
    lt("all prior pair ratios below 2: h/r", hmax/rmin, 2)

    qbar = record("barrier_q_lower", xi-(alpha*(1-xi)+eps+delta/gamma)/cmin)
    lowodds = rmax*((xi-I(".5"))/vlo).exp()
    lowmean = record("low_scale_mean_upper", lowodds/(1+lowodds))
    db = record("barrier_distance_upper", amax**2+dmax**2+2*amax*xi+2*dmax*eta)
    highodds = hmin*(-db/(2*vhi)).exp()
    highmean = record("high_scale_mean_upper", (rmax-amin*highodds)/(1+rmax+highodds))
    lt("low-scale barrier", lowmean, qbar)
    lt("high-scale barrier", highmean, qbar)
    record("low_scale_outward_step_upper", gamma*(alpha*(1-xi)+eps+cmin*(lowmean-xi))+delta)
    record("high_scale_outward_step_upper", gamma*(alpha*(1-xi)+eps+cmin*(highmean-xi))+delta)
    lowcut = record("sufficient_low_cutoff_lower", (I(".5")-xi)/(rmax*(1-qbar)/qbar).log())
    hreq = record("high_scale_required_odds_upper", (rmax*(1-qbar)-qbar)/(amin+qbar))
    highcut = record("sufficient_high_cutoff_upper", db/(2*(hmin/hreq).log()))
    lt("chosen low cutoff", vlo, lowcut)
    lt("chosen high cutoff", highcut, vhi)

    sleft, sentry, ttop, sright = map(I, ("-.01", ".53", ".05", "1.1"))
    dc = record("channel_distance_lower", amin**2+dmin**2+2*amax*sleft-2*dmax*ttop)
    lt("positive channel distance", 0, dc)
    rc = rmin*((sleft-I(".5"))/vminus).exp()
    hc = hmax*(-dc/(2*vplus)).exp()
    wc = record("channel_w2_upper", hc/(1+rc+hc))
    kappa = record("channel_horizontal_derivative_upper", (I(".25")+amax*(amax+2)*wc)/vminus)
    lt("rounded posterior leakage", wc, ".044")
    lt("rounded derivative bound", kappa, ".56")
    de = record("entry_distance_lower", amin**2+dmin**2+2*amin*sentry-2*dmax*ttop)
    lt("positive entry distance", 0, de)
    re = rmin*((sentry-I(".5"))/vplus).exp()
    he = hmax*(-de/(2*vplus)).exp()
    deltaentry = record("entry_mean_surplus_lower", (re-amax*he)/(1+re+he)-sentry)
    lt("rounded entry surplus", ".016", deltaentry)
    # The following deliberately use simple rounded constants from the theorem.
    zeta, surplus = I(".44"), I(".016")
    lt("observation weak-coordinate term is nonnegative", I(1)/500, 1-sentry)
    lt("progress survives update perturbation", delta/gamma, cmin*surplus)
    lt("transverse bottom", delta, gamma*((1+cmin)*eta-eps))
    transverse = record("channel_top_drift_upper", eps-ttop+cmax*(dmax*I(".044")-ttop)+delta/gamma)
    lt("transverse top", transverse, 0)
    q = record("channel_comparison_q", 1-gamma*cmin*zeta)
    star = record("channel_comparison_fixed_point", sentry+(surplus-delta/(gamma*cmin))/zeta)
    s24 = record("hypothetical_s24_lower", star-(star-sleft)*q**24)
    s25 = record("hypothetical_s25_lower", star-(star-sleft)*q**25)
    lt("25-call entry", sentry, s25)
    upperexit = record("first_crossing_upper", sentry+gamma*((alpha+cmax)*(1-sentry)+eps)+delta)
    lt("first crossing does not overshoot", upperexit, sright)
    rightdrift = record("prepared_right_inward_step_lower", gamma*(cmin*(sright-1)-eps)-delta)
    lt("prepared right face", 0, rightdrift)

    # Distance differences are affine in (s,t); parameter monotonicity gives
    # these six exact extrema. All are over full rectangles and parameter box.
    gaps = {
        "G0_to_1": 1-2*I(".4"),
        "G0_to_2": amin**2+dmin**2-2*amin*I(".1")-2*dmin*I(".15"),
        "G1_to_0": 2*I(".53")-1,
        "G1_to_2": amin**2+dmin**2-1+2*(amin+1)*I(".53")-2*dmin*I(".15"),
        "G2_to_0": -2*amax*I(".01")+2*dmax*I("1.73")-amax**2-dmax**2,
        "G2_to_1": -2*(amax+1)*I(".01")+2*dmax*I("1.73")+1-amax**2-dmax**2,
    }
    for key, val in gaps.items():
        record(key+"_gap_lower", val)
        le(key+" gap at least .06", ".06", val)
    lt("template diameter below 3", (amax+1)**2+dmax**2, 9)
    # The minimal template-to-face distance is .1; use .09 to retain slack.
    le("G2 left face distance", ".1", I("1.12")-amax)
    lt("G2 lower face distance", ".1", dmin-I("1.73"))
    lt("G2 upper face distance", ".1", I("2.27")-dmax)
    lt("class 2 initial horizontal lower bound", I("-1.12"), -amax*alpha-eps)
    lt("class 2 initial horizontal upper bound", eps, ".01")
    lt("class 2 initial vertical lower bound", "1.73", dmin-eps)
    lt("class 2 initial vertical upper bound", dmax+eps, "2.27")
    vl = alpha**2+alpha**4
    leakage = record("terminal_leakage_upper_at_alpha_max", 4*(-I(".03")/vl).exp())
    lt("terminal gap to exponential envelope", alpha*(1+alpha**2), ".99")
    lt("terminal invariant-face slack", eps+(1+alpha)*3*leakage+delta/gamma, cmin*I(".09"))
    lt("terminal center initially within 6", I("1.03")+I(".29")+2*eps/cmin, 6)
    # Optional stable fixed-point conclusion on each terminal rectangle.
    lipschitz = record("terminal_lipschitz_upper", I(".75")+gamma*(1+alpha)*9*leakage/vl)
    lt("terminal contraction", lipschitz, ".751")

    # Containment of the original balanced construction. Relax c(v0) to
    # [.999,1.002]. The source note proves monotonicity in all five arguments.
    ao0, ao1, do0, do1 = map(I, (".99", "1.01", "1.99", "2.01"))
    wo0, wo1, vo0, vo1 = map(I, (".039", ".041", ".99", "1.01"))
    roriglo = record("original_prior_r_lower", (I(".5")+ao0*wo0)/(I(".5")-(ao0+1)*wo0))
    rorighi = record("original_prior_r_upper", (I(".5")+ao1*wo1)/(I(".5")-(ao1+1)*wo1))
    def original_h(a, d, w, v, c):
        return w/(I(".5")-(a+1)*w)*((a*a+a+d*d*(1-2*c*w/(1+c)))/(2*v)).exp()
    horiglo = record("original_prior_h_lower", original_h(ao0, do0, wo0, vo1, cmax))
    horighi = record("original_prior_h_upper", original_h(ao1, do1, wo1, vo0, cmin))
    lt("original r lower contained", rmin, roriglo)
    lt("original r upper contained", rorighi, rmax)
    lt("original h lower contained", hmin, horiglo)
    lt("original h upper contained", horighi, hmax)
    derivative_w = record("original_h_log_omega_derivative_lower", 1/wo1-do1**2*cmax/(vo0*(1+cmin)))
    lt("original h increases with omega", 0, derivative_w)
    lt("original h positive d derivative factor", 0, 1-2*cmax*wo1/(1+cmin))

    # Unconditional Gaussian risk endpoints. Analytic monotonicity extends
    # these checks to every 0 < alpha <= 1e-5; no division by a zero interval.
    arisk = I(".00001")
    exp_remainder_ratio = record("residual_exponential_over_alpha3", 12*(-1/(64*arisk)).exp()/arisk**3)
    lt("A_ralpha <= alpha^3 exponential term", exp_remainder_ratio, ".5")
    tailratio = record("two_tail_terms_over_alpha6", 2*128*I(8).sqrt()*(-1/(4000000*arisk**2)).exp()/arisk**6)
    lt("2 T_alpha <= alpha^6", tailratio, 1)
    # Bound the mismatch factor by its worst denominator over alpha <= .001.
    mismatch_factor = record("mismatch_over_alpha6_upper", alpha/cmin**2+1/(2*(1+cmin)**2))
    lt("coefficient mismatch L2 <= alpha^3", mismatch_factor, 1)
    # Simple rational upper bounds sqrt(2)<1.415, 3^(1/4)<1.317,
    # alpha^(3/2)<.000032 avoid a fractional-power dependency in the kernel.
    lt("sqrt 2 rational bound", 2, I("1.415")**2)
    lt("fourth root 3 rational bound", 3, I("1.317")**4)
    lt("alpha power bound", alpha**3, I(".000032")**2)
    l4 = record("global_L4_envelope_upper", I("1.415")*(6+4*delta+2*I("1.317")*(alpha**2+I(".000032"))))
    lt("global L4 envelope below 9", l4, 9)
    lt("fourth moment implies original tail constant", I(9)**4, I(128)**2)
    # (sqrt(A^2+T)+alpha^3)^2 <= 4 alpha^6 + 2T <= 5 alpha^6.

    result = {
        "status": "PASS",
        "precision_decimal_digits": args.precision,
        "arithmetic": "Decimal outward-rounded +,-,*,/; exp,ln,sqrt enlarged by adjacent representables",
        "coverage": "Full continuous parameter box via analytic monotonic reductions; no grid",
        "scope": "Numerical inequalities only; analytic assembly and arithmetic kernel require independent review",
        "parameter_box": {"alpha": "(0,.001]", "a": "[.98,1.02]", "d": "[1.98,2.02]",
                          "pi1/pi0": "[1.27,1.31]", "pi2/pi0": "[1.60,1.96]", "theta": "[-1,1]"},
        "values": values,
        "number_of_checks": len(checks),
        "checks": checks,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": result["status"], "precision": args.precision,
                      "checks": len(checks), "output": str(args.output) if args.output else None}))
    if not args.output:
        print(payload)


if __name__ == "__main__":
    main()
