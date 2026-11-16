"""Outward-rounded checks of the scalar bounds in Supplement S27."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import mpmath


def certify(digits):
    iv = mpmath.iv
    iv.dps = digits
    N = iv.mpf
    checks = {}

    def lt(name, left, right):
        left, right = N(left), N(right)
        if not left.b < right.a:
            raise ArithmeticError(f"{name}: {left} is not below {right}")
        checks[name] = {"lhs_interval": str(left), "rhs_interval": str(right)}

    amin, amax, dmin, dmax = map(N, (".98", "1.02", "1.98", "2.02"))
    rmin, rmax, hmin, hmax = map(N, ("1.27", "1.31", "1.60", "1.96"))
    alpha, gamma, f = map(N, (".001", ".25", ".020"))
    e1, e2, cmin, cmax = map(N, (".00025", ".005", ".999", "1.002"))
    blo, bhi, bottom, btop = map(N, ("-.5", ".3", "-.027", ".51"))
    vlo, vhi = N(".17"), N("2.5")
    lt("own-coordinate monotonicity", gamma*(1+cmax), 1)
    lt("class-1 mass for risk bound", ".147", (1-bhi)**2*rmin/(1+rmin+hmax))

    qmin = (amin**2+dmin**2+2*amax*blo-2*dmax*btop)/2
    low_h = hmax*iv.exp(-qmin/vlo)
    lt("barrier low third weight", low_h/(1+low_h), ".01")
    lt("barrier low negative mean", "-.30", -amax*low_h/(1+low_h))
    for l, u in (("2.5", "4"), ("4", "8"), ("8", None)):
        r = rmin*iv.exp(-1/N(l))
        h = hmax*iv.exp(-qmin/N(u)) if u else hmax
        lt(f"barrier high third weight from {l}", h/(1+r+h), ".481")
        lt(f"barrier high negative mean from {l}", "-.30", (r-amax*h)/(1+r+h))
    r = rmax*iv.exp((bhi-N(".5"))/vlo)
    lt("right face low mean", r/(1+r), ".287729")
    for l, dstar, mean in ((bottom, dmax, ".29300"), (N(".15"), dmin, ".25984")):
        qmax = (amax**2+dmax**2+2*amax*bhi-2*dstar*l)/2
        h = hmin*iv.exp(-qmax/vhi)
        lt(f"right face high mean {mean}", (rmax-amin*h)/(1+rmax+h), mean)
    for mean, t, bound in ((".287729", btop, "-.00020"),
                            (".29300", N(".15"), "-.00069"),
                            (".25984", btop, "-.0071")):
        drift = gamma*(alpha*(1-bhi)+e1+cmin*(N(mean)-bhi)+(1-bhi)*f**2+t*f)
        lt(f"right face drift from mean {mean}", drift, bound)
    lt("left face inward drift", ".047",
       gamma*(alpha*(1-blo)-e1+cmin*(-blo-N(".30"))-f*btop))
    lt("bottom face inward drift", ".0047",
       gamma*(-e2-(1+cmin)*bottom-(1-blo)*f+bottom*f**2))
    lt("top face outward drift",
       gamma*(e2-btop+cmax*(dmax*N(".481")-btop)+(1-blo)*f+btop*f**2), "-.0030")

    smin, smax, tmax = alpha-e1, alpha+e1+f**2, e2+f
    lt("class-1 classifier horizontal threshold", alpha/2, smin)
    lt("class-1 initial right barrier clearance", smax, bhi)
    lt("class-1 initial bottom clearance", bottom, -tmax)
    lt("class-1 initial top barrier clearance", tmax, btop)
    lt("class-1 classifier vertical threshold", tmax, 1)
    lt("class-0 classifier horizontal threshold", e1, alpha/2)
    lt("class-0 classifier vertical threshold", e2, 1)
    lt("class-0 initial rectangle clearance", max(e1, e2), ".1")
    rot2 = (f+f**2)*dmax
    lt("class-2 initial horizontal clearance", alpha*amax+e1+rot2, ".043")
    lt("class-2 initial vertical lower bound", "1.933", dmin-e2-rot2)
    lt("class-2 initial vertical upper bound", dmax+e2+rot2, "2.067")

    sl, se, su, top = map(N, ("-.01", ".53", "1.1", ".06"))
    vm, vp = N(".69"), N(".71")
    lt("class-1 starts right of channel left", sl, smin)
    lt("class-1 starts left of channel entrance", smax, se)
    lt("class-1 starts below channel top", tmax, top)
    lt("aligned data force through channel", e1, alpha*(1-se))
    r = rmin*iv.exp((sl-N(".5"))/vm)
    h = hmax*iv.exp(-(amin**2+dmin**2+2*amax*sl-2*dmax*top)/(2*vp))
    w = h/(1+r+h)
    lt("channel third weight", w, ".04508")
    lt("channel derivative", (N(".25")+amax*(amax+2)*w)/vm, ".564")
    r = rmin*iv.exp((se-N(".5"))/vp)
    h = hmax*iv.exp(-(amin**2+dmin**2+2*amin*se-2*dmax*top)/(2*vp))
    lt("entrance surplus", ".01563", (r-amax*h)/(1+r+h)-se)
    rot_s = gamma*((1-sl)*f**2+top*f)
    rot_t = gamma*((1-sl)*f+top*f**2)
    lt("horizontal rotation correction", rot_s, ".000402")
    lt("channel top drift", gamma*(e2-top+cmax*(dmax*N(".04508")-top))+rot_t, "-.0009")
    lt("channel bottom drift", ".007", gamma*(-e2-(1+cmin)*bottom)-rot_t)
    advance = gamma*cmin*N(".01563")-N(".000402")
    rate = 1-gamma*cmin*N(".436")
    lt("comparison rate positive", 0, rate)
    lt("comparison rate below one", rate, 1)
    lt("prepared left face inward", 0, advance)
    s = smin
    for _ in range(25):
        s = rate*s+(1-rate)*se+advance
    lt("entry at 25", ".5306", s)
    lt("overshoot", se+gamma*((alpha+cmax)*(1-se)+e1)+N(".000402"), ".649")
    lt("prepared right face outward drift", gamma*((alpha+cmin)*(1-su)+e1)+N(".000402"), 0)

    gaps = {"G0 to 1": 1-2*N(".4"),
            "G0 to 2": amin**2+dmin**2-2*amin*N(".1")-2*dmin*N(".15"),
            "G1 to 2": amin**2+dmin**2-1+2*(amin+1)*se-2*dmin*N(".15"),
            "G2 to 0": 2*dmin*N("1.73")-2*amax*N(".05")-amax**2-dmax**2}
    # On G2 the distance gap to mu1 is even larger than that to mu0.
    for pair, gap in gaps.items():
        lt(f"terminal gap {pair}", "1.62" if pair == "G2 to 0" else ".06", gap)
    assert 2*Fraction(".53")-1 == Fraction(".06")  # Exact G1 to 0 gap.
    lt("class-2 gap increment", 0, 1-2*N(".05"))
    lt("prior pair ratio envelope", max(hmax, rmax, 1/hmin, 1/rmin, hmax/rmin, rmax/hmin), 2)
    lt("template diameter", (1+amax)**2+dmax**2, 9)
    vmax = N(".001")
    leakage = iv.exp(-N(".03")/vmax)
    lt("rotation correction on terminal faces", gamma*(f+f**2)*N("1.1"), ".021")
    lt("aligned terminal face margin", ".021", gamma*(cmin*N(".09")-e2-12*cmax*leakage))
    lt("terminal linear contraction bound", 1-gamma*(alpha+cmin), ".751")
    lt("terminal contraction including leakage", N(".751")+36*gamma*cmax*leakage/vmax, 1)
    return {"precision": digits, "checks_passed": len(checks),
            "exact_terminal_gap_g1_to_g0": ".06", "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=45)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.precision < 40:
        parser.error("Use at least 40 decimal digits")
    result = certify(args.precision)
    if args.out:
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(f"PASS: {result['checks_passed']} strict scalar bounds at {args.precision} digits")


if __name__ == "__main__":
    main()
