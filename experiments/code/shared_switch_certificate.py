"""Enclose shared-switch states on one explicit local parameter box."""
import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'red_geometry_computation_risk/code'))
from interval import I


def certify():
    alpha = I('.000009', '.000011')
    a, d = I('.99999', '1.00001'), I('1.99999', '2.00001')
    omega, v0, theta = I('.03999', '.04001'), I('.99999', '1.00001'), I('-.00001', '.00001')
    lam = (1+alpha.sq())*(1+theta*alpha)
    tau2 = alpha.sq().sq()
    v = I('.70')
    c = lam*(1-tau2/v)
    c0 = lam*(1-tau2/v0)
    anchor_t = c0*d*omega/(1+c0)
    w0 = I('.5')-(a+1)*omega
    r1 = (I('.5')+a*omega)/w0
    r2 = (omega/w0)*((a.sq()+d.sq()+a-2*d*anchor_t)/(2*v0)).exp()
    e = I('-.000000022', '.000000022')
    gamma = I('.25')
    rows = {}
    for z in (0, 1):
        b1, b2 = z*alpha+e, e
        def step(s, t):
            p1 = r1*((s-I('.5'))/v).exp()
            p2 = r2*((-a.sq()-d.sq()-2*a*s+2*d*t)/(2*v)).exp()
            den = 1+p1+p2
            return ((1-gamma*(alpha+c))*s+gamma*(b1+c*(p1-a*p2)/den),
                    (1-gamma*(1+c))*t+gamma*(b2+c*d*p2/den))

        s, t = b1, b2
        rows[z] = []
        for h in range(26):
            # Monotonicity is in (s,-t), uniformly on this box.
            assert s.lo > Decimal('-.1') and s.hi < Decimal('1.1')
            assert t.lo > Decimal('-.1') and t.hi < Decimal('.15')
            if z == 1 and h <= 12:
                assert s.hi < Decimal('.497')
            if z == 0 and h >= 13:
                assert s.lo > Decimal('.503')
            rows[z].append({'h': h, 's': s.json(), 't': t.json()})
            if h < 25:
                left, top = step(I(s.lo), I(t.hi))
                right, bottom = step(I(s.hi), I(t.lo))
                s, t = I(left.lo, right.hi), I(bottom.lo, top.hi)
    return {'status': 'PASS', 'alpha': alpha.json(), 'a': a.json(), 'd': d.json(),
            'omega': omega.json(), 'v0': v0.json(), 'theta': theta.json(),
            'v': '.70', 'e': e.json(), 'classes': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = certify()
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
