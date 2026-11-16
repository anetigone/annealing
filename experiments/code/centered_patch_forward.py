"""Enclose complete selected development trajectories through terminal-ball entry."""
import argparse
import json
from pathlib import Path
import numpy as np
from mpmath import iv
from patch_terminal_interval import exact,norm


def step(state,b,means,pi,diagonal,v,c,gamma):
    if v is None:
        w=pi
        average=[sum(pi[i]*means[i][j] for i in range(len(means))) for j in range(len(b))]
    else:
        raw=[pi[i]*iv.exp(-sum((state[j]-mu[j])**2 for j in range(len(b)))/(2*v))
             for i,mu in enumerate(means)]
        total=sum(raw)
        w=[r/total for r in raw]
        average=[sum(w[i]*means[i][j] for i in range(len(means))) for j in range(len(b))]
    result=[(1-gamma*(diagonal[j]+c))*state[j]+gamma*b[j]+gamma*c*average[j]
            for j in range(len(b))]
    return result,w


def tube_step(state,radius,b,means,pi,diagonal,v,c,gamma,pair_distances,diameter,
              normalized_pairs=None):
    enclosed,w=step(state,b,means,pi,diagonal,v,c,gamma)
    middle=[value.mid for value in enclosed]
    errors=[max(abs((value-center).a),abs((value-center).b))
            for value,center in zip(enclosed,middle)]
    arithmetic_error=iv.sqrt(sum(error**2 for error in errors))
    # On the radius-R ball, w_i(x)w_j(x) <=
    # exp(2 R D/v) w_i(center)w_j(center).
    # C is a sum of PSD pair terms, hence its spectral norm is
    # bounded by this factor times the center covariance trace.
    base=1-gamma*(min(diagonal,key=lambda x:float(x.a))+c)
    factor=base
    if v is not None:
        trace=sum(w[i]*w[j]*pair_distances[i,j]
                  for i in range(len(means)) for j in range(i))
        inflation=iv.exp(2*radius*diameter/v)
        factor+=gamma*c/v*inflation*trace
        if normalized_pairs is not None:
            normalized_trace=sum(w[i]*w[j]*normalized_pairs[i,j]
                                 for i in range(len(means)) for j in range(i))
            rho=(c/v*inflation*normalized_trace).b
            eigenvalue=(min(diagonal,key=lambda x:float(x.a)) if rho<1 else
                        max(diagonal,key=lambda x:float(x.b)))
            curvature=(1-rho)*(eigenvalue+c)
            alternative=1-gamma*curvature
            factor=min(factor.b,alternative.b)
    next_radius=(factor*radius+arithmetic_error).b
    return middle,next_radius


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,help="Fresh tube-certificate output directory")
    args=parser.parse_args()
    folder=args.out or args.data/"forward_tube_interval"
    folder.mkdir(exist_ok=False)
    iv.dps=60
    source=np.load(args.data/"k8_train_tau/candidates.npz")
    model=np.load(args.data/"model_8.npz")
    balls=np.load(args.data/"certification_1000/candidates.npz")
    meta=json.loads((args.data/"certification_1000/checks.json").read_text())
    means=[[exact(x) for x in row] for row in model["mu"]]
    pi=[exact(x) for x in model["pi"]]
    diagonal=[exact(x) for x in np.diag(source["gram"])]
    pair_distances={(i,j):sum((means[i][k]-means[j][k])**2 for k in range(15))
                    for i in range(len(means)) for j in range(i)}
    diameter=max([iv.sqrt(value).b for value in pair_distances.values()],
                 key=lambda value:float(value.b))
    tau2=float(model["tau2"])
    sigma_l=float(np.sqrt(tau2))
    coefficients={}
    for name,sigma in [("terminal",sigma_l),("middle",.1),("high",.3),("high1",1.)]:
        variance=float(tau2+sigma**2)
        coefficients[name]=(exact(variance),exact(.1*sigma**2/variance))
    coefficients["limit"]=(None,exact(.1))
    gamma=exact(.25)
    selected=sorted({r["source_sample"] for r in meta["rows"]})
    records=[]
    for sample in selected:
        b=[exact(x) for x in source["b"][sample]]
        for policy in ["terminal","intermediate","low_high","high1","high_limit"]:
            state=b.copy()
            radius=iv.mpf(0)
            for k in range(1000):
                key="terminal"
                if k<25:
                    if policy=="intermediate":
                        key="middle"
                    elif policy=="low_high":
                        key="terminal" if k%2==0 else "high"
                    elif policy=="high1":
                        key="high1"
                    elif policy=="high_limit":
                        key="limit"
                variance,c=coefficients[key]
                state,radius=tube_step(state,radius,b,means,pi,diagonal,variance,c,gamma,
                                       pair_distances,diameter)
            mode="terminal" if policy=="terminal" else "intermediate"
            row=next(r for r in meta["rows"] if r["source_sample"]==sample and r["policy"]==mode)
            center=balls["centers"][row["sample"]]
            distance=norm([x-exact(y) for x,y in zip(state,center)])
            margin=exact(row["radius"])-distance-radius
            assert margin.a>0,(sample,policy,str(margin))
            records.append({"sample":sample,"policy":policy,"ball":mode,
                            "entry_margin":str(margin),
                            "propagated_radius":str(radius),
                            "scope":"Exact map at exported observation and coefficients"})
            print(f"PASS sample {sample}, {policy}, entry margin {margin}",flush=True)
    (folder/"certificate.json").write_text(json.dumps({
        "status":"PASS","digits":60,"time":1000,"records":records,
        "selection":"Two development samples selected by persistent endpoint distance, not error",
        "method":"Euclidean error tube, interval point steps and pair-trace Lipschitz envelopes",
        "scope":"Full exact-arithmetic trajectory enclosure from exported b, fixed fitted model. "
                "No noise-event or population-risk guarantee. Infinite high scale is a limit map."
    },indent=2))


if __name__=="__main__":
    main()
