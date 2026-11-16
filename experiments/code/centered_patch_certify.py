"""Export disjoint terminal balls and verify physical negative curvature."""
import argparse
import json
from pathlib import Path
import numpy as np
from mpmath import iv
from centered_patch_development import centers,check_ball,curvature
from patch_terminal_interval import exact,norm


def negative_rayleigh(means,pi,gram,v,c,state):
    hess,_ = curvature(state[None],means,pi,v,c,gram)
    _,vectors = np.linalg.eigh(hess[0])
    direction = vectors[:,0]
    iv.dps=60
    size=len(state)
    x=[exact(z) for z in state]
    u=[exact(z) for z in direction]
    mu=[[exact(z) for z in row] for row in means]
    raw=[exact(pi[i])*iv.exp(-sum((x[k]-row[k])**2 for k in range(size))/(2*exact(v)))
         for i,row in enumerate(mu)]
    weights=[r/sum(raw) for r in raw]
    projections=[sum(u[k]*row[k] for k in range(size)) for row in mu]
    variance=sum(w*t*t for w,t in zip(weights,projections))-sum(
        w*t for w,t in zip(weights,projections))**2
    squared=sum(t*t for t in u)
    value=(sum(u[i]*exact(gram[i,j])*u[j] for i in range(size) for j in range(size))
           +exact(c)*squared-exact(c)/exact(v)*variance)/squared
    assert value.b<0
    return {"state":state.tolist(),"direction":direction.tolist(),"rayleigh_interval":str(value)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--time",type=int,choices=[1000,5000],default=5000)
    parser.add_argument("--out",type=Path,help="Fresh certificate output directory")
    args=parser.parse_args()
    out=args.out or args.data/("certification" if args.time==5000 else "certification_1000")
    out.mkdir(exist_ok=False)
    data=np.load(args.data/"data.npz")
    witnesses=[]
    for components in [8,16]:
        for terminal in ["absolute_.02","train_tau"]:
            folder=args.data/f"k{components}_{terminal}"
            summary=json.loads((folder/"summary.json").read_text())
            model=np.load(args.data/f"model_{components}.npz")
            source=np.load(folder/"candidates.npz")
            means,pi=model["mu"],model["pi"]
            state=np.array(summary["geometry"]["witness_structure"])
            witness=negative_rayleigh(means,pi,source["gram"],float(source["v"]),
                                      float(source["c"]),state)
            decoded=[]
            for row in data["basis"]:
                value=iv.mpf(".5")+sum(exact(row[k])*exact(state[k]) for k in range(15))
                assert value.a>=0 and value.b<=1
                decoded.append(str(value))
            witness.update({"case":folder.name,"pixel_intervals":decoded})
            witnesses.append(witness)
    (out/"physical_curvature.json").write_text(json.dumps({
        "status":"PASS","digits":60,"witnesses":witnesses,
        "scope":"Four selected development states, not a whole-domain nonconvexity map"},indent=2))
    folder=args.data/"k8_train_tau"
    summary=json.loads((folder/"summary.json").read_text())
    selected=sorted({r["sample"] for r in summary["distinct_endpoint_float_certificates"]})
    source=np.load(folder/"candidates.npz")
    saved=np.load(folder/"outputs.npz")
    means,pi=source["means"],source["pi"]
    v,c=float(source["v"]),float(source["c"])
    gram=source["gram"]
    ball_rows=[]
    ball_centers=[]
    ball_b=[]
    ball_states=[]
    ordering=[]
    for sample in selected:
        for policy,offset in [("terminal",0),("intermediate",4)]:
            snapshot_index=2 if args.time==1000 else 3
            point=saved["states"][offset+snapshot_index,sample]
            points=np.repeat(point[None],3,axis=0)
            if policy=="intermediate":
                points[1]=saved["states"][8+snapshot_index,sample]
            late=saved["states"][offset+3,sample]
            center=centers(late[None],source["b"][sample:sample+1],means,pi,v,c,gram)[0]
            checked=check_ball(center,points,source["b"][sample],means,pi,v,c,gram)
            assert checked["passed"]
            row_id=len(ball_rows)
            checked.update({"sample":row_id,"time":args.time,"source_sample":sample,"policy":policy})
            ball_rows.append(checked)
            ball_centers.append(center)
            ball_b.append(source["b"][sample])
            ball_states.append(points)
    for position,sample in enumerate(selected):
        first,second=2*position,2*position+1
        difference=[exact(x)-exact(y) for x,y in zip(ball_centers[first],ball_centers[second])]
        separation=norm(difference)-exact(ball_rows[first]["radius"])-exact(ball_rows[second]["radius"])
        assert separation.a>0
        errors=[]
        exact_truth=[sum(exact(data["truth"][sample,j])*exact(data["basis"][j,k])
                         for j in range(16)) for k in range(15)]
        for ball in [first,second]:
            distance=norm([exact(x)-y for x,y in zip(ball_centers[ball],exact_truth)])
            radius=exact(ball_rows[ball]["radius"])
            lower=distance-radius
            upper=distance+radius
            assert lower.a>0
            errors.append((lower,upper))
        better=0 if errors[0][1].b<errors[1][0].a else 1
        assert errors[better][1].b<errors[1-better][0].a
        ordering.append({"source_sample":sample,
                         "separation_minus_radii":str(separation),
                         "terminal_error_distance":list(map(str,errors[0])),
                         "intermediate_error_distance":list(map(str,errors[1])),
                         "lower_error_ball":["terminal","intermediate"][better]})
    (out/"checks.json").write_text(json.dumps({"rows":ball_rows,"ordering":ordering},indent=2))
    np.savez_compressed(out/"candidates.npz",gram=gram,operator=source["operator"],
                        b=np.array(ball_b),means=means,pi=pi,v=v,c=c,gamma=.25,
                        centers=np.array(ball_centers),
                        states=np.array(ball_states).transpose(1,0,2))
    print(json.dumps({"negative_curvature_witnesses":len(witnesses),
                      "disjoint_ball_pairs":len(ordering),"risk_ordering":ordering}))


if __name__=="__main__":
    main()
