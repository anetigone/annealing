"""Local disjoint balls and full interval trajectories for the raw development candidate."""
import argparse
import json
from pathlib import Path
import numpy as np
from mpmath import iv
from centered_patch_model import structure_basis
from centered_patch_development import centers,check_ball,curvature
from centered_patch_certify import negative_rayleigh
from centered_patch_forward import tube_step
from patch_terminal_interval import exact,norm


def export_balls(root,out):
    data=np.load(root/"raw_reference_detail/absolute_.02_states.npz")
    gram=data["operator"]@data["operator"].T
    v,c=float(data["v"]),float(data["c"])
    differences=np.linalg.norm(data["states"][7]-data["states"][3],axis=1)
    selected=np.flatnonzero(differences>1e-3)
    assert len(selected)==1
    sample=int(selected[0])
    mean,pi=data["means"],data["pi"]
    rows=[]
    states=[]
    fixed=[]
    for label,offset in [("terminal",0),("intermediate",4)]:
        point=data["states"][offset+3,sample]
        center=centers(point[None],data["b"][sample:sample+1],mean,pi,v,c,gram)[0]
        starts=np.repeat(point[None],3,axis=0)
        if label=="terminal":
            starts[1]=data["states"][11,sample]
        checked=check_ball(center,starts,data["b"][sample],mean,pi,v,c,gram)
        assert checked["passed"]
        checked.update({"sample":len(rows),"time":5000,"policy":label,"source_sample":sample})
        rows.append(checked);states.append(starts);fixed.append(center)
    iv.dps=60
    separation=norm([exact(x)-exact(y) for x,y in zip(*fixed)])-sum(
        exact(row["radius"]) for row in rows)
    assert separation.a>0
    error_intervals=[]
    for center,row in zip(fixed,rows):
        distance=norm([exact(x)-exact(y) for x,y in zip(center,data["truth"][sample])])
        error_intervals.append((distance-exact(row["radius"]),distance+exact(row["radius"])))
    assert error_intervals[1][1].b<error_intervals[0][0].a
    line=np.linspace(fixed[0],fixed[1],101)
    _,rho=curvature(line,mean,pi,v,c,gram)
    witness=negative_rayleigh(mean,pi,gram,v,c,line[np.argmax(rho)])
    (out/"checks.json").write_text(json.dumps({"rows":rows,
        "source_sample":sample,"separation_minus_radii":str(separation),
        "terminal_error_norm_interval":list(map(str,error_intervals[0])),
        "intermediate_error_norm_interval":list(map(str,error_intervals[1])),
        "line_negative_curvature":witness},indent=2))
    np.savez_compressed(out/"candidates.npz",gram=gram,operator=data["operator"],
                        b=np.repeat(data["b"][sample:sample+1],2,axis=0),means=mean,pi=pi,
                        v=v,c=c,gamma=.25,centers=np.array(fixed),
                        states=np.array(states).transpose(1,0,2))
    return data,sample,rows,fixed


def forward(data,sample,rows,fixed,out):
    dc,basis=structure_basis()
    transform=np.column_stack([dc,basis])
    exact_transform=[[exact(x) for x in row] for row in transform]
    original_mu=[[exact(x) for x in row] for row in data["means"]]
    means=[[sum(row[j]*exact_transform[j][k] for j in range(16)) for k in range(16)]
           for row in original_mu]
    b=[sum(exact(data["b"][sample,j])*exact_transform[j][k] for j in range(16))
       for k in range(16)]
    diagonal_np=np.diag(transform.T@data["operator"]@data["operator"].T@transform)
    assert np.array_equal(transform.T@data["operator"]@data["operator"].T@transform,
                          np.diag(diagonal_np))
    diagonal=list(map(exact,diagonal_np))
    pairs={(i,j):sum((means[i][k]-means[j][k])**2 for k in range(16))
           for i in range(len(means)) for j in range(i)}
    diameter=max([iv.sqrt(d).b for d in pairs.values()],key=lambda d:float(d.b))
    tau2=float(data["tau2"])
    stages={}
    for name,sigma in [("terminal",.02),("middle",.1),("high",.3),("high1",1.)]:
        v=float(tau2+sigma**2);c=.1*sigma**2/v
        stages[name]=(exact(v),exact(c))
    stages["limit"]=(None,exact(.1))
    normalized={}
    for name,(_,c) in stages.items():
        normalized[name]={(i,j):sum((means[i][k]-means[j][k])**2/(diagonal[k]+c)
                                   for k in range(16)) for i,j in pairs}
    pi=[exact(x) for x in data["pi"]]
    result=[]
    for policy in ["terminal","intermediate","low_high","high1","high_limit"]:
        state=b.copy();radius=iv.mpf(0)
        for k in range(5000):
            name="terminal"
            if k<25:
                if policy=="intermediate":name="middle"
                elif policy=="low_high":name="terminal" if k%2==0 else "high"
                elif policy=="high1":name="high1"
                elif policy=="high_limit":name="limit"
            v,c=stages[name]
            state,radius=tube_step(state,radius,b,means,pi,diagonal,v,c,exact(.25),
                                   pairs,diameter,normalized[name])
        ball=1 if policy=="intermediate" else 0
        transformed_center=[sum(exact(fixed[ball][j])*exact_transform[j][k]
                                for j in range(16)) for k in range(16)]
        distance=norm([x-y for x,y in zip(state,transformed_center)])
        margin=exact(rows[ball]["radius"])-distance-radius
        assert margin.a>0,(policy,str(margin))
        record={"policy":policy,"ball":ball,"entry_margin":str(margin),
                "propagated_radius":str(radius)}
        result.append(record)
        print(f"PASS raw sample {sample}, {policy}, margin {margin}",flush=True)
    (out/"forward_certificate.json").write_text(json.dumps({
        "status":"PASS","sample":sample,"digits":60,"time":5000,"records":result,
        "method":"Dyadic orthogonal coordinates and Euclidean error tubes",
        "scope":"Exact trajectories at exported observation and coefficients. "
                "One selected development patch, no population or restricted-controller guarantee."
    },indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,help="Fresh certificate output directory")
    args=parser.parse_args()
    out=args.out or args.data/"raw_certification"
    out.mkdir(exist_ok=False)
    iv.dps=60
    data,sample,rows,fixed=export_balls(args.data,out)
    forward(data,sample,rows,fixed,out)


if __name__=="__main__":
    main()
