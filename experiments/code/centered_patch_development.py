"""Check reachable geometry for the declared structure-GMM development cases."""
import argparse
import json
from pathlib import Path
import numpy as np
from terminal_ball import terminal_ball_check


POLICIES = ["terminal","intermediate","low_high"]
TIMES = [25,154,1000,5000]


def weights(x,means,pi,v):
    log = np.log(pi)-np.sum((x[:,None]-means)**2,axis=2)/(2*v)
    w = np.exp(log-log.max(axis=1,keepdims=True))
    return w/w.sum(axis=1,keepdims=True)


def curvature(x,means,pi,v,c,gram):
    w = weights(x,means,pi,v)
    shifted = means[None]-(w@means)[:,None]
    cov = np.einsum("nk,nki,nkj->nij",w,shifted,shifted)
    hessian = gram+c*(np.eye(gram.shape[0])-cov/v)
    diagonal = np.diag(gram)+c
    normalized = cov/np.sqrt(diagonal[:,None]*diagonal[None,:])
    rho = c/v*np.linalg.eigvalsh(normalized)[:,-1]
    return hessian,rho


def centers(x,b,means,pi,v,c,gram):
    center = x.copy()
    for _ in range(10):
        w = weights(center,means,pi,v)
        hessian,_ = curvature(center,means,pi,v,c,gram)
        field = center@gram-b+c*(center-w@means)
        center -= np.linalg.solve(hessian,field[...,None])[...,0]
    return center


def check_ball(center,points,b,means,pi,v,c,gram):
    if not np.isfinite(center).all():
        return {"passed":False,"reason":"nonfinite candidate center"}
    radius = max(1.01*np.linalg.norm(points-center,axis=1).max(),1e-4)
    checks = [terminal_ball_check(gram,b,means,pi,v,c,.25,center,radius,j,points)
              for j in range(len(means))]
    best = min(checks,key=lambda item:item["rho_upper"])
    best.update({"radius":float(radius),"reference":int(np.argmin(
        [item["rho_upper"] for item in checks]))})
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--components",type=int,choices=[8,16],required=True)
    parser.add_argument("--terminal",choices=["absolute_.02","train_tau"],required=True)
    args = parser.parse_args()
    plan = json.loads((args.data/"plan.json").read_text())
    folder = args.data/f"k{args.components}_{args.terminal}"
    folder.mkdir(exist_ok=False)
    data = np.load(args.data/"data.npz")
    model = np.load(args.data/f"model_{args.components}.npz")
    means,pi,tau2 = [model[key] for key in ["mu","pi","tau2"]]
    operator,basis,dc = [data[key] for key in ["shape_operator","basis","dc"]]
    gram = operator@operator.T
    b = data["shape_y"]@operator.T
    truth = data["truth"]
    ztrue = truth@basis
    sigma_l = .02 if args.terminal=="absolute_.02" else float(np.sqrt(tau2))
    v_l = float(tau2+sigma_l**2)
    c_l = .1*sigma_l**2/v_l
    assert .25*(np.linalg.eigvalsh(gram)[-1]+.1)<=1
    snapshot = {}
    metrics = {}
    for policy in POLICIES:
        x = b.copy()
        metrics[policy] = {}
        for k in range(5000):
            sigma = sigma_l
            if k<25:
                if policy=="intermediate":
                    sigma=.1
                elif policy=="low_high":
                    sigma=sigma_l if k%2==0 else .3
            v = tau2+sigma**2
            c = .1*sigma**2/v
            x -= .25*(x@gram-b+c*(x-weights(x,means,pi,v)@means))
            if k+1 in TIMES:
                assert np.isfinite(x).all()
                snapshot[policy,k+1]=x.copy()
                reconstructed = x@basis.T+data["dc_y"][:,None]*dc
                error = np.sum((reconstructed-truth)**2,axis=1)
                shape_error = np.sum((x-ztrue)**2,axis=1)
                dc_error = (data["dc_y"]-truth@dc)**2
                assert np.allclose(error,shape_error+dc_error,rtol=1e-10,atol=1e-12)
                field = x@gram-b+c_l*(x-weights(x,means,pi,v_l)@means)
                hess,rho = curvature(x,means,pi,v_l,c_l,gram)
                metrics[policy][str(k+1)] = {
                    "full_risk":float(error.mean()),"structure_risk":float(shape_error.mean()),
                    "brightness_risk":float(dc_error.mean()),
                    "gradient_rms":float(np.sqrt(np.mean(np.sum(field**2,axis=1)))),
                    "negative_curvature_points":int((rho>1).sum()),
                    "maximum_rho":float(rho.max()),
                    "minimum_hessian_eigenvalue":float(np.linalg.eigvalsh(hess)[:,0].min())}
    probes = [ztrue,means]
    probes.append(np.array([(means[i]+means[j])/2 for i in range(len(means)) for j in range(i)]))
    pointset = np.concatenate(probes)
    physical = pointset@basis.T+.5
    admissible = (physical.min(axis=1)>=0)&(physical.max(axis=1)<=1)
    _,rho = curvature(pointset,means,pi,v_l,c_l,gram)
    witnesses = np.flatnonzero(admissible&(rho>1))
    geometry = {"physical_probes":int(admissible.sum()),
                "physical_negative_curvature_probes":len(witnesses),
                "maximum_physical_rho":float(rho[admissible].max())}
    if len(witnesses):
        geometry["witness_structure"]=pointset[witnesses[np.argmax(rho[witnesses])]].tolist()
    center = centers(snapshot["terminal",5000],b,means,pi,v_l,c_l,gram)
    rows = []
    distinct = []
    local_records = []
    common_end = []
    for sample in range(len(truth)):
        starts = np.array([snapshot[p,154][sample] for p in POLICIES])
        checked = check_ball(center[sample],starts,b[sample],means,pi,v_l,c_l,gram)
        checked.update({"sample":sample,"time":154})
        rows.append(checked)
    for policy in POLICIES[1:]:
        reference = snapshot["terminal",5000]
        endpoint = snapshot[policy,5000]
        distance = np.linalg.norm(endpoint-reference,axis=1)
        candidates = np.flatnonzero(distance>1e-3)
        for sample in candidates:
            points = np.array([reference[sample],endpoint[sample]])
            force = np.repeat(b[sample:sample+1],2,axis=0)
            local_centers = centers(points,force,means,pi,v_l,c_l,gram)
            local = [check_ball(local_centers[j],points[j:j+1],b[sample],
                                means,pi,v_l,c_l,gram) for j in range(2)]
            separation = np.linalg.norm(local_centers[0]-local_centers[1])
            disjoint = all(row["passed"] for row in local) and separation>sum(
                row["radius"] for row in local)
            local_records.append({"sample":int(sample),"policy":policy,
                                  "endpoint_distance":float(distance[sample]),
                                  "center_separation":float(separation),
                                  "disjoint_float_balls":bool(disjoint),"local":local})
            if disjoint:
                distinct.append({"sample":int(sample),"policy":policy})
        common_end.append({"policy":policy,"distant_endpoints":len(candidates),
                           "mean_squared_output_distance154":float(np.mean(np.sum(
                               (snapshot[policy,154]-snapshot["terminal",154])**2,axis=1))),
                           "mean_squared_output_distance5000":float(np.mean(distance**2))})
    summary = {"components":args.components,"terminal_rule":args.terminal,
               "terminal_sigma":sigma_l,"tau2":float(tau2),"terminal_c":float(c_l),
               "development_samples":len(truth),"metrics":metrics,"geometry":geometry,
               "common_ball_float_pass":sum(row["passed"] for row in rows),
               "distinct_endpoint_float_certificates":distinct,"output_distances":common_end,
               "scope":plan["selection"]+"; floating sufficient checks, not interval certification."}
    (folder/"summary.json").write_text(json.dumps(summary,indent=2))
    (folder/"checks.json").write_text(json.dumps({"rows":rows,"local":local_records},indent=2))
    np.savez_compressed(folder/"candidates.npz",gram=gram,operator=operator,b=b,means=means,
                        pi=pi,v=v_l,c=c_l,gamma=.25,centers=center,
                        states=np.array([snapshot[p,154] for p in POLICIES]))
    np.savez_compressed(folder/"outputs.npz",truth=truth,truth_structure=ztrue,
                        states=np.array([snapshot[p,n] for p in POLICIES for n in TIMES]))
    print(json.dumps({"components":args.components,"terminal":args.terminal,
                      "risk154":{p:metrics[p]["154"]["full_risk"] for p in POLICIES},
                      "geometry":{k:v for k,v in geometry.items() if k!="witness_structure"},
                      "common_float_pass":summary["common_ball_float_pass"],
                      "disjoint_candidates":len(distinct)}))


if __name__=="__main__":
    main()
