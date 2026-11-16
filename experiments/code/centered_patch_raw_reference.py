"""Post-hoc matched-split raw-16D reference for the centered development study."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from centered_patch_development import weights,curvature


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--model",type=Path,help="Reuse an exported matched-split model")
    args=parser.parse_args()
    folder=args.out or args.data/"raw_reference"
    folder.mkdir(exist_ok=False)
    (folder/"plan.json").write_text(json.dumps({
        "role":"Post-hoc development attribution, not a primary preregistered comparison",
        "training":"Same 2048 spatially separated raw china patches, K=8, n_init=5, seed20261120",
        "evaluation":"Same 256 development patches and same observations",
        "terminal_rules":["absolute_.02","train_tau"],"gamma":.25,"lambda":.1,
        "policies":["terminal","intermediate","low_high"],
        "scope":"Raw16D proper isotropic GMM changes the DC prior as well as structure fitting"
    },indent=2))
    data=np.load(args.data/"data.npz")
    if args.model is None:
        fit=KMeans(n_clusters=8,random_state=20261120,n_init=5).fit(data["train"])
        means=fit.cluster_centers_
        pi=np.bincount(fit.labels_,minlength=8)/len(data["train"])
        tau2=float(np.mean((data["train"]-means[fit.labels_])**2))
    else:
        saved=np.load(args.model)
        means,pi,tau2=saved["mu"],saved["pi"],float(saved["tau2"])
    centered=means-pi@means
    covariance=(centered.T*pi)@centered
    values,vectors=np.linalg.eigh(covariance)
    gram=data["operator"]@data["operator"].T
    b=data["observation"]@data["operator"].T
    results={}
    for rule in ["absolute_.02","train_tau"]:
        sigma_l=.02 if rule=="absolute_.02" else float(np.sqrt(tau2))
        v_l=tau2+sigma_l**2
        c_l=.1*sigma_l**2/v_l
        risk={}
        ends={}
        snapshots=[]
        for policy in ["terminal","intermediate","low_high"]:
            x=b.copy()
            risk[policy]={}
            for k in range(5000):
                sigma=sigma_l
                if k<25:
                    if policy=="intermediate":
                        sigma=.1
                    elif policy=="low_high":
                        sigma=sigma_l if k%2==0 else .3
                v=tau2+sigma**2
                c=.1*sigma**2/v
                x-=.25*(x@gram-b+c*(x-weights(x,means,pi,v)@means))
                if k+1 in [25,154,1000,5000]:
                    snapshots.append(x.copy())
                    error=x-data["truth"]
                    shape=np.sum((error@data["basis"])**2,axis=1)
                    brightness=(error@data["dc"])**2
                    full=np.sum(error**2,axis=1)
                    assert np.allclose(full,shape+brightness,atol=1e-12)
                    force=x@gram-b+c_l*(x-weights(x,means,pi,v_l)@means)
                    risk[policy][str(k+1)]={"full_risk":float(full.mean()),
                                            "structure_risk":float(shape.mean()),
                                            "brightness_risk":float(brightness.mean()),
                                            "gradient_rms":float(np.sqrt(np.mean(np.sum(force**2,axis=1))))}
            ends[policy]=x.copy()
        distances={p:float(np.mean(np.sum((ends[p]-ends["terminal"])**2,axis=1)))
                   for p in ["intermediate","low_high"]}
        results[rule]={"terminal_sigma":sigma_l,"c_l":c_l,"metrics":risk,
                       "output_squared_distances5000":distances,
                       "distant_endpoints":{p:int(np.sum(np.linalg.norm(
                           ends[p]-ends["terminal"],axis=1)>1e-3))
                           for p in ["intermediate","low_high"]}}
        np.savez_compressed(folder/f"{rule}_states.npz",states=np.array(snapshots),
                            b=b,truth=data["truth"],operator=data["operator"],
                            means=means,pi=pi,tau2=tau2,v=v_l,c=c_l,gamma=.25)
    output={"tau2":tau2,"tau":float(np.sqrt(tau2)),
            "top_mean_covariance_fraction":float(values[-1]/values.sum()),
            "top_direction_dc_alignment_squared":float((vectors[:,-1]@data["dc"])**2),
            "results":results,
            "scope":"Post-hoc matched-split reference, no independent performance guarantee"}
    (folder/"summary.json").write_text(json.dumps(output,indent=2))
    np.savez_compressed(folder/"model.npz",mu=means,pi=pi,tau2=tau2)
    print(json.dumps({"tau":output["tau"],"top_fraction":output["top_mean_covariance_fraction"],
                      "distances":{k:r["distant_endpoints"] for k,r in results.items()}}))


if __name__=="__main__":
    main()
