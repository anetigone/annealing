"""Summarize frozen validation by photograph and noise repeat."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.data/"plan.json").read_text())
    photos = plan["photos"]
    rows = []
    for name in photos:
        folder = args.data/name
        source = json.loads((folder/"summary.json").read_text())
        certificate = json.loads((folder/"interval/certificate.json").read_text())
        assert certificate["status"] == "PASS" and certificate["samples"] == 768
        outputs = np.load(folder/"outputs.npz")
        reference = outputs["states"][1]
        truth = outputs["truth"]
        ref_error = np.sum((reference-truth)**2, axis=1).reshape(3,256)
        comparisons = {}
        for policy, index in [("intermediate", 1), ("low_high", 2)]:
            prediction = outputs["states"][4*index+1]
            delta = np.sum((prediction-truth)**2, axis=1).reshape(3,256)-ref_error
            comparison = {"mean_difference": float(delta.mean()),
                          "per_seed_difference": delta.mean(axis=1).tolist(),
                          "relative_percent": float(100*delta.mean()/ref_error.mean())}
            comparisons[policy] = comparison
        rho = [float(r["rho_upper"]) for r in certificate["records"]]
        rows.append({
            "photo": name, "observations": certificate["samples"],
            "fixed_rho_interval_pass": sum(r == .45 for r in rho),
            "general_interval_pass": len(rho),
            "max_verified_rho": max(rho), "worst_q_interval": certificate["q"],
            "risk154": {p: source["risk"][p]["154"]["mean"]
                        for p in plan["policies"][:3]},
            "comparisons": comparisons,
            "output_differences": {
                p: {str(n): source["distances"][p][str(n)]["mean_output_squared_distance"]
                    for n in [154,1000,5000]} for p in ["intermediate","low_high"]},
            "different_components154": {
                p: source["distances"][p]["154"]["different_dominant_components"]
                for p in ["intermediate","low_high"]}})
    result = {
        "unit": "Per-photo description; noise repeats share truth patches",
        "photos": rows, "observations": sum(r["observations"] for r in rows),
        "fixed_rho_interval_pass": sum(r["fixed_rho_interval_pass"] for r in rows),
        "general_interval_pass": sum(r["general_interval_pass"] for r in rows),
        "selection": plan["selection"],
        "conclusion": "No consistent positive schedule advantage across these photos. "
                      "All exported triplets have interval-certified common terminal balls."
    }
    (args.data/"summary.json").write_text(json.dumps(result, indent=2))
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    names = ["Flower", "Cameraman", "House", "Boat"]
    styles = [("terminal","Terminal","C0","-"), ("intermediate","Intermediate","C1","-"),
              ("low_high","Low/high","C2","-"),
              ("intermediate_frozen_c","Smoothing only","C3",":"),
              ("intermediate_frozen_weights","Coefficient only","C4","--")]
    fig, axes = plt.subplots(1,4,figsize=(12,3),layout="constrained")
    for ax, name, title in zip(axes,photos,names):
        source=json.loads((args.data/name/"summary.json").read_text())
        for policy,label,color,linestyle in styles:
            ax.plot(plan["times"],[source["risk"][policy][str(n)]["mean"]
                    for n in plan["times"]],label=label,color=color,ls=linestyle,marker=".")
        ax.set(xscale="log",xlabel="Calls",title=title)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Squared Euclidean error")
    axes[0].legend(fontsize=7,frameon=False)
    fig.savefig(args.data/"risk_curves.pdf")
    fig.savefig(args.data/"risk_curves.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(12,3),layout="constrained")
    for ax,name,title in zip(axes,photos,names):
        folder=args.data/name
        output=np.load(folder/"outputs.npz")
        cert=json.loads((folder/"interval/certificate.json").read_text())
        candidates=np.load(folder/"candidates.npz")
        q=np.array([1-.25*(1-float(r["rho_upper"]))*float(candidates["c"])
                    for r in cert["records"]])
        ref=output["states"][1]
        delta0=np.sum((output["states"][5]-ref)**2,axis=1)
        times=[154,1000,5000]
        actual=[float(np.mean(np.sum((output["states"][4+i]-
                                     output["states"][i])**2,axis=1))) for i in [1,2,3]]
        upper=[float(np.mean(q**(2*(n-154))*delta0)) for n in times]
        ax.plot(times,actual,"o-",label="Observed float distance")
        ax.plot(times,upper,"s--",label="Exact-map bound")
        ax.set(xscale="log",yscale="log",xlabel="Total calls",title=title)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Mean squared output distance")
    axes[0].legend(fontsize=7,frameon=False)
    fig.savefig(args.data/"distance_curves.pdf")
    fig.savefig(args.data/"distance_curves.png",dpi=150)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!="photos"}))


if __name__ == "__main__":
    main()
