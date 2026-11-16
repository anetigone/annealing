"""Collect the small centered-prior development study without selecting a winner."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    args=parser.parse_args()
    models=json.loads((args.data/"models.json").read_text())["models"]
    records=[]
    for components in [8,16]:
        for terminal in ["absolute_.02","train_tau"]:
            source=json.loads((args.data/f"k{components}_{terminal}/summary.json").read_text())
            records.append({
                "components":components,"terminal_rule":terminal,
                "sigma_l":source["terminal_sigma"],"c_l":source["terminal_c"],
                "risk154":{p:source["metrics"][p]["154"]["full_risk"]
                           for p in source["metrics"]},
                "risk5000":{p:source["metrics"][p]["5000"]["full_risk"]
                            for p in source["metrics"]},
                "physical_negative_curvature_probes":source["geometry"]["physical_negative_curvature_probes"],
                "physical_probes":source["geometry"]["physical_probes"],
                "common_ball_float_pass":source["common_ball_float_pass"],
                "distinct_samples":sorted({r["sample"] for r in
                    source["distinct_endpoint_float_certificates"]})})
    raw=json.loads((args.data/"raw_probe/summary.json").read_text())
    base=raw["metrics"]["terminal"]["5000"]["risk"]
    middle=raw["metrics"]["intermediate"]["5000"]["risk"]
    summary={
        "models":models,"primary_settings":records,
        "scope":"Four declared development settings, one spatially held-out part of china.jpg",
        "raw_reference":"Post-hoc matched-split 16D control revealed a separate candidate",
        "selected_raw_patch":37,"raw_microvalidation_noise_draws":256,
        "raw_microvalidation_risk5000":{p:raw["metrics"][p]["5000"]["risk"]
                                       for p in raw["metrics"]},
        "raw_microvalidation_middle_relative_percent":100*(middle-base)/base,
        "certificates":{
            "centered_physical_negative_states":4,
            "centered_disjoint_ball_pairs":2,
            "centered_forward_trajectories":10,
            "raw_disjoint_ball_pairs":1,
            "raw_forward_trajectories":5},
        "unfinished":"No common restricted-region proof or population risk separation",
        "external_test_photos":"Not used for the new models"}
    (args.data/"summary.json").write_text(json.dumps(summary,indent=2))
    data=np.load(args.data/"data.npz")
    shape=np.load(args.data/"k8_train_tau/outputs.npz")
    raw_source=np.load(args.data/"raw_reference_detail/absolute_.02_states.npz")
    examples=[]
    for sample in [68,90]:
        terminal=shape["states"][3,sample]@data["basis"].T+data["dc_y"][sample]*data["dc"]
        middle=shape["states"][7,sample]@data["basis"].T+data["dc_y"][sample]*data["dc"]
        examples.append((f"Structure dev {sample}",data["truth"][sample],
                         data["observation"][sample],terminal,middle))
    examples.append(("Raw dev 37",raw_source["truth"][37],data["observation"][37],
                     raw_source["states"][3,37],raw_source["states"][7,37]))
    fig,axes=plt.subplots(3,5,figsize=(8.5,5.5),layout="constrained")
    titles=["Truth","Noisy blur","Terminal 5000","Intermediate 5000","Absolute difference"]
    for row,(name,truth,observed,terminal,middle) in enumerate(examples):
        images=[truth,observed,terminal,middle,np.abs(terminal-middle)]
        for column,(ax,values) in enumerate(zip(axes[row],images)):
            ax.imshow(values.reshape(4,4),cmap="gray" if column<4 else "magma",
                      vmin=0,vmax=1 if column<4 else .2,interpolation="nearest")
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[column],fontsize=9)
            if column==0:ax.set_ylabel(name,fontsize=9)
            if column in [2,3]:
                ax.set_xlabel(f"SE {np.sum((values-truth)**2):.4f}",fontsize=8)
    fig.savefig(args.data/"development_examples.png",dpi=180)
    fig.savefig(args.data/"development_examples.svg")
    plt.close(fig)
    print(json.dumps({"raw_microvalidation_middle_relative_percent":
                       summary["raw_microvalidation_middle_relative_percent"]}))


if __name__=="__main__":
    main()
