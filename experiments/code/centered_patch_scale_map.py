"""Explore scale/holding-time responses only for the two discovered dev examples."""
import argparse
import json
from pathlib import Path
import numpy as np
from centered_patch_development import weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    args = parser.parse_args()
    folder = args.data/"scale_map"
    folder.mkdir(exist_ok=False)
    source = json.loads((args.data/"k8_train_tau/summary.json").read_text())
    selected = sorted({r["sample"] for r in source["distinct_endpoint_float_certificates"]})
    assert selected
    data = np.load(args.data/"data.npz")
    model = np.load(args.data/"model_8.npz")
    means,pi,tau2 = [model[k] for k in ["mu","pi","tau2"]]
    terminal = float(np.sqrt(tau2))
    sigmas = [.005,.01,.02,.035,terminal,.075,.1,.15,.3,.6,1.,np.inf]
    holds = [5,25,50]
    gram = data["shape_operator"]@data["shape_operator"].T
    b = data["shape_y"]@data["shape_operator"].T
    saved = np.load(args.data/"k8_train_tau/outputs.npz")
    rows = []
    for hold in holds:
        inputs = np.repeat(b[selected],len(sigmas),axis=0)
        x = inputs.copy()
        scales = np.tile(sigmas,len(selected))
        for k in range(5000):
            variance = tau2+(scales**2 if k<hold else terminal**2)
            variance = np.broadcast_to(variance,(len(x),))
            log = np.log(pi)-np.sum((x[:,None]-means)**2,axis=2)/(2*variance[:,None])
            w = np.exp(log-log.max(axis=1,keepdims=True))
            w /= w.sum(axis=1,keepdims=True)
            c = .1*(1-tau2/variance)
            x -= .25*(x@gram-inputs+c[:,None]*(x-w@means))
        for position,sample in enumerate(selected):
            for index,sigma in enumerate(sigmas):
                z = x[position*len(sigmas)+index]
                baseline = saved["states"][3,sample]
                mid = saved["states"][7,sample]
                error = np.sum((z-data["truth"][sample]@data["basis"])**2)
                rows.append({"sample":sample,"hold":hold,
                             "sigma":"infinity_limit" if np.isinf(sigma) else float(sigma),
                             "admissible_sigma_ge_terminal":bool(sigma>=terminal),
                             "distance_to_terminal":float(np.linalg.norm(z-baseline)),
                             "distance_to_intermediate":float(np.linalg.norm(z-mid)),
                             "full_squared_error":float(error+(
                                 data["dc_y"][sample]-data["truth"][sample]@data["dc"])**2)})
    (folder/"scale_map.json").write_text(json.dumps({
        "selection":"Post-hoc development examples selected by endpoint separation, not error",
        "samples":selected,"holds":holds,"rows":rows,
        "scope":"Small discrete exploratory response map, no continuous scale certificate"},indent=2))
    print(json.dumps([r for r in rows if r["hold"]==25 and (
        r["sigma"]=="infinity_limit" or r["sigma"]==.1)]))


if __name__=="__main__":
    main()
