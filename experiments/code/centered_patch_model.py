"""Fit small structure-only GMMs on a spatial training split."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_images


def structure_basis():
    h = np.array([[1,1,1,1],[1,-1,1,-1],
                  [1,1,-1,-1],[1,-1,-1,1]], dtype=float)/2
    orthogonal = np.kron(h,h)
    assert np.array_equal(orthogonal.T@orthogonal,np.eye(16))
    assert np.array_equal(orthogonal[:,0],np.ones(16)/4)
    return orthogonal[:,0],orthogonal[:,1:]


def blur_operator():
    basis = np.eye(16).reshape(16,4,4)
    blurred = .5*basis+.25*np.roll(basis,1,1)+.25*np.roll(basis,-1,1)
    blurred = .5*blurred+.25*np.roll(blurred,1,2)+.25*np.roll(blurred,-1,2)
    return blurred.reshape(16,16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    plan = {
        "stage": "Development only, no prior test-photo evaluation",
        "photo": "sklearn china.jpg, spatial upper/lower train/development split",
        "split": "Upper half minus 4 patch rows trains; lower half plus 4 rows develops",
        "train_patches": 2048,"development_patches": 256,
        "patch_size": [4,4],"components": [8,16],
        "training": "KMeans n_init=5, common isotropic variance in 15D structure coordinates",
        "seed_train": 20261120,"seed_development": 20261121,
        "seed_noise": 20261122,"noise_std": .01,
        "lambda": .1,"gamma": .25,"terminal_rules": ["absolute_.02","train_tau"],
        "intermediate_sigma": .1,"high_sigma": .3,"intermediate_calls": 25,
        "budgets": [25,154,1000,5000],
        "policies": ["terminal","intermediate","low_high"],
        "full_reconstruction": "Structure RED plus observed DC coefficient, no truth brightness input",
        "change": "New 15D centered prior and unregularized observed brightness; "
                  "not the original full-16D isotropic prior",
        "selection": "Two fixed component counts and two terminal rules, declared before fitting. "
                     "All development settings reported; no population performance claim",
        "gates": "Physical-domain curvature, reachable distinct terminal fixed points, "
                 "then barrier/channel certification if candidates exist"}
    (args.out/"plan.json").write_text(json.dumps(plan,indent=2))
    photos = load_sample_images()
    gray = np.asarray(Image.fromarray(photos.images[0]).convert("L"),dtype=float)/255
    height,width = [n//4*4 for n in gray.shape]
    patches = gray[:height,:width].reshape(height//4,4,width//4,4).transpose(0,2,1,3)
    split = (height//4)//2
    training = patches[:split-4].reshape(-1,16)
    development = patches[split+4:].reshape(-1,16)
    train_rng = np.random.default_rng(plan["seed_train"])
    dev_rng = np.random.default_rng(plan["seed_development"])
    train = training[train_rng.choice(len(training),2048,replace=False)]
    truth = development[dev_rng.choice(len(development),256,replace=False)]
    dc,basis = structure_basis()
    operator = blur_operator()
    shape_operator = basis.T@operator@basis
    assert np.array_equal(shape_operator,np.diag(np.diag(shape_operator)))
    assert np.array_equal(operator@dc,dc)
    assert np.array_equal(operator@basis,basis@shape_operator)
    noise = np.random.default_rng(plan["seed_noise"]).standard_normal(truth.shape)
    observation = truth@operator+.01*noise
    shape_y = observation@basis
    dc_y = observation@dc
    np.savez_compressed(args.out/"data.npz",train=train,truth=truth,
                        observation=observation,shape_y=shape_y,dc_y=dc_y,
                        basis=basis,dc=dc,operator=operator,shape_operator=shape_operator)
    z = train@basis
    models = []
    for components in plan["components"]:
        fit = KMeans(n_clusters=components,random_state=plan["seed_train"],n_init=5).fit(z)
        means = fit.cluster_centers_
        pi = np.bincount(fit.labels_,minlength=components)/len(z)
        tau2 = float(np.mean((z-means[fit.labels_])**2))
        assert np.isfinite(means).all() and (pi>0).all() and tau2>0
        cov = ((means-pi@means).T*pi)@(means-pi@means)
        values = np.linalg.eigvalsh(cov)
        np.savez_compressed(args.out/f"model_{components}.npz",mu=means,pi=pi,tau2=tau2)
        models.append({"components":components,"tau2":tau2,"tau":float(np.sqrt(tau2)),
                       "top_mean_covariance_fraction":float(values[-1]/values.sum()),
                       "model_path":f"model_{components}.npz"})
    (args.out/"models.json").write_text(json.dumps({
        "source":photos.filenames[0],"training_patch_rows":[0,split-4],
        "development_patch_rows":[split+4,height//4],
        "shape_operator_spectrum":np.diag(shape_operator).tolist(),"models":models},indent=2))
    print(json.dumps(models))


if __name__ == "__main__":
    main()
