#!/usr/bin/env python3
"""R1 feasibility study. Original, unnormalised RED-GD; no trained model.

Run: python3 r1_experiment.py --out results --samples 200000
All methods within a comparison share observations, initialisations and K.
The success event is the Euclidean ball of radius 0.2 about the true template.
Squared error is squared Euclidean error, not per-coordinate MSE.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass(frozen=True)
class Config:
    m: float = 1.0
    tau: float = 0.15
    lam: float = 1.0
    a: float = 0.25
    a2: float = 1.0
    nu: float = 0.05
    gamma: float = 0.4
    epsilon: float = 0.35
    sigma_high: float = 2.0
    K: int = 64
    H: int = 12
    R: float = 2.0
    radius: float = 0.2
    pi_plus: float = 0.5


def schedule(c: Config, kind: str, value: float | None = None):
    if kind == "constant":
        return np.full(c.K, c.epsilon)
    if kind == "two_stage":
        h = c.H if value is None else int(value)
        if not 0 <= h < c.K:
            raise ValueError("H must leave at least one terminal-noise call")
        return np.r_[np.full(h, c.sigma_high), np.full(c.K-h, c.epsilon)]
    if kind == "power":
        p = float(value)
        return c.epsilon + (c.sigma_high-c.epsilon)*(1-np.arange(c.K)/(c.K-1))**p
    raise ValueError(kind)


def scalar_run(x0, b, sigmas, c: Config):
    x = np.array(x0, dtype=float, copy=True)
    bias = 0.5*np.log(c.pi_plus/(1-c.pi_plus))
    for sigma in sigmas:
        s2 = c.tau**2 + sigma**2
        q = sigma**2/s2
        x = x - c.gamma*((c.a**2+c.lam*q)*x-b
                         -c.lam*q*c.m*np.tanh(c.m*x/s2+bias))
    return x


def run_2d(x0, y, sigmas, c: Config, trajectory=False):
    x = np.array(x0, dtype=float, copy=True)
    bias = 0.5*np.log(c.pi_plus/(1-c.pi_plus))
    hist = [x.copy()] if trajectory else None
    for sigma in sigmas:
        s2 = c.tau**2 + sigma**2
        q = sigma**2/s2
        x[:, 0] -= c.gamma*((c.a**2+c.lam*q)*x[:, 0]-c.a*y[:, 0]
                              -c.lam*q*c.m*np.tanh(c.m*x[:, 0]/s2+bias))
        x[:, 1] -= c.gamma*((c.a2**2+c.lam*q)*x[:, 1]-c.a2*y[:, 1])
        if trajectory:
            hist.append(x.copy())
    return np.array(hist) if trajectory else x


def uniform_interval_probability(sigmas, c: Config, b, lower, upper):
    """Continuous-uniform initial law; numerical inverse of monotone composition.
    Returns a probability and the preimage interval, not a grid/MC estimate.
    """
    f = lambda z: float(scalar_run(z, b, sigmas, c))
    left, right = f(-c.R), f(c.R)
    if right < lower or left > upper:
        return 0.0, None, None
    lo = -c.R if left >= lower else brentq(lambda z:f(z)-lower, -c.R, c.R, xtol=1e-14)
    hi = c.R if right <= upper else brentq(lambda z:f(z)-upper, -c.R, c.R, xtol=1e-14)
    return max(0.0, hi-lo)/(2*c.R), lo, hi


def stationary_roots(c: Config, sigma, b):
    s2 = c.tau**2+sigma**2
    q = sigma**2/s2
    bias = .5*np.log(c.pi_plus/(1-c.pi_plus))
    f = lambda u:(c.a**2+c.lam*q)*u-b-c.lam*q*c.m*np.tanh(c.m*u/s2+bias)
    grid = np.linspace(-4, 4, 16001)
    val = f(grid)
    indices = np.flatnonzero(val[:-1]*val[1:] < 0)
    roots = [brentq(f,grid[i],grid[i+1]) for i in indices]
    roots.extend(grid[val == 0].tolist())
    return sorted(set(roots))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def formula_checks(c: Config, rng):
    """Independent Bayes mixture denoiser vs tanh formula and potential gradient."""
    worst_d, worst_g = 0.0, 0.0
    for sigma in [.35, .7, 2.0]:
        x = rng.normal(size=(80, 2))
        for pi in [.23, .5, .77]:
            s2 = c.tau**2+sigma**2
            means = np.array([[c.m, 0.], [-c.m, 0.]])
            logits = np.log([pi, 1-pi])[None, :] - np.sum((x[:,None,:]-means[None,:,:])**2,axis=2)/(2*s2)
            resp = np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
            means_cond = c.tau**2/s2*x[:,None,:]+sigma**2/s2*means[None,:,:]
            direct = np.sum(resp[:,:,None]*means_cond, axis=1)
            closed = c.tau**2/s2*x
            closed[:,0] += sigma**2/s2*c.m*np.tanh(c.m*x[:,0]/s2+.5*np.log(pi/(1-pi)))
            worst_d = max(worst_d, float(np.max(abs(direct-closed))))
            def prior_log(z):
                return logsumexp(np.log([pi,1-pi])[None,:]-np.sum((z[:,None,:]-means[None,:,:])**2,axis=2)/(2*s2),axis=1)
            h = 1e-5
            gradient = np.empty_like(x)
            for j in range(2):
                step = np.zeros_like(x); step[:,j]=h
                gradient[:,j] = -sigma**2*(prior_log(x+step)-prior_log(x-step))/(2*h)
            worst_g = max(worst_g, float(np.max(abs(gradient-(x-direct)))))
    assert worst_d < 1e-12 and worst_g < 1e-8
    return {"direct_bayes_denoiser_max_abs_error":worst_d,
            "finite_difference_regularizer_gradient_max_abs_error":worst_g}


def certificate(c: Config):
    s2 = c.tau**2+c.epsilon**2
    q = c.epsilon**2/s2
    aL = c.a**2+q
    beta = 1/s2
    sH2 = c.tau**2+c.sigma_high**2
    qH = c.sigma_high**2/sH2
    muH = c.a**2+qH-qH/sH2
    bmin,bmax=.025,.1
    local_mu = aL-q*beta/np.cosh(beta*.5)**2
    grow = 2*q*np.tanh(beta*.5)-aL
    Hsafe = int(np.ceil(np.log(1+2*.8/bmin)/(-np.log(.68))))
    M = int(np.ceil(np.log(1+.77*.5/bmin)/np.log(1+.4*.77)))
    N = 8
    uerror=.05+1.05*.648**N
    verror=.15/(1+q)+2*.263**(Hsafe+M+N)
    pE=float((2*norm.cdf(3)-1)**2)
    failure_mass=(2-.025)/4
    p_one=2*norm.cdf(3)-1
    moment1=c.nu*np.sqrt(2/np.pi)*(1-(1-np.exp(-4.5))*p_one)
    moment2=c.nu**2*(1-(1-2*(3*norm.pdf(3)+norm.sf(3)))*p_one)
    cu=1-c.gamma*aL; cv=1-c.gamma*(1+q)
    C0=1+c.R*cu**c.K+(c.a**2+1)/aL; C1=c.a/aL
    D0=c.R*cv**c.K; D1=1/(1+q)
    tail_mse=(C0*C0+D0*D0)*(1-pE)+2*(C0*C1+D0*D1)*moment1+(C1*C1+D1*D1)*moment2
    annealed_mse_upper=pE*(uerror*uerror+verror*verror)+tail_mse
    constant_mse_lower=pE*failure_mass*1.025**2
    h = lambda u,b:aL*u-b-q*np.tanh(beta*u)
    values={
        "high_curvature":muH,"high_curvature_lower_bound":.8,
        "terminal_local_curvature":local_mu,"terminal_local_curvature_lower_bound":.88,
        "terminal_growth_coefficient":grow,"terminal_growth_lower_bound":.77,
        "H_safe":Hsafe,"low_growth_steps":M,"low_local_steps":N,
        "min_total_steps":Hsafe+M+N,"low_steps_required":M+N,
        "first_coordinate_error_bound":uerror,"second_coordinate_error_bound":verror,
        "euclidean_error_bound":float(np.hypot(uerror,verror)),
        "H_epsilon_minus_0025_at_bmax":h(-.025,bmax),
        "H_epsilon_095_at_bmin":h(.95,bmin),
        "H_epsilon_105_at_bmax":h(1.05,bmax),
        "good_noise_event_probability":pE,
        "annealed_success_lower_bound":pE,
        "constant_success_upper_bound":1-pE*failure_mass,
        "success_gain_lower_bound":pE*(1+failure_mass)-1,
        "annealed_expected_squared_l2_upper_bound_K64":annealed_mse_upper,
        "constant_expected_squared_l2_lower_bound":constant_mse_lower,
        "expected_squared_l2_improvement_lower_bound_K64":constant_mse_lower-annealed_mse_upper,
        "tail_moment_abs_coordinate":moment1,
        "tail_moment_squared_coordinate":moment2,
        "annealed_tail_squared_error_upper_bound":tail_mse,
    }
    assert muH>.8 and local_mu>.88 and grow>.77
    assert h(-.025,bmax)>0 and h(.95,bmin)<0 and h(1.05,bmax)>0
    assert np.hypot(uerror,verror)<c.radius
    return values


def metrics(x, truth, templates):
    success=np.linalg.norm(x-templates,axis=1)<=.2
    correct=np.sign(x[:,0])==np.sign(templates[:,0])
    truth_near=np.linalg.norm(x-truth,axis=1)<=.2
    err=np.sum((x-truth)**2,axis=1)
    return {"template_success":float(success.mean()),"mode_accuracy":float(correct.mean()),
            "truth_ball_success":float(truth_near.mean()),"mean_squared_l2_error":float(err.mean()),
            "squared_error_standard_error":float(err.std(ddof=1)/np.sqrt(len(err))),
            "template_success_count":int(success.sum()),"sample_count":len(success)}


def initialization_check(c, out):
    rng=np.random.default_rng(4242);n=50000
    z=rng.choice([-1.,1.],n);truth=np.c_[z,np.zeros(n)]
    y=truth*np.array([c.a,c.a2])+c.nu*rng.normal(size=(n,2))
    rows=[]
    starts=[("zero",np.zeros_like(truth)),("backprojection",y*np.array([c.a,c.a2])),("least_squares",y/np.array([c.a,c.a2]))]
    for initial,x0 in starts:
        for method in ["constant","two_stage"]:
            x=run_2d(x0,y,schedule(c,method),c)
            rows.append({"initialization":initial,"method":method,**metrics(x,truth,truth)})
    write_csv(out/"initialization_check.csv",rows)
    return rows


def make_figure(out, c, dwell, powers, priors):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes=plt.subplots(2,2,figsize=(12,8.2),constrained_layout=True)
    blue,orange,gray="#126A97","#CD633C","#697684"
    u=np.linspace(-1.65,1.65,1000); b=c.a**2
    for sigma,color in [(2.,blue),(.8,gray),(.35,orange)]:
        s2=c.tau**2+sigma**2
        lp=logsumexp(np.stack([-(u-1)**2/(2*s2),-(u+1)**2/(2*s2)]),axis=0)-np.log(2)
        F=.5*c.a**2*u**2-b*u-c.lam*sigma**2*lp
        axes[0,0].plot(u,F-F.min(),label=f"sigma = {sigma:g}",color=color,lw=2)
    axes[0,0].set(title="A. Objective changes with denoising level",xlabel="Mode coordinate u",ylabel="F_sigma(u) - min F_sigma")
    axes[0,0].legend(frameon=False)
    for name,ss,color in [("Constant",schedule(c,"constant"),orange),("Annealed: H=12",schedule(c,"two_stage"),blue)]:
        for initial in [-2,-1.2,-.25,.8]:
            x=np.array([[initial,0.]])
            y=np.array([[c.a,0.]])
            history=run_2d(x,y,ss,c,trajectory=True)[:,0,0]
            axes[0,1].plot(np.arange(c.K+1),history,color=color,alpha=.8,lw=1.7,label=name if initial==-2 else None)
    axes[0,1].axhspan(.8,1.2,color=blue,alpha=.08)
    axes[0,1].axvline(c.H,color=gray,ls=":",lw=1)
    axes[0,1].set(title="B. Same starts and budget, different final basin",xlabel="Denoiser calls",ylabel="Mode coordinate u",ylim=(-2.05,1.25))
    axes[0,1].legend(frameon=False,loc="lower right")
    axes[1,0].plot([d["power"] for d in powers],[100*d["template_success"] for d in powers],color=blue,lw=2,label="Within 0.2 of true template")
    axes[1,0].plot([d["power"] for d in powers],[100*d["terminal_basin_probability"] for d in powers],color=gray,lw=1.5,ls="--",label="In correct terminal basin")
    axes[1,0].axhline(100*dwell[0]["template_success"],color=orange,lw=1,ls=":",label="Constant terminal sigma")
    axes[1,0].set(xscale="log",title="C. Finite-budget scheduling window",xlabel="Power p (small: late drop; large: early drop)",ylabel="Probability (%)",ylim=(-3,103))
    axes[1,0].legend(frameon=False,fontsize=8,loc="lower right")
    axes[1,1].plot([d["pi_plus"] for d in priors],[100*d["constant_success"] for d in priors],color=orange,lw=2,label="Constant")
    axes[1,1].plot([d["pi_plus"] for d in priors],[100*d["annealed_success"] for d in priors],color=blue,lw=2,label="Annealed: H=12")
    axes[1,1].axvline(.5,color=gray,ls=":",lw=1)
    axes[1,1].set(title="D. Wrong prior weights reverse the benefit",xlabel="Assumed prior mass of + template (truth is +)",ylabel="Template-neighborhood success (%)",ylim=(-3,103))
    axes[1,1].legend(frameon=False,loc="lower right")
    fig.suptitle("R1 feasibility: exact-MMSE RED-GD on a two-Gaussian prior",fontsize=15,fontweight="bold")
    fig.text(.5,-.015,"Noiseless slice; u0 ~ Uniform[-2,2]; K=64; final sigma=0.35; gamma=0.4. No claim of novelty or learned-model performance.",ha="center",fontsize=9,color=gray)
    fig.savefig(out/"r1_feasibility.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",default=str(Path(__file__).resolve().parents[1]/"results"/"reproduction"));parser.add_argument("--samples",type=int,default=200000)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    c=Config();seed=20261028;rng=np.random.default_rng(seed)
    checks=formula_checks(c,rng);cert=certificate(c)
    b=c.a**2*c.m;roots=stationary_roots(c,c.epsilon,b);saddle=roots[1]
    dwell=[]
    for H in range(c.K):
        ss=schedule(c,"two_stage",H)
        near,lo,hi=uniform_interval_probability(ss,c,b,.8,1.2)
        basin,_,_=uniform_interval_probability(ss,c,b,saddle,100)
        dwell.append({"H":H,"terminal_calls":c.K-H,"template_success":near,"terminal_basin_probability":basin,"success_preimage_lower":lo,"success_preimage_upper":hi})
    powers=[]
    for p in np.unique(np.r_[np.logspace(-3,3,73),.005,.02,.1,.3,1,3,10,100,1000]):
        ss=schedule(c,"power",p)
        near,_,_=uniform_interval_probability(ss,c,b,.8,1.2)
        basin,_,_=uniform_interval_probability(ss,c,b,saddle,100)
        powers.append({"power":float(p),"template_success":near,"terminal_basin_probability":basin})
    priors=[]
    for pi in np.unique(np.r_[np.linspace(.1,.9,81),.45,.47,.48,.49,.5]):
        cc=replace(c,pi_plus=float(pi))
        p0,*_=uniform_interval_probability(schedule(cc,"constant"),cc,b,.8,1.2)
        p1,*_=uniform_interval_probability(schedule(cc,"two_stage"),cc,b,.8,1.2)
        priors.append({"pi_plus":float(pi),"constant_success":p0,"annealed_success":p1})
    write_csv(out/"dwell_sweep.csv",dwell);write_csv(out/"power_sweep.csv",powers);write_csv(out/"prior_weight_sweep.csv",priors)
    n=args.samples
    z=rng.choice([-1.,1.],size=n);templates=np.column_stack((z,np.zeros(n)))
    x0=rng.uniform(-c.R,c.R,size=(n,2));standard_noise=rng.normal(size=(n,2));within=rng.normal(size=(n,2))
    method_specs=[("constant","constant",None),("annealed_H12","two_stage",12),("too_late_H63","two_stage",63),("power_1","power",1.),("power_3","power",3.),("too_early_power_100","power",100.)]
    rows=[]
    for case,nu,use_gmm in [("templates_low_noise",.05,False),("templates_moderate_noise",.2,False),("templates_MAP_scaled_noise",.35,False),("truth_from_same_GMM",.05,True)]:
        truth=templates+c.tau*within if use_gmm else templates.copy()
        y=truth*np.array([c.a,c.a2])+nu*standard_noise
        for name,kind,value in method_specs:
            ss=schedule(c,kind,value)
            x=run_2d(x0,y,ss,c)
            rows.append({"case":case,"noise_std":nu,"method":name,**metrics(x,truth,templates)})
    write_csv(out/"paired_2d_results.csv",rows)
    # Moderate prior imbalance; balanced evaluation population. Fixed paired inputs.
    imbalance=[];y=templates*np.array([c.a,c.a2])+c.nu*standard_noise
    for pi in [.45,.47,.48,.49,.5,.51,.52,.53,.55]:
        cc=replace(c,pi_plus=pi)
        for name,kind in [("constant","constant"),("annealed_H12","two_stage")]:
            x=run_2d(x0,y,schedule(cc,kind),cc)
            for label,mask in [("all",np.ones(n,dtype=bool)),("plus",z>0),("minus",z<0)]:
                imbalance.append({"pi_plus":pi,"method":name,"population":label,**metrics(x[mask],templates[mask],templates[mask])})
    write_csv(out/"paired_prior_imbalance.csv",imbalance)
    init_results=initialization_check(c,out)
    summary={"config":asdict(c),"seed":seed,"samples_per_case":n,"checks":checks,"certificate":cert,
             "noiseless_terminal_stationary_points":roots,
             "noiseless_high_stationary_points":stationary_roots(c,c.sigma_high,b),
             "noiseless_selected_dwell":[dwell[h] for h in [0,1,4,8,11,12,32,45,60,63]],
             "initialization_check":init_results,
             "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
             "notes":["Exact-MMSE, unnormalised RED residual; no sigma^-2 correction.",
                      "Parameters selected through a feasibility search; not preregistered or held-out.",
                      "The theorem concerns fixed templates; the same-GMM truth row is numerical only.",
                      "nu=.35 makes lambda*epsilon^2=nu^2: terminal smoothed-MAP scaling.",
                      "Constant baselines have the same final objective, step size and K; no universal hyperparameter-optimality claim."]}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    make_figure(out,c,dwell,powers,priors)
    print(json.dumps({"certificate":cert,"checks":checks,"selected_noiseless":summary["noiseless_selected_dwell"],"main_results":[r for r in rows if r["method"] in ["constant","annealed_H12","too_late_H63"]]},indent=2))


if __name__=="__main__":
    main()
