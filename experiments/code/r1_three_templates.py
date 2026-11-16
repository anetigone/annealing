#!/usr/bin/env python3
"""Exploratory extension: three templates and measurement initialisation.
Synthetic settings are selected by a noiseless feasibility search, not a benchmark.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax


def restore(x0,y,means,A,sigmas,tau=.15,gamma=.35):
    x=x0.copy()
    for sigma in sigmas:
        v=tau*tau+sigma*sigma;q=sigma*sigma/v
        logits=-np.sum((x[:,None,:]-means[None,:,:])**2,axis=2)/(2*v)
        mean=softmax(logits,axis=1)@means
        x-=gamma*((x@A.T-y)@A+q*(x-mean))
    return x


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results"/"reproduction")
    out=parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(3132026)
    K=64;eps=.35;high=2.
    schedules={'constant':np.full(K,eps),'two_stage_H12':np.r_[np.full(12,high),np.full(52,eps)],'linear_anneal':np.linspace(high,eps,K)}
    candidates=[]
    for i in range(300):
        means=rng.normal(size=(3,2));means-=means.mean(axis=0)
        means*=1.4/np.sqrt(np.mean(np.sum(means*means,axis=1)))
        distances=np.linalg.norm(means[:,None,:]-means[None,:,:],axis=2)+np.eye(3)*100
        if distances.min()<.9:
            continue
        theta=rng.uniform(0,np.pi);U=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        A=np.diag([.25,1.])@U.T
        y=means@A.T;x0=y@A
        scores={};errors={}
        for name,ss in schedules.items():
            est=restore(x0,y,means,A,ss)
            errors[name]=np.linalg.norm(est-means,axis=1)
            scores[name]=int(np.sum(errors[name]<.2))
        gain=max(scores['two_stage_H12'],scores['linear_anneal'])-scores['constant']
        if gain>0:
            candidates.append({'index':i,'means':means.tolist(),'A':A.tolist(),'scores':scores,'errors':{k:v.tolist() for k,v in errors.items()},'gain':gain})
    candidates.sort(key=lambda c:(-c['gain'],max(c['errors']['linear_anneal'])))
    (out/'three_template_search.json').write_text(json.dumps({'seed':3132026,'draws':300,'selection':'exploratory: largest noiseless gain, then smallest worst-template linear-anneal error; not preregistered','positive_cases':len(candidates),'candidates':candidates},indent=2))
    if not candidates:
        print('No positive case found');return
    chosen=candidates[0];means=np.array(chosen['means']);A=np.array(chosen['A'])
    n=100000;test_rng=np.random.default_rng(3132027);labels=test_rng.integers(0,3,size=n);truth=means[labels];noise=test_rng.normal(size=(n,2))
    rows=[]
    for nu in [.0,.05,.1]:
        y=truth@A.T+nu*noise
        for initial,x0 in [('backprojection',y@A),('zero',np.zeros_like(truth))]:
            for name,ss in schedules.items():
                est=restore(x0,y,means,A,ss)
                error=np.sum((est-truth)**2,axis=1)
                predicted=np.argmin(np.sum((est[:,None,:]-means[None,:,:])**2,axis=2),axis=1)
                row={'noise_std':nu,'initialization':initial,'method':name,'samples':n,'template_success':float(np.mean(error<=.2**2)),'mode_accuracy':float(np.mean(predicted==labels)),'mean_squared_l2_error':float(error.mean())}
                rows.append(row)
    with (out/'three_template_validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'three_template_selected.json').write_text(json.dumps(chosen,indent=2))
    print(json.dumps({'positive_cases':len(candidates),'selected':chosen,'validation':rows},indent=2))


if __name__=='__main__':
    main()
