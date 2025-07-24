#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 10:33:30 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
import os
import argparse
import numpy as np
from scipy.stats import ttest_ind, false_discovery_control

def is_topk(a: np.ndarray, k: int):
    _, rix = np.unique(-a, return_inverse=True)
    return (rix < k).astype(int).reshape(a.shape)

def is_bottomk(a: np.ndarray, k: int):
    _, rix = np.unique(a, return_inverse=True)
    return (rix < k).astype(int).reshape(a.shape)

def select_neurons(args, pos, neg):
    ns, nl, hd = pos.shape
    total = nl * hd
    t_vals = np.zeros((nl, hd)); p_vals = np.zeros((nl, hd))
    for i in range(nl):
        t_vals[i], p_vals[i] = ttest_ind(
            np.abs(pos[:,i,:]), np.abs(neg[:,i,:]),
            axis=0, equal_var=False
        )
    # FDR
    adj_p = false_discovery_control(p_vals.flatten()).reshape(p_vals.shape)
    # how many to pick
    k = int((args.percentage/100) * total)
    start, end = map(int, args.localize_range.split('-'))
    np.random.seed(args.seed)
    if start == end == 100:
        mask = is_topk(t_vals, k)
    elif start == end == 0:
        mask = is_bottomk(t_vals, k)
    else:
        low, high = np.percentile(t_vals, start), np.percentile(t_vals, end)
        rng = ((t_vals>=low)&(t_vals<=high)).flatten()
        idx = np.where(rng)[0]
        choose = np.random.choice(idx, size=k, replace=False)
        mask = np.zeros(nl*hd, int); mask[choose]=1
        mask = mask.reshape((nl,hd))
    # save
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f"mask_{args.size}_{args.pooling}.npy"), mask)
    np.save(os.path.join(args.output_dir, f"pvalues_{args.size}_{args.pooling}.npy"), adj_p)
    print("select_neurons done:", mask.shape)

def compute_mean(args, pos, neg):
    pos_mean = pos.mean(axis=0)  # (layers, hidden)
    neg_mean = neg.mean(axis=0)
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f"positive_{args.size}_{args.pooling}.npy"), pos_mean)
    np.save(os.path.join(args.output_dir, f"negative_{args.size}_{args.pooling}.npy"), neg_mean)
    print("compute_mean done:", pos_mean.shape)
    
def nmd(args, pos, neg):
    pos = pos.mean(axis=0)  # (layers, hidden)
    neg = neg.mean(axis=0)
    diff = pos - neg
    nl, hd = diff.shape
    topk = hd // 200
    # args.layer_range is 1-based, convert to 0-based indices
    start, end = map(int, args.layer_range.split('-'))
    start_idx = max(0, start - 1)
    end_idx = min(nl, end - 1)
    mask = np.zeros((nl, hd), dtype=int)
    for i in range(start_idx, end_idx):
        idxs = np.argsort(np.abs(diff[i]))[-topk:]
        mask[i, idxs] = 1
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f"mask_{args.size}_{args.pooling}_nmd.npy"), mask)
    print("nmd done:", mask.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection utilities")
    sub = parser.add_subparsers(dest="mode", required=True)
    # select
    sel = sub.add_parser("select", help="Select neurons via t-test")
    sel.add_argument("--input_dir", required=True)
    sel.add_argument("--output_dir", required=True)
    sel.add_argument("--size", required=True)
    sel.add_argument("--pooling", required=True)
    sel.add_argument("--percentage", type=float, required=True)
    sel.add_argument("--localize_range", default="100-100")
    sel.add_argument("--seed", type=int, default=42)
    sel.set_defaults(func=select_neurons)
    # mean
    mu = sub.add_parser("mean", help="Average hidden states")
    mu.add_argument("--input_dir", required=True)
    mu.add_argument("--output_dir", required=True)
    mu.add_argument("--size", required=True)
    mu.add_argument("--pooling", required=True)
    mu.set_defaults(func=compute_mean)
    # nmd
    nn = sub.add_parser("nmd", help="Neuron-wise mean difference (NMD) mask generation")
    nn.add_argument("--input_dir", required=True)
    nn.add_argument("--output_dir", required=True)
    nn.add_argument("--size", required=True)
    nn.add_argument("--pooling", required=True)
    nn.add_argument("--layer_range", default="0-32", help="Layer range [start-end)")
    nn.set_defaults(func=nmd)

    args = parser.parse_args()
    # load pos/neg or average for select/mean/nmd
    pos = np.load(os.path.join(args.input_dir, f"positive_{args.size}_{args.pooling}.npy"))
    neg = np.load(os.path.join(args.input_dir, f"negative_{args.size}_{args.pooling}.npy"))
    
    args.func(args, pos, neg)
