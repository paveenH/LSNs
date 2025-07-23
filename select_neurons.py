#!/usr/bin/env python3
import os
import argparse
import numpy as np
from scipy.stats import ttest_ind, false_discovery_control


# def is_topk(t_values: np.ndarray, k: int):
#     """Return a 0/1 mask with the same shape as t_values, marking the top k units by absolute value (flattened)."""
#     flat = np.abs(t_values).flatten()
#     thresh = np.partition(flat, -k)[-k]
#     return (np.abs(t_values) >= thresh).astype(int)

def is_topk(a: np.ndarray, k: int):
    _, rix = np.unique(-a, return_inverse=True)
    return (rix < k).astype(int).reshape(a.shape)


def is_bottomk(t_values: np.ndarray, k: int):
    """Return a mask for the units with the smallest absolute k values."""
    flat = np.abs(t_values).flatten()
    thresh = np.partition(flat, k - 1)[k - 1]
    return (np.abs(t_values) <= thresh).astype(int)


def select_neurons(positive: np.ndarray, negative: np.ndarray, percentage: float, localize_range: str, seed: int):
    """
    positive, negative: shape (num_samples, num_layers, hidden_dim)
    percentage: e.g. 1.0 means top 1%
    localize_range: e.g. "100-100", "0-0", or "80-90"
    """
    num_samples, num_layers, hidden_dim = positive.shape
    total_units = num_layers * hidden_dim

    # 1) Compute t and p values
    t_vals = np.zeros((num_layers, hidden_dim), dtype=float)
    p_vals = np.zeros((num_layers, hidden_dim), dtype=float)
    for i in range(num_layers):
        pos = np.abs(positive[:, i, :])
        neg = np.abs(negative[:, i, :])
        t_vals[i], p_vals[i] = ttest_ind(pos, neg, axis=0, equal_var=False)

    # 2) FDR correction
    flat_p = p_vals.flatten()
    adj_p = false_discovery_control(flat_p)
    adj_p = adj_p.reshape(p_vals.shape)

    # 3) Select mask
    np.random.seed(seed)
    start, end = map(int, localize_range.split("-"))
    # Calculate number of units to select
    k = int((percentage / 100) * total_units)

    if start < end:
        # Randomly select k units whose t-values are between start and end percentiles
        low, high = np.percentile(t_vals, start), np.percentile(t_vals, end)
        rng_mask = (t_vals >= low) & (t_vals <= high)
        rng_indices = np.nonzero(rng_mask.flatten())[0]
        chosen = np.random.choice(rng_indices, size=k, replace=False)
        mask = np.zeros(total_units, dtype=int)
        mask[chosen] = 1
        mask = mask.reshape((num_layers, hidden_dim))
    elif start == end == 100:
        # top k units
        mask = is_topk(t_vals, k)
    elif start == end == 0:
        # bottom k units
        mask = is_bottomk(t_vals, k)
    else:
        raise ValueError(f"unsupported localize_range {localize_range}")

    return mask, adj_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="select_neurons.py", description="Compute neuron mask and FDR-corrected p-values from positive/negative hidden states")
    parser.add_argument("--input_dir", required=True, help="Directory containing positive_<size>_<pooling>.npy and negative_<size>_<pooling>.npy")
    parser.add_argument("--output_dir", required=True, help="Directory to save mask and p-values")
    parser.add_argument("--size", required=True, help="Model identifier, e.g., '8B'")
    parser.add_argument("--pooling", required=True, help="Pooling strategy used during feature extraction (mean, last, sum, orig)")
    parser.add_argument("--percentage", type=float, required=True, help="Percentage of units to select as top neurons (e.g., 1.0 for top 1%)")
    parser.add_argument("--localize_range", default="100-100", help="Percentile range for unit selection: '100-100' means top units, '0-0' bottom units, '80-90' randomly in 80~90 percentile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling units in percentile range")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Load data
    pos_path = os.path.join(args.input_dir, f"positive_{args.size}_{args.pooling}.npy")
    neg_path = os.path.join(args.input_dir, f"negative_{args.size}_{args.pooling}.npy")
    positive = np.load(pos_path)  # e.g. (240, 32, 4096)
    negative = np.load(neg_path)
    
    print("positive.shape:", positive.shape)
    print("negative.shape:", negative.shape)

    # Compute mask and adjusted p-values
    mask, adj_p = select_neurons(positive, negative, percentage=args.percentage, localize_range=args.localize_range, seed=args.seed)

    # Save results
    mask_path = os.path.join(args.output_dir, f"mask_{args.size}_{args.pooling}.npy")
    pval_path = os.path.join(args.output_dir, f"pvalues_{args.size}_{args.pooling}.npy")

    np.save(mask_path, mask)
    np.save(pval_path, adj_p)
    print(f"> Mask saved to   {mask_path}  shape={mask.shape}")
    print(f"> P-values saved to {pval_path} shape={adj_p.shape}")
