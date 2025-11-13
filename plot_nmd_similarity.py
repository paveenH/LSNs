#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pos vs Neg Hidden-State Similarity Analysis
Matching paper-style metrics:
 - Pearson
 - Spearman
 - Cosine
 - Wasserstein (Z-scored)
 - Euclidean (unit-normalized)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# User Config
# ======================================================
CACHE_DIR = "cache"

model = "llama3"
size = "8B"
percentage = 0.5        # e.g., 1.0 => 1.0pct
pooling = "last"        # last / mean / orig / sum
method = "nmd"          # nmd / ttest_abs / ttest_signed

LAYERS_START_AT_ONE = True

# ======================================================
# Build filenames
# ======================================================
base_prefix = f"{model}_{size}_{percentage}pct_{pooling}_{method}"

pos_path = os.path.join(CACHE_DIR, f"{base_prefix}_pos_mean.npy")
neg_path = os.path.join(CACHE_DIR, f"{base_prefix}_neg_mean.npy")

if not os.path.exists(pos_path) or not os.path.exists(neg_path):
    raise FileNotFoundError(
        f"Missing pos/neg mean files:\n{pos_path}\n{neg_path}\n"
        f"Available: {os.listdir(CACHE_DIR)}"
    )

print(f"[Load] {pos_path}")
print(f"[Load] {neg_path}")

# ======================================================
# Load data
# ======================================================
pos_mean = np.load(pos_path)     # (layers, hidden)
neg_mean = np.load(neg_path)     # (layers, hidden)

pos_hs = np.squeeze(pos_mean)
neg_hs = np.squeeze(neg_mean)

num_layers, hidden_dim = pos_hs.shape

# ======================================================
# Compute similarity metrics layer-wise
# ======================================================
pearson_list, spearman_list, cosine_list = [], [], []
wasserstein_list, euclidean_list = [], []

for layer in range(num_layers):

    pos_layer  = pos_hs[layer]
    neg_layer  = neg_hs[layer]

    # Pearson & Spearman
    if np.std(pos_layer) < 1e-8 or np.std(neg_layer) < 1e-8:
        pearson  = np.nan
        spearman = np.nan
    else:
        pearson  = np.corrcoef(pos_layer, neg_layer)[0, 1]
        spearman = stats.spearmanr(pos_layer, neg_layer)[0]

    pearson_list.append(pearson)
    spearman_list.append(spearman)

    # Cosine similarity
    cos_sim = cosine_similarity(
        pos_layer.reshape(1, -1),
        neg_layer.reshape(1, -1)
    )[0, 0]
    cosine_list.append(cos_sim)

    # Wasserstein distance (Z-scored)
    pos_norm = (pos_layer - pos_layer.mean()) / (pos_layer.std() + 1e-8)
    neg_norm = (neg_layer - neg_layer.mean()) / (neg_layer.std() + 1e-8)
    w_dist = stats.wasserstein_distance(pos_norm, neg_norm)
    wasserstein_list.append(w_dist)

    # Euclidean distance (unit normalized)
    pos_unit = pos_layer / (np.linalg.norm(pos_layer) + 1e-8)
    neg_unit = neg_layer / (np.linalg.norm(neg_layer) + 1e-8)
    euc_dist = np.linalg.norm(pos_unit - neg_unit)
    euclidean_list.append(euc_dist)

# ======================================================
# Print summary
# ======================================================
print("\n=== Similarity Metrics (Pos ↔ Neg) ===")
print("Average Pearson Correlation:        ", np.nanmean(pearson_list))
print("Average Spearman Correlation:       ", np.nanmean(spearman_list))
print("Average Cosine Similarity:          ", np.mean(cosine_list))
print("Average Wasserstein Distance (Z):   ", np.mean(wasserstein_list))
print("Average Euclidean Distance (unit):  ", np.mean(euclidean_list))

# ======================================================
# Plot (paper-style)
# ======================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,
    "figure.figsize": (6.9, 4.5),
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

# Layer index convention
LAYERS_START_AT_ONE = True

if LAYERS_START_AT_ONE:
    layers = np.arange(1, num_layers + 1)   # 1..33
else:
    layers = np.arange(num_layers)          # 0..32

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

fig.suptitle(
    f"Pos vs Neg Hidden-State Similarity ({model}-{size}, {pooling})",
    fontsize=14,
    fontweight="bold",
    y=0.97
)

# -------------- Panels ----------------
axes[0].plot(layers, pearson_list, marker='o', color='tab:blue')
axes[0].set_title("Pearson Correlation", fontweight="bold")
axes[0].set_ylabel("Correlation")
axes[0].set_xlabel("Layer Index")

axes[1].plot(layers, spearman_list, marker='s', color='tab:green')
axes[1].set_title("Spearman Correlation", fontweight="bold")
axes[1].set_xlabel("Layer Index")

axes[2].plot(layers, cosine_list, marker='^', color='tab:red')
axes[2].set_title("Cosine Similarity", fontweight="bold")
axes[2].set_ylabel("Similarity")
axes[2].set_xlabel("Layer Index")

axes[3].plot(layers, euclidean_list, marker='d', color='tab:purple')
axes[3].set_title("Euclidean Distance", fontweight="bold")
axes[3].set_ylabel("Distance")
axes[3].set_xlabel("Layer Index")

# -------------- Axis style --------------
for ax in axes:
    ax.set_xticks(layers)       # ensure ticks start at correct index
    ax.set_xlim(layers[0], layers[-1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', pad=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_dir = f"{model}/plots"
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, "pos_neg_similarity_metrics.pdf")
png_path = pdf_path.replace(".pdf", ".png")
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
fig.savefig(png_path, dpi=400, bbox_inches="tight")

plt.show()
plt.close()