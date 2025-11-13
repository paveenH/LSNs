#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

sns.set_theme(context="paper", style="white", font_scale=1.3)

# ======================================================
# User Config
# ======================================================
CACHE_DIR = "cache"

model = "llama3"
size = "8B"
percentage = 1.0         # e.g., 1.0 for "1.0pct"
pooling = "orig"         # last / orig / mean / sum
method = "nmd"           # ttest_abs / ttest_signed / nmd

freq_threshold = 15

# ======================================================
# Build filename (new naming convention)
# ======================================================
mask_filename = f"{model}_{size}_{percentage}pct_{pooling}_{method}_mask.npy"
mask_path = os.path.join(CACHE_DIR, mask_filename)

if not os.path.exists(mask_path):
    raise FileNotFoundError(
        f"Mask not found:\n{mask_path}\n"
        f"Available files: {os.listdir(CACHE_DIR)}"
    )

print(f"Loading: {mask_path}")
mask = np.load(mask_path)

num_layers, hidden_dim = mask.shape
print(f"mask shape = {mask.shape}")

# ======================================================
# 1. Scatter plot of selected neurons
# ======================================================
nonzero = np.nonzero(mask)
layer_positions = nonzero[0] + 1       # 1-based layer indexing
neuron_indices = nonzero[1]
values = mask[nonzero]

abs_max = max(abs(values.min()), abs(values.max()))

plt.figure(figsize=(11, 6))

scatter = plt.scatter(
    neuron_indices,
    layer_positions,
    c=values,
    cmap="crest",         
    s=55,
    alpha=0.85,
    linewidth=0,           
)

plt.xlabel("Neuron Index", fontsize=12)
plt.ylabel("Layer", fontsize=12)
plt.title(f"{method.upper()} Selective Neurons ({model}-{size})", fontsize=14, pad=10)

# Colorbar
cbar = plt.colorbar(scatter)
cbar.ax.tick_params(labelsize=10)

sns.despine()            
plt.tight_layout()
plt.show()

# ======================================================
# 2. Highlight high-frequency neurons
# ======================================================
# neuron_indices, layer_positions, values, num_layers, freq_threshold already defined
# Count frequency per neuron index
freq = Counter(neuron_indices)
high_freq = {idx for idx, cnt in freq.items() if cnt >= freq_threshold}
mask_high = np.isin(neuron_indices, list(high_freq))

# ---- Paper-style visualization ----
sns.set_theme(context="paper", style="white", font_scale=1.3)

plt.figure(figsize=(11, 6))

# Main scatter plot (soft color map, no edge lines)
scatter = plt.scatter(
    neuron_indices,
    layer_positions,
    c=values,
    cmap="crest",          # Soft paper-style colormap
    s=55,
    alpha=0.80,
    linewidth=0
)

# Highlight high-frequency neurons
plt.scatter(
    neuron_indices[mask_high],
    layer_positions[mask_high],
    facecolors='none',
    edgecolors='orange',
    linewidths=1.8,
    s=70,
    label=f"Freq ≥ {freq_threshold}"
)

# Labels & formatting
plt.title(f"High-Frequency Neurons (Freq ≥ {freq_threshold}) - {model}-{size}",
          fontsize=14, pad=12)
plt.xlabel("Neuron Index", fontsize=12)
plt.ylabel("Layer", fontsize=12)
plt.yticks(np.arange(1, num_layers + 1))

# Colorbar (paper-friendly)
cbar = plt.colorbar(scatter)
cbar.ax.tick_params(labelsize=9)

plt.legend(loc='upper right', frameon=False, fontsize=10)
sns.despine()
plt.tight_layout()
plt.show()

# ======================================================
# 3. Frequency bar plot
# ======================================================
freq_counter = Counter(neuron_indices)
sorted_items = sorted(freq_counter.items(), key=lambda x: -x[1])

filtered = [(idx, cnt) for idx, cnt in sorted_items if cnt >= freq_threshold]
filtered_indices = [x[0] for x in filtered]
filtered_counts = [x[1] for x in filtered]

plt.figure(figsize=(10, 5))
plt.bar(
    np.arange(len(filtered_indices)), filtered_counts,
    color='slateblue', edgecolor='black', linewidth=0.8
)

plt.xticks(
    np.arange(len(filtered_indices)),
    filtered_indices, fontsize=8,
    rotation=45, ha='right'
)

plt.xlabel("Neuron Index", fontsize=12, fontweight='bold')
plt.ylabel("Frequency", fontsize=12, fontweight='bold')
plt.title(
    f"Neurons appearing ≥ {freq_threshold} times ({model}-{size})",
    fontsize=14, fontweight='bold'
)

plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()