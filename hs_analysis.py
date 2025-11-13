#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ======================================================
# User Config
# ======================================================
CACHE_DIR = "cache"

model = "llama3"
size = "8B"
percentage = 1.0         # e.g., 1.0 for "1.0pct"
pooling = "orig"         # last / orig / mean / sum
method = "nmd"           # ttest_abs / ttest_signed / nmd

freq_threshold = 5

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

plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    neuron_indices, layer_positions, c=values,
    cmap='coolwarm', edgecolor='k', s=80,
    vmin=-abs_max, vmax=abs_max
)

plt.title(
    f"Selected Neurons per Layer - {method.upper()} ({model}-{size}, {pooling}-pool)",
    fontsize=14, fontweight='bold', pad=15
)
plt.xlabel("Neuron Index", fontsize=12, fontweight='bold')
plt.ylabel("Layer", fontsize=12, fontweight='bold')
plt.yticks(np.arange(1, num_layers + 1))
plt.grid(True, linestyle='--', alpha=0.5)

cbar = plt.colorbar(scatter)
cbar.set_label("Mask Value", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ======================================================
# 2. Highlight high-frequency neurons
# ======================================================
freq = Counter(neuron_indices)
high_freq = {idx for idx, count in freq.items() if count >= freq_threshold}
mask_high = np.isin(neuron_indices, list(high_freq))

plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    neuron_indices, layer_positions, c=values,
    cmap='coolwarm', edgecolor='k', s=80,
    vmin=-abs_max, vmax=abs_max
)

plt.scatter(
    neuron_indices[mask_high], layer_positions[mask_high],
    facecolors='none', edgecolors='lime', s=120, linewidths=2,
    label=f'Freq ≥ {freq_threshold}'
)

plt.title(
    f"High-Frequency Neurons (Freq ≥ {freq_threshold}) - {model}-{size}",
    fontsize=14, fontweight='bold', pad=15
)
plt.xlabel("Neuron Index", fontsize=12, fontweight='bold')
plt.ylabel("Layer", fontsize=12, fontweight='bold')
plt.yticks(np.arange(1, num_layers + 1))
plt.grid(True, linestyle='--', alpha=0.5)

plt.colorbar(scatter)
plt.legend()

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