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


# -----------------------------------------------------------
# 2. Visualize distribution of all non-zero neurons
# -----------------------------------------------------------
nonzero_indices = np.nonzero(mask)
layer_positions = nonzero_indices[0] + 1
neuron_indices = nonzero_indices[1]
neuron_values = mask[nonzero_indices]
abs_max = max(abs(np.min(neuron_values)), abs(np.max(neuron_values)))

fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(
    neuron_indices, layer_positions, c=neuron_values,
    cmap='pink', edgecolor='k', s=80, vmin=-abs_max, vmax=abs_max
)

plt.title(
    f"Selected Neurons per Layer - {method.upper()} ({model}-{size}, {pooling}-pool)",
    fontsize=14, fontweight='bold', pad=15
)
ax.set_xlabel("Neuron Index", fontsize=12, fontweight='bold')
ax.set_ylabel("Layer", fontsize=12, fontweight='bold')
ax.set_yticks(np.arange(1, num_layers + 1))
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# Colorbar
cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label("Mask Value", fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 3. Highlight high-frequency neurons
# -----------------------------------------------------------

freq = Counter(neuron_indices)
high_freq_neurons = {idx for idx, count in freq.items() if count >= freq_threshold}
high_mask = np.isin(neuron_indices, list(high_freq_neurons))

fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(
    neuron_indices, layer_positions, c=neuron_values,
    cmap='coolwarm', edgecolor='k', s=80, vmin=-abs_max, vmax=abs_max
)
ax.scatter(
    neuron_indices[high_mask], layer_positions[high_mask],
    facecolors='none', edgecolors='lime', s=120, linewidths=2, label=f'Freq ≥ {freq_threshold}'
)

ax.set_title(
    f"High-Frequency Neurons (Freq ≥ {freq_threshold}) in Mask",
    fontsize=14, fontweight='bold', pad=15
)
ax.set_xlabel("Neuron Index", fontsize=12, fontweight='bold')
ax.set_ylabel("Layer", fontsize=12, fontweight='bold')
ax.set_yticks(np.arange(1, num_layers + 1))
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label("Mask Value", fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

ax.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 4. Count frequency of top neurons indices across layers
# -----------------------------------------------------------
nonzero_indices = np.nonzero(mask)
neuron_indices = nonzero_indices[1]

# count frequency of neuron index
freq_counter = Counter(neuron_indices)

# sort
sorted_items = sorted(freq_counter.items(), key=lambda x: -x[1])
sorted_indices = [item[0] for item in sorted_items]
sorted_counts = [item[1] for item in sorted_items]

# only keep high frequent neuron index
freq_threshold = 5
filtered_indices = [idx for idx, cnt in zip(sorted_indices, sorted_counts) if cnt >= freq_threshold]
filtered_counts = [cnt for cnt in sorted_counts if cnt >= freq_threshold]

# Plot
plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(filtered_indices)), filtered_counts,
        color='slateblue', edgecolor='black', linewidth=0.8)
plt.xticks(np.arange(len(filtered_indices)), filtered_indices,
           fontsize=8, rotation=45, ha='right')
plt.xlabel("Neuron Index", fontsize=12, fontweight='bold')
plt.ylabel("Frequency", fontsize=12, fontweight='bold')
plt.title(f"Neurons appearing ≥ {freq_threshold} times ({model})", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()