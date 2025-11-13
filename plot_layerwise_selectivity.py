#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot top-1% selective neuron distribution per layer (Figure 2a style).

Input:
    cache/<model>_<mask_type>_mask.npy
Output:
    cache/heatmap_<model>_<mask_type>.png
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model_utils import model_name_map

CACHE_DIR = "cache"

def load_mask(model_prefix, mask_type):
    """
    Load mask file with explicit mask type:
        <model_prefix>_<mask_type>_mask.npy
    
    mask_type ∈ {"ttest_abs", "ttest_signed", "nmd"}
    """
    target_name = f"{model_prefix}_{mask_type}_mask.npy"
    path = os.path.join(CACHE_DIR, target_name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mask file not found:\n  {path}\n"
            f"Available files: {os.listdir(CACHE_DIR)}"
        )
    print(f"> Loading mask: {target_name}")
    return np.load(path)


def plot_selectivity_heatmap(model_name, mask, mask_type):
    """
    Plot per-layer % of selective units (mask shape = num_layers × hidden_dim)
    """
    num_layers, hidden_dim = mask.shape
    print(f"> Mask shape = {num_layers} × {hidden_dim}")

    per_layer_percent = (mask.sum(axis=1) / hidden_dim) * 100

    data = per_layer_percent.reshape(-1, 1)

    plt.figure(figsize=(2.4, 8))
    sns.set_theme(context="paper", font_scale=1.5, style="white")

    ax = sns.heatmap(
        data,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar=False,
        yticklabels=[str(i+1) for i in range(num_layers)],
        xticklabels=[model_name_map.get(model_name, model_name)]
    )

    for t in ax.texts:
        t.set_text(t.get_text() + "%")

    plt.ylabel("Layer")
    plt.xlabel("")
    plt.title(f"{model_name} – {mask_type}")

    plt.tight_layout()

    out_path = f"{CACHE_DIR}/heatmap_{model_name}_{mask_type}.png"
    plt.savefig(out_path, dpi=300)
    print(f"> Saved: {out_path}")


def main():
    # Models & mask types you want to draw
    # models = ["llama3_1B", "llama3_3B", "gpt2_BASE"]
    models = ["llama3_8B"]
    mask_types = ["ttest_abs", "ttest_signed", "nmd"]   # customize here

    for model_prefix in models:
        for mask_type in mask_types:
            print(f"\n=== Processing {model_prefix} [{mask_type}] ===")

            try:
                mask = load_mask(model_prefix, mask_type)

            except FileNotFoundError as e:
                print(f"  [!] Skip: {e}")
                continue

            plot_selectivity_heatmap(model_prefix, mask, mask_type)


if __name__ == "__main__":
    main()