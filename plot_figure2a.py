#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:07:37 2025

@author: paveenhuang
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model_utils import model_name_map, get_num_blocks, get_hidden_dim

percentage = 1.0
network = "language"
model_name = "Llama-3.1-8B-Instruct"

plot_data = {"selectivity": [], "layer_num": [], "model_name": []}

model_name = os.path.basename(model_name)
num_layers = get_num_blocks(model_name)  # 32
hidden_dim = get_hidden_dim(model_name)  # 4096

pooling = "last-token"  # for language, always last-token

# ## Path 1 ##
# model_loc_path = (
#     f"{model_name}_network={network}_pooling={pooling}_range=100-100_perc={percentage}_nunits=None_pretrained=True.npy"
# )
# lang_mask_path = os.path.join("cache", model_loc_path)


# Path 2 Plot From saved HS##
model_name = "llama3"
size = "8B"
pooling = "fix12"
model_loc_path = f"mask_{size}_{pooling}.npy"
lang_mask_path = os.path.join("representation_ttest", model_name, model_loc_path)


if not os.path.exists(lang_mask_path):
    raise ValueError(f"Path does not exist: {lang_mask_path}")
# Read language_mask
language_mask = np.load(lang_mask_path)  # shape: (32, 4096)


for i in range(num_layers):
    num_selected = language_mask[i].sum()
    percentage_selected = (num_selected / hidden_dim) * 100

    plot_data["selectivity"].append(percentage_selected)
    plot_data["layer_num"].append(i + 1)
    plot_data["model_name"].append(model_name_map.get(model_name, model_name))

# Plot heatmap
df = pd.DataFrame(plot_data)
pivot_df = df.pivot_table(index="layer_num", columns="model_name", values="selectivity", aggfunc="mean")

plt.figure(figsize=(4, 6))
sns.set_theme(context="paper", font_scale=2, style="white")

ax = sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".1f", cbar=False)
for t in ax.texts:
    t.set_fontsize(10)
    t.set_text(t.get_text() + " %")

sns.despine()
plt.xlabel("")
plt.ylabel("")
plt.title("")
plt.tight_layout()
plt.savefig(f"cache/heatmap_model={model_name}_network={network}.png")
