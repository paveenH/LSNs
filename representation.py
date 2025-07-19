#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:41:25 2025

@author: paveenhuang
"""
#!/usr/bin/env python3
# This script uses LSNsModel to extract hidden state representations and saves them

import os
import numpy as np
from llms import LSNsModel, get_layer_names

model_path = 'mistralai/Mistral-7B-v0.3'
batch_size = 8
max_length = 12
pooling = "last-token" # choices=["last-token", "mean", "sum"]
output_dir = os.path.join(os.getcwd(), "hidden")
os.makedirs(output_dir, exist_ok=True)


# Initialize model wrapper
model = LSNsModel(model_path)

# Extract representations
reps = model.extract_representations(
    max_length=max_length,
    pooling=pooling,
    batch_size=batch_size,
)

# Get ordered layer names
num_layers = model.num_layers
layer_names = get_layer_names(model_path, num_layers)

# Stack across layers: shape = (num_examples, num_layers, hidden_size)
# by stacking on axis=1 instead of axis=0
pos_stack = np.stack([reps["positive"][ln] for ln in layer_names], axis=1)
neg_stack = np.stack([reps["negative"][ln] for ln in layer_names], axis=1)

# Save as .npy
pos_file = os.path.join(output_dir, "positive.npy")
neg_file = os.path.join(output_dir, "negative.npy")
np.save(pos_file, pos_stack)
np.save(neg_file, neg_stack)

print(f"Saved positive states to {pos_file} (shape {pos_stack.shape})")
print(f"Saved negative states to {neg_file} (shape {neg_stack.shape})")

np.save(neg_file, neg_stack)

print(f"Saved positive states to {pos_file} (shape {pos_stack.shape})")
print(f"Saved negative states to {neg_file} (shape {neg_stack.shape})")
