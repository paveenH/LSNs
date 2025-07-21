#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and save layer-wise hidden states as .npy files using LSNsModel.
"""

import os
import argparse
import numpy as np
from llms import LSNsModel, get_layer_names

def main():
    parser = argparse.ArgumentParser(description="Extract hidden states and save as npy files")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Model name or path, e.g., 'mistralai/Mistral-7B-v0.3'")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for DataLoader")
    parser.add_argument("--max-length", type=int, default=12,
                        help="Maximum token length for tokenizer")
    parser.add_argument("--pooling", type=str, choices=["last-token", "mean", "sum"], default="last-token",
                        help="Pooling strategy to apply to each layer's output")
    parser.add_argument("--output-dir", type=str, default="hidden",
                        help="Directory to save output .npy files")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    model = LSNsModel(args.model_path)

    # Extract representations
    reps = model.extract_representations(
        max_length=args.max_length,
        pooling=args.pooling,
        batch_size=args.batch_size,
    )

    # Get ordered layer names
    layer_names = get_layer_names(args.model_path, model.num_layers)

    # Stack into (N, L, D)
    pos_stack = np.stack([reps["positive"][ln] for ln in layer_names], axis=1)
    neg_stack = np.stack([reps["negative"][ln] for ln in layer_names], axis=1)

    # Save
    pos_file = os.path.join(args.output_dir, "positive.npy")
    neg_file = os.path.join(args.output_dir, "negative.npy")
    np.save(pos_file, pos_stack)
    np.save(neg_file, neg_stack)

    print(f"Saved positive states to {pos_file} (shape {pos_stack.shape})")
    print(f"Saved negative states to {neg_file} (shape {neg_stack.shape})")

if __name__ == "__main__":
    main()