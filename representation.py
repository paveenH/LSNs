#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and save layer-wise hidden states as .npy files using LSNsModel.
"""

import os
import argparse
import numpy as np
from llms import LSNsModel, get_layer_names
from datasets import LangLocDataset


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states and save as npy files")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name, e.g., 'Llama3'")
    parser.add_argument("--model-size", type=str, required=True,
                        help="Model size, e.g., '8B'")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Model name or path, e.g., 'mistralai/Mistral-7B-v0.3'")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for DataLoader")
    parser.add_argument("--pooling", type=str, choices=["last", "mean", "sum"], default="last",
                        help="Pooling strategy to apply to each layer's output")
    parser.add_argument("--output-dir", type=str, default="hidden",
                        help="Top-level directory to save output .npy files")

    args = parser.parse_args()
    
    dataset = LangLocDataset()

    # Construct full output path: e.g., hidden/Llama3
    mode_name_save = f"{args.model_name}"
    full_output_dir = os.path.join(args.output_dir, mode_name_save)
    os.makedirs(full_output_dir, exist_ok=True)

    # Initialize model wrapper
    model = LSNsModel(args.model_path)

    # Extract representations
    reps = model.extract_representations(
        pooling=args.pooling,
        batch_size=args.batch_size,
        dataset=dataset
    )

    # Get ordered layer names
    layer_names = get_layer_names(args.model_path, model.num_layers)

    # Stack and save
    pos_stack = np.stack([reps["positive"][ln] for ln in layer_names], axis=1)
    neg_stack = np.stack([reps["negative"][ln] for ln in layer_names], axis=1)

    pos_file = os.path.join(full_output_dir, f"positive_{args.model_size}_{args.pooling}.npy")
    neg_file = os.path.join(full_output_dir, f"negative_{args.model_size}_{args.pooling}.npy")
    np.save(pos_file, pos_stack)
    np.save(neg_file, neg_stack)

    print(f"Saved positive states to {pos_file} (shape {pos_stack.shape})")
    print(f"Saved negative states to {neg_file} (shape {neg_stack.shape})")

if __name__ == "__main__":
    main()
    