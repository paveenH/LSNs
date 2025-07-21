#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:03:26 2025

@author: paveenhuang
"""

import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Average hidden states across samples.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing positive_*.npy and negative_*.npy")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the averaged files")
    parser.add_argument("--size", type=str, required=True, help="Model identifier, e.g., 8B")
    args = parser.parse_args()

    positive_path = os.path.join(args.input_dir, f"positive_{args.model_name}.npy")
    negative_path = os.path.join(args.input_dir, f"negative_{args.model_name}.npy")

    # Load original hidden states
    positive_hidden = np.load(positive_path)  # shape: (240, 32, 4096)
    negative_hidden = np.load(negative_path)

    # shape (32, 4096)
    positive_mean = positive_hidden.mean(axis=0)
    negative_mean = negative_hidden.mean(axis=0)

    # output
    os.makedirs(args.output_dir, exist_ok=True)

    positive_out = os.path.join(args.output_dir, f"positive_{args.size}.npy")
    negative_out = os.path.join(args.output_dir, f"negative_{args.size}.npy")
    np.save(positive_out, positive_mean)
    np.save(negative_out, negative_mean)

    print(f"> Saved mean positive to {positive_out} (shape {positive_mean.shape})")
    print(f"> Saved mean negative to {negative_out} (shape {negative_mean.shape})")

if __name__ == "__main__":
    main()