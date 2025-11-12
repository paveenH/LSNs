#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Comparing T-test (abs vs signed) vs NMD neuron selection
on real model activations using unified ModelFactory interface.
Supports command-line arguments.
"""

import os
import torch
import numpy as np
import argparse
from models.factory import ModelFactory
from datasets import LangLocDataset, TOMLocDataset, MDLocDataset
from analysis.ttest_analyzer import TTestAnalyzer
from analysis.ttest_signed_analyzer import TTestSignedAnalyzer  
from analysis.nmd_analyzer import NMDAnalyzer


# ======================================================
# STEP 1 — Extract activations
# ======================================================
def extract_data(model_name, network, pooling, batch_size):
    print(f"[1] Extracting activations from {model_name} on {network} stimuli...")        

    config = {
        "device_map": "auto",
        "torch_dtype": "float16", 
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    model = ModelFactory.create_model(model_name, config)
    layer_names = model.get_layer_names()

    # Dataset selection
    if network == "language":
        dataset = LangLocDataset()
    elif network == "theory-of-mind":
        dataset = TOMLocDataset()
    elif network == "multiple-demand":
        dataset = MDLocDataset()
    else:
        raise ValueError(f"Unsupported network type: {network}")

    pos_texts = [str(x) for x in dataset.positive]
    neg_texts = [str(x) for x in dataset.negative]

    print(f"Extracting {len(pos_texts)} positive and {len(neg_texts)} negative examples...")

    def batched_extract(texts, batch_size=8):
        all_batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            acts = model.extract_activations(batch, layer_names, pooling)
            batch_stack = np.stack([acts[layer] for layer in layer_names], axis=1)
            all_batches.append(batch_stack)
            torch.cuda.empty_cache()
        return np.concatenate(all_batches, axis=0)

    positive = batched_extract(pos_texts, batch_size)
    negative = batched_extract(neg_texts, batch_size)

    print(f"✅ Done: positive {positive.shape}, negative {negative.shape}")
    return positive, negative, layer_names


# ======================================================
# STEP 2 — Run analyzers
# ======================================================
def run_all_analyses(positive, negative, layer_names, percentage=5.0):
    print("[2] Running analysis methods...")

    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    # (a) Absolute-value T-test
    ttest_abs = TTestAnalyzer({"percentage": percentage, "localize_range": "100-100"})
    ttest_abs_mask, ttest_abs_meta = ttest_abs.analyze(positive, negative)

    # (b) Signed T-test 
    ttest_signed = TTestSignedAnalyzer({"percentage": percentage, "localize_range": "100-100"})
    ttest_signed_mask, ttest_signed_meta = ttest_signed.analyze(positive, negative)

    # (c) NMD
    nmd_analyzer = NMDAnalyzer({"topk_ratio": percentage / 100.0})
    nmd_mask, nmd_meta = nmd_analyzer.analyze(positive, negative)

    print(f"T-test (abs) selected:   {ttest_abs_mask.sum()} neurons ({ttest_abs_meta['selection_ratio']:.3f})")
    print(f"T-test (signed) selected:{ttest_signed_mask.sum()} neurons ({ttest_signed_meta['selection_ratio']:.3f})")
    print(f"NMD selected:             {nmd_mask.sum()} neurons ({nmd_meta['selection_ratio']:.3f})")

    # Save masks
    np.save(os.path.join(cache_dir, "real_ttest_abs_mask.npy"), ttest_abs_mask)
    np.save(os.path.join(cache_dir, "real_ttest_signed_mask.npy"), ttest_signed_mask)
    np.save(os.path.join(cache_dir, "real_nmd_mask.npy"), nmd_mask)

    return {
        "ttest_abs_mask": ttest_abs_mask,
        "ttest_signed_mask": ttest_signed_mask,
        "nmd_mask": nmd_mask,
        "layer_names": layer_names,
    }


# ======================================================
# STEP 3 — Layerwise Comparison
# ======================================================
def compare_selection(results):
    ttest_abs = results["ttest_abs_mask"]
    ttest_signed = results["ttest_signed_mask"]
    nmd = results["nmd_mask"]

    per_layer_abs = ttest_abs.sum(axis=1)
    per_layer_signed = ttest_signed.sum(axis=1)
    per_layer_nmd = nmd.sum(axis=1)

    print("\n[3] Per-layer Comparison:")
    print(f"{'Layer':<6} {'AbsT':<8} {'SignedT':<8} {'NMD':<8}")
    print("-" * 40)
    for i in range(len(per_layer_abs)):
        print(f"{i:<6} {int(per_layer_abs[i]):<8} {int(per_layer_signed[i]):<8} {int(per_layer_nmd[i]):<8}")

    overlap_abs_nmd = np.logical_and(ttest_abs, nmd).sum()
    overlap_signed_nmd = np.logical_and(ttest_signed, nmd).sum()
    overlap_abs_signed = np.logical_and(ttest_abs, ttest_signed).sum()

    print(f"\nOverlap(abs↔signed): {overlap_abs_signed / ttest_abs.sum():.3f}")
    print(f"Overlap(abs↔nmd):    {overlap_abs_nmd / ttest_abs.sum():.3f}")
    print(f"Overlap(signed↔nmd): {overlap_signed_nmd / ttest_signed.sum():.3f}")


# ======================================================
# STEP 4 — Test ablation
# ======================================================
def test_ablation(results, model_name):
    print("\n[4] Testing ablation effects via BaseModel...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelFactory.create_model(model_name, config={})
    model.to_device(device)
    model.model.eval()

    prompt = "The quick brown fox"

    abs_mask = results.get("ttest_abs_mask")
    signed_mask = results.get("ttest_signed_mask")
    nmd_mask = results.get("nmd_mask")

    methods = {
        "Baseline (no ablation)": None,
        "T-test abs ablation": 1 - abs_mask,
        "T-test signed ablation": 1 - signed_mask,
        "NMD ablation": 1 - nmd_mask,
    }

    for name, mask in methods.items():
        print(f"\n--- {name} ---")
        try:
            mask_tensor = None
            if mask is not None:
                mask_tensor = torch.tensor(mask, dtype=torch.float16, device=device)
            model.set_language_selective_mask(mask_tensor)
            output_text = model.generate(prompt, max_new_tokens=15, do_sample=False)
            print(output_text)
        except Exception as e:
            print(f"[!] Error during {name}: {e}")


# ======================================================
# MAIN ENTRY
# ======================================================
def main():
    parser = argparse.ArgumentParser(description="Compare neuron selectivity methods (T-test, NMD)")
    parser.add_argument("--model", type=str, default="llama3", help="Base model name (e.g., llama3, gpt2)")
    parser.add_argument("--size", type=str, default="1B", help="Model size (e.g., 1B, 3B, 8B)")
    parser.add_argument("--network", type=str, default="language", 
                        choices=["language", "theory-of-mind", "multiple-demand"],
                        help="Network type (stimuli domain)")
    parser.add_argument("--pooling", type=str, default="last",
                        choices=["last", "mean", "sum", "orig"],
                        help="Pooling strategy for activations")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for extraction")
    parser.add_argument("--percentage", type=float, default=5.0, help="Percentage of neurons to select")
    args = parser.parse_args()

    # Mapping table
    model_map = {
        # llama3 family
        ("llama3", "8B"): "meta-llama/Llama-3.1-8B-Instruct",
        ("llama3", "3B"): "meta-llama/Llama-3.2-3B-Instruct",
        ("llama3", "1B"): "meta-llama/Llama-3.2-1B-Instruct",
        # GPT2 (single entry, no size)
        ("gpt2", None): "gpt2",
    }

    # Resolve model path
    key = (args.model.lower(), args.size.upper() if hasattr(args, "size") else None)
    if key not in model_map:
        if args.model.lower() == "gpt2":
            model_path = "gpt2"
        else:
            raise ValueError(f"Unknown model combination: {args.model}, {args.size}")
    else:
        model_path = model_map[key]

    print(f"🧩 Using model: {model_path}")

    # Step 1
    positive, negative, layer_names = extract_data(
        model_name=model_path,
        network=args.network,
        pooling=args.pooling,
        batch_size=args.batch_size,
    )

    # Step 2
    results = run_all_analyses(
        positive, negative, layer_names, args.percentage, model_name=model_path
    )

    # Step 3
    compare_selection(results)

    # Step 4
    test_ablation(results, model_name=model_path)