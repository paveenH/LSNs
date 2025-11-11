#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Comparing T-test (abs vs signed) vs NMD neuron selection
on real model activations using unified ModelFactory interface.
"""

import os
import torch
import numpy as np
from models.factory import ModelFactory
from datasets import LangLocDataset, TOMLocDataset, MDLocDataset
from analysis.ttest_analyzer import TTestAnalyzer
from analysis.ttest_signed_analyzer import TTestSignedAnalyzer  
from analysis.nmd_analyzer import NMDAnalyzer


# ======================================================
# STEP 1 — Extract real activations using ModelFactory
# ======================================================
def extract_data(model_name="gpt2", network="language", pooling="last", batch_size=4, device=None):
    print(f"[1] Extracting real activations from {model_name} on {network} stimuli...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "device_map": device,
        "torch_dtype": "float32",
        "trust_remote_code": True,
    }
    model = ModelFactory.create_model(model_name, config)
    layer_names = model.get_layer_names()

    # Load stimuli set
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

    pos_acts = model.extract_activations(pos_texts, layer_names, pooling)
    neg_acts = model.extract_activations(neg_texts, layer_names, pooling)

    positive = np.stack([pos_acts[layer] for layer in layer_names], axis=1)
    negative = np.stack([neg_acts[layer] for layer in layer_names], axis=1)

    print(f"✅ Done: positive {positive.shape}, negative {negative.shape}")
    return positive, negative, layer_names


# ======================================================
# STEP 2 — Run all analyzers (T-test abs, signed, NMD)
# ======================================================
def run_all_analyses(positive, negative, layer_names):
    print("[2] Running analysis methods...")

    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    # (a) Absolute-value T-test
    ttest_abs = TTestAnalyzer({"percentage": 5.0, "localize_range": "100-100"})
    ttest_abs_mask, ttest_abs_meta = ttest_abs.analyze(positive, negative)

    # (b) Signed T-test 
    ttest_signed = TTestSignedAnalyzer({"percentage": 5.0, "localize_range": "100-100"})
    ttest_signed_mask, ttest_signed_meta = ttest_signed.analyze(positive, negative)

    # (c) NMD
    nmd_analyzer = NMDAnalyzer({"topk_ratio": 0.05})
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
# STEP 3 — Compare layerwise selection results
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

    # Pairwise overlaps
    overlap_abs_nmd = np.logical_and(ttest_abs, nmd).sum()
    overlap_signed_nmd = np.logical_and(ttest_signed, nmd).sum()
    overlap_abs_signed = np.logical_and(ttest_abs, ttest_signed).sum()

    print(f"\nOverlap(abs↔signed): {overlap_abs_signed / ttest_abs.sum():.3f}")
    print(f"Overlap(abs↔nmd):    {overlap_abs_nmd / ttest_abs.sum():.3f}")
    print(f"Overlap(signed↔nmd): {overlap_signed_nmd / ttest_signed.sum():.3f}")


# ======================================================
# STEP 4 — Optional: Test ablation using BaseModel wrapper
# ======================================================
def test_ablation(results, model_name="gpt2", device=None):
    print("\n[4] Testing ablation effects via BaseModel...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ModelFactory.create_model(model_name, config={})
    model.to_device(device)
    model.model.eval()

    prompt = "The quick brown fox"

    # Retrieve masks
    abs_mask = results.get("ttest_abs_mask")
    signed_mask = results.get("ttest_signed_mask")
    nmd_mask = results.get("nmd_mask")

    # Ablation methods
    methods = {
        "Baseline (no ablation)": None,
        "T-test abs ablation": 1 - abs_mask if abs_mask is not None else None,
        "T-test signed ablation": 1 - signed_mask if signed_mask is not None else None,
        "NMD ablation": 1 - nmd_mask if nmd_mask is not None else None,
    }

    for name, mask in methods.items():
        print(f"\n--- {name} ---")
        try:
            if mask is not None:
                mask_tensor = torch.tensor(mask, dtype=model.model.dtype, device=device)
                model.set_language_selective_mask(mask_tensor)
            else:
                model.set_language_selective_mask(None)
            output_text = model.generate(prompt, max_new_tokens=15, do_sample=False)
            print(output_text)
        except Exception as e:
            print(f"[!] Error during {name}: {e}")


# ======================================================
# MAIN PIPELINE
# ======================================================
def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    network = "language"
    pooling = "last"

    # Step 1
    positive, negative, layer_names = extract_data(
        model_name=model_name,
        network=network,
        pooling=pooling,
        batch_size=4,
    )

    # Step 2
    results = run_all_analyses(positive, negative, layer_names)

    # Step 3
    compare_selection(results)

    # Step 4
    test_ablation(results, model_name=model_name)


if __name__ == "__main__":
    main()