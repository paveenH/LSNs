#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: T-test vs NMD on real activations (Language localizer stimuli)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from analysis.ttest_analyzer import TTestAnalyzer
from analysis.nmd_analyzer import NMDAnalyzer
from generate_lesion import PaperCorrectMaskedGPT2
from localize import extract_representations
from model_utils import get_layer_names, get_hidden_dim


# ======================
# STEP 1 — extract real activations
# ======================
def extract_data(model_name="gpt2", network="language", pooling="last-token", batch_size=4, device=None):
    print(f"[1] Extracting real activations from {model_name} on {network} stimuli...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    layer_names = get_layer_names(model_name)
    hidden_dim = get_hidden_dim(model_name)
    
    # Run localization data extraction (LangLocDataset etc.)
    activations = extract_representations(
        network=network,
        pooling=pooling,
        model=model,
        tokenizer=tokenizer,
        layer_names=layer_names,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        device=device,
    )

    # Reshape to match analyzers’ expected format, shape = (N, L, H)
    positive = np.stack([activations["positive"][l] for l in layer_names], axis=1)
    negative = np.stack([activations["negative"][l] for l in layer_names], axis=1) 

    print(f"✅ Done: positive {positive.shape}, negative {negative.shape}")
    return positive, negative, layer_names


# ======================
# STEP 2 — run both analyses
# ======================
def run_both_analyses(positive, negative, layer_names):
    print("[2] Running analysis methods...")

    # T-test (global)
    ttest_analyzer = TTestAnalyzer({"percentage": 5.0, "localize_range": "100-100"})
    ttest_mask, ttest_meta = ttest_analyzer.analyze(positive, negative)

    # NMD (per-layer)
    nmd_analyzer = NMDAnalyzer({"topk_ratio": 0.05})
    nmd_mask, nmd_meta = nmd_analyzer.analyze(positive, negative)

    print(f"T-test selected: {ttest_mask.sum()} neurons ({ttest_meta['selection_ratio']:.3f})")
    print(f"NMD selected: {nmd_mask.sum()} neurons ({nmd_meta['selection_ratio']:.3f})")

    np.save("cache/real_ttest_mask.npy", ttest_mask)
    np.save("cache/real_nmd_mask.npy", nmd_mask)

    return {"ttest_mask": ttest_mask, "nmd_mask": nmd_mask, "layer_names": layer_names}


# ======================
# STEP 3 — compare layerwise results
# ======================
def compare_selection(results):
    ttest_mask = results["ttest_mask"]
    nmd_mask = results["nmd_mask"]

    ttest_per_layer = ttest_mask.sum(axis=1)
    nmd_per_layer = nmd_mask.sum(axis=1)

    print("\n[3] Per-layer Comparison:")
    print(f"{'Layer':<6} {'T-test':<8} {'NMD':<6} {'Diff':<8}")
    print("-" * 35)

    for i in range(len(ttest_per_layer)):
        diff = int(ttest_per_layer[i]) - int(nmd_per_layer[i])
        print(f"{i:<6} {int(ttest_per_layer[i]):<8} {int(nmd_per_layer[i]):<6} {diff:+4d}")

    overlap = np.logical_and(ttest_mask, nmd_mask).sum()
    print(f"\nOverlap: {overlap} / {ttest_mask.sum()} = {overlap / ttest_mask.sum():.3f}")


# ======================
# STEP 4 — test ablation (optional)
# ======================
def test_ablation(results, model_name="gpt2"):
    print("\n[4] Testing ablation effects...")

    model = PaperCorrectMaskedGPT2.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")

    methods = {
        "Baseline (no ablation)": None,
        "T-test ablation": 1 - results["ttest_mask"],
        "NMD ablation": 1 - results["nmd_mask"],
    }

    for name, mask in methods.items():
        if mask is not None:
            model.set_language_selective_mask(torch.tensor(mask, dtype=torch.float32))
        else:
            model.set_language_selective_mask(None)

        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        print(f"{name:<20}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


# ======================
# MAIN PIPELINE
# ======================
def main():
    model_name = "gpt2"
    network = "language"
    pooling = "last-token"

    positive, negative, layer_names = extract_data(
        model_name=model_name, network=network, pooling=pooling, batch_size=4
        )

    results = run_both_analyses(positive, negative, layer_names)
    compare_selection(results)
    test_ablation(results, model_name=model_name)


if __name__ == "__main__":
    main()