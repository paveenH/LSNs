#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Comparing T-test vs NMD neuron selection on real model activations
(using language-localizer stimuli and unified model.extract_activations()).

This demo replicates the NMD vs T-test comparison from the paper,
but now uses real hidden activations obtained directly from the model,
through the standardized ModelFactory wrapper (same API as in brain_alignment.py).
"""

import torch
import numpy as np
from models.factory import ModelFactory
from datasets import LangLocDataset, TOMLocDataset, MDLocDataset
from analysis.ttest_analyzer import TTestAnalyzer
from analysis.nmd_analyzer import NMDAnalyzer
from generate_lesion import PaperCorrectMaskedGPT2
from transformers import AutoTokenizer


# ======================================================
# STEP 1 — Extract real activations using ModelFactory
# ======================================================
def extract_data(model_name="gpt2", network="language", pooling="last", batch_size=4, device=None):
    """
    Extracts hidden activations from the given model using the same mechanism
    as brain_alignment.py (through model.extract_activations()).

    Args:
        model_name: str, model identifier (e.g. "gpt2", "meta-llama/Llama-3.2-1B")
        network: str, one of ["language", "theory-of-mind", "multiple-demand"]
        pooling: str, token pooling method ("last", "mean", "sum", "orig")
        batch_size: int, number of stimuli per batch
        device: torch.device or str, e.g. "cuda" or "cpu"

    Returns:
        positive: np.ndarray, shape (N_pos, L, H)
        negative: np.ndarray, shape (N_neg, L, H)
        layer_names: list of layer identifiers
    """
    print(f"[1] Extracting real activations from {model_name} on {network} stimuli...")

    # Select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model via unified factory (BaseModel wrapper)
    config = {
        "device_map": device,
        "torch_dtype": "float32",
        "trust_remote_code": True,
    }
    model = ModelFactory.create_model(model_name, config)
    layer_names = model.get_layer_names()

    # Load localization dataset
    if network == "language":
        dataset = LangLocDataset()
    elif network == "theory-of-mind":
        dataset = TOMLocDataset()
    elif network == "multiple-demand":
        dataset = MDLocDataset()
    else:
        raise ValueError(f"Unsupported network type: {network}")

    # Extract per-layer activations for positive and negative stimuli
    pos_texts = [str(x) for x in dataset.positive]
    neg_texts = [str(x) for x in dataset.negative]

    print(f"Extracting {len(pos_texts)} positive and {len(neg_texts)} negative examples...")

    pos_acts = model.extract_activations(pos_texts, layer_names, pooling)
    neg_acts = model.extract_activations(neg_texts, layer_names, pooling)

    # Stack into (N, L, H)
    positive = np.stack([pos_acts[layer] for layer in layer_names], axis=1)
    negative = np.stack([neg_acts[layer] for layer in layer_names], axis=1)

    print(f"✅ Done: positive {positive.shape}, negative {negative.shape}")
    return positive, negative, layer_names


# ======================================================
# STEP 2 — Run both analyzers (T-test & NMD)
# ======================================================
def run_both_analyses(positive, negative, layer_names):
    print("[2] Running analysis methods...")

    # Global T-test analyzer
    ttest_analyzer = TTestAnalyzer({"percentage": 5.0, "localize_range": "100-100"})
    ttest_mask, ttest_meta = ttest_analyzer.analyze(positive, negative)

    # Layerwise NMD analyzer
    nmd_analyzer = NMDAnalyzer({"topk_ratio": 0.05})
    nmd_mask, nmd_meta = nmd_analyzer.analyze(positive, negative)

    print(f"T-test selected: {ttest_mask.sum()} neurons ({ttest_meta['selection_ratio']:.3f})")
    print(f"NMD selected: {nmd_mask.sum()} neurons ({nmd_meta['selection_ratio']:.3f})")

    np.save("cache/real_ttest_mask.npy", ttest_mask)
    np.save("cache/real_nmd_mask.npy", nmd_mask)

    return {"ttest_mask": ttest_mask, "nmd_mask": nmd_mask, "layer_names": layer_names}


# ======================================================
# STEP 3 — Compare layerwise selection results
# ======================================================
def compare_selection(results):
    ttest_mask = results["ttest_mask"]
    nmd_mask = results["nmd_mask"]

    ttest_per_layer = ttest_mask.sum(axis=1)
    nmd_per_layer = nmd_mask.sum(axis=1)

    print("\n[3] Per-layer Comparison:")
    print(f"{'Layer':<6} {'T-test':<8} {'NMD':<8} {'Diff':<8}")
    print("-" * 36)

    for i in range(len(ttest_per_layer)):
        diff = int(ttest_per_layer[i]) - int(nmd_per_layer[i])
        print(f"{i:<6} {int(ttest_per_layer[i]):<8} {int(nmd_per_layer[i]):<8} {diff:+4d}")

    overlap = np.logical_and(ttest_mask, nmd_mask).sum()
    print(f"\nOverlap: {overlap} / {ttest_mask.sum()} = {overlap / ttest_mask.sum():.3f}")


# ======================================================
# STEP 4 — Optional: Test ablation using PaperCorrectMaskedGPT2
# ======================================================
def test_ablation(results, model_name="gpt2"):
    """
    Performs a quick qualitative ablation test using GPT-2 and the generated masks.
    This uses the PaperCorrectMaskedGPT2 class with forward hooks.
    """
    print("\n[4] Testing ablation effects (GPT-2 only)...")

    try:
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
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"{name:<22}: {text}")

    except Exception as e:
        print(f"Ablation test skipped due to error: {e}")
        import traceback
        traceback.print_exc()


# ======================================================
# MAIN PIPELINE
# ======================================================
def main():
    model_name = "gpt2"
    network = "language"
    pooling = "last"

    # Step 1: Extract real activations
    positive, negative, layer_names = extract_data(
        model_name=model_name,
        network=network,
        pooling=pooling,
        batch_size=4,
    )

    # Step 2: Run both analyses
    results = run_both_analyses(positive, negative, layer_names)

    # Step 3: Compare layerwise selection
    compare_selection(results)

    # Step 4: Test ablation (optional)
    test_ablation(results, model_name=model_name)


if __name__ == "__main__":
    main()