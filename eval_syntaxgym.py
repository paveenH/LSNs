#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate LLaMA with mask ablation on SyntaxGym minimal-pair tasks.

Supports:
- baseline (no mask)
- NMD ablation
- T-test ablation

SyntaxGym dataset from HuggingFace datasets:
https://huggingface.co/datasets/syntaxgym

Evaluation metric:
- For each test item: choose the condition with higher mean logprob
- Accuracy = (# correct) / total
"""

import os
import numpy as np
import datasets as hf_datasets
import torch
from models.factory import ModelFactory


# ============================================================
# Utility: Compute avg logprob for a sentence
# ============================================================
def sentence_logprob(model, text: str):
    """
    Return mean logprob over tokens for the given text using BaseModel.
    Your BaseModel must support .score(text) returning per-token logprobs.
    """
    try:
        # Many LLaMA inference wrappers return dict { "logprobs": [...] }
        out = model.score(text)
        # Expect out["logprobs"]: shape (seq_len,)
        logprobs = out["logprobs"]
        return float(np.mean(logprobs))
    except Exception as e:
        print(f"[!] score() failed on text: {text}")
        print("Error:", e)
        return float("-inf")


# ============================================================
# Evaluate one SyntaxGym task
# ============================================================
def eval_syntaxgym_task(task_dataset, model):
    """
    Evaluate one SyntaxGym benchmark (e.g., "agreement", "island_effects"...)
    """
    correct = 0
    total   = 0

    for ex in task_dataset:
        # SyntaxGym format:
        # ex["conditioned_sentences"] = list of different sentence versions
        # ex["targets"] = index of the correct condition
        sentences = ex["conditioned_sentences"]
        gold_idx  = ex["targets"]

        # Compute logprob for each version
        scores = []
        for s in sentences:
            lp = sentence_logprob(model, s)
            scores.append(lp)

        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc, total


# ============================================================
# Mask loading
# ============================================================
def load_mask(model, size, pct, pooling, method):
    fname = f"{model}_{size}_{pct}pct_{pooling}_{method}_mask.npy"
    path  = os.path.join("cache", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found:\n{path}")
    print(f"Loaded mask: {path}")
    return np.load(path)


# ============================================================
# Main Evaluation
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",  type=str, default="llama3")
    parser.add_argument("--size",   type=str, default="3B")
    parser.add_argument("--pct",    type=float, default=0.5)
    parser.add_argument("--pooling", type=str, default="last")
    parser.add_argument("--method", type=str, default="nmd",
                        choices=["nmd", "ttest_abs", "ttest_signed"])

    parser.add_argument("--limit_tasks", type=int, default=None,
                        help="Only evaluate first N SyntaxGym subtasks")

    args = parser.parse_args()

    # HuggingFace model path
    model_map = {
        ("llama3", "3B"): "meta-llama/Llama-3.2-3B-Instruct",
        ("llama3", "8B"): "meta-llama/Llama-3.1-8B-Instruct",
    }

    key = (args.model, args.size)
    model_path = model_map[key]

    print("========================================")
    print(f"Model:   {model_path}")
    print(f"Mask:    {args.pct}pct, pooling={args.pooling}, method={args.method}")
    print("========================================")

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------
    model = ModelFactory.create_model(model_path, config={
        "device_map": "auto",
        "torch_dtype": "float16",
    })
    device = model.device
    model.model.eval()

    # Load mask
    mask = load_mask(args.model, args.size, args.pct, args.pooling, args.method)
    mask_tensor = torch.tensor(1.0 - mask, dtype=torch.float16, device=device)

    # ----------------------------------------------------------
    # Load SyntaxGym dataset
    # ----------------------------------------------------------
    syntaxgym = hf_datasets.load_dataset("syntaxgym")
    subtask_names = list(syntaxgym.keys())

    if args.limit_tasks:
        subtask_names = subtask_names[:args.limit_tasks]

    print("SyntaxGym subtasks:")
    for name in subtask_names:
        print(" -", name)

    # ----------------------------------------------------------
    # Evaluate under 3 conditions:
    # baseline, NMD ablation, t-test ablation
    # ----------------------------------------------------------

    results = {}

    # 1) Baseline (no mask)
    print("\n=== Baseline Evaluation ===")
    model.set_language_selective_mask(None)
    baseline_scores = {}
    for name in subtask_names:
        acc, n = eval_syntaxgym_task(syntaxgym[name]["test"], model)
        baseline_scores[name] = acc
        print(f"{name:25s}  ACC = {acc:.3f}   (n={n})")
    results["baseline"] = baseline_scores

    # 2) Mask ablation
    print(f"\n=== Ablation: {args.method} ===")
    model.set_language_selective_mask(mask_tensor)
    mask_scores = {}
    for name in subtask_names:
        acc, n = eval_syntaxgym_task(syntaxgym[name]["test"], model)
        mask_scores[name] = acc
        print(f"{name:25s}  ACC = {acc:.3f}   (n={n})")
    results["ablation"] = mask_scores

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n========================================")
    print("Summary:")
    print("Subtask                  | Baseline | Ablation | Δ")
    print("----------------------------------------")

    for name in subtask_names:
        b  = results["baseline"][name]
        a  = results["ablation"][name]
        d  = a - b
        print(f"{name:25s}  {b:6.3f}   {a:6.3f}   {d:+6.3f}")

    print("========================================")
    print("Done.")


if __name__ == "__main__":
    main()