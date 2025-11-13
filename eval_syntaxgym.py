#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate LLaMA with mask ablation on SyntaxGym minimal-pair tasks.

Supports:
- baseline (no mask)
- NMD ablation
- T-test ablation

SyntaxGym JSON suites must be downloaded locally:
https://github.com/cpllab/syntaxgym/tree/master/suites

Evaluation metric:
- For each test item: choose the condition with higher mean logprob
- Accuracy = (# correct) / total
"""

import os
import json
import glob
import numpy as np
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
        out = model.score(text)
        logprobs = out["logprobs"]  # shape (seq_len,)
        return float(np.mean(logprobs))
    except Exception as e:
        print(f"[!] score() failed on text: {text}")
        print("Error:", e)
        return float("-inf")


# ============================================================
# Evaluate one SyntaxGym task (JSON structure)
# ============================================================
def eval_syntaxgym_task(task_items, model):
    """
    Each JSON suite has:
    {
       "items": [
          {
             "conditioned_sentences": [...],
             "targets": index
          },
          ...
       ]
    }
    """
    correct = 0
    total = 0

    for ex in task_items:
        sentences = ex["conditioned_sentences"]
        gold_idx = ex["targets"]

        # Compute logprob for each version
        scores = [sentence_logprob(model, s) for s in sentences]

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
    path = os.path.join("cache", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found:\n{path}")
    print(f"Loaded mask: {path}")
    return np.load(path)


# ============================================================
# Load SyntaxGym JSON suites from local directory
# ============================================================
def load_syntaxgym_suites(suite_dir):
    suite_paths = sorted(glob.glob(os.path.join(suite_dir, "*.json")))
    if len(suite_paths) == 0:
        raise FileNotFoundError(f"No JSON suite files found in: {suite_dir}")

    suites = {}
    for path in suite_paths:
        with open(path, "r") as f:
            data = json.load(f)
            # key: filename without ".json"
            name = os.path.splitext(os.path.basename(path))[0]
            suites[name] = data
    return suites


# ============================================================
# Main Evaluation
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--suite_dir", type=str, required=True,
                        help="Path to SyntaxGym/suites directory")

    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="3B")
    parser.add_argument("--pct", type=float, default=0.5)
    parser.add_argument("--pooling", type=str, default="last")
    parser.add_argument("--method", type=str, default="nmd",
                        choices=["nmd", "ttest_abs", "ttest_signed"])

    parser.add_argument("--limit_tasks", type=int, default=None,
                        help="Evaluate only first N subtasks")

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Map model name
    # ----------------------------------------------------------
    model_map = {
        ("llama3", "3B"): "meta-llama/Llama-3.2-3B-Instruct",
        ("llama3", "8B"): "meta-llama/Llama-3.1-8B-Instruct",
    }

    key = (args.model, args.size)
    model_path = model_map[key]

    print("========================================")
    print(f"Model:   {model_path}")
    print(f"Mask:    {args.pct}pct, pooling={args.pooling}, method={args.method}")
    print(f"SuiteDir: {args.suite_dir}")
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
    # Load local SyntaxGym JSON suites
    # ----------------------------------------------------------
    suites = load_syntaxgym_suites(args.suite_dir)
    subtask_names = list(suites.keys())

    if args.limit_tasks:
        subtask_names = subtask_names[:args.limit_tasks]

    print("SyntaxGym subtasks:")
    for name in subtask_names:
        print(" -", name)

    results = {}

    # ----------------------------------------------------------
    # 1) Baseline
    # ----------------------------------------------------------
    print("\n=== Baseline Evaluation ===")
    model.set_language_selective_mask(None)
    baseline_scores = {}

    for name in subtask_names:
        acc, n = eval_syntaxgym_task(suites[name]["items"], model)
        baseline_scores[name] = acc
        print(f"{name:30s} ACC = {acc:.3f} (n={n})")

    results["baseline"] = baseline_scores

    # ----------------------------------------------------------
    # 2) Mask Ablation
    # ----------------------------------------------------------
    print(f"\n=== Ablation: {args.method} ===")
    model.set_language_selective_mask(mask_tensor)

    mask_scores = {}
    for name in subtask_names:
        acc, n = eval_syntaxgym_task(suites[name]["items"], model)
        mask_scores[name] = acc
        print(f"{name:30s} ACC = {acc:.3f} (n={n})")

    results["ablation"] = mask_scores

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n========================================")
    print("Summary:")
    print("Subtask                     | Baseline | Ablation | Δ")
    print("----------------------------------------")

    for name in subtask_names:
        b = results["baseline"][name]
        a = results["ablation"][name]
        print(f"{name:25s}  {b:6.3f}   {a:6.3f}   {a-b:+6.3f}")

    print("========================================")
    print("Done.")


if __name__ == "__main__":
    main()