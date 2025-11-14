#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate LLaMA with mask ablation on SyntaxGym tasks (region-level surprisal).

This version:
- Uses HuggingFace dataset: cpllab/syntaxgym, config: all-2020
- Groups items by suite_name
- For each item:
    * For each condition, computes region-level surprisal S[cond][region_number]
    * Parses the SyntaxGym 'predictions' inequality expression
    * Substitutes (k;%cond%) → S['cond'][k] and evaluates the expression
- Accuracy = (# items where expression is True) / total

Requires:
- BaseModel.score(text) to return:
    {
        "input_ids": [...],
        "tokens": [...],
        "offsets": [(start, end), ...],   # per token char span
        "logprobs": [...],                # next-token logprobs, length = T-1
        "mean_logprob": float
    }

"""

import os
import re
import numpy as np
import torch
from datasets import load_dataset
from models.factory import ModelFactory


# ============================================================
# Mask loading
# ============================================================
def load_mask(model, size, pct, pooling, method):
    fname = f"{model}_{size}_{pct}pct_{pooling}_{method}_mask.npy"
    path = os.path.join("cache", fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found:\n{path}")

    print(f"Loaded mask: {path}")
    return np.load(path)


# ============================================================
# Region-level surprisal computation
# ============================================================
def compute_region_surprisals(ex, model):
    cond_names = ex["conditions"]["condition_name"]
    regions_per_cond = ex["conditions"]["regions"]   # list over conditions

    S = {}  # cond_name -> {region_number: surprisal(float)}

    for cond_idx, cond_name in enumerate(cond_names):
        sentence = ex["conditions"]["content"][cond_idx]
        region_info = regions_per_cond[cond_idx]
        words = region_info["content"]            # ['The', 'painting', 'that', ...]
        region_numbers = region_info["region_number"]  # [1, 2, 3, ..., N]

        word_spans = []
        pos = 0
        for w in words:
            start = sentence.find(w, pos)
            if start == -1:
                start = pos
            end = start + len(w)
            word_spans.append((start, end))
            pos = end + 1

        out = model.score(sentence)
        logprobs = out["logprobs"]      # T-1
        offsets = out["offsets"]        # T

        word_surprisal = [0.0 for _ in words]

        for tok_idx, (t_start, t_end) in enumerate(offsets):
            if t_start == t_end:
                continue

            for w_idx, (w_start, w_end) in enumerate(word_spans):
                if t_start >= w_start and t_end <= w_end:
                    if tok_idx > 0 and (tok_idx - 1) < len(logprobs):
                        lp = logprobs[tok_idx - 1]
                        word_surprisal[w_idx] += -float(lp)   # surprisal = -log p
                    break

        cond_S = {}
        for w_idx, r_id in enumerate(region_numbers):
            cond_S[int(r_id)] = float(word_surprisal[w_idx])

        S[cond_name] = cond_S

    return S


# ============================================================
# Prediction evaluation
# ============================================================
def evaluate_prediction(pred_str, S):
    def repl(m):
        region = int(m.group(1))
        cond = m.group(2)
        return f"S['{cond}'][{region}]"

    expr = re.sub(r"\((\d+);%([^%]+)%\)", repl, pred_str.strip())

    try:
        result = eval(expr, {"__builtins__": {}}, {"S": S})
        if isinstance(result, bool):
            return result
        return bool(result)
    except Exception as e:
        print(f"[!] Failed to evaluate prediction: {pred_str}")
        print(f"    Transformed expr: {expr}")
        print(f"    Error: {e}")
        return None


# ============================================================
# Evaluate SyntaxGym items (region-level)
# ============================================================
def eval_syntaxgym_task(task_dataset, model):
    correct = 0
    total = 0

    for ex in task_dataset:
        S = compute_region_surprisals(ex, model)
        pred_str = ex["predictions"][0]

        ok = evaluate_prediction(pred_str, S)
        if ok is None:
            continue

        if ok:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc, total


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--pct", type=float, default=0.5)
    parser.add_argument("--pooling", type=str, default="last")
    parser.add_argument("--method", type=str, default="nmd")
    parser.add_argument("--limit_tasks", type=int, default=None)

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Model mapping
    # ----------------------------------------------------------
    model_map = {
        ("llama3", "3B"): "meta-llama/Llama-3.2-3B-Instruct",
        ("llama3", "8B"): "meta-llama/Llama-3.1-8B-Instruct",
    }
    model_path = model_map[(args.model, args.size)]

    print("========================================")
    print(f"Model: {model_path}")
    print(f"Mask : {args.pct}pct, {args.pooling}, {args.method}")
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
    # Load SyntaxGym
    # ----------------------------------------------------------
    print("Loading SyntaxGym...")
    raw_data = load_dataset("cpllab/syntaxgym", "all-2020")
    all_items = raw_data["test"]   # ONLY one split!

    # group items by suite_name
    from collections import defaultdict
    subtasks = defaultdict(list)
    for ex in all_items:
        subtasks[ex["suite_name"]].append(ex)

    subtask_names = sorted(subtasks.keys())

    if args.limit_tasks:
        subtask_names = subtask_names[:args.limit_tasks]

    print("Found SyntaxGym subtasks:")
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
        acc, n = eval_syntaxgym_task(subtasks[name], model)
        baseline_scores[name] = acc
        print(f"{name:25s} ACC={acc:.3f} (n={n})")

    results["baseline"] = baseline_scores

    # ----------------------------------------------------------
    # 2) Ablation
    # ----------------------------------------------------------
    print(f"\n=== Ablation: {args.method} ===")
    model.set_language_selective_mask(mask_tensor)

    ablation_scores = {}
    for name in subtask_names:
        acc, n = eval_syntaxgym_task(subtasks[name], model)
        ablation_scores[name] = acc
        print(f"{name:25s} ACC={acc:.3f} (n={n})")

    results["ablation"] = ablation_scores

    # ----------------------------------------------------------
    # 3) Summary
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