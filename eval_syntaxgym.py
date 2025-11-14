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
    """
    For one SyntaxGym item:
        - For each condition, compute surprisal per region_number.
        - Surprisal(region k) = sum_{tokens in region k} -log p(token_t | prefix).

    Returns:
        S: dict
            cond_name -> { region_number(int): surprisal(float) }
    """
    cond_names = ex["conditions"]["condition_name"]
    regions_per_cond = ex["conditions"]["regions"]       # list over conditions
    contents_per_cond = ex["conditions"]["content"]      # full sentences per cond

    S = {}  # cond_name -> {region_number: surprisal(float)}

    for cond_idx, cond_name in enumerate(cond_names):
        sentence = contents_per_cond[cond_idx]
        region_info = regions_per_cond[cond_idx]

        region_texts = region_info["content"]            # e.g. ['The', 'painting', 'that', ...]
        region_numbers = region_info["region_number"]    # e.g. [1, 2, 3, ..., N]

        # --------------------------------------------------
        # 1) Compute char-span for each region (in sentence)
        #    region_spans[i] = (start_char, end_char) for region i
        # --------------------------------------------------
        region_spans = []
        pos = 0
        for chunk in region_texts:
            chunk = str(chunk)
            start = sentence.find(chunk, pos)
            if start == -1:
                # Fallback: try from beginning; if still -1, approximate with pos
                start = sentence.find(chunk)
                if start == -1:
                    start = pos
            end = start + len(chunk)
            region_spans.append((start, end))
            # move pos to end to avoid matching earlier occurrences
            pos = end

        # --------------------------------------------------
        # 2) Get token-level logprobs & offsets from model
        # --------------------------------------------------
        try:
            out = model.score(sentence)
        except Exception as e:
            print(f"[!] score() failed on sentence: {sentence}")
            print("    Error:", e)
            # skip this condition (very rare)
            continue

        logprobs = out["logprobs"]   # length T-1
        offsets = out["offsets"]     # length T (one span per token)

        # --------------------------------------------------
        # 3) Aggregate surprisal per region
        # --------------------------------------------------
        # initialize region surprisal dict
        cond_S = {int(r): 0.0 for r in region_numbers}

        for tok_idx, (t_start, t_end) in enumerate(offsets):
            # some tokenizers may produce empty spans
            if t_start == t_end:
                continue

            # token 0 has no next-token logprob (we use next-token prediction)
            if tok_idx == 0 or (tok_idx - 1) >= len(logprobs):
                continue

            lp = logprobs[tok_idx - 1]
            surpr = -float(lp)  # surprisal = -log p(token_t | prefix)

            # assign this token to the first region whose span fully covers it
            for r_idx, (r_start, r_end) in enumerate(region_spans):
                if t_start >= r_start and t_end <= r_end:
                    region_id = int(region_numbers[r_idx])
                    cond_S[region_id] += surpr
                    break

        S[cond_name] = cond_S

    return S


# ============================================================
# Prediction evaluation
# ============================================================
def evaluate_prediction(pred_str, S):
    """
    Evaluate a SyntaxGym prediction expression, e.g.:

        ((6;%plaus%) + (7;%plaus%)) < ((6;%implaus%) + (7;%implaus%))

    After replacement:
        ((S['plaus'][6] + S['plaus'][7])) < ((S['implaus'][6] + S['implaus'][7]))

    Returns:
        True / False / None (if evaluation fails)
    """

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
            # skip items where the expression could not be evaluated
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