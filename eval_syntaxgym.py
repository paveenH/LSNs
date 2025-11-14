#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate LLaMA with mask ablation on SyntaxGym minimal-pair tasks.

Fully FIXED version for HuggingFace SyntaxGym:
- Correct dataset loading
- Group items by suite_name
- Convert regions → sentences
- Parse prediction inequalities
- Evaluate baseline + ablation

Author: ChatGPT (fixed for Paveen)
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from models.factory import ModelFactory
import re

# ============================================================
# Utility: Compute avg logprob for a sentence
# ============================================================
def sentence_logprob(model, text: str):
    try:
        out = model.score(text)
        logprobs = out["logprobs"]
        return float(np.mean(logprobs))
    except Exception as e:
        print(f"[!] score() failed on: {text}")
        print("Error:", e)
        return float("-inf")


# ============================================================
# Build full sentences from SyntaxGym item
# ============================================================
def build_sentences_from_item(ex):
    # ex["conditions"] is a dict
    cond_names = ex["conditions"]["condition_name"]   # ['plaus', 'implaus']
    sentences  = ex["conditions"]["content"]          # ['sentence1', 'sentence2']

    return dict(zip(cond_names, sentences))


# ============================================================
# Parse SyntaxGym inequality like:
# "( (6;%match%) + (7;%match%) ) < ( (6;%mismatch%) )"
#
# For minimal-pair classification, we only need:
#   which condition should have *higher* logprob (lower surprisal)
#
# If prediction is   match < mismatch
# → mismatch must have higher surprisal → match is correct answer.
# ============================================================
def get_gold_condition(pred_str, available_conds):
    """
    Robustly parse SyntaxGym inequality to decide which condition
    should have higher logprob (i.e., lower surprisal).

    We:
    - detect operator (< or >)
    - extract all %cond% occurrences
    - deduplicate in order and intersect with available_conds
    """

    # 1) 判断是 < 还是 >
    if "<" in pred_str:
        op = "<"
    elif ">" in pred_str:
        op = ">"
    else:
        # 无法解析的情况
        print(f"[!] Cannot find < or > in prediction: {pred_str}")
        return None

    # 2) 提取所有 %cond% 里的 cond 名
    conds = re.findall(r"%([^%]+)%", pred_str)  # ['plaus', 'plaus', 'implaus', 'implaus'] 等

    if not conds:
        print(f"[!] No %cond% pattern found in prediction: {pred_str}")
        return None

    # 3) 去重，同时只保留当前 item 里真正存在的 condition
    seen = set()
    uniq = []
    for c in conds:
        c = c.strip()
        if c in available_conds and c not in seen:
            seen.add(c)
            uniq.append(c)

    if len(uniq) == 0:
        print(f"[!] No usable condition names found in prediction: {pred_str}")
        return None
    if len(uniq) == 1:
        # 只有一个条件名，勉强返回这个（大多数任务不会这样）
        return uniq[0]
    # 正常情况，取前两个
    cond1, cond2 = uniq[0], uniq[1]

    # 4) 根据不等式方向决定 gold condition
    if op == "<":
        # cond1_surprisal < cond2_surprisal → cond1 logprob 更大
        return cond1
    else:  # op == ">"
        # cond1_surprisal > cond2_surprisal → cond2 logprob 更大
        return cond2

# ============================================================
# Evaluate SyntaxGym items
# ============================================================
def eval_syntaxgym_task(task_dataset, model):
    correct = 0
    total = 0

    for ex in task_dataset:

        sentences = build_sentences_from_item(ex)

        # prediction list (usually length=1)
        pred_str = ex["predictions"][0]        
        cond_names = ex["conditions"]["condition_name"]
        gold_cond = get_gold_condition(pred_str, cond_names)

        # compute scores for all sentences
        scores = {
            cond: sentence_logprob(model, sent)
            for cond, sent in sentences.items()
        }

        # choose highest logprob
        pred_cond = max(scores, key=scores.get)

        if pred_cond == gold_cond:
            correct += 1
        total += 1

    return (correct / total if total else 0.0), total


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