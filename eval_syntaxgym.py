#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate LLaMA with mask ablation on SyntaxGym tasks (region-level surprisal).

"""

import os
import re
import numpy as np
import torch
from datasets import load_dataset
from collections import defaultdict
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
    Correct SyntaxGym region-level surprisal computation.
    Minimal changes only:
      ✓ Use norm_sentence for model.score()
      ✓ Sequential region matching (avoid matching wrong region)
    """

    cond_names = ex["conditions"]["condition_name"]
    regions_per_cond = ex["conditions"]["regions"]
    contents_per_cond = ex["conditions"]["content"]

    S = {}

    # Helper: normalize punctuation spacing
    def normalize_text(txt: str) -> str:
        if not isinstance(txt, str):
            txt = str(txt)
        txt = re.sub(r"\s+([.,!?;:])", r"\1", txt)
        return txt.strip()

    for cond_idx, cond_name in enumerate(cond_names):
        raw_sentence = contents_per_cond[cond_idx]
        raw_region_info = regions_per_cond[cond_idx]

        # ------------------------------------------------
        # 1. Normalize region info
        # ------------------------------------------------
        if isinstance(raw_region_info, dict):
            region_numbers = [int(x) for x in raw_region_info["region_number"]]
            region_texts   = [normalize_text(x) for x in raw_region_info["content"]]
        else:
            region_numbers = [int(r["region_number"]) for r in raw_region_info]
            region_texts   = [normalize_text(r["content"]) for r in raw_region_info]

        # Normalize entire sentence
        norm_sentence = normalize_text(raw_sentence)

        # ------------------------------------------------
        # 2. Score sentence using **normalized text** (critical fix)
        # ------------------------------------------------
        try:
            out = model.score(norm_sentence)
        except Exception as e:
            print(f"[!] score() failed:\n  {norm_sentence}\nError: {e}")
            continue

        logprobs = out["logprobs"]
        offsets  = out["offsets"]

        # ------------------------------------------------
        # 3. Find region spans sequentially (minimal change)
        # ------------------------------------------------
        region_spans = []
        search_pos = 0  # NEW — sequential matching

        for rtxt in region_texts:
            s = norm_sentence.find(rtxt, search_pos)
            if s == -1:
                print(f"[Warning] Region not found:\n '{rtxt}'\n in:\n '{norm_sentence}'")
                region_spans.append(None)
                continue

            e = s + len(rtxt)
            region_spans.append((s, e))
            search_pos = e  # move forward to avoid matching earlier substring

        # ------------------------------------------------
        # 4. Map to token spans
        # ------------------------------------------------
        region_token_spans = []

        for span in region_spans:
            if span is None:
                region_token_spans.append(None)
                continue

            start_char, end_char = span
            start_tok = None
            end_tok = None

            for i, (ts, te) in enumerate(offsets):
                if start_tok is None and ts <= start_char < te:
                    start_tok = i
                if ts < end_char <= te:
                    end_tok = i + 1
                    break

            if start_tok is not None and end_tok is None:
                end_tok = len(offsets)

            if start_tok is None or end_tok is None:
                print(f"[Warning] Token alignment failed for region span {span}")
                region_token_spans.append(None)
            else:
                region_token_spans.append((start_tok, end_tok))

        # ------------------------------------------------
        # 5. Sum surprisal per region
        # ------------------------------------------------
        cond_S = {r: 0.0 for r in region_numbers}

        for ridx, tok_span in enumerate(region_token_spans):
            if tok_span is None:
                continue

            region_id = region_numbers[ridx]
            t0, t1 = tok_span

            for t in range(t0, t1):
                if t == 0:
                    continue
                if t - 1 >= len(logprobs):
                    break
                cond_S[region_id] += -float(logprobs[t - 1])

        S[cond_name] = cond_S

    return S

# ============================================================
# Prediction evaluation
# ============================================================
def evaluate_prediction(pred_str, S):
    """
    Evaluate a SyntaxGym prediction expression
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
    import pandas as pd
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--pct", type=float, default=0.5)
    parser.add_argument("--pooling", type=str, default="last")
    parser.add_argument("--method", nargs="+", type=str, default=["nmd"])
    parser.add_argument("--baseline", action="store_true", default=False)
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
    print(f"Methods: {args.method}")
    print(f"Mask pct: {args.pct} | pooling={args.pooling}")
    print(f"Evaluate baseline: {args.baseline}")
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

    # ----------------------------------------------------------
    # Load SyntaxGym
    # ----------------------------------------------------------
    print("Loading SyntaxGym...")
    raw_data = load_dataset("cpllab/syntaxgym", "all-2020")
    all_items = raw_data["test"]

    subtasks = defaultdict(list)
    for ex in all_items:
        subtasks[ex["suite_name"]].append(ex)

    subtask_names = sorted(subtasks.keys())
    if args.limit_tasks:
        subtask_names = subtask_names[:args.limit_tasks]

    print("Found SyntaxGym subtasks:")
    for name in subtask_names:
        print(" -", name)

    # ----------------------------------------------------------
    # Result collector
    # ----------------------------------------------------------
    results = { "subtask": subtask_names }

    # ----------------------------------------------------------
    # 1) Baseline
    # ----------------------------------------------------------
    if args.baseline:
        print("\n=== Baseline Evaluation ===")
        model.set_language_selective_mask(None)

        baseline_scores = {}
        for name in subtask_names:
            acc, n = eval_syntaxgym_task(subtasks[name], model)
            baseline_scores[name] = acc
            print(f"{name:25s} ACC={acc:.3f} (n={n})")

        results["baseline"] = [baseline_scores[n] for n in subtask_names]

    # ----------------------------------------------------------
    # 2) Evaluate each method
    # ----------------------------------------------------------
    for method in args.method:

        print(f"\n=== Ablation: {method} ===")

        # load mask
        mask = load_mask(args.model, args.size, args.pct, args.pooling, method)
        mask_tensor = torch.tensor(1.0 - mask, dtype=torch.float16, device=device)

        # apply mask
        model.set_language_selective_mask(mask_tensor)

        method_scores = {}
        for name in subtask_names:
            acc, n = eval_syntaxgym_task(subtasks[name], model)
            method_scores[name] = acc
            print(f"{name:25s} ACC={acc:.3f} (n={n})")

        results[method] = [method_scores[n] for n in subtask_names]

    # ----------------------------------------------------------
    # 3) Save results to CSV
    # ----------------------------------------------------------
    df = pd.DataFrame(results)
    out_name = f"cache/syntaxgym_results_{args.model}_{args.size}_{args.pct}.csv"
    df.to_csv(out_name, index=False)

    print("\n========================================")
    print("Saved results to:", out_name)
    print("========================================")

if __name__ == "__main__":
    main()