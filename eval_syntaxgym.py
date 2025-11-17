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
    Corrected SyntaxGym region-level surprisal computation.
    Uses character offset alignment for robust region matching.
    """
    tokenizer = model.tokenizer

    cond_names = ex["conditions"]["condition_name"]
    regions_per_cond = ex["conditions"]["regions"]
    contents_per_cond = ex["conditions"]["content"]

    S = {}

    for cond_idx, cond_name in enumerate(cond_names):
        sentence = contents_per_cond[cond_idx]
        region_info = regions_per_cond[cond_idx]

        # ------------------------------------------------
        # 1) Normalize region_info into list-of-dicts
        # ------------------------------------------------
        if isinstance(region_info, dict):
            region_numbers = [int(x) for x in region_info["region_number"]]
            region_texts = [str(x) for x in region_info["content"]]
        else:
            region_numbers = [int(r["region_number"]) for r in region_info]
            region_texts = [str(r["content"]) for r in region_info]

        # ------------------------------------------------
        # 2) Compute model logprobs
        # ------------------------------------------------
        try:
            out = model.score(sentence)
        except Exception as e:
            print(f"[!] score() failed on sentence: {sentence}")
            print(f"    Error: {e}")
            continue

        # Assume logprobs[i] = -log P(token_{i+1} | prefix)
        # So logprobs has length (num_tokens - 1)
        logprobs = out["logprobs"]

        # ------------------------------------------------
        # 3) Tokenize sentence with offsets
        # ------------------------------------------------
        enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
        
        # Get tokens (compatible with both Fast and standard tokenizers)
        if hasattr(enc, 'tokens'):
            sent_tokens = enc.tokens()
        else:
            sent_tokens = tokenizer.convert_ids_to_tokens(enc.input_ids)
        
        sent_offsets = enc.offset_mapping

        # ------------------------------------------------
        # 4) Character-based region alignment
        # ------------------------------------------------
        region_token_spans = []

        for chunk in region_texts:
            # Find character position in sentence
            start_char = sentence.find(chunk)
            
            if start_char == -1:
                print(f"[Warning] Region '{chunk}' not found in: '{sentence}'")
                region_token_spans.append(None)
                continue
            
            end_char = start_char + len(chunk)
            
            # Map character range to token indices
            start_tok = None
            end_tok = None
            
            for i, (tok_start, tok_end) in enumerate(sent_offsets):
                # Token that starts at or before region start
                if start_tok is None and tok_start <= start_char < tok_end:
                    start_tok = i
                
                # Token that ends at or after region end
                if tok_start < end_char <= tok_end:
                    end_tok = i + 1
                    break
            
            # Handle edge case: region extends to end of sentence
            if start_tok is not None and end_tok is None:
                end_tok = len(sent_tokens)
            
            if start_tok is None or end_tok is None:
                print(f"[Warning] Failed to align region '{chunk}' (chars {start_char}-{end_char})")
                region_token_spans.append(None)
            else:
                region_token_spans.append((start_tok, end_tok))

        # ------------------------------------------------
        # 5) Sum surprisal per region
        # ------------------------------------------------
        cond_S = {r: 0.0 for r in region_numbers}

        for ridx, span in enumerate(region_token_spans):
            if span is None:
                continue

            region_id = region_numbers[ridx]
            start_tok, end_tok = span

            # Accumulate surprisal for tokens in this region
            # logprobs[i] corresponds to the surprisal of token i+1
            # So for token at position t, use logprobs[t-1]
            for t in range(start_tok, end_tok):
                if t == 0:
                    # First token has no preceding context, skip
                    continue
                if t - 1 >= len(logprobs):
                    # Out of bounds check
                    break
                
                # Add negative log probability (surprisal)
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