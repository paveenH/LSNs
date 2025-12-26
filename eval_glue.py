#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot GLUE evaluation for LLaMA-style models
with optional neuron ablation masks.

This version:
- Uses BaseModel.classify() and classify_pair()
- Fully compatible with SyntaxGym mask pipeline
- Pure inference-only zero-shot evaluation based on logprob scoring
- Same coding style as your SyntaxGym eval script
"""

import os
if 'HF_HUB_ENABLE_HF_TRANSFER' not in os.environ:
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import numpy as np
import torch
from datasets import load_dataset
from models.factory import ModelFactory

# ======================================================
# Mask loading
# ======================================================
def load_mask(model, size, pct, pooling, method):
    import glob
    
    fname1 = f"{model}_{size}_{pct}pct_{pooling}_{method}_mask.npy"
    path1 = os.path.join("cache", fname1)
    
    pooling_localize = "last-token" if pooling == "last" else pooling
    model_id = f"Llama-3.1-8B-Instruct" if (model, size) == ("llama3", "8B") else f"{model}_{size}"
    fname2_pattern = f"{model_id}_network=language_pooling={pooling_localize}_range=100-100_perc={pct}_nunits=*_pretrained=True.npy"
    path2_pattern = os.path.join("cache", fname2_pattern)
    
    if os.path.exists(path1):
        print(f"Loaded mask: {path1}")
        return np.load(path1)
    
    matches = glob.glob(path2_pattern)
    if matches:
        path2 = matches[0]
        print(f"Loaded mask: {path2}")
        return np.load(path2)
    
    all_masks = glob.glob(os.path.join("cache", "*language*.npy"))
    if all_masks:
        print(f"Warning: Could not find exact mask match. Found {len(all_masks)} language masks.")
        print(f"Available masks: {[os.path.basename(m) for m in all_masks[:5]]}")
        print(f"Trying to use: {all_masks[0]}")
        return np.load(all_masks[0])
    
    raise FileNotFoundError(
        f"Mask not found. Tried:\n"
        f"  - {path1}\n"
        f"  - Pattern: {fname2_pattern}\n"
        f"Run 'localize.py' first to create masks, or use --baseline flag for baseline evaluation only."
    )


# ======================================================
# Metric computation
# ======================================================
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

def compute_metric(task, preds, labels):
    if task == "cola":
        return matthews_corrcoef(labels, preds)
    if task == "stsb":
        pear = pearsonr(labels, preds)[0]
        spear = spearmanr(labels, preds)[0]
        return (pear + spear) / 2
    if task in ["mrpc", "qqp"]:
        return f1_score(labels, preds)
    return accuracy_score(labels, preds)


# ======================================================
# Task-specific zero-shot prediction logic
# ======================================================
def zero_shot_predict(model, task, ex):
    """
    Returns predicted label (int)
    GLUE format:
        - label is 0/1 for most tasks
        - STSB uses a real-valued similarity score
    """

    if task in ["sst2", "cola"]:
        return model.classify(ex["sentence"])

    elif task == "mrpc":
        return model.classify_pair(ex["sentence1"], ex["sentence2"])

    elif task == "qqp":
        return model.classify_pair(ex["question1"], ex["question2"])

    elif task == "qnli":
        # GLUE uses fields "question" & "sentence"
        return model.classify_pair(ex["question"], ex["sentence"])

    elif task == "rte":
        return model.classify_pair(ex["sentence1"], ex["sentence2"])

    elif task == "stsb":
        # regression: 0~5
        out = model.score(ex["sentence1"] + " " + ex["sentence2"])
        return out["mean_logprob"]   # usable continuous regression score

    else:
        raise NotImplementedError(f"Task not supported: {task}")


# ======================================================
# Evaluate GLUE validation split
# ======================================================
def eval_glue_subset(dataset, task, model):
    preds = []
    labels = []

    for ex in dataset:
        y_hat = zero_shot_predict(model, task, ex)
        y = ex["label"]

        preds.append(y_hat)
        labels.append(y)

    return compute_metric(task, preds, labels), len(labels)


# ======================================================
# Main
# ======================================================
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
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc", "qnli", "rte"])
    args = parser.parse_args()

    # ----------------------------
    # Model path mapping
    # ----------------------------
    model_map = {
        ("llama3", "3B"): "meta-llama/Llama-3.2-3B-Instruct",
        ("llama3", "8B"): "meta-llama/Llama-3.1-8B-Instruct",
    }
    model_path = model_map[(args.model, args.size)]

    print("========================================")
    print(f"Model: {model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Methods: {args.method}")
    print(f"Mask pct={args.pct}, pooling={args.pooling}")
    print(f"Baseline={args.baseline}")
    print("========================================")

    # ----------------------------
    # Load model
    # ----------------------------
    model = ModelFactory.create_model(
        model_path,
        config={"device_map": "auto", "torch_dtype": "float16"}
    )
    device = model.device
    model.model.eval()

    # ----------------------------
    # Load GLUE datasets
    # ----------------------------
    print("Loading GLUE subtasks...")
    glue_data = {}
    for task in args.tasks:
        glue_data[task] = load_dataset("glue", task)["validation"]

    # ----------------------------
    # Results container
    # ----------------------------
    results = {"task": args.tasks}

    # ----------------------------
    # 1. Baseline
    # ----------------------------
    if args.baseline:
        print("\n=== Baseline Evaluation ===")
        model.set_language_selective_mask(None)

        base_scores = {}
        for task in args.tasks:
            score, n = eval_glue_subset(glue_data[task], task, model)
            base_scores[task] = score
            print(f"{task:10s} Score={score:.3f} (n={n})")

        results["baseline"] = [base_scores[t] for t in args.tasks]

    # ----------------------------
    # 2. Mask ablation
    # ----------------------------
    for method in args.method:
        print(f"\n=== Ablation: {method} ===")

        try:
            mask = load_mask(args.model, args.size, args.pct, args.pooling, method)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("\nSkipping ablation evaluation for this method.")
            continue
            
        mask_tensor = torch.tensor(1.0 - mask, dtype=torch.float16, device=device)
        model.set_language_selective_mask(mask_tensor)

        m_scores = {}
        for task in args.tasks:
            score, n = eval_glue_subset(glue_data[task], task, model)
            m_scores[task] = score
            print(f"{task:10s} Score={score:.3f} (n={n})")

        results[method] = [m_scores[t] for t in args.tasks]

    # ----------------------------
    # Save CSV
    # ----------------------------
    df = pd.DataFrame(results)
    out = f"cache/glue_results_{args.model}_{args.size}_{args.pct}.csv"
    df.to_csv(out, index=False)

    print("\n========================================")
    print("Saved results to:", out)
    print("========================================")


if __name__ == "__main__":
    main()