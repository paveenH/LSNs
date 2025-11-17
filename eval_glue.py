#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a LLaMA-style model (with optional neuron-mask ablation)
on GLUE benchmark tasks.

This script mirrors the SyntaxGym evaluation structure:
- supports baseline and multiple ablation methods
- supports ModelFactory with set_language_selective_mask
- loads masks from cache, format: MODEL_SIZE_PCT_pooling_method_mask.npy

This version is inference-only (no fine-tuning).
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from models.factory import ModelFactory

# ======================================================
# Mask loading
# ======================================================
def load_mask(model, size, pct, pooling, method):
    fname = f"{model}_{size}_{pct}pct_{pooling}_{method}_mask.npy"
    path = os.path.join("cache", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found:\n{path}")
    print(f"Loaded mask: {path}")
    return np.load(path)

# ======================================================
# Simple (inference only) GLUE predictors
# Each returns: (prediction_label, gold_label)
# ======================================================

def predict_single_sentence(model, tokenizer, text):
    """
    For SST-2, CoLA (sentence-level classification)
    """
    out = model.classify(text)
    # expected: {"probs": [...], "label": int}
    return out["label"]

def predict_sentence_pair(model, tokenizer, text1, text2):
    """
    For MRPC, QQP, QNLI, RTE, WNLI
    """
    out = model.classify_pair(text1, text2)
    return out["label"]

def predict_stsb(model, tokenizer, text1, text2):
    """
    Regression task (0-5).
    If your model does not support regression directly,
    you may approximate by softmax over classes or use model.score on concatenated pairs.
    For now: use model.score(text1 + ' ' + text2) as placeholder.
    """
    out = model.score_pair(text1, text2)
    return float(out["score"])

# ======================================================
# Metric computation
# ======================================================
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

def compute_metric(task_name, preds, labels):
    if task_name == "cola":
        return matthews_corrcoef(labels, preds)
    if task_name == "stsb":
        pear = pearsonr(labels, preds)[0]
        spear = spearmanr(labels, preds)[0]
        return (pear + spear) / 2
    if task_name in ["mrpc", "qqp"]:
        return f1_score(labels, preds)
    return accuracy_score(labels, preds)

# ======================================================
# Evaluate one GLUE subset (train / validation / test)
# We evaluate on validation set by default
# ======================================================
def eval_glue_subset(dataset, task_name, model, tokenizer):
    preds = []
    golds = []

    for ex in dataset:
        if task_name in ["sst2", "cola"]:
            y_hat = predict_single_sentence(model, tokenizer, ex["sentence"])
            y = ex["label"]
        elif task_name == "stsb":
            y_hat = predict_stsb(model, tokenizer, ex["sentence1"], ex["sentence2"])
            y = ex["label"]
        else:
            y_hat = predict_sentence_pair(model, tokenizer, ex["sentence1"], ex["sentence2"])
            y = ex["label"]

        preds.append(y_hat)
        golds.append(y)

    return compute_metric(task_name, preds, golds), len(golds)

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

    # --------------------------------------------------
    # Model path mapping
    # --------------------------------------------------
    model_map = {
        ("llama3", "3B"): "meta-llama/Llama-3.2-3B-Instruct",
        ("llama3", "8B"): "meta-llama/Llama-3.1-8B-Instruct",
    }
    model_path = model_map[(args.model, args.size)]

    print("========================================")
    print(f"Model: {model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Methods: {args.method}")
    print(f"Mask pct: {args.pct} | pooling={args.pooling}")
    print(f"Evaluate baseline: {args.baseline}")
    print("========================================")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = ModelFactory.create_model(model_path, config={
        "device_map": "auto",
        "torch_dtype": "float16",
    })
    tokenizer = model.tokenizer
    device = model.device
    model.model.eval()

    # --------------------------------------------------
    # Load GLUE datasets
    # --------------------------------------------------
    glue_data = {}
    print("Loading GLUE subtasks...")
    for task in args.tasks:
        glue_data[task] = load_dataset("glue", task)["validation"]

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    results = {"task": args.tasks}

    # --------------------------------------------------
    # 1) Baseline
    # --------------------------------------------------
    if args.baseline:
        print("\n=== Baseline Evaluation ===")
        model.set_language_selective_mask(None)

        base_scores = {}
        for task in args.tasks:
            score, n = eval_glue_subset(glue_data[task], task, model, tokenizer)
            base_scores[task] = score
            print(f"{task:10s} Score={score:.3f} (n={n})")

        results["baseline"] = [base_scores[t] for t in args.tasks]

    # --------------------------------------------------
    # 2) Evaluate each ablation method
    # --------------------------------------------------
    for method in args.method:
        print(f"\n=== Ablation: {method} ===")

        mask = load_mask(args.model, args.size, args.pct, args.pooling, method)
        mask_tensor = torch.tensor(1.0 - mask, dtype=torch.float16, device=device)
        model.set_language_selective_mask(mask_tensor)

        m_scores = {}
        for task in args.tasks:
            score, n = eval_glue_subset(glue_data[task], task, model, tokenizer)
            m_scores[task] = score
            print(f"{task:10s} Score={score:.3f} (n={n})")

        results[method] = [m_scores[t] for t in args.tasks]

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    df = pd.DataFrame(results)
    out_name = f"cache/glue_results_{args.model}_{args.size}_{args.pct}.csv"
    df.to_csv(out_name, index=False)

    print("\n========================================")
    print("Saved results to:", out_name)
    print("========================================")


if __name__ == "__main__":
    main()