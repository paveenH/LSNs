#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from datasets import load_dataset
from models.factory import ModelFactory
from collections import defaultdict

def load_mask(model, size, pct, pooling, method):
    fname = f"{model}_{size}_{pct}pct_{pooling}_{method}_mask.npy"
    path = os.path.join("cache", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    print(f"Loaded mask: {path}")
    return np.load(path)


from sklearn.metrics import accuracy_score, f1_score

def compute_metric(task, preds, labels):
    if task in ["boolq", "rte", "wic", "wsc"]:
        return accuracy_score(labels, preds)
    elif task == "cb":
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        return (acc + f1) / 2
    elif task in ["copa", "multirc"]:
        return accuracy_score(labels, preds)
    elif task == "record":
        exact_matches = sum(1 for p, l in zip(preds, labels) if p.lower().strip() == l.lower().strip())
        return exact_matches / len(labels) if labels else 0.0
    else:
        return accuracy_score(labels, preds)


def zero_shot_predict(model, task, ex):
    if task == "boolq":
        question = ex["question"]
        passage = ex["passage"]
        prompt = f"Passage: {passage}\nQuestion: {question}"
        return model.classify(prompt)

    elif task == "cb":
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        return model.classify_three_way(premise, hypothesis)

    elif task == "copa":
        premise = ex["premise"]
        question = ex["question"]
        choice1 = ex["choice1"]
        choice2 = ex["choice2"]
        
        if question == "cause":
            prompt = f"Effect: {premise}\nWhat was the cause?"
        else:
            prompt = f"Cause: {premise}\nWhat was the effect?"
        
        options = [choice1, choice2]
        return model.choose_best_option(prompt, options)

    elif task == "multirc":
        paragraph = ex.get("paragraph", ex.get("passage", ""))
        question = ex["question"]
        answer = ex["answer"]
        
        prompt = f"Paragraph: {paragraph}\nQuestion: {question}\nAnswer: {answer}\nIs this answer correct?"
        return model.classify(prompt)

    elif task == "record":
        passage = ex.get("passage", ex.get("text", ""))
        query = ex["query"]
        
        question = query.replace("@placeholder", "_____")
        answer = model.extract_answer(passage, question)
        
        return answer

    elif task == "rte":
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        return model.classify_pair(premise, hypothesis)

    elif task == "wic":
        word = ex["word"]
        sentence1 = ex["sentence1"]
        sentence2 = ex["sentence2"]
        
        prompt = f"""
        Word: "{word}"
        Sentence 1: "{sentence1}"
        Sentence 2: "{sentence2}"
        Does the word have the same meaning in both sentences?
        Answer Yes or No.
        """
        return model.classify(prompt)

    elif task == "wsc":
        text = ex["text"]
        span1_text = ex.get("span1_text", ex.get("option1", ""))
        span2_text = ex.get("span2_text", ex.get("option2", ""))
        
        prompt = f"""
        Text: "{text}"
        Option 1: Use "{span1_text}"
        Option 2: Use "{span2_text}"
        Which option makes the text more grammatically and semantically correct?
        """
        options = [f"Use '{span1_text}'", f"Use '{span2_text}'"]
        return model.choose_best_option(prompt, options)

    else:
        raise NotImplementedError(f"Task not supported: {task}")


def map_label(task, label):
    if task == "boolq":
        return 1 if label else 0
    elif task == "cb":
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        if isinstance(label, str):
            return label_map.get(label.lower(), 1)
        return label
    elif task == "copa":
        return int(label) if not isinstance(label, int) else label  # 0 or 1
    elif task == "multirc":
        return 1 if label else 0
    elif task == "record":
        if isinstance(label, list):
            return label[0] if label else ""
        return str(label) if label else ""
    elif task == "rte":
        label_map = {"entailment": 1, "not_entailment": 0}
        if isinstance(label, str):
            return label_map.get(label.lower(), 0)
        return label
    elif task == "wic":
        return 1 if label else 0
    elif task == "wsc":
        return int(label) if not isinstance(label, int) else label  # 0 or 1
    else:
        return label


def eval_superglue_subset(dataset, task, model):
    preds = []
    labels = []

    for ex in dataset:
        try:
            y_hat = zero_shot_predict(model, task, ex)
            label = ex.get("label", ex.get("labels", None))
            if label is None:
                continue
            y = map_label(task, label)
            
            preds.append(y_hat)
            labels.append(y)
        except Exception as e:
            print(f"  Warning: Error processing example: {e}")
            continue

    if not preds:
        return 0.0, 0
    
    return compute_metric(task, preds, labels), len(labels)


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
    parser.add_argument("--tasks", nargs="+", default=["boolq", "cb", "copa", "rte", "wic"])
    args = parser.parse_args()

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

    model = ModelFactory.create_model(
        model_path,
        config={"device_map": "auto", "torch_dtype": "float16"}
    )
    device = model.device
    model.model.eval()

    print("Loading SuperGLUE subtasks...")
    superglue_data = {}
    for task in args.tasks:
        try:
            superglue_data[task] = load_dataset("super_glue", task)["validation"]
            print(f"  Loaded {task}: {len(superglue_data[task])} examples")
        except Exception as e:
            print(f"  Warning: Failed to load {task}: {e}")
            continue

    if not superglue_data:
        raise ValueError("No SuperGLUE tasks loaded successfully!")

    args.tasks = list(superglue_data.keys())

    results = {"task": args.tasks}

    if args.baseline:
        print("\n=== Baseline Evaluation ===")
        model.set_language_selective_mask(None)

        base_scores = {}
        for task in args.tasks:
            try:
                score, n = eval_superglue_subset(superglue_data[task], task, model)
                base_scores[task] = score
                print(f"{task:10s} Score={score:.3f} (n={n})")
            except Exception as e:
                print(f"{task:10s} Error: {e}")
                base_scores[task] = 0.0

        results["baseline"] = [base_scores[t] for t in args.tasks]

    for method in args.method:
        print(f"\n=== Ablation: {method} ===")

        mask = load_mask(args.model, args.size, args.pct, args.pooling, method)
        mask_tensor = torch.tensor(1.0 - mask, dtype=torch.float16, device=device)
        model.set_language_selective_mask(mask_tensor)

        m_scores = {}
        for task in args.tasks:
            try:
                score, n = eval_superglue_subset(superglue_data[task], task, model)
                m_scores[task] = score
                print(f"{task:10s} Score={score:.3f} (n={n})")
            except Exception as e:
                print(f"{task:10s} Error: {e}")
                m_scores[task] = 0.0

        results[method] = [m_scores[t] for t in args.tasks]

    df = pd.DataFrame(results)
    out = f"cache/superglue_results_{args.model}_{args.size}_{args.pct}.csv"
    df.to_csv(out, index=False)

    print("\n========================================")
    print("Saved results to:", out)
    print("========================================")


if __name__ == "__main__":
    main()

