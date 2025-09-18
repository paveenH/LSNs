
from typing import List, Dict, Optional, Tuple
import os
import torch
import argparse
import numpy as np
from pathlib import Path

from tqdm import tqdm

from models.factory import ModelFactory
from datasets import LangLocDataset, TOMLocDataset, MDLocDataset
from analysis.ttest_analyzer import TTestAnalyzer
from analysis.nmd_analyzer import NMDAnalyzer
from analysis.mean_analyzer import MeanAnalyzer
from brain_alignment import BrainAlignmentMetrics, RealBrainData
from logger import get_logger

CACHE_DIR = os.environ.get("LOC_CACHE", "cache")

def extract_step_wise_brain_aligned_representations(
    network: str,
    pooling: str,
    model,
    layer_names: List[str],
    hidden_dim: int,
    device: torch.device,
    num_denoising_steps: int = 10,
    brain_data_path: str = None
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[Dict[str, float]]]:
    logger = get_logger()

    if network == "language":
        loc_dataset = LangLocDataset()
    elif network == "theory-of-mind":
        loc_dataset = TOMLocDataset()
    elif network == "multiple-demand":
        loc_dataset = MDLocDataset()
    else:
        raise ValueError(f"Unsupported network: {network}")

    if not brain_data_path:
        raise ValueError(
            "Brain data path is required. Mock data is no longer supported. "
            "Please provide --brain-data-path argument pointing to real fMRI data. "
            "Supported formats: .npy, .npz, .csv, .h5, .hdf5"
        )

    logger.info(f"Loading real brain data from {brain_data_path}...")
    brain_data = RealBrainData(brain_data_path)

    brain_metrics = BrainAlignmentMetrics()

    is_diffusion_model = hasattr(model, 'extract_diffusion_activations')
    logger.info(f"Model type: {'Diffusion' if is_diffusion_model else 'Standard'}")

    if is_diffusion_model:
        step_wise_data = _extract_diffusion_step_wise(
            model, loc_dataset, layer_names, hidden_dim,
            pooling, num_denoising_steps, logger
        )
    else:
        step_wise_data = _extract_standard_with_simulation(
            model, loc_dataset, layer_names, hidden_dim,
            pooling, num_denoising_steps, logger
        )

    logger.info("Computing brain alignment scores across diffusion steps...")
    brain_scores = []

    middle_layer_idx = len(layer_names) // 2
    target_layer = layer_names[middle_layer_idx]
    logger.info(f"Using layer {target_layer} for brain alignment")

    for step in range(num_denoising_steps):
        logger.info(f"Computing brain alignment for step {step + 1}/{num_denoising_steps}")

        positive_acts = step_wise_data["positive"][target_layer][step]
        negative_acts = step_wise_data["negative"][target_layer][step]

        all_model_acts = np.vstack([positive_acts, negative_acts])

        brain_acts = brain_data.get_brain_activations()

        alignment_scores = brain_metrics.compute_all_metrics(
            all_model_acts, brain_acts
        )

        brain_scores.append(alignment_scores)

        logger.debug(f"Step {step} brain scores: {alignment_scores}")

    return step_wise_data, brain_scores

def _extract_diffusion_step_wise(
    model,
    dataset,
    layer_names: List[str],
    hidden_dim: int,
    pooling: str,
    num_steps: int,
    logger
) -> Dict[str, Dict[str, np.ndarray]]:
    logger.info(f"Extracting true diffusion activations for {num_steps} steps")

    logger.info("Processing positive samples...")
    positive_texts = [str(text) for text in dataset.positive]
    positive_activations = model.extract_diffusion_activations(
        positive_texts,
        layer_names=layer_names,
        pooling=pooling,
        num_steps=num_steps
    )

    logger.info("Processing negative samples...")
    negative_texts = [str(text) for text in dataset.negative]
    negative_activations = model.extract_diffusion_activations(
        negative_texts,
        layer_names=layer_names,
        pooling=pooling,
        num_steps=num_steps
    )

    return {
        "positive": positive_activations,
        "negative": negative_activations
    }

def _extract_standard_with_simulation(
    model,
    dataset,
    layer_names: List[str],
    hidden_dim: int,
    pooling: str,
    num_steps: int,
    logger
) -> Dict[str, Dict[str, np.ndarray]]:
    logger.info(f"Simulating {num_steps} diffusion steps with standard model")

    step_wise_positive = {
        layer_name: np.zeros((num_steps, len(dataset.positive), hidden_dim))
        for layer_name in layer_names
    }
    step_wise_negative = {
        layer_name: np.zeros((num_steps, len(dataset.negative), hidden_dim))
        for layer_name in layer_names
    }

    for step in range(num_steps):
        logger.info(f"Simulating step {step + 1}/{num_steps}")

        noise_scale = (num_steps - step) / num_steps * 0.1

        positive_texts = [str(text) for text in dataset.positive]
        negative_texts = [str(text) for text in dataset.negative]

        pos_activations = model.extract_activations(
            positive_texts, layer_names, pooling
        )
        neg_activations = model.extract_activations(
            negative_texts, layer_names, pooling
        )

        for layer_name in layer_names:
            pos_acts = pos_activations[layer_name]
            if noise_scale > 0:
                noise = np.random.normal(0, noise_scale, pos_acts.shape)
                pos_acts = pos_acts + noise
            step_wise_positive[layer_name][step] = pos_acts

            neg_acts = neg_activations[layer_name]
            if noise_scale > 0:
                noise = np.random.normal(0, noise_scale, neg_acts.shape)
                neg_acts = neg_acts + noise
            step_wise_negative[layer_name][step] = neg_acts

    return {
        "positive": step_wise_positive,
        "negative": step_wise_negative
    }

def analyze_brain_alignment_trajectory(
    brain_scores: List[Dict[str, float]],
    num_steps: int
) -> Dict[str, any]:
    logger = get_logger()
    logger.info("Analyzing brain alignment trajectory...")

    trajectories = {}
    metrics = brain_scores[0].keys()

    for metric in metrics:
        trajectories[metric] = [scores[metric] for scores in brain_scores]

    optimal_steps = {}
    for metric in metrics:
        scores = trajectories[metric]
        optimal_step = np.argmax(scores)
        optimal_steps[metric] = {
            'step': optimal_step,
            'score': scores[optimal_step],
            'improvement': scores[optimal_step] - scores[0] if scores[0] != 0 else 0
        }

    trajectory_stats = {}
    for metric in metrics:
        scores = trajectories[metric]
        trajectory_stats[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'range': np.max(scores) - np.min(scores),
            'trend': 'increasing' if scores[-1] > scores[0] else 'decreasing'
        }

    logger.info("Brain alignment trajectory analysis:")
    for metric in metrics:
        opt = optimal_steps[metric]
        stats = trajectory_stats[metric]
        logger.info(f"  {metric.upper()}:")
        logger.info(f"    Best step: {opt['step']} (score: {opt['score']:.4f})")
        logger.info(f"    Improvement: {opt['improvement']:.4f}")
        logger.info(f"    Range: {stats['min']:.4f} - {stats['max']:.4f}")
        logger.info(f"    Trend: {stats['trend']}")

    return {
        'trajectories': trajectories,
        'optimal_steps': optimal_steps,
        'trajectory_stats': trajectory_stats
    }

def save_brain_aligned_results(
    step_wise_data: Dict[str, Dict[str, np.ndarray]],
    brain_scores: List[Dict[str, float]],
    trajectory_analysis: Dict,
    model_name: str,
    network: str,
    output_dir: str = "results"
):
    logger = get_logger()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    brain_scores_file = output_path / f"{model_name}_{network}_brain_alignment_trajectory.npy"

    metrics = list(brain_scores[0].keys())
    trajectory_array = np.array([[scores[metric] for metric in metrics] for scores in brain_scores])
    np.save(brain_scores_file, trajectory_array)

    analysis_file = output_path / f"{model_name}_{network}_trajectory_analysis.json"
    import json

    json_analysis = {}
    for key, value in trajectory_analysis.items():
        if isinstance(value, dict):
            json_analysis[key] = {
                k: {kk: float(vv) if isinstance(vv, np.number) else vv for kk, vv in v.items()}
                if isinstance(v, dict) else [float(x) for x in v] if isinstance(v, list) else v
                for k, v in value.items()
            }
        else:
            json_analysis[key] = value

    with open(analysis_file, 'w') as f:
        json.dump(json_analysis, f, indent=2)

    for metric in metrics:
        best_step = trajectory_analysis['optimal_steps'][metric]['step']

        logger.info(f"Generating LSN mask for best {metric} step ({best_step})...")

        analyzer_config = {'percentage': 5.0, 'localize_range': '100-100', 'alpha': 0.05}
        analyzer = TTestAnalyzer(analyzer_config)

        positive_step = []
        negative_step = []
        layer_names = list(step_wise_data["positive"].keys())

        for layer_name in layer_names:
            pos_step = step_wise_data["positive"][layer_name][best_step]
            neg_step = step_wise_data["negative"][layer_name][best_step]
            positive_step.append(pos_step)
            negative_step.append(neg_step)

        positive_step = np.stack(positive_step, axis=1)
        negative_step = np.stack(negative_step, axis=1)

        mask, metadata = analyzer.analyze(positive_step, negative_step)

        mask_file = output_path / f"{model_name}_{network}_best_{metric}_step_mask.npy"
        np.save(mask_file, mask)

    logger.info(f"Brain-aligned results saved to {output_dir}/")
    logger.info(f"  Trajectories: {brain_scores_file}")
    logger.info(f"  Analysis: {analysis_file}")
    logger.info(f"  Best step masks: {output_dir}/{model_name}_{network}_best_*_step_mask.npy")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--network", type=str, default="language",
                        choices=["language", "theory-of-mind", "multiple-demand"])
    parser.add_argument("--pooling", type=str, default="last",
                        choices=["last", "mean", "sum"])
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")

    # Brain data arguments
    parser.add_argument("--brain-data-path", type=str, required=True)

    args = parser.parse_args()

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("BRAIN-ALIGNED STEP-WISE DIFFUSION LSN LOCALIZATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Network: {args.network}")
    logger.info(f"Steps: {args.num_steps}")

    logger.info("Loading model...")
    config = {
        'device_map': args.device,
        'torch_dtype': 'float32',
        'trust_remote_code': True,
        'num_denoising_steps': args.num_steps
    }

    model = ModelFactory.create_model(args.model_name, config)
    logger.info(f"Model loaded: {model}")

    logger.info("Extracting step-wise representations with brain alignment...")
    logger.info(f"Using real brain data from: {args.brain_data_path}")

    step_wise_data, brain_scores = extract_step_wise_brain_aligned_representations(
        network=args.network,
        pooling=args.pooling,
        model=model,
        layer_names=model.get_layer_names(),
        hidden_dim=model.hidden_size,
        device=torch.device(args.device),
        num_denoising_steps=args.num_steps,
        brain_data_path=args.brain_data_path
    )

    logger.info("Analyzing brain alignment trajectory...")
    trajectory_analysis = analyze_brain_alignment_trajectory(
        brain_scores=brain_scores,
        num_steps=args.num_steps
    )

    logger.info("Saving brain-aligned results...")
    model_name_clean = args.model_name.replace("/", "_").replace("-", "_")
    save_brain_aligned_results(
        step_wise_data=step_wise_data,
        brain_scores=brain_scores,
        trajectory_analysis=trajectory_analysis,
        model_name=model_name_clean,
        network=args.network,
        output_dir=args.output_dir
    )

    logger.info("=" * 80)
    logger.info("BRAIN-ALIGNED STEP-WISE ANALYSIS COMPLETE")
    logger.info("=" * 80)

    best_pearson_step = trajectory_analysis['optimal_steps']['pearson']['step']
    best_pearson_score = trajectory_analysis['optimal_steps']['pearson']['score']
    best_rsa_step = trajectory_analysis['optimal_steps']['rsa']['step']
    best_rsa_score = trajectory_analysis['optimal_steps']['rsa']['score']

    logger.info(f"BRAIN ALIGNMENT FINDINGS:")
    logger.info(f"  Best Pearson alignment: Step {best_pearson_step} (score: {best_pearson_score:.4f})")
    logger.info(f"  Best RSA alignment: Step {best_rsa_step} (score: {best_rsa_score:.4f})")
    logger.info(f"  Results saved to: {args.output_dir}/")

    if args.brain_data_path:
        logger.info("Analysis completed with real brain data!")
        logger.info("Results are scientifically meaningful for neuroscience research.")
    else:
        logger.warning("Analysis completed with MOCK brain data!")
        logger.warning("For actual neuroscience research, provide real brain data with --brain-data-path")

    logger.info("Ready for diffusion model brain alignment analysis!")

if __name__ == "__main__":
    main()