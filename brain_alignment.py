#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Optional
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from logger import get_logger

class BrainAlignmentMetrics:

    def __init__(self):
        self.logger = get_logger()

    def pearson_correlation(
        self,
        model_activations: np.ndarray,
        brain_activations: np.ndarray
    ) -> float:
        correlations = []

        for model_unit in range(model_activations.shape[1]):
            for brain_voxel in range(brain_activations.shape[1]):
                model_resp = model_activations[:, model_unit]
                brain_resp = brain_activations[:, brain_voxel]

                if np.std(model_resp) == 0 or np.std(brain_resp) == 0:
                    continue

                corr, _ = pearsonr(model_resp, brain_resp)
                if not np.isnan(corr):
                    correlations.append(corr)

        return np.mean(correlations) if correlations else 0.0

    def representational_similarity_analysis(
        self,
        model_activations: np.ndarray,
        brain_activations: np.ndarray,
        metric: str = 'correlation'
    ) -> float:
        model_rdm = self._compute_rdm(model_activations, metric)
        brain_rdm = self._compute_rdm(brain_activations, metric)

        model_rdm_flat = model_rdm[np.triu_indices_from(model_rdm, k=1)]
        brain_rdm_flat = brain_rdm[np.triu_indices_from(brain_rdm, k=1)]

        if len(model_rdm_flat) == 0 or len(brain_rdm_flat) == 0:
            return 0.0

        rsa_corr, _ = spearmanr(model_rdm_flat, brain_rdm_flat)
        return rsa_corr if not np.isnan(rsa_corr) else 0.0

    def _compute_rdm(self, activations: np.ndarray, metric: str) -> np.ndarray:
        if metric == 'correlation':
            rdm = 1 - np.corrcoef(activations)
        elif metric == 'euclidean':
            distances = pdist(activations, metric='euclidean')
            rdm = squareform(distances)
        elif metric == 'cosine':
            similarities = cosine_similarity(activations)
            rdm = 1 - similarities
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return rdm

    def centered_kernel_alignment(
        self,
        model_activations: np.ndarray,
        brain_activations: np.ndarray
    ) -> float:
        model_centered = model_activations - np.mean(model_activations, axis=0)
        brain_centered = brain_activations - np.mean(brain_activations, axis=0)

        model_gram = np.dot(model_centered, model_centered.T)
        brain_gram = np.dot(brain_centered, brain_centered.T)

        n = model_gram.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n

        model_gram_centered = np.dot(H, np.dot(model_gram, H))
        brain_gram_centered = np.dot(H, np.dot(brain_gram, H))

        numerator = np.trace(np.dot(model_gram_centered, brain_gram_centered))
        denominator = np.sqrt(
            np.trace(np.dot(model_gram_centered, model_gram_centered)) *
            np.trace(np.dot(brain_gram_centered, brain_gram_centered))
        )

        return numerator / denominator if denominator != 0 else 0.0

    def compute_all_metrics(
        self,
        model_activations: np.ndarray,
        brain_activations: np.ndarray
    ) -> Dict[str, float]:
        metrics = {}

        try:
            metrics['pearson'] = self.pearson_correlation(
                model_activations, brain_activations
            )
        except Exception as e:
            self.logger.warning(f"Pearson correlation failed: {e}")
            metrics['pearson'] = 0.0

        try:
            metrics['rsa'] = self.representational_similarity_analysis(
                model_activations, brain_activations
            )
        except Exception as e:
            self.logger.warning(f"RSA failed: {e}")
            metrics['rsa'] = 0.0

        try:
            metrics['cka'] = self.centered_kernel_alignment(
                model_activations, brain_activations
            )
        except Exception as e:
            self.logger.warning(f"CKA failed: {e}")
            metrics['cka'] = 0.0

        return metrics

class RealBrainData:

    def __init__(self, brain_data_path: str):
        self.logger = get_logger()

        if not brain_data_path or brain_data_path == "mock":
            raise ValueError(
                "Mock brain data is no longer supported. "
                "Please provide a real brain data path using --brain-data-path argument. "
                "Supported formats: BIDS, HDF5, NPY, or CSV files with fMRI data."
            )

        self.brain_data_path = brain_data_path
        self.brain_activations = self._load_brain_data()

        self.n_stimuli, self.n_voxels = self.brain_activations.shape
        self.logger.info(f"Brain data loaded: {self.n_stimuli} stimuli, {self.n_voxels} voxels")

    def _load_brain_data(self) -> np.ndarray:
        import os

        if not os.path.exists(self.brain_data_path):
            raise FileNotFoundError(f"Brain data file not found: {self.brain_data_path}")

        if self.brain_data_path.endswith('.npy'):
            return np.load(self.brain_data_path)
        elif self.brain_data_path.endswith('.npz'):
            data = np.load(self.brain_data_path)
            if 'brain_activations' in data:
                return data['brain_activations']
            else:
                return data[list(data.keys())[0]]
        elif self.brain_data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(self.brain_data_path)
            return df.values
        elif self.brain_data_path.endswith('.h5') or self.brain_data_path.endswith('.hdf5'):
            import h5py
            with h5py.File(self.brain_data_path, 'r') as f:
                if 'brain_data' in f:
                    return f['brain_data'][:]
                else:
                    key = list(f.keys())[0]
                    return f[key][:]
        else:
            raise ValueError(
                f"Unsupported brain data format: {self.brain_data_path}. "
                "Supported formats: .npy, .npz, .csv, .h5, .hdf5"
            )

    def get_brain_activations(
        self,
        stimuli_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        if stimuli_indices is None:
            return self.brain_activations
        else:
            return self.brain_activations[stimuli_indices]