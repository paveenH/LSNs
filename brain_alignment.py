
import numpy as np
from typing import Dict, List, Tuple, Optional
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

                corr, p_value = pearsonr(model_resp, brain_resp)
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

class MockBrainData:

    def __init__(self, n_stimuli: int = 48, n_voxels: int = 1000, seed: int = 42):
        np.random.seed(seed)
        self.n_stimuli = n_stimuli
        self.n_voxels = n_voxels

        self.brain_activations = self._generate_mock_brain_data()

        self.logger = get_logger()
        self.logger.info(f"Mock brain data created: {n_stimuli} stimuli, {n_voxels} voxels")

    def _generate_mock_brain_data(self) -> np.ndarray:
        brain_data = np.random.normal(0, 1, (self.n_stimuli, self.n_voxels))

        language_signal = np.random.normal(0.5, 0.2, (self.n_stimuli // 2, self.n_voxels))
        brain_data[:self.n_stimuli // 2] += language_signal

        for i in range(0, self.n_voxels, 50):
            end_idx = min(i + 10, self.n_voxels)
            correlation_strength = np.random.uniform(0.3, 0.7)
            base_signal = np.random.normal(0, 1, self.n_stimuli)
            for j in range(i, end_idx):
                if j < self.n_voxels:
                    brain_data[:, j] += correlation_strength * base_signal

        return brain_data

    def get_brain_activations(
        self,
        stimuli_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        if stimuli_indices is None:
            return self.brain_activations
        else:
            return self.brain_activations[stimuli_indices]

def test_brain_alignment_metrics():
    logger = get_logger()

    logger.info("Testing brain alignment metrics...")

    n_stimuli = 48
    n_model_units = 768
    n_brain_voxels = 500

    np.random.seed(42)
    model_activations = np.random.normal(0, 1, (n_stimuli, n_model_units))

    language_boost = np.random.normal(0.3, 0.1, (n_stimuli // 2, n_model_units))
    model_activations[:n_stimuli // 2] += language_boost

    brain_data = MockBrainData(n_stimuli, n_brain_voxels)
    brain_activations = brain_data.get_brain_activations()

    metrics = BrainAlignmentMetrics()

    logger.info("Computing alignment metrics...")
    alignment_scores = metrics.compute_all_metrics(
        model_activations, brain_activations
    )

    logger.info("Brain alignment test results:")
    for metric_name, score in alignment_scores.items():
        logger.info(f"  {metric_name.upper()}: {score:.4f}")

    logger.info("Testing with random data (no signal)...")
    random_model = np.random.normal(0, 1, (n_stimuli, n_model_units))
    random_brain = np.random.normal(0, 1, (n_stimuli, n_brain_voxels))

    random_scores = metrics.compute_all_metrics(random_model, random_brain)

    logger.info("Random data alignment (should be lower):")
    for metric_name, score in random_scores.items():
        logger.info(f"  {metric_name.upper()}: {score:.4f}")

    return alignment_scores, random_scores

if __name__ == "__main__":
    test_brain_alignment_metrics()