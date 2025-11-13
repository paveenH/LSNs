#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-accurate T-test analyzer (matching LLM Language Network localize.py).
Uses rank-based is_topk selection instead of standard top-k.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import ttest_ind

from .base import BaseAnalyzer


class TTestPaperAnalyzer(BaseAnalyzer):
    """Paper-correct T-test analyzer using is_topk as in localize.py."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.percentage = config.get("percentage", 1.0)
        self.localize_range = config.get("localize_range", "100-100")

    # ========================================================
    # === replicate paper’s is_topk function ================
    # ========================================================
    def _is_topk(self, a: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Paper's original top-k selection:
        - rank all values
        - pick all units whose rank is within the k highest
        - not necessarily exactly k units!
        """
        flat = a.flatten()
        # reverse sort for top, but preserve ties correctly
        uniq_sorted, rix = np.unique(-flat, return_inverse=True)
        # rix is rank index; smaller rix = larger value
        mask = (rix < k).astype(int)
        return mask.reshape(a.shape)

    # ========================================================
    # === main analysis ======================================
    # ========================================================
    def analyze(
        self,
        positive_activations: np.ndarray,
        negative_activations: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Paper-accurate t-test + is_topk neuron selection.

        Inputs:
            pos/neg activations: shape (N, L, H)

        Returns:
            mask: binary (L, H) matrix
            metadata: (unused for now)
        """
        n_samples, n_layers, hidden_dim = positive_activations.shape
        total_neurons = n_layers * hidden_dim

        # =============================
        # 1. Compute t-values (|act|)
        # =============================
        t_values = np.zeros((n_layers, hidden_dim))
        p_values = np.zeros((n_layers, hidden_dim))

        for i in range(n_layers):
            t_values[i], p_values[i] = ttest_ind(
                np.abs(positive_activations[:, i, :]),
                np.abs(negative_activations[:, i, :]),
                axis=0,
                equal_var=False
            )

        # =============================
        # 2. percentage → how many to pick?
        # =============================
        k = int(self.percentage / 100 * total_neurons)
        if k < 1:
            k = 1

        # =============================
        # 3. Paper's selection logic
        # =============================
        start, end = map(int, self.localize_range.split("-"))

        if start == end == 100:
            # === paper uses is_topk(t_values, k) ***
            mask = self._is_topk(t_values, k=k)

        elif start == end == 0:
            # bottom-k (rarely used)
            flat = t_values.flatten()
            uniq_sorted, rix = np.unique(flat, return_inverse=True)
            mask = (rix < k).astype(int).reshape(t_values.shape)

        else:
            # percentile range (not used for language network)
            low = np.percentile(t_values, start)
            high = np.percentile(t_values, end)
            in_range = ((t_values >= low) & (t_values <= high)).flatten()
            idx = np.where(in_range)[0]
            if len(idx) <= k:
                selected = idx
            else:
                selected = np.random.choice(idx, size=k, replace=False)

            mask = np.zeros(total_neurons, dtype=int)
            mask[selected] = 1
            mask = mask.reshape(t_values.shape)

        # =============================
        # 4. metadata
        # =============================
        metadata = {
            "percentage": self.percentage,
            "localize_range": self.localize_range,
            "selected_neurons": int(mask.sum()),
            "total_neurons": total_neurons,
            "selection_ratio": mask.sum() / total_neurons
        }

        return mask, metadata

    def get_analyzer_name(self) -> str:
        return "ttest_paper"