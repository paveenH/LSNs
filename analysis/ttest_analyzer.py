#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-test analyzer for LSNs experiments.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import ttest_ind, false_discovery_control

from .base import BaseAnalyzer


class TTestAnalyzer(BaseAnalyzer):
    """T-test based analyzer for selecting neurons."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.percentage = config.get('percentage', 5.0)
        self.localize_range = config.get('localize_range', '100-100')
        self.fdr_correction = config.get('fdr_correction', True)
    
    def analyze(
        self, 
        positive_activations: np.ndarray, 
        negative_activations: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform t-test analysis to select neurons.
        
        Args:
            positive_activations: Shape (n_samples, n_layers, hidden_dim)
            negative_activations: Shape (n_samples, n_layers, hidden_dim)
            
        Returns:
            mask: Binary mask of shape (n_layers, hidden_dim)
            metadata: Analysis results including t-values, p-values, etc.
        """
        n_samples, n_layers, hidden_dim = positive_activations.shape
        total_neurons = n_layers * hidden_dim
        
        # Calculate t-values and p-values
        t_values = np.zeros((n_layers, hidden_dim))
        p_values = np.zeros((n_layers, hidden_dim))
        
        for i in range(n_layers):
            t_values[i], p_values[i] = ttest_ind(
                np.abs(positive_activations[:, i, :]), 
                np.abs(negative_activations[:, i, :]),
                axis=0, 
                equal_var=False
            )
        
        # Apply FDR correction if requested
        if self.fdr_correction:
            adjusted_p_values = false_discovery_control(p_values.flatten()).reshape(p_values.shape)
        else:
            adjusted_p_values = p_values
        
        # Generate mask based on selection criteria
        mask = self._generate_mask(t_values, total_neurons)
        
        # Prepare metadata
        metadata = {
            'analyzer': self.get_analyzer_name(),
            't_values': t_values.tolist(),
            'p_values': p_values.tolist(),
            'adjusted_p_values': adjusted_p_values.tolist(),
            'percentage': self.percentage,
            'localize_range': self.localize_range,
            'fdr_correction': self.fdr_correction,
            'total_neurons': total_neurons,
            'selected_neurons': int(mask.sum()),
            'selection_ratio': float(mask.sum() / total_neurons)
        }
        
        return mask, metadata
    
    def _generate_mask(self, t_values: np.ndarray, total_neurons: int) -> np.ndarray:
        """Generate binary mask based on t-values and selection criteria."""
        n_layers, hidden_dim = t_values.shape
        k = int((self.percentage / 100) * total_neurons)
        
        start, end = map(int, self.localize_range.split('-'))
        
        if start == end == 100:
            # Select top-k neurons
            mask = self._select_topk(t_values, k)
        elif start == end == 0:
            # Select bottom-k neurons
            mask = self._select_bottomk(t_values, k)
        else:
            # Select neurons within percentile range
            mask = self._select_percentile_range(t_values, k, start, end)
        
        return mask
    
    def _select_topk(self, values: np.ndarray, k: int) -> np.ndarray:
        """Select top-k values."""
        flat_values = values.flatten()
        # Get indices of top-k values
        top_indices = np.argpartition(-flat_values, k)[:k]
        
        # Create mask
        mask = np.zeros_like(flat_values, dtype=int)
        mask[top_indices] = 1
        return mask.reshape(values.shape)
    
    def _select_bottomk(self, values: np.ndarray, k: int) -> np.ndarray:
        """Select bottom-k values."""
        flat_values = values.flatten()
        # Get indices of bottom-k values
        bottom_indices = np.argpartition(flat_values, k)[:k]
        
        # Create mask
        mask = np.zeros_like(flat_values, dtype=int)
        mask[bottom_indices] = 1
        return mask.reshape(values.shape)
    
    def _select_percentile_range(self, values: np.ndarray, k: int, start: int, end: int) -> np.ndarray:
        """Select neurons within percentile range."""
        n_layers, hidden_dim = values.shape
        
        # Calculate percentile bounds
        low = np.percentile(values, start)
        high = np.percentile(values, end)
        
        # Find neurons within range
        in_range = ((values >= low) & (values <= high)).flatten()
        candidate_indices = np.where(in_range)[0]
        
        # Randomly select k neurons from candidates
        if len(candidate_indices) >= k:
            selected_indices = np.random.choice(candidate_indices, size=k, replace=False)
        else:
            # If not enough candidates, select all and fill with random
            selected_indices = candidate_indices
            remaining = k - len(candidate_indices)
            if remaining > 0:
                all_indices = np.arange(n_layers * hidden_dim)
                available = np.setdiff1d(all_indices, candidate_indices)
                additional = np.random.choice(available, size=remaining, replace=False)
                selected_indices = np.concatenate([selected_indices, additional])
        
        # Create mask
        mask = np.zeros(n_layers * hidden_dim, dtype=int)
        mask[selected_indices] = 1
        return mask.reshape((n_layers, hidden_dim))
    
    def get_analyzer_name(self) -> str:
        """Return the name of this analyzer."""
        return "ttest" 