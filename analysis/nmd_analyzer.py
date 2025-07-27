#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuron-wise Mean Difference (NMD) analyzer for LSNs experiments.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base import BaseAnalyzer


class NMDAnalyzer(BaseAnalyzer):
    """Neuron-wise Mean Difference based analyzer for selecting neurons."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # NMD uses topk_ratio (per-layer) not percentage (global)
        self.topk_ratio = config.get('topk_ratio', 0.005)  # Default: top 0.5% per layer
        # For compatibility, convert percentage to topk_ratio if provided
        if 'percentage' in config:
            # Convert global percentage to approximate per-layer ratio
            self.topk_ratio = config['percentage'] / 100
    
    def analyze(
        self, 
        positive_activations: np.ndarray, 
        negative_activations: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform NMD analysis to select neurons.
        
        Args:
            positive_activations: Shape (n_samples, n_layers, hidden_dim)
            negative_activations: Shape (n_samples, n_layers, hidden_dim)
            
        Returns:
            mask: Binary mask of shape (n_layers, hidden_dim)
            metadata: Analysis results including differences, etc.
        """
        n_samples, n_layers, hidden_dim = positive_activations.shape
        
        # Calculate mean activations
        pos_mean = positive_activations.mean(axis=0)  # (n_layers, hidden_dim)
        neg_mean = negative_activations.mean(axis=0)  # (n_layers, hidden_dim)
        
        # Calculate differences
        differences = pos_mean - neg_mean
        
        # Generate mask based on differences (per-layer selection)
        mask = self._generate_mask(differences, hidden_dim)
        
        # Prepare metadata
        metadata = {
            'analyzer': self.get_analyzer_name(),
            'positive_mean': pos_mean.tolist(),
            'negative_mean': neg_mean.tolist(),
            'differences': differences.tolist(),
            'topk_ratio': self.topk_ratio,
            'total_neurons': n_layers * hidden_dim,
            'selected_neurons': int(mask.sum()),
            'selection_ratio': float(mask.sum() / (n_layers * hidden_dim))
        }
        
        return mask, metadata
    
    def _generate_mask(self, differences: np.ndarray, hidden_dim: int) -> np.ndarray:
        """Generate binary mask based on absolute differences (per-layer selection)."""
        n_layers, _ = differences.shape
        topk = int(hidden_dim * self.topk_ratio)
        
        mask = np.zeros((n_layers, hidden_dim), dtype=int)
        
        for i in range(n_layers):
            # Select top-k neurons with largest absolute differences IN THIS LAYER
            indices = np.argsort(np.abs(differences[i]))[-topk:]
            mask[i, indices] = 1
        
        return mask
    
    def get_analyzer_name(self) -> str:
        """Return the name of this analyzer."""
        return "nmd" 