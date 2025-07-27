#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean analyzer for computing average activations in LSNs experiments.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base import BaseAnalyzer


class MeanAnalyzer(BaseAnalyzer):
    """Mean analyzer for computing average activations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def analyze(
        self, 
        positive_activations: np.ndarray, 
        negative_activations: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute mean activations for positive and negative stimuli.
        
        Args:
            positive_activations: Shape (n_samples, n_layers, hidden_dim)
            negative_activations: Shape (n_samples, n_layers, hidden_dim)
            
        Returns:
            mask: Dummy mask (all zeros) since this is just for computing means
            metadata: Mean activations and statistics
        """
        n_samples, n_layers, hidden_dim = positive_activations.shape
        
        # Calculate mean activations
        pos_mean = positive_activations.mean(axis=0)  # (n_layers, hidden_dim)
        neg_mean = negative_activations.mean(axis=0)  # (n_layers, hidden_dim)
        
        # Calculate standard deviations
        pos_std = positive_activations.std(axis=0)  # (n_layers, hidden_dim)
        neg_std = negative_activations.std(axis=0)  # (n_layers, hidden_dim)
        
        # Create dummy mask (all zeros since this analyzer doesn't select neurons)
        mask = np.zeros((n_layers, hidden_dim), dtype=int)
        
        # Prepare metadata
        metadata = {
            'analyzer': self.get_analyzer_name(),
            'positive_mean': pos_mean.tolist(),
            'negative_mean': neg_mean.tolist(),
            'positive_std': pos_std.tolist(),
            'negative_std': neg_std.tolist(),
            'n_samples': n_samples,
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'total_neurons': n_layers * hidden_dim
        }
        
        return mask, metadata
    
    def get_analyzer_name(self) -> str:
        """Return the name of this analyzer."""
        return "mean" 