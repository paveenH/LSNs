#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base analyzer class for LSNs experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from pathlib import Path


class BaseAnalyzer(ABC):
    """Base class for all analyzers in LSNs experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.seed = config.get('seed', 42)
        np.random.seed(self.seed)
    
    @abstractmethod
    def analyze(
        self, 
        positive_activations: np.ndarray, 
        negative_activations: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Analyze the activations and return mask and metadata.
        
        Args:
            positive_activations: Shape (n_samples, n_layers, hidden_dim)
            negative_activations: Shape (n_samples, n_layers, hidden_dim)
            
        Returns:
            mask: Binary mask of shape (n_layers, hidden_dim)
            metadata: Additional analysis results
        """
        pass
    
    @abstractmethod
    def get_analyzer_name(self) -> str:
        """Return the name of this analyzer."""
        pass
    
    def save_results(
        self, 
        mask: np.ndarray, 
        metadata: Dict[str, Any], 
        output_path: Path
    ) -> None:
        """Save analysis results to files."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save mask
        mask_path = output_path.with_suffix('.npy')
        np.save(mask_path, mask)
        
        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_results(self, output_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load analysis results from files."""
        mask_path = output_path.with_suffix('.npy')
        metadata_path = output_path.with_suffix('.json')
        
        mask = np.load(mask_path)
        
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return mask, metadata 