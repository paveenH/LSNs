#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma model implementation for LSNs experiments.
"""

from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseModel
from logger import get_logger


class GemmaModel(BaseModel):
    """Gemma model implementation."""
    
    def _load_model(self) -> None:
        """Load Gemma model."""
        logger = get_logger()
        
        # Load model with configuration
        torch_dtype = getattr(torch, self.config.get('torch_dtype', 'float16'))
        device_map = self.config.get('device_map', 'auto')
        trust_remote_code = self.config.get('trust_remote_code', True)
        
        logger.info(f"Loading Gemma model from {self.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code
        )
        
        # Set device
        if device_map == 'auto':
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device_map if device_map else "cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
    
    def _setup_tokenizer(self) -> None:
        """Setup Gemma tokenizer."""
        logger = get_logger()
        
        logger.info(f"Loading tokenizer from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Tokenizer setup completed")
    
    def _get_model_info(self) -> None:
        """Get Gemma model information."""
        logger = get_logger()
        
        # Get number of layers
        self.num_layers = len(self.model.model.layers)
        
        # Get hidden size
        self.hidden_size = self.model.config.hidden_size
        
        # Generate layer names
        self.layer_names = [f"model.layers.{i}" for i in range(self.num_layers)]
        
        logger.info(f"Gemma model info: {self.num_layers} layers, {self.hidden_size} hidden size")
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names for hooking."""
        return self.layer_names 