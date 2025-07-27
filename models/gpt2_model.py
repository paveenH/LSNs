#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT2 model implementation for LSNs experiments.
"""

from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from .base import BaseModel
from logger import get_logger


class PaperCorrectMaskedGPT2(GPT2LMHeadModel):
    """GPT2 model with language selective masking exactly as described in the paper."""
    
    def __init__(self, config):
        super().__init__(config)
        self.language_selective_mask = None
        self.original_forwards = {}
        self.hooks_registered = False
    
    def set_language_selective_mask(self, mask):
        """Set the language selective mask.
        
        Args:
            mask: Tensor of shape (num_layers, hidden_dim) where 0 = ablate, 1 = keep
        """
        self.language_selective_mask = mask
        if mask is not None:
            self._register_ablation_hooks()
        else:
            self._remove_ablation_hooks()
    
    def _register_ablation_hooks(self):
        """Register hooks on transformer blocks to apply ablation at each layer output."""
        if self.hooks_registered:
            return
            
        def create_ablation_hook(layer_idx):
            def ablation_hook(module, input, output):
                if self.language_selective_mask is not None:
                    # output is either a tensor or tuple with hidden states as first element
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    # Apply layer-specific mask: mask shape (hidden_dim,), hidden_states shape (batch, seq, hidden)
                    layer_mask = self.language_selective_mask[layer_idx]  # (hidden_dim,)
                    masked_hidden = hidden_states * layer_mask.unsqueeze(0).unsqueeze(0)
                    
                    if isinstance(output, tuple):
                        return (masked_hidden,) + output[1:]
                    else:
                        return masked_hidden
                return output
            return ablation_hook
        
        # Register hooks on each transformer block
        self.ablation_hooks = []
        for i, block in enumerate(self.transformer.h):
            hook = block.register_forward_hook(create_ablation_hook(i))
            self.ablation_hooks.append(hook)
        
        self.hooks_registered = True
    
    def _remove_ablation_hooks(self):
        """Remove ablation hooks."""
        if hasattr(self, 'ablation_hooks'):
            for hook in self.ablation_hooks:
                hook.remove()
            self.ablation_hooks = []
        self.hooks_registered = False


class GPT2Model(BaseModel):
    """GPT2 model implementation using paper-correct masked model."""
    
    def _load_model(self) -> None:
        """Load GPT2 model."""
        logger = get_logger()
        
        # Load model with configuration
        torch_dtype = getattr(torch, self.config.get('torch_dtype', 'float16'))
        device_map = self.config.get('device_map', None)
        trust_remote_code = self.config.get('trust_remote_code', True)
        
        logger.info(f"Loading GPT2 model from {self.model_path}")
        
        # Use our paper-correct implementation
        load_kwargs = {
            'torch_dtype': torch_dtype,
            'trust_remote_code': trust_remote_code
        }
        
        # Only add device_map if it's not None
        if device_map is not None:
            load_kwargs['device_map'] = device_map
        
        self.model = PaperCorrectMaskedGPT2.from_pretrained(
            self.model_path,
            **load_kwargs
        )
        
        # Set device
        if device_map == 'auto':
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(self.config.get('device', "cuda" if torch.cuda.is_available() else "cpu"))
            self.model = self.model.to(self.device)
    
    def _setup_tokenizer(self) -> None:
        """Setup GPT2 tokenizer."""
        logger = get_logger()
        
        logger.info(f"Loading tokenizer from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Tokenizer setup completed")
    
    def _get_model_info(self) -> None:
        """Get GPT2 model information."""
        logger = get_logger()
        
        # Get number of layers
        self.num_layers = len(self.model.transformer.h)
        
        # Get hidden size
        self.hidden_size = self.model.config.hidden_size
        
        # Generate layer names
        self.layer_names = [f"transformer.h.{i}" for i in range(self.num_layers)]
        
        logger.info(f"GPT2 model info: {self.num_layers} layers, {self.hidden_size} hidden size")
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names for hooking."""
        return self.layer_names 