#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory for creating different types of models.
"""

from typing import Dict, Any, Type
import torch
from transformers import AutoTokenizer

from .base import BaseModel
from .llama_model import LlamaModel
from .gpt2_model import GPT2Model
from .phi3_model import Phi3Model
from .gemma_model import GemmaModel
from .falcon_model import FalconModel
from .mistral_model import MistralModel
from .llada_model import LLaDAModel
from .dream_model import DreamModel

from logger import get_logger


class ModelFactory:
    """Factory for creating different types of models."""
    
    _models = {
        'llama': LlamaModel,
        'gpt2': GPT2Model,
        'phi3': Phi3Model,
        'gemma': GemmaModel,
        'falcon': FalconModel,
        'mistral': MistralModel,
        'llada': LLaDAModel,
        'dream': DreamModel
    }
    
    @classmethod
    def create_model(cls, model_path: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on the model path.
        
        Args:
            model_path: Path or name of the model
            config: Model configuration
            
        Returns:
            Model instance
        """
        logger = get_logger()
        
        # Determine model type from path
        model_type = cls._detect_model_type(model_path)
        
        if model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(cls._models.keys())}")
        
        logger.info(f"Creating {model_type} model from {model_path}")
        
        # Create model instance
        model_class = cls._models[model_type]
        return model_class(model_path, config)
    
    @classmethod
    def _detect_model_type(cls, model_path: str) -> str:
        """
        Detect model type from model path.
        
        Args:
            model_path: Path or name of the model
            
        Returns:
            Model type string
        """
        model_path_lower = model_path.lower()
        
        # Check for diffusion models first (more specific patterns)
        if 'llada' in model_path_lower:
            return 'llada'
        elif 'dream' in model_path_lower:
            return 'dream'
        # Check for base model types
        elif 'llama' in model_path_lower:
            return 'llama'
        elif 'gpt2' in model_path_lower:
            return 'gpt2'
        elif 'phi' in model_path_lower:
            return 'phi3'
        elif 'gemma' in model_path_lower:
            return 'gemma'
        elif 'falcon' in model_path_lower:
            return 'falcon'
        elif 'mistral' in model_path_lower:
            return 'mistral'
        else:
            # Default to GPT2 for unknown models
            return 'gpt2'
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type.
        
        Args:
            name: Model type name
            model_class: Model class
        """
        cls._models[name] = model_class
    
    @classmethod
    def get_supported_models(cls) -> list:
        """Get list of supported model types."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseModel]:
        """
        Get model class by type.
        
        Args:
            model_type: Model type name
            
        Returns:
            Model class
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type] 