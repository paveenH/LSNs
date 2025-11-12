#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 model implementation for LSNs experiments.
Now simplified: uses BaseModel's universal masking & hook system.
"""

from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from .base import BaseModel
from logger import get_logger


class GPT2Model(BaseModel):
    """Unified GPT-2 implementation compatible with BaseModel ablation hooks."""

    def _load_model(self) -> None:
        """Load GPT-2 model."""
        logger = get_logger()

        torch_dtype = getattr(torch, self.config.get("torch_dtype", "float16"))
        device_map = self.config.get("device_map", None)
        trust_remote_code = self.config.get("trust_remote_code", True)

        logger.info(f"Loading GPT-2 model from {self.model_path}")

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        # Directly use the official HuggingFace GPT2 model (no subclass)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path, **load_kwargs)

        # Device assignment
        if device_map == "auto":
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(
                self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            self.model = self.model.to(self.device)

        logger.info("GPT-2 model loaded successfully")

    def _setup_tokenizer(self) -> None:
        """Setup tokenizer."""
        logger = get_logger()

        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Tokenizer setup completed")

    def _get_model_info(self) -> None:
        """Gather GPT-2 architecture info."""
        logger = get_logger()

        self.num_layers = len(self.model.transformer.h)
        self.hidden_size = self.model.config.hidden_size
        self.layer_names = [f"transformer.h.{i}" for i in range(self.num_layers)]

        logger.info(
            f"GPT-2 model info: {self.num_layers} layers, {self.hidden_size} hidden size"
        )

    def get_layer_names(self) -> List[str]:
        """Return names of transformer blocks."""
        return self.layer_names