#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base model class for LSNs experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from pathlib import Path

from logger import get_logger


class BaseModel(ABC):
    """Base class for all models in LSNs experiments."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.logger = get_logger()
        
        # Model attributes
        self.model = None
        self.tokenizer = None
        self.device = None
        self.num_layers = None
        self.hidden_size = None
        self.layer_names = None
        
        # Initialize model
        self._load_model()
        self._setup_tokenizer()
        self._get_model_info()
        
        self.language_selective_mask = None
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model from path."""
        pass
    
    @abstractmethod
    def _setup_tokenizer(self) -> None:
        """Setup the tokenizer."""
        pass
    
    @abstractmethod
    def _get_model_info(self) -> None:
        """Get model information (layers, hidden size, etc.)."""
        pass
    
    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Get list of layer names for hooking."""
        pass
    
    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        if self.model is not None:
            self.model = self.model.to(device)
            self.device = device
            self.logger.info(f"Model moved to device: {device}")
    
    def set_language_selective_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Set language selective mask for ablation experiments."""
        self.language_selective_mask = mask
        if mask is not None:
            self._register_ablation_hooks()
            self.logger.info(f"Language selective mask set with shape: {mask.shape}")
        else:
            self._remove_ablation_hooks()
            self.logger.info("Language selective mask cleared")
    
    def _register_ablation_hooks(self):
        """Attach forward hooks to transformer blocks for applying ablation mask."""
        if hasattr(self, "_ablation_hooks") and self._ablation_hooks:
            return

        def make_hook(layer_idx):
            def hook_fn(module, inputs, outputs):
                if self.language_selective_mask is None:
                    return outputs
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                mask_vec = self.language_selective_mask[layer_idx].to(hidden.device, dtype=hidden.dtype)
                masked = hidden * mask_vec.unsqueeze(0).unsqueeze(0)
                return (masked,) + outputs[1:] if isinstance(outputs, tuple) else masked
            return hook_fn

        # Unified block resolution
        blocks = (
            getattr(getattr(self.model, "model", None), "layers", None)
            or getattr(getattr(self.model, "transformer", None), "h", None)
            or getattr(getattr(self.model, "encoder", None), "layers", None)
        )
        if blocks is None:
            self.logger.warning("No recognizable transformer blocks found.")
            return

        self._ablation_hooks = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(blocks)]
        self.logger.info(f"Registered {len(self._ablation_hooks)} ablation hooks")
    
    
    def _remove_ablation_hooks(self):
        """Remove any registered ablation hooks."""
        if getattr(self, "_ablation_hooks", None):
            for h in self._ablation_hooks:
                h.remove()
            self._ablation_hooks = []
            self.logger.info("Removed ablation hooks")
    
    
    @torch.no_grad()
    def extract_activations(
        self, 
        texts: List[str], 
        layer_names: Optional[List[str]] = None,
        pooling: str = "last"
    ) -> Dict[str, np.ndarray]:
        """
        Extract activations from specified layers.
        
        Args:
            texts: List of input texts
            layer_names: List of layer names to extract from (if None, use all)
            pooling: Pooling strategy ('last', 'mean', 'sum', 'orig')
            
        Returns:
            Dictionary mapping layer names to activations
        """
        if layer_names is None:
            layer_names = self.get_layer_names()
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Setup hooks
        from utils import setup_hooks
        hooks, layer_reps = setup_hooks(self.model, layer_names)
        
        # Forward pass
        _ = self.model(**inputs)
        
        # Extract activations
        activations = {}
        for layer_name in layer_names:
            reps = layer_reps[layer_name]  # (batch_size, seq_len, hidden_dim)
            
            if pooling == "mean":
                pooled = reps.mean(dim=1)  # (batch_size, hidden_dim)
            elif pooling == "sum":
                pooled = reps.sum(dim=1)  # (batch_size, hidden_dim)
            elif pooling == "last":
                # Get last token for each sequence
                last_token_idxs = inputs['attention_mask'].sum(dim=1) - 1
                idx = last_token_idxs.unsqueeze(1).unsqueeze(2).expand(-1, 1, reps.size(-1))
                pooled = reps.gather(dim=1, index=idx).squeeze(1)
            elif pooling == "orig":
                # Use token at fixed position 11 (12th token)
                pooled = reps[:, 11, :]  # (batch_size, hidden_dim)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            
            activations[layer_name] = pooled.cpu().numpy()
        
        # Cleanup hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 20,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def save_model_info(self, output_path: Path) -> None:
        """Save model information to file."""
        info = {
            'model_path': self.model_path,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'layer_names': self.get_layer_names(),
            'device': str(self.device)
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Model info saved to {output_path}")
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(path={self.model_path}, layers={self.num_layers}, hidden={self.hidden_size})" 
    
    @torch.no_grad()
    def score(self, text: str) -> Dict[str, Any]:
        """
        Compute per-token logprobs and character offsets.
        Returns:
            {
                "input_ids": [...],
                "tokens": [...],
                "offsets": [(start,end), ...],   # char span for EACH token
                "logprobs": [...],
                "mean_logprob": float
            }
        """

        # -----------------------------------------------------
        # 1. Tokenize WITHOUT add_special_tokens
        # -----------------------------------------------------
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        input_ids = enc["input_ids"]  # [1, T]
        attention_mask = enc.get("attention_mask", None)

        # convert ids → tokens
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].tolist()
        )

        # -----------------------------------------------------
        # 2. Compute offsets manually (because LLaMA is NOT fast tokenizer)
        # -----------------------------------------------------
        offsets = []
        pos = 0
        for tok in tokens:
            # remove special token prefix
            clean_tok = tok.replace("▁", "")     # LLaMA sentencepiece marker
            clean_tok = clean_tok.lstrip()       # safety

            # find match in original text
            start = text.find(clean_tok, pos)
            if start == -1:
                start = pos
            end = start + len(clean_tok)
            offsets.append((start, end))

            pos = end

        # -----------------------------------------------------
        # 3. Forward pass with ablation mask
        # -----------------------------------------------------
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits  # [1, T, vocab]

        # -----------------------------------------------------
        # 4. Next-token log-prob
        # -----------------------------------------------------
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        logprobs = torch.log_softmax(shift_logits, dim=-1)

        token_logprobs = logprobs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1).squeeze(0)   # [T-1]

        return {
            "input_ids": input_ids[0].tolist(),
            "tokens": tokens,
            "offsets": [(int(a), int(b)) for a, b in offsets],
            "logprobs": token_logprobs.cpu().tolist(),
            "mean_logprob": float(token_logprobs.mean().item())
        }
    
    
    # =========================================================
    # Universal Zero-shot Classification APIs for GLUE tasks
    # =========================================================
    @torch.no_grad()
    def _score_choice(self, prompt: str, choice: str) -> float:
        """
        Compute the total logprob of generating `choice` after `prompt`.
        Uses model.score() so ablation hooks automatically apply.
        """
        full_text = prompt.strip() + " " + choice
        out = self.score(full_text)
        return sum(out["logprobs"])   # total logprob

    @torch.no_grad()
    def classify(self, text: str) -> int:
        """
        Zero-shot boolean classification using Yes/No.
        Output: 1 for yes, 0 for no.
        Works for CoLA / SST-2 style tasks.
        """

        # Default binary question prompt
        prompt = f"""
        Text: "{text}"
        Is this true? Answer Yes or No.
        """

        s_yes = self._score_choice(prompt, "Yes")
        s_no  = self._score_choice(prompt, "No")

        return 1 if s_yes > s_no else 0

    @torch.no_grad()
    def classify_pair(self, text1: str, text2: str) -> int:
        """
        Zero-shot boolean classification for sentence-pair tasks.
        Output: 1 for yes, 0 for no.
        Works for RTE / MRPC / QQP / QNLI style tasks.
        """
        prompt = f"""
        Sentence 1: "{text1}"
        Sentence 2: "{text2}"
        Do these sentences have the same meaning or logically entail each other?
        Answer Yes or No.
        """

        s_yes = self._score_choice(prompt, "Yes")
        s_no  = self._score_choice(prompt, "No")

        return 1 if s_yes > s_no else 0




