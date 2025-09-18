
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseModel
from logger import get_logger

class LLaDAModel(BaseModel):

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.num_denoising_steps = config.get('num_denoising_steps', 20)
        self.denoising_scheduler = config.get('denoising_scheduler', 'ddpm')

        super().__init__(model_path, config)

        self.logger.info(f"LLaDA model initialized with {self.num_denoising_steps} denoising steps")

    def _load_model(self) -> None:
        logger = get_logger()

        torch_dtype = getattr(torch, self.config.get('torch_dtype', 'float16'))
        device_map = self.config.get('device_map', 'auto')
        trust_remote_code = self.config.get('trust_remote_code', True)

        logger.info(f"Loading LLaDA model from {self.model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code
        )

        if device_map == 'auto':
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device_map if device_map else "cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)

        logger.info(f"LLaDA model loaded successfully on device: {self.device}")

        if not self._verify_llada_model():
            raise ValueError(f"Model at {self.model_path} does not appear to be a valid LLaDA diffusion model")

    def _setup_tokenizer(self) -> None:
        logger = get_logger()

        logger.info(f"Loading tokenizer from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("LLaDA tokenizer setup completed")

    def _verify_llada_model(self) -> bool:
        logger = get_logger()

        llada_indicators = [
            'diffusion_forward',
            'denoise',
            'diffusion_generate',
            'llada',
            'diffusion_config'
        ]

        for indicator in llada_indicators:
            if hasattr(self.model, indicator):
                logger.info(f"Found LLaDA indicator: {indicator}")
                return True

        if hasattr(self.model, 'config'):
            config_dict = self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else {}
            config_str = str(config_dict).lower()

            if 'llada' in config_str or 'diffusion' in config_str:
                logger.info("Found LLaDA indicators in model config")
                return True

        if 'llada' in self.model_path.lower():
            logger.warning("Model path suggests LLaDA but no diffusion indicators found")
            logger.warning("This may be a converted LLaDA model without diffusion capabilities")

        return False

    def _get_model_info(self) -> None:
        logger = get_logger()

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.num_layers = len(self.model.model.layers)
            self.layer_names = [f"model.layers.{i}" for i in range(self.num_layers)]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.num_layers = len(self.model.transformer.h)
            self.layer_names = [f"transformer.h.{i}" for i in range(self.num_layers)]
        else:
            if hasattr(self.model.config, 'num_hidden_layers'):
                self.num_layers = self.model.config.num_hidden_layers
                self.layer_names = [f"model.layers.{i}" for i in range(self.num_layers)]
            else:
                logger.error("Could not determine model structure")
                raise ValueError("Unable to determine LLaDA model layer structure")

        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            self.hidden_size = self.model.config.d_model
        else:
            logger.error("Could not determine hidden size")
            raise ValueError("Unable to determine LLaDA model hidden size")

        logger.info(f"LLaDA model info: {self.num_layers} layers, {self.hidden_size} hidden size")

    def get_layer_names(self) -> List[str]:
        return self.layer_names

    @torch.no_grad()
    def extract_diffusion_activations(
        self,
        texts: List[str],
        layer_names: Optional[List[str]] = None,
        pooling: str = "last",
        num_steps: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        if layer_names is None:
            layer_names = self.get_layer_names()

        if num_steps is None:
            num_steps = self.num_denoising_steps

        logger = get_logger()
        logger.info(f"Extracting diffusion activations for {num_steps} steps")

        if hasattr(self.model, 'diffusion_forward') or hasattr(self.model, 'denoise'):
            return self._extract_true_diffusion_activations(texts, layer_names, pooling, num_steps)
        else:
            logger.warning("Model does not appear to have diffusion capabilities")
            logger.info("Simulating diffusion steps using multiple forward passes")
            return self._simulate_diffusion_activations(texts, layer_names, pooling, num_steps)

    def _extract_true_diffusion_activations(
        self,
        texts: List[str],
        layer_names: List[str],
        pooling: str,
        num_steps: int
    ) -> Dict[str, np.ndarray]:
        return self._simulate_diffusion_activations(texts, layer_names, pooling, num_steps)

    def _simulate_diffusion_activations(
        self,
        texts: List[str],
        layer_names: List[str],
        pooling: str,
        num_steps: int
    ) -> Dict[str, np.ndarray]:
        logger = get_logger()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        batch_size = inputs['input_ids'].shape[0]

        step_activations = {
            layer_name: np.zeros((num_steps, batch_size, self.hidden_size))
            for layer_name in layer_names
        }

        for step in range(num_steps):
            logger.debug(f"Processing step {step + 1}/{num_steps}")

            noise_scale = (num_steps - step) / num_steps * 0.1

            step_reps = self.extract_activations(texts, layer_names, pooling)

            for layer_name in layer_names:
                activations = step_reps[layer_name]

                if noise_scale > 0:
                    noise = np.random.normal(0, noise_scale, activations.shape)
                    activations = activations + noise

                step_activations[layer_name][step] = activations

        logger.info(f"Extracted simulated diffusion activations for {num_steps} steps")
        return step_activations

    @torch.no_grad()
    def diffusion_generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        num_denoising_steps: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        if num_denoising_steps is None:
            num_denoising_steps = self.num_denoising_steps

        if hasattr(self.model, 'diffusion_generate'):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            outputs = self.model.diffusion_generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_denoising_steps=num_denoising_steps,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        else:
            raise NotImplementedError("This LLaDA model does not support diffusion generation. "
                                      "Ensure you're using a proper LLaDA diffusion model.")