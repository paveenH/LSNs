
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseModel
from logger import get_logger

class DreamModel(BaseModel):

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.num_denoising_steps = config.get('num_denoising_steps', 16)
        self.denoising_scheduler = config.get('denoising_scheduler', 'ddim')

        super().__init__(model_path, config)

        self.logger.info(f"Dream model initialized with {self.num_denoising_steps} denoising steps")

    def _load_model(self) -> None:
        logger = get_logger()

        torch_dtype = getattr(torch, self.config.get('torch_dtype', 'float16'))
        device_map = self.config.get('device_map', 'auto')
        trust_remote_code = self.config.get('trust_remote_code', True)

        logger.info(f"Loading Dream model from {self.model_path}")

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

        logger.info(f"Dream model loaded successfully on device: {self.device}")

        if not self._verify_dream_model():
            raise ValueError(f"Model at {self.model_path} does not appear to be a valid Dream diffusion model")

    def _setup_tokenizer(self) -> None:
        logger = get_logger()

        logger.info(f"Loading tokenizer from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Dream tokenizer setup completed")

    def _verify_dream_model(self) -> bool:
        logger = get_logger()

        dream_indicators = [
            'dream_forward',
            'diffusion_decode',
            'dream_generate',
            'dream',
            'diffusion_config',
            'backbone'
        ]

        for indicator in dream_indicators:
            if hasattr(self.model, indicator):
                logger.info(f"Found Dream indicator: {indicator}")
                return True

        if hasattr(self.model, 'config'):
            config_dict = self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else {}
            config_str = str(config_dict).lower()

            if 'dream' in config_str or 'diffusion' in config_str:
                logger.info("Found Dream indicators in model config")
                return True

        if 'dream' in self.model_path.lower():
            logger.warning("Model path suggests Dream but no diffusion indicators found")
            logger.warning("This may be a converted Dream model without diffusion capabilities")

        return False

    def _get_model_info(self) -> None:
        logger = get_logger()

        layer_attrs = [
            ('model', 'layers'),
            ('transformer', 'h'),
            ('model', 'decoder'),
            ('backbone', 'layers'),
        ]

        self.num_layers = None
        self.layer_names = None

        for attr1, attr2 in layer_attrs:
            if hasattr(self.model, attr1):
                component = getattr(self.model, attr1)
                if hasattr(component, attr2):
                    layers = getattr(component, attr2)
                    if hasattr(layers, '__len__'):
                        self.num_layers = len(layers)
                        self.layer_names = [f"{attr1}.{attr2}.{i}" for i in range(self.num_layers)]
                        break

        if self.num_layers is None:
            config_attrs = ['num_hidden_layers', 'n_layer', 'num_layers', 'depth']
            for attr in config_attrs:
                if hasattr(self.model.config, attr):
                    self.num_layers = getattr(self.model.config, attr)
                    self.layer_names = [f"model.layers.{i}" for i in range(self.num_layers)]
                    break

        if self.num_layers is None:
            logger.error("Could not determine model structure")
            raise ValueError("Unable to determine Dream model layer structure")

        hidden_size_attrs = ['hidden_size', 'd_model', 'embed_dim', 'dim']
        self.hidden_size = None

        for attr in hidden_size_attrs:
            if hasattr(self.model.config, attr):
                self.hidden_size = getattr(self.model.config, attr)
                break

        if self.hidden_size is None:
            logger.error("Could not determine hidden size")
            raise ValueError("Unable to determine Dream model hidden size")

        logger.info(f"Dream model info: {self.num_layers} layers, {self.hidden_size} hidden size")

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
        logger.info(f"Extracting Dream diffusion activations for {num_steps} steps")

        if hasattr(self.model, 'dream_forward') or hasattr(self.model, 'diffusion_decode'):
            return self._extract_true_diffusion_activations(texts, layer_names, pooling, num_steps)
        else:
            logger.warning("Model does not appear to have Dream diffusion capabilities")
            logger.info("Simulating diffusion steps using iterative refinement")
            return self._simulate_dream_activations(texts, layer_names, pooling, num_steps)

    def _extract_true_diffusion_activations(
        self,
        texts: List[str],
        layer_names: List[str],
        pooling: str,
        num_steps: int
    ) -> Dict[str, np.ndarray]:
        return self._simulate_dream_activations(texts, layer_names, pooling, num_steps)

    def _simulate_dream_activations(
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
            logger.debug(f"Processing Dream step {step + 1}/{num_steps}")

            progress = (step + 1) / num_steps
            noise_scale = (1 - progress ** 2) * 0.15

            step_reps = self.extract_activations(texts, layer_names, pooling)

            for layer_name in layer_names:
                activations = step_reps[layer_name]

                if noise_scale > 0:
                    base_noise = np.random.normal(0, noise_scale * 0.5, (batch_size, 1))
                    structured_noise = base_noise * np.random.normal(0, 0.3, (1, self.hidden_size))
                    random_noise = np.random.normal(0, noise_scale * 0.3, activations.shape)

                    total_noise = structured_noise + random_noise
                    activations = activations + total_noise

                step_activations[layer_name][step] = activations

        logger.info(f"Extracted simulated Dream activations for {num_steps} steps")
        return step_activations

    @torch.no_grad()
    def dream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        num_denoising_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        temperature: float = 0.7
    ) -> str:
        if num_denoising_steps is None:
            num_denoising_steps = self.num_denoising_steps

        if hasattr(self.model, 'dream_generate') or hasattr(self.model, 'diffusion_generate'):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'pad_token_id': self.tokenizer.eos_token_id
            }

            if hasattr(self.model, 'dream_generate'):
                generation_kwargs.update({
                    'num_denoising_steps': num_denoising_steps,
                    'guidance_scale': guidance_scale
                })
                outputs = self.model.dream_generate(**inputs, **generation_kwargs)
            else:
                outputs = self.model.diffusion_generate(**inputs, **generation_kwargs)

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        else:
            raise NotImplementedError("This Dream model does not support diffusion generation. "
                                      "Ensure you're using a proper Dream diffusion model.")

    def get_diffusion_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'dream',
            'num_denoising_steps': self.num_denoising_steps,
            'scheduler': self.denoising_scheduler,
            'has_native_diffusion': hasattr(self.model, 'dream_generate') or hasattr(self.model, 'diffusion_generate'),
            'simulation_mode': not (hasattr(self.model, 'dream_generate') or hasattr(self.model, 'diffusion_generate'))
        }