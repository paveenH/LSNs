import os
import sys
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel
from collections import OrderedDict

CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")

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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--prompt", type=str, required=True)
    argparser.add_argument("--percentage", type=float, required=True)
    argparser.add_argument("--network", type=str, default="language", choices=["language", "random", "none"])   
    argparser.add_argument("--device", type=str, default=None)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--pooling", type=str, default="last-token", choices=["last-token", "mean"])
    argparser.add_argument("--localize-range", type=str, default="100-100")

    args = argparser.parse_args()

    seed = args.seed
    percentage = args.percentage
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name
    network = args.network
    prompt = args.prompt
    pooling = args.pooling
    loc_range = args.localize_range

    print(f"> Running with model {model_name}")

    # Use paper-correct model implementation
    if "gpt2" in model_name:
        model = PaperCorrectMaskedGPT2.from_pretrained(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported in paper-correct version")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    model_name = os.path.basename(model_name)

    print(f"> Running with {network} mask")

    if network in ["language", "random"]:
        mask_path = f"{model_name}_network=language_pooling={pooling}_range={loc_range}_perc={percentage}_nunits=None_pretrained=True.npy"
    else:
        mask_path = None

    if mask_path is not None:
        language_mask = np.load(f"{CACHE_DIR}/{mask_path}")
        num_active_units = int(language_mask.sum())

        if network == "random":
            num_layers, hidden_dim = language_mask.shape
            total_num_units = np.prod(language_mask.shape)
            invlang_mask_indices = np.arange(total_num_units)[(1 - language_mask).flatten().astype(bool)]
            np.random.seed(seed)
            rand_indices = np.random.choice(invlang_mask_indices, size=num_active_units, replace=False)
            lang_mask_rand = np.full(total_num_units, 0)
            lang_mask_rand[rand_indices] = 1
            assert np.sum(lang_mask_rand) == num_active_units
            language_mask = lang_mask_rand.reshape((num_layers, hidden_dim))

        # PAPER CORRECT: Invert the mask so 0 = ablate, 1 = keep
        inverted_mask = 1 - language_mask
        model.set_language_selective_mask(torch.tensor(inverted_mask, dtype=torch.float32).to(device))
        print("Loaded language mask with", num_active_units, "units, with shape", language_mask.shape)
        print("Mask inverted: 0 = ablate, 1 = keep")
        print("Applying ablation at each transformer block output (paper methodology)")
    else:
        model.set_language_selective_mask(None)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))