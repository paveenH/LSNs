from typing import List, Dict

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from utils import setup_hooks, get_layer_names
from datasets import LangLocDataset


class LSNsModel:
    def __init__(
        self,
        model_path: str,
        diffusion_mode: str = None,
        cache_dir: str = "cache"
    ) -> None:
        self.model_path = model_path
        self.diffusion_mode = diffusion_mode
        self.cache_dir = cache_dir

        if diffusion_mode == "dream":
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        # get number of layers and hidden size
        try:
            self.num_layers = len(self.model.model.layers)
        except AttributeError:
            self.num_layers = len(self.model.transformer.h)
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def extract_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str = "last-token",
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Run a forward pass and extract per-layer activations for a batch.
        """
        layer_names = get_layer_names(self.model_path, self.num_layers)
        batch_activations = {ln: [] for ln in layer_names}
        hooks, layer_reps = setup_hooks(self.model, layer_names)

        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        for idx in range(input_ids.size(0)):
            for ln in layer_names:
                reps = layer_reps[ln][idx]
                if pooling == "mean":
                    act = reps.mean(dim=0).cpu()
                elif pooling == "sum":
                    act = reps.sum(dim=0).cpu()
                else:
                    act = reps[-1].cpu()
                batch_activations[ln].append(act)

        for hook in hooks:
            hook.remove()

        return batch_activations

    def extract_representations(
        self,
        max_length: int, 
        pooling: str,
        batch_size: int,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract activations for positive/negative stimuli across specified layers.
        """
        dataset = LangLocDataset()
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        layer_names = get_layer_names(self.model_path, self.num_layers)
        hidden_dim = self.hidden_size
        
        self.model.eval()
        self.model.to(self.model.device)
        
        reps = {
            "positive": {ln: np.zeros((len(dataset.positive), hidden_dim)) for ln in layer_names},
            "negative": {ln: np.zeros((len(dataset.negative), hidden_dim)) for ln in layer_names}
        }

        offset = 0
        for batch_data in tqdm(loader, desc="Extracting reps"):
            sents, nonwords = batch_data
            # tokenize
            pos = self.tokenizer(sents, truncation=True, padding=True,
                                 max_length=max_length, return_tensors='pt').to(self.model.device)
            neg = self.tokenizer(nonwords, truncation=True, padding=True,
                                 max_length=max_length, return_tensors='pt').to(self.model.device)

            batch_pos = self.extract_batch(pos.input_ids, pos.attention_mask, pooling)
            batch_neg = self.extract_batch(neg.input_ids, neg.attention_mask, pooling)

            bsz = len(sents)
            for ln in layer_names:
                reps["positive"][ln][offset:offset+bsz] = torch.stack(batch_pos[ln]).numpy()
                reps["negative"][ln][offset:offset+bsz] = torch.stack(batch_neg[ln]).numpy()
            offset += bsz

        return reps
