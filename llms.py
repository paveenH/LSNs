from typing import List, Dict
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from utils import setup_hooks, get_layer_names


class LSNsModel:
    def __init__(self, model_path: str, diffusion_mode: str = None, cache_dir: str = "cache") -> None:
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._ensure_padding_token()

        # get number of layers and hidden size
        try:
            self.num_layers = len(self.model.model.layers)
        except AttributeError:
            self.num_layers = len(self.model.transformer.h)
        self.hidden_size = self.model.config.hidden_size
        
        self.layer_names = get_layer_names(self.model_path, self.num_layers)

    def _ensure_padding_token(self) -> None:
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_max_length(self, dataset):
        """
        Compute the maximum tokenized length for both positive and negative samples.
        """
        max_pos = max(len(self.tokenizer.encode(sent, truncation=False)) for sent in dataset.positive)
        max_neg = max(len(self.tokenizer.encode(sent, truncation=False)) for sent in dataset.negative)
        return max(max_pos, max_neg)
            
    @torch.no_grad()
    def extract_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str = "last-token",
    ) -> Dict[str, List[torch.Tensor]]:
        
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        hooks, layer_reps = setup_hooks(self.model, self.layer_names)

        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        batch_activations = {ln: [] for ln in self.layer_names}
        for i in range(input_ids.size(0)):
            for ln in self.layer_names:
                reps = layer_reps[ln][i]  # (T, H)
                if pooling == "mean":
                    vec = reps.mean(dim=0)
                elif pooling == "sum":
                    vec = reps.sum(dim=0)
                else:  # last-token
                    vec = reps[-1]
                batch_activations[ln].append(vec.cpu())

        for hook in hooks:
            hook.remove()

        return batch_activations
            
    
    def extract_representations(self, pooling: str, batch_size: int, dataset) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract activations for positive/negative stimuli across specified layers.
        """

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        layer_names = get_layer_names(self.model_path, self.num_layers)
        hidden_dim = self.hidden_size

        # max_length = self.get_max_length(dataset)
        max_length = 12
        print(f"[INFO] Auto-detected max token length: {max_length}")

        self.model.eval()

        reps = {
            "positive": {ln: np.zeros((len(dataset.positive), hidden_dim)) for ln in layer_names},
            "negative": {ln: np.zeros((len(dataset.negative), hidden_dim)) for ln in layer_names},
        }

        offset = 0
        for batch_data in tqdm(loader, desc="Extracting reps"):
            sents, nonwords = batch_data
    
            pos = self.tokenizer(sents, truncation=True, max_length=max_length, return_tensors="pt")
            neg = self.tokenizer(nonwords, truncation=True, max_length=max_length, return_tensors="pt")

            batch_pos = self.extract_batch(pos.input_ids, pos.attention_mask, pooling)
            batch_neg = self.extract_batch(neg.input_ids, neg.attention_mask, pooling)

            bsz = len(sents)
            for ln in layer_names:
                reps["positive"][ln][offset : offset + bsz] = torch.stack(batch_pos[ln]).numpy()
                reps["negative"][ln][offset : offset + bsz] = torch.stack(batch_neg[ln]).numpy()
            offset += bsz

        return reps
