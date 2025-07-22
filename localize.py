from typing import List, Dict

import os
import torch
import argparse
import numpy as np
import transformers

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind, false_discovery_control

from model_utils import get_layer_names, get_hidden_dim
from utils import setup_hooks
from datasets import LangLocDataset, TOMLocDataset, MDLocDataset

CACHE_DIR = os.environ.get("LOC_CACHE", "cache")

def extract_batch(
    model: torch.nn.Module, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor,
    layer_names: List[str],
    pooling: str = "last-token",
):
    
    batch_activations = {layer_name: [] for layer_name in layer_names}
    hooks, layer_representations = setup_hooks(model, layer_names)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    for sample_idx in range(len(input_ids)):
        for layer_idx, layer_name in enumerate(layer_names):
            if pooling == "mean":
                activations = layer_representations[layer_name][sample_idx].mean(dim=0).cpu()
            elif pooling == "sum":
                activations = layer_representations[layer_name][sample_idx].sum(dim=0).cpu()
            else:
                activations = layer_representations[layer_name][sample_idx][-1].cpu()    
            batch_activations[layer_name] += [activations]

    for hook in hooks:
        hook.remove()

    return batch_activations

def extract_representations(
    network: str,
    pooling: str,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    layer_names: List[str],
    hidden_dim: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, np.array]]:

    if network == "language":
        loc_dataset = LangLocDataset()
    elif network == "theory-of-mind":
        loc_dataset = TOMLocDataset()
    elif network == "multiple-demand":
        loc_dataset = MDLocDataset()
    else:
        raise ValueError(f"Unsupported network: {network}")

    # Get the activations of the model on the dataset
    langloc_dataloader = DataLoader(loc_dataset, batch_size=batch_size, num_workers=0)

    print(f"> Using Device: {device}")

    model.eval()
    model.to(device)

    final_layer_representations = {
        "positive": {layer_name: np.zeros((len(loc_dataset.positive), hidden_dim)) for layer_name in layer_names},
        "negative": {layer_name: np.zeros((len(loc_dataset.negative), hidden_dim)) for layer_name in layer_names}
    }
    
    for batch_idx, batch_data in tqdm(enumerate(langloc_dataloader), total=len(langloc_dataloader)):

        sents, non_words = batch_data
        if network == "language":
            sent_tokens = tokenizer(sents, truncation=True, max_length=12, return_tensors='pt').to(device)
            non_words_tokens = tokenizer(non_words, truncation=True, max_length=12, return_tensors='pt').to(device)
        else:
            sent_tokens = tokenizer(sents, padding=True, return_tensors='pt').to(device)
            non_words_tokens = tokenizer(non_words, padding=True, return_tensors='pt').to(device)
        
        batch_real_actv = extract_batch(model, sent_tokens["input_ids"], sent_tokens["attention_mask"], layer_names, pooling)
        batch_rand_actv = extract_batch(model, non_words_tokens["input_ids"], non_words_tokens["attention_mask"], layer_names, pooling)

        for layer_name in layer_names:
            final_layer_representations["positive"][layer_name][batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.stack(batch_real_actv[layer_name]).numpy()
            final_layer_representations["negative"][layer_name][batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.stack(batch_rand_actv[layer_name]).numpy()

    return final_layer_representations

def localize(model_id: str,
    network: str,
    pooling: str,
    model: torch.nn.Module, 
    num_units: int, 
    tokenizer: transformers.PreTrainedTokenizer, 
    hidden_dim: int, 
    layer_names: List[str], 
    batch_size: int,
    seed: int,
    device: torch.device,
    percentage: float = None,
    localize_range: str = None,
    pretrained: bool = True,
    overwrite: bool = False,
):
    """
    Localize network selective units in the model.
    """

    range_start, range_end = map(int, localize_range.split("-"))

    save_path = f"{CACHE_DIR}/{model_id}_network={network}_pooling={pooling}_range={localize_range}_perc={percentage}_nunits={num_units}_pretrained={pretrained}.npy"
    save_path_pvalues = f"{CACHE_DIR}/{model_id}_network={network}_pooling={pooling}_pretrained={pretrained}_pvalues.npy"

    if os.path.exists(save_path) and not overwrite:
        print(f"> Loading mask from {save_path}")
        return np.load(save_path)

    representations = extract_representations(
        network=network, 
        pooling=pooling,
        model=model, 
        tokenizer=tokenizer, 
        layer_names=layer_names, 
        hidden_dim=hidden_dim, 
        batch_size=batch_size, 
        device=device,
    )

    p_values_matrix = np.zeros((len(layer_names), hidden_dim))
    t_values_matrix = np.zeros((len(layer_names), hidden_dim))

    for layer_idx, layer_name in tqdm(enumerate(layer_names), total=len(layer_names)):

        positive_actv = np.abs(representations["positive"][layer_name])
        negative_actv = np.abs(representations["negative"][layer_name])

        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = ttest_ind(positive_actv, negative_actv, axis=0, equal_var=False)
 
    def is_topk(a, k=1):
        _, rix = np.unique(-a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)
    
    def is_bottomk(a, k=1):
        _, rix = np.unique(a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)
    
    np.random.seed(seed)
    if percentage is not None:
        num_units = int((percentage/100) * hidden_dim*len(layer_names))
        print(f"> Percentage: {percentage}% --> Num Units: {num_units}")

    if localize_range is not None and range_start < range_end:
        range_start_val = np.percentile(t_values_matrix, range_start)
        range_end_val = np.percentile(t_values_matrix, range_end)
        # take random num_units from that percentile range
        mask_range = (t_values_matrix >= range_start_val) & (t_values_matrix <= range_end_val)
        total_num_units = np.prod(mask_range.shape)
        mask_range_indices = np.arange(total_num_units)[mask_range.flatten()]
        rand_indices = np.random.choice(mask_range_indices, size=num_units, replace=False)
        language_mask = np.full(total_num_units, 0)
        language_mask[rand_indices] = 1
        language_mask = language_mask.reshape(mask_range.shape)
        print(f"> Num units in range {range_start}-{range_end}: {language_mask.sum()}")
    elif localize_range and range_start == range_end and int(range_start) == 0:
        language_mask = is_bottomk(t_values_matrix, k=num_units)
    else:
        language_mask = is_topk(t_values_matrix, k=num_units)

    print(f"> Num units: {language_mask.sum()}")
    num_layers, num_units = p_values_matrix.shape
    adjusted_p_values = false_discovery_control(p_values_matrix.flatten())
    adjusted_p_values = adjusted_p_values.reshape((num_layers, num_units))

    np.save(save_path, language_mask)
    np.save(save_path_pvalues, adjusted_p_values)
    print(f"> {model_id} {network} mask cached to {save_path}")
    return language_mask

if  __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Localize Units in LLMs")
    parser.add_argument("--model-name", type=str, required=True, help="huggingface model name")
    parser.add_argument("--percentage", type=float, default=None, help="percentage of units to localize")
    parser.add_argument("--localize-range", type=str, default="100-100", help="percentile in which to localize, 100-100 and 0-0 indicate top and least selective units respectively")
    parser.add_argument("--network", type=str, default="language", help="network to localize")
    parser.add_argument("--pooling", type=str, default="last-token", choices=["last-token", "mean"], help="token aggregation method")
    parser.add_argument("--num-units", type=int, default=None, help="number of units to localize, percentage overrides it")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default=None, help="device to use")
    parser.add_argument("--untrained", action="store_true", help="use an untrained version of the model")
    parser.add_argument("--overwrite", action="store_true", help="overwrite current mask if cached")
    args = parser.parse_args()

    assert args.percentage or args.num_units, "You must either provide percentage of units to localize or number of units"
    assert args.network in {"language", "theory-of-mind", "multiple-demand"}, "Unsupported network"

    model_name = args.model_name
    pretrained = not args.untrained
    localize_range = args.localize_range
    num_units = args.num_units
    percentage = args.percentage
    pooling = args.pooling
    network = args.network
    seed = args.seed
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device is None else args.device

    if pretrained:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        model_config = transformers.AutoConfig.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_config(config=model_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_name = os.path.basename(model_name)

    model_layer_names = get_layer_names(model_name)
    hidden_dim = get_hidden_dim(model_name)

    model.eval()

    localize(
        model_id=model_name,
        network=network,
        pooling=pooling,
        model=model,
        num_units=num_units,
        percentage=percentage,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        layer_names=model_layer_names,
        batch_size=batch_size,
        seed=seed,
        device=device,
        localize_range=localize_range,
        pretrained=pretrained,
        overwrite=args.overwrite,
    )
