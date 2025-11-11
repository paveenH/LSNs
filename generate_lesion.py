import os
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer

from models import ModelFactory


CACHE_DIR = os.environ.get("LOC_CACHE", "cache")

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
    
    # === NEW: Load hook from BaseModel ===
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = ModelFactory.create_model(model_name, config={})
    model.to_device(device)
    model.model.eval()  

    print(f"> Running with {network} mask")


    if network in ["language", "random"]:
        # mask_path = f"{model_name}_network=language_pooling={pooling}_range={loc_range}_perc={percentage}_nunits=None_pretrained=True.npy"
        mask_path = "real_nmd_mask.npy"
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
        # model.set_language_selective_mask(torch.tensor(inverted_mask, dtype=torch.float32).to(device))
        mask_dtype = next(model.model.parameters()).dtype
        model.set_language_selective_mask(torch.tensor(inverted_mask, dtype=mask_dtype).to(device))

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
        prompt,
        max_new_tokens=10,
        do_sample=False
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    