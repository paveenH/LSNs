from typing import OrderedDict

import json
import torch


def write_txt(path, data):
    with open(path, "w") as f:
        f.writelines(data)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]


def read_txt(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def read_fewshot_tom(path):
    with open(path, "r") as f:
        return f.read().split("\n\n")


def read_story(path):
    with open(path, "r") as f:
        story = [line.strip() for line in f.readlines()]
    return " ".join(story)


def append_options(question):
    options = ["True", "False"]
    return f"{question}\nOptions:\n- {options[0]}\n- {options[1]}"


def read_question(path):
    with open(path, "r") as f:
        question = f.read().split("\n\n")[0]
    return append_options(question.replace("\n", ""))


def read_story_v2(path):
    with open(path, "r") as f:
        stories = f.read()
    stories = [append_options(story) for story in stories.split("\n\n")]
    return stories


def _get_layer(module, layer_name: str) -> torch.nn.Module:
    SUBMODULE_SEPARATOR = "."
    for part in layer_name.split(SUBMODULE_SEPARATOR):
        module = module._modules.get(part)
        assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
    return module


def _register_hook(layer: torch.nn.Module, key: str, target_dict: dict):
    # instantiate parameters to function defaults; otherwise they would change on next function call
    def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
        # fix for when taking out only the hidden state, this is different from dropout because of residual state
        # see:  https://github.com/huggingface/transformers/blob/c06d55564740ebdaaf866ffbbbabf8843b34df4b/src/transformers/models/gpt2/modeling_gpt2.py#L428
        output = output[0] if isinstance(output, (tuple, list)) else output
        target_dict[key] = output

    hook = layer.register_forward_hook(hook_function)
    return hook


def get_layer_names(model_name: str, num_blocks: int) -> list:
    model_name = model_name.lower()

    if any(key in model_name for key in ["gpt2", "falcon"]):
        prefix = "transformer.h"
    elif any(key in model_name for key in ["llama", "gemma", "phi", "mistral", "qwen"]):
        prefix = "model.layers"
    else:
        raise ValueError(f"Model type for '{model_name}' not supported!")

    return [f"{prefix}.{i}" for i in range(num_blocks)]


def setup_hooks(model, layer_names):
    """set up the hooks for recording internal neural activity from the model (aka layer activations)"""
    hooks = []
    layer_representations = OrderedDict()

    for layer_name in layer_names:
        layer = _get_layer(model, layer_name)
        hook = _register_hook(layer, key=layer_name, target_dict=layer_representations)
        hooks.append(hook)

    return hooks, layer_representations


def lesion_hooks(
        model,
        layer_names,
        mask,
        start: int = 1,
        end: int | None = None,
    ):
    
    """
    Register forward hooks to zero-out (lesion) neurons according to `mask`
    for layers in [start, end) (1-based indices).
    """
    # Convert mask to tensor
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32)
    
    device = next(model.parameters()).device
    mask = mask.to(device)
    num_layers, hidden = mask.shape
    
    # setup layer range; default 1, 33
    if end is None or end > num_layers + 1:
        end = num_layers + 1
    start_idx = max(0, start - 1)
    end_idx = min(num_layers, end - 1)

    hooks = []
    for idx in range(start_idx, end_idx):
        name = layer_names[idx]
        layer = _get_layer(model, name)
        vec = mask[idx]
        def make_hook(mask_vec: torch.Tensor):
            def hook(module, inp, out):
                # out shape: (batch, seq_len, hidden)
                return out * mask_vec
            return hook
        hooks.append(layer.register_forward_hook(make_hook(vec)))

    return hooks
