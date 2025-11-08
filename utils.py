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
        return f.read().split('\n\n')

def read_story(path):
    with open(path, "r") as f:
        story = [line.strip() for line in f.readlines()]
    return ' '.join(story)

def append_options(question):
    options = ["True", "False"]
    return f"{question}\nOptions:\n- {options[0]}\n- {options[1]}"

def read_question(path):
    with open(path, "r") as f:
        question = f.read().split('\n\n')[0]
    return append_options(question.replace("\n", ""))

def read_story_v2(path):
    with open(path, "r") as f:
        stories = f.read()
    stories = [append_options(story) for story in stories.split("\n\n")]
    return stories

def _get_layer(module, layer_name: str) -> torch.nn.Module:
    SUBMODULE_SEPARATOR = '.'
    for part in layer_name.split(SUBMODULE_SEPARATOR):
        module = module._modules.get(part)
        assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
    return module
    
def _register_hook(layer: torch.nn.Module,
                    key: str,
                    target_dict: dict):
    def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
        output = output[0] if isinstance(output, (tuple, list)) else output
        target_dict[key] = output

    hook = layer.register_forward_hook(hook_function)
    return hook

def setup_hooks(model, layer_names):
    hooks = []
    layer_representations = OrderedDict()

    for layer_name in layer_names:
        layer = _get_layer(model, layer_name)
        hook = _register_hook(layer, key=layer_name,
                                target_dict=layer_representations)
        hooks.append(hook)

    return hooks, layer_representations
