# LSNs Experiment Guide

This guide shows how to reproduce the "Language-Selective Neurons in Large Language Models" paper experiments.

## Setup

### 1. Clone repository
```bash
git clone <repository-url>
cd LSNs
```

### 2. Create virtual environment
```bash
python3 -m venv lsns_env
source lsns_env/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running Experiments

### Step 1: Localize language-selective neurons
```bash
python localize.py --model-name gpt2 --percentage 5.0 --network language --pooling last-token --device cpu
```

### Step 2: Test language-selective ablation
```bash
python generate_lesion.py --model-name gpt2 --prompt "The quick brown fox" --percentage 1.0 --network language --device cpu
```

### Step 3: Test baseline (no ablation)
```bash
python generate_lesion.py --model-name gpt2 --prompt "The quick brown fox" --percentage 1.0 --network none --device cpu
```

### Step 4: Test random control ablation
```bash
python generate_lesion.py --model-name gpt2 --prompt "The quick brown fox" --percentage 1.0 --network random --device cpu
```

## Parameters

- `--model-name`: Model to use (gpt2)
- `--percentage`: Percentage of neurons to identify/ablate
- `--network`: Network type (language) or ablation type (language/random/none)
- `--pooling`: Token pooling method (last-token)
- `--device`: Device to use (cpu/cuda)

## Extended Usage (Scalable Framework)

### Using Different Models

```bash
# GPT2 variants
python localize.py --model-name gpt2-medium --percentage 5.0 --network language --pooling last-token
python localize.py --model-name gpt2-large --percentage 5.0 --network language --pooling last-token

# Other model architectures (requires model factory integration)
python localize.py --model-name llama-7b --percentage 5.0 --network language --pooling last-token
python localize.py --model-name gemma-2b --percentage 5.0 --network language --pooling last-token
```

### Using Different Analysis Methods

```python
# Create custom localization with different methods
from analysis.mean_analyzer import MeanAnalyzer
from analysis.nmd_analyzer import NMDAnalyzer

# Mean-based analysis (global percentage)
analyzer = MeanAnalyzer({'percentage': 5.0, 'localize_range': '100-100'})
mask, metadata = analyzer.analyze(positive_activations, negative_activations)

# NMD-based analysis (per-layer ratio)
analyzer = NMDAnalyzer({'topk_ratio': 0.05})  # 5% per layer
mask, metadata = analyzer.analyze(positive_activations, negative_activations)

# NMD with percentage (converted to topk_ratio)
analyzer = NMDAnalyzer({'percentage': 5.0})  # Approximates 5% per layer
mask, metadata = analyzer.analyze(positive_activations, negative_activations)
```

### Using Configuration Files

Create `my_experiment.yaml`:
```yaml
model:
  name: "gpt2"
  device: "cpu"
  torch_dtype: "float32"

analysis:
  method: "ttest"
  percentage: 5.0
  pooling: "last"
  
output_dir: "my_results"
cache_dir: "my_cache"
```

Use with Python:
```python
from config import ExperimentConfig
config = ExperimentConfig.from_yaml('my_experiment.yaml')
```

### Adding New Models

1. Create model class:
```python
from models.base import BaseModel

class MyModel(BaseModel):
    def _load_model(self): # implement
    def _setup_tokenizer(self): # implement  
    def _get_model_info(self): # implement
    def get_layer_names(self): # implement
```

2. Register in factory:
```python
from models.factory import ModelFactory
ModelFactory.register_model('mymodel', MyModel)
```

### Adding New Analysis Methods

1. Create analyzer:
```python
from analysis.base import BaseAnalyzer

class MyAnalyzer(BaseAnalyzer):
    def analyze(self, positive_activations, negative_activations):
        # implement analysis logic
        return mask, metadata
    
    def get_analyzer_name(self):
        return "my_method"
```

2. Use directly:
```python
analyzer = MyAnalyzer({'percentage': 5.0})
mask, metadata = analyzer.analyze(pos_data, neg_data)
```

## Example: T-test vs NMD Comparison

Run the demo to see how different analysis methods work:

```bash
python demo_nmd_experiment.py
```

This demonstrates:
- **T-test**: Global selection (finds strongest neurons across all layers)
- **NMD**: Per-layer selection (finds top neurons within each layer)
- **Key difference**: T-test concentrates selections, NMD distributes evenly

Example output:
```
Layer  T-test   NMD    Difference
0      25       38     -13
1      45       38     +7  
2      67       38     +29
...
Total  460      456    +4
```

Shows how T-test finds more neurons in important layers (like layer 2), while NMD maintains consistent distribution.

## Files

Results are saved in `cache/` directory:
- `gpt2_network=language_pooling=last-token_range=100-100_perc=5.0_nunits=None_pretrained=True.npy`
- `gpt2_network=language_pooling=last-token_pretrained=True_pvalues.npy`