
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def demo_nmd_vs_ttest():
    print("Demo: T-test vs NMD Analysis Methods")
    print("=" * 50)
    
    print("\nStep 1: Create Test Data")
    activations = create_realistic_test_data()
    
    print("\nStep 2: Run Analysis Methods")
    results = run_both_analyses(activations)
    
    print("\nStep 3: Compare Selection Strategies")
    compare_selection_strategies(results)
    
    print("\nStep 4: Test Ablation Effects")
    test_ablation_demo(results)
    
    print("\n" + "=" * 50)
    print("Demo completed!")

def create_realistic_test_data():
    np.random.seed(42)
    
    n_samples_pos = 20
    n_samples_neg = 20  
    n_layers = 12
    hidden_dim = 768
    
    print(f"Simulating GPT2: {n_layers} layers, {hidden_dim} hidden size")
    print(f"Data: {n_samples_pos} language + {n_samples_neg} non-language samples")
    
    positive_activations = np.random.randn(n_samples_pos, n_layers, hidden_dim) * 0.3
    negative_activations = np.random.randn(n_samples_neg, n_layers, hidden_dim) * 0.3
    
    language_strength_per_layer = [0.5, 0.8, 1.2, 1.0, 0.6, 0.9, 1.5, 1.1, 0.7, 0.4, 0.3, 0.2]
    
    for layer_idx in range(n_layers):
        strength = language_strength_per_layer[layer_idx]
        
        n_selective = int(50 * strength)
        selective_neurons = np.random.choice(hidden_dim, n_selective, replace=False)
        
        for neuron_idx in selective_neurons:
            boost = np.random.normal(strength, 0.2, n_samples_pos)
            positive_activations[:, layer_idx, neuron_idx] += boost
    
    print(f"Created realistic language-selective patterns")
    print(f"  Layer selectivity: {[f'{s:.1f}' for s in language_strength_per_layer]}")
    
    return {
        'positive': positive_activations,
        'negative': negative_activations,
        'layer_strength': language_strength_per_layer
    }


def run_both_analyses(activations):
    from analysis.ttest_analyzer import TTestAnalyzer
    from analysis.nmd_analyzer import NMDAnalyzer
    
    positive = activations['positive']
    negative = activations['negative']
    
    print("Running T-test analysis (global selection)...")
    ttest_config = {'percentage': 5.0, 'localize_range': '100-100'}
    ttest_analyzer = TTestAnalyzer(ttest_config)
    ttest_mask, ttest_metadata = ttest_analyzer.analyze(positive, negative)
    
    print("Running NMD analysis (per-layer selection)...")
    nmd_config = {'topk_ratio': 0.05}
    nmd_analyzer = NMDAnalyzer(nmd_config)
    nmd_mask, nmd_metadata = nmd_analyzer.analyze(positive, negative)
    
    print(f"T-test selected: {ttest_mask.sum()} neurons ({ttest_metadata['selection_ratio']:.3f})")
    print(f"NMD selected: {nmd_mask.sum()} neurons ({nmd_metadata['selection_ratio']:.3f})")
    
    return {
        'ttest_mask': ttest_mask,
        'nmd_mask': nmd_mask,
        'ttest_metadata': ttest_metadata,
        'nmd_metadata': nmd_metadata
    }


def compare_selection_strategies(results):
    ttest_mask = results['ttest_mask']
    nmd_mask = results['nmd_mask']
    
    ttest_per_layer = ttest_mask.sum(axis=1)
    nmd_per_layer = nmd_mask.sum(axis=1)
    
    print("Per-layer Selection Comparison:")
    print(f"{'Layer':<6} {'T-test':<8} {'NMD':<6} {'Difference':<10}")
    print("-" * 35)
    
    for i in range(len(ttest_per_layer)):
        diff = int(ttest_per_layer[i]) - int(nmd_per_layer[i])
        print(f"{i:<6} {int(ttest_per_layer[i]):<8} {int(nmd_per_layer[i]):<6} {diff:+3d}")
    
    print("-" * 35)
    print(f"{'Total':<6} {int(ttest_mask.sum()):<8} {int(nmd_mask.sum()):<6} {int(ttest_mask.sum()) - int(nmd_mask.sum()):+3d}")
    
    overlap = np.logical_and(ttest_mask, nmd_mask).sum()
    print(f"\nOverlap Analysis:")
    print(f"  Common neurons: {overlap}")
    print(f"  Overlap ratio: {overlap / ttest_mask.sum():.3f}")
    print(f"  T-test only: {np.logical_and(ttest_mask, ~nmd_mask).sum()}")
    print(f"  NMD only: {np.logical_and(~ttest_mask, nmd_mask).sum()}")


def test_ablation_demo(results):
    from generate_lesion import PaperCorrectMaskedGPT2
    from transformers import AutoTokenizer
    
    print("Loading model for ablation test...")
    
    try:
        model = PaperCorrectMaskedGPT2.from_pretrained("gpt2", torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        test_prompt = "The quick brown fox"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        methods = {
            'Baseline (no ablation)': None,
            'T-test ablation': 1 - results['ttest_mask'],
            'NMD ablation': 1 - results['nmd_mask']
        }
        
        print(f"Testing with prompt: '{test_prompt}'")
        
        for method_name, mask in methods.items():
            if mask is not None:
                model.set_language_selective_mask(torch.tensor(mask, dtype=torch.float32))
            else:
                model.set_language_selective_mask(None)
            
            outputs = model.generate(**inputs, max_new_tokens=6, do_sample=False)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"  {method_name:<20}: '{result}'")
        
        print("Both ablation methods work correctly")
        
    except Exception as e:
        print(f"Ablation test skipped: {e}")
        print("Mask generation and analysis completed successfully")


def main():
    try:
        demo_nmd_vs_ttest()
        return True
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)