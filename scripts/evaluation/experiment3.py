import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import entropy

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import EVALUATION_DIR, ensure_dir

# METRICS
def compute_top_k_mass(attributions, k=10):
    """
    Compute proportion of total attribution in top-k tokens.
    
    Higher values = more concentrated (focused on few tokens)
    Lower values = more diffuse (spread across many tokens)
    """
    abs_attr = np.abs(attributions)
    total = abs_attr.sum()
    
    if total == 0:
        return 0.0
    
    # Get top-k values
    top_k_sum = np.sort(abs_attr)[-k:].sum()
    
    return top_k_sum / total


def compute_attribution_entropy(attributions):
    """
    Compute Shannon entropy over attribution distribution.
    
    Higher entropy = more diffuse (spread across tokens)
    Lower entropy = more concentrated (focused on few tokens)
    """
    abs_attr = np.abs(attributions)
    
    # Normalize to probability distribution
    total = abs_attr.sum()
    if total == 0:
        return 0.0
    
    prob_dist = abs_attr / total
    
    # Remove zeros for entropy calculation
    prob_dist = prob_dist[prob_dist > 0]
    
    return entropy(prob_dist)


def analyze_example_concentration(example_df, top_k=10):
    """
    Compute concentration metrics for a single example.
    """
    attributions = example_df['attribution'].values
    
    return {
        'top_k_mass': compute_top_k_mass(attributions, k=top_k),
        'entropy': compute_attribution_entropy(attributions),
        'n_tokens': len(attributions),
        'mean_abs_attr': np.abs(attributions).mean(),
        'max_abs_attr': np.abs(attributions).max(),
    }


# EXPERIMENT 3
def run_concentration_analysis(ig_results_path, top_k=10, model_name=None):
    """
    Analyze concentration metrics by confidence level.
    """
    # Load data
    df = pd.read_csv(ig_results_path)

    # Add model column if not present
    if 'model' not in df.columns:
        if model_name is None:
            # Try to infer from filename
            filename = Path(ig_results_path).stem
            if 'single' in filename.lower():
                model_name = 'single-task'
            elif 'multi' in filename.lower() or '4task' in filename.lower():
                model_name = 'multi-task'
            else:
                model_name = 'unknown'
        df['model'] = model_name
    
    # Add task column if not present
    if 'task' not in df.columns:
        df['task'] = 'Q_overall'

    # Compute metrics for each example
    results = []
    
    grouped = df.groupby(['model', 'task', 'confidence_level', 'example_idx'])
    
    for (model, task, conf_level, ex_idx), example_df in grouped:
        metrics = analyze_example_concentration(example_df, top_k)
        
        # Get example-level info
        pred_prob = example_df['pred_prob'].iloc[0]
        true_label = example_df['true_label'].iloc[0]
        convergence = example_df['convergence_delta'].iloc[0]
        
        results.append({
            'model': model,
            'task': task,
            'confidence_level': conf_level,
            'example_idx': ex_idx,
            'pred_prob': pred_prob,
            'true_label': true_label,
            'convergence_delta': convergence,
            **metrics
        })
    
    results_df = pd.DataFrame(results)
    
    # Aggregate by confidence level
    print("CONCENTRATION METRICS BY CONFIDENCE LEVEL")
    
    agg_results = []
    
    for (model, task, conf_level), group in results_df.groupby(['model', 'task', 'confidence_level']):
        agg = {
            'model': model,
            'task': task,
            'confidence_level': conf_level,
            'n_examples': len(group),
            'mean_top_k_mass': group['top_k_mass'].mean(),
            'std_top_k_mass': group['top_k_mass'].std(),
            'mean_entropy': group['entropy'].mean(),
            'std_entropy': group['entropy'].std(),
            'mean_n_tokens': group['n_tokens'].mean(),
            'mean_pred_prob': group['pred_prob'].mean(),
        }
        agg_results.append(agg)
        
        # Print summary
        print(f"\n{model} - {task} - {conf_level}:")
        print(f"  Examples: {agg['n_examples']}")
        print(f"  Top-{top_k} mass: {agg['mean_top_k_mass']:.3f} ± {agg['std_top_k_mass']:.3f}")
        print(f"  Entropy: {agg['mean_entropy']:.3f} ± {agg['std_entropy']:.3f}")
        print(f"  Avg pred prob: {agg['mean_pred_prob']:.3f}")
    
    agg_df = pd.DataFrame(agg_results)
    
    # Statistical comparison
    if 'high' in agg_df['confidence_level'].values and 'borderline' in agg_df['confidence_level'].values:
        print("HIGH vs BORDERLINE COMPARISON")

        for (model, task), group in agg_df.groupby(['model', 'task']):
            high = group[group['confidence_level'] == 'high']
            borderline = group[group['confidence_level'] == 'borderline']
            
            if len(high) > 0 and len(borderline) > 0:
                high_mass = high['mean_top_k_mass'].iloc[0]
                borderline_mass = borderline['mean_top_k_mass'].iloc[0]
                high_entropy = high['mean_entropy'].iloc[0]
                borderline_entropy = borderline['mean_entropy'].iloc[0]
                
                print(f"\n{model} - {task}:")
                print(f"  Top-{top_k} mass: high={high_mass:.3f}, borderline={borderline_mass:.3f}, "
                      f"Δ={high_mass - borderline_mass:+.3f}")
                print(f"  Entropy: high={high_entropy:.3f}, borderline={borderline_entropy:.3f}, "
                      f"Δ={high_entropy - borderline_entropy:+.3f}")
                
                if high_mass > borderline_mass:
                    print(f"  High-confidence MORE concentrated (as expected)")
                else:
                    print(f"  Borderline MORE concentrated (unexpected)")
    
    return results_df, agg_df


# MAIN
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ig_results', type=str, required=True, help="Path to IG results CSV from run_integrated_gradients.py")
    parser.add_argument('--metric', type=str, default='concentration', choices=['concentration'], help="Analysis type (currently only 'concentration')")
    parser.add_argument('--output', type=str,help="Output path for results CSV")
    parser.add_argument('--top_k', type=int, default=10, help="Number of top tokens for mass computation")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = EVALUATION_DIR / "experiment3.csv"
    
    ensure_dir(output_path.parent)
    
    # Run analysis
    if args.metric == 'concentration':
        detailed_results, summary_results = run_concentration_analysis(
            args.ig_results,
            top_k=args.top_k
        )
        
        # Save detailed results
        detailed_path = output_path.parent / f"{output_path.stem}_detailed.csv"
        detailed_results.to_csv(detailed_path, index=False)
        print(f"\nSaved detailed results: {detailed_path}")
        
        # Save summary
        summary_results.to_csv(output_path, index=False)
        print(f"Saved summary results: {output_path}")
    
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

if __name__ == "__main__":
    main()