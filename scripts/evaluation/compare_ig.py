import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_2task", required=True, help="Path to 2-task IG results CSV")
    parser.add_argument("--results_4task", required=True, help="Path to 4-task IG results CSV")
    parser.add_argument("--results_single", required=True, help="Path to single-task IG results CSV")
    parser.add_argument("--output", required=True,help="Path to output analysis report (txt file)")
    return parser.parse_args()


def calculate_metrics(df, model_name):
    """
    Calculate comprehensive metrics for a single model's IG results
    """
    # Prediction quality
    tp = ((df['pred'] == 1) & (df['true_label'] == 1)).sum()
    pred_pos = (df['pred'] == 1).sum()
    actual_pos = (df['true_label'] == 1).sum()
    
    precision = tp / pred_pos if pred_pos > 0 else 0
    recall = tp / actual_pos if actual_pos > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Special tokens
    special_tokens = ['<s>', '</s>', '<pad>']
    special_count = df['tok_1'].isin(special_tokens).sum()
    
    return {
        'model': model_name,
        'n_examples': len(df),
        'task': df['task'].iloc[0],
        
        # Predictions
        'true_unsafe_count': int(actual_pos),
        'pred_unsafe_count': int(pred_pos),
        'true_unsafe_pct': actual_pos / len(df) * 100,
        'pred_unsafe_pct': pred_pos / len(df) * 100,
        
        # Performance
        'accuracy': (df['pred'] == df['true_label']).mean(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        
        # Confidence distribution
        'mean_prob': df['prob_pos'].mean(),
        'std_prob': df['prob_pos'].std(),
        'min_prob': df['prob_pos'].min(),
        'max_prob': df['prob_pos'].max(),
        'high_conf_count': int((df['prob_pos'] > 0.8).sum()),
        'med_conf_count': int(((df['prob_pos'] >= 0.5) & (df['prob_pos'] <= 0.8)).sum()),
        'low_conf_count': int((df['prob_pos'] < 0.5).sum()),
        'high_conf_pct': (df['prob_pos'] > 0.8).sum() / len(df) * 100,
        'low_conf_pct': (df['prob_pos'] < 0.5).sum() / len(df) * 100,
        
        # IG quality
        'mean_conv_delta': abs(df['convergence_delta']).mean(),
        'median_conv_delta': abs(df['convergence_delta']).median(),
        'max_conv_delta': abs(df['convergence_delta']).max(),
        'good_conv_count': int((abs(df['convergence_delta']) < 0.05).sum()),
        'acceptable_conv_count': int((abs(df['convergence_delta']) < 0.10).sum()),
        'good_conv_pct': (abs(df['convergence_delta']) < 0.05).sum() / len(df) * 100,
        
        # Token quality
        'special_token_count': int(special_count),
        'special_token_pct': special_count / len(df) * 100,
    }


def main():
    args = parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_2task = pd.read_csv(args.results_2task)
    df_4task = pd.read_csv(args.results_4task)
    df_single = pd.read_csv(args.results_single)
    
    # Calculate metrics for each model
    metrics_2task = calculate_metrics(df_2task, "2-Task Multi-Task")
    metrics_4task = calculate_metrics(df_4task, "4-Task Multi-Task")
    metrics_single = calculate_metrics(df_single, "Single-Task")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame([metrics_2task, metrics_4task, metrics_single])
    
    # Save to CSV
    comparison_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()

'''
python -m  scripts.compare_ig \    
    --results_2task results/ig/ig_2task_q2.csv \
    --results_4task results/ig/ig_4task_q2.csv \
    --results_single results/ig/ig_single_q.csv \
    --output results/evaluation/ig_comparison.csv
'''