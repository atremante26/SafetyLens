import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--logreg", required=True, help="Path to logistic regression SHAP results CSV")
    parser.add_argument("--single", required=True, help="Path to single-task transformer SHAP results CSV")
    parser.add_argument("--multitask", required=True, help="Path to multi-task transformer SHAP results CSV")
    parser.add_argument("--output", required=True, help="Path to output comparison CSV")
    return parser.parse_args()

def calculate_shap_metrics(df, model_name):
    """
    Calculate comprehensive metrics for a single model's SHAP results.
    """
    # Determine feature column name (differs between models)
    if 'token' in df.columns:
        feature_col = 'token'
    elif 'feature' in df.columns:
        feature_col = 'feature'
    else:
        feature_col = None
    
    # Basic counts
    n_examples = df['row_index'].nunique() if 'row_index' in df.columns else len(df)
    n_attributions = len(df)
    n_unique_features = df[feature_col].nunique() if feature_col else 0
    
    # SHAP value statistics
    mean_abs_shap = df['abs_shap_value'].mean()
    median_abs_shap = df['abs_shap_value'].median()
    max_abs_shap = df['abs_shap_value'].max()
    std_abs_shap = df['abs_shap_value'].std()
    
    # Get top features by total importance
    if feature_col:
        # Sum absolute SHAP values for each feature across all examples
        feature_importance = df.groupby(feature_col)['abs_shap_value'].sum().sort_values(ascending=False)
        
        top_1_feature = feature_importance.index[0] if len(feature_importance) > 0 else None
        top_1_importance = feature_importance.values[0] if len(feature_importance) > 0 else 0
        
        # Get top 5 features
        top_5_features = feature_importance.head(5).index.tolist()
        top_5_importance = feature_importance.head(5).values.tolist()
        
        # Feature diversity: How many features account for 80% of total importance?
        total_importance = feature_importance.sum()
        cumsum = feature_importance.cumsum()
        n_features_80pct = (cumsum <= 0.8 * total_importance).sum() + 1
    else:
        top_1_feature = None
        top_1_importance = 0
        top_5_features = []
        top_5_importance = []
        n_features_80pct = 0
    
    # Average attributions per example
    avg_attributions_per_example = n_attributions / n_examples if n_examples > 0 else 0
    
    return {
        'model': model_name,
        'n_examples': n_examples,
        'n_attributions': n_attributions,
        'n_unique_features': n_unique_features,
        'avg_attributions_per_example': avg_attributions_per_example,
        
        # SHAP value statistics
        'mean_abs_shap': mean_abs_shap,
        'median_abs_shap': median_abs_shap,
        'max_abs_shap': max_abs_shap,
        'std_abs_shap': std_abs_shap,
        
        # Top features
        'top_1_feature': top_1_feature,
        'top_1_importance': top_1_importance,
        'top_5_features': ', '.join(str(f) for f in top_5_features[:5]),
        
        # Feature diversity
        'n_features_80pct': n_features_80pct,
    }


def main():
    args = parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading SHAP results:")
    df_logreg = pd.read_csv(args.logreg)
    df_single = pd.read_csv(args.single)
    df_multitask = pd.read_csv(args.multitask)
    
    print(f"  Logistic Regression: {len(df_logreg)} attributions")
    print(f"  Single-Task: {len(df_single)} attributions")
    print(f"  Multi-Task: {len(df_multitask)} attributions")
    
    # Calculate metrics for each model
    print("\nCalculating metrics:")
    metrics_logreg = calculate_shap_metrics(df_logreg, "Logistic Regression")
    metrics_single = calculate_shap_metrics(df_single, "Single-Task Transformer")
    metrics_multitask = calculate_shap_metrics(df_multitask, "Multi-Task Transformer")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame([metrics_logreg, metrics_single, metrics_multitask])
    
    # Save to CSV
    comparison_df.to_csv(args.output, index=False)
    print(f"\nComparison saved to: {args.output}")

if __name__ == "__main__":
    main()

'''
python -m scripts.compare_shap \
    --logreg results/shap/logreg_shap_q_overall.csv \
    --single results/shap/single_shap_q_overall.csv \
    --multitask results/shap/multi_shap_q2_harmful.csv \
     --output results/evaluation/shap_comparison.csv
'''