import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import entropy as scipy_entropy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import EVALUATION_DIR, EXPLAINABILITY_DIR, ensure_dir

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Output directory
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
ensure_dir(FIGURES_DIR)


# EXPERIMENT 1
def plot_experiment1():
    """
    Create visualizations for Experiment 1.
    """
    # Load data
    exp1_path = EVALUATION_DIR / "experiment1.csv"
    if not exp1_path.exists():
        print(f"  Experiment 1 results not found: {exp1_path}")
        return
    
    df = pd.read_csv(exp1_path)
    
    # FIGURE 1: F1 SCORES BY TASK
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    tasks = df['task'].unique()
    models = df['model'].unique()
    
    x = np.arange(len(tasks))
    width = 0.2
    
    colors = ['#6B2C91', '#8B5BAE', '#004B73', '#4FC3F7'] 
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        f1_scores = [
            model_data[model_data['task'] == task]['F1_pos'].values[0] 
            if task in model_data['task'].values else 0 
            for task in tasks
        ]
        
        ax.bar(x + i * width, f1_scores, width, 
               label=model.replace('_', ' ').title(), 
               color=colors[i % len(colors)])
    
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('F1 Score (Positive Class)', fontweight='bold')
    ax.set_title('Experiment 1: Model Performance Across Safety Tasks', fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tasks)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment1/exp1_f1_by_task.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 2: PERFORMANCE HEATMAP
    pivot_f1 = df.pivot(index='model', columns='task', values='F1_pos')
    pivot_prauc = df.pivot(index='model', columns='task', values='PR_AUC')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 heatmap
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='Purples', 
                vmin=0.2, vmax=0.7, ax=axes[0], cbar_kws={'label': 'F1 Score'})
    axes[0].set_title('F1 Score by Model and Task', fontweight='bold')
    axes[0].set_xlabel('Task', fontweight='bold')
    axes[0].set_ylabel('Model', fontweight='bold')
    
    # PR-AUC heatmap
    sns.heatmap(pivot_prauc, annot=True, fmt='.3f', cmap='Purples', 
                vmin=0.1, vmax=0.7, ax=axes[1], cbar_kws={'label': 'PR-AUC'})
    axes[1].set_title('PR-AUC by Model and Task', fontweight='bold')
    axes[1].set_xlabel('Task', fontweight='bold')
    axes[1].set_ylabel('Model', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment1/exp1_performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 3: PERFORMANCE VS. CLASS IMBALANCE
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Group by task and compute mean F1 and positive rate
    task_summary = df.groupby('task').agg({
        'F1_pos': 'mean',
        'true_pos_rate': 'mean'
    }).reset_index()
    
    # Sort by positive rate
    task_summary = task_summary.sort_values('true_pos_rate', ascending=False)
    
    # Plot
    ax.scatter(task_summary['true_pos_rate'] * 100, 
           task_summary['F1_pos'], 
           s=200, alpha=0.7, color='#6B2C91', edgecolors='black', linewidth=2)
    
    # Annotate points
    for _, row in task_summary.iterrows():
        ax.annotate(row['task'], 
                   (row['true_pos_rate'] * 100, row['F1_pos']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Class Imbalance (% Unsafe Examples)', fontweight='bold')
    ax.set_ylabel('Average F1 Score', fontweight='bold')
    ax.set_title('Experiment 1: Performance Degrades with Severe Class Imbalance', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment1/exp1_imbalance_impact.png", dpi=300, bbox_inches='tight')
    plt.close()

# EXPERIMENT 2
def plot_experiment2():
    """
    Create visualizations for Experiment 2.
    """
    # Load data
    exp2_path = EVALUATION_DIR / "experiment2.csv"
    if not exp2_path.exists():
        print(f"  Experiment 2 results not found: {exp2_path}")
        return
    
    df = pd.read_csv(exp2_path)
    
    # FIGURE 1: SIDE BY SIDE COMPARISON (2-HEAD VS. 4-HEAD) 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    tasks = df['task'].unique()
    x = np.arange(len(tasks))
    width = 0.35
    
    # F1 comparison
    mt2_f1 = [df[(df['model'] == 'multi_task_2') & (df['task'] == t)]['F1_pos'].values[0] for t in tasks]
    mt4_f1 = [df[(df['model'] == 'multi_task_4') & (df['task'] == t)]['F1_pos'].values[0] for t in tasks]
    
    axes[0].bar(x - width/2, mt2_f1, width, label='2-Head', color='#6B2C91', alpha=0.8)
    axes[0].bar(x + width/2, mt4_f1, width, label='4-Head', color='#8B5BAE', alpha=0.8)
    axes[0].set_xlabel('Task', fontweight='bold')
    axes[0].set_ylabel('F1 Score', fontweight='bold')
    axes[0].set_title('F1 Score Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks)
    axes[0].legend()
    axes[0].set_ylim(0.3, 0.5)
    
    # PR-AUC comparison
    mt2_prauc = [df[(df['model'] == 'multi_task_2') & (df['task'] == t)]['PR_AUC'].values[0] for t in tasks]
    mt4_prauc = [df[(df['model'] == 'multi_task_4') & (df['task'] == t)]['PR_AUC'].values[0] for t in tasks]
    
    axes[1].bar(x - width/2, mt2_prauc, width, label='2-Head', color='#6B2C91', alpha=0.8)
    axes[1].bar(x + width/2, mt4_prauc, width, label='4-Head', color='#8B5BAE', alpha=0.8)
    axes[1].set_xlabel('Task', fontweight='bold')
    axes[1].set_ylabel('PR-AUC', fontweight='bold')
    axes[1].set_title('PR-AUC Comparison', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks)
    axes[1].legend()
    axes[1].set_ylim(0.3, 0.5)
    
    plt.suptitle('Experiment 2: 2-Head vs 4-Head Multi-Task Models', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment2/exp2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 2: DELTA PLOT
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute deltas (2-head - 4-head)
    deltas = []
    for task in tasks:
        mt2_f1_val = df[(df['model'] == 'multi_task_2') & (df['task'] == task)]['F1_pos'].values[0]
        mt4_f1_val = df[(df['model'] == 'multi_task_4') & (df['task'] == task)]['F1_pos'].values[0]
        delta = mt2_f1_val - mt4_f1_val
        deltas.append(delta)
    
    colors = ['#6B2C91' if d > 0 else '#916B2C' for d in deltas]  
    
    ax.barh(tasks, deltas, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Δ F1 Score (2-Head - 4-Head)', fontweight='bold')
    ax.set_ylabel('Task', fontweight='bold')
    ax.set_title('Experiment 2: Evidence of Negative Transfer from Imbalanced Tasks', 
                 fontweight='bold', pad=20)
    
    # Add value labels
    for i, (task, delta) in enumerate(zip(tasks, deltas)):
        ax.text(delta + 0.001 if delta > 0 else delta - 0.001, i, 
               f'{delta:+.3f}', 
               ha='left' if delta > 0 else 'right', 
               va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment2/exp2_delta_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

# EXPERIMENT 3
def plot_experiment3():
    """
    Create visualizations for Experiment 3.
    """
    # Load raw IG data
    single_path = EVALUATION_DIR / "experiment3_single.csv"
    multi_path = EVALUATION_DIR / "experiment3_multi4.csv"
    
    # Alternative paths (if files are in different location)
    if not single_path.exists():
        single_path = EVALUATION_DIR / "experiment3_single.csv"
    if not multi_path.exists():
        multi_path = EVALUATION_DIR / "experiment3_multi4.csv"
    
    if not single_path.exists() or not multi_path.exists():
        print(f"  Experiment 3 results not found")
        print(f"Checked: {single_path}")
        print(f"Checked: {multi_path}")
        return
    
    # Load data
    single_df = pd.read_csv(single_path)
    multi_df = pd.read_csv(multi_path)
    
    # Compute metrics per example
    def compute_metrics(df, model_name):
        results = []
        for (ex_idx, conf_level), group in df.groupby(['example_idx', 'confidence_level']):
            attrs = group['abs_attribution'].values
            total = attrs.sum()
            
            # Top-10 mass
            top_10_mass = np.sort(attrs)[-10:].sum() / total if total > 0 else 0
            
            # Entropy
            if total > 0:
                prob_dist = attrs / total
                prob_dist = prob_dist[prob_dist > 0]
                attr_entropy = scipy_entropy(prob_dist)
            else:
                attr_entropy = 0
            
            results.append({
                'model': model_name,
                'example_idx': ex_idx,
                'confidence_level': conf_level,
                'top_10_mass': top_10_mass,
                'entropy': attr_entropy,
                'n_tokens': len(attrs),
                'pred_prob': group['pred_prob'].iloc[0]
            })
        return pd.DataFrame(results)
    
    single_metrics = compute_metrics(single_df, 'Single-Task')
    multi_metrics = compute_metrics(multi_df, 'Multi-Task')
    all_metrics = pd.concat([single_metrics, multi_metrics])
    
    # FIGURE 1: TOP-10 MASS COMPARISON (BAR CHART)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    summary = all_metrics.groupby(['model', 'confidence_level'])['top_10_mass'].mean().reset_index()
    
    models = ['Single-Task', 'Multi-Task']
    conf_levels = ['high', 'borderline']
    x = np.arange(len(models))
    width = 0.35
    
    high_vals = [summary[(summary['model'] == m) & (summary['confidence_level'] == 'high')]['top_10_mass'].values[0] 
                 for m in models]
    border_vals = [summary[(summary['model'] == m) & (summary['confidence_level'] == 'borderline')]['top_10_mass'].values[0] 
                   for m in models]
    
    ax.bar(x - width/2, high_vals, width, label='High Confidence', color='#6B2C91', alpha=0.8, edgecolor='black', linewidth=2)
    ax.bar(x + width/2, border_vals, width, label='Borderline', color='#8B5BAE', alpha=0.8, edgecolor='black', linewidth=2)
    # Add value labels
    for i, (h, b) in enumerate(zip(high_vals, border_vals)):
        ax.text(i - width/2, h + 0.02, f'{h:.3f}', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, b + 0.02, f'{b:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Top-10 Attribution Mass', fontweight='bold', fontsize=12)
    ax.set_title('Experiment 3: Attribution Concentration by Confidence Level', 
                 fontweight='bold', pad=20, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment3/exp3_top10_mass.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 2: ENTROPY COMPARISON (BAR CHART)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    entropy_summary = all_metrics.groupby(['model', 'confidence_level'])['entropy'].mean().reset_index()
    
    high_entropy = [entropy_summary[(entropy_summary['model'] == m) & (entropy_summary['confidence_level'] == 'high')]['entropy'].values[0] 
                    for m in models]
    border_entropy = [entropy_summary[(entropy_summary['model'] == m) & (entropy_summary['confidence_level'] == 'borderline')]['entropy'].values[0] 
                      for m in models]
    
    ax.bar(x - width/2, high_entropy, width, label='High Confidence', color='#6B2C91', alpha=0.8, edgecolor='black', linewidth=2)
    ax.bar(x + width/2, border_entropy, width, label='Borderline', color='#8B5BAE', alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels
    for i, (h, b) in enumerate(zip(high_entropy, border_entropy)):
        ax.text(i - width/2, h + 0.15, f'{h:.2f}', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, b + 0.15, f'{b:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Attribution Entropy', fontweight='bold', fontsize=12)
    ax.set_title('Experiment 3: Attribution Diffuseness by Confidence Level', 
                 fontweight='bold', pad=20, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment3/exp3_entropy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 3: COMBINED HEATMAP
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Top-10 mass heatmap
    mass_pivot = all_metrics.pivot_table(
        index='model', 
        columns='confidence_level', 
        values='top_10_mass', 
        aggfunc='mean'
    )[['high', 'borderline']]  # Order columns
    
    sns.heatmap(mass_pivot, annot=True, fmt='.3f', cmap='Purples', 
                vmin=0.1, vmax=0.5, ax=axes[0], 
                cbar_kws={'label': 'Top-10 Mass'}, 
                linewidths=2, linecolor='black')
    axes[0].set_title('Top-10 Attribution Mass', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Confidence Level', fontweight='bold')
    axes[0].set_ylabel('Model', fontweight='bold')
    
    # Entropy heatmap
    entropy_pivot = all_metrics.pivot_table(
        index='model', 
        columns='confidence_level', 
        values='entropy', 
        aggfunc='mean'
    )[['high', 'borderline']]
    
    sns.heatmap(entropy_pivot, annot=True, fmt='.2f', cmap='Blues', 
                vmin=3, vmax=5.5, ax=axes[1], 
                cbar_kws={'label': 'Entropy'}, 
                linewidths=2, linecolor='black')
    axes[1].set_title('Attribution Entropy', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Confidence Level', fontweight='bold')
    axes[1].set_ylabel('Model', fontweight='bold')
    
    plt.suptitle('Experiment 3: Concentration Metrics Heatmap', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment3/exp3_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 4: TEXT LENGTH VS. CONCENTRATION
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot by model and confidence
    colors = {'high': '#6B2C91', 'borderline': '#8B5BAE'}
    markers = {'Single-Task': 'o', 'Multi-Task': 's'}
    
    for model in models:
        for conf in conf_levels:
            data = all_metrics[(all_metrics['model'] == model) & 
                              (all_metrics['confidence_level'] == conf)]
            ax.scatter(data['n_tokens'], data['top_10_mass'], 
                      s=100, alpha=0.7, 
                      color=colors[conf], 
                      marker=markers[model],
                      label=f'{model} - {conf.capitalize()}',
                      edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Text Length (# Tokens)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Top-10 Attribution Mass', fontweight='bold', fontsize=12)
    ax.set_title('Experiment 3: Text Length vs Attribution Concentration', 
                 fontweight='bold', pad=20, fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add annotation about the pattern
    ax.text(0.05, 0.95, 
           'High-confidence:\nShort texts (60 tokens)\nConcentrated attribution\n\nBorderline:\nLong texts (254 tokens)\nDiffuse attribution',
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiment3/exp3_length_vs_concentration.png", dpi=300, bbox_inches='tight')
    plt.close()



# LIME
def plot_lime():
    """
    Analyze LIME feature attributions across models.
    """
    # Load LIME explanations
    models = ['logreg', 'single_task', 'multi_task']
    lime_data = []
    
    for model in models:
        path = EXPLAINABILITY_DIR / "lime" / f"{model}_lime_explanations.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['model_name'] = model
            lime_data.append(df)
    
    if not lime_data:
        print("  No LIME results found")
        return
    
    lime_df = pd.concat(lime_data, ignore_index=True)
    
    # Define function words (simplified heuristic)
    # In reality, use POS tagging, but for visualization this works
    function_words = {
        'not', 'don', 'I', 'you', 'a', 'the', 'is', 'are', 'was', 'were',
        'm', 't', 'But', 'on', 'that', 'this', 'it', 'to', 'of', 'in'
    }
    
    lime_df['is_content_word'] = ~lime_df['feature'].isin(function_words)
    
    # FIGURE 1: CONTENT WORD PERCENTAGE BY MODEL
    fig, ax = plt.subplots(figsize=(8, 6))
    
    content_pct = lime_df.groupby('model_name')['is_content_word'].apply(
        lambda x: (x.sum() / len(x)) * 100
    ).reset_index()
    content_pct.columns = ['model', 'content_pct']
    
    model_labels = {
        'logreg': 'Logistic\nRegression',
        'single_task': 'Single-Task\nTransformer',
        'multi_task': 'Multi-Task\nTransformer'
    }
    content_pct['model_label'] = content_pct['model'].map(model_labels)
    
    colors = ['#6B2C91', '#8B5BAE', '#4A1F66'] 
    bars = ax.bar(content_pct['model_label'], content_pct['content_pct'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Content Words (%)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('LIME: Feature Type Distribution Across Models', fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars, content_pct['content_pct']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lime/lime_content_word_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 2: ATTRIBUTION STRENGTH DISTRIBUTION
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, model in enumerate(models):
        model_data = lime_df[lime_df['model_name'] == model]
        
        axes[i].hist(model_data[model_data['is_content_word']]['weight'].abs(), 
            bins=20, alpha=0.7, color='#8B5BAE', label='Content Words', edgecolor='black')
        axes[i].hist(model_data[~model_data['is_content_word']]['weight'].abs(), 
            bins=20, alpha=0.7, color='#004B73', label='Function Words', edgecolor='black')
        
        axes[i].set_xlabel('|Attribution Weight|', fontweight='bold')
        axes[i].set_ylabel('Frequency', fontweight='bold')
        axes[i].set_title(model_labels[model], fontweight='bold')
        axes[i].legend()
    
    plt.suptitle('LIME: Attribution Strength by Feature Type', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lime/lime_attribution_strength.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# Main
# =============================================================================

def main():
    print("Creating Experiment 1 visualizations...")
    plot_experiment1()
    
    print("\nCreating Experiment 2 visualizations...")
    plot_experiment2()
    
    print("\nCreating LIME visualizations...")
    plot_lime()
    
    print("\nCreating Experiment 3 visualizations...")
    plot_experiment3()
    
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

'''
python src/visualizations.py
'''