import sys
from pathlib import Path
import argparse

import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import PREDICTIONS_DIR, EVALUATION_DIR, ensure_dir

# Default paths to prediction CSVs
DEFAULT_PATHS = {
    'logreg': PREDICTIONS_DIR / 'logistic_regression' / 'test_preds.csv',
    'single': PREDICTIONS_DIR / 'single_task_transformer' / 'test_preds.csv',
    'multi2': PREDICTIONS_DIR / 'multi_task_transformer' / 'test_preds_2.csv',
    'multi4': PREDICTIONS_DIR / 'multi_task_transformer' / 'test_preds_4.csv',
}

# Task configurations
ALL_TASKS = ['Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy']
SHARED_TASKS = ['Q_overall', 'Q2_harmful']  


# METRICS
def compute_pr_auc(y_true, y_probs):
    """
    Compute precision-recall AUC for binary classification.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


def evaluate_model_on_task(df, task, threshold=0.5):
    """
    Compute metrics for a single task.
    """
    true_col = f"{task}_true"
    prob_col = f"{task}_prob"
    
    # Check if columns exist
    if true_col not in df.columns or prob_col not in df.columns:
        return None
    
    y_true = df[true_col].values
    y_prob = df[prob_col].values

    # Generate predictions from probabilities
    y_pred = (y_prob >= threshold).astype(int)

    # Compute metrics
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    pr_auc = compute_pr_auc(y_true, y_prob)
    pred_pos_rate = y_pred.mean()
    true_pos_rate = y_true.mean()

    return {
        "F1_pos": f1,
        "PR_AUC": pr_auc,
        "pred_pos_rate": pred_pos_rate,
        "true_pos_rate": true_pos_rate,
    }


def load_and_validate(path, tasks):
    """
    Load CSV and check which tasks are available.
    """
    if not Path(path).exists():
        print(f"File not found: {path}")
        return None, []
    
    df = pd.read_csv(path)
    
    # Check which tasks are available
    available_tasks = []
    for task in tasks:
        if f"{task}_true" in df.columns and f"{task}_prob" in df.columns:
            available_tasks.append(task)
    
    return df, available_tasks


# EXPERIMENT 1
def run_experiment_1(paths):
    """
    Experiment 1: Compare all models across all tasks.
    Tests:
    - H1: Transformers > LogReg (per task)
    - H3: Imbalanced tasks (Q3_bias, Q6_policy) have lower performance
    - Partial H2: Multi-task vs single-task comparison
    
    """
    print("\nEXPERIMENT 1: Performance Table Across Tasks")
    
    results = []
    
    # Evaluate each model on each task
    for model_name, path in paths.items():
        df, available_tasks = load_and_validate(path, ALL_TASKS)
        
        if df is None:
            print(f"Skipping {model_name} (file not found)")
            continue
        
        print(f"\n{model_name}:")
        print(f"  Available tasks: {available_tasks}")
        
        for task in available_tasks:
            metrics = evaluate_model_on_task(df, task)
            if metrics:
                results.append({
                    "model": model_name,
                    "task": task,
                    **metrics
                })
                print(f"  {task}: F1={metrics['F1_pos']:.3f}, PR-AUC={metrics['PR_AUC']:.3f}, "
                      f"true_pos_rate={metrics['true_pos_rate']:.3f}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_path = EVALUATION_DIR / "experiment1.csv"
        ensure_dir(EVALUATION_DIR)
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        
        # Print summary
        print("\nSUMMARY - Experiment 1")
        
        # H1: Transformer superiority
        print("\nH1: Transformer Superiority (Q_overall)")
        q_overall = results_df[results_df['task'] == 'Q_overall']
        for _, row in q_overall.iterrows():
            print(f"  {row['model']:15s}: F1={row['F1_pos']:.3f}, PR-AUC={row['PR_AUC']:.3f}")
        
        # H3: Imbalanced task performance
        print("\nH3: Performance on Imbalanced Tasks")
        for task in ALL_TASKS:
            task_data = results_df[results_df['task'] == task]
            if len(task_data) > 0:
                avg_f1 = task_data['F1_pos'].mean()
                avg_pos_rate = task_data['true_pos_rate'].mean()
                print(f"  {task:15s}: avg_F1={avg_f1:.3f}, pos_rate={avg_pos_rate:.3f}")
        
        return results_df
    else:
        print("\nNo results generated")
        return None

# EXPERIMENT 2
def run_experiment_2(paths):
    """
    Experiment 2: 2-head vs 4-head ablation on shared tasks.
    
    Tests:
    - H2: Does multi-task learning help or hurt?
    - H3: Does adding imbalanced tasks (Q3_bias, Q6_policy) cause negative transfer?
    
    Compare Multi-Task-2 vs Multi-Task-4 on Q_overall and Q2_harmful only.
    """
    print("\nEXPERIMENT 2: 2-Head vs 4-Head Ablation")
    
    # Only compare multi-task models
    models_to_compare = {
        'multi_task_2': paths.get('multi2'),
        'multi_task_4': paths.get('multi4'),
    }
    
    results = []
    
    for model_name, path in models_to_compare.items():
        if path is None:
            print(f"   {model_name} path not provided")
            continue
            
        df, available_tasks = load_and_validate(path, SHARED_TASKS)
        
        if df is None:
            print(f"Skipping {model_name} (file not found)")
            continue
        
        print(f"\n{model_name}:")
        
        for task in available_tasks:
            metrics = evaluate_model_on_task(df, task)
            if metrics:
                results.append({
                    "model": model_name,
                    "task": task,
                    **metrics
                })
                print(f"  {task}: F1={metrics['F1_pos']:.3f}, PR-AUC={metrics['PR_AUC']:.3f}, "
                      f"pred_pos_rate={metrics['pred_pos_rate']:.3f}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_path = EVALUATION_DIR / "experiment2.csv"
        ensure_dir(EVALUATION_DIR)
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        
        # Print comparison
        print("\nSUMMARY - Experiment 2: Does Adding Imbalanced Tasks Hurt?")
        
        for task in SHARED_TASKS:
            print(f"\n{task}:")
            task_data = results_df[results_df['task'] == task]
            
            if len(task_data) == 2:
                mt2 = task_data[task_data['model'] == 'multi_task_2'].iloc[0]
                mt4 = task_data[task_data['model'] == 'multi_task_4'].iloc[0]
                
                f1_diff = mt2['F1_pos'] - mt4['F1_pos']
                pr_diff = mt2['PR_AUC'] - mt4['PR_AUC']
                
                print(f"  2-head: F1={mt2['F1_pos']:.3f}, PR-AUC={mt2['PR_AUC']:.3f}")
                print(f"  4-head: F1={mt4['F1_pos']:.3f}, PR-AUC={mt4['PR_AUC']:.3f}")
                print(f"  Δ (2-head - 4-head): F1={f1_diff:+.3f}, PR-AUC={pr_diff:+.3f}")
                
                if f1_diff > 0:
                    print(f"  RESULT: 2-head performs BETTER (negative transfer from imbalanced tasks)")
                else:
                    print(f"  RESULT: 4-head performs BETTER (multi-task helps)")
        
        return results_df
    else:
        print("\nNo results generated")
        return None


# MAIN
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--experiment',
        type=int,
        choices=[1, 2],
        required=True,
        help="Which experiment to run: 1 (all tasks) or 2 (ablation)"
    )
    # Optional custom paths
    parser.add_argument('--logreg', type=str, help="Path to logistic regression predictions")
    parser.add_argument('--single', type=str, help="Path to single-task predictions")
    parser.add_argument('--multi2', type=str, help="Path to multi-task 2-head predictions")
    parser.add_argument('--multi4', type=str, help="Path to multi-task 4-head predictions")
    
    args = parser.parse_args()
    
    # Use custom paths if provided, otherwise use defaults
    paths = {
        'logreg': Path(args.logreg) if args.logreg else DEFAULT_PATHS['logreg'],
        'single_task': Path(args.single) if args.single else DEFAULT_PATHS['single'],
        'multi2': Path(args.multi2) if args.multi2 else DEFAULT_PATHS['multi2'],
        'multi4': Path(args.multi4) if args.multi4 else DEFAULT_PATHS['multi4'],
    }
    
    # Run selected experiment
    if args.experiment == 1:
        run_experiment_1(paths)
    elif args.experiment == 2:
        run_experiment_2(paths)


if __name__ == "__main__":
    main()