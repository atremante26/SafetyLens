import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score

# Paths to prediction CSVs
LOGREG_PATH = "results/logistic_regression/test_preds.csv"
#SINGLE_TASK_PATH = "results/single_task/test_preds.csv"
MULTI_TASK_2_PATH = "results/multi_task_transformer/test_predictions_2.csv"
MULTI_TASK_4_PATH = "results/multi_task_transformer/test_predictions_4.csv"

# Only evaluating overall safety
TASK = "Q_overall"

def compute_pr_auc(y_true, y_probs):
    """Compute precision-recall AUC for binary classification."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def evaluate_model(df):
    """Compute PR-AUC, F1 (positive class), and positive rate for Q_overall."""
    y_true = df[f"{TASK}_true"].values
    y_prob = df[f"{TASK}_prob"].values

    # Generate predictions from probabilities (threshold 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_true, y_pred, pos_label=1)
    pr_auc = compute_pr_auc(y_true, y_prob)
    pos_rate = y_true.mean()

    return {
        "F1_pos": f1,
        "PR_AUC": pr_auc,
        "positive_rate": pos_rate
    }

def load_and_prepare(path):
    """Load CSV and verify necessary columns exist."""
    df = pd.read_csv(path)
    required_cols = [f"{TASK}_true", f"{TASK}_prob"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")
    return df

def main():
    # Load predictions
    logreg_df = load_and_prepare(LOGREG_PATH)
    # single_df = load_and_prepare(SINGLE_TASK_PATH)
    multi_2_df = load_and_prepare(MULTI_TASK_2_PATH)
    multi_4_df = load_and_prepare(MULTI_TASK_4_PATH)

    # Evaluate each model
    logreg_results = evaluate_model(logreg_df)
    # single_results = evaluate_model(single_df)
    multi_2_results = evaluate_model(multi_2_df)
    multi_4_results = evaluate_model(multi_4_df)

    # Aggregate results
    rows = [
        {"model": "logreg", "task": TASK, **logreg_results},
        # {"model": "single_task", "task": TASK, **single_results},
        {"model": "multi_task_2", "task": TASK, **multi_2_results},
        {"model": "multi_task_4", "task": TASK, **multi_4_results},
    ]

    results_df = pd.DataFrame(rows)
    results_df.to_csv("results/evaluation/model_comparison_q_overall.csv", index=False)
    print("Saved evaluation results to results/evaluation/model_comparison_q_overall.csv\n")


    """
    # --- H1: Transformer superiority ---
    print("H1: Transformer Superiority over Linear Baselines")
    print(f"LR F1={logreg_results['F1_pos']:.3f}, ST F1={single_results['F1_pos']:.3f}, "
          f"MT_2 F1={multi_2_results['F1_pos']:.3f}, MT_4 F1={multi_4_results['F1_pos']:.3f}")
    print(f"LR PR-AUC={logreg_results['PR_AUC']:.3f}, ST PR-AUC={single_results['PR_AUC']:.3f}, "
          f"MT_2 PR-AUC={multi_2_results['PR_AUC']:.3f}, MT_4 PR-AUC={multi_4_results['PR_AUC']:.3f}\n")

    # --- H2: Multi-task benefits ---
    print("H2: Multi-Task Learning Benefits")
    print(f"Single-task F1={single_results['F1_pos']:.3f}, Multi-task 2 F1={multi_2_results['F1_pos']:.3f}, "
          f"Multi-task 4 F1={multi_4_results['F1_pos']:.3f}")
    print(f"Single-task PR-AUC={single_results['PR_AUC']:.3f}, Multi-task 2 PR-AUC={multi_2_results['PR_AUC']:.3f}, "
          f"Multi-task 4 PR-AUC={multi_4_results['PR_AUC']:.3f}\n")

    # --- H3: Class imbalance effect ---
    print("H3: Positive class prevalence for Q_overall")
    print(f"Positive rate={logreg_results['positive_rate']:.3f}")

    """

if __name__ == "__main__":
    main()
