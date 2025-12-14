import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paths
EVAL_CSV = "results/evaluation/model_comparison_q_overall.csv"
PRED_CSVS = {
    "logreg": "results/logistic_regression/test_preds.csv",
    "single_task": "results/single_task_transformer/test_preds.csv",
    "multi_task_2": "results/multi_task_transformer/test_predictions_2.csv",
    "multi_task_4": "results/multi_task_transformer/test_predictions_4.csv",
}

TASK = "Q_overall"

# -----------------------
# 1. Load aggregated metrics
# -----------------------
eval_df = pd.read_csv(EVAL_CSV)

# -----------------------
# 2. Grouped bar chart: F1 and PR-AUC
# -----------------------
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ["F1_pos", "PR_AUC"]
width = 0.35
x = range(len(eval_df))

ax.bar([i - width/2 for i in x], eval_df["F1_pos"], width=width, label="F1")
ax.bar([i + width/2 for i in x], eval_df["PR_AUC"], width=width, label="PR-AUC")
ax.set_xticks(x)
ax.set_xticklabels(eval_df["model"])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Performance on Q_overall")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/q_overall_metrics_bar.png")
plt.close()

# -----------------------
# 3. Probability distributions
# -----------------------
# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (model_name, csv_path) in enumerate(PRED_CSVS.items()):
    df = pd.read_csv(csv_path)
    # Generate predictions if not already present
    df["pred"] = (df[f"{TASK}_prob"] >= 0.5).astype(int)

    ax = axes[i]
    # Plot distribution of probabilities per true label
    for label in [0, 1]:
        subset = df[df[f"{TASK}_true"] == label]
        sns.kdeplot(
            subset[f"{TASK}_prob"],
            ax=ax,
            label=f"{'unsafe' if label==1 else 'safe'}",
            fill=True,
            alpha=0.3
        )
    ax.set_title(model_name.replace("_", " ").title())
    ax.set_xlabel("Predicted probability of unsafe")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig("results/figures/q_overall_probability_distributions_2x2.png")
plt.close()
print("Saved 2x2 probability distribution plot to results/evaluation/q_overall_probability_distributions_2x2.png")

# -----------------------
# 4. Confusion matrices
# -----------------------
for model_name, csv_path in PRED_CSVS.items():
    df = pd.read_csv(csv_path)
    y_true = df[f"{TASK}_true"].values
    y_pred = (df[f"{TASK}_prob"] >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["safe", "unsafe"], yticklabels=["safe", "unsafe"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.tight_layout()
    plt.savefig(f"results/figures/q_overall_confusion_{model_name}.png")
    plt.close()

print("All visualizations saved to results/figures/")
