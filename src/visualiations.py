import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)

TASKS_4 = ["Q_overall", "Q2_harmful", "Q3_bias", "Q6_policy"]
TASKS_2 = ["Q_overall", "Q2_harmful"]

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _cols(task: str):
    return f"{task}_true", f"{task}_prob"

def plot_pr_curve(preds_csv: str, task: str, out_dir="results/figures", show=True):
    """
    Saves: PR curve PNG for task + prints AP (PR-AUC).
    Requires columns: {task}_true, {task}_prob
    """
    _ensure_dir(out_dir)
    df = pd.read_csv(preds_csv)

    y_col, p_col = _cols(task)
    y_true = df[y_col].astype(int).values
    y_prob = df[p_col].astype(float).values

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve: {task} (AP={ap:.3f})")

    out_path = os.path.join(out_dir, f"pr_{os.path.basename(preds_csv).replace('.csv','')}_{task}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    print(f"[PR-AUC] {task}: {ap:.4f}")
    return out_path, ap

def plot_confusion(preds_csv: str, task: str, threshold=0.5, out_dir="results/figures", show=True):
    """
    Saves: confusion matrix PNG for task at a threshold.
    Requires columns: {task}_true, {task}_prob
    """
    _ensure_dir(out_dir)
    df = pd.read_csv(preds_csv)

    y_col, p_col = _cols(task)
    y_true = df[y_col].astype(int).values
    y_prob = df[p_col].astype(float).values
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Safe(0)", "NotSafe(1)"])

    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion: {task} (thr={threshold:.2f})")

    out_path = os.path.join(out_dir, f"cm_{os.path.basename(preds_csv).replace('.csv','')}_{task}_thr{threshold:.2f}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return out_path

def save_metrics_row(model_name: str, metrics: dict, out_csv="results/metrics_summary.csv"):
    """
    Appends one row per task into a metrics CSV.
    metrics should look like:
      {"Q2_harmful": {"pr_auc":0.40, "acc":0.74, "f1_pos":0.41, "f1_macro":0.62}, ...}
    """
    _ensure_dir(os.path.dirname(out_csv) or ".")
    rows = []
    for task, m in metrics.items():
        row = {"model": model_name, "task": task}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)

    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(out_csv, index=False)
    return out_csv

def plot_metric_bars(metrics_csv="results/metrics_summary.csv", metric="pr_auc", out_dir="results/figures", show=True):
    """
    Bar chart: metric by task, grouped by model.
    Requires columns: model, task, metric
    """
    _ensure_dir(out_dir)
    df = pd.read_csv(metrics_csv)
    if metric not in df.columns:
        raise ValueError(f"{metrics_csv} has no column '{metric}'")

    models = df["model"].unique().tolist()
    tasks = df["task"].unique().tolist()

    x = np.arange(len(tasks))
    width = 0.8 / max(len(models), 1)

    plt.figure()
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("task")
        vals = [float(sub.loc[t][metric]) if t in sub.index else np.nan for t in tasks]
        plt.bar(x + i*width, vals, width=width, label=model)

    plt.xticks(x + width*(len(models)-1)/2, tasks, rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} by task")
    plt.legend()

    out_path = os.path.join(out_dir, f"{metric}_by_task.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return out_path
