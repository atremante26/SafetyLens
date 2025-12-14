import pickle
import shap
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Configuration
LOGREG_PATH = Path("results/models/logistic_regression_model.pkl")
VECTORIZER_PATH = Path("results/models/tfidf_vectorizer.pkl")
OUTPUT_DIR = Path("results/explainability/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

    
# -------------------
# LOGISTIC REGRESSION
# -------------------
def run_shap_logistic_regression(
    df,
    text_col,
    model_path,
    vectorizer_path,
    out_path,
    max_features=20,
):
    """
    Run SHAP for a TF-IDF + logistic regression model over the full dataset.
    Saves per-feature SHAP weights for each row.
    """

    ensure_dir(out_path.parent)

    # Load model + vectorizer
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()

    X = vectorizer.transform(df[text_col].astype(str))

    explainer = shap.LinearExplainer(model, X, feature_names=feature_names)
    shap_values = explainer(X)

    rows = []

    for i in range(X.shape[0]):
        sv = shap_values[i].values
        top_idx = np.argsort(np.abs(sv))[::-1][:max_features]

        for j in top_idx:
            rows.append({
                "model": "logistic_regression",
                "row_index": i,
                "feature": feature_names[j],
                "shap_value": float(sv[j]),
                "abs_shap_value": float(abs(sv[j])),
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)


# -----------------------
# SINGLE TASK TRANSFORMER
# -----------------------
def predict_single_task(texts, model, tokenizer, device):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

    return probs[:, 1].cpu().numpy().reshape(-1, 1)


def run_shap_single_task_transformer(
    df,
    text_col,
    model_path,
    out_path,
    max_tokens=20,
    device="cpu",
):
    """
    Run SHAP for a single-task transformer and save token-level attributions.
    """

    ensure_dir(out_path.parent)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    explainer = shap.Explainer(
        lambda x: predict_single_task(x, model, tokenizer, device),
        shap.maskers.Text(tokenizer),
    )

    rows = []

    for i, text in enumerate(df[text_col].astype(str)):
        sv = explainer([text])[0]

        tokens = sv.data
        values = sv.values

        top_idx = np.argsort(np.abs(values))[::-1][:max_tokens]

        for j in top_idx:
            rows.append({
                "model": "single_task_transformer",
                "row_index": i,
                "token": tokens[j],
                "shap_value": float(values[j]),
                "abs_shap_value": float(abs(values[j])),
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)


# ----------------------
# MULTI TASK TRANSFORMER
# ----------------------
def predict_multi_task(texts, model, tokenizer, task_idx, device):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits[:, task_idx]
        probs = torch.sigmoid(logits)

    return probs.cpu().numpy().reshape(-1, 1)


def run_shap_multi_task_transformer(
    df,
    text_col,
    model_path,
    task_idx,
    out_path,
    max_tokens=20,
    device="cpu",
):
    """
    Run SHAP for a multi-task transformer on a specified task head.
    """

    ensure_dir(out_path.parent)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    explainer = shap.Explainer(
        lambda x: predict_multi_task(x, model, tokenizer, task_idx, device),
        shap.maskers.Text(tokenizer),
    )

    rows = []

    for i, text in enumerate(df[text_col].astype(str)):
        sv = explainer([text])[0]

        tokens = sv.data
        values = sv.values

        top_idx = np.argsort(np.abs(values))[::-1][:max_tokens]

        for j in top_idx:
            rows.append({
                "model": "multi_task_transformer",
                "task_idx": task_idx,
                "row_index": i,
                "token": tokens[j],
                "shap_value": float(values[j]),
                "abs_shap_value": float(abs(values[j])),
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)
