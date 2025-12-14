import pickle
import shap
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from models import MultiTaskRoBERTa

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
def predict_multi_task(texts, model, tokenizer, task_name, device):
    # texts is a list[str]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(enc["input_ids"], enc["attention_mask"])  # MultiTaskRoBERTa forward
        logits = outputs[task_name].squeeze(-1)  # [B]
        probs = torch.sigmoid(logits)            # [B]

    return probs.detach().cpu().numpy().reshape(-1, 1)


def run_shap_multi_task_transformer(
    df,
    text_col,
    model_path,   # this is your ckpt path
    task_name,
    out_path,
    max_tokens=20,
    device="cpu",
    n_examples=25,
):
    """
    Run SHAP for MultiTaskRoBERTa checkpoint on a specified task head.
    """

    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    print("\n=== DEBUG: loading multitask checkpoint ===")
    print("ckpt path:", model_path)
    ckpt = torch.load(model_path, map_location=device)
    print("ckpt keys:", list(ckpt.keys()))
    print("ckpt model_name:", ckpt.get("model_name"))
    print("ckpt tasks:", ckpt.get("tasks"))

    model_name = ckpt.get("model_name", "roberta-base")
    tasks = ckpt.get("tasks", None)
    if tasks is None:
        raise ValueError("Checkpoint missing 'tasks' key — not a multitask checkpoint?")

    if task_name not in tasks:
        raise ValueError(f"task_name={task_name} not in ckpt tasks={tasks}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultiTaskRoBERTa(model_name=model_name, tasks=tasks).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\n=== DEBUG: sampling texts ===")
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    df_small = df.sample(n=min(n_examples, len(df)), random_state=42).reset_index(drop=True)
    print("n_examples:", len(df_small))
    print("sample text:", df_small.loc[0, text_col][:120])

    # Quick smoke test: can we get a probability?
    print("\n=== DEBUG: forward smoke test ===")
    p = predict_multi_task([df_small.loc[0, text_col]], model, tokenizer, task_name, device)
    print("pred prob shape:", p.shape, "value:", float(p[0, 0]))

    # SHAP explainer
    explainer = shap.Explainer(
        lambda x: predict_multi_task(x, model, tokenizer, task_name, device),
        shap.maskers.Text(tokenizer),
    )

    rows = []
    for i, text in enumerate(df_small[text_col].astype(str).tolist()):
        sv = explainer([text])[0]
        tokens = sv.data
        values = sv.values

        # values can come back as shape (num_tokens, 1) sometimes
        values = np.array(values).squeeze()

        top_idx = np.argsort(np.abs(values))[::-1][:max_tokens]
        for j in top_idx:
            rows.append({
                "model": "multi_task_transformer",
                "task": task_name,
                "row_index": i,
                "token": tokens[j],
                "shap_value": float(values[j]),
                "abs_shap_value": float(abs(values[j])),
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved SHAP:", str(out_path))