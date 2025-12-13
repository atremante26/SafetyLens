import pickle
import shap
import torch
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Configuration
LOGREG_PATH = Path("results/models/logistic_regression_model.pkl")
VECTORIZER_PATH = Path("results/models/tfidf_vectorizer.pkl")
OUTPUT_DIR = Path("results/explainability/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def predict_single_task(texts, model, tokenizer, device="cpu"):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()


def predict_multi_task(texts, model, tokenizer, task_idx=0, device="cpu"):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, task_idx]
        probs = torch.sigmoid(logits)

    return probs.unsqueeze(1).cpu().numpy()


# LOGISTIC REGRESSION
# -------------------

def explain_logreg (
        LOGREG_PATH = LOGREG_PATH,
        VECTORIZER_PATH = VECTORIZER_PATH
):
    
    # Load model + vectorizer
    with open(LOGREG_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()


    # Sample texts for testing
    texts = [
        "you are a worthless piece of trash",
        "I strongly disagree with you but respect your opinion"
    ]

    X = vectorizer.transform(texts)

    # SHAP explainer
    explainer = shap.LinearExplainer(model, X, feature_names=feature_names)
    shap_values = explainer(X)

    for i, text in enumerate(texts):
        print(f"\nText {i}: {text}")
        shap.plots.bar(shap_values[i], max_display=15)


# SINGLE-TASK TRANSFORMER MODEL
# -----------------------------

def shap_single_task_transformer():
    model_path = "results/models/single_task_q_overall.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    explainer = shap.Explainer(
        lambda x: predict_single_task(x, model, tokenizer),
        shap.maskers.Text(tokenizer)
    )

    texts = [
        "you are a worthless piece of trash",
        "I disagree with you but this is not hateful"
    ]

    shap_values = explainer(texts)

    shap.plots.text(shap_values[0])


# MULTI-TASK TRANSFORMER MODEL
# ----------------------------

def shap_multi_task_transformer():
    model_path = "results/models/multi_task.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    explainer = shap.Explainer(
        lambda x: predict_multi_task(x, model, tokenizer, task_idx=0),
        shap.maskers.Text(tokenizer)
    )

    texts = [
        "you people are disgusting",
    ]

    shap_values = explainer(texts)

    shap.plots.text(shap_values[0])

if __name__ == "__main__":
    explain_logreg(LOGREG_PATH, VECTORIZER_PATH)