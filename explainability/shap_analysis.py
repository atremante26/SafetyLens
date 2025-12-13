import pickle
import shap
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

# Configuration
LOGREG_PATH = Path("results/models/logistic_regression.pkl")
VECTORIZER_PATH = Path("results/models/tfidf_vectorizer.pkl")
OUTPUT_DIR = Path("results/explainability/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    explainer = shap.LinearExplainer(
        model,
        X,
        feature_names=feature_names
    )

    shap_values = explainer(X)

    # Visualization (text-based)
    for i, text in enumerate(texts):
        print(f"\nText {i}: {text}")
        shap.plots.text(shap_values[i])