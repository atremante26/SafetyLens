import os
import random
import numpy as np
import pandas as pd
import pickle
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer
from models.single_task_transformer import load_finetuned, label2id, id2label, MODEL_NAME
from models.multi_task_transformer import load_model


def predict_proba_logreg(texts, model, vectorizer):
    """
    texts: List[str] -> np.array shape (N, 2)
    """
    X = vectorizer.transform(texts)            # sparse matrix
    probs = model.predict_proba(X)             # numpy array (N, 2)
    return probs

def explanation_to_rows(exp, row_index, model_name, label_name, label_id, num_features):
    """
    Returns list[dict] where each dict is one (feature, weight) row from LIME.
    """
    rows = []
    for feat, weight in exp.as_list(label=label_id)[:num_features]:
        rows.append({
            "model": model_name,
            "row_index": row_index,
            "explained_label": label_name,
            "explained_label_id": label_id,
            "feature": feat,
            "weight": float(weight),
        })
    return rows

def run_lime_logreg_on_samples(
    df,
    logreg_path,
    vectorizer_path,
    text_col="text",
    label_col="Q_overall_binary",
    n_samples=5,
    seed=67,
    num_features=8,
    out_dir="results/LIME",
):
    np.random.seed(seed)

    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    class_names = ["safe", "unsafe"]
    explainer = LimeTextExplainer(class_names=class_names)

    sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed)

    sample_rows = []
    explanation_rows = []

    for i, row in sample_df.iterrows():
        text = str(row[text_col])
        true_label = row[label_col] if label_col in sample_df.columns else None

        probs = predict_proba_logreg([text], model, vectorizer)[0]
        pred_id = int(np.argmax(probs))

        print("=" * 80)
        print(f"Row index: {i}")
        if true_label is not None:
            print(f"True label: {class_names[int(true_label)]}")
        print(f"Pred label: {class_names[pred_id]} | probs={probs}")

        # Explain UNSAFE class (index 1)
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=lambda xs: predict_proba_logreg(xs, model, vectorizer),
            labels=[1],
            num_features=num_features,
        )

        print("\nTop features pushing toward UNSAFE (positive = more unsafe):")
        for feat, weight in exp.as_list(label=1):
            print(f"{weight:+.4f}\t{feat}")
        
        explanation_rows.extend(
            explanation_to_rows(
                exp,
                row_index=i,
                label_name="unsafe",
                label_id=1,
                num_features=num_features,
            )
        )

        samples_df = pd.DataFrame(sample_rows)
        expl_df = pd.DataFrame(explanation_rows)
        
        samples_df.to_csv("lime_logreg_samples.csv", index=False)
        expl_df.to_csv("lime_logreg_explanations.csv", index=False)



def predict_proba(texts, model, tokenizer, device, max_length=256):
    """
    LIME expects a function that takes a list[str] and returns np.array shape (N, num_classes)
    with probabilities for each class.
    """
    model.eval()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    return probs


def run_lime_on_samples(
    df,
    text_col="text",
    label_col="Q_overall_binary",
    n_samples=5,
    seed=67,
    max_length=128,
):
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model, tokenizer = load_model(model_name=MODEL_NAME, device=device)

    class_names = ['safe', 'unsafe']

    print("Loading explainer...")
    explainer = LimeTextExplainer(class_names=class_names)

    sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed)

    sample_rows = []    
    explanation_rows = [] 

    print("Explaining...")
    for i, row in sample_df.iterrows():
        text = str(row[text_col])
        true_label = row[label_col] if label_col in sample_df.columns else None

        probs = predict_proba([text], model, tokenizer, device, max_length=max_length)[0]
        pred_id = int(np.argmax(probs))
        pred_label = id2label[pred_id]

        print("=" * 80)
        print(f"Row index: {i}")
        if true_label is not None:
            print(f"True label: {id2label[true_label]}")
        print(f"Pred label: {pred_label} | probs={probs}")

        sample_rows.append({
            "model": "roberta",
            "row_index": i,
            "true_label": None if true_label is None else int(true_label),
            "true_label_name": None if true_label is None else class_names[int(true_label)],
            "pred_label": pred_id,
            "pred_label_name": class_names[pred_id],
            "prob_safe": float(probs[label2id["safe"]]),
            "prob_unsafe": float(probs[label2id["unsafe"]]),
            "text": text,
        })

        explanation_rows += explanation_to_rows(
            unsafe_exp, i, "roberta", "unsafe", unsafe_id, num_features,
        )

        unsafe_id = label2id["unsafe"]
        safe_id = label2id["safe"]

        unsafe_exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=lambda xs: predict_proba(xs, model, tokenizer, device, max_length=max_length),
            labels=[unsafe_id],
            num_features=5,
        )



        print("\nTop features pushing toward UNSAFE (positive = more unsafe):")
        for feat, weight in unsafe_exp.as_list(label=unsafe_id):
            print(f"{weight:+.4f}\t{feat}")
        
    samples_df = pd.DataFrame(sample_rows)
    expl_df = pd.DataFrame(explanation_rows)

    samples_df.to_csv('/results/LIME/lime_samples.csv', index=False)
    expl_df.to_csv('/results/LIME/lime_explainations.csv', index=False)

def main():
    data_path = "data/processed/dices_350_binary.csv"
    df = pd.read_csv(data_path)

    run_lime_on_samples(
        df=df,
        text_col="text",
        label_col="Q_overall_binary",
        n_samples=5,
        seed=67,
        max_length=128,
    )

    run_lime_logreg_on_samples(
        df,
        logreg_path="results/models/logistic_regression_model.pkl",
        vectorizer_path="results/models/tfidf_vectorizer.pkl",
        n_samples=5,
    )




if __name__ == "__main__":
    main()