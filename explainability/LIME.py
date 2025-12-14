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
import argparse

#https://github.com/marcotcr/lime?tab=readme-ov-file#text-explanation

def predict_proba_multitask(texts, model, tokenizer, device, task="Q_overall", max_length=256):
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
        out = model(**enc)             
        logit = out[task].squeeze(-1) 

        p_unsafe = torch.sigmoid(logit)         
        p_safe = 1.0 - p_unsafe                
        probs = torch.stack([p_safe, p_unsafe], dim=1) 

    return probs.cpu().numpy()

def predict_proba_logreg(texts, model, vectorizer):

    X = vectorizer.transform(texts)           
    probs = model.predict_proba(X)       
    return probs

def explanation_to_rows(exp, row_index, model_name, label_name, label_id, num_features):
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

        e