from typing import List, Dict, Any, Callable
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

# PREDICTION FUNCTIONS
def predict_proba_logreg(
    texts, 
    model, 
    vectorizer
):
    """
    Predict class probabilities using logistic regression model.
    """
    # Transform texts to TF-IDF features
    X = vectorizer.transform(texts)
    
    # Get probability predictions from logistic regression
    probs = model.predict_proba(X)
    
    return probs


def predict_proba_singletask(
    texts, 
    model, 
    tokenizer, 
    device, 
    max_length=256
):
    """
    Predict class probabilities using single-task RoBERTa model.
    """
    model.eval()

    # Tokenize all texts in a single batch
    enc = tokenizer(
        texts,
        padding=True,           # Pad shorter sequences to match longest in batch
        truncation=True,        # Truncate sequences longer than max_length
        max_length=max_length,
        return_tensors="pt",    # Return PyTorch tensors
    )

    # Move all tensors to model's device
    enc = {k: v.to(device) for k, v in enc.items()}

    # Run inference without gradient computation
    with torch.no_grad():
        # Get logits from model
        logits = model(**enc).logits
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    return probs


def predict_proba_multitask(
    texts, 
    model, 
    tokenizer, 
    device, 
    task="Q_overall", 
    max_length=256
):
    """
    Predict class probabilities using multi-task RoBERTa model.
    """
    model.eval()

    # Tokenize all texts in a single batch
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Move tensors to model's device
    enc = {k: v.to(device) for k, v in enc.items()}

    # Run inference without gradient computation
    with torch.no_grad():
        # Multi-task model returns dict of task outputs
        out = model(**enc)
        
        # Extract logits for the specified task and remove extra dimensions
        logit = out[task].squeeze(-1)

        # Convert logits to probabilities
        p_unsafe = torch.sigmoid(logit)
        p_safe = 1.0 - p_unsafe
        
        # Stack into [P(safe), P(unsafe)] format expected by LIME
        probs = torch.stack([p_safe, p_unsafe], dim=1)

    return probs.cpu().numpy()


# LIME EXPLAINER
def create_lime_explainer(class_names):
    """
    Create a LIME text explainer with specified class names.
    """
    if class_names is None:
        class_names = ['safe', 'unsafe']
    
    return LimeTextExplainer(class_names=class_names)


# EXPLANATION PROCESSING
def explanation_to_rows(
    exp,
    row_index,
    model_name,
    label_name,
    label_id,
    num_features
):
    """
    Convert LIME explanation to list of dictionaries for DataFrame storage.
    
    This function extracts the top feature attributions from a LIME explanation
    and formats them as rows suitable for pandas DataFrame creation.
    """
    rows = []
    
    # Extract top features and their weights
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

# CLASSIFIER
def create_classifier(
    model_type,
    model,
    tokenizer,
    device,
    vectorizer=None,
    task="Q_overall",
    max_length=256
):
    """
    Create a classifier function for LIME based on model type.
    """
    if model_type == 'logreg':
        if vectorizer is None:
            raise ValueError("vectorizer required for 'logreg' model_type")
        return lambda texts: predict_proba_logreg(texts, model, vectorizer)
    
    elif model_type == 'single_task':
        if tokenizer is None or device is None:
            raise ValueError("tokenizer and device required for 'single_task' model_type")
        return lambda texts: predict_proba_singletask(texts, model, tokenizer, device, max_length)
    
    elif model_type == 'multi_task':
        if tokenizer is None or device is None:
            raise ValueError("tokenizer and device required for 'multi_task' model_type")
        return lambda texts: predict_proba_multitask(texts, model, tokenizer, device, task, max_length)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'logreg', 'single_task', or 'multi_task'")