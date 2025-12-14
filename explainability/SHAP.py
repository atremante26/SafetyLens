import pickle
import shap
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# LOGISTIC REGRESSION
def run_shap_logistic_regression(
    df,
    text_col,
    model_path,
    vectorizer_path,
    out_path, 
    max_features=20,
):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    # Load model + vectorizer
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()
    X = vectorizer.transform(df[text_col].astype(str))

    explainer = shap.LinearExplainer(model, X, feature_names=feature_names)
    shap_values = explainer(X).values  # NumPy array shape: (n_rows, n_features)

    rows = []
    for i in range(X.shape[0]):
        sv = shap_values[i]
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
    print(f"Saved logistic regression SHAP results to {out_path}")

# SINGLE TASK TRANSFORMER
def run_shap_single_task_transformer(
    df,
    text_col,
    model_path,
    out_path,
    max_tokens=20,
    device="cpu",
):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    # Convert device string to torch.device
    device = torch.device(device)

    # Load checkpoint to get model_name
    ckpt = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and 'roberta.embeddings.word_embeddings.weight' in ckpt:
        # Direct state_dict
        state_dict = ckpt
        model_name = "roberta-base"
    else:
        # Wrapped format
        model_name = ckpt.get("model_name", "roberta-base")
        state_dict = ckpt.get("model_state_dict", ckpt)
    
    # Load tokenizer from base model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load single-task model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "safe", 1: "unsafe"},
        label2id={"safe": 0, "unsafe": 1}
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Running SHAP for single-task model...")

    # Define prediction function for SHAP that handles both strings and lists
    def predict_for_shap(texts):
        # SHAP may pass either strings or lists of strings
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # Filter out any None or non-string values
        texts = [str(t) if t is not None else "" for t in texts]
        
        # Tokenize
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            logits = model(**enc).logits  
            probs = torch.softmax(logits, dim=-1)  
        
        # Return probability of "unsafe" class (index 1)
        return probs[:, 1].cpu().numpy().reshape(-1, 1)

    # Create SHAP explainer
    explainer = shap.Explainer(
        predict_for_shap,
        shap.maskers.Text(tokenizer),
        output_names=["unsafe_probability"]
    )

    rows = []
    print(f"Computing SHAP values for {len(df)} examples...")
    
    for i, text in enumerate(df[text_col].astype(str)):
        if i % 5 == 0:
            print(f"  Processing example {i}/{len(df)}...")
        
        try:
            # Compute SHAP values for this text
            sv = explainer([text])
            
            # Extract tokens and values
            if hasattr(sv, '__getitem__'):
                explanation = sv[0]
                tokens = explanation.data
                values = explanation.values
            else:
                tokens = sv.data
                values = sv.values
            
            # Get top tokens by absolute SHAP value
            top_idx = np.argsort(np.abs(values))[::-1][:max_tokens]

            for j in top_idx:
                rows.append({
                    "model": "single_task_transformer",
                    "row_index": i,
                    "token": tokens[j],
                    "shap_value": float(values[j]),
                    "abs_shap_value": float(abs(values[j])),
                })
        
        except Exception as e:
            print(f"  Warning: Failed to compute SHAP for example {i}: {e}")
            continue

    # Save results
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    print(f"  Saved single-task transformer SHAP results to {out_path}")
    print(f"  Total rows saved: {len(result_df)}")

# MULTI TASK TRANSFORMER
def run_shap_multi_task_transformer(
    df,
    text_col,
    model_path,
    task_idx,
    out_path,
    max_tokens=20,
    device="cpu",
):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    # Convert device string to torch.device
    device = torch.device(device)

    # Load checkpoint to get model_name
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt.get("model_name", "roberta-base")
    
    # Load tokenizer from the base model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the multi-task model architecture
    from models import MultiTaskRoBERTa
    model = MultiTaskRoBERTa(
        model_name=model_name,
        tasks=ckpt["tasks"]
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get task name from index
    task_name = ckpt["tasks"][task_idx]
    print(f"Running SHAP for task: {task_name} (index {task_idx})")
    
    # Define prediction function for SHAP that handles both strings and lists
    def predict_for_shap(texts):
        # SHAP may pass either strings or lists of strings
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # Filter out any None or non-string values
        texts = [str(t) if t is not None else "" for t in texts]
        
        # Tokenize
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(enc["input_ids"], enc["attention_mask"])
            logits = outputs[task_name]  
            probs = torch.sigmoid(logits).squeeze(-1)  
        
        return probs.cpu().numpy().reshape(-1, 1)

    # Create SHAP explainer with text masker
    explainer = shap.Explainer(
        predict_for_shap,
        shap.maskers.Text(tokenizer),
        output_names=["unsafe_probability"]
    )

    rows = []
    print(f"Computing SHAP values for {len(df)} examples...")
    
    for i, text in enumerate(df[text_col].astype(str)):
        if i % 5 == 0:
            print(f"  Processing example {i}/{len(df)}...")
        
        try:
            # Compute SHAP values for this text
            sv = explainer([text])
            
            # Extract tokens and values
            if hasattr(sv, '__getitem__'):
                explanation = sv[0]
                tokens = explanation.data
                values = explanation.values
            else:
                tokens = sv.data
                values = sv.values
            
            # Get top tokens by absolute SHAP value
            top_idx = np.argsort(np.abs(values))[::-1][:max_tokens]

            for j in top_idx:
                rows.append({
                    "model": "multi_task_transformer",
                    "task_idx": task_idx,
                    "task_name": task_name,
                    "row_index": i,
                    "token": tokens[j],
                    "shap_value": float(values[j]),
                    "abs_shap_value": float(abs(values[j])),
                })
        
        except Exception as e:
            print(f"  Warning: Failed to compute SHAP for example {i}: {e}")
            continue

    # Save results
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False)
    print(f"  Saved multi-task transformer SHAP results to {out_path}")
    print(f"  Total rows saved: {len(result_df)}")