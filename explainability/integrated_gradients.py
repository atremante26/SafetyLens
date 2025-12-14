import torch
import numpy as np
from captum.attr import LayerIntegratedGradients

# TOKENIZATION
def prepare_inputs(text, tokenizer, device, max_length=128):
    '''
    Tokenize input text are prepare tensors for the model
    '''
    # Tokenize with padding to max_length and truncation
    enc = tokenizer(
        text,
        truncation=True,        # Cut off text longer than max_length
        padding="max_length",   # Pad to max_length with PAD tokens
        max_length=max_length,
        return_tensors="pt",    # Return PyTorch tensors
    )
    
    # Move tensors to appropriate device (CPU or GPU)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Convert token IDs back to string tokens for interpretability
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    return input_ids, attention_mask, tokens

# PREDICTION
@torch.no_grad()
def predict_binary(model, input_ids, attention_mask, model_type, task=None, threshold=0.5):
    '''
    Generate binary prediction (safe=0, unsafe=1) for a single example
    '''
    model.eval()

    # Get logits based on model type
    if model_type == "multitask" and task is None:
        logits = model(input_ids, attention_mask)[task]  # [B,1] or [B]
    else:  # single_task
        out = model(input_ids, attention_mask)
        logits = out.logits if hasattr(out, "logits") else out # Handle possible SequenceClassifierOutput or raw tensor

        # Handle different output shapes
        if logits.shape[-1] == 2:  # Cross-entropy
            logits = logits[:, 1:2] # Extract positive class

    # Convert to scalars
    logits = logits.squeeze()
    prob = torch.sigmoid(logits).item() # Convert logit to probability
    pred = int(prob >= threshold) # Apply threshold

    return prob, pred, float(logits.item())

# COMPUTE INTEGRATED GRADIENTS
def compute_integrated_gradients(
    text,
    model,
    tokenizer,
    model_type,
    task=None,
    device="cuda",
    max_length=128,
    n_steps=25,
):
    '''
    Compute Integrated Gradients token attribution for a text example

    Integrated Gradients explains predictions by:
    1. Starting from a baseline (all PAD tokens = "no content")
    2. Gradually interpolating to the actual input
    3. Integrating gradients along this path
    4. Attributing prediction to each token based on gradient magnitude
    
    Higher attribution = token had more influence on the prediction
    '''
    # Multitask models require task specification
    if model_type == "multitask" and task is None:
        raise ValueError("task must be provided for multitask models")

    model.eval()

    # Tokenize the input text
    input_ids, attention_mask, tokens = prepare_inputs(
        text, tokenizer, device, max_length
    )

    # Create baseline (all PAD tokens)
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

    # Define forward function 
    if model_type == "multitask":
        def forward_func(ids, mask):
            '''
            Extract specific task output from multitask model
            '''
            out = model(ids, mask)[task]
            return out.squeeze(-1)
    else:
        def forward_func(ids, mask):
            '''
            Handle single-task model output
            '''
            out = model(ids, mask)
            logits = out.logits if hasattr(out, "logits") else out
            if logits.shape[-1] == 2:
                logits = logits[:, 1:2]
            return logits.squeeze(-1)

    # Embedding layer (IG computes attributions at embedding layer)
    if hasattr(model, "roberta"):
        embedding_layer = model.roberta.embeddings
    elif hasattr(model, "bert"):
        embedding_layer = model.bert.embeddings
    else:
        raise ValueError("Unsupported model architecture")

    # Initialize LayerIntegratedGradients
    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    # Compute attributions 
    # Integrates gradients from baseline to input over n_steps
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum over embedding dimension to get per-token attribution
    token_attrs = attributions.sum(dim=-1).squeeze(0).cpu().numpy()

    # Remove padding from output
    mask = attention_mask.squeeze(0).cpu().numpy().astype(bool)
    tokens = [t for t, m in zip(tokens, mask) if m]
    token_attrs = token_attrs[mask]

    # Normalize attributions to [-1, 1]
    token_attrs = token_attrs / (np.max(np.abs(token_attrs)) + 1e-9)

    # Get model's prediction for this example
    prob, pred, logit = predict_binary(
        model,
        input_ids,
        attention_mask,
        model_type=model_type,
        task=task,
    )

    # Return results dictionary
    return {
        "tokens": tokens,
        "attributions": token_attrs,
        "prob": prob,
        "pred": pred,
        "logit": logit,
        "convergence_delta": float(delta.item()),
        "text": text,
        "task": task,
        "model_type": model_type,
    }

# TOP-K TOKEN HELPER
def top_k_tokens(tokens, attributions, k=10):
    idx = np.argsort(np.abs(attributions))[-k:][::-1]
    return [(tokens[i], float(attributions[i])) for i in idx]
