import torch
import numpy as np
from captum.attr import LayerIntegratedGradients


def prepare_inputs(text, tokenizer, device, max_length=128):
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return input_ids, attention_mask, tokens


@torch.no_grad()
def predict_binary(model, input_ids, attention_mask, task="Q2_harmful", threshold=0.5):
    model.eval()
    logits = model(input_ids=input_ids, attention_mask=attention_mask)[task]  # [B,1]
    prob = torch.sigmoid(logits).squeeze(-1)  # [B]
    pred = (prob >= threshold).long()
    return {
        "logit": float(logits.squeeze().item()),
        "prob": float(prob.item()),
        "pred": int(pred.item())
    }


def metrics(attributions):
    attributions = np.asarray(attributions)
    non_zero = attributions[np.abs(attributions) > 1e-12]
    if len(non_zero) == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "sparsity": 1.0}
    return {
        "mean": float(np.mean(np.abs(non_zero))),
        "std": float(np.std(np.abs(non_zero))),
        "max": float(np.max(np.abs(non_zero))),
        "sparsity": float(1.0 - (len(non_zero) / len(attributions)))
    }


def compute_ig_attributions(
    text,
    model,
    tokenizer,
    task="Q2_harmful",
    n_steps=25,
    device="cuda",
    max_length=128,
):
    """
    Integrated Gradients on RoBERTa embeddings using LayerIntegratedGradients.

    Returns:
        dict with tokens, attributions per token, prob/pred, convergence delta.
    """
    model.eval()
    input_ids, attention_mask, tokens = prepare_inputs(text, tokenizer, device, max_length=max_length)

    # Baseline: all PAD tokens, same attention mask
    pad_id = tokenizer.pad_token_id
    baseline_ids = torch.full_like(input_ids, pad_id)

    # Forward function returns scalar logit per example (Captum expects shape [B,1] or [B])
    def forward_func(ids, mask):
        return model(input_ids=ids, attention_mask=mask)[task]  # [B,1]

    lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True
    )

    # Sum over embedding dim -> [B,L]
    token_attrs = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # Mask out padding for readability
    mask = attention_mask.squeeze(0).detach().cpu().numpy().astype(bool)
    tokens = [t for t, m in zip(tokens, mask) if m]
    token_attrs = np.array([a for a, m in zip(token_attrs, mask) if m], dtype=float)

    # Safe normalization (optional)
    denom = np.max(np.abs(token_attrs)) + 1e-9
    token_attrs_norm = token_attrs / denom

    pred_info = predict_binary(model, input_ids, attention_mask, task=task)

    return {
        "text": text,
        "task": task,
        "tokens": tokens,
        "attributions": token_attrs_norm,
        "raw_attributions": token_attrs,
        "prob": pred_info["prob"],
        "pred": pred_info["pred"],
        "logit": pred_info["logit"],
        "convergence_delta": float(delta.item()),
        "metrics": metrics(token_attrs_norm)
    }
