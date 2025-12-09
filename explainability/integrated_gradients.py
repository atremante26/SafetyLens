import torch
import numpy as np
from captum.attr import IntegratedGradients


def prepare_input(text, tokenizer, device):
    # Tokenize text and prepare for model input 
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device),
        'tokens': tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    }


def get_prediction(model, input_ids, attention_mask, task='Q_overall'):
    # Get model prediction for a specific task
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs[task]
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
    
    return {
        'predicted_class': pred_class.item(),
        'probabilities': probs[0].cpu().numpy(),
        'class_names': ['Safe', 'Ambiguous', 'Unsafe']
    }


def construct_baseline(input_ids, attention_mask, baseline_token_id=1):
    # Construct baseline (all padding tokens) for Integrated Gradients
    ref_input_ids = torch.ones_like(input_ids) * baseline_token_id
    ref_attention_mask = torch.zeros_like(attention_mask)
    return (input_ids, attention_mask), (ref_input_ids, ref_attention_mask)


def compute_attribution_metrics(attributions):
    # Compute statistics for attribution analysis (mean, std, max, sparsity)
    non_zero_attrs = attributions[attributions != 0]
    
    if len(non_zero_attrs) == 0:
        return {'mean': 0, 'std': 0, 'max': 0, 'sparsity': 1.0}
    
    return {
        'mean': float(np.mean(np.abs(non_zero_attrs))),
        'std': float(np.std(non_zero_attrs)),
        'max': float(np.max(np.abs(non_zero_attrs))),
        'sparsity': float(1.0 - (len(non_zero_attrs) / len(attributions)))
    }


def compute_ig_attributions(text, model, tokenizer, task='Q_overall', n_steps=50, device='cuda'):
    """
    Compute Integrated Gradients attributions for a text input.
    
    Returns dictionary with tokens, attributions, predictions, and metrics.
    """
    # Prepare input
    inputs = prepare_input(text, tokenizer, device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    tokens = inputs['tokens']
    
    # Get prediction
    prediction = get_prediction(model, input_ids, attention_mask, task)
    predicted_class = prediction['predicted_class']
    
    # Construct baseline
    (input_ids, attention_mask), (ref_input_ids, ref_attention_mask) = \
        construct_baseline(input_ids, attention_mask)
    
    # Define forward function for this task
    def forward_func(ids, mask):
        outputs = model(ids, mask)
        return outputs[task]
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(forward_func)
    
    # Compute attributions
    attributions, delta = ig.attribute(
        inputs=(input_ids, attention_mask),
        baselines=(ref_input_ids, ref_attention_mask),
        target=predicted_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    
    # Extract token attributions (sum over embedding dimensions)
    token_attributions = attributions[0].sum(dim=2).squeeze(0)
    token_attributions = token_attributions.cpu().detach().numpy()
    
    # Normalize attributions
    token_attributions = token_attributions / np.linalg.norm(token_attributions)
    
    return {
        'text': text,
        'task': task,
        'tokens': tokens,
        'attributions': token_attributions,
        'predicted_class': predicted_class,
        'class_name': prediction['class_names'][predicted_class],
        'probabilities': prediction['probabilities'],
        'convergence_delta': delta.item(),
        'metrics': compute_attribution_metrics(token_attributions)
    }