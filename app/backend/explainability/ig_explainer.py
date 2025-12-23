import torch
from captum.attr import LayerIntegratedGradients
import numpy as np
import logging
logger = logging.getLogger(__name__)

class IGExplainer:
    """Wrapper for Integrated Gradients explanations"""
    def __init__(self):
        pass
    
    def explain_transformer(self, text, model, tokenizer, n_steps=25, num_features=10, task='Q_overall'):
        """
        Explain transformer prediction using Integrated Gradients.
        """
        device = next(model.parameters()).device
        
        # Check if multi-task
        is_multitask = hasattr(model, 'heads')
        
        # Tokenize
        max_length = 256 if is_multitask else 512
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding='max_length' if is_multitask else True
        ).to(device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Define forward function for IG
        def forward_func(input_ids, attention_mask):
            if is_multitask:
                # Multi-task model returns dict: {'Q_overall': tensor([[logit]]), ...}
                logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)
                logit = logits_dict[task]  # Shape: (batch_size, 1)
                return logit.squeeze(-1)  # Return shape: (batch_size,)
            else:
                # Single-task model
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits[:, 1]  # Unsafe class logit: (batch_size,)
        
        # Get embeddings layer
        embeddings_layer = model.roberta.embeddings
        
        # Create IG instance
        lig = LayerIntegratedGradients(forward_func, embeddings_layer)
        
        try:
            # Compute attributions
            attributions, delta = lig.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(input_ids * 0, attention_mask),
                n_steps=n_steps,
                return_convergence_delta=True
            )
        except Exception as e:
            logger.warning(f"IG attribution failed: {e}. Trying without convergence delta...")
            # Fallback: compute without delta
            attributions = lig.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(input_ids * 0, attention_mask),
                n_steps=n_steps,
                return_convergence_delta=False
            )
        
        # Sum across embedding dimension
        attributions = attributions.sum(dim=-1).squeeze(0)
        
        # Normalize
        attr_norm = torch.norm(attributions)
        if attr_norm > 0:
            attributions = attributions / attr_norm
        
        attributions = attributions.cpu().detach().numpy()
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter out special tokens and combine
        results = []
        for token, attr in zip(tokens, attributions):
            if token not in ['<s>', '</s>', '<pad>']:
                # Clean up token (remove Ġ prefix from RoBERTa)
                clean_token = token.replace('Ġ', '')
                if clean_token.strip():  # Skip empty tokens
                    results.append({
                        "token": clean_token,
                        "attribution": float(attr)
                    })
        
        # Sort by absolute attribution
        results.sort(key=lambda x: abs(x['attribution']), reverse=True)
        
        return results[:num_features] 