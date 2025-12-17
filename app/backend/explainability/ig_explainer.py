"""
Integrated Gradients explainer for transformers
"""

import torch
from captum.attr import LayerIntegratedGradients
import numpy as np

class IGExplainer:
    """Wrapper for Integrated Gradients explanations"""
    
    def __init__(self):
        pass
    
    def explain_transformer(self, text, model, tokenizer, n_steps=25):
        """
        Explain transformer prediction using Integrated Gradients
        
        Returns: List of (token, attribution) tuples
        """
        device = next(model.parameters()).device
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Define forward function for IG
        def forward_func(input_ids, attention_mask):
            if hasattr(model, 'roberta'):  # Single-task
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits[:, 1]  # Unsafe class logit
            else:  # Multi-task
                logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)
                task = list(logits_dict.keys())[0]
                return logits_dict[task].squeeze()
        
        # Get embeddings layer
        if hasattr(model, 'roberta'):
            embeddings_layer = model.roberta.embeddings
        else:
            embeddings_layer = model.roberta.embeddings
        
        # Create IG instance
        lig = LayerIntegratedGradients(forward_func, embeddings_layer)
        
        # Compute attributions
        attributions, delta = lig.attribute(
            inputs=(input_ids, attention_mask),
            baselines=(input_ids * 0, attention_mask),
            n_steps=n_steps,
            return_convergence_delta=True
        )
        
        # Sum across embedding dimension
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter out special tokens and combine
        results = []
        for token, attr in zip(tokens, attributions):
            if token not in ['<s>', '</s>', '<pad>']:
                # Clean up token (remove Ġ prefix from RoBERTa)
                clean_token = token.replace('Ġ', '')
                results.append({
                    "token": clean_token,
                    "attribution": float(attr)
                })
        
        # Sort by absolute attribution
        results.sort(key=lambda x: abs(x['attribution']), reverse=True)
        
        return results[:10]  # Top 10