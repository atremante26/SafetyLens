from lime.lime_text import LimeTextExplainer
import numpy as np

class LIMEExplainer:
    """Wrapper for LIME text explanations"""
    
    def __init__(self):
        self.explainer = LimeTextExplainer(class_names=['Safe', 'Unsafe'])
    
    def explain_logreg(self, text, model, vectorizer, num_features=10, num_samples=500):
        """
        Explain logistic regression prediction
        
        Returns: List of (token, attribution) tuples
        """
        def predict_proba(texts):
            X = vectorizer.transform(texts)
            return model.predict_proba(X)
        
        exp = self.explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get feature weights for unsafe class (class 1)
        weights = exp.as_list()
        
        # Format as list of dicts for API response
        return [
            {"token": token, "attribution": float(weight)}
            for token, weight in weights
        ]
    
    def explain_transformer(self, text, model, tokenizer, num_features=10, num_samples=500):
        """
        Explain transformer prediction (single-task or multi-task)
        
        Returns: List of (token, attribution) tuples
        """
        import torch
        
        device = next(model.parameters()).device
        
        def predict_proba(texts):
            results = []
            for t in texts:
                inputs = tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)
                
                with torch.no_grad():
                    # Handle different model types
                    if hasattr(model, 'roberta'):  # RobertaForSequenceClassification
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                    else:  # Multi-task model
                        logits_dict = model(**inputs)
                        # Use first task for explanation
                        task = list(logits_dict.keys())[0]
                        logit = logits_dict[task]
                        prob_unsafe = torch.sigmoid(logit).item()
                        probs = torch.tensor([[1 - prob_unsafe, prob_unsafe]])
                
                results.append(probs[0].cpu().numpy())
            
            return np.array(results)
        
        exp = self.explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        weights = exp.as_list()
        
        return [
            {"token": token, "attribution": float(weight)}
            for token, weight in weights
        ]