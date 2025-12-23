from lime.lime_text import LimeTextExplainer
import numpy as np
import torch

class LIMEExplainer:
    """Wrapper for LIME text explanations"""
    def __init__(self):
        self.explainer = LimeTextExplainer(class_names=['Safe', 'Unsafe'])
    
    def explain_logreg(self, text, model, vectorizer, num_features=10, num_samples=500):
        """
        Explain logistic regression prediction.
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
    
    def explain_transformer(self, text, model, tokenizer, num_features=10, num_samples=500, task='Q_overall'):
        """
        Explain transformer prediction (single-task or multi-task).
        """
        device = next(model.parameters()).device
        
        # Check if this is a multi-task model
        is_multitask = hasattr(model, 'heads')
        
        def predict_proba(texts):
            results = []
            for t in texts:
                inputs = tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256 if is_multitask else 512,
                    padding='max_length' if is_multitask else True
                ).to(device)
                
                with torch.no_grad():
                    if is_multitask:
                        # Multi-task model returns dict of single logits
                        logits_dict = model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask']
                        )
                        
                        # Get logit for specified task - Shape: (batch_size, 1)
                        logit = logits_dict[task]
                        prob_unsafe = torch.sigmoid(logit).squeeze().item()
                        probs = torch.tensor([[1 - prob_unsafe, prob_unsafe]])
                    else:
                        # Single-task model (RobertaForSequenceClassification)
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                
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