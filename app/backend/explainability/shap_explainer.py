import shap
import numpy as np

class SHAPExplainer:
    """Wrapper for SHAP explanations (Logistic Regression only)"""
    def __init__(self):
        pass
    
    def explain_logreg(self, text, model, vectorizer, num_features=10):
        """
        Explain logistic regression prediction using SHAP
        """
        # Transform text
        X = vectorizer.transform([text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get non-zero features (words that appear in the text)
        non_zero_idx = X.nonzero()[1]
        
        if len(non_zero_idx) == 0:
            return []
        
        # Create explainer with just the training data mean
        # Use a simple background dataset
        background = np.zeros((1, X.shape[1]))
        explainer = shap.LinearExplainer(model, background)
        
        # Get SHAP values
        shap_values = explainer.shap_values(X.toarray())
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # Take the unsafe class (class 1)
            shap_values_unsafe = shap_values[1][0]
        else:
            shap_values_unsafe = shap_values[0]
        
        # Create results for non-zero features only
        results = []
        for idx in non_zero_idx:
            token = feature_names[idx]
            attribution = float(shap_values_unsafe[idx])
            
            # Only include if attribution is non-zero
            if abs(attribution) > 1e-6:
                results.append({
                    "token": token,
                    "attribution": attribution
                })
        
        # If still no results, try a different approach
        if len(results) == 0:
            # Fall back to using coefficients directly
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            
            for idx in non_zero_idx:
                token = feature_names[idx]
                # Multiply coefficient by feature value
                feature_value = X[0, idx]
                attribution = float(coef[idx] * feature_value)
                
                if abs(attribution) > 1e-6:
                    results.append({
                        "token": token,
                        "attribution": attribution
                    })
        
        # Sort by absolute attribution
        results.sort(key=lambda x: abs(x['attribution']), reverse=True)
        
        return results[:num_features]