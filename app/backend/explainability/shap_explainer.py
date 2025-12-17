"""
SHAP explainer for logistic regression
"""

import shap
import numpy as np

class SHAPExplainer:
    """Wrapper for SHAP explanations (LogReg only)"""
    
    def __init__(self):
        pass
    
    def explain_logreg(self, text, model, vectorizer, num_features=10):
        """
        Explain logistic regression prediction using SHAP
        
        Returns: List of (token, attribution) tuples
        """
        # Transform text
        X = vectorizer.transform([text])
        
        # Create SHAP explainer
        explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
        
        # Get SHAP values
        shap_values = explainer.shap_values(X)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get non-zero features
        non_zero_idx = X.nonzero()[1]
        
        if len(non_zero_idx) == 0:
            return []
        
        # Create results
        results = []
        for idx in non_zero_idx:
            token = feature_names[idx]
            # SHAP values for unsafe class
            attribution = float(shap_values[1][0, idx]) if len(shap_values) > 1 else float(shap_values[0, idx])
            results.append({
                "token": token,
                "attribution": attribution
            })
        
        # Sort by absolute attribution
        results.sort(key=lambda x: abs(x['attribution']), reverse=True)
        
        return results[:num_features]