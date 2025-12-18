"""
Explainability endpoint
POST /api/explain - Returns token attributions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import traceback

router = APIRouter()

class ExplainRequest(BaseModel):
    text: str
    model: str  # "logreg" | "singletask" | "multi2" | "multi4"
    method: str  # "lime" | "shap" | "ig"
    task: Optional[str] = "Q_overall"
    n_samples: Optional[int] = 500  # For LIME
    n_steps: Optional[int] = 25  # For IG
    num_features: Optional[int] = 10

class TokenAttribution(BaseModel):
    token: str
    attribution: float

class ExplainResponse(BaseModel):
    tokens: List[TokenAttribution]
    method: str
    model: str

# Will be set by main.py
_model_loader = None

def set_model_loader(loader):
    global _model_loader
    _model_loader = loader

def get_model_loader():
    if _model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return _model_loader

@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Generate explainability for prediction
    """
    from ..explainability.lime_explainer import LIMEExplainer
    from ..explainability.ig_explainer import IGExplainer
    from ..explainability.shap_explainer import SHAPExplainer
    
    model_loader = get_model_loader()
    
    try:
        print(f"🔍 Received explanation request:")
        print(f"   Model: {request.model}")
        print(f"   Method: {request.method}")
        print(f"   Text length: {len(request.text)} chars")
        
        # Around line 70-95, replace the LIME section:

        if request.method == "lime":
            explainer = LIMEExplainer()
            
            if request.model == "logreg":
                tokens = explainer.explain_logreg(
                    request.text,
                    model_loader.logreg_model,
                    model_loader.vectorizer,
                    num_features=request.num_features,
                    num_samples=request.n_samples
                )
            elif request.model == "singletask":
                tokens = explainer.explain_transformer(
                    request.text,
                    model_loader.singletask_model,
                    model_loader.singletask_tokenizer,
                    num_features=request.num_features,
                    num_samples=request.n_samples,
                    task=None  # Single-task doesn't need task parameter
                )
            elif request.model == "multi2":
                tokens = explainer.explain_transformer(
                    request.text,
                    model_loader.multitask_2_model,
                    model_loader.multitask_2_tokenizer,
                    num_features=request.num_features,
                    num_samples=request.n_samples,
                    task=request.task  # Pass task parameter for multi-task
                )
            elif request.model == "multi4":
                tokens = explainer.explain_transformer(
                    request.text,
                    model_loader.multitask_4_model,
                    model_loader.multitask_4_tokenizer,
                    num_features=request.num_features,
                    num_samples=request.n_samples,
                    task=request.task  # Pass task parameter for multi-task
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
        
        elif request.method == "shap":
            if request.model != "logreg":
                raise HTTPException(status_code=400, detail="SHAP only available for Logistic Regression")
            
            explainer = SHAPExplainer()
            tokens = explainer.explain_logreg(
                request.text,
                model_loader.logreg_model,
                model_loader.vectorizer,
                num_features=request.num_features
            )
        
        elif request.method == "ig":
            if request.model == "logreg":
                raise HTTPException(status_code=400, detail="Integrated Gradients not available for Logistic Regression")
            
            explainer = IGExplainer()
            
            if request.model == "singletask":
                tokens = explainer.explain_transformer(
                    request.text,
                    model_loader.singletask_model,
                    model_loader.singletask_tokenizer,
                    n_steps=request.n_steps,
                    task=None
                )
            elif request.model == "multi2":
                tokens = explainer.explain_transformer(
                    request.text,
                    model_loader.multitask_2_model,
                    model_loader.multitask_2_tokenizer,
                    n_steps=request.n_steps,
                    task=request.task
                )
            elif request.model == "multi4":
                tokens = explainer.explain_transformer(
                    request.text,
                    model_loader.multitask_4_model,
                    model_loader.multitask_4_tokenizer,
                    n_steps=request.n_steps,
                    task=request.task
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        print(f"✅ Explanation generated: {len(tokens)} tokens")
        
        return {
            "tokens": tokens,
            "method": request.method,
            "model": request.model
        }
        
    except Exception as e:
        print(f"❌ Explanation failed:")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")