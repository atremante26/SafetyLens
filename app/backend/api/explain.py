from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

from ..explainability.lime_explainer import LIMEExplainer
from ..explainability.ig_explainer import IGExplainer
from ..explainability.shap_explainer import SHAPExplainer    

router = APIRouter()
logger = logging.getLogger(__name__)

class ExplainRequest(BaseModel):
    """Request model for explanation endpoint"""
    text: str
    model: str
    method: str  # "lime" | "shap" | "ig"
    task: Optional[str] = "Q_overall"
    n_samples: Optional[int] = 500
    n_steps: Optional[int] = 25
    num_features: Optional[int] = 10

class TokenAttribution(BaseModel):
    """Token attribution pair"""
    token: str
    attribution: float

class ExplainResponse(BaseModel):
    """Response model for explanation endpoint"""
    tokens: List[TokenAttribution]
    method: str
    model: str

# Global model loader (injected by main.py)
_model_loader = None

def set_model_loader(loader):
    """Set the model loader instance"""
    global _model_loader
    _model_loader = loader

def get_model_loader():
    """Get the model loader instance"""
    if _model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return _model_loader

@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Generate explainability for prediction
    """
    model_loader = get_model_loader()
    
    try:
        logger.info(f"Explanation request: model={request.model}, method={request.method}")
        
        # Route based on method
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
                    task=None
                )
            elif request.model == "multi2":
                tokens = explainer.explain_transformer(
                    request.text,
                    model_loader.multitask_2_model,
                    model_loader.multitask_2_tokenizer,
                    num_features=request.num_features,
                    num_samples=request.n_samples,
                    task=request.task
                )
            elif request.model == "multi4":
                tokens = explainer.explain_transformer(
                    request.text,
                    model_loader.multitask_4_model,
                    model_loader.multitask_4_tokenizer,
                    num_features=request.num_features,
                    num_samples=request.n_samples,
                    task=request.task
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
        
        elif request.method == "shap":
            if request.model != "logreg":
                raise HTTPException(
                    status_code=400, 
                    detail="SHAP only available for Logistic Regression"
                )
            
            explainer = SHAPExplainer()
            tokens = explainer.explain_logreg(
                request.text,
                model_loader.logreg_model,
                model_loader.vectorizer,
                num_features=request.num_features
            )
        
        elif request.method == "ig":
            if request.model == "logreg":
                raise HTTPException(
                    status_code=400, 
                    detail="Integrated Gradients not available for Logistic Regression"
                )
            
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
        
        logger.info(f"Explanation successful: {len(tokens)} tokens")
        
        return {
            "tokens": tokens,
            "method": request.method,
            "model": request.model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")