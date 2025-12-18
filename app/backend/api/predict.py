from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    """Request model for prediction endpoint"""
    text: str
    model: str  # "logreg" | "singletask" | "multi2" | "multi4"
    task: Optional[str] = "Q_overall"

class PredictResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: int
    probability: float
    label: str
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

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict safety of input text using selected model
    """
    model_loader = get_model_loader()
    
    try:
        logger.info(f"Prediction request: model={request.model}, text_length={len(request.text)}")
        
        # Route to appropriate model
        if request.model == "logreg":
            result = model_loader.predict_logreg(request.text)
        elif request.model == "singletask":
            result = model_loader.predict_singletask(request.text)
        elif request.model in ["multi2", "multi4"]:
            num_heads = 2 if request.model == "multi2" else 4
            result = model_loader.predict_multitask(
                request.text, 
                task=request.task, 
                num_heads=num_heads
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
        
        result["model"] = request.model
        logger.info(f"Prediction successful: {result['label']} ({result['probability']:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")