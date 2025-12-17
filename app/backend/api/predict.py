"""
Prediction endpoint
POST /api/predict - Returns safety prediction
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class PredictRequest(BaseModel):
    text: str
    model: str  # "logreg" | "singletask" | "multi2" | "multi4"
    task: Optional[str] = "Q_overall"  # For multi-task models

class PredictResponse(BaseModel):
    prediction: int  # 0 = safe, 1 = unsafe
    probability: float  # Probability of unsafe
    label: str  # "Safe" or "Unsafe"
    model: str

# This will be set by main.py
_model_loader = None

def set_model_loader(loader):
    """Called by main.py to inject the model loader"""
    global _model_loader
    _model_loader = loader

def get_model_loader():
    """Dependency to get the model loader"""
    if _model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return _model_loader

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict safety of input text using selected model
    """
    import traceback
    
    model_loader = get_model_loader()
    
    try:
        print(f"🔍 Received prediction request:")
        print(f"   Model: {request.model}")
        print(f"   Task: {request.task}")
        print(f"   Text length: {len(request.text)} chars")
        
        # Route to appropriate model
        if request.model == "logreg":
            print("   Using LogReg model...")
            result = model_loader.predict_logreg(request.text)
        elif request.model == "singletask":
            print("   Using Single-Task model...")
            result = model_loader.predict_singletask(request.text)
        elif request.model in ["multi2", "multi4"]:
            num_heads = 2 if request.model == "multi2" else 4
            print(f"   Using Multi-Task ({num_heads} heads) model...")
            result = model_loader.predict_multitask(
                request.text, 
                task=request.task, 
                num_heads=num_heads
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
        
        result["model"] = request.model
        print(f"✅ Prediction successful: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Prediction failed with error:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")