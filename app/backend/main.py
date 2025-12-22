from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download models on Render startup
try:
    from .utils.download_models import setup_models
    setup_models()
except Exception as e:
    logger.warning(f"Model setup issue: {e}")
    # Continue anyway - models might already be there or it's local dev

from .api.predict import router as predict_router, set_model_loader as set_predict_loader
from .api.explain import router as explain_router, set_model_loader as set_explain_loader
from .models.loader import ModelLoader

# Global model loader
model_loader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup (model loading) and shutdown
    """
    global model_loader
    
    logger.info("Starting SafetyLens API...")
    logger.info("Loading LogReg model (transformers will lazy-load on demand)...")
    
    try:
        model_loader = ModelLoader()
        
        # Only load LogReg on startup (tiny memory footprint)
        model_loader.load_logreg()
        
        # Inject model loader into API routes
        set_predict_loader(model_loader)
        set_explain_loader(model_loader)
        
        logger.info("LogReg loaded! Transformers will load on first use (~30 sec each).")
        
        yield
        
    finally:
        logger.info("Shutting down SafetyLens API...")

# Create FastAPI app
app = FastAPI(
    title="SafetyLens API",
    description="Multi-model content safety detection with explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",              # Local development
        "https://atremante26.github.io",      # GitHub Pages
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, prefix="/api", tags=["Prediction"])
app.include_router(explain_router, prefix="/api", tags=["Explainability"])

@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "message": "SafetyLens API is running",
        "version": "1.0.0",
        "status": "ok",
        "models_loaded": model_loader is not None
    }

@app.get("/api/health")
async def health():
    """Health check endpoint - detailed model status"""
    if model_loader is None:
        return {"status": "error", "message": "Models not loaded"}
    
    return {
        "status": "ok",
        "models": {
            "logreg": model_loader.logreg_model is not None,
            "singletask": model_loader.singletask_model is not None,
            "multitask_2": model_loader.multitask_2_model is not None,
            "multitask_4": model_loader.multitask_4_model is not None,
        }
    }