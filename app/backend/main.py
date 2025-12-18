from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .api.predict import router as predict_router, set_model_loader as set_predict_loader
from .api.explain import router as explain_router, set_model_loader as set_explain_loader
from .models.loader import ModelLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    logger.info("Loading models...")
    
    try:
        model_loader = ModelLoader()
        model_loader.load_all_models()
        
        # Inject model loader into API routes
        set_predict_loader(model_loader)
        set_explain_loader(model_loader)
        
        logger.info("All models loaded successfully!")
        
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
    allow_origins=["http://localhost:5173"],
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