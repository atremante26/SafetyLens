"""
SafetyLens FastAPI Backend
Main application entry point with CORS enabled.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Use relative imports (note the dots)
from .api.predict import router as predict_router, set_model_loader
from .api.explain import router as explain_router
from .api.explain import set_model_loader as set_explain_model_loader
from .models.loader import ModelLoader

# Global model loader instance
model_loader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load all models into memory
    Shutdown: Clean up resources
    """
    global model_loader
    
    print("🚀 Starting SafetyLens API...")
    print("📦 Loading models...")
    
    model_loader = ModelLoader()
    model_loader.load_all_models()
    
    # Inject model loader into API routes
    set_model_loader(model_loader)
    set_explain_model_loader(model_loader) 
    
    print("✅ All models loaded successfully!")
    
    yield
    
    print("🛑 Shutting down SafetyLens API...")

# Create FastAPI app
app = FastAPI(
    title="SafetyLens API",
    description="Content safety detection with explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, prefix="/api", tags=["Prediction"])
app.include_router(explain_router, prefix="/api", tags=["Explainability"])

# Root endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SafetyLens API is running",
        "version": "1.0.0",
        "models_loaded": model_loader is not None
    }

@app.get("/api/health")
async def health():
    """Detailed health check"""
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