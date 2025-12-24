#!/bin/bash

echo "  Starting SafetyLens with Docker..."
echo ""
echo "  This will:"
echo "   - Build Docker image (auto-downloads models if needed)"
echo "   - Start the backend (FastAPI on port 8000) with ALL models loaded"
echo "   - Start the frontend (React + Vite on port 5173)"
echo ""

# Check if models exist locally
if [ -f "models/checkpoints/logistic_regression_model.pkl" ]; then
    echo "   Models found locally - build will be faster!"
    echo ""
    echo "   First build: ~2-3 minutes"
else
    echo "   Models not found locally"
    echo "   Docker will download them during build (~5-10 minutes first time)"
    echo "   Or you can download manually now for faster builds:"
    echo ""
    echo "   pip install gdown"
    echo "   mkdir -p models/checkpoints && cd models/checkpoints"
    echo "   gdown 1NSGLiM2M8l_h2N0m0DDKjVWsh2aQlnL4  # logreg"
    echo "   gdown 1WHjq8UaTlRb2SqudGZCv5RiUQL42NQSB  # vectorizer"
    echo "   gdown 1DX2oY2zPX7DgH6_F2j6BxvL6CNAd1IUH  # singletask"
    echo "   gdown 1FZdKRT3E4mISFQ2HnRCODxXJQkF1xeBf  # multitask_2"
    echo "   gdown 11AQtrY6veTF_j337g2f9YaSBxipfsdor  # multitask_4"
    echo "   cd ../.."
    echo ""
    read -p "Continue with automatic download? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Download models and try again."
        exit 1
    fi
    echo ""
    echo "⏱  First build with downloads: ~5-10 minutes"
fi

echo ""
echo "Building and starting..."
echo ""
echo "When ready, access:"
echo ""
echo "   Frontend:  http://localhost:5173/SafetyLens/"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "   ALL MODELS LOADED - No 30-second wait!"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Build and start
docker-compose up --build

# Clean exit message
echo ""
echo "SafetyLens stopped!"