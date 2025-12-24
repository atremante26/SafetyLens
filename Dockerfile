# Backend Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (including gdown for model downloads)
RUN pip install --no-cache-dir -r requirements.txt gdown

# Copy entire project
COPY . .

# Download models if they don't exist
# This runs during build, so models are baked into the image
RUN mkdir -p models/checkpoints && \
    cd models/checkpoints && \
    if [ ! -f "logistic_regression_model.pkl" ]; then \
        echo "Downloading models (this may take 5-10 minutes)..."; \
        gdown 1NSGLiM2M8l_h2N0m0DDKjVWsh2aQlnL4 || echo "Failed to download logistic_regression_model.pkl"; \
    fi && \
    if [ ! -f "tfidf_vectorizer.pkl" ]; then \
        gdown 1WHjq8UaTlRb2SqudGZCv5RiUQL42NQSB || echo "Failed to download tfidf_vectorizer.pkl"; \
    fi && \
    if [ ! -f "best_singletask.pt" ]; then \
        gdown 1DX2oY2zPX7DgH6_F2j6BxvL6CNAd1IUH || echo "Failed to download best_singletask.pt"; \
    fi && \
    if [ ! -f "best_multitask_2.pt" ]; then \
        gdown 1FZdKRT3E4mISFQ2HnRCODxXJQkF1xeBf || echo "Failed to download best_multitask_2.pt"; \
    fi && \
    if [ ! -f "best_multitask_4.pt" ]; then \
        gdown 11AQtrY6veTF_j337g2f9YaSBxipfsdor || echo "Failed to download best_multitask_4.pt"; \
    fi && \
    cd ../.. && \
    echo "Model download complete!"

# Verify at least LogReg exists (minimum requirement)
RUN if [ ! -f "models/checkpoints/logistic_regression_model.pkl" ]; then \
        echo "ERROR: Critical model files missing after download attempt"; \
        echo "Please check your internet connection or download models manually"; \
        exit 1; \
    fi

# Expose backend port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOAD_ALL_MODELS=true

# Run the FastAPI application
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
