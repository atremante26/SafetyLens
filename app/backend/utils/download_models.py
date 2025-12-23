import os
from pathlib import Path
import sys
import gdown

# Google Drive file IDs
MODEL_FILE_IDS = {
    'logistic_regression_model.pkl': '1NSGLiM2M8l_h2N0m0DDKjVWsh2aQlnL4',
    'tfidf_vectorizer.pkl': '1WHjq8UaTlRb2SqudGZCv5RiUQL42NQSB',
    'best_singletask.pt': '1DX2oY2zPX7DgH6_F2j6BxvL6CNAd1IUH',
    'best_multitask_2.pt': '1FZdKRT3E4mISFQ2HnRCODxXJQkF1xeBf',
    'best_multitask_4.pt': '11AQtrY6veTF_j337g2f9YaSBxipfsdor',
}

def download_file(file_id, destination):
    """Download file from Google Drive using gdown"""
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(destination), quiet=False, fuzzy=True)
        
        # Verify download
        if not destination.exists() or destination.stat().st_size == 0:
            print(f"Download failed or file is empty", flush=True)
            return False
        
        # Calculate size
        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"Downloaded {destination.name} ({size_mb:.1f} MB)", flush=True)
        return True
        
    except Exception as e:
        print(f"Error downloading {destination.name}: {e}", flush=True)
        return False

def setup_models():
    """Download models if running on Render"""
    
    # Only download on Render, not locally
    if not os.getenv("RENDER"):
        print("Local environment - skipping model download", flush=True)
        return True
    
    print("Running on Render - checking models...", flush=True)
    
    # Create models directory
    models_dir = Path(__file__).parent.parent.parent.parent.parent / "models" / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Models directory: {models_dir}", flush=True)
    
    # Check and download each model
    all_success = True
    for filename, file_id in MODEL_FILE_IDS.items():
        filepath = models_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 0:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"{filename} exists ({size_mb:.1f} MB)", flush=True)
            continue
        
        print(f"Need to download {filename}", flush=True)
        success = download_file(file_id, filepath)
        
        if not success:
            all_success = False
            print(f"Could not download {filename}", flush=True)
    
    if all_success:
        print("All models ready!", flush=True)
    else:
        print("Some models failed to download", flush=True)
    
    return all_success

if __name__ == "__main__":
    success = setup_models()
    sys.exit(0 if success else 1)