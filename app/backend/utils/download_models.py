import os
from pathlib import Path
import requests
import sys

# Google Drive direct download URLs
# Replace FILE_ID with your actual IDs
MODEL_URLS = {
    'logistic_regression_model.pkl': 'https://drive.google.com/file/d/1NSGLiM2M8l_h2N0m0DDKjVWsh2aQlnL4/view?usp=sharing',
    'tfidf_vectorizer.pkl': 'https://drive.google.com/file/d/1WHjq8UaTlRb2SqudGZCv5RiUQL42NQSB/view?usp=sharing',
    'best_singletask.pt': 'https://drive.google.com/file/d/1DX2oY2zPX7DgH6_F2j6BxvL6CNAd1IUH/view?usp=sharing',
    'best_multitask_2.pt': 'https://drive.google.com/file/d/1FZdKRT3E4mISFQ2HnRCODxXJQkF1xeBf/view?usp=sharing',
    'best_multitask_4.pt': 'https://drive.google.com/file/d/11AQtrY6veTF_j337g2f9YaSBxipfsdor/view?usp=sharing',
}

def download_file(url, destination):
    """Download file from Google Drive"""   
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100
                        print(f"  Progress: {percent:.1f}%", flush=True)
        
        print(f"Downloaded {destination.name}", flush=True)
        return True
        
    except Exception as e:
        print(f"Failed to download {destination.name}: {e}", flush=True)
        return False

def setup_models():
    """Download models if running on Render"""
    # Only download on Render, not locally
    if not os.getenv("RENDER"):
        print("Local environment - skipping model download", flush=True)
        return True
    
    # Create models directory
    models_dir = Path(__file__).parent.parent.parent.parent / "models" / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Models directory: {models_dir}", flush=True)
    
    # Check and download each model
    all_success = True
    for filename, url in MODEL_URLS.items():
        filepath = models_dir / filename
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"{filename} exists ({size_mb:.1f} MB)", flush=True)
            continue
        
        print(f"Need to download {filename}", flush=True)
        success = download_file(url, filepath)
        
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