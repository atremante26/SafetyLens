import os
from pathlib import Path
import requests
import sys

# Google Drive direct download URLs
# Replace FILE_ID with your actual IDs
MODEL_URLS = {
    'logistic_regression_model.pkl': 'https://drive.google.com/uc?export=download&id=1NSGLiM2M8l_h2N0m0DDKjVWsh2aQlnL4&confirm=t',
    'tfidf_vectorizer.pkl': 'https://drive.google.com/uc?export=download&id=1WHjq8UaTlRb2SqudGZCv5RiUQL42NQSB&confirm=t',
    'best_singletask.pt': 'https://drive.google.com/uc?export=download&id=1DX2oY2zPX7DgH6_F2j6BxvL6CNAd1IUH&confirm=t',
    'best_multitask_2.pt': 'https://drive.google.com/uc?export=download&id=1FZdKRT3E4mISFQ2HnRCODxXJQkF1xeBf&confirm=t',
    'best_multitask_4.pt': 'https://drive.google.com/uc?export=download&id=11AQtrY6veTF_j337g2f9YaSBxipfsdor&confirm=t',
}

def download_file(url, destination):
    """Download file from Google Drive with proper handling"""
    print(f"Downloading {destination.name}...", flush=True)
    try:
        session = requests.Session()
        response = session.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Check if we got HTML instead of binary file
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            print(f"Got HTML instead of file - check sharing permissions", flush=True)
            return False
        
        # Save file
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (5*1024*1024) < 1024*1024:  # Print every ~5MB
                                print(f"  Progress: {percent:.0f}%", flush=True)
        
        # Verify file
        if destination.stat().st_size == 0:
            print(f"File is empty", flush=True)
            return False
        
        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"Downloaded {destination.name} ({size_mb:.1f} MB)", flush=True)
        return True
        
    except Exception as e:
        print(f"Error: {e}", flush=True)
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