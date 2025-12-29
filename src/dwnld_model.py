import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import requests

# Configure logging
logger = logging.getLogger(__name__)


def get_default_onnx_dir():
    """
    Get the default ONNX models directory.
    Uses ONNX_HOME environment variable if set, otherwise defaults to ./models.
    """
    onnx_home = os.environ.get("ONNX_HOME")
    if onnx_home:
        return Path(onnx_home)
    else:
        default_dir = Path("./models")
        logger.info(f"Using default ONNX directory: {default_dir}")
        return default_dir


# Model registry mapping model indices to Zenodo ONNX files
MODEL_REGISTRY = {
    "a": {
        "url": "https://zenodo.org/api/records/15699208/files/model_a.onnx/content",
        "md5": "7a8a627129817418963c7b77b962e0bd",
        "size_mb": 114.80,
    },
    "b": {
        "url": "https://zenodo.org/api/records/15699208/files/model_b.onnx/content",
        "md5": "aecb4c022e673f80f6cf73ced9e4c373",
        "size_mb": 87.28,
    },
    "c": {
        "url": "https://zenodo.org/api/records/15699208/files/model_c.onnx/content",
        "md5": "027ab7bf064944b9782ab51ef1dd8416",
        "size_mb": 139.52,  # estimated based on pattern
    },
    "d": {
        "url": "https://zenodo.org/api/records/15699208/files/model_d.onnx/content",
        "md5": "9d33f3c9e3db15b5903ada9043e93126",
        "size_mb": 157.33,
    },
    "e": {
        "url": "https://zenodo.org/api/records/15699208/files/model_e.onnx/content",
        "md5": "d40fff195b94c75d0842660b913a3bc5",
        "size_mb": 164.86,
    },
    "f": {
        "url": "https://zenodo.org/api/records/15699208/files/model_f.onnx/content",
        "md5": "e5391f7b501f2b50666438015b2221f1",
        "size_mb": 114.00,
    },
    "g": {
        "url": "https://zenodo.org/api/records/15699208/files/model_g.onnx/content",
        "md5": "1282ebaff6999e93d091dc0a689131af",
        "size_mb": 139.52,
    },
    "h": {
        "url": "https://zenodo.org/api/records/15699208/files/model_h.onnx/content",
        "md5": "60c5d33688573a7498190238840c8752",
        "size_mb": 114.80,
    },
    "i": {
        "url": "https://zenodo.org/api/records/15699208/files/model_i.onnx/content",
        "md5": "5c21e392e2517bdede04034f188876d3",
        "size_mb": 132.58,
    },
    "j": {
        "url": "https://zenodo.org/api/records/15699208/files/model_j.onnx/content",
        "md5": "7beac5476ff03f6fa96ec304bf44acb1",
        "size_mb": 301.24,
    },
    "k": {
        "url": "https://zenodo.org/api/records/15699208/files/model_k.onnx/content",
        "md5": "b8e0c9a24275a1813bd007088e1b19f7",
        "size_mb": 116.16,
    },
}


def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_model(model_idx, onnx_models_dir=None):
    """
    Download ONNX model from Zenodo if not already cached.
    
    :param model_idx: Model index (a-k)
    :param onnx_models_dir: Directory to store ONNX models (string or Path). 
                            If None, uses default directory.
    :return: Path to downloaded model file
    """
    if model_idx not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_idx} not found in registry")
    
    # Use default directory if not specified
    if onnx_models_dir is None:
        onnx_models_dir = get_default_onnx_dir()
    
    # Ensure onnx_models_dir is a Path object
    onnx_models_dir = Path(onnx_models_dir)
    model_info = MODEL_REGISTRY[model_idx]
    model_path = onnx_models_dir / f"model_{model_idx}.onnx"
    
    # Create directory if it doesn't exist
    onnx_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists and has correct hash
    if model_path.exists():
        logger.info(f"Model {model_idx} found locally, verifying hash...")
        if calculate_md5(model_path) == model_info["md5"]:
            logger.info(f"Model {model_idx} hash verified, using cached version")
            return model_path
        else:
            logger.warning(f"Model {model_idx} hash mismatch, re-downloading...")
            model_path.unlink()
    
    # Download model
    logger.info(
        f"Downloading model {model_idx} ({model_info['size_mb']:.1f} MB) from Zenodo..."
    )
    
    try:
        response = requests.get(model_info["url"], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\rDownload progress: {progress:.1f}%", end="", flush=True
                        )
        
        print()  # New line after progress
        
        # Verify downloaded file hash
        logger.info(f"Verifying hash for model {model_idx}...")
        if calculate_md5(model_path) != model_info["md5"]:
            model_path.unlink()
            raise ValueError(f"Downloaded model {model_idx} hash verification failed")
        
        logger.info(f"Model {model_idx} downloaded and verified successfully")
        return model_path
        
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download model {model_idx}: {str(e)}")


# Test the function
if __name__ == "__main__":
    # Test with default directory (./models)
    try:
        result = download_model("a")
        print(f"Download successful: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with explicit directory
    try:
        result = download_model("b", "./models")
        print(f"Download successful: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with Path object
    try:
        result = download_model("c", Path("./models"))
        print(f"Download successful: {result}")
    except Exception as e:
        print(f"Error: {e}")
