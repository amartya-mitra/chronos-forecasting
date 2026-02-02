#!/usr/bin/env python3
"""
Model Utilities for Reasoning Finetuning

This module provides utilities for:
1. Checking if base Chronos model is available locally
2. Checking if finetuned reasoning model is available
3. Downloading/training models if not present
"""

import os
import sys
from pathlib import Path

# Ensure we can import from the project
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Model paths (relative to project root)
BASE_MODEL_ID = "amazon/chronos-t5-small"
FINETUNED_MODEL_DIR = "output/reasoning-v4-from-amazon"

# Data paths (relative to reasoning-finetuning/)
REASONING_DIR = Path(__file__).parent
TRAINING_DATA_PATH = REASONING_DIR / "data" / "gifteval-reasoning.arrow"
FIGURES_DIR = REASONING_DIR / "figures"
LOGS_DIR = REASONING_DIR / "logs"

TRAINING_CONFIG_PATH = "reasoning-finetuning/configs/default.yaml"


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def check_base_model_available() -> bool:
    """
    Check if the base Chronos model is available in HuggingFace cache.
    
    Returns:
        True if model is cached locally, False otherwise
    """
    try:
        from huggingface_hub import try_to_load_from_cache, scan_cache_dir
        
        # Check if model files exist in cache
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if "chronos-t5-small" in repo.repo_id:
                print(f"✓ Base model found in cache: {repo.repo_path}")
                return True
        
        print(f"✗ Base model not found in local cache")
        return False
    except Exception as e:
        print(f"Warning: Could not check cache: {e}")
        return False


def download_base_model():
    """Download the base Chronos model from HuggingFace."""
    print(f"Downloading base model: {BASE_MODEL_ID}")
    print("This may take a few minutes...")
    
    try:
        import torch
        from chronos import ChronosPipeline
        
        pipeline = ChronosPipeline.from_pretrained(
            BASE_MODEL_ID,
            device_map="cpu",  # Download only, don't load to GPU
            torch_dtype=torch.float32,
        )
        print(f"✓ Base model downloaded successfully")
        print(f"  Vocab size: {pipeline.model.model.shared.num_embeddings}")
        return True
    except Exception as e:
        print(f"✗ Failed to download base model: {e}")
        return False


def check_finetuned_model_available() -> bool:
    """
    Check if the finetuned reasoning model is available locally.
    
    Returns:
        True if model exists, False otherwise
    """
    model_path = PROJECT_ROOT / FINETUNED_MODEL_DIR
    config_file = model_path / "config.json"
    model_file = model_path / "model.safetensors"
    
    if model_path.exists() and config_file.exists() and model_file.exists():
        print(f"✓ Finetuned model found: {model_path}")
        return True
    else:
        print(f"✗ Finetuned model not found at: {model_path}")
        return False


def check_training_data_available() -> bool:
    """Check if training data exists."""
    if TRAINING_DATA_PATH.exists():
        size_mb = TRAINING_DATA_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Training data found: {TRAINING_DATA_PATH} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ Training data not found at: {TRAINING_DATA_PATH}")
        return False


def generate_training_data():
    """Generate the reasoning training dataset."""
    print("Generating training data...")
    
    prepare_script = PROJECT_ROOT / "reasoning-finetuning" / "prepare_dataset.py"
    if prepare_script.exists():
        os.system(f"python {prepare_script}")
    else:
        # Fallback to original script
        legacy_script = PROJECT_ROOT / "data_reasoning.py"
        if legacy_script.exists():
            os.system(f"python {legacy_script}")
        else:
            print("✗ No data preparation script found")
            return False
    
    return check_training_data_available()


def train_finetuned_model():
    """Train the finetuned reasoning model."""
    print("Training finetuned model (this will take ~50 minutes)...")
    
    train_script = PROJECT_ROOT / "reasoning-finetuning" / "train.py"
    config_path = PROJECT_ROOT / TRAINING_CONFIG_PATH
    
    if train_script.exists() and config_path.exists():
        os.system(f"python {train_script} {config_path}")
    else:
        # Fallback to original training script
        legacy_script = PROJECT_ROOT / "scripts" / "training" / "train_reasoning.py"
        legacy_config = PROJECT_ROOT / "scripts" / "training" / "configs" / "chronos-2-small-gifteval-subset.yaml"
        if legacy_script.exists() and legacy_config.exists():
            os.system(f"python {legacy_script} {legacy_config}")
        else:
            print("✗ No training script found")
            return False
    
    return check_finetuned_model_available()


def ensure_base_model():
    """Ensure base model is available, download if not."""
    if not check_base_model_available():
        return download_base_model()
    return True


def ensure_finetuned_model():
    """Ensure finetuned model is available, train if not."""
    if check_finetuned_model_available():
        return True
    
    # Need to train - first ensure prerequisites
    if not ensure_base_model():
        print("Cannot train: base model not available")
        return False
    
    if not check_training_data_available():
        print("Training data not found, generating...")
        if not generate_training_data():
            print("Cannot train: training data generation failed")
            return False
    
    return train_finetuned_model()


def load_finetuned_model():
    """Load the finetuned model, ensuring it exists first."""
    import torch
    from chronos import ChronosPipeline
    
    if not ensure_finetuned_model():
        raise RuntimeError("Could not load or create finetuned model")
    
    model_path = PROJECT_ROOT / FINETUNED_MODEL_DIR
    pipeline = ChronosPipeline.from_pretrained(
        str(model_path),
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
        torch_dtype=torch.float32,
    )
    return pipeline


def load_base_model():
    """Load the base Chronos model, ensuring it exists first."""
    import torch
    from chronos import ChronosPipeline
    
    if not ensure_base_model():
        raise RuntimeError("Could not load or download base model")
    
    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
        torch_dtype=torch.float32,
    )
    return pipeline


def main():
    """CLI for model utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model utilities for reasoning finetuning")
    parser.add_argument("--check-base", action="store_true", help="Check if base model is available")
    parser.add_argument("--check-finetuned", action="store_true", help="Check if finetuned model is available")
    parser.add_argument("--check-data", action="store_true", help="Check if training data is available")
    parser.add_argument("--download-base", action="store_true", help="Download base model if not present")
    parser.add_argument("--ensure-finetuned", action="store_true", help="Ensure finetuned model exists (train if needed)")
    parser.add_argument("--status", action="store_true", help="Show status of all components")
    
    args = parser.parse_args()
    
    if args.status or len(sys.argv) == 1:
        print("=" * 60)
        print("REASONING FINETUNING - STATUS")
        print("=" * 60)
        print("\n[Base Model]")
        check_base_model_available()
        print("\n[Training Data]")
        check_training_data_available()
        print("\n[Finetuned Model]")
        check_finetuned_model_available()
        print()
        return
    
    if args.check_base:
        check_base_model_available()
    
    if args.check_finetuned:
        check_finetuned_model_available()
    
    if args.check_data:
        check_training_data_available()
    
    if args.download_base:
        ensure_base_model()
    
    if args.ensure_finetuned:
        ensure_finetuned_model()


if __name__ == "__main__":
    main()
