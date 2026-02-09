"""
Configuration file for Privacy-Aware Person Re-Identification Baseline.
All hyperparameters and paths in one place.
"""

import os
import torch

# Random seed for reproducibility
RANDOM_SEED = 42

# Paths
DATA_ROOT = "./data/Market-1501"
RESULTS_DIR = "./results"

# Training hyperparameters - Baseline
BASELINE_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.0003,
    "num_epochs": 30,
    "weight_decay": 5e-4,
    "step_size": 10,  # for StepLR scheduler
    "gamma": 0.1,     # for StepLR scheduler
}

# Training hyperparameters - Differential Privacy
DP_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.0003,
    "num_epochs": 15,
    "weight_decay": 5e-4,
    "max_grad_norm": 1.0,
    "delta": 1e-5,
}

# Epsilon values to test for privacy-utility trade-off
EPSILON_VALUES = [1.0, 3.0, 5.0, 8.0, 10.0, 50.0]

# Model architecture
MODEL_CONFIG = {
    "backbone": "resnet50",
    "pretrained": True,
    "embedding_dim": 512,  # Project from 2048 to 512
    "dropout_rate": 0.5,
}

# Loss function weights
LOSS_CONFIG = {
    "ce_weight": 1.0,      # Cross-entropy loss weight
    "triplet_weight": 1.0,  # Triplet loss weight
    "triplet_margin": 0.3,  # Triplet loss margin
}

# Image preprocessing
IMAGE_CONFIG = {
    "height": 256,
    "width": 128,
    "mean": [0.485, 0.456, 0.406],  # ImageNet mean
    "std": [0.229, 0.224, 0.225],    # ImageNet std
}

# Evaluation
EVAL_CONFIG = {
    "batch_size": 64,
    "rerank": False,  # Re-ranking disabled by default (slower but more accurate)
}

# Privacy attacks
ATTACK_CONFIG = {
    "train_test_split": 0.5,  # Split for membership inference
    "attack_epochs": 50,
    "attack_lr": 0.001,
    "attack_batch_size": 128,
}

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)


def get_model_path(model_type: str, epsilon: float = None) -> str:
    """
    Get the path for a model checkpoint.

    Args:
        model_type: 'baseline' or 'dp'
        epsilon: Privacy budget (only for DP models)

    Returns:
        Path to model checkpoint
    """
    if model_type == "baseline":
        return os.path.join(RESULTS_DIR, "baseline_model.pth")
    elif model_type == "dp":
        if epsilon is None:
            raise ValueError("Epsilon must be provided for DP models")
        return os.path.join(RESULTS_DIR, f"dp_model_eps_{epsilon}.pth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_embeddings_path(model_name: str, split: str) -> str:
    """
    Get the path for saved embeddings.

    Args:
        model_name: Name of the model (e.g., 'baseline', 'dp_eps_8.0')
        split: 'query' or 'gallery'

    Returns:
        Path to embeddings file
    """
    return os.path.join(RESULTS_DIR, f"{model_name}_{split}_embeddings.npy")


def get_log_path(model_type: str, epsilon: float = None) -> str:
    """
    Get the path for training log.

    Args:
        model_type: 'baseline' or 'dp'
        epsilon: Privacy budget (only for DP models)

    Returns:
        Path to log file
    """
    if model_type == "baseline":
        return os.path.join(RESULTS_DIR, "baseline_training_log.json")
    elif model_type == "dp":
        if epsilon is None:
            raise ValueError("Epsilon must be provided for DP models")
        return os.path.join(RESULTS_DIR, f"dp_training_log_eps_{epsilon}.json")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
