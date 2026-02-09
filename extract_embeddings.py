"""
Extract feature embeddings from trained Re-ID models.
Supports both baseline and DP-trained models.
"""

import os
import sys
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import config
from dataset import create_dataloaders
from model import create_model


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, num_classes)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    num_classes = checkpoint['num_classes']
    model = create_model(num_classes, device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")

    return model, num_classes


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract embeddings from a data loader.

    Args:
        model: Trained Re-ID model
        data_loader: Data loader
        device: Device to run inference on

    Returns:
        Tuple of (embeddings, person_ids, camera_ids)
    """
    model.eval()

    embeddings_list = []
    person_ids_list = []
    camera_ids_list = []

    for images, person_ids, camera_ids in tqdm(data_loader, desc="Extracting embeddings"):
        images = images.to(device)

        # Extract embeddings
        embeddings = model(images, return_embeddings=True)

        embeddings_list.append(embeddings.cpu().numpy())
        person_ids_list.append(person_ids.numpy())
        camera_ids_list.append(camera_ids.numpy())

    # Concatenate all batches
    embeddings = np.vstack(embeddings_list)
    person_ids = np.concatenate(person_ids_list)
    camera_ids = np.concatenate(camera_ids_list)

    return embeddings, person_ids, camera_ids


def extract_and_save(
    model_path: str,
    model_name: str,
    data_root: str,
    device: torch.device,
    batch_size: int = 64
):
    """
    Extract and save embeddings for query and gallery sets.

    Args:
        model_path: Path to model checkpoint
        model_name: Name of the model (for saving files)
        data_root: Path to Market-1501 dataset
        device: Device to run inference on
        batch_size: Batch size for extraction
    """
    print("=" * 80)
    print(f"Extracting Embeddings: {model_name}")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("=" * 80 + "\n")

    # Load model
    model, num_classes = load_model(model_path, device)

    # Create dataloaders
    print("\nLoading dataset...")
    _, query_loader, gallery_loader, _ = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4
    )
    print(f"✓ Query set: {len(query_loader.dataset)} images")
    print(f"✓ Gallery set: {len(gallery_loader.dataset)} images\n")

    # Extract query embeddings
    print("Extracting query embeddings...")
    query_embeddings, query_person_ids, query_camera_ids = extract_embeddings(
        model, query_loader, device
    )
    print(f"✓ Query embeddings shape: {query_embeddings.shape}\n")

    # Extract gallery embeddings
    print("Extracting gallery embeddings...")
    gallery_embeddings, gallery_person_ids, gallery_camera_ids = extract_embeddings(
        model, gallery_loader, device
    )
    print(f"✓ Gallery embeddings shape: {gallery_embeddings.shape}\n")

    # Save embeddings
    print("Saving embeddings...")

    query_emb_path = config.get_embeddings_path(model_name, "query")
    np.save(query_emb_path, query_embeddings)
    np.save(query_emb_path.replace("_embeddings.npy", "_person_ids.npy"), query_person_ids)
    np.save(query_emb_path.replace("_embeddings.npy", "_camera_ids.npy"), query_camera_ids)
    print(f"✓ Saved query embeddings to {query_emb_path}")

    gallery_emb_path = config.get_embeddings_path(model_name, "gallery")
    np.save(gallery_emb_path, gallery_embeddings)
    np.save(gallery_emb_path.replace("_embeddings.npy", "_person_ids.npy"), gallery_person_ids)
    np.save(gallery_emb_path.replace("_embeddings.npy", "_camera_ids.npy"), gallery_camera_ids)
    print(f"✓ Saved gallery embeddings to {gallery_emb_path}\n")

    print("=" * 80)
    print("✓ Embedding extraction completed!")
    print("=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from trained Re-ID models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model (for saving files)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.DATA_ROOT,
        help="Path to Market-1501 dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EVAL_CONFIG["batch_size"],
        help="Batch size for extraction"
    )

    args = parser.parse_args()

    device = config.DEVICE

    try:
        extract_and_save(
            model_path=args.model_path,
            model_name=args.model_name,
            data_root=args.data_root,
            device=device,
            batch_size=args.batch_size
        )
        return 0
    except Exception as e:
        print(f"\n✗ Embedding extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
