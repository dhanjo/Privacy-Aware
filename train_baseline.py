"""
Train baseline Re-ID model without any privacy mechanisms.
Uses cross-entropy loss + triplet loss for person re-identification.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

import config
from dataset import create_dataloaders
from model import create_model, CombinedLoss


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Re-ID model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_triplet_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for images, person_ids, _ in pbar:
        images = images.to(device)
        person_ids = person_ids.to(device)

        # Forward pass
        logits, embeddings = model(images)

        # Compute loss
        loss, ce_loss, triplet_loss = criterion(logits, embeddings, person_ids)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_triplet_loss += triplet_loss.item()

        _, predicted = logits.max(1)
        total += person_ids.size(0)
        correct += predicted.eq(person_ids).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    metrics = {
        'loss': total_loss / len(train_loader),
        'ce_loss': total_ce_loss / len(train_loader),
        'triplet_loss': total_triplet_loss / len(train_loader),
        'accuracy': 100.0 * correct / total
    }

    return metrics


def train(
    data_root: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    step_size: int,
    gamma: float,
    device: torch.device,
    save_path: str,
    log_path: str
) -> Dict[str, List[float]]:
    """
    Train baseline Re-ID model.

    Args:
        data_root: Path to Market-1501 dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        step_size: Step size for learning rate scheduler
        gamma: Gamma for learning rate scheduler
        device: Device to train on
        save_path: Path to save the model
        log_path: Path to save training log

    Returns:
        Dictionary with training history
    """
    print("=" * 80)
    print("Training Baseline Re-ID Model (No Privacy)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Model will be saved to: {save_path}")
    print("=" * 80 + "\n")

    # Create dataloaders
    print("Loading dataset...")
    train_loader, _, _, num_classes = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4
    )
    print(f"✓ Loaded {len(train_loader.dataset)} training images")
    print(f"✓ Number of identities: {num_classes}\n")

    # Create model
    print("Creating model...")
    model = create_model(num_classes, device)
    print(f"✓ Model created\n")

    # Create loss function
    criterion = CombinedLoss(
        num_classes=num_classes,
        ce_weight=config.LOSS_CONFIG["ce_weight"],
        triplet_weight=config.LOSS_CONFIG["triplet_weight"],
        triplet_margin=config.LOSS_CONFIG["triplet_margin"]
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training history
    history = {
        'loss': [],
        'ce_loss': [],
        'triplet_loss': [],
        'accuracy': [],
        'learning_rate': []
    }

    # Training loop
    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save metrics
        history['loss'].append(metrics['loss'])
        history['ce_loss'].append(metrics['ce_loss'])
        history['triplet_loss'].append(metrics['triplet_loss'])
        history['accuracy'].append(metrics['accuracy'])
        history['learning_rate'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Loss: {metrics['loss']:.4f} "
              f"(CE: {metrics['ce_loss']:.4f}, Triplet: {metrics['triplet_loss']:.4f})")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}\n")

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds "
          f"({elapsed_time / 60:.2f} minutes)\n")

    # Save model
    print(f"Saving model to {save_path}...")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes,
        'history': history,
    }, save_path)
    print("✓ Model saved\n")

    # Save training log
    print(f"Saving training log to {log_path}...")
    with open(log_path, 'w') as f:
        json.dump({
            'config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'step_size': step_size,
                'gamma': gamma,
                'num_classes': num_classes,
            },
            'history': history,
            'training_time_seconds': elapsed_time,
        }, f, indent=2)
    print("✓ Training log saved\n")

    return history


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Train baseline Re-ID model without privacy"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.DATA_ROOT,
        help="Path to Market-1501 dataset"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=config.BASELINE_CONFIG["num_epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BASELINE_CONFIG["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.BASELINE_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config.BASELINE_CONFIG["weight_decay"],
        help="Weight decay"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = config.DEVICE

    # Get save paths
    save_path = config.get_model_path("baseline")
    log_path = config.get_log_path("baseline")

    # Train model
    try:
        train(
            data_root=args.data_root,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            step_size=config.BASELINE_CONFIG["step_size"],
            gamma=config.BASELINE_CONFIG["gamma"],
            device=device,
            save_path=save_path,
            log_path=log_path
        )
        print("=" * 80)
        print("✓ Training completed successfully!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
