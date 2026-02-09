"""
Train Re-ID model with Differential Privacy using Opacus.
Implements privacy-preserving training with configurable epsilon budget.
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
from tqdm import tqdm
import numpy as np

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import config
from dataset import create_dataloaders
from model import create_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model_private_compatible(model: nn.Module) -> nn.Module:
    """
    Make model compatible with Opacus by replacing BatchNorm with GroupNorm.

    Args:
        model: Original model

    Returns:
        Modified model compatible with Opacus
    """
    # Use Opacus's ModuleValidator to fix the model
    model = ModuleValidator.fix(model)
    return model


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    privacy_engine: PrivacyEngine
) -> Dict[str, float]:
    """
    Train for one epoch with differential privacy.

    Args:
        model: Re-ID model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        privacy_engine: Opacus privacy engine

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for images, person_ids, _ in pbar:
        images = images.to(device)
        person_ids = person_ids.to(device)

        # Forward pass
        logits, _ = model(images)

        # Compute loss (only cross-entropy for DP training to avoid complexity)
        loss = criterion(logits, person_ids)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()

        _, predicted = logits.max(1)
        total += person_ids.size(0)
        correct += predicted.eq(person_ids).sum().item()

        # Get current epsilon
        epsilon = privacy_engine.get_epsilon(delta=config.DP_CONFIG["delta"])

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%',
            'ε': f'{epsilon:.2f}'
        })

    # Get final epsilon for this epoch
    epsilon = privacy_engine.get_epsilon(delta=config.DP_CONFIG["delta"])

    metrics = {
        'loss': total_loss / len(train_loader),
        'accuracy': 100.0 * correct / total,
        'epsilon': epsilon
    }

    return metrics


def train(
    data_root: str,
    target_epsilon: float,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    delta: float,
    device: torch.device,
    save_path: str,
    log_path: str
) -> Dict[str, List[float]]:
    """
    Train Re-ID model with differential privacy.

    Args:
        data_root: Path to Market-1501 dataset
        target_epsilon: Target privacy budget
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_grad_norm: Maximum gradient norm for clipping
        delta: Target delta for (ε, δ)-DP
        device: Device to train on
        save_path: Path to save the model
        log_path: Path to save training log

    Returns:
        Dictionary with training history
    """
    print("=" * 80)
    print("Training Re-ID Model with Differential Privacy")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target Epsilon: {target_epsilon}")
    print(f"Delta: {delta}")
    print(f"Max Grad Norm: {max_grad_norm}")
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

    # Make model compatible with Opacus
    print("Making model compatible with Opacus (replacing BatchNorm with GroupNorm)...")
    model = make_model_private_compatible(model)
    model = model.to(device)
    print(f"✓ Model is now Opacus-compatible\n")

    # Create loss function (only cross-entropy for DP training)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create privacy engine
    print("Attaching privacy engine...")
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=delta,
        epochs=num_epochs,
        max_grad_norm=max_grad_norm,
    )

    print(f"✓ Privacy engine attached")
    print(f"✓ Noise multiplier: {optimizer.noise_multiplier:.4f}\n")

    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'epsilon': []
    }

    # Training loop
    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, privacy_engine
        )

        # Save metrics
        history['loss'].append(metrics['loss'])
        history['accuracy'].append(metrics['accuracy'])
        history['epsilon'].append(metrics['epsilon'])

        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Privacy: (ε = {metrics['epsilon']:.2f}, δ = {delta})\n")

    elapsed_time = time.time() - start_time
    final_epsilon = history['epsilon'][-1]

    print(f"Training completed in {elapsed_time:.2f} seconds "
          f"({elapsed_time / 60:.2f} minutes)")
    print(f"Final privacy guarantee: (ε = {final_epsilon:.2f}, δ = {delta})\n")

    # Save model (need to extract the original model from DDP wrapper)
    print(f"Saving model to {save_path}...")

    # Get the underlying model (Opacus wraps it)
    if hasattr(model, '_module'):
        model_to_save = model._module
    else:
        model_to_save = model

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_to_save.state_dict(),
        'num_classes': num_classes,
        'history': history,
        'epsilon': final_epsilon,
        'delta': delta,
        'max_grad_norm': max_grad_norm,
    }, save_path)
    print("✓ Model saved\n")

    # Save training log
    print(f"Saving training log to {log_path}...")
    with open(log_path, 'w') as f:
        json.dump({
            'config': {
                'target_epsilon': target_epsilon,
                'final_epsilon': final_epsilon,
                'delta': delta,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'max_grad_norm': max_grad_norm,
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
        description="Train Re-ID model with differential privacy"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.DATA_ROOT,
        help="Path to Market-1501 dataset"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8.0,
        help="Target privacy budget (epsilon)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=config.DP_CONFIG["num_epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.DP_CONFIG["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.DP_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config.DP_CONFIG["weight_decay"],
        help="Weight decay"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=config.DP_CONFIG["max_grad_norm"],
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=config.DP_CONFIG["delta"],
        help="Target delta for (ε, δ)-DP"
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
    save_path = config.get_model_path("dp", args.epsilon)
    log_path = config.get_log_path("dp", args.epsilon)

    # Train model
    try:
        train(
            data_root=args.data_root,
            target_epsilon=args.epsilon,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            delta=args.delta,
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
