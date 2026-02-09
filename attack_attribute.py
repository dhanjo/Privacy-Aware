"""
Attribute Inference Attack on Re-ID embeddings.
Tests whether sensitive attributes (camera ID as proxy for location) can be inferred from embeddings.
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import config


def prepare_attribute_data(
    embeddings: np.ndarray,
    camera_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for attribute inference attack.

    Use camera ID as a proxy for location/context attribute.

    Args:
        embeddings: Feature embeddings (num_samples, embedding_dim)
        camera_ids: Camera IDs (num_samples,)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        camera_ids,
        test_size=0.3,
        random_state=config.RANDOM_SEED,
        stratify=camera_ids
    )

    return X_train, X_test, y_train, y_test


def train_attack_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int
) -> Tuple[MLPClassifier, Dict[str, float]]:
    """
    Train an attribute inference attack classifier.

    Args:
        X_train: Training embeddings
        y_train: Training labels (camera IDs)
        X_test: Test embeddings
        y_test: Test labels
        num_classes: Number of camera classes

    Returns:
        Tuple of (classifier, metrics)
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train attack classifier
    print("Training attack classifier...")
    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=config.ATTACK_CONFIG["attack_epochs"],
        learning_rate_init=config.ATTACK_CONFIG["attack_lr"],
        random_state=config.RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )

    classifier.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    # Calculate random baseline (uniform guess across classes)
    random_baseline = 1.0 / num_classes

    metrics = {
        'accuracy': accuracy,
        'random_baseline': random_baseline,
        'accuracy_improvement_over_random': accuracy - random_baseline,
    }

    return classifier, metrics


def attack(model_name: str) -> Dict[str, float]:
    """
    Perform attribute inference attack on a model's embeddings.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with attack metrics
    """
    print("=" * 80)
    print(f"Attribute Inference Attack: {model_name}")
    print("=" * 80 + "\n")

    # Load embeddings
    print("Loading embeddings...")

    query_emb_path = config.get_embeddings_path(model_name, "query")
    gallery_emb_path = config.get_embeddings_path(model_name, "gallery")

    if not os.path.exists(query_emb_path):
        raise FileNotFoundError(f"Query embeddings not found: {query_emb_path}")
    if not os.path.exists(gallery_emb_path):
        raise FileNotFoundError(f"Gallery embeddings not found: {gallery_emb_path}")

    query_embeddings = np.load(query_emb_path)
    query_camera_ids = np.load(query_emb_path.replace("_embeddings.npy", "_camera_ids.npy"))

    gallery_embeddings = np.load(gallery_emb_path)
    gallery_camera_ids = np.load(gallery_emb_path.replace("_embeddings.npy", "_camera_ids.npy"))

    # Combine query and gallery for attack
    all_embeddings = np.vstack([query_embeddings, gallery_embeddings])
    all_camera_ids = np.concatenate([query_camera_ids, gallery_camera_ids])

    # Remap camera IDs to start from 0
    unique_cameras = sorted(np.unique(all_camera_ids))
    camera_id_map = {cid: idx for idx, cid in enumerate(unique_cameras)}
    all_camera_ids = np.array([camera_id_map[cid] for cid in all_camera_ids])

    num_cameras = len(unique_cameras)

    print(f"✓ Total embeddings: {all_embeddings.shape}")
    print(f"✓ Number of cameras: {num_cameras}\n")

    # Prepare attack data
    print("Preparing attack data...")
    X_train, X_test, y_train, y_test = prepare_attribute_data(
        all_embeddings,
        all_camera_ids
    )

    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples\n")

    # Train attack classifier
    classifier, metrics = train_attack_classifier(
        X_train, y_train, X_test, y_test, num_cameras
    )

    # Print results
    print("\n" + "=" * 80)
    print("Attribute Inference Attack Results")
    print("=" * 80)
    print(f"Attribute: Camera ID (proxy for location/context)")
    print(f"Number of classes: {num_cameras}")
    print(f"Random baseline accuracy: {metrics['random_baseline'] * 100:.2f}%")
    print(f"Attack accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Improvement over random: {metrics['accuracy_improvement_over_random'] * 100:.2f}%")
    print("=" * 80)

    # Interpret results
    if metrics['accuracy_improvement_over_random'] > 0.3:
        print("\n⚠️  HIGH PRIVACY RISK: Attack significantly better than random")
        print("   The model leaks sensitive attribute information.")
    elif metrics['accuracy_improvement_over_random'] > 0.15:
        print("\n⚠️  MODERATE PRIVACY RISK: Attack moderately better than random")
        print("   The model leaks some attribute information.")
    else:
        print("\n✓ LOW PRIVACY RISK: Attack close to random guessing")
        print("  The model does not significantly leak attribute information.")

    print("\n")

    # Save results
    results_path = os.path.join(config.RESULTS_DIR, f"{model_name}_attribute_attack.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Results saved to {results_path}\n")

    return metrics


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Perform attribute inference attack on Re-ID embeddings"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model (e.g., 'baseline', 'dp_eps_8.0')"
    )

    args = parser.parse_args()

    try:
        attack(model_name=args.model_name)
        print("=" * 80)
        print("✓ Attack completed!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n✗ Attack failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
