"""
Membership Inference Attack on Re-ID embeddings.
Tests whether an identity was in the training set by training a binary classifier.
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config


def prepare_membership_data(
    embeddings: np.ndarray,
    person_ids: np.ndarray,
    split_ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for membership inference attack.

    Split identities into "member" and "non-member" sets.
    The attack classifier will learn to distinguish between them.

    Args:
        embeddings: Feature embeddings (num_samples, embedding_dim)
        person_ids: Person IDs (num_samples,)
        split_ratio: Ratio to split identities into members/non-members

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Get unique person IDs
    unique_person_ids = np.unique(person_ids)
    num_unique = len(unique_person_ids)

    # Split person IDs into two sets
    np.random.seed(config.RANDOM_SEED)
    member_ids = np.random.choice(
        unique_person_ids,
        size=int(num_unique * split_ratio),
        replace=False
    )
    member_set = set(member_ids)

    # Create labels: 1 for members, 0 for non-members
    labels = np.array([1 if pid in member_set else 0 for pid in person_ids])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.3,
        random_state=config.RANDOM_SEED,
        stratify=labels
    )

    return X_train, X_test, y_train, y_test


def train_attack_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[MLPClassifier, Dict[str, float]]:
    """
    Train a membership inference attack classifier.

    Args:
        X_train: Training embeddings
        y_train: Training labels (1=member, 0=non-member)
        X_test: Test embeddings
        y_test: Test labels

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
    y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
    }

    return classifier, metrics


def attack(model_name: str) -> Dict[str, float]:
    """
    Perform membership inference attack on a model's embeddings.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with attack metrics
    """
    print("=" * 80)
    print(f"Membership Inference Attack: {model_name}")
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
    query_person_ids = np.load(query_emb_path.replace("_embeddings.npy", "_person_ids.npy"))

    gallery_embeddings = np.load(gallery_emb_path)
    gallery_person_ids = np.load(gallery_emb_path.replace("_embeddings.npy", "_person_ids.npy"))

    # Combine query and gallery for attack
    all_embeddings = np.vstack([query_embeddings, gallery_embeddings])
    all_person_ids = np.concatenate([query_person_ids, gallery_person_ids])

    print(f"✓ Total embeddings: {all_embeddings.shape}")
    print(f"✓ Unique identities: {len(np.unique(all_person_ids))}\n")

    # Prepare attack data
    print("Preparing attack data...")
    X_train, X_test, y_train, y_test = prepare_membership_data(
        all_embeddings,
        all_person_ids,
        split_ratio=config.ATTACK_CONFIG["train_test_split"]
    )

    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Member ratio (train): {y_train.mean():.2%}")
    print(f"✓ Member ratio (test): {y_test.mean():.2%}\n")

    # Train attack classifier
    classifier, metrics = train_attack_classifier(X_train, y_train, X_test, y_test)

    # Print results
    print("\n" + "=" * 80)
    print("Membership Inference Attack Results")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score: {metrics['f1_score'] * 100:.2f}%")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print("=" * 80)

    # Interpret results
    if metrics['accuracy'] > 0.6:
        print("\n⚠️  HIGH PRIVACY RISK: Attack accuracy > 60%")
        print("   The model leaks membership information.")
    elif metrics['accuracy'] > 0.55:
        print("\n⚠️  MODERATE PRIVACY RISK: Attack accuracy > 55%")
        print("   The model leaks some membership information.")
    else:
        print("\n✓ LOW PRIVACY RISK: Attack accuracy ≈ random guessing")
        print("  The model does not significantly leak membership information.")

    print("\n")

    # Save results
    results_path = os.path.join(config.RESULTS_DIR, f"{model_name}_membership_attack.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Results saved to {results_path}\n")

    return metrics


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Perform membership inference attack on Re-ID embeddings"
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
