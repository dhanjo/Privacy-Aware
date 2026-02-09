"""
Evaluate Re-ID performance using standard metrics.
Computes mAP, Rank-1/5/10 accuracy, and CMC curve.
"""

import os
import sys
import json
import argparse
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import config


def compute_distance_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine distance between query and gallery.

    Args:
        query_embeddings: Query embeddings (num_query, embedding_dim)
        gallery_embeddings: Gallery embeddings (num_gallery, embedding_dim)

    Returns:
        Distance matrix (num_query, num_gallery)
    """
    # Compute cosine similarity
    similarity = cosine_similarity(query_embeddings, gallery_embeddings)

    # Convert to distance (1 - similarity)
    distance = 1 - similarity

    return distance


def evaluate_market1501(
    dist_mat: np.ndarray,
    query_person_ids: np.ndarray,
    query_camera_ids: np.ndarray,
    gallery_person_ids: np.ndarray,
    gallery_camera_ids: np.ndarray,
    max_rank: int = 50
) -> Tuple[float, np.ndarray]:
    """
    Evaluate using Market-1501 protocol.

    Args:
        dist_mat: Distance matrix (num_query, num_gallery)
        query_person_ids: Query person IDs
        query_camera_ids: Query camera IDs
        gallery_person_ids: Gallery person IDs
        gallery_camera_ids: Gallery camera IDs
        max_rank: Maximum rank for CMC computation

    Returns:
        Tuple of (mAP, cmc_scores)
    """
    num_query = dist_mat.shape[0]

    all_ap = []
    all_cmc = []

    for q_idx in range(num_query):
        # Get query info
        q_pid = query_person_ids[q_idx]
        q_camid = query_camera_ids[q_idx]

        # Sort gallery by distance
        order = np.argsort(dist_mat[q_idx])

        # Create masks for valid gallery images
        gallery_pids = gallery_person_ids[order]
        gallery_camids = gallery_camera_ids[order]

        # Remove gallery samples that have the same pid and camid with query
        valid_mask = ~((gallery_pids == q_pid) & (gallery_camids == q_camid))

        # Find matches (same person ID, different camera)
        match_mask = (gallery_pids == q_pid)

        # Apply valid mask
        match_mask = match_mask & valid_mask

        if not np.any(match_mask):
            # No valid matches for this query
            continue

        # Compute CMC
        cmc = match_mask.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])

        # Compute average precision
        num_matches = match_mask.sum()
        tmp_cmc = match_mask.cumsum()
        tmp_cmc = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        tmp_cmc = tmp_cmc * match_mask
        ap = tmp_cmc.sum() / num_matches

        all_ap.append(ap)

    if len(all_ap) == 0:
        raise ValueError("No valid query-gallery pairs found!")

    all_cmc = np.vstack(all_cmc)
    cmc_scores = all_cmc.mean(axis=0)
    mAP = np.mean(all_ap)

    return mAP, cmc_scores


def plot_cmc_curve(cmc_scores: np.ndarray, model_name: str, save_path: str):
    """
    Plot CMC curve.

    Args:
        cmc_scores: CMC scores for each rank
        model_name: Name of the model
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    ranks = np.arange(1, len(cmc_scores) + 1)
    plt.plot(ranks, cmc_scores * 100, linewidth=2, label=model_name)

    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Matching Rate (%)', fontsize=12)
    plt.title('Cumulative Matching Characteristic (CMC) Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save as both PNG and PDF
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ CMC curve saved to {save_path}")


def evaluate(
    model_name: str,
    data_root: str = None,
    plot: bool = True
) -> Dict[str, float]:
    """
    Evaluate Re-ID performance from saved embeddings.

    Args:
        model_name: Name of the model
        data_root: Path to Market-1501 dataset (unused, kept for compatibility)
        plot: Whether to plot CMC curve

    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 80)
    print(f"Evaluating Re-ID Performance: {model_name}")
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
    query_camera_ids = np.load(query_emb_path.replace("_embeddings.npy", "_camera_ids.npy"))

    gallery_embeddings = np.load(gallery_emb_path)
    gallery_person_ids = np.load(gallery_emb_path.replace("_embeddings.npy", "_person_ids.npy"))
    gallery_camera_ids = np.load(gallery_emb_path.replace("_embeddings.npy", "_camera_ids.npy"))

    print(f"✓ Query embeddings: {query_embeddings.shape}")
    print(f"✓ Gallery embeddings: {gallery_embeddings.shape}\n")

    # Compute distance matrix
    print("Computing distance matrix...")
    dist_mat = compute_distance_matrix(query_embeddings, gallery_embeddings)
    print(f"✓ Distance matrix: {dist_mat.shape}\n")

    # Evaluate
    print("Evaluating...")
    mAP, cmc_scores = evaluate_market1501(
        dist_mat,
        query_person_ids,
        query_camera_ids,
        gallery_person_ids,
        gallery_camera_ids
    )

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"mAP: {mAP * 100:.2f}%")
    print(f"Rank-1: {cmc_scores[0] * 100:.2f}%")
    print(f"Rank-5: {cmc_scores[4] * 100:.2f}%")
    print(f"Rank-10: {cmc_scores[9] * 100:.2f}%")
    print("=" * 80 + "\n")

    # Save metrics
    metrics = {
        'mAP': float(mAP),
        'rank1': float(cmc_scores[0]),
        'rank5': float(cmc_scores[4]),
        'rank10': float(cmc_scores[9]),
    }

    metrics_path = os.path.join(config.RESULTS_DIR, f"{model_name}_reid_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}\n")

    # Plot CMC curve
    if plot:
        print("Plotting CMC curve...")
        plot_path = os.path.join(config.RESULTS_DIR, f"{model_name}_cmc_curve.png")
        plot_cmc_curve(cmc_scores, model_name, plot_path)
        print()

    return metrics


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate Re-ID performance"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model (e.g., 'baseline', 'dp_eps_8.0')"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.DATA_ROOT,
        help="Path to Market-1501 dataset"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't plot CMC curve"
    )

    args = parser.parse_args()

    try:
        evaluate(
            model_name=args.model_name,
            data_root=args.data_root,
            plot=not args.no_plot
        )
        print("=" * 80)
        print("✓ Evaluation completed!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
