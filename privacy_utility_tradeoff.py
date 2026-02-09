"""
Privacy-Utility Trade-off Analysis.
Trains models across multiple epsilon values and analyzes the trade-off between
privacy (lower attack success) and utility (higher Re-ID accuracy).
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command to run (as list)
        description: Description of the command

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}")

    # Replace 'python' with sys.executable to use the current Python interpreter
    if cmd[0] == 'python':
        cmd[0] = sys.executable

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Command failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Command failed: {e}")
        return False


def train_and_evaluate_baseline() -> Dict[str, float]:
    """
    Train and evaluate baseline model.

    Returns:
        Dictionary with all metrics
    """
    print("\n" + "=" * 80)
    print("BASELINE MODEL (NO PRIVACY)")
    print("=" * 80)

    model_name = "baseline"

    # Check if baseline model already exists
    model_path = config.get_model_path("baseline")
    if os.path.exists(model_path):
        print(f"\n✓ Baseline model already exists at {model_path}")
        print("Skipping training...\n")
    else:
        # Train baseline
        if not run_command(
            ["python", "train_baseline.py"],
            "Training baseline model"
        ):
            raise RuntimeError("Baseline training failed")

    # Extract embeddings
    if not run_command(
        ["python", "extract_embeddings.py", "--model-path", model_path, "--model-name", model_name],
        "Extracting baseline embeddings"
    ):
        raise RuntimeError("Baseline embedding extraction failed")

    # Evaluate Re-ID
    if not run_command(
        ["python", "evaluate_reid.py", "--model-name", model_name],
        "Evaluating baseline Re-ID performance"
    ):
        raise RuntimeError("Baseline Re-ID evaluation failed")

    # Run membership attack
    if not run_command(
        ["python", "attack_membership.py", "--model-name", model_name],
        "Running membership attack on baseline"
    ):
        raise RuntimeError("Baseline membership attack failed")

    # Run attribute attack
    if not run_command(
        ["python", "attack_attribute.py", "--model-name", model_name],
        "Running attribute attack on baseline"
    ):
        raise RuntimeError("Baseline attribute attack failed")

    # Collect metrics
    metrics = collect_metrics(model_name, epsilon=None)

    return metrics


def train_and_evaluate_dp(epsilon: float) -> Dict[str, float]:
    """
    Train and evaluate DP model with given epsilon.

    Args:
        epsilon: Privacy budget

    Returns:
        Dictionary with all metrics
    """
    print("\n" + "=" * 80)
    print(f"DP MODEL (ε = {epsilon})")
    print("=" * 80)

    model_name = f"dp_eps_{epsilon}"
    model_path = config.get_model_path("dp", epsilon)

    # Check if model already exists
    if os.path.exists(model_path):
        print(f"\n✓ DP model (ε={epsilon}) already exists at {model_path}")
        print("Skipping training...\n")
    else:
        # Train DP model
        if not run_command(
            ["python", "train_dp.py", "--epsilon", str(epsilon)],
            f"Training DP model (ε = {epsilon})"
        ):
            raise RuntimeError(f"DP training failed for epsilon={epsilon}")

    # Extract embeddings
    if not run_command(
        ["python", "extract_embeddings.py", "--model-path", model_path, "--model-name", model_name],
        f"Extracting DP embeddings (ε = {epsilon})"
    ):
        raise RuntimeError(f"DP embedding extraction failed for epsilon={epsilon}")

    # Evaluate Re-ID
    if not run_command(
        ["python", "evaluate_reid.py", "--model-name", model_name],
        f"Evaluating DP Re-ID performance (ε = {epsilon})"
    ):
        raise RuntimeError(f"DP Re-ID evaluation failed for epsilon={epsilon}")

    # Run membership attack
    if not run_command(
        ["python", "attack_membership.py", "--model-name", model_name],
        f"Running membership attack (ε = {epsilon})"
    ):
        raise RuntimeError(f"DP membership attack failed for epsilon={epsilon}")

    # Run attribute attack
    if not run_command(
        ["python", "attack_attribute.py", "--model-name", model_name],
        f"Running attribute attack (ε = {epsilon})"
    ):
        raise RuntimeError(f"DP attribute attack failed for epsilon={epsilon}")

    # Collect metrics
    metrics = collect_metrics(model_name, epsilon=epsilon)

    return metrics


def collect_metrics(model_name: str, epsilon: float = None) -> Dict[str, float]:
    """
    Collect all metrics for a model.

    Args:
        model_name: Name of the model
        epsilon: Privacy budget (None for baseline)

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'model': model_name,
        'epsilon': epsilon,
    }

    # Load Re-ID metrics
    reid_path = os.path.join(config.RESULTS_DIR, f"{model_name}_reid_metrics.json")
    with open(reid_path, 'r') as f:
        reid_metrics = json.load(f)
    metrics.update({
        'mAP': reid_metrics['mAP'],
        'rank1': reid_metrics['rank1'],
        'rank5': reid_metrics['rank5'],
        'rank10': reid_metrics['rank10'],
    })

    # Load membership attack metrics
    membership_path = os.path.join(config.RESULTS_DIR, f"{model_name}_membership_attack.json")
    with open(membership_path, 'r') as f:
        membership_metrics = json.load(f)
    metrics['membership_attack_acc'] = membership_metrics['accuracy']

    # Load attribute attack metrics
    attribute_path = os.path.join(config.RESULTS_DIR, f"{model_name}_attribute_attack.json")
    with open(attribute_path, 'r') as f:
        attribute_metrics = json.load(f)
    metrics['attribute_attack_acc'] = attribute_metrics['accuracy']

    return metrics


def plot_tradeoff_curves(results: List[Dict[str, float]]):
    """
    Plot privacy-utility trade-off curves.

    Args:
        results: List of metric dictionaries
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # Sort by epsilon (baseline has epsilon=None, treat as infinity)
    df['epsilon_sort'] = df['epsilon'].fillna(float('inf'))
    df = df.sort_values('epsilon_sort')

    # Prepare data for plotting
    epsilons = []
    labels = []
    for _, row in df.iterrows():
        if row['epsilon'] is None:
            epsilons.append(100)  # Use large value for baseline
            labels.append('Baseline\n(No DP)')
        else:
            epsilons.append(row['epsilon'])
            labels.append(f'ε={row["epsilon"]}')

    epsilons = np.array(epsilons)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Privacy-Utility Trade-off Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Re-ID Performance (mAP and Rank-1)
    ax = axes[0, 0]
    ax.plot(epsilons, df['mAP'] * 100, 'o-', linewidth=2, markersize=8, label='mAP', color='#1f77b4')
    ax.plot(epsilons, df['rank1'] * 100, 's-', linewidth=2, markersize=8, label='Rank-1', color='#ff7f0e')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax.set_ylabel('Re-ID Accuracy (%)', fontsize=11)
    ax.set_title('Re-ID Performance vs Privacy Budget', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if len(epsilons) > 1:
        ax.set_xscale('log')

    # Plot 2: Membership Attack Success
    ax = axes[0, 1]
    ax.plot(epsilons, df['membership_attack_acc'] * 100, 'o-', linewidth=2, markersize=8, color='#d62728')
    ax.axhline(y=50, color='gray', linestyle='--', label='Random Guess (50%)')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax.set_ylabel('Attack Accuracy (%)', fontsize=11)
    ax.set_title('Membership Inference Attack Success', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if len(epsilons) > 1:
        ax.set_xscale('log')

    # Plot 3: Attribute Attack Success
    ax = axes[1, 0]
    ax.plot(epsilons, df['attribute_attack_acc'] * 100, 'o-', linewidth=2, markersize=8, color='#9467bd')
    random_baseline = df['attribute_attack_acc'].iloc[0] * 0.167  # Approximate random guess for 6 cameras
    ax.axhline(y=random_baseline * 100, color='gray', linestyle='--', label=f'Random Guess (~{random_baseline*100:.1f}%)')
    ax.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax.set_ylabel('Attack Accuracy (%)', fontsize=11)
    ax.set_title('Attribute Inference Attack Success', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if len(epsilons) > 1:
        ax.set_xscale('log')

    # Plot 4: Privacy-Utility Scatter
    ax = axes[1, 1]
    # Use membership attack as privacy metric (lower is better)
    privacy_score = 100 - df['membership_attack_acc'] * 100  # Convert to privacy score (higher is better)
    utility_score = df['mAP'] * 100

    scatter = ax.scatter(privacy_score, utility_score, s=150, c=np.log10(epsilons + 1), cmap='viridis', edgecolors='black', linewidths=1.5)
    for i, label in enumerate(labels):
        ax.annotate(label, (privacy_score.iloc[i], utility_score.iloc[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Privacy Score (100 - Membership Attack Acc) (%)', fontsize=11)
    ax.set_ylabel('Utility Score (mAP) (%)', fontsize=11)
    ax.set_title('Privacy vs Utility Trade-off', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(ε + 1)', fontsize=10)

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(config.RESULTS_DIR, 'privacy_utility_tradeoff.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Trade-off plots saved to {plot_path}")


def save_summary_table(results: List[Dict[str, float]]):
    """
    Save summary table as CSV and print to console.

    Args:
        results: List of metric dictionaries
    """
    df = pd.DataFrame(results)

    # Sort by epsilon
    df['epsilon_sort'] = df['epsilon'].fillna(float('inf'))
    df = df.sort_values('epsilon_sort')

    # Format for display
    display_df = df[[
        'model', 'epsilon', 'mAP', 'rank1', 'rank5',
        'membership_attack_acc', 'attribute_attack_acc'
    ]].copy()

    # Convert to percentages
    for col in ['mAP', 'rank1', 'rank5', 'membership_attack_acc', 'attribute_attack_acc']:
        display_df[col] = (display_df[col] * 100).round(2)

    # Rename columns
    display_df.columns = [
        'Model', 'Epsilon', 'mAP (%)', 'Rank-1 (%)', 'Rank-5 (%)',
        'Membership Attack (%)', 'Attribute Attack (%)'
    ]

    # Print table
    print("\n" + "=" * 120)
    print("PRIVACY-UTILITY TRADE-OFF SUMMARY")
    print("=" * 120)
    print(display_df.to_string(index=False))
    print("=" * 120)

    # Save as CSV
    csv_path = os.path.join(config.RESULTS_DIR, 'privacy_utility_summary.csv')
    display_df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary table saved to {csv_path}")


def run_full_analysis(epsilon_values: List[float] = None, skip_baseline: bool = False):
    """
    Run full privacy-utility trade-off analysis.

    Args:
        epsilon_values: List of epsilon values to test
        skip_baseline: Whether to skip baseline training
    """
    if epsilon_values is None:
        epsilon_values = config.EPSILON_VALUES

    print("\n" + "=" * 80)
    print("PRIVACY-UTILITY TRADE-OFF ANALYSIS")
    print("=" * 80)
    print(f"Epsilon values to test: {epsilon_values}")
    print(f"Skip baseline: {skip_baseline}")
    print("=" * 80)

    results = []

    # Train and evaluate baseline
    if not skip_baseline:
        try:
            baseline_metrics = train_and_evaluate_baseline()
            results.append(baseline_metrics)
        except Exception as e:
            print(f"\n✗ Baseline failed: {e}")
            print("Continuing with DP models...\n")

    # Train and evaluate DP models
    for epsilon in epsilon_values:
        try:
            dp_metrics = train_and_evaluate_dp(epsilon)
            results.append(dp_metrics)
        except Exception as e:
            print(f"\n✗ DP model (ε={epsilon}) failed: {e}")
            print(f"Continuing with remaining epsilon values...\n")
            continue

    if len(results) == 0:
        raise RuntimeError("All experiments failed!")

    # Generate plots and summary
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_tradeoff_curves(results)
    save_summary_table(results)

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to: {config.RESULTS_DIR}/")
    print("  - privacy_utility_tradeoff.png/pdf: Trade-off plots")
    print("  - privacy_utility_summary.csv: Summary table")
    print("=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run privacy-utility trade-off analysis"
    )
    parser.add_argument(
        "--epsilon-values",
        type=float,
        nargs='+',
        default=config.EPSILON_VALUES,
        help="List of epsilon values to test"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline training"
    )

    args = parser.parse_args()

    try:
        run_full_analysis(
            epsilon_values=args.epsilon_values,
            skip_baseline=args.skip_baseline
        )
        return 0
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
