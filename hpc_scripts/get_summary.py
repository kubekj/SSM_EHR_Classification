import json
import glob
import os
import numpy as np
import argparse
from typing import Dict, List


def load_split_results(base_dir: str, split_folder_pattern: str = "split_*") -> List[Dict]:
    """Load test results from all split folders."""
    results = []
    pattern = os.path.join(base_dir, split_folder_pattern)
    for folder in glob.glob(pattern):
        result_path = os.path.join(folder, "test_result.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                results.append(json.load(f))
    return results


def calculate_summary_statistics(results: List[Dict]) -> Dict:
    """Calculate mean and standard deviation for key metrics."""
    losses = []
    accuracies = []
    auprcs = []
    aurocs = []

    # Collect metrics from each split
    for result in results:
        losses.append(result['test_loss'])
        accuracies.append(result['accuracy'])
        auprcs.append(result['AUPRC'])
        aurocs.append(result['AUROC'])

    # Calculate summary statistics
    summary = {
        'mean_loss': np.mean(losses),
        'mean_accuracy': np.mean(accuracies),
        'mean_auprc': np.mean(auprcs),
        'mean_auroc': np.mean(aurocs),
        'std_loss': np.std(losses),
        'std_accuracy': np.std(accuracies),
        'std_auprc': np.std(auprcs),
        'std_auroc': np.std(aurocs)
    }

    # Round all values to 4 decimal places
    summary = {k: round(v, 4) for k, v in summary.items()}

    return summary


def save_summary(summary: Dict, base_dir: str, output_filename: str = "summary.json") -> None:
    """Save summary statistics to a JSON file."""
    output_path = os.path.join(base_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)


def get_summary():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Generate summary statistics from split folders')
    parser.add_argument('--dir', type=str, default='.',
                        help='Base directory containing split_* folders (default: current directory)')
    args = parser.parse_args()

    # Convert relative path to absolute path
    base_dir = os.path.abspath(args.dir)

    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist!")
        return

    results = load_split_results(base_dir)

    if not results:
        print(f"No test results found in split folders under '{base_dir}'!")
        return

    summary = calculate_summary_statistics(results)

    save_summary(summary, base_dir)
    print(f"\nSummary statistics saved to {os.path.join(base_dir, 'summary.json')}")

    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    get_summary()
