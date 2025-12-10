#!/usr/bin/env python3
"""
Analyze Marmousi experiment results and generate comparison plots.

Usage:
    python scripts/analyze_marmousi_results.py --results_dir experiment/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def parse_exp_name(exp_name):
    """
    Parse experiment name to extract method, sigma, noise type, and noise level.

    Examples:
        - marmousi_red-diffeq_sigma20 -> method='red-diffeq', sigma=20, noise_type=None, noise_level=0.0
        - marmousi_tv_sigma25 -> method='tv', sigma=25, noise_type=None, noise_level=0.0
        - marmousi_baseline_sigma20_gaussian0.3 -> method='baseline', sigma=20, noise_type='gaussian', noise_level=0.3
    """
    parts = exp_name.replace('marmousi_', '').split('_')

    method = parts[0]
    sigma = None
    noise_type = None
    noise_level = 0.0

    for part in parts[1:]:
        if part.startswith('sigma'):
            sigma = float(part.replace('sigma', ''))
        elif part.startswith('gaussian'):
            noise_type = 'gaussian'
            noise_level = float(part.replace('gaussian', ''))
        elif part.startswith('laplace'):
            noise_type = 'laplace'
            noise_level = float(part.replace('laplace', ''))

    return method, sigma, noise_type, noise_level


def load_results(results_dir):
    """Load all experiment results."""
    results_dir = Path(results_dir)
    experiments = defaultdict(list)

    # Find all result directories
    for exp_dir in results_dir.glob('marmousi_*'):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        method, sigma, noise_type, noise_level = parse_exp_name(exp_name)

        # Find result files
        result_files = list(exp_dir.rglob('*_results.npz'))

        for result_file in result_files:
            try:
                data = np.load(result_file)

                # Extract final metrics
                final_ssim = float(data['ssim'][-1])
                final_mae = float(data['mae'][-1])
                final_rmse = float(data['rmse'][-1])

                experiments['results'].append({
                    'method': method,
                    'sigma': sigma,
                    'noise_type': noise_type if noise_type else 'none',
                    'noise_level': noise_level,
                    'ssim': final_ssim,
                    'mae': final_mae,
                    'rmse': final_rmse,
                    'exp_name': exp_name,
                    'file': result_file,
                })

            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}")

    return pd.DataFrame(experiments['results'])


def plot_sigma_comparison(df, output_dir):
    """Plot: Final SSIM vs. Initial Sigma for each method."""
    # Filter for no-noise experiments
    df_sigma = df[df['noise_type'] == 'none'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['ssim', 'mae', 'rmse']
    titles = ['SSIM (↑ better)', 'MAE (↓ better)', 'RMSE (↓ better)']

    for ax, metric, title in zip(axes, metrics, titles):
        for method in df_sigma['method'].unique():
            method_data = df_sigma[df_sigma['method'] == method].sort_values('sigma')
            ax.plot(method_data['sigma'], method_data[metric],
                   marker='o', linewidth=2, markersize=8, label=method)

        ax.set_xlabel('Initial Model Sigma', fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.set_title(f'{title} vs. Initial Smoothing', fontsize=16)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sigma_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sigma_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'sigma_comparison.png'}")
    plt.close()


def plot_noise_robustness(df, output_dir):
    """Plot: Noise robustness comparison."""
    # Filter for sigma=20 with noise
    df_noise = df[(df['sigma'] == 20) & (df['noise_type'] != 'none')].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['ssim', 'mae', 'rmse']
    titles = ['SSIM (↑ better)', 'MAE (↓ better)', 'RMSE (↓ better)']
    noise_types = ['gaussian', 'laplace']

    for row, noise_type in enumerate(noise_types):
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[row, col]

            df_filtered = df_noise[df_noise['noise_type'] == noise_type]

            for method in df_filtered['method'].unique():
                method_data = df_filtered[df_filtered['method'] == method].sort_values('noise_level')
                ax.plot(method_data['noise_level'], method_data[metric],
                       marker='o', linewidth=2, markersize=8, label=method)

            ax.set_xlabel(f'{noise_type.capitalize()} Noise Level', fontsize=14)
            ax.set_ylabel(title, fontsize=14)
            ax.set_title(f'{title} - {noise_type.capitalize()} Noise', fontsize=16)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'noise_robustness.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'noise_robustness.png'}")
    plt.close()


def plot_method_comparison_heatmap(df, output_dir):
    """Plot: Heatmap of SSIM for all conditions."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    methods = sorted(df['method'].unique())

    for idx, method in enumerate(methods):
        ax = axes[idx]
        method_data = df[df['method'] == method].copy()

        # Create pivot table for heatmap
        # Rows: sigma values, Columns: noise conditions
        conditions = []
        ssim_values = []
        sigma_values = sorted(method_data['sigma'].unique())

        for sigma in sigma_values:
            row = []
            # No noise
            no_noise = method_data[(method_data['sigma'] == sigma) &
                                  (method_data['noise_type'] == 'none')]
            row.append(no_noise['ssim'].values[0] if len(no_noise) > 0 else np.nan)

            # Gaussian noise
            for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]:
                gaussian = method_data[(method_data['sigma'] == sigma) &
                                      (method_data['noise_type'] == 'gaussian') &
                                      (np.abs(method_data['noise_level'] - noise_level) < 0.01)]
                row.append(gaussian['ssim'].values[0] if len(gaussian) > 0 else np.nan)

            # Laplace noise
            for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]:
                laplace = method_data[(method_data['sigma'] == sigma) &
                                     (method_data['noise_type'] == 'laplace') &
                                     (np.abs(method_data['noise_level'] - noise_level) < 0.01)]
                row.append(laplace['ssim'].values[0] if len(laplace) > 0 else np.nan)

            ssim_values.append(row)

        # Create heatmap
        ssim_array = np.array(ssim_values)

        col_labels = ['Clean'] + [f'G{x:.1f}' for x in [0.1, 0.2, 0.3, 0.4, 0.5]] + \
                                [f'L{x:.1f}' for x in [0.1, 0.2, 0.3, 0.4, 0.5]]

        im = ax.imshow(ssim_array, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(sigma_values)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels([f'{int(s)}' for s in sigma_values], fontsize=10)

        ax.set_xlabel('Noise Condition', fontsize=12)
        ax.set_ylabel('Initial Sigma', fontsize=12)
        ax.set_title(f'{method.upper()}', fontsize=14)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='SSIM')

    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'method_comparison_heatmap.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'method_comparison_heatmap.png'}")
    plt.close()


def generate_summary_table(df, output_dir):
    """Generate summary statistics table."""
    summary = []

    for method in sorted(df['method'].unique()):
        method_data = df[df['method'] == method]

        # Overall statistics
        summary.append({
            'Method': method.upper(),
            'Avg SSIM': f"{method_data['ssim'].mean():.3f} ± {method_data['ssim'].std():.3f}",
            'Avg MAE': f"{method_data['mae'].mean():.4f} ± {method_data['mae'].std():.4f}",
            'Avg RMSE': f"{method_data['rmse'].mean():.4f} ± {method_data['rmse'].std():.4f}",
            'Best SSIM': f"{method_data['ssim'].max():.3f}",
            'Worst SSIM': f"{method_data['ssim'].min():.3f}",
        })

    summary_df = pd.DataFrame(summary)

    # Save as CSV
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"Saved: {output_dir / 'summary_statistics.csv'}")

    # Print to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze Marmousi experiment results')
    parser.add_argument('--results_dir', type=str, default='experiment/',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='experiment/analysis/',
                       help='Directory to save analysis plots')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_results(args.results_dir)

    if len(df) == 0:
        print("Error: No results found!")
        return

    print(f"Loaded {len(df)} experiment results")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Sigma values: {sorted(df['sigma'].unique())}")
    print(f"Noise types: {sorted(df['noise_type'].unique())}")

    # Generate plots
    print("\nGenerating plots...")
    plot_sigma_comparison(df, output_dir)
    plot_noise_robustness(df, output_dir)
    plot_method_comparison_heatmap(df, output_dir)

    # Generate summary table
    generate_summary_table(df, output_dir)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
