#!/usr/bin/env python3
"""
Analyze diffusion model comparison results.

Compares performance of model-1, model-2, model-3, model-4 across:
- Different datasets (Marmousi, Overthrust)
- Different initial conditions (sigma values)
- Different noise levels

Usage:
    python scripts/analyze_model_comparison.py \
        --results_dir experiment/ \
        --output_dir experiment/analysis/model_comparison/
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
    Parse experiment name to extract dataset, model, sigma, and noise.

    Examples:
        - openfwi_model-1_sigma10_clean -> dataset='openfwi', model='model-1', sigma=10, noise=0.0
        - marmousi_model-1_sigma20_clean -> dataset='marmousi', model='model-1', sigma=20, noise=0.0
        - overthrust_model-3_sigma25_gaussian0.3 -> dataset='overthrust', model='model-3', sigma=25, noise=0.3
    """
    parts = exp_name.split('_')

    dataset = parts[0]  # openfwi, marmousi, or overthrust
    model = parts[1]  # model-1, model-2, model-3, model-4

    sigma = None
    noise = 0.0
    noise_type = 'clean'

    for part in parts[2:]:
        if part.startswith('sigma'):
            sigma = float(part.replace('sigma', ''))
        elif part.startswith('gaussian'):
            noise_type = 'gaussian'
            noise = float(part.replace('gaussian', ''))
        elif part == 'clean':
            noise_type = 'clean'

    return dataset, model, sigma, noise_type, noise


def load_results(results_dir):
    """Load all model comparison results."""
    results_dir = Path(results_dir)
    experiments = []

    # Search for model comparison results
    for dataset_dir in results_dir.glob('*/'):
        if dataset_dir.name not in ['Marmousi', 'Overthrust', 'OpenFWI']:
            continue

        dataset = dataset_dir.name.lower()

        # Find experiment directories with model-X pattern
        for exp_dir in dataset_dir.glob('*_model-*'):
            exp_name = exp_dir.name

            # Parse experiment name
            try:
                ds, model, sigma, noise_type, noise = parse_exp_name(exp_name)
            except:
                continue

            # Find result files
            result_files = list(exp_dir.rglob('*_results.npz'))

            for result_file in result_files:
                try:
                    data = np.load(result_file)

                    # Extract final metrics
                    final_ssim = float(data['ssim'][-1])
                    final_mae = float(data['mae'][-1])
                    final_rmse = float(data['rmse'][-1])

                    experiments.append({
                        'dataset': ds,
                        'model': model,
                        'sigma': sigma,
                        'noise_type': noise_type,
                        'noise': noise,
                        'ssim': final_ssim,
                        'mae': final_mae,
                        'rmse': final_rmse,
                        'exp_name': exp_name,
                        'file': result_file,
                    })

                except Exception as e:
                    print(f"Warning: Failed to load {result_file}: {e}")

    return pd.DataFrame(experiments)


def plot_model_comparison_by_dataset(df, output_dir):
    """Plot: Model comparison for each dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    datasets = ['marmousi', 'overthrust']
    metrics = ['ssim', 'mae', 'rmse']
    metric_labels = ['SSIM (↑ better)', 'MAE (↓ better)', 'RMSE (↓ better)']

    for row, dataset in enumerate(datasets):
        df_dataset = df[df['dataset'] == dataset]

        for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]

            # Group by model and calculate mean across all conditions
            model_stats = df_dataset.groupby('model')[metric].agg(['mean', 'std'])
            model_stats = model_stats.sort_index()

            x_pos = np.arange(len(model_stats))
            ax.bar(x_pos, model_stats['mean'], yerr=model_stats['std'],
                   capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_stats.index, rotation=0)
            ax.set_xlabel('Model', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.set_title(f'{dataset.capitalize()} - {label}', fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_by_dataset.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison_by_dataset.png'}")
    plt.close()


def plot_model_vs_sigma(df, output_dir):
    """Plot: Model performance vs initial smoothing."""
    # Filter for clean data only
    df_clean = df[df['noise_type'] == 'clean']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    datasets = ['marmousi', 'overthrust']
    metrics = ['ssim', 'mae', 'rmse']
    metric_labels = ['SSIM (↑ better)', 'MAE (↓ better)', 'RMSE (↓ better)']

    for row, dataset in enumerate(datasets):
        df_dataset = df_clean[df_clean['dataset'] == dataset]

        for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]

            for model in sorted(df_dataset['model'].unique()):
                model_data = df_dataset[df_dataset['model'] == model].sort_values('sigma')
                ax.plot(model_data['sigma'], model_data[metric],
                       marker='o', linewidth=2, markersize=8, label=model)

            ax.set_xlabel('Initial Sigma', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.set_title(f'{dataset.capitalize()} - {label} vs Sigma', fontsize=16)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_vs_sigma.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_vs_sigma.png'}")
    plt.close()


def plot_model_vs_noise(df, output_dir):
    """Plot: Model robustness to noise."""
    # Filter for sigma=20 with noise
    df_noise = df[(df['sigma'] == 20) & (df['noise_type'] == 'gaussian')]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    datasets = ['marmousi', 'overthrust']
    metrics = ['ssim', 'mae', 'rmse']
    metric_labels = ['SSIM (↑ better)', 'MAE (↓ better)', 'RMSE (↓ better)']

    for row, dataset in enumerate(datasets):
        df_dataset = df_noise[df_noise['dataset'] == dataset]

        for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]

            for model in sorted(df_dataset['model'].unique()):
                model_data = df_dataset[df_dataset['model'] == model].sort_values('noise')
                if len(model_data) > 0:
                    ax.plot(model_data['noise'], model_data[metric],
                           marker='o', linewidth=2, markersize=8, label=model)

            ax.set_xlabel('Gaussian Noise Level', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.set_title(f'{dataset.capitalize()} - {label} vs Noise', fontsize=16)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_vs_noise.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_vs_noise.png'}")
    plt.close()


def plot_model_heatmap(df, output_dir):
    """Plot: Heatmap of model performance across conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    datasets = ['marmousi', 'overthrust']
    models = sorted(df['model'].unique())

    for row, dataset in enumerate(datasets):
        df_dataset = df[df['dataset'] == dataset]

        # Prepare data for heatmap
        conditions = []
        for sigma in sorted(df_dataset['sigma'].unique()):
            conditions.append(f'σ={int(sigma)}')

        for noise in sorted(df_dataset[df_dataset['noise'] > 0]['noise'].unique()):
            conditions.append(f'N={noise:.1f}')

        ssim_matrix = np.zeros((len(models), len(conditions)))
        mae_matrix = np.zeros((len(models), len(conditions)))

        for i, model in enumerate(models):
            model_data = df_dataset[df_dataset['model'] == model]

            col_idx = 0
            # Sigma conditions (clean)
            for sigma in sorted(df_dataset['sigma'].unique()):
                data = model_data[(model_data['sigma'] == sigma) & (model_data['noise'] == 0.0)]
                if len(data) > 0:
                    ssim_matrix[i, col_idx] = data['ssim'].values[0]
                    mae_matrix[i, col_idx] = data['mae'].values[0]
                col_idx += 1

            # Noise conditions (sigma=20)
            for noise in sorted(df_dataset[df_dataset['noise'] > 0]['noise'].unique()):
                data = model_data[(model_data['sigma'] == 20) & (model_data['noise'] == noise)]
                if len(data) > 0:
                    ssim_matrix[i, col_idx] = data['ssim'].values[0]
                    mae_matrix[i, col_idx] = data['mae'].values[0]
                col_idx += 1

        # SSIM heatmap
        ax = axes[row, 0]
        im = ax.imshow(ssim_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(models, fontsize=11)
        ax.set_title(f'{dataset.capitalize()} - SSIM', fontsize=14, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        plt.colorbar(im, ax=ax, label='SSIM')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(conditions)):
                if ssim_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{ssim_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)

        # MAE heatmap
        ax = axes[row, 1]
        im = ax.imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(models, fontsize=11)
        ax.set_title(f'{dataset.capitalize()} - MAE', fontsize=14, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        plt.colorbar(im, ax=ax, label='MAE')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(conditions)):
                if mae_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{mae_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_heatmap.png'}")
    plt.close()


def generate_summary_table(df, output_dir):
    """Generate summary statistics table."""
    summary = []

    for dataset in sorted(df['dataset'].unique()):
        for model in sorted(df['model'].unique()):
            model_data = df[(df['dataset'] == dataset) & (df['model'] == model)]

            if len(model_data) > 0:
                summary.append({
                    'Dataset': dataset.capitalize(),
                    'Model': model,
                    'Avg SSIM': f"{model_data['ssim'].mean():.3f} ± {model_data['ssim'].std():.3f}",
                    'Avg MAE': f"{model_data['mae'].mean():.4f} ± {model_data['mae'].std():.4f}",
                    'Avg RMSE': f"{model_data['rmse'].mean():.4f} ± {model_data['rmse'].std():.4f}",
                    'Best SSIM': f"{model_data['ssim'].max():.3f}",
                    'Worst SSIM': f"{model_data['ssim'].min():.3f}",
                    'N_runs': len(model_data),
                })

    summary_df = pd.DataFrame(summary)

    # Save as CSV
    summary_df.to_csv(output_dir / 'model_comparison_summary.csv', index=False)
    print(f"Saved: {output_dir / 'model_comparison_summary.csv'}")

    # Print to console
    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")

    # Find best model overall
    print("BEST MODEL ANALYSIS:")
    print("-" * 100)

    for dataset in sorted(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]
        best_model = dataset_data.groupby('model')['ssim'].mean().idxmax()
        best_ssim = dataset_data.groupby('model')['ssim'].mean().max()

        print(f"{dataset.capitalize():12} - Best: {best_model:10} (Avg SSIM: {best_ssim:.3f})")

    print("-" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze diffusion model comparison results')
    parser.add_argument('--results_dir', type=str, default='experiment/',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='experiment/analysis/model_comparison/',
                       help='Directory to save analysis plots')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model comparison results...")
    df = load_results(args.results_dir)

    if len(df) == 0:
        print("Error: No results found!")
        print("Make sure you've run: experiment/exp_sh/model_comparison/diffusion_model_study.sh")
        return

    print(f"Loaded {len(df)} experiment results")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Conditions: {len(df['sigma'].unique())} sigma values, {len(df[df['noise'] > 0]['noise'].unique())} noise levels")

    # Generate plots
    print("\nGenerating plots...")
    plot_model_comparison_by_dataset(df, output_dir)
    plot_model_vs_sigma(df, output_dir)
    plot_model_vs_noise(df, output_dir)
    plot_model_heatmap(df, output_dir)

    # Generate summary table
    generate_summary_table(df, output_dir)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
