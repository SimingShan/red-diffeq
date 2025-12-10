#!/usr/bin/env python3
"""
Quick summary of OpenFWI model comparison results.

Usage:
    python scripts/quick_model_summary.py
"""

import numpy as np
from pathlib import Path
import pandas as pd

def load_model_results():
    """Load results for all 4 models."""
    results_dir = Path('experiment/OpenFWI')

    models = ['model-1', 'model-2', 'model-3', 'model-4']
    results = []

    for model in models:
        exp_dir = results_dir / f'openfwi_{model}_sigma10_clean'

        if not exp_dir.exists():
            print(f"Warning: {exp_dir} not found, skipping...")
            continue

        # Find all result files
        result_files = list(exp_dir.rglob('*_results.npz'))

        if len(result_files) == 0:
            print(f"Warning: No results found in {exp_dir}")
            continue

        print(f"\n{model}: Found {len(result_files)} result files")

        # Aggregate metrics across all samples
        ssims = []
        maes = []
        rmses = []

        for result_file in result_files:
            try:
                data = np.load(result_file)
                ssims.append(float(data['ssim'][-1]))
                maes.append(float(data['mae'][-1]))
                rmses.append(float(data['rmse'][-1]))
            except Exception as e:
                print(f"  Error loading {result_file}: {e}")

        if len(ssims) > 0:
            results.append({
                'model': model,
                'n_samples': len(ssims),
                'avg_ssim': np.mean(ssims),
                'std_ssim': np.std(ssims),
                'avg_mae': np.mean(maes),
                'std_mae': np.std(maes),
                'avg_rmse': np.mean(rmses),
                'std_rmse': np.std(rmses),
            })

    return pd.DataFrame(results)

def main():
    print("="*70)
    print("DIFFUSION MODEL COMPARISON - QUICK SUMMARY")
    print("="*70)

    df = load_model_results()

    if len(df) == 0:
        print("\nNo results found!")
        print("Make sure you've run: experiment/exp_sh/model_comparison/diffusion_model_study.sh")
        return

    print("\n" + "-"*70)
    print("RESULTS SUMMARY")
    print("-"*70)

    # Sort by SSIM (descending)
    df_sorted = df.sort_values('avg_ssim', ascending=False)

    for idx, row in df_sorted.iterrows():
        print(f"\n{row['model'].upper():10}")
        print(f"  Samples:    {int(row['n_samples'])}")
        print(f"  Avg SSIM:   {row['avg_ssim']:.4f} ± {row['std_ssim']:.4f}")
        print(f"  Avg MAE:    {row['avg_mae']:.4f} ± {row['std_mae']:.4f}")
        print(f"  Avg RMSE:   {row['avg_rmse']:.4f} ± {row['std_rmse']:.4f}")

    print("\n" + "-"*70)
    print("RANKING (by SSIM)")
    print("-"*70)

    for rank, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        if rank == 1:
            marker = "🏆"
        elif rank == 2:
            marker = "🥈"
        elif rank == 3:
            marker = "🥉"
        else:
            marker = "  "

        print(f"{marker} {rank}. {row['model']:10} - SSIM: {row['avg_ssim']:.4f}")

    print("\n" + "-"*70)
    best_model = df_sorted.iloc[0]['model']
    best_ssim = df_sorted.iloc[0]['avg_ssim']

    print(f"\n✓ BEST MODEL: {best_model} (SSIM: {best_ssim:.4f})")
    print(f"  Use this model for future experiments!")
    print(f"  Update configs to: pretrained_models/{best_model}.pt")

    print("\n" + "="*70)

    # Save summary
    output_file = 'experiment/model_comparison_summary.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"\nSummary saved to: {output_file}")

    print("\nFor detailed analysis with plots, run:")
    print("  python scripts/analyze_model_comparison.py")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
