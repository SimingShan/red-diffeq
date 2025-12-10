#!/usr/bin/env python
"""
Wrapper script to run uncertainty analysis with specific RED-DiffEq parameters.

Parameters:
- regularization: diffusion (RED-DiffEq)
- reg_lambda: 0.75
- lr: 0.03
- ts: 300
- n_runs: 10 per family (default, can be overridden)
- batch_size: 5 (default, can be overridden)
- dataset_families: All families by default (CF, CV, FF, FV), or specify with --dataset_families

Usage:
    python scripts/run_red_diffeq_uncertainty.py [--batch_size BATCH_SIZE] [--n_runs N_RUNS] [--dataset_families FAMILY1 FAMILY2 ...]
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from red_diffeq import load_config
from scripts import uncertainty_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run uncertainty analysis with RED-DiffEq parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default batch_size=5, n_runs=10, all families
  python scripts/run_red_diffeq_uncertainty.py
  
  # Process only CF family
  python scripts/run_red_diffeq_uncertainty.py --dataset_families CF
  
  # Process CF and CV families
  python scripts/run_red_diffeq_uncertainty.py --dataset_families CF CV
  
  # Use batch_size=2, n_runs=10, only CF
  python scripts/run_red_diffeq_uncertainty.py --batch_size 2 --dataset_families CF
  
  # Use batch_size=5, n_runs=20, all families
  python scripts/run_red_diffeq_uncertainty.py --n_runs 20
        """
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for each run (default: 5)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of inversion runs per family (default: 10)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Sample index to use (default: None to process all samples)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--dataset_families",
        type=str,
        nargs="+",
        default=None,
        help="Dataset families to process (default: None to auto-detect all: CF, CV, FF, FV)",
    )
    
    args = parser.parse_args()
    
    # Default config path
    base_config = Path(args.config)
    
    if not base_config.exists():
        print(f"Error: Base config not found: {base_config}")
        sys.exit(1)
    
    n_batches = (args.n_runs + args.batch_size - 1) // args.batch_size  # Ceiling division
    
    # Determine title based on families
    if args.dataset_families:
        families_title = ", ".join(args.dataset_families)
    else:
        families_title = "All Families"
    
    print(f"\n{'='*70}")
    print(f"UNCERTAINTY ANALYSIS - RED-DiffEq ({families_title})")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Regularization: diffusion")
    print(f"  Lambda: 0.75")
    print(f"  Learning rate: 0.03")
    print(f"  Timesteps: 300")
    print(f"  Number of runs per family: {args.n_runs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of batches per family: {n_batches} ({args.n_runs}/{args.batch_size})")
    if args.sample_idx is not None:
        print(f"  Sample index: {args.sample_idx} (single sample mode)")
    else:
        print(f"  Processing: All samples")
    if args.dataset_families:
        families_str = ", ".join(args.dataset_families)
        print(f"  Dataset families: {families_str} (specified)")
        n_families = len(args.dataset_families)
    else:
        print(f"  Dataset families: Auto-detect (CF, CV, FF, FV)")
        n_families = 4  # Default assumption
    if args.sample_idx is not None:
        print(f"  Total batch runs: {n_families} families × {n_batches} batches = {n_families * n_batches}")
    else:
        print(f"  Total batch runs: {n_families} families × {n_batches} batches × N samples per family")
    print(f"{'='*70}\n")
    
    # Monkey-patch load_config to return modified config
    original_load_config = uncertainty_analysis.load_config
    def patched_load_config(config_path):
        config = original_load_config(config_path)
        # Modify config parameters
        config.optimization.regularization = "diffusion"
        config.optimization.reg_lambda = 0.75
        config.optimization.lr = 0.03
        config.optimization.ts = 300
        return config
    
    # Temporarily replace load_config
    uncertainty_analysis.load_config = patched_load_config
    
    # Call the uncertainty analysis main function
    original_argv = sys.argv
    try:
        sys.argv = [
            'run_red_diffeq_uncertainty.py',
            '--config', str(base_config),
            '--n_runs', str(args.n_runs),
            '--batch_size', str(args.batch_size),
        ]
        # Only add --sample_idx if it's specified (not None)
        if args.sample_idx is not None:
            sys.argv.extend(['--sample_idx', str(args.sample_idx)])
        # Add --dataset_families if specified
        if args.dataset_families:
            sys.argv.extend(['--dataset_families'] + args.dataset_families)
        uncertainty_analysis.main()
    finally:
        sys.argv = original_argv
        # Restore original load_config
        uncertainty_analysis.load_config = original_load_config


if __name__ == "__main__":
    main()

