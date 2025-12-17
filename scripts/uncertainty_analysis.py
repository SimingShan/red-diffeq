#!/usr/bin/env python
"""
Script to run inversion multiple times and analyze uncertainty vs error.

Runs the same inversion 10 times (with batch_size=2) for the first sample of CF,
then evaluates uncertainty (std across runs) vs error (difference from ground truth).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.dpi'] = 300

from red_diffeq import (
    REDDiffEqConfig,
    load_config,
    GaussianDiffusion,
    Unet,
    FWIForward,
    InversionEngine,
    DataTransformer,
    SSIM,
    prepare_initial_model,
    s_normalize_none,
    v_denormalize,
)
from red_diffeq.utils.data_trans import v_normalize
from red_diffeq.config import RegularizationType


def setup_device() -> torch.device:
    """Setup computation device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_diffusion_model(config: REDDiffEqConfig, device: torch.device) -> GaussianDiffusion:
    """Initialize and load pretrained diffusion model."""
    model = Unet(
        dim=config.model.dim,
        dim_mults=config.model.dim_mults,
        flash_attn=config.model.flash_attn,
        channels=config.model.channels,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=config.diffusion.image_size,
        timesteps=config.diffusion.timesteps,
        sampling_timesteps=config.diffusion.sampling_timesteps,
        objective=config.diffusion.objective,
    ).to(device)

    model_path = Path(config.diffusion.model_path)
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        diffusion.load_state_dict(checkpoint["model"])
        print(f"Loaded pretrained model from: {model_path}")
    else:
        print(f"WARNING: Pretrained model not found at {model_path}")
        print("Continuing with randomly initialized model...")

    diffusion.eval()
    return diffusion


def initialize_forward_operator(config: REDDiffEqConfig, device: torch.device) -> FWIForward:
    """Initialize PDE forward modeling operator."""
    ctx = config.pde.to_dict()

    fwi_forward = FWIForward(
        ctx,
        device,
        normalize=True,
        v_denorm_func=v_denormalize,
        s_norm_func=s_normalize_none,
    )

    return fwi_forward


def get_data_info(
    config: REDDiffEqConfig,
    dataset_family: str
) -> tuple:
    """Get information about dataset without loading all data.
    
    Args:
        config: Configuration instance.
        dataset_family: Dataset family name (e.g., 'CF', 'CV', 'FF', 'FV').
        
    Returns:
        Tuple of (vel_path, seis_path, n_samples, use_mmap).
    """
    vel_path = Path(config.data.velocity_data_dir) / f"{dataset_family}.npy"
    seis_path = Path(config.data.seismic_data_dir) / f"{dataset_family}.npy"
    
    if not vel_path.exists():
        raise FileNotFoundError(f"Velocity file not found: {vel_path}")
    if not seis_path.exists():
        raise FileNotFoundError(f"Seismic file not found: {seis_path}")
    
    # Load with memory mapping to check shape without loading all data
    use_mmap = getattr(config.data, 'use_mmap', True)
    mmap_mode = "r" if use_mmap else None
    
    vel_info = np.load(vel_path, mmap_mode=mmap_mode)
    seis_info = np.load(seis_path, mmap_mode=mmap_mode)
    
    # Determine number of samples
    if vel_info.ndim == 4:
        n_samples = vel_info.shape[0]
    else:
        n_samples = 1
    
    print(f"{dataset_family} data info:")
    print(f"  Velocity shape: {vel_info.shape}")
    print(f"  Seismic shape: {seis_info.shape}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Using memory mapping: {use_mmap}")
    
    return vel_path, seis_path, n_samples, use_mmap


def load_single_sample_from_file(
    vel_path: Path,
    seis_path: Path,
    sample_idx: int,
    use_mmap: bool = True
) -> tuple:
    """Load a single sample from data files using memory mapping.
    
    Args:
        vel_path: Path to velocity data file.
        seis_path: Path to seismic data file.
        sample_idx: Index of sample to load.
        use_mmap: Whether to use memory mapping.
        
    Returns:
        Tuple of (velocity_sample, seismic_sample) with batch dimension.
    """
    mmap_mode = "r" if use_mmap else None
    
    # Load with memory mapping (doesn't load entire file into memory)
    vel_data = np.load(vel_path, mmap_mode=mmap_mode)
    seis_data = np.load(seis_path, mmap_mode=mmap_mode)
    
    # Extract single sample
    if vel_data.ndim == 4:
        vel_sample = vel_data[sample_idx:sample_idx+1].copy()  # Copy to avoid keeping mmap reference
        seis_sample = seis_data[sample_idx:sample_idx+1].copy()
    else:
        vel_sample = vel_data[np.newaxis, ...].copy()
        seis_sample = seis_data[np.newaxis, ...].copy()
    
    return vel_sample, seis_sample


def load_single_sample(
    config: REDDiffEqConfig,
    dataset_family: str,
    sample_idx: int = 0
) -> tuple:
    """Load a single velocity and seismic sample.
    
    Args:
        config: Configuration instance.
        dataset_family: Dataset family name (e.g., 'CF', 'CV', 'FF', 'FV').
        sample_idx: Index of sample to load (default: 0 for first sample).
        
    Returns:
        Tuple of (velocity, seismic) tensors.
    """
    vel_path = Path(config.data.velocity_data_dir) / f"{dataset_family}.npy"
    seis_path = Path(config.data.seismic_data_dir) / f"{dataset_family}.npy"
    
    if not vel_path.exists():
        raise FileNotFoundError(f"Velocity file not found: {vel_path}")
    if not seis_path.exists():
        raise FileNotFoundError(f"Seismic file not found: {seis_path}")
    
    vel_data = np.load(vel_path)
    seis_data = np.load(seis_path)
    
    # Extract single sample
    if vel_data.ndim == 4:
        vel_sample = vel_data[sample_idx:sample_idx+1]  # Keep batch dimension
        seis_sample = seis_data[sample_idx:sample_idx+1]
    else:
        vel_sample = vel_data[np.newaxis, ...]
        seis_sample = seis_data[np.newaxis, ...]
    
    print(f"Loaded {dataset_family} sample {sample_idx}:")
    print(f"  Velocity shape: {vel_sample.shape}")
    print(f"  Seismic shape: {seis_sample.shape}")
    
    return vel_sample, seis_sample


def run_single_inversion(
    vel_sample: np.ndarray,
    seis_sample: np.ndarray,
    config: REDDiffEqConfig,
    inversion_engine: InversionEngine,
    fwi_forward: FWIForward,
    device: torch.device,
    run_idx: int,
    batch_size: int = 2
) -> Dict:
    """Run a single inversion.
    
    Args:
        vel_sample: Ground truth velocity (single sample, will be repeated).
        seis_sample: Seismic data (single sample, will be repeated).
        config: Configuration instance.
        inversion_engine: Inversion engine.
        fwi_forward: Forward modeling operator.
        device: Computation device.
        run_idx: Index of this run.
        batch_size: Batch size for processing.
        
    Returns:
        Dictionary with results.
    """
    # Repeat sample to batch_size
    vel_batch = np.repeat(vel_sample, batch_size, axis=0)
    seis_batch = np.repeat(seis_sample, batch_size, axis=0)
    
    # Convert to tensor (raw velocity, NOT normalized - metrics calculation will normalize it)
    vel_batch_tensor = torch.from_numpy(vel_batch).float()
    
    # Prepare initial model (prepare_initial_model expects raw velocity and normalizes internally)
    initial_model = prepare_initial_model(
        vel_batch_tensor,
        initial_type=config.optimization.initial_type.value,
        sigma=config.optimization.sigma,
    )
    
    # Pad initial model for forward modeling (but NOT ground truth - metrics expects unpadded GT)
    pad_size = 1
    initial_model_padded = F.pad(initial_model, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    # Note: vel_batch_tensor stays unpadded - metrics calculation removes padding from result internally
    
    # Convert seismic to tensor
    seis_batch_tensor = torch.from_numpy(seis_batch).float()
    
    # Run inversion
    print(f"\nRun {run_idx + 1}: Starting inversion...")
    mu_batch, results_per_model = inversion_engine.optimize(
        initial_model_padded,
        vel_batch_tensor,  # Ground truth is NOT padded (matches run_inversion.py)
        seis_batch_tensor,
        fwi_forward,
        ts=config.optimization.ts,
        lr=config.optimization.lr,
        reg_lambda=config.optimization.reg_lambda,
        loss_type=config.optimization.loss_type.value,
        noise_std=config.optimization.noise_std,
        noise_type=config.optimization.noise_type,
        missing_number=config.optimization.missing_number,
        regularization=config.optimization.regularization.value
        if config.optimization.regularization
        and config.optimization.regularization != RegularizationType.NONE
        else None,
        scheduler_type=config.optimization.lr_scheduler.value,
        optimizer_type=config.optimization.optimizer.value,
        lbfgs_max_iter=config.optimization.lbfgs_max_iter,
        lbfgs_max_eval=config.optimization.lbfgs_max_eval,
        lbfgs_history_size=config.optimization.lbfgs_history_size,
        lbfgs_line_search=config.optimization.lbfgs_line_search,
    )
    
    # Remove padding from result
    mu_result = mu_batch[:, :, pad_size:-pad_size, pad_size:-pad_size]
    
    # Convert to numpy (unpadded) - collect ALL results from batch
    batch_results = []
    for i in range(batch_size):
        mu_result_np = mu_result[i, 0].detach().cpu().numpy()  # i-th sample, first channel, unpadded (normalized)
        vel_gt_np = v_normalize(vel_batch[i:i+1, 0:1, :, :])[0, 0]  # Ground truth (normalized, unpadded)
        
        final_metrics = results_per_model[i]
        batch_results.append({
            'result': mu_result_np,
            'ground_truth': vel_gt_np,
            'ssim': final_metrics['ssim'][-1],
            'mae': final_metrics['mae'][-1],
            'rmse': final_metrics['rmse'][-1],
            'total_time': final_metrics['total_time'],
        })
    
    # Return list of results (one per sample in batch)
    return batch_results


def calculate_uncertainty_and_error(all_results: List[Dict]) -> Dict:
    """Calculate uncertainty (std) and error (difference from GT) for all runs.
    
    Args:
        all_results: List of result dictionaries from all runs.
        
    Returns:
        Dictionary with uncertainty and error metrics.
    """
    # Stack all results
    results_array = np.stack([r['result'] for r in all_results])  # Shape: (n_runs, H, W)
    gt = all_results[0]['ground_truth']  # Ground truth (same for all runs)
    
    # Calculate uncertainty: std across runs at each pixel
    uncertainty = np.std(results_array, axis=0)  # Shape: (H, W)
    
    # Calculate error: mean absolute difference from GT at each pixel
    mean_result = np.mean(results_array, axis=0)
    error = np.abs(mean_result - gt)  # Shape: (H, W)
    
    # Calculate per-run metrics
    ssim_values = np.array([r['ssim'] for r in all_results])
    mae_values = np.array([r['mae'] for r in all_results])
    rmse_values = np.array([r['rmse'] for r in all_results])
    
    return {
        'uncertainty': uncertainty,
        'error': error,
        'mean_result': mean_result,
        'ground_truth': gt,
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values),
        'all_results': results_array,
    }


def plot_uncertainty_vs_error(analysis: Dict, output_dir: Path):
    """Plot uncertainty vs error analysis.
    
    Args:
        analysis: Dictionary with uncertainty and error metrics.
        output_dir: Output directory for plots.
    """
    uncertainty = analysis['uncertainty']
    error = analysis['error']
    
    # Flatten for scatter plot
    uncertainty_flat = uncertainty.flatten()
    error_flat = error.flatten()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 4))
    
    # 1. Scatter plot: uncertainty vs error
    ax1 = plt.subplot(1, 4, 1)
    scatter = ax1.scatter(uncertainty_flat, error_flat, alpha=0.3, s=1, c='blue')
    ax1.set_xlabel('Uncertainty (std)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error (|mean - GT|)', fontsize=12, fontweight='bold')
    ax1.set_title('Uncertainty vs Error', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(uncertainty_flat, error_flat)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Uncertainty map
    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(uncertainty, cmap='hot', aspect='equal')
    ax2.set_title('Uncertainty Map', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Horizontal Distance', fontsize=11)
    ax2.set_ylabel('Depth', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Uncertainty (std)')
    
    # 3. Error map
    ax3 = plt.subplot(1, 4, 3)
    im3 = ax3.imshow(error, cmap='hot', aspect='equal')
    ax3.set_title('Error Map', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Horizontal Distance', fontsize=11)
    ax3.set_ylabel('Depth', fontsize=11)
    plt.colorbar(im3, ax=ax3, label='Error (|mean - GT|)')
    
    # 4. Histogram of uncertainty vs error
    ax4 = plt.subplot(1, 4, 4)
    ax4.hist2d(uncertainty_flat, error_flat, bins=50, cmap='Blues')
    ax4.set_xlabel('Uncertainty (std)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Error (|mean - GT|)', fontsize=12, fontweight='bold')
    ax4.set_title('2D Histogram', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_vs_error.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'uncertainty_vs_error.svg', dpi=300, bbox_inches='tight', format='svg')
    print(f"Saved: {output_dir / 'uncertainty_vs_error.png'}")
    plt.close()
    
    # Plot mean result vs ground truth
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    im0 = axes[0].imshow(analysis['ground_truth'], cmap='jet', aspect='equal')
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Horizontal Distance', fontsize=11)
    axes[0].set_ylabel('Depth', fontsize=11)
    plt.colorbar(im0, ax=axes[0], label='Velocity (normalized)')
    
    # Mean result
    im1 = axes[1].imshow(analysis['mean_result'], cmap='jet', aspect='equal')
    axes[1].set_title(f'Mean Result (10 runs)\nSSIM: {analysis["ssim_mean"]:.4f} ± {analysis["ssim_std"]:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Horizontal Distance', fontsize=11)
    axes[1].set_ylabel('Depth', fontsize=11)
    plt.colorbar(im1, ax=axes[1], label='Velocity (normalized)')
    
    # Difference
    diff = analysis['mean_result'] - analysis['ground_truth']
    vmax = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap='RdBu', aspect='equal', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Difference (Mean - GT)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Horizontal Distance', fontsize=11)
    axes[2].set_ylabel('Depth', fontsize=11)
    plt.colorbar(im2, ax=axes[2], label='Velocity Difference')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_result_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'mean_result_comparison.svg', dpi=300, bbox_inches='tight', format='svg')
    print(f"Saved: {output_dir / 'mean_result_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run inversion multiple times and analyze uncertainty vs error"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of inversion runs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for each run (default: 2)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Sample index to use (default: None to process all samples)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per family (default: None for all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: uncertainty_analysis/{timestamp})",
    )
    parser.add_argument(
        "--dataset_families",
        type=str,
        nargs="+",
        default=None,
        help="Dataset families to process (default: auto-detect CF, CV, FF, FV)",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    config = load_config(Path(args.config))
    
    # Determine dataset families
    if args.dataset_families:
        dataset_families = args.dataset_families
    else:
        # Auto-detect available families
        vel_dir = Path(config.data.velocity_data_dir)
        dataset_families = []
        for npy_file in vel_dir.glob("*.npy"):
            family_name = npy_file.stem
            if family_name not in ["README"]:  # Skip non-data files
                dataset_families.append(family_name)
        dataset_families = sorted(dataset_families)
    
    print(f"Found dataset families: {dataset_families}")
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("uncertainty_analysis") / timestamp
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("UNCERTAINTY ANALYSIS")
    print(f"{'='*70}")
    print(f"Number of ensemble runs per sample: {args.n_runs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches per sample: {args.n_runs // args.batch_size}")
    if args.sample_idx is not None:
        print(f"Sample index: {args.sample_idx} (single sample mode)")
    else:
        print(f"Processing: All samples")
        if args.max_samples is not None:
            print(f"Maximum samples per family: {args.max_samples}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load models and operators (shared across all families)
    print("Loading models and operators...")
    diffusion_model = load_diffusion_model(config, device)
    fwi_forward = initialize_forward_operator(config, device)
    
    # Initialize inversion engine
    data_transformer = DataTransformer()
    ssim_loss = SSIM(window_size=11, size_average=True)
    inversion_engine = InversionEngine(
        diffusion_model,
        data_transformer,
        ssim_loss,
        regularization=config.optimization.regularization.value
        if config.optimization.regularization
        and config.optimization.regularization != RegularizationType.NONE
        else None,
        use_time_weight=getattr(config.optimization, 'use_time_weight', False),
        sigma_x0=getattr(config.optimization, 'sigma_x0', 0.0001),
    )
    
    # Process each dataset family
    for family_idx, dataset_family in enumerate(dataset_families):
        print(f"\n{'='*70}")
        print(f"PROCESSING DATASET FAMILY: {dataset_family} ({family_idx + 1}/{len(dataset_families)})")
        print(f"{'='*70}")
        
        # Get data info without loading all data
        print(f"\nGetting {dataset_family} data info...")
        try:
            vel_path, seis_path, n_samples, use_mmap = get_data_info(config, dataset_family=dataset_family)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print(f"Skipping {dataset_family}...")
            continue
        
        # Determine which samples to process
        if args.sample_idx is not None:
            sample_indices = [args.sample_idx]
        else:
            sample_indices = list(range(n_samples))
            if args.max_samples is not None:
                sample_indices = sample_indices[:args.max_samples]
        
        print(f"Processing {len(sample_indices)} samples for {dataset_family}...")
        
        # Create family output directory
        family_output_dir = output_dir / dataset_family
        family_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each sample (load on-demand)
        for sample_pos, sample_idx in enumerate(sample_indices):
            print(f"\n{'='*70}")
            print(f"Processing {dataset_family} sample {sample_pos + 1}/{len(sample_indices)} (index {sample_idx})")
            print(f"{'='*70}")
            
            # Load single sample on-demand (using memory mapping)
            vel_sample, seis_sample = load_single_sample_from_file(
                vel_path, seis_path, sample_idx, use_mmap=use_mmap
            )
            
            # Run ensemble inversions
            print(f"\nRunning {args.n_runs} ensemble inversions...")
            all_results = []
            
            # Calculate number of batches needed
            n_batches = (args.n_runs + args.batch_size - 1) // args.batch_size  # Ceiling division
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * args.batch_size
                batch_end = min(batch_start + args.batch_size, args.n_runs)
                current_batch_size = batch_end - batch_start
                
                print(f"\nBatch {batch_idx + 1}/{n_batches} (runs {batch_start + 1}-{batch_end})...")
                
                result_batch = run_single_inversion(
                    vel_sample,
                    seis_sample,
                    config,
                    inversion_engine,
                    fwi_forward,
                    device,
                    run_idx=batch_idx,  # Use batch_idx as run_idx for logging
                    batch_size=current_batch_size
                )
                
                # result_batch is a list of results (one per sample in batch)
                all_results.extend(result_batch)
                
                for i, result in enumerate(result_batch):
                    run_num = batch_start + i + 1
                    print(f"  Run {run_num}/{args.n_runs}: SSIM={result['ssim']:.4f}, MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")
            
            # Stack all ensemble results: shape (n_runs, H, W)
            ensemble_results = np.stack([r['result'] for r in all_results])
            
            # Save ensemble results: {family}_{sample_idx+1}_ensemble.npy
            # Note: sample_idx+1 because user wants 1-indexed filenames
            ensemble_filename = family_output_dir / f"{dataset_family}_{sample_idx + 1}_ensemble.npy"
            np.save(ensemble_filename, ensemble_results)
            print(f"\nSaved ensemble results: {ensemble_filename}")
            print(f"  Shape: {ensemble_results.shape} (n_runs={args.n_runs}, H={ensemble_results.shape[1]}, W={ensemble_results.shape[2]})")
        
        print(f"\n{dataset_family} processing complete! Processed {len(sample_indices)} samples.")
    
    print(f"\n{'='*70}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(dataset_families)} dataset families")


if __name__ == "__main__":
    main()

