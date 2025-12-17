#!/usr/bin/env python
"""
Wrapper script to run the old implementation directly using red_diff_save.py.

This script uses the old red_diff_save.py logic to ensure exact matching with previous results.
It's useful for:
1. Comparing new vs old implementation
2. Reproducing exact old results
3. Debugging differences

The old implementation is self-contained in red_diff_save.py, so we can use it directly.

Usage:
    python scripts/run_old_implementation.py --regularization diffusion --lr 0.03 --ts 300 --sigma 10 --noise_std 0.5
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import os

# Import from red_diff_save.py (old implementation)
from red_diff_save import RED_DiffEq
from red_diffeq.models.diffusion import GaussianDiffusion, Unet
from red_diffeq.solvers.pde import FWIForward
from red_diffeq.utils.data_trans import DataTransformer, v_denormalize, s_normalize_none
from red_diffeq.utils.ssim import SSIM
from accelerate import Accelerator
from torch.optim import Adam


def run_old_implementation(
    regularization=None,
    lr=0.03,
    ts=300,
    sigma=10.0,
    loss_type='l1',
    noise_std=0.0,
    initial_type='smoothed',
    missing_number=0,
    method='red_diff',
    reg_lambda=0.01,
    seismic_data_dir="dataset/Test_Data/Seismic_Data_Test/",
    velocity_data_dir="dataset/Test_Data/Velocity_Data_Test/",
    model_path="model-4.pt",
    results_dir_base="experiment/"
):
    """
    Run the old implementation using red_diff_save.py logic.
    
    This matches the old main.py behavior but uses the new package structure where possible.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Context setup (matching old implementation)
    ctx = {
        'n_grid': 70, 'nt': 1000, 'dx': 10, 'nbc': 120,
        'dt': 1e-3, 'f': 15, 'sz': 10, 'gz': 10, 'ng': 70, 'ns': 5
    }
    
    # Initialize forward operator
    data_trans = DataTransformer()
    fwi_forward = FWIForward(
        ctx, device, normalize=True,
        v_denorm_func=v_denormalize,
        s_norm_func=s_normalize_none
    )
    
    # Initialize the model and diffusion process (matching old implementation)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
        channels=1
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size=72,
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_noise'
    ).to(device)
    
    # Use Accelerator with mixed precision (matching old implementation exactly)
    accelerator = Accelerator(
        split_batches=True,
        mixed_precision='fp16'
    )
    opt = Adam(diffusion.parameters(), lr=20, betas=(0.9, 0.99))
    diffusion, opt = accelerator.prepare(diffusion, opt)
    diffusion = accelerator.unwrap_model(diffusion)
    
    # Load model
    model_path_expanded = os.path.expanduser(model_path)
    if os.path.exists(model_path_expanded):
        diffusion.load_state_dict(torch.load(model_path_expanded, map_location=device)['model'])
        print(f"Loaded pretrained model from: {model_path_expanded}")
    else:
        print(f"WARNING: Pretrained model not found at {model_path_expanded}")
        print("Continuing with randomly initialized model...")
    
    diffusion.eval()
    
    # Create results directory
    dir_identifier = regularization if regularization else 'pure'
    args_str = f"{method}_{dir_identifier}_lr{lr}_ts{ts}_sigma{sigma}_loss{loss_type}_noisestd{noise_std}_missing{missing_number}"
    results_dir = os.path.join(results_dir_base, f"{args_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Process data files
    seismic_base_dir = Path(seismic_data_dir)
    velocity_base_dir = Path(velocity_data_dir)
    
    if not seismic_base_dir.exists():
        raise FileNotFoundError(f"Seismic data directory not found: {seismic_base_dir}")
    if not velocity_base_dir.exists():
        raise FileNotFoundError(f"Velocity data directory not found: {velocity_base_dir}")
    
    family_name_list = [f.name for f in seismic_base_dir.glob("*.npy")]
    
    if not family_name_list:
        raise ValueError(f"No .npy files found in {seismic_base_dir}")
    
    # Initialize RED_DiffEq (from old implementation)
    # Note: We need to adapt this to work with new structure
    # For now, we'll use a simplified approach that calls the old method
    
    print("\n" + "=" * 70)
    print("Processing families...")
    print("=" * 70)
    
    for family_name in family_name_list:
        family_results_dir = os.path.join(results_dir, family_name)
        os.makedirs(family_results_dir, exist_ok=True)
        
        seismic_path = seismic_base_dir / family_name
        velocity_path = velocity_base_dir / family_name
        
        if not seismic_path.exists():
            print(f"Warning: Seismic file not found: {seismic_path}, skipping...")
            continue
        if not velocity_path.exists():
            print(f"Warning: Velocity file not found: {velocity_path}, skipping...")
            continue
        
        seis_data = np.load(seismic_path)
        velocity_data = np.load(velocity_path)
        print(f"\nProcessing {family_name}: {seis_data.shape[0]} samples")
        
        mus, losses, reg_losses_raw_all = [], [], []
        matrices = {'MAE': [], 'RMSE': [], 'SSIM': []}
        
        ssim_loss = SSIM(window_size=11)
        l1_loss = torch.nn.L1Loss()
        l2_loss = torch.nn.MSELoss()
        
        for j in range(seis_data.shape[0]):
            seis_slice = torch.from_numpy(seis_data[j:j+1]).float().to(device)
            vel_slice = torch.from_numpy(velocity_data[j:j+1]).float()
            
            # Prepare initial model (matching old implementation)
            initial_model = data_trans.prepare_initial_model(vel_slice, initial_type, sigma=sigma)
            initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)
            
            # Use the old RED_DiffEq class from red_diff_save.py
            # Note: This requires the old class structure
            # For now, we'll use a workaround by importing the class directly
            try:
                # Try to use the old implementation
                from red_diff_save import RED_DiffEq as OldRED_DiffEq
                
                red_diff = OldRED_DiffEq(
                    diffusion_model=diffusion,
                    data_trans_module=data_trans,
                    data_vis_module=None,  # Not needed for computation
                    pytorch_ssim_module=None  # We'll use new SSIM
                )
                
                if method == 'red_diff':
                    mu, total_losses, obs_losses, reg_losses, reg_losses_raw, current_MAE, current_RMSE, current_SSIM = red_diff.red_diff_sample(
                        mu=initial_model,
                        mu_true=vel_slice,
                        y=seis_slice,
                        ts=ts,
                        lr=lr,
                        fwi_forward=fwi_forward,
                        regularization=regularization,
                        plot_show=False,
                        loss_type=loss_type,
                        noise_std=noise_std,
                        missing_number=missing_number,
                        reg_lambda=reg_lambda
                    )
                else:
                    raise ValueError(f"Invalid method: {method}")
                
                # Process results (matching old implementation)
                mu_tensor = mu[:, :, 1:-1, 1:-1].detach().cpu()
                mu_numpy = mu[:, :, 1:-1, 1:-1].detach().cpu().numpy()
                vm_data_norm = torch.tensor(data_trans.v_normalize(velocity_data[j:j+1]))
                mae = l1_loss(mu_tensor, vm_data_norm).item()
                mse = l2_loss(mu_tensor, vm_data_norm).item()
                rmse = np.sqrt(mse)
                ssim_val = ssim_loss((mu_tensor + 1) / 2, (vm_data_norm + 1) / 2).item()
                
                matrices['MAE'].append(mae)
                matrices['RMSE'].append(rmse)
                matrices['SSIM'].append(ssim_val)
                
                mus.append(mu_numpy)
                losses.append(obs_losses)
                reg_losses_raw_all.append(reg_losses_raw)
                
                print(f"Sample {j}: MAE: {mae:.6f}, RMSE: {rmse:.6f}, SSIM: {ssim_val:.6f}")
                
            except Exception as e:
                print(f"Error processing sample {j}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save collective results for the family
        if mus:
            mu_result = np.concatenate(mus)
            losses_result = np.concatenate(losses)
            reg_losses_raw_result = np.concatenate(reg_losses_raw_all)
            
            average_mae = np.mean(matrices['MAE'])
            average_rmse = np.mean(matrices['RMSE'])
            average_ssim = np.mean(matrices['SSIM'])
            
            np.save(os.path.join(family_results_dir, 'velocity_sample.npy'), mu_result)
            np.save(os.path.join(family_results_dir, 'losses.npy'), losses_result)
            np.save(os.path.join(family_results_dir, 'reg_losses_raw.npy'), reg_losses_raw_result)
            
            print('*' * 50)
            print(f'{family_name} sampling is DONE')
            print(f'Average MAE: {average_mae:.6f}, RMSE: {average_rmse:.6f}, SSIM: {average_ssim:.6f}')
            
            with open(os.path.join(family_results_dir, 'metrics_summary.txt'), 'w') as f:
                f.write(f'Average MAE: {average_mae}\n')
                f.write(f'Average RMSE: {average_rmse}\n')
                f.write(f'Average SSIM: {average_ssim}\n')


def parse_args():
    """Parse command-line arguments matching old implementation."""
    parser = argparse.ArgumentParser(
        description="Run old RED-DiffEq implementation using red_diff_save.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--regularization",
        default=None,
        help="Regularization type (e.g., 'diffusion', 'tv')"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--ts",
        type=int,
        default=300,
        help="Number of timesteps for training"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10,
        help="Sigma for Gaussian blurring in initial model preparation"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default='l1',
        choices=['l1', 'l2', 'Huber'],
        help="Type of the loss function"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0,
        help="The standard deviation of the gaussian noise"
    )
    parser.add_argument(
        "--missing_number",
        type=int,
        default=0,
        help="The missing trace of the seismic data"
    )
    parser.add_argument(
        "--initial_type",
        type=str,
        default='smoothed',
        choices=['smoothed', 'homogeneous', 'linear'],
        help="Type of initial velocity model"
    )
    parser.add_argument(
        "--method",
        type=str,
        default='red_diff',
        help="Choose to run the benchmark or our method"
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.01,
        help="The regularization coefficient lambda"
    )
    parser.add_argument(
        "--seismic_data_dir",
        type=str,
        default="dataset/Test_Data/Seismic_Data_Test/",
        help="Directory containing seismic data"
    )
    parser.add_argument(
        "--velocity_data_dir",
        type=str,
        default="dataset/Test_Data/Velocity_Data_Test/",
        help="Directory containing velocity data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model-4.pt",
        help="Path to pretrained model file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiment/",
        help="Base directory for results"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 70)
    print("Running OLD Implementation (using red_diff_save.py)")
    print("=" * 70)
    print(f"Regularization: {args.regularization}")
    print(f"Learning rate: {args.lr}")
    print(f"Timesteps: {args.ts}")
    print(f"Sigma: {args.sigma}")
    print(f"Loss type: {args.loss_type}")
    print(f"Noise std: {args.noise_std}")
    print(f"Missing traces: {args.missing_number}")
    print(f"Initial type: {args.initial_type}")
    print(f"Method: {args.method}")
    print(f"Reg lambda: {args.reg_lambda}")
    print("=" * 70)
    print()
    
    # Call the old implementation
    run_old_implementation(
        regularization=args.regularization,
        lr=args.lr,
        ts=args.ts,
        sigma=args.sigma,
        loss_type=args.loss_type,
        noise_std=args.noise_std,
        initial_type=args.initial_type,
        missing_number=args.missing_number,
        method=args.method,
        reg_lambda=args.reg_lambda,
        seismic_data_dir=args.seismic_data_dir,
        velocity_data_dir=args.velocity_data_dir,
        model_path=args.model_path,
        results_dir_base=args.results_dir
    )
