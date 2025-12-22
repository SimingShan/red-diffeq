#!/usr/bin/env python
"""
Batch post-processing script for all .npz result files in an experiment directory.

Processes all *_results.npz files and saves post-processed versions in the same .npz format,
preserving all original arrays and adding/replacing the 'result' with post-processed version.

Usage:
    python scripts/batch_post_process.py experiment/OpenFWI/OpenFWI_Clean/m4_clean --config configs/openfwi/model_4.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Tuple, Dict
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from red_diffeq import (
    load_config,
    GaussianDiffusion,
    Unet,
    RED_DiffEq_POST_PROCESS,
)


def setup_device() -> torch.device:
    """Setup computation device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_diffusion_model(config_path: str, device: torch.device) -> GaussianDiffusion:
    """Initialize and load pretrained diffusion model from config."""
    config = load_config(Path(config_path))

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
        raise FileNotFoundError(f"Pretrained model not found at {model_path}")

    diffusion.eval()
    return diffusion


def post_process_single(
    data: np.ndarray,
    post_processor: RED_DiffEq_POST_PROCESS,
    timesteps: int,
    device: torch.device,
    target_size: int = 72,
) -> np.ndarray:
    """Post-process a single velocity model (2D array).

    Args:
        data: Input 2D array (H, W)
        post_processor: RED_DiffEq_POST_PROCESS instance
        timesteps: Number of diffusion timesteps
        device: Computation device
        target_size: Target image size for padding

    Returns:
        Post-processed 2D array (same shape as input)
    """
    original_h, original_w = data.shape

    # Convert to tensor (1, 1, H, W)
    mu_tensor = torch.from_numpy(data[np.newaxis, np.newaxis, :, :]).float().to(device)

    # Calculate padding
    pad_h = max(0, target_size - original_h)
    pad_w = max(0, target_size - original_w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad if needed
    if original_h != target_size or original_w != target_size:
        mu_tensor_padded = F.pad(mu_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
    else:
        mu_tensor_padded = mu_tensor

    # Post-process (expects [-1, 1] input, outputs [0, 1])
    with torch.no_grad():
        denoised_padded = post_processor.diffusion_denoise(mu_tensor_padded, timesteps=timesteps)

    # Unpad if needed
    if original_h != target_size or original_w != target_size:
        denoised_tensor = denoised_padded[:, :, pad_top:pad_top+original_h, pad_left:pad_left+original_w]
    else:
        denoised_tensor = denoised_padded

    # Convert output from [0, 1] back to [-1, 1]
    denoised_tensor = (denoised_tensor * 2) - 1

    # Convert back to numpy and remove batch/channel dims
    result = denoised_tensor[0, 0].cpu().numpy()

    return result


def process_npz_file(
    npz_path: Path,
    post_processor: RED_DiffEq_POST_PROCESS,
    timesteps: int,
    device: torch.device,
    target_size: int,
    output_suffix: str = "_post",
    result_key: str = "result",
) -> Path:
    """Process a single .npz file and save post-processed version.

    Args:
        npz_path: Path to input .npz file
        post_processor: Post-processor instance
        timesteps: Number of diffusion timesteps
        device: Computation device
        target_size: Target size for padding
        output_suffix: Suffix to add to output filename
        result_key: Key to extract and post-process from npz

    Returns:
        Path to saved output file
    """
    # Load npz file
    data = dict(np.load(npz_path))

    # Check if result key exists
    if result_key not in data:
        raise KeyError(f"Key '{result_key}' not found in {npz_path}. Available: {list(data.keys())}")

    # Extract and post-process the result
    original_result = data[result_key]

    # Post-process
    post_processed_result = post_process_single(
        original_result,
        post_processor,
        timesteps,
        device,
        target_size,
    )

    # Replace the result with post-processed version
    data[result_key] = post_processed_result

    # Determine output path
    # Change 0_results.npz -> 0_results_post_t100.npz
    stem = npz_path.stem  # e.g., "0_results"
    output_name = f"{stem}{output_suffix}_t{timesteps}.npz"
    output_path = npz_path.parent / output_name

    # Save as npz with all arrays
    np.savez_compressed(output_path, **data)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch post-process all .npz result files in an experiment directory"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (will process all subdirectories recursively)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100,
        help="Number of diffusion timesteps for post-processing (default: 100)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_results.npz",
        help="File pattern to match (default: *_results.npz)"
    )
    parser.add_argument(
        "--result-key",
        type=str,
        default="result",
        help="Key to extract and post-process from npz (default: 'result')"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_post",
        help="Suffix to add to output filenames (default: '_post')"
    )

    args = parser.parse_args()

    # Setup
    device = setup_device()
    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Find all matching files
    npz_files = sorted(experiment_dir.rglob(args.pattern))

    if not npz_files:
        print(f"No files matching '{args.pattern}' found in {experiment_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Batch Post-Processing Configuration")
    print(f"{'='*80}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Config file: {args.config}")
    print(f"File pattern: {args.pattern}")
    print(f"Files found: {len(npz_files)}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Result key: '{args.result_key}'")
    print(f"Output suffix: '{args.output_suffix}'")
    print(f"{'='*80}\n")

    # Load config
    config = load_config(Path(args.config))
    target_size = config.diffusion.image_size

    # Load diffusion model
    print("Loading diffusion model...")
    diffusion_model = load_diffusion_model(args.config, device)
    post_processor = RED_DiffEq_POST_PROCESS(diffusion_model)
    print()

    # Process all files
    print(f"Processing {len(npz_files)} files...\n")

    successful = 0
    failed = 0

    for npz_file in tqdm(npz_files, desc="Post-processing files"):
        try:
            output_path = process_npz_file(
                npz_file,
                post_processor,
                args.timesteps,
                device,
                target_size,
                args.output_suffix,
                args.result_key,
            )
            successful += 1

        except Exception as e:
            print(f"\nError processing {npz_file}: {e}")
            failed += 1
            continue

    # Summary
    print(f"\n{'='*80}")
    print(f"Post-Processing Complete")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(npz_files)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(npz_files)}")
    print(f"{'='*80}\n")

    # Show example output location
    if successful > 0:
        example_input = npz_files[0]
        example_output = example_input.parent / f"{example_input.stem}{args.output_suffix}_t{args.timesteps}.npz"
        print("Output files saved with naming pattern:")
        print(f"  Input:  {example_input.name}")
        print(f"  Output: {example_output.name}\n")


if __name__ == "__main__":
    main()
