#!/usr/bin/env python
"""
Post-processing script for .npz result files (modified to handle npz directly).

Usage:
    python scripts/post_process_npz.py experiment/OpenFWI/OpenFWI_Clean/m4_clean/CF/0_results.npz --config configs/openfwi/model_4.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Tuple
import numpy as np
import torch
from tqdm import tqdm

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


def load_npz_file(file_path: Path, array_key: str = 'result') -> Tuple[np.ndarray, tuple]:
    """Load array from .npz file and ensure correct shape for processing.

    Args:
        file_path: Path to .npz file.
        array_key: Key to extract from npz file (default: 'result')

    Returns:
        Tuple of (data_4d, original_shape)
    """
    npz_data = np.load(file_path)

    # Check if key exists
    if array_key not in npz_data:
        available_keys = list(npz_data.keys())
        raise KeyError(f"Key '{array_key}' not found in {file_path}. Available keys: {available_keys}")

    data = npz_data[array_key]
    original_shape = data.shape

    # Handle different input shapes - convert to 4D for processing
    if data.ndim == 2:
        # (height, width) -> (1, 1, height, width)
        data_4d = data[np.newaxis, np.newaxis, :, :]
    elif data.ndim == 3:
        # Assume (channels, height, width) -> (1, channels, height, width)
        data_4d = data[np.newaxis, :, :, :]
    elif data.ndim == 4:
        # Already (batch, channels, height, width)
        data_4d = data
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}. Expected 2D, 3D, or 4D.")

    print(f"Loaded '{array_key}' from npz: {original_shape} -> Processing shape: {data_4d.shape}")
    return data_4d, original_shape


def post_process_batch(
    data: np.ndarray,
    post_processor: RED_DiffEq_POST_PROCESS,
    timesteps: int,
    device: torch.device,
    target_size: int = 72,
    batch_size: int = 1,
) -> Tuple[np.ndarray, tuple]:
    """Post-process a batch of velocity models."""
    import torch.nn.functional as F

    original_shape = data.shape
    current_h, current_w = data.shape[2], data.shape[3]

    total_batches = data.shape[0]
    results = []

    # Calculate padding
    pad_h = max(0, target_size - current_h)
    pad_w = max(0, target_size - current_w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    for i in tqdm(range(0, total_batches, batch_size), desc="Post-processing"):
        batch_end = min(i + batch_size, total_batches)
        batch_data = data[i:batch_end]

        mu_tensor = torch.from_numpy(batch_data).float().to(device)

        # Pad to target_size if needed
        if current_h != target_size or current_w != target_size:
            mu_tensor_padded = F.pad(mu_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        else:
            mu_tensor_padded = mu_tensor

        # Post-process
        with torch.no_grad():
            denoised_padded = post_processor.diffusion_denoise(mu_tensor_padded, timesteps=timesteps)

        # Unpad
        if current_h != target_size or current_w != target_size:
            denoised_tensor = denoised_padded[:, :, pad_top:pad_top+current_h, pad_left:pad_left+current_w]
        else:
            denoised_tensor = denoised_padded

        # Convert output from [0, 1] back to [-1, 1]
        denoised_tensor = (denoised_tensor * 2) - 1

        denoised_numpy = denoised_tensor.cpu().numpy()
        results.append(denoised_numpy)

    result = np.concatenate(results, axis=0)
    return result, original_shape


def restore_shape(result_4d: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Restore the original shape from 4D processed result."""
    if len(original_shape) == 2:
        return result_4d[0, 0]
    elif len(original_shape) == 3:
        return result_4d[0]
    elif len(original_shape) == 4:
        return result_4d
    else:
        raise ValueError(f"Unsupported original shape: {original_shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process .npz result files using diffusion denoising"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input .npz file (e.g., *_results.npz)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--array-key",
        type=str,
        default="result",
        help="Key to extract from npz file (default: 'result')"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100,
        help="Number of diffusion timesteps for post-processing (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: input_file with '_post_tN' suffix)"
    )

    args = parser.parse_args()

    # Setup
    device = setup_device()
    input_path = Path(args.input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if args.output is None:
        # Replace _results.npz with _result_post_t100.npy
        stem = input_path.stem.replace("_results", "_result")
        output_path = input_path.parent / f"{stem}_post_t{args.timesteps}.npy"
    else:
        output_path = Path(args.output)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Extracting array key: '{args.array_key}'")
    print(f"Post-processing timesteps: {args.timesteps}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Load config
    config = load_config(Path(args.config))
    target_size = config.diffusion.image_size

    # Load diffusion model
    print("Loading diffusion model...")
    diffusion_model = load_diffusion_model(args.config, device)
    post_processor = RED_DiffEq_POST_PROCESS(diffusion_model)
    print()

    # Load input data from npz
    print("Loading input data from npz...")
    data_4d, original_shape = load_npz_file(input_path, args.array_key)

    # Check input range
    data_min, data_max = data_4d.min(), data_4d.max()
    print(f"Input data range: [{data_min:.4f}, {data_max:.4f}]")
    if data_min < -1.1 or data_max > 1.1:
        print(f"WARNING: Input data appears to be in physical scale (not normalized).")
        print(f"         Expected range: [-1, 1] (normalized velocity space)")
    print()

    # Post-process
    print("Post-processing...")
    result_4d, _ = post_process_batch(
        data_4d,
        post_processor,
        timesteps=args.timesteps,
        device=device,
        target_size=target_size,
        batch_size=args.batch_size,
    )

    # Restore original shape
    result = restore_shape(result_4d, original_shape)

    # Check output range
    result_min, result_max = result.min(), result.max()
    print(f"\nOutput data range: [{result_min:.4f}, {result_max:.4f}]")
    print()

    # Save result
    print(f"Saving result to {output_path}...")
    np.save(output_path, result)
    print(f"Done! Result saved to: {output_path}")
    print(f"Output shape: {result.shape}")


if __name__ == "__main__":
    main()
