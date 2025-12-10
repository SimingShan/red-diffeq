#!/usr/bin/env python3
"""
Generate seismic data from velocity models using the FWI forward solver.

Usage:
    python scripts/generate_seismic_data.py \
        --velocity_file dataset/Overthrust/Velocity_Data/overthurst.npy \
        --output_file dataset/Overthrust/Seismic_Data/overthrust.npy \
        --config configs/marmousi/baseline.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from tqdm import tqdm

from red_diffeq import (
    load_config,
    FWIForward,
    v_denormalize,
    s_normalize_none,
)


def normalize_velocity(v_phys):
    """
    Normalize velocity from physical range [1500, 4500] to [-1, 1].

    Args:
        v_phys: Velocity in m/s, shape (N, 1, H, W)

    Returns:
        v_norm: Normalized velocity in [-1, 1]
    """
    return (v_phys - 1500) / 3000 * 2 - 1


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_velocity_model(velocity_file):
    """Load velocity model from file."""
    print(f"Loading velocity model from: {velocity_file}")

    velocity_data = np.load(velocity_file)
    print(f"  Shape: {velocity_data.shape}")
    print(f"  Range: [{velocity_data.min():.2f}, {velocity_data.max():.2f}] m/s")
    print(f"  Mean: {velocity_data.mean():.2f} m/s")

    return velocity_data


def generate_seismic_data(velocity_data, config, device):
    """
    Generate seismic data using FWI forward solver.

    Args:
        velocity_data: Velocity models in physical units (N, 1, H, W)
        config: Configuration dict
        device: torch device

    Returns:
        seismic_data: Generated seismic data (N, ns, nt, ng)
    """
    print("\nInitializing forward solver...")

    # Create PDE context
    ctx = config.pde.to_dict()
    print(f"  Grid size: {ctx['n_grid']}")
    print(f"  Time steps: {ctx['nt']}")
    print(f"  Sources: {ctx['ns']}")
    print(f"  Receivers: {ctx['ng']}")
    print(f"  Frequency: {ctx['f']} Hz")

    # Initialize forward operator
    fwi_forward = FWIForward(
        ctx,
        device,
        normalize=True,
        v_denorm_func=v_denormalize,
        s_norm_func=s_normalize_none,
    )

    # Convert velocity to torch tensor and normalize
    print("\nProcessing velocity models...")
    v_tensor = torch.from_numpy(velocity_data).float().to(device)

    # Normalize velocity to [-1, 1]
    v_norm = normalize_velocity(v_tensor)
    print(f"  Normalized range: [{v_norm.min():.3f}, {v_norm.max():.3f}]")

    # Generate seismic data
    print("\nGenerating seismic data...")
    n_samples = velocity_data.shape[0]
    seismic_list = []

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Samples", unit="sample"):
            # Forward solve for this velocity model
            v_single = v_norm[i:i+1]  # Keep batch dimension
            s_single = fwi_forward(v_single)

            # Move to CPU and convert to numpy
            s_np = s_single.cpu().numpy()
            seismic_list.append(s_np)

    # Concatenate all samples
    seismic_data = np.concatenate(seismic_list, axis=0)

    print(f"\nGenerated seismic data shape: {seismic_data.shape}")
    print(f"  Range: [{seismic_data.min():.6f}, {seismic_data.max():.6f}]")
    print(f"  Mean: {seismic_data.mean():.6f}, Std: {seismic_data.std():.6f}")

    return seismic_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate seismic data from velocity models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--velocity_file",
        type=Path,
        required=True,
        help="Path to velocity model file (.npy)",
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to output seismic data file (.npy)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file (for PDE parameters)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )

    args = parser.parse_args()

    # Check if output file exists
    if args.output_file.exists() and not args.overwrite:
        print(f"Error: Output file already exists: {args.output_file}")
        print("Use --overwrite to overwrite existing file")
        return 1

    # Create output directory if needed
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Setup device
    device = setup_device()

    # Load velocity model
    velocity_data = load_velocity_model(args.velocity_file)

    # Generate seismic data
    seismic_data = generate_seismic_data(velocity_data, config, device)

    # Save seismic data
    print(f"\nSaving seismic data to: {args.output_file}")
    np.save(args.output_file, seismic_data)

    # Verify saved file
    print("\nVerifying saved file...")
    loaded = np.load(args.output_file)
    print(f"  Saved shape: {loaded.shape}")
    print(f"  Saved dtype: {loaded.dtype}")

    if np.allclose(loaded, seismic_data):
        print(f"\n✓ Seismic data generated successfully!")
    else:
        print(f"\n✗ Warning: Loaded data doesn't match generated data")
        return 1

    print(f"\n" + "="*60)
    print("Summary:")
    print(f"  Input velocity: {args.velocity_file}")
    print(f"  Output seismic: {args.output_file}")
    print(f"  Shape: {seismic_data.shape}")
    print(f"  Size: {args.output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
