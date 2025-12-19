#!/usr/bin/env python3
"""
Evaluate diffusion models by sampling and computing FID scores.

This script:
1. Loads diffusion models (model-1.pt, model-2.pt, model-3.pt, model-4.pt)
2. Samples 100 velocity models from each
3. Computes FID score against real data
4. Plots FID scores as a line plot
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime

from red_diffeq import GaussianDiffusion, Unet
from pytorch_fid import fid_score
from scipy import linalg
import torch.nn.functional as F


def load_diffusion_model(model_path: Path, device: torch.device):
    """Load a pretrained diffusion model."""
    print(f"Loading model from {model_path}...")

    # Create model architecture (standard for all models)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
        channels=1,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=72,
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_noise',
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    diffusion.load_state_dict(checkpoint['model'])
    diffusion.eval()

    print(f"✓ Model loaded successfully")
    return diffusion


def sample_from_model(diffusion, num_samples: int, batch_size: int, device: torch.device):
    """Sample velocity models from a diffusion model.

    Args:
        diffusion: Diffusion model
        num_samples: Total number of samples to generate
        batch_size: Batch size for sampling
        device: Device to use

    Returns:
        Tensor of shape (num_samples, 1, 72, 72) with values in [0, 1]
        (diffusion model auto-unnormalizes from [-1,1] to [0,1])
    """
    print(f"Sampling {num_samples} images...")

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Sampling batches"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            # Sample from diffusion model
            # Note: diffusion.sample() returns [0, 1] due to auto_normalize=True
            samples = diffusion.sample(batch_size=current_batch_size)
            all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    print(f"✓ Sampled {all_samples.shape[0]} images, shape: {all_samples.shape}")
    print(f"  Range: [{all_samples.min():.4f}, {all_samples.max():.4f}] (should be ~[0, 1])")

    return all_samples


def load_real_data(data_dir: Path, num_samples: int):
    """Load real velocity models from dataset.

    Args:
        data_dir: Directory containing .npy files
        num_samples: Number of samples to load

    Returns:
        Tensor of shape (num_samples, 1, H, W) with values in [0, 1]
        (normalized from physical range [1500, 4500] to match sampled data)
    """
    print(f"Loading {num_samples} real samples from {data_dir}...")

    # Find all .npy files
    npy_files = sorted(list(data_dir.glob("*.npy")))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    print(f"Found {len(npy_files)} data files: {[f.name for f in npy_files]}")

    # Load balanced samples from all files (matching training distribution)
    # Training used mix of CV, FV, CF, FF - so we sample equally from each
    all_data = []
    samples_per_file = num_samples // len(npy_files)
    remainder = num_samples % len(npy_files)

    for idx, npy_file in enumerate(npy_files):
        data = np.load(npy_file, mmap_mode='r')

        # Get equal samples from each file, with remainder distributed
        needed = samples_per_file + (1 if idx < remainder else 0)
        needed = min(needed, data.shape[0])  # Don't exceed file size

        # Load needed samples
        batch = data[:needed]
        all_data.append(batch)

        print(f"  Loaded {needed} from {npy_file.name}")

    # Concatenate and convert to tensor
    real_data = np.concatenate(all_data, axis=0)[:num_samples]
    real_tensor = torch.from_numpy(real_data).float()

    print(f"  Raw data range: [{real_tensor.min():.2f}, {real_tensor.max():.2f}] (physical units)")

    # Normalize to [0, 1] to match diffusion model output
    # Physical range is approximately [1500, 4500]
    # Map to [0, 1]: (v - 1500) / 3000
    real_normalized = (real_tensor - 1500.0) / 3000.0

    print(f"✓ Loaded {real_normalized.shape[0]} real samples, shape: {real_normalized.shape}")
    print(f"  Normalized range: [{real_normalized.min():.4f}, {real_normalized.max():.4f}] (should be ~[0, 1])")

    return real_normalized


def extract_inception_features(images: torch.Tensor, device: torch.device):
    """Extract features using InceptionV3 for FID calculation.

    Args:
        images: Tensor of shape (N, 1, H, W) in [0, 1] range
        device: Device to use

    Returns:
        Features tensor of shape (N, 2048)
    """
    from torchvision.models import inception_v3, Inception_V3_Weights

    # Load InceptionV3
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    inception.fc = torch.nn.Identity()  # Remove final classification layer
    inception = inception.to(device)
    inception.eval()

    # Images are already in [0, 1] range (both real and sampled)
    # Convert grayscale to RGB (repeat channel 3 times)
    if images.shape[1] == 1:
        images_rgb = images.repeat(1, 3, 1, 1)
    else:
        images_rgb = images

    # Resize to 299x299 (InceptionV3 input size)
    images_resized = F.interpolate(images_rgb, size=(299, 299), mode='bilinear', align_corners=False)

    # Apply ImageNet normalization (CRITICAL for correct FID!)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images_resized.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images_resized.device).view(1, 3, 1, 1)
    images_normalized = (images_resized - mean) / std

    # Extract features in batches
    batch_size = 32
    all_features = []

    with torch.no_grad():
        for i in tqdm(range(0, images.shape[0], batch_size), desc="Extracting features"):
            batch = images_normalized[i:i + batch_size].to(device)
            features = inception(batch)
            all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=0)
    return all_features.numpy()


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray):
    """Calculate Fréchet Inception Distance.

    Args:
        real_features: Features from real images (N, D)
        fake_features: Features from generated images (M, D)

    Returns:
        FID score (lower is better)
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        print("Warning: FID calculation produced singular product; adding small epsilon to diagonal")
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return fid


def save_sample_images(samples: torch.Tensor, output_path: Path, num_display: int = 10):
    """Save sample images for visualization.

    Args:
        samples: Sampled images (N, 1, H, W)
        output_path: Path to save the figure
        num_display: Number of samples to display
    """
    num_display = min(num_display, samples.shape[0])

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_display):
        img = samples[i, 0].cpu().numpy()
        axes[i].imshow(img, cmap='seismic', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved sample images to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion models using FID")
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate per model')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for sampling')
    parser.add_argument('--data_dir', type=Path,
                       default=Path('dataset/OpenFWI/Velocity_Data/'),
                       help='Directory containing real velocity data')
    parser.add_argument('--models_dir', type=Path,
                       default=Path('pretrained_models'),
                       help='Directory containing pretrained models')
    parser.add_argument('--output_dir', type=Path,
                       default=Path('fid_evaluation'),
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Model paths
    model_names = ['model-1', 'model-2', 'model-3', 'model-4']
    model_paths = [args.models_dir / f"{name}.pt" for name in model_names]

    # Verify all models exist
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    # Load real data once (for all comparisons)
    print("\n" + "="*70)
    print("LOADING REAL DATA")
    print("="*70)
    real_data = load_real_data(args.data_dir, args.num_samples)

    # Extract features from real data once
    print("\n" + "="*70)
    print("EXTRACTING FEATURES FROM REAL DATA")
    print("="*70)
    real_features = extract_inception_features(real_data, device)

    # Evaluate each model
    fid_scores = []

    for model_idx, (model_name, model_path) in enumerate(zip(model_names, model_paths)):
        print("\n" + "="*70)
        print(f"EVALUATING {model_name.upper()} ({model_idx + 1}/{len(model_names)})")
        print("="*70)

        # Load model
        diffusion = load_diffusion_model(model_path, device)

        # Sample from model
        fake_data = sample_from_model(diffusion, args.num_samples, args.batch_size, device)

        # Save sample images
        sample_img_path = output_dir / f"{model_name}_samples.png"
        save_sample_images(fake_data, sample_img_path)

        # Extract features from generated data
        print("Extracting features from generated data...")
        fake_features = extract_inception_features(fake_data, device)

        # Calculate FID
        print("Calculating FID score...")
        fid = calculate_fid(real_features, fake_features)
        fid_scores.append(fid)

        print(f"\n✓ {model_name} FID Score: {fid:.4f}")

        # Clean up to save memory
        del diffusion, fake_data, fake_features
        torch.cuda.empty_cache()

    # Plot results
    print("\n" + "="*70)
    print("PLOTTING RESULTS")
    print("="*70)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(model_names) + 1), fid_scores, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('FID Score (lower is better)', fontsize=14)
    plt.title('Diffusion Model Evaluation - FID Scores', fontsize=16)
    plt.xticks(range(1, len(model_names) + 1), model_names)
    plt.grid(True, alpha=0.3)

    # Annotate points with values
    for i, (model_name, fid) in enumerate(zip(model_names, fid_scores)):
        plt.annotate(f'{fid:.2f}',
                    xy=(i + 1, fid),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10)

    plot_path = output_dir / 'fid_scores_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_path}")
    plt.close()

    # Save results to text file
    results_path = output_dir / 'fid_results.txt'
    with open(results_path, 'w') as f:
        f.write("FID Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Data directory: {args.data_dir}\n\n")
        f.write("FID Scores:\n")
        f.write("-" * 50 + "\n")
        for model_name, fid in zip(model_names, fid_scores):
            f.write(f"{model_name:15s}: {fid:8.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best model: {model_names[np.argmin(fid_scores)]} (FID: {min(fid_scores):.4f})\n")

    print(f"✓ Saved results to {results_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nFID Scores:")
    for model_name, fid in zip(model_names, fid_scores):
        print(f"  {model_name:15s}: {fid:8.4f}")
    print(f"\nBest model: {model_names[np.argmin(fid_scores)]} (FID: {min(fid_scores):.4f})")
    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
