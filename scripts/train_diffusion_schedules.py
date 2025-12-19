#!/usr/bin/env python3
"""
Train diffusion models with different beta schedules.

This script trains 3 diffusion models with:
- Linear schedule
- Cosine schedule
- Sigmoid schedule

Each model is trained for 100k iterations with:
- Training loss logging
- FID evaluation during training
- Checkpointing every 10k steps
"""

import sys
import torch
import numpy as np
import torch.nn.functional as F
import json
from datetime import datetime
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from red_diffeq import Unet, GaussianDiffusion
from trainer_with_logging import TrainerWithLogging


def load_training_data():
    """
    Load training data from OpenFWI dataset.

    Data loading pipeline (DO NOT MODIFY):
    - CurveVel_b: 54 samples (90% of 60)
    - FlatVel_b: 54 samples (90% of 60)
    - CurveFault_b: 54 samples (3 velocities × 18 samples)
    - FlatFault_b: 54 samples (3 velocities × 18 samples)

    Total: 216 training samples
    """
    print("="*70)
    print("LOADING TRAINING DATA")
    print("="*70)

    training_images = []
    num_dataset_loaded = 0

    # CurveVel_b (54 samples)
    print("\nLoading CurveVel_b...")
    for i in range(1, 55):
        training_images.append(np.load(f'dataset/Velocity_Data/CurveVel_b/model{i}.npy'))
        num_dataset_loaded += 1
    print(f'  ✓ Loaded CurveVel_b/model1-54.npy ({num_dataset_loaded} samples)')

    # FlatVel_b (54 samples)
    print("\nLoading FlatVel_b...")
    start_count = num_dataset_loaded
    for i in range(1, 55):
        training_images.append(np.load(f'dataset/Velocity_Data/FlatVel_b/model{i}.npy'))
        num_dataset_loaded += 1
    print(f'  ✓ Loaded FlatVel_b/model1-54.npy ({num_dataset_loaded - start_count} samples)')

    # CurveFault_b (54 samples)
    print("\nLoading CurveFault_b...")
    start_count = num_dataset_loaded
    for i in [6, 7, 8]:
        for j in range(18):
            training_images.append(np.load(f'dataset/Velocity_Data/CurveFault_b/vel{i}_1_{j}.npy'))
            num_dataset_loaded += 1
    print(f'  ✓ Loaded CurveFault_b/vel[6,7,8]_1_[0-17].npy ({num_dataset_loaded - start_count} samples)')

    # FlatFault_b (54 samples)
    print("\nLoading FlatFault_b...")
    start_count = num_dataset_loaded
    for i in [6, 7, 8]:
        for j in range(18):
            training_images.append(np.load(f'dataset/Velocity_Data/FlatFault_b/vel{i}_1_{j}.npy'))
            num_dataset_loaded += 1
    print(f'  ✓ Loaded FlatFault_b/vel[6,7,8]_1_[0-17].npy ({num_dataset_loaded - start_count} samples)')

    # Concatenate and normalize
    training_images = np.concatenate(training_images)
    print(f"\n✓ Total samples loaded: {num_dataset_loaded}")
    print(f"  Raw data shape: {training_images.shape}")
    print(f"  Raw data range: [{training_images.min():.2f}, {training_images.max():.2f}] m/s")

    # Normalize to [0, 1]
    training_images = (training_images - 1500) / 3000
    print(f"  Normalized range: [{training_images.min():.4f}, {training_images.max():.4f}]")

    # Convert to tensor and pad to 72x72
    training_images = torch.as_tensor(training_images)
    training_images = F.pad(training_images, (1, 1, 1, 1), "constant", 0)
    print(f"  Final shape after padding: {training_images.shape}")

    print("="*70)
    return training_images


def train_model_with_schedule(schedule_name, training_images, train_num_steps=600000):
    """
    Train a diffusion model with specified beta schedule.

    Args:
        schedule_name: 'linear', 'cosine', or 'sigmoid'
        training_images: Training dataset tensor
        train_num_steps: Number of training iterations (default: 600k)
    """
    print("\n" + "="*70)
    print(f"TRAINING MODEL WITH {schedule_name.upper()} SCHEDULE")
    print("="*70)
    print(f"Schedule: {schedule_name}")
    print(f"Training steps: {train_num_steps:,}")
    print(f"Save/sample every: 100,000 steps")
    print("="*70)

    # Create model architecture
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
        channels=1
    )

    # Create diffusion model with specified schedule
    diffusion = GaussianDiffusion(
        model,
        image_size=72,
        timesteps=1000,           # number of diffusion steps
        sampling_timesteps=250,    # DDIM sampling steps
        objective='pred_noise',
        beta_schedule=schedule_name  # linear, cosine, or sigmoid
    )

    # Setup results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f'./diffusion_models_{schedule_name}_{timestamp}'

    # Create trainer with logging
    trainer = TrainerWithLogging(
        diffusion,
        training_images,
        train_batch_size=32,
        train_lr=0.0002,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=True,                      # mixed precision training
        calculate_fid=True,            # ENABLE FID calculation
        save_and_sample_every=100000,  # Save every 100k steps
        num_fid_samples=5000,          # Number of samples for FID
        results_folder=results_folder
    )

    print(f"\n✓ Model created with {schedule_name} schedule")
    print(f"✓ Results will be saved to: {results_folder}")
    print(f"✓ FID calculation: ENABLED")
    print(f"✓ Training loss will be logged to: {results_folder}/training_loss.json")
    print(f"✓ FID scores will be logged to: {results_folder}/fid_scores.json")
    print("\nStarting training...")
    print("="*70)

    # Train the model
    trainer.train()

    print("\n" + "="*70)
    print(f"✓ TRAINING COMPLETE: {schedule_name.upper()} SCHEDULE")
    print(f"✓ Results saved to: {results_folder}")
    print("="*70)

    return results_folder


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("DIFFUSION MODEL TRAINING - BETA SCHEDULE COMPARISON")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    print("  - Schedules: linear, cosine, sigmoid")
    print("  - Iterations per schedule: 600,000")
    print("  - Batch size: 32")
    print("  - Learning rate: 0.0002")
    print("  - FID calculation: Enabled")
    print("  - Checkpoint frequency: Every 100,000 steps")
    print("="*70)

    # Load training data once
    training_images = load_training_data()

    # Train models with different schedules
    schedules = ['linear', 'cosine', 'sigmoid']
    results = {}

    for schedule in schedules:
        try:
            results_folder = train_model_with_schedule(
                schedule,
                training_images,
                train_num_steps=600000
            )
            results[schedule] = {
                'status': 'success',
                'results_folder': results_folder
            }
        except Exception as e:
            print(f"\n✗ ERROR training {schedule} schedule: {e}")
            results[schedule] = {
                'status': 'failed',
                'error': str(e)
            }

    # Save summary
    summary_path = Path('training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'schedules': schedules,
            'iterations_per_schedule': 600000,
            'results': results
        }, f, indent=2)

    # Print final summary
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE")
    print("="*70)
    print("\nSummary:")
    for schedule, result in results.items():
        if result['status'] == 'success':
            print(f"  ✓ {schedule:10s}: {result['results_folder']}")
        else:
            print(f"  ✗ {schedule:10s}: FAILED - {result.get('error', 'Unknown error')}")

    print(f"\nFull summary saved to: {summary_path}")
    print("="*70)


if __name__ == '__main__':
    main()
