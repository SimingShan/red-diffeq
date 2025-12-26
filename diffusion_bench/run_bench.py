import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from tqdm import tqdm

from red_diffeq import (
    load_config,
    save_config,
    GaussianDiffusion,
    Unet,
    FWIForward,
    SSIM,
    prepare_initial_model,
    s_normalize_none,
    v_denormalize,
)
from diffusion_bench import DiffusionFWI, ILVR_FWI
import ml_collections


def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_diffusion_model(config: ml_collections.ConfigDict, device: torch.device) -> GaussianDiffusion:
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
        print(f"✅ Loaded pretrained model from: {model_path}")
    else:
        print(f"WARNING: Pretrained model not found at {model_path}")
        print("Continuing with randomly initialized model...")

    diffusion.eval()
    return diffusion


def initialize_forward_operator(config: ml_collections.ConfigDict, device: torch.device) -> FWIForward:
    ctx = config.pde.to_dict()

    fwi_forward = FWIForward(
        ctx,
        device,
        normalize=True,
        v_denorm_func=v_denormalize,
        s_norm_func=s_normalize_none,
    )

    return fwi_forward


def get_data_files(config: ml_collections.ConfigDict) -> list:
    seismic_dir = Path(config.data.seismic_data_dir)

    if not seismic_dir.exists():
        raise FileNotFoundError(f"Seismic data directory not found: {seismic_dir}")

    pattern = config.data.data_pattern
    family_files = sorted(seismic_dir.glob(pattern))

    if not family_files:
        raise ValueError(f"No data files found matching {pattern} in {seismic_dir}")

    return [f.name for f in family_files]


def process_batch(
    batch_start: int,
    batch_end: int,
    seis_mmap: np.ndarray,
    vel_mmap: np.ndarray,
    config: ml_collections.ConfigDict,
    bench_method,  # DiffusionFWI or ILVR_FWI instance
    fwi_forward: FWIForward,
    device: torch.device,
) -> tuple:
    current_batch_size = batch_end - batch_start

    seis_batch = torch.from_numpy(seis_mmap[batch_start:batch_end].copy()).float().to(device)
    vel_batch = torch.from_numpy(vel_mmap[batch_start:batch_end].copy()).float()

    initial_models = []
    for i in range(current_batch_size):
        vel_slice = vel_batch[i : i + 1]
        initial_model = prepare_initial_model(
            vel_slice,
            config.optimization.initial_type,
            sigma=config.optimization.sigma,
        )
        initial_models.append(initial_model)

    initial_model_batch = torch.cat(initial_models, dim=0)

    opt_kwargs = {
        'ts': config.optimization.ts,
        'diffusion_ts': config.optimization.diffusion_ts,
        'lr': config.optimization.lr,
        'noise_std': config.optimization.noise_std,
        'noise_type': config.optimization.noise_type,
        'missing_number': config.optimization.missing_number,
        'grad_norm': config.optimization.get('grad_norm', True),  # Notebook default: True
        'grad_smooth': config.optimization.get('grad_smooth', None),  # Notebook default: disabled (None)
        'model_blur': config.optimization.get('model_blur', False),  # Notebook default: False
        'grad_clip': config.optimization.get('grad_clip', 1.0),  # Notebook default: 1.0
    }

    if isinstance(bench_method, ILVR_FWI):
        opt_kwargs['use_ilvr'] = config.optimization.get('use_ilvr', True)
        opt_kwargs['ilvr_weight'] = config.optimization.get('ilvr_weight', 0.05)
        opt_kwargs['ilvr_down_schedule'] = config.optimization.get('ilvr_down_schedule', 'linear')

    opt_kwargs['use_patches'] = config.optimization.get('use_patches', False)
    opt_kwargs['patch_kernel_size'] = config.optimization.get('patch_kernel_size', None)
    opt_kwargs['patch_stride'] = config.optimization.get('patch_stride', None)

    mu_batch, final_results_per_model = bench_method.optimize(
        initial_model_batch,
        vel_batch,
        seis_batch,
        fwi_forward,
        **opt_kwargs
    )

    return mu_batch, final_results_per_model, initial_model_batch, vel_batch


def save_batch_results(
    batch_start: int,
    batch_end: int,
    mu_batch: torch.Tensor,
    results_per_model: list,
    initial_model_batch: torch.Tensor,
    vel_batch: torch.Tensor,
    output_dir: Path,
) -> None:
    mu_batch_np = mu_batch.detach().cpu().numpy()
    vel_batch_np = vel_batch.cpu().numpy()
    initial_model_batch_np = initial_model_batch.detach().cpu().numpy()

    for i, model_idx in enumerate(range(batch_start, batch_end)):
        mu_result_2d = mu_batch_np[i, 0, :, :]
        initial_velocity_2d = initial_model_batch_np[i, 0, :, :]
        ground_truth_2d = vel_batch_np[i, 0, :, :]
        model_metrics = results_per_model[i]

        npz_data = {
            "result": mu_result_2d,
            "initial_velocity": initial_velocity_2d,
            "ground_truth": ground_truth_2d,
            "total_losses": np.array(model_metrics["total_losses"]),
            "obs_losses": np.array(model_metrics["obs_losses"]),
            "ssim": np.array(model_metrics["ssim"]),
            "mae": np.array(model_metrics["mae"]),
            "rmse": np.array(model_metrics["rmse"]),
        }

        npz_path = output_dir / f"{model_idx}_results.npz"
        np.savez(npz_path, **npz_data)


def run_experiment(config: ml_collections.ConfigDict, method: str = 'diffusionfwi') -> None:
    print("\n" + "=" * 70)
    print(f"BENCHMARK METHOD: {method.upper()}")
    print("=" * 70)
    print("Configuration:")
    print("=" * 70)
    for key, value in sorted(config.items()):
        if not isinstance(value, ml_collections.ConfigDict):
            print(f"  {key}: {value}")

    # Print key optimization parameters explicitly
    print("\n  Key optimization parameters:")
    print(f"    lr:           {config.optimization.lr}")
    print(f"    ts:           {config.optimization.ts}")
    print(f"    diffusion_ts: {config.optimization.diffusion_ts}")
    print(f"    grad_norm:    {config.optimization.get('grad_norm', True)}")
    print(f"    grad_smooth:  {config.optimization.get('grad_smooth', None)}")
    print(f"    model_blur:   {config.optimization.get('model_blur', False)}")
    print(f"    grad_clip:    {config.optimization.get('grad_clip', 1.0)}")
    if method.lower() in ['ilvr', 'ilvr_fwi']:
        print(f"    use_ilvr:     {config.optimization.get('use_ilvr', True)}")
        print(f"    ilvr_weight:  {config.optimization.get('ilvr_weight', 0.05)}")
    print("=" * 70 + "\n")

    base_seed = config.experiment.random_seed
    if base_seed is not None:
        from red_diffeq.utils.seed_utils import set_seed
        set_seed(base_seed, verbose=True, allow_tf32=True)

    device = setup_device()

    print("Initializing models...")
    diffusion = load_diffusion_model(config, device)
    fwi_forward = initialize_forward_operator(config, device)

    ssim_loss = SSIM(window_size=11, size_average=True)

    # Select benchmark method
    if method.lower() == 'ilvr' or method.lower() == 'ilvr_fwi':
        print("Using ILVR-FWI method")
        bench_method = ILVR_FWI(diffusion, fwi_forward, ssim_loss)
    else:
        print("Using DiffusionFWI method")
        bench_method = DiffusionFWI(diffusion, fwi_forward, ssim_loss)

    seismic_dir = Path(config.data.seismic_data_dir).resolve()
    dataset_name = seismic_dir.parts[-2] if len(seismic_dir.parts) >= 2 else None

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dataset_name:
        results_dir = Path(config.experiment.results_dir) / dataset_name / config.experiment.name / timestamp
    else:
        results_dir = Path(config.experiment.results_dir) / config.experiment.name / timestamp
    print(f"Results will be saved to: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = results_dir / "config.yaml"
    save_config(config, config_save_path)
    print(f"Configuration saved to: {config_save_path}")
    print(f"  → ts={config.optimization.ts}, diffusion_ts={config.optimization.diffusion_ts}")

    print("Loading data files...")
    family_files = get_data_files(config)
    print(f"Found {len(family_files)} data families to process")

    for family_name in family_files:
        print(f"\n{'='*70}")
        print(f"Processing: {family_name}")
        print(f"{'='*70}")

        family_results_dir = results_dir / Path(family_name).stem
        family_results_dir.mkdir(exist_ok=True)

        seismic_path = Path(config.data.seismic_data_dir) / family_name
        velocity_path = Path(config.data.velocity_data_dir) / family_name

        seis_mmap = np.load(seismic_path, mmap_mode="r")
        vel_mmap = np.load(velocity_path, mmap_mode="r")
        num_models = seis_mmap.shape[0]

        print(f"Number of models: {num_models}")
        print(f"Batch size: {config.data.batch_size}")

        num_batches = (num_models + config.data.batch_size - 1) // config.data.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Batches"):
            batch_start = batch_idx * config.data.batch_size
            batch_end = min(batch_start + config.data.batch_size, num_models)

            mu_batch, results, initial_batch, vel_batch = process_batch(
                batch_start,
                batch_end,
                seis_mmap,
                vel_mmap,
                config,
                bench_method,
                fwi_forward,
                device,
            )

            save_batch_results(
                batch_start,
                batch_end,
                mu_batch,
                results,
                initial_batch,
                vel_batch,
                family_results_dir,
            )

    print(f"\n{'='*70}")
    print(f"Experiment complete! Results saved to: {results_dir}")
    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark diffusion FWI methods (DiffusionFWI or ILVR-FWI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=['diffusionfwi', 'ilvr', 'ilvr_fwi'],
        default='diffusionfwi',
        help="Benchmark method: 'diffusionfwi' (standard) or 'ilvr'/'ilvr_fwi' (ILVR-enhanced)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )

    # Optimization parameters
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--ts", type=int, help="Number of FWI iterations per diffusion step")
    parser.add_argument("--diffusion_ts", type=int, help="Number of diffusion timesteps")

    # Gradient tricks
    parser.add_argument("--grad_norm", type=lambda x: x.lower() == 'true',
                       help="Enable gradient normalization (true/false)")
    parser.add_argument("--grad_smooth", type=float,
                       help="Gaussian gradient smoothing sigma (e.g., 1.0)")
    parser.add_argument("--model_blur", type=lambda x: x.lower() == 'true',
                       help="Enable model Gaussian blur (true/false)")
    parser.add_argument("--grad_clip", type=float,
                       help="Gradient clipping factor (e.g., 1.1)")

    # ILVR-specific parameters
    parser.add_argument("--use_ilvr", type=lambda x: x.lower() == 'true',
                       help="Enable ILVR conditioning (true/false, only for ilvr method)")
    parser.add_argument("--ilvr_weight", type=float,
                       help="ILVR weight (0.01-0.5, default 0.05)")
    parser.add_argument("--ilvr_down_schedule", type=str,
                       choices=['linear', 'stepwise'],
                       help="ILVR downsampling schedule: 'linear' (16→2) or 'stepwise' ([32,16,8,4])")

    # Patch parameters (for non-square images like Marmousi 70x190)
    parser.add_argument("--use_patches", type=lambda x: x.lower() == 'true',
                       help="Enable patch-based processing for large/non-square images (true/false)")
    parser.add_argument("--patch_height", type=int,
                       help="Patch height (default: 70)")
    parser.add_argument("--patch_width", type=int,
                       help="Patch width (default: 70)")
    parser.add_argument("--patch_stride_h", type=int,
                       help="Patch stride in height dimension (default: 1)")
    parser.add_argument("--patch_stride_w", type=int,
                       help="Patch stride in width dimension (default: 60 for Marmousi)")

    # Data parameters
    parser.add_argument("--noise_type", choices=["gaussian", "laplace"],
                       help="Noise type")
    parser.add_argument("--noise_std", type=float, help="Noise level")
    parser.add_argument("--sigma", type=float, help="Initial model smoothing sigma")
    parser.add_argument("--missing_number", type=int, help="Number of missing traces")
    parser.add_argument("--batch_size", type=int, help="Batch size")

    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--random_seed", type=int, help="Random seed")

    args = parser.parse_args()

    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        print("No config file specified. Using default configuration.")
        from red_diffeq import get_config
        config = get_config()

    # Update config from command-line args
    if args.lr is not None:
        config.optimization.lr = args.lr
    if args.ts is not None:
        config.optimization.ts = args.ts
    if args.diffusion_ts is not None:
        config.optimization.diffusion_ts = args.diffusion_ts
    if args.grad_norm is not None:
        config.optimization.grad_norm = args.grad_norm
    if args.grad_smooth is not None:
        config.optimization.grad_smooth = args.grad_smooth
    if args.model_blur is not None:
        config.optimization.model_blur = args.model_blur
    if args.grad_clip is not None:
        config.optimization.grad_clip = args.grad_clip
    if args.use_ilvr is not None:
        config.optimization.use_ilvr = args.use_ilvr
    if args.ilvr_weight is not None:
        config.optimization.ilvr_weight = args.ilvr_weight
    if args.ilvr_down_schedule is not None:
        config.optimization.ilvr_down_schedule = args.ilvr_down_schedule

    # Patch parameters
    if args.use_patches is not None:
        config.optimization.use_patches = args.use_patches
    if args.patch_height is not None and args.patch_width is not None:
        config.optimization.patch_kernel_size = [args.patch_height, args.patch_width]
    if args.patch_stride_h is not None and args.patch_stride_w is not None:
        config.optimization.patch_stride = [args.patch_stride_h, args.patch_stride_w]

    if args.noise_type is not None:
        config.optimization.noise_type = args.noise_type
    if args.noise_std is not None:
        config.optimization.noise_std = args.noise_std
    if args.sigma is not None:
        config.optimization.sigma = args.sigma
    if args.missing_number is not None:
        config.optimization.missing_number = args.missing_number
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.experiment_name is not None:
        config.experiment.name = args.experiment_name
    if args.random_seed is not None:
        config.experiment.random_seed = args.random_seed

    run_experiment(config, method=args.method)


if __name__ == "__main__":
    main()
