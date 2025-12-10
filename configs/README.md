# RED-DiffEq Configuration Files

This directory contains YAML configuration files for running RED-DiffEq experiments.

## Quick Start

Run an experiment with a configuration file:

```bash
red-diffeq --config configs/default.yaml
```

Or using the Python script directly:

```bash
python scripts/run_inversion.py --config configs/default.yaml
```

## Available Configurations

### `default.yaml`
Standard configuration with diffusion regularization. Good starting point for most experiments.

### Example Configurations (`examples/`)

- **`quick_test.yaml`**: Fast configuration for testing (50 iterations, batch size 5)
- **`no_regularization.yaml`**: Pure gradient descent baseline (no regularization)
- **`tv_regularization.yaml`**: Total Variation regularization
- **`noisy_incomplete_data.yaml`**: Robust configuration for noisy/incomplete data

## Configuration Structure

Configuration files are organized into six sections:

### 1. PDE Parameters (`pde`)
Physical parameters for the wave equation solver:
```yaml
pde:
  n_grid: 70        # Grid size (70x70)
  nt: 1000          # Time steps
  dx: 10.0          # Spatial step (meters)
  dt: 0.001         # Time step (seconds)
  f: 15.0           # Source frequency (Hz)
  # ... more parameters
```

### 2. Model Architecture (`model`)
U-Net architecture for diffusion model:
```yaml
model:
  dim: 64
  dim_mults: [1, 2, 4, 8]
  flash_attn: false
  channels: 1
```

### 3. Diffusion Parameters (`diffusion`)
Gaussian diffusion model configuration:
```yaml
diffusion:
  image_size: 72
  timesteps: 1000
  sampling_timesteps: 250
  objective: "pred_noise"
  model_path: "pretrained_models/model.pt"
```

### 4. Optimization (`optimization`)
Inversion optimization parameters:
```yaml
optimization:
  lr: 0.03                        # Learning rate
  ts: 300                         # Optimization steps
  regularization: "diffusion"     # Options: diffusion, tv, l2, null
  reg_lambda: 0.01                # Regularization weight
  loss_type: "l1"                 # Options: l1, l2, Huber
  lr_scheduler: "cosine_annealing"
  sigma: 10.0                     # Initial model blur
  initial_type: "smoothed"        # Options: smoothed, homogeneous, linear
  noise_std: 0.0                  # Observation noise
  missing_number: 0               # Missing traces
```

### 5. Data (`data`)
Data loading configuration:
```yaml
data:
  seismic_data_dir: "dataset/OpenFWI/Seismic_Data/"
  velocity_data_dir: "dataset/OpenFWI/Velocity_Data/"
  batch_size: 25
  data_pattern: "*.npy"
  use_mmap: true
```

### 6. Experiment (`experiment`)
Experiment tracking and results:
```yaml
experiment:
  name: "my_experiment"
  results_dir: "experiment/"
  save_intermediate: false
  log_interval: 10
  save_metrics: true
  random_seed: null              # Set to integer for reproducibility
```

## Creating Custom Configurations

1. Copy an existing configuration:
   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   ```

2. Edit parameters as needed

3. Run with your custom config:
   ```bash
   red-diffeq --config configs/my_experiment.yaml
   ```

## Command-Line Overrides

Override specific parameters from command line:

```bash
red-diffeq --config configs/default.yaml \
    --lr 0.05 \
    --ts 500 \
    --regularization tv \
    --batch_size 50
```

Command-line arguments take precedence over config file values.

## Validation

All configurations are automatically validated:
- Type checking (float, int, string)
- Range validation (positive values, valid choices)
- Consistency checks (e.g., sampling_timesteps ≤ timesteps)
- Path existence warnings

If validation fails, you'll see a helpful error message explaining what needs to be fixed.

## Tips

- **Start with `quick_test.yaml`** for rapid iteration
- **Use `random_seed`** for reproducible experiments
- **Increase `reg_lambda`** for stronger regularization
- **Try `Huber` loss** for noisy data (robust to outliers)
- **Set `save_intermediate: true`** to save checkpoints (slower, larger storage)

## Examples

### High-Resolution Experiment
```yaml
optimization:
  ts: 1000                        # More iterations
  lr: 0.01                        # Lower learning rate
  lr_scheduler: "cosine_annealing"
```

### Fast Baseline Comparison
```yaml
optimization:
  ts: 100
  regularization: null            # No regularization
```

### Robust to Noise
```yaml
optimization:
  loss_type: "Huber"              # Robust loss
  reg_lambda: 0.02                # Stronger regularization
  noise_std: 0.1                  # 10% noise
```

## Need Help?

- See full documentation: `docs/`
- Check example notebooks: `examples/`
- Report issues: GitHub Issues
