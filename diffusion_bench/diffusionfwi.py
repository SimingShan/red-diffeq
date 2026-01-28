"""
DiffusionFWI Implementation - Acoustic FWI Adaptation

This file reproduces the DiffeFWI algorithm from the original repository:
Repository: diffefwi/
Main algorithm reference: diffefwi/src/diffefwi/diffusion.py (lines 573-685)
FWI loop reference: diffefwi/src/diffefwi/fwi.py (lines 99-299)

Key differences from original:
- Uses acoustic FWI instead of elastic FWI (deepwave.scalar vs deepwave.elastic)
- Single velocity parameter instead of (Vp, Vs, rho)
- Otherwise identical diffusion algorithm structure

Algorithm Overview (from diffefwi/src/diffefwi/diffusion.py):
1. Reverse diffusion loop: timestep T-1 down to 0 (line 621)
2. At each timestep:
   a) Denoise current model using p_sample_wf/p_mean_variance (lines 626-633)
   b) Run FWI optimization on denoised model (lines 638-650)
   c) Update current model with optimized result (line 219 in fwi.py)
"""

import torch
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from red_diffeq.core.metrics import MetricsCalculator
from red_diffeq.utils.diffusion_utils import diffusion_pad, diffusion_crop
from red_diffeq.utils.data_trans import add_noise_to_seismic, missing_trace


def split_data_to_patches(data, kernel_size, stride):
    """
    Helper function for patch-based processing
    Used when model size exceeds diffusion model's image_size
    """
    B, C, H, W = data.shape
    patch_h, patch_w = kernel_size
    stride_h, stride_w = stride
    patches = data.unfold(2, patch_h, stride_h).unfold(3, patch_w, stride_w)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, patch_h, patch_w)

    return patches


def merge_patches_to_data(patches, output_size, kernel_size, stride):
    """
    Merges denoised patches back to full-size model
    Handles overlapping regions by averaging
    """
    num_patches, C, patch_h, patch_w = patches.shape
    H, W = output_size
    stride_h, stride_w = stride

    n_patches_h = (H - patch_h) // stride_h + 1
    n_patches_w = (W - patch_w) // stride_w + 1

    device = patches.device
    merged = torch.zeros(1, C, H, W, device=device)
    count = torch.zeros(1, C, H, W, device=device)

    patch_idx = 0
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            h_start = i * stride_h
            w_start = j * stride_w
            h_end = h_start + patch_h
            w_end = w_start + patch_w

            merged[:, :, h_start:h_end, w_start:w_end] += patches[patch_idx]
            count[:, :, h_start:h_end, w_start:w_end] += 1
            patch_idx += 1

    merged = merged / count.clamp(min=1)

    return merged


class DiffusionFWI:
    """
    DiffusionFWI: Diffusion-Guided Full Waveform Inversion

    Reproduces the algorithm from:
    - diffefwi/src/diffefwi/diffusion.py::fwi_sample() (lines 573-685)
    - diffefwi/src/diffefwi/fwi.py::fwi_loop() (lines 99-299)

    Main reference: Algorithm from DiffeFWI paper
    "Diffusion Model-based Full Waveform Inversion"
    """

    def __init__(self, diffusion_model, fwi_forward, ssim_loss):
        self.diffusion_model = diffusion_model
        self.fwi_forward = fwi_forward
        self.ssim_loss = ssim_loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _apply_diffusion_denoising_with_patches(self, current_model, diffusion_step,
                                               kernel_size=None, stride=None, use_patches=False):
        """
        Apply diffusion denoising step

        Original reference: diffefwi/src/diffefwi/diffusion.py
        - p_sample_wf() method (lines 488-510)
        - p_mean_variance() method (lines 414-460)

        Algorithm:
        1. Pass current model through diffusion model at timestep t
        2. Predict noise: pred_noise = model(x_t, t) (line 451)
        3. Predict x_0 from noise (line 454)
        4. Compute posterior mean (line 458-459)

        Returns denoised model (equivalent to img_mean in original)
        """

        batch_size = current_model.shape[0]
        height = current_model.shape[2]
        width = current_model.shape[3]

        if kernel_size is None:
            kernel_size = [height, height]
        if stride is None:
            stride = [1, 1]

        image_size = self.diffusion_model.image_size
        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]

        unpadded_size = image_size - 2

        needs_patching = use_patches and (width != height or width > image_size)

        if not needs_patching:
            # Standard denoising via p_mean_variance
            # Reference: diffefwi/src/diffefwi/diffusion.py, lines 414-460, 626-633
            current_padded = diffusion_pad(current_model)
            t = torch.full((batch_size,), diffusion_step, device=self.device, dtype=torch.long)

            model_mean_padded, _, _, _ = self.diffusion_model.p_mean_variance(
                x=current_padded, t=t, x_self_cond=None, clip_denoised=True
            )

            denoised = diffusion_crop(model_mean_padded).clamp(-1.0, 1.0)

        else:
            # Patch-based denoising for large models (similar to diffefwi lines 425-443)
            patches = split_data_to_patches(current_model, kernel_size, stride)

            denoised_patches = []
            t = torch.full((1,), diffusion_step, device=self.device, dtype=torch.long)

            for i in range(patches.shape[0]):
                patch = patches[i:i+1]

                # Resize to model input size, denoise, then resize back
                patch_resized = F.interpolate(patch, size=(unpadded_size, unpadded_size),
                                             mode='bilinear', align_corners=False)
                patch_resized_padded = diffusion_pad(patch_resized)

                with torch.no_grad():
                    model_mean_padded, _, _, _ = self.diffusion_model.p_mean_variance(
                        x=patch_resized_padded, t=t, x_self_cond=None, clip_denoised=True
                    )

                denoised_patch_resized = diffusion_crop(model_mean_padded).clamp(-1.0, 1.0)

                denoised_patch = F.interpolate(
                    denoised_patch_resized,
                    size=(kernel_size[0], kernel_size[1]),
                    mode='bilinear',
                    align_corners=False
                )

                denoised_patches.append(denoised_patch)

            denoised_patches = torch.cat(denoised_patches, dim=0)
            denoised = merge_patches_to_data(denoised_patches, [height, width], kernel_size, stride)

        return denoised

    def optimize(self, mu: torch.Tensor, mu_true: torch.Tensor, y: torch.Tensor,
                fwi_forward, ts: int = 300, diffusion_ts: int = 500, lr: float = 0.03,
                noise_std: float = 0.0, noise_type: str = 'gaussian',
                missing_number: int = 0, grad_norm: bool = True,
                grad_smooth: float = None, model_blur: bool = False,
                grad_clip: float = 1.0,
                use_patches: bool = False, patch_kernel_size: list = None,
                patch_stride: list = None):
        """
        Main DiffusionFWI optimization loop

        Original reference: diffefwi/src/diffefwi/diffusion.py::fwi_sample()
        Main loop: lines 621-685

        Algorithm structure (from original):
        FOR each diffusion timestep i from (timesteps - init_timestep - 1) down to 0:
            1. DENOISE: img_mean = p_sample_wf(model, img, t) [line 626-633]
            2. FWI: If i != 0, run fwi_loop on denoised model [line 638-650]
            3. UPDATE: current model = optimized result [line 219 in fwi.py]

        Parameters match original fwi_sample() parameters:
        - mu: initial velocity model (vp_img in original)
        - y: observed seismic data (receiver_amplitudes_true)
        - ts: num_epochs in fwi_loop (line 103 in fwi.py)
        - diffusion_ts: number of reverse diffusion steps
        - lr: learning_rate for Adam optimizer (line 104 in fwi.py)
        - grad_norm, grad_smooth, grad_clip: gradient processing options
        """

        if mu.shape[0] != y.shape[0]:
            raise ValueError('Batch size mismatch between velocity and seismic data')
        if fwi_forward is None or not callable(fwi_forward):
            raise ValueError('fwi_forward must be a callable forward modeling function')

        fwi_forward = fwi_forward.to(self.device)
        batch_size = mu.shape[0]
        mu = mu.float().clone().detach().to(self.device)
        mu_true = mu_true.float().to(self.device)

        metrics_calc = MetricsCalculator(self.ssim_loss)

        metrics_history = {
            'total_losses': [], 'obs_losses': [],
            'ssim': [], 'mae': [], 'rmse': []
        }

        y = add_noise_to_seismic(y, noise_std, noise_type=noise_type)
        y, mask = missing_trace(y, missing_number, return_mask=True)
        y = y.to(self.device)
        mask = mask.to(self.device)

        current_model = mu

        # Main diffusion loop - reverse timesteps from T-1 to 0
        # Reference: diffefwi/src/diffefwi/diffusion.py, line 621
        pbar_diffusion = tqdm(range(diffusion_ts - 1, -1, -1),
                            desc='DiffusionFWI', unit='step', position=0)
        for diffusion_step in pbar_diffusion:
            # Step 1: Denoise using p_mean_variance
            # Reference: diffefwi/src/diffefwi/diffusion.py, lines 626-633
            with torch.no_grad():
                denoised = self._apply_diffusion_denoising_with_patches(
                    current_model, diffusion_step,
                    kernel_size=patch_kernel_size,
                    stride=patch_stride,
                    use_patches=use_patches
                )

            # Step 2: Run FWI optimization on denoised model (skip at t=0)
            # Reference: diffefwi/src/diffefwi/diffusion.py, lines 635-650
            #            diffefwi/src/diffefwi/fwi.py::fwi_loop, lines 99-299
            if diffusion_step != 0:
                # Initialize optimizer (diffefwi/src/diffefwi/fwi.py, lines 129-136)
                mu_opt = denoised.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([mu_opt], lr=lr)

                grad_max = None

                # Inner FWI loop (diffefwi/src/diffefwi/fwi.py, lines 147-299)
                for fwi_iter in range(ts):
                    optimizer.zero_grad(set_to_none=True)

                    # Forward modeling (original uses deepwave.elastic at line 188-196)
                    predicted_seismic = fwi_forward(mu_opt)

                    # Data loss (original supports L2/cross-correlation/optimal transport at lines 238-253)
                    # Apply mask to properly handle missing traces (fair comparison with other methods)
                    loss = F.l1_loss(
                        y.float(),
                        predicted_seismic.float(),
                        reduction='none'
                    )
                    # Weighted loss: only compute on observed traces
                    loss = loss * mask
                    num_observed = mask.sum(dim=tuple(range(1, len(mask.shape)))).clamp(min=1.0)
                    loss_obs = loss.sum(dim=tuple(range(1, len(loss.shape)))) / num_observed

                    loss_obs.sum().backward()

                    # Gradient processing tricks
                    with torch.no_grad():
                        # Normalize gradients by max value
                        if grad_norm:
                            if fwi_iter == 0:
                                grad_max = torch.max(torch.abs(mu_opt.grad)).item()
                            if grad_max is not None and grad_max > 0:
                                mu_opt.grad /= grad_max

                        # Smooth gradients for stability
                        if grad_smooth is not None and grad_smooth > 0:
                            grad_np = mu_opt.grad.detach().cpu().numpy()
                            grad_smoothed = gaussian_filter(
                                grad_np,
                                sigma=[0, 0, grad_smooth, grad_smooth]
                            )
                            mu_opt.grad = torch.from_numpy(grad_smoothed).to(mu_opt.device)
                            grad_max = torch.max(torch.abs(mu_opt.grad)).item()

                        # Clip gradients to prevent instability
                        if grad_clip is not None and grad_clip > 0:
                            if grad_max is not None and grad_max > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    [mu_opt],
                                    grad_clip * grad_max
                                )

                    optimizer.step()

                    # Apply optional model smoothing and clamp values
                    # Reference: diffefwi/src/diffefwi/fwi.py, lines 159-179
                    with torch.no_grad():
                        if model_blur:
                            mu_opt.data = torchvision.transforms.functional.gaussian_blur(
                                mu_opt.data,
                                kernel_size=[3, 3],
                                sigma=[0.4, 0.4]
                            )
                        mu_opt.data.clamp_(-1.0, 1.0)

                current_model = mu_opt.detach()
            else:
                # At t=0, skip optimization and use denoised directly
                current_model = denoised.detach()

            # Track metrics at each timestep
            with torch.no_grad():
                predicted_seismic = fwi_forward(current_model)
                # Apply mask to properly handle missing traces
                loss = F.l1_loss(
                    y.float(),
                    predicted_seismic.float(),
                    reduction='none'
                )
                loss = loss * mask
                num_observed = mask.sum(dim=tuple(range(1, len(mask.shape)))).clamp(min=1.0)
                loss_obs = loss.sum(dim=tuple(range(1, len(loss.shape)))) / num_observed

                mae, rmse, ssim = metrics_calc.calculate(current_model, mu_true)

                metrics_history['total_losses'].append(loss_obs.detach().cpu().numpy())
                metrics_history['obs_losses'].append(loss_obs.detach().cpu().numpy())
                metrics_history['ssim'].append(ssim.detach().cpu().numpy())
                metrics_history['mae'].append(mae.detach().cpu().numpy())
                metrics_history['rmse'].append(rmse.detach().cpu().numpy())

            with torch.no_grad():
                _, _, ssim_display = metrics_calc.calculate(current_model, mu_true)
            pbar_diffusion.set_postfix({
                'timestep': diffusion_step,
                'SSIM': ssim_display.mean().item()
            })

        mu = current_model

        final_results_per_model = []
        num_timesteps = len(metrics_history['total_losses'])
        for i in range(batch_size):
            model_results = {
                'total_losses': [metrics_history['total_losses'][t][i] for t in range(num_timesteps)],
                'obs_losses': [metrics_history['obs_losses'][t][i] for t in range(num_timesteps)],
                'ssim': [metrics_history['ssim'][t][i] for t in range(num_timesteps)],
                'mae': [metrics_history['mae'][t][i] for t in range(num_timesteps)],
                'rmse': [metrics_history['rmse'][t][i] for t in range(num_timesteps)]
            }
            final_results_per_model.append(model_results)

        return mu, final_results_per_model
