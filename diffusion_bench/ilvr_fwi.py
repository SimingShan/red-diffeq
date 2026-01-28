"""
ILVR-FWI Implementation - Acoustic FWI Adaptation with ILVR Conditioning

This file reproduces the ILBrEFWI (ILVR-based EFWI) algorithm from the original repository:
Repository: ilvrefwi/
Main algorithm reference: ilvrefwi/src/ilvrefwi/diffusion.py (lines 573-730)
ILVR conditioning: ilvrefwi/src/ilvrefwi/diffusion.py (lines 670-700)
Resizer: ilvrefwi/src/ilvrefwi/resizer.py

Key differences from original:
- Uses acoustic FWI instead of elastic FWI (deepwave.scalar vs deepwave.elastic)
- Single velocity parameter instead of (Vp, Vs, rho)
- Otherwise identical ILVR diffusion algorithm structure

ILVR (Iterative Latent Variable Refinement) Overview:
Paper: "ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models"
        Choi et al., ICCV 2021

Algorithm (from ilvrefwi/src/ilvrefwi/diffusion.py):
1. Reverse diffusion loop: timestep T-1 down to 0 (line 621)
2. At each timestep:
   a) Denoise current model using p_sample_wf/p_mean_variance (lines 626-633)
   b) APPLY ILVR CONDITIONING to preserve low-frequency content (lines 671-700)
      Formula: denoised' = denoised - α*LF(denoised) + α*LF(noised_current)
      where LF is low-frequency extraction via downsampling/upsampling
   c) Run FWI optimization on conditioned model (lines 638-650)
   d) Update current model with optimized result
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from diffusion_bench.diffusionfwi import DiffusionFWI
from diffusion_bench.resizer import Resizer
from red_diffeq.core.metrics import MetricsCalculator
from red_diffeq.utils.diffusion_utils import diffusion_pad, diffusion_crop
from red_diffeq.utils.data_trans import add_noise_to_seismic, missing_trace


class ILVR_FWI(DiffusionFWI):
    """
    ILVR-FWI: ILVR-Conditioned Diffusion-Guided Full Waveform Inversion

    Reproduces the algorithm from:
    - ilvrefwi/src/ilvrefwi/diffusion.py::fwi_sample() (lines 573-730)
    - ilvrefwi/src/ilvrefwi/diffusion.py::ILVR conditioning (lines 670-700)

    Main reference: ILBrEFWI paper
    "ILVR-Based Full Waveform Inversion"

    Inherits from DiffusionFWI and adds ILVR conditioning step
    """

    def __init__(self, diffusion_model, fwi_forward, ssim_loss):
        super().__init__(diffusion_model, fwi_forward, ssim_loss)

    def optimize(self, mu: torch.Tensor, mu_true: torch.Tensor, y: torch.Tensor,
                fwi_forward, ts: int = 300, diffusion_ts: int = 500, lr: float = 0.03,
                noise_std: float = 0.0, noise_type: str = 'gaussian',
                missing_number: int = 0, grad_norm: bool = True,
                grad_smooth: float = None, model_blur: bool = False,
                grad_clip: float = 1.0,
                use_ilvr: bool = True, ilvr_weight: float = 0.05,
                ilvr_down_schedule: str = 'linear',
                use_patches: bool = False, patch_kernel_size: list = None,
                patch_stride: list = None):
        """
        Main ILVR-FWI optimization loop

        Original reference: ilvrefwi/src/ilvrefwi/diffusion.py::fwi_sample()
        Main loop: lines 621-730
        ILVR conditioning: lines 670-700

        Key addition vs DiffusionFWI:
        - ILVR conditioning applied after denoising (lines 671-700)
        - Downsampling schedule controls which frequencies are preserved

        Parameters:
        - use_ilvr: Enable ILVR conditioning (default: True)
        - ilvr_weight: Weight for low-frequency conditioning (α in formula, default: 0.05)
        - ilvr_down_schedule: Schedule for downsampling factor
            'linear': linear interpolation from 16 to 2 (original default)
            'stepwise': [32, 16, 8, 4] repeated (original alternative)
        """

        self.use_ilvr = use_ilvr
        self.ilvr_weight = ilvr_weight

        # Setup ILVR downsampling schedule (controls which frequencies to preserve)
        # Reference: ilvrefwi paper and implementation
        if ilvr_down_schedule == 'linear':
            # Linear: gradually reduce downsampling from 16x to 2x
            self.down_n = np.linspace(16, 2, diffusion_ts).astype(int)
        elif ilvr_down_schedule == 'stepwise':
            # Stepwise: constant factors in blocks [32, 16, 8, 4]
            Ns = [32, 16, 8, 4]
            self.down_n = np.repeat(Ns, diffusion_ts // len(Ns))
            if len(self.down_n) < diffusion_ts:
                self.down_n = np.pad(self.down_n, (0, diffusion_ts - len(self.down_n)),
                                    constant_values=Ns[-1])
        else:
            raise ValueError(f"Unknown ilvr_down_schedule: {ilvr_down_schedule}")

        return self._optimize_with_ilvr(
            mu, mu_true, y, fwi_forward, ts, diffusion_ts, lr,
            noise_std, noise_type, missing_number,
            grad_norm, grad_smooth, model_blur, grad_clip,
            use_patches, patch_kernel_size, patch_stride
        )

    def _optimize_with_ilvr(self, mu, mu_true, y, fwi_forward, ts, diffusion_ts,
                           lr, noise_std, noise_type, missing_number,
                           grad_norm, grad_smooth, model_blur, grad_clip,
                           use_patches, patch_kernel_size, patch_stride):
        if mu.shape[0] != y.shape[0]:
            raise ValueError('Batch size mismatch')

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

        # Main diffusion loop (same as DiffeFWI but with ILVR conditioning)
        # Reference: ilvrefwi/src/ilvrefwi/diffusion.py, line 621
        pbar_diffusion = tqdm(range(diffusion_ts - 1, -1, -1),
                            desc='ILVR-FWI' if self.use_ilvr else 'DiffusionFWI',
                            unit='step', position=0)

        for diffusion_step in pbar_diffusion:
            # Step 1: Denoise (same as DiffeFWI)
            # Reference: ilvrefwi/src/ilvrefwi/diffusion.py, lines 626-633
            with torch.no_grad():
                denoised = self._apply_diffusion_denoising_with_patches(
                    current_model, diffusion_step,
                    kernel_size=patch_kernel_size,
                    stride=patch_stride,
                    use_patches=use_patches
                )

            # Step 2: ILVR conditioning - KEY difference from DiffeFWI
            # Preserves low-frequency structure from current model
            # Reference: ilvrefwi/src/ilvrefwi/diffusion.py, lines 670-700
            if self.use_ilvr and diffusion_step > 0:
                denoised = self._apply_ilvr(denoised, current_model, diffusion_step)

            # Step 3: FWI optimization (operates on ILVR-conditioned model)
            # Reference: ilvrefwi/src/ilvrefwi/diffusion.py, lines 710-728
            if diffusion_step != 0:
                mu_opt = denoised.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([mu_opt], lr=lr)
                grad_max = None

                for fwi_iter in range(ts):
                    optimizer.zero_grad(set_to_none=True)

                    predicted_seismic = fwi_forward(mu_opt)

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

                    with torch.no_grad():
                        if grad_norm:
                            if fwi_iter == 0:
                                grad_max = torch.max(torch.abs(mu_opt.grad)).item()
                            if grad_max is not None and grad_max > 0:
                                mu_opt.grad /= grad_max

                        if grad_smooth is not None and grad_smooth > 0:
                            from scipy.ndimage import gaussian_filter
                            grad_np = mu_opt.grad.detach().cpu().numpy()
                            grad_smoothed = gaussian_filter(
                                grad_np,
                                sigma=[0, 0, grad_smooth, grad_smooth]
                            )
                            mu_opt.grad = torch.from_numpy(grad_smoothed).to(mu_opt.device)
                            grad_max = torch.max(torch.abs(mu_opt.grad)).item()

                        if grad_clip is not None and grad_clip > 0:
                            if grad_max is not None and grad_max > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    [mu_opt],
                                    grad_clip * grad_max
                                )

                    optimizer.step()

                    with torch.no_grad():
                        if model_blur:
                            import torchvision
                            mu_opt.data = torchvision.transforms.functional.gaussian_blur(
                                mu_opt.data,
                                kernel_size=[3, 3],
                                sigma=[0.4, 0.4]
                            )
                        mu_opt.data.clamp_(-1.0, 1.0)

                current_model = mu_opt.detach()
            else:
                current_model = denoised.detach()

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

    def _apply_ilvr(self, denoised, current_model, t):
        """
        Apply ILVR conditioning to preserve low-frequency structure

        Reference: ilvrefwi/src/ilvrefwi/diffusion.py, lines 670-700
        Formula: denoised' = denoised - α*LF(denoised) + α*LF(q_sample(current, t))

        This keeps structural information from the current model while letting
        the diffusion model generate high-frequency details.
        """

        with torch.no_grad():
            down_factor = self.down_n[t]
            orig_h, orig_w = denoised.shape[2], denoised.shape[3]

            # Setup resizers for downsampling/upsampling (acts as low-pass filter)
            # Reference: ilvrefwi/src/ilvrefwi/resizer.py
            down = Resizer(denoised.shape, 1 / down_factor).to(self.device)
            up = Resizer(
                (denoised.shape[0], denoised.shape[1],
                 int(denoised.shape[2] / down_factor),
                 int(denoised.shape[3] / down_factor)),
                down_factor
            ).to(self.device)

            # Add noise to current model to match diffusion timestep t
            # Reference: ilvrefwi/src/ilvrefwi/diffusion.py, line 699
            t_tensor = torch.tensor([t], device=self.device)
            noised_current = self.diffusion_model.q_sample(
                current_model,
                t_tensor,
                torch.randn_like(current_model)
            )

            # Extract low frequencies via down-up (low-pass filtering)
            low_freq_denoised = up(down(denoised))
            low_freq_current = up(down(noised_current))

            # Handle size mismatches
            if low_freq_denoised.shape[2:] != (orig_h, orig_w):
                low_freq_denoised = F.interpolate(
                    low_freq_denoised, size=(orig_h, orig_w),
                    mode='bilinear', align_corners=False
                )
            if low_freq_current.shape[2:] != (orig_h, orig_w):
                low_freq_current = F.interpolate(
                    low_freq_current, size=(orig_h, orig_w),
                    mode='bilinear', align_corners=False
                )

            # ILVR formula: replace low-freq of denoised with low-freq of current
            # Reference: ilvrefwi/src/ilvrefwi/diffusion.py, lines 696-700
            conditioned = (
                denoised
                - self.ilvr_weight * low_freq_denoised
                + self.ilvr_weight * low_freq_current
            )

            return conditioned.clamp(-1.0, 1.0)
