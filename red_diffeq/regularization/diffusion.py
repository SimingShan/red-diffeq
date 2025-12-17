import math
import torch
from typing import List, Tuple
from red_diffeq.utils.diffusion_utils import extract, diffusion_pad, diffusion_crop


def calculate_patches(width: int, height: int) -> Tuple[List[Tuple[int, int]], List[int]]:
    m = height
    n = width
    k = math.ceil(n / m)

    if k == 1:
        return [(0, n)], []

    s = (n - m) / (k - 1)

    positions = []
    for i in range(k):
        if i == k - 1:
            positions.append((n - m, n))
        else:
            start = int(i * s)
            positions.append((start, min(start + m, n)))

    overlaps = [positions[i][1] - positions[i + 1][0] for i in range(k - 1)]

    return positions, overlaps


class RED_DiffEq:

    def __init__(self, diffusion_model, use_time_weight: bool = False, share_noise_across_batch: bool = False, sigma_x0: float = 0.0001):
        self.diffusion_model = diffusion_model
        self.use_time_weight = use_time_weight
        self.share_noise_across_batch = share_noise_across_batch
        self.sigma_x0 = sigma_x0

        image_size = getattr(diffusion_model, 'image_size', 72)
        self.input_size = image_size[0] if isinstance(image_size, (tuple, list)) else image_size

    def _apply_time_weight(self, tensor: torch.Tensor, time_tensor: torch.Tensor) -> torch.Tensor:
        if not self.use_time_weight:
            return tensor

        gamma_t = extract(self.diffusion_model.alphas_cumprod, time_tensor, tensor.shape)
        w_t = torch.sqrt((1.0 - gamma_t) / gamma_t)
        return tensor * w_t

    def get_reg_loss(self, mu: torch.Tensor, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute regularization loss using diffusion model.

        IMPORTANT: The input 'mu' is actually x0_pred (already perturbed in inversion.py).
        This ensures the SAME perturbation is used for both forward modeling and regularization,
        matching the original implementation's behavior.
        """
        batch_size = mu.shape[0]

        if seed is not None:
            # Save current RNG state
            rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()

            # Set manual seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        if self.share_noise_across_batch:
            time_tensor = torch.randint(
                0, self.diffusion_model.num_timesteps,
                (1,), device=mu.device, dtype=torch.long
            ).expand(batch_size)

            noise_single = torch.randn(1, mu.shape[1], mu.shape[2], mu.shape[3], device=mu.device)
            noise = noise_single.expand(batch_size, -1, -1, -1)
        else:
            time_tensor = torch.randint(
                0, self.diffusion_model.num_timesteps,
                (batch_size,), device=mu.device, dtype=torch.long
            )

            noise = torch.randn_like(mu)

        if seed is not None:
            # Restore RNG state
            torch.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

        # Use mu directly as x0_pred (already perturbed in inversion.py)
        # DO NOT create a new perturbation here - this was the bug!
        # IMPORTANT: mu is now 72×72 (padded in run_inversion.py), matching original
        # NO need to pad/crop - use directly
        x0_pred = mu

        x_t = self.diffusion_model.q_sample(x0_pred, t=time_tensor, noise=noise)

        # CRITICAL: Do NOT use torch.no_grad() here to match original implementation
        # Original allows gradients to flow through model predictions
        predictions = self.diffusion_model.model_predictions(
            x_t, t=time_tensor, x_self_cond=None,
            clip_x_start=True, rederive_pred_noise=True
        )

        # Do NOT detach to match original: et = pred_noise (without .detach())
        # pred_noise is 72×72, use directly (matching original)
        pred_noise = predictions.pred_noise

        # Use original noise tensor (matching official RED-Diff implementation)
        # Original code: noise_xt = torch.randn_like(mu) then uses it directly
        # Both pred_noise and noise are 72×72
        gradient_field = pred_noise - noise

        # Use x0_pred for gradient computation to match old implementation
        reg_field = gradient_field * x0_pred

        reg_field = self._apply_time_weight(reg_field, time_tensor)

        gradient_per_model = gradient_field.view(batch_size, -1).mean(dim=1)
        reg_per_model = reg_field.view(batch_size, -1).mean(dim=1)

        return reg_per_model, gradient_per_model

    def get_reg_loss_patched(self, mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute regularization loss using patched diffusion model for large inputs.

        IMPORTANT: The input 'mu' is actually x0_pred (already perturbed in inversion.py).
        """
        batch_size = mu.shape[0]
        height = mu.shape[2]
        width = mu.shape[3]

        patch_positions, overlaps = calculate_patches(width, height)

        if self.share_noise_across_batch:
            time_tensor = torch.randint(
                0, self.diffusion_model.num_timesteps,
                (1,), device=mu.device, dtype=torch.long
            ).expand(batch_size)

            noise_single = torch.randn(1, mu.shape[1], height, width, device=mu.device)
            noise = noise_single.expand(batch_size, -1, -1, -1)
        else:
            time_tensor = torch.randint(
                0, self.diffusion_model.num_timesteps,
                (batch_size,), device=mu.device, dtype=torch.long
            )

            noise = torch.randn_like(mu)

        # Use mu directly as x0_pred (already perturbed in inversion.py)
        # DO NOT create a new perturbation here - this was the bug!
        # NOTE: This patched version is for inputs > 72×72
        # For standard 72×72 inputs, get_reg_loss() is used instead
        x0_pred = mu

        gradient_field = torch.zeros_like(mu)
        weight_map = torch.zeros_like(mu)

        for patch_idx, (start_x, end_x) in enumerate(patch_positions):
            x0_pred_patch = x0_pred[:, :, :, start_x:end_x]
            noise_patch = noise[:, :, :, start_x:end_x]

            # For patches, we may still need padding if patch size < 72×72
            # This code path is only used for very large models
            x0_pred_patch_padded = diffusion_pad(x0_pred_patch)
            noise_patch_padded = diffusion_pad(noise_patch)

            x_t = self.diffusion_model.q_sample(
                x0_pred_patch_padded, t=time_tensor, noise=noise_patch_padded
            )

            # CRITICAL: Do NOT use torch.no_grad() here to match original implementation
            predictions = self.diffusion_model.model_predictions(
                x_t, t=time_tensor, x_self_cond=None,
                clip_x_start=True, rederive_pred_noise=True
            )

            # Do NOT detach to match original
            pred_noise_patch = diffusion_crop(predictions.pred_noise)
            noise_patch_cropped = diffusion_crop(noise_patch_padded)

            # Use original noise tensor (matching official RED-Diff implementation)
            gradient_patch = pred_noise_patch - noise_patch_cropped

            patch_width = end_x - start_x
            weight = torch.ones(patch_width, device=mu.device)

            if patch_idx > 0:
                weight[:overlaps[patch_idx - 1]] = 0.5

            if patch_idx < len(patch_positions) - 1:
                weight[-overlaps[patch_idx]:] = 0.5

            weight = weight.view(1, 1, 1, -1)

            gradient_field[:, :, :, start_x:end_x] += gradient_patch * weight
            weight_map[:, :, :, start_x:end_x] += weight

        gradient_field = gradient_field / weight_map.clamp(min=1e-8)

        # Use x0_pred (mu + pre-noise) for gradient computation to match old implementation
        reg_field = gradient_field * x0_pred

        reg_field = self._apply_time_weight(reg_field, time_tensor)

        gradient_per_model = gradient_field.view(batch_size, -1).mean(dim=1)
        reg_per_model = reg_field.view(batch_size, -1).mean(dim=1)

        return reg_per_model, gradient_per_model


class RED_DiffEq_POST_PROCESS:

    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model

    def generate_time_tensor(self, timesteps: int, mu: torch.Tensor) -> torch.Tensor:
        return torch.full((mu.shape[0],), timesteps, device=mu.device, dtype=torch.long)

    def generate_noisy_sample(self, mu: torch.Tensor, time_tensor: torch.Tensor):
        x0_pred = mu
        noise = torch.randn_like(mu)
        mu_norm = self.diffusion_model.normalize(mu)
        x_t_norm = self.diffusion_model.q_sample(mu_norm, t=time_tensor, noise=noise)
        x_t = self.diffusion_model.unnormalize(x_t_norm)
        return (x_t, noise, x0_pred)

    def diffusion_denoise(self, mu: torch.Tensor, timesteps: int):
        max_timesteps = self.diffusion_model.num_timesteps
        if timesteps > max_timesteps:
            raise ValueError(
                f"timesteps ({timesteps}) exceeds model's num_timesteps ({max_timesteps})"
            )

        mu_01 = (mu + 1) / 2

        time_tensor = self.generate_time_tensor(timesteps, mu_01)
        x_t, _, _ = self.generate_noisy_sample(mu_01, time_tensor)

        x_start = None
        for t in reversed(range(timesteps)):
            self_cond = x_start if self.diffusion_model.self_condition else None
            x_t_norm = self.diffusion_model.normalize(x_t)
            x_t_norm, x_start_norm = self.diffusion_model.p_sample_deterministic(
                x_t_norm, t=t, x_self_cond=self_cond
            )
            x_t = self.diffusion_model.unnormalize(x_t_norm)
            x_start = (
                self.diffusion_model.unnormalize(x_start_norm)
                if x_start_norm is not None
                else None
            )

        return x_t