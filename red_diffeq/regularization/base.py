import torch
from typing import Optional
from red_diffeq.regularization.diffusion import RED_DiffEq
from red_diffeq.regularization.benchmark import total_variation_loss, tikhonov_loss


class RegularizationMethod:

    def __init__(self, regularization_type: Optional[str], diffusion_model=None,
                 use_time_weight: bool = False, share_noise_across_batch: bool = False):
        self.regularization_type = regularization_type
        self.diffusion_model = diffusion_model
        self.use_time_weight = use_time_weight
        self.share_noise_across_batch = share_noise_across_batch
        if regularization_type == 'diffusion':
            self.red_diffeq = RED_DiffEq(diffusion_model, use_time_weight=use_time_weight,
                                          share_noise_across_batch=share_noise_across_batch)

    def get_reg_loss(self, mu: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if self.regularization_type == 'diffusion':
            if self.diffusion_model is None:
                raise ValueError("Diffusion model required for 'diffusion' regularization")

            height = mu.shape[2]
            width = mu.shape[3]

            if width > self.red_diffeq.input_size or height > self.red_diffeq.input_size:
                reg_loss, _ = self.red_diffeq.get_reg_loss_patched(mu) # Patched version might need seed support too later
            else:
                reg_loss, _ = self.red_diffeq.get_reg_loss(mu, seed=seed)

            return reg_loss

        elif self.regularization_type == 'l2':
            return tikhonov_loss(mu)

        elif self.regularization_type == 'tv':
            return total_variation_loss(mu)

        else:
            return torch.zeros(mu.shape[0], device=mu.device, dtype=mu.dtype)
