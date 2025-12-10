import torch
from tqdm.auto import tqdm
from typing import Optional

from red_diffeq.regularization.base import RegularizationMethod
from red_diffeq.utils.data_trans import DataTransformer
from red_diffeq.utils.ssim import SSIM
from red_diffeq.core.metrics import MetricsCalculator
from red_diffeq.core.losses import LossCalculator


class InversionEngine:

    def __init__(self, diffusion_model, data_transformer: DataTransformer,
                 ssim_loss: SSIM, regularization: Optional[str] = None,
                 use_time_weight: bool = False, share_noise_across_batch: bool = False):
        self.diffusion_model = diffusion_model
        self.data_transformer = data_transformer
        self.ssim_loss = ssim_loss
        self.device = diffusion_model.device
        self.regularization_method = RegularizationMethod(
            regularization, diffusion_model, use_time_weight=use_time_weight,
            share_noise_across_batch=share_noise_across_batch
        )

    def optimize(self, mu: torch.Tensor, mu_true: torch.Tensor, y: torch.Tensor,
                fwi_forward, ts: int = 300, lr: float = 0.03, reg_lambda: float = 0.01,
                noise_std: float = 0.0, noise_type: str = 'gaussian',
                missing_number: int = 0, regularization: Optional[str] = None):
        if mu.shape[0] != y.shape[0]:
            raise ValueError('Batch size mismatch between velocity and seismic data')
        if regularization not in ['diffusion', 'l2', 'tv', 'hybrid', None]:
            raise ValueError(f'Unknown regularization: {regularization}')
        if fwi_forward is None or not callable(fwi_forward):
            raise ValueError('fwi_forward must be a callable forward modeling function')

        fwi_forward = fwi_forward.to(self.device)
        if regularization is not None:
            use_time_weight = getattr(
                self.regularization_method, 'use_time_weight', False
            ) if hasattr(self, 'regularization_method') else False
            share_noise_across_batch = getattr(
                self.regularization_method, 'share_noise_across_batch', False
            ) if hasattr(self, 'regularization_method') else False
            self.regularization_method = RegularizationMethod(
                regularization, self.diffusion_model, use_time_weight=use_time_weight,
                share_noise_across_batch=share_noise_across_batch
            )

        batch_size = mu.shape[0]
        mu = mu.float().clone().detach().to(self.device).requires_grad_(True)
        mu_true = mu_true.float().to(self.device)

        optimizer = torch.optim.Adam([mu], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ts, eta_min=0.0
        )

        metrics_calc = MetricsCalculator(self.data_transformer, self.ssim_loss)
        loss_calc = LossCalculator(self.regularization_method)

        metrics_history = {
            'total_losses': [], 'obs_losses': [], 'reg_losses': [],
            'ssim': [], 'mae': [], 'rmse': []
        }

        y = self.data_transformer.add_noise_to_seismic(y, noise_std, noise_type=noise_type)
        y = self.data_transformer.missing_trace(y, missing_number)
        y = y.to(self.device)

        pbar = tqdm(range(ts), desc='Optimizing', unit='step')
        for step in pbar:
            predicted_seismic = fwi_forward(mu)
            loss_obs = loss_calc.observation_loss(predicted_seismic, y)

            reg_loss = loss_calc.regularization_loss(mu)

            total_loss = loss_calc.total_loss(loss_obs, reg_loss, reg_lambda)

            optimizer.zero_grad(set_to_none=True)
            total_loss.sum().backward()
            optimizer.step()

            with torch.no_grad():
                mu.data.clamp_(-1, 1)

            scheduler.step()

            mae, rmse, ssim = metrics_calc.calculate(mu, mu_true)

            metrics_history['total_losses'].append(total_loss.detach().cpu().numpy())
            metrics_history['obs_losses'].append(loss_obs.detach().cpu().numpy())
            metrics_history['reg_losses'].append(reg_loss.detach().cpu().numpy())
            metrics_history['ssim'].append(ssim.detach().cpu().numpy())
            metrics_history['mae'].append(mae.detach().cpu().numpy())
            metrics_history['rmse'].append(rmse.detach().cpu().numpy())

            pbar.set_postfix({
                'total_loss': total_loss.sum().item() / batch_size,
                'obs_loss': loss_obs.sum().item() / batch_size,
                'reg_loss': reg_loss.mean().item(),
                'SSIM': ssim.mean().item()
            })

        final_results_per_model = []
        num_timesteps = len(metrics_history['total_losses'])
        for i in range(batch_size):
            model_results = {
                'total_losses': [metrics_history['total_losses'][t][i] for t in range(num_timesteps)],
                'obs_losses': [metrics_history['obs_losses'][t][i] for t in range(num_timesteps)],
                'reg_losses': [metrics_history['reg_losses'][t][i] for t in range(num_timesteps)],
                'ssim': [metrics_history['ssim'][t][i] for t in range(num_timesteps)],
                'mae': [metrics_history['mae'][t][i] for t in range(num_timesteps)],
                'rmse': [metrics_history['rmse'][t][i] for t in range(num_timesteps)]
            }
            final_results_per_model.append(model_results)

        return mu, final_results_per_model
