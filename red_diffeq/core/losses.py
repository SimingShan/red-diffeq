import torch
import torch.nn as nn
from typing import Optional

from red_diffeq.regularization.base import RegularizationMethod


class LossCalculator:
    """Calculate observation and regularization losses for FWI optimization."""

    def __init__(self, regularization_method: RegularizationMethod):
        self.regularization_method = regularization_method

    def observation_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute observation loss (data fidelity term).

        Args:
            predicted: Predicted seismic data
            target: Observed seismic data

        Returns:
            Per-model observation loss (batch_size,)
        """
        loss = nn.L1Loss(reduction='none')(target.float(), predicted.float())
        # Average over spatial/temporal dimensions, keep batch dimension
        loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
        return loss

    def regularization_loss(self, mu: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """Compute regularization loss.

        Args:
            mu: Velocity model (batch, 1, height, width)
            seed: Optional random seed for deterministic noise sampling

        Returns:
            Per-model regularization loss (batch_size,)
        """
        return self.regularization_method.get_reg_loss(mu, seed=seed)

    def total_loss(self, obs_loss: torch.Tensor, reg_loss: torch.Tensor, reg_lambda: float) -> torch.Tensor:
        """Compute total loss = observation + λ * regularization.

        Args:
            obs_loss: Observation loss (batch_size,)
            reg_loss: Regularization loss (batch_size,)
            reg_lambda: Regularization weight

        Returns:
            Total loss (batch_size,)
        """
        return obs_loss + reg_lambda * reg_loss
