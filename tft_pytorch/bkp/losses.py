"""
Loss functions for Temporal Fusion Transformer models.

Includes quantile, MSE, MAE, Huber, Tweedie, combined, and adaptive losses,
all with support for optional padding masks and sample weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class QuantileLoss(nn.Module):
    """
    Quantile (pinball) loss for multi-quantile predictions from TFT models.

    Expects:
        y_pred: [batch_size, prediction_steps, num_quantiles]
        y_true: [batch_size, prediction_steps] or [batch_size, prediction_steps, 1]
    """

    def __init__(self, quantiles: Optional[List[float]] = None, reduction: str = 'mean'):
        """
        Args:
            quantiles: List of quantile values (e.g., [0.1, 0.5, 0.9]).
                       Defaults to [0.1, 0.25, 0.5, 0.75, 0.9].
            reduction: 'none' | 'mean' | 'sum' | 'batch_mean'
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantiles = quantiles
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: Predictions  [batch, time, num_quantiles]
            y_true: Targets      [batch, time] or [batch, time, 1]
            mask:   Binary mask  [batch, time]  (1=valid, 0=ignore)
            sample_weight: Per-sample weights [batch] or [batch, time]

        Returns:
            Scalar loss (or tensor if reduction='none'/'batch_mean').
        """
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1)  # [batch, time, 1]

        assert y_pred.size(-1) == len(self.quantiles), (
            f"Expected {len(self.quantiles)} quantiles, got {y_pred.size(-1)}"
        )

        losses = []
        for i, q in enumerate(self.quantiles):
            y_pred_q = y_pred[..., i:i+1]
            errors = y_true - y_pred_q
            loss_q = torch.where(errors >= 0, q * errors, (q - 1) * errors)
            losses.append(loss_q)

        # [batch, time, num_quantiles]
        losses = torch.cat(losses, dim=-1)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            losses = losses * mask

        if sample_weight is not None:
            if sample_weight.dim() == 1:
                sample_weight = sample_weight.view(-1, 1, 1)
            elif sample_weight.dim() == 2:
                sample_weight = sample_weight.unsqueeze(-1)
            losses = losses * sample_weight

        return self._reduce(losses, mask)

    def _reduce(self, losses, mask=None):
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            if mask is not None:
                return losses.sum() / (mask.sum() * len(self.quantiles) + 1e-8)
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'batch_mean':
            return losses.mean(dim=[1, 2])
        raise ValueError(f"Unknown reduction: {self.reduction}")


class MSELoss(nn.Module):
    """Mean Squared Error with mask and sample-weight support."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        errors = (y_pred - y_true) ** 2
        errors = _apply_mask_and_weight(errors, mask, sample_weight)
        return _reduce(errors, self.reduction, mask)


class RMSELoss(MSELoss):
    """Root Mean Squared Error (wraps MSELoss with sqrt)."""

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return torch.sqrt(super().forward(*args, **kwargs) + 1e-8)


class MAELoss(nn.Module):
    """Mean Absolute Error with mask and sample-weight support."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        errors = torch.abs(y_pred - y_true)
        errors = _apply_mask_and_weight(errors, mask, sample_weight)
        return _reduce(errors, self.reduction, mask)


class HuberLoss(nn.Module):
    """Huber (smooth L1) loss with mask and sample-weight support."""

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self._huber = nn.HuberLoss(reduction='none', delta=delta)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_shape = y_pred.shape
        if y_pred.dim() > 2:
            bs = y_pred.size(0)
            y_pred = y_pred.reshape(bs, -1)
            y_true = y_true.reshape(bs, -1)

        loss = self._huber(y_pred, y_true)

        if len(original_shape) > 2:
            loss = loss.reshape(original_shape)

        loss = _apply_mask_and_weight(loss, mask, sample_weight)
        return _reduce(loss, self.reduction, mask)


class TweedieLoss(nn.Module):
    """
    Tweedie loss for non-negative continuous targets (e.g., sales, counts).

    The trainer will inverse-transform predictions and targets before computing
    this loss, so raw (unscaled) values are expected at forward time.

    Args:
        p: Power parameter in (1, 2). Typical choices:
           1.5 → compound Poisson-Gamma, 1.0 → Poisson, 2.0 → Gamma
    """

    def __init__(self, p: float = 1.5, reduction: str = 'mean'):
        super().__init__()
        assert 1 < p < 2, "p must be in the open interval (1, 2)"
        self.p = p
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_pred = F.softplus(y_pred) + 1e-8
        loss = (
            -y_true * torch.pow(y_pred, 1 - self.p) / (1 - self.p)
            + torch.pow(y_pred, 2 - self.p) / (2 - self.p)
        )
        loss = _apply_mask_and_weight(loss, mask, sample_weight)
        return _reduce(loss, self.reduction, mask)


class CombinedLoss(nn.Module):
    """
    Linearly combine multiple loss functions with fixed or learnable weights.

    Example::

        criterion = CombinedLoss(
            losses=[QuantileLoss([0.1, 0.5, 0.9]), MAELoss()],
            weights=[0.7, 0.3],
        )
    """

    def __init__(
        self,
        losses: List[nn.Module],
        weights: Optional[List[float]] = None,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)

        if weights is None:
            weights = [1.0] * len(losses)

        if learnable_weights:
            self.log_weights = nn.Parameter(torch.tensor(weights).log())
        else:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
            self.log_weights = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.log_weights is not None:
            weights = F.softmax(self.log_weights, dim=0) * len(self.losses)
        else:
            weights = self.weights

        total = sum(w * fn(y_pred, y_true, **kwargs) for w, fn in zip(weights, self.losses))
        return total


class AdaptiveLoss(nn.Module):
    """
    Automatically balance multiple loss components using exponential moving
    averages of their magnitudes.  During the warmup phase equal weights are used.

    Example::

        criterion = AdaptiveLoss(
            losses=[QuantileLoss([0.1, 0.5, 0.9]), MAELoss()],
            ema_decay=0.99,
            warmup_steps=200,
        )
    """

    def __init__(
        self,
        losses: List[nn.Module],
        ema_decay: float = 0.99,
        warmup_steps: int = 100,
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps

        n = len(losses)
        self.register_buffer('loss_emas', torch.ones(n))
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('weights', torch.ones(n) / n)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:
        individual = [fn(y_pred, y_true, **kwargs) for fn in self.losses]
        losses_tensor = torch.stack(individual)

        with torch.no_grad():
            self.step_count += 1
            if self.step_count > self.warmup_steps:
                self.loss_emas = (
                    self.ema_decay * self.loss_emas
                    + (1 - self.ema_decay) * losses_tensor.detach()
                )
                inv = 1.0 / (self.loss_emas + 1e-8)
                self.weights = inv / inv.sum()
            else:
                self.weights = torch.ones_like(self.weights) / len(self.losses)

        return (self.weights * losses_tensor).sum()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_mask_and_weight(errors, mask, sample_weight):
    if mask is not None:
        while mask.dim() < errors.dim():
            mask = mask.unsqueeze(-1)
        errors = errors * mask
    if sample_weight is not None:
        while sample_weight.dim() < errors.dim():
            sample_weight = sample_weight.unsqueeze(-1)
        errors = errors * sample_weight
    return errors


def _reduce(errors, reduction, mask=None):
    if reduction == 'none':
        return errors
    elif reduction == 'mean':
        if mask is not None:
            while mask.dim() < errors.dim():
                mask = mask.unsqueeze(-1)
            return errors.sum() / (mask.sum() + 1e-8)
        return errors.mean()
    elif reduction == 'sum':
        return errors.sum()
    elif reduction == 'batch_mean':
        return errors.mean(dim=list(range(1, errors.dim())))
    raise ValueError(f"Unknown reduction: {reduction}")
