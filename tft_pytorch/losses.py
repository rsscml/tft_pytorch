#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class QuantileLoss(nn.Module):
    """
    Quantile loss for multi-quantile predictions from TFT models.
    
    Expects predictions with shape [batch_size, prediction_steps, num_quantiles]
    and targets with shape [batch_size, prediction_steps] or [batch_size, prediction_steps, 1]
    """
    
    def __init__(self, quantiles: Optional[List[float]] = None, reduction: str = 'mean'):
        """
        Args:
            quantiles: List of quantile values (e.g., [0.1, 0.5, 0.9])
            reduction: 'none', 'mean', 'sum', or 'batch_mean'
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantiles = quantiles
        self.reduction = reduction

    def forward(self, 
                y_pred: torch.Tensor, 
                y_true: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate quantile loss.
        
        Args:
            y_pred: Predictions [batch_size, prediction_steps, num_quantiles]
            y_true: Targets [batch_size, prediction_steps] or [batch_size, prediction_steps, 1]
            mask: Optional mask [batch_size, prediction_steps] where 1=valid, 0=ignore
            sample_weight: Optional weights [batch_size] or [batch_size, prediction_steps]
        
        Returns:
            Loss value (scalar if reduction='mean'/'sum', tensor otherwise)
        """
        # Ensure y_true has same dims as y_pred for broadcasting
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1)  # [batch, time, 1]
        
        # Ensure we have the right number of quantiles
        assert y_pred.size(-1) == len(self.quantiles), \
            f"Expected {len(self.quantiles)} quantiles, got {y_pred.size(-1)}"
        
        losses = []
        for i, q in enumerate(self.quantiles):
            # Extract predictions for this quantile
            y_pred_q = y_pred[..., i:i+1]  # [batch, time, 1]
            
            # Calculate errors
            errors = y_true - y_pred_q
            
            # Quantile loss formula
            loss_q = torch.where(
                errors >= 0,
                q * errors,
                (q - 1) * errors
            )
            losses.append(loss_q)
        
        # Stack losses: [batch, time, num_quantiles]
        losses = torch.cat(losses, dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match losses shape
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [batch, time, 1]
            losses = losses * mask
        
        # Apply sample weights if provided
        if sample_weight is not None:
            if sample_weight.dim() == 1:
                sample_weight = sample_weight.view(-1, 1, 1)
            elif sample_weight.dim() == 2:
                sample_weight = sample_weight.unsqueeze(-1)
            losses = losses * sample_weight
        
        # Reduce according to specified method
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            if mask is not None:
                # Weighted mean accounting for mask
                return losses.sum() / (mask.sum() * len(self.quantiles) + 1e-8)
            else:
                return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'batch_mean':
            # Mean over time and quantiles, keep batch dimension
            return losses.mean(dim=[1, 2])
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class MSELoss(nn.Module):
    """
    Mean Squared Error loss with proper shape handling for TFT outputs.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, 
                y_pred: torch.Tensor, 
                y_true: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate MSE loss.
        
        Args:
            y_pred: Predictions [batch, time, features] or [batch, features]
            y_true: Targets (same shape as y_pred or broadcastable)
            mask: Optional mask where 1=valid, 0=ignore
            sample_weight: Optional weights
        """
        # Calculate squared errors
        squared_errors = (y_pred - y_true) ** 2
        
        # Apply mask
        if mask is not None:
            if mask.dim() < squared_errors.dim():
                # Expand mask to match squared_errors
                for _ in range(squared_errors.dim() - mask.dim()):
                    mask = mask.unsqueeze(-1)
            squared_errors = squared_errors * mask
        
        # Apply sample weights
        if sample_weight is not None:
            if sample_weight.dim() < squared_errors.dim():
                for _ in range(squared_errors.dim() - sample_weight.dim()):
                    sample_weight = sample_weight.unsqueeze(-1)
            squared_errors = squared_errors * sample_weight
        
        # Reduce
        if self.reduction == 'none':
            return squared_errors
        elif self.reduction == 'mean':
            if mask is not None:
                return squared_errors.sum() / (mask.sum() + 1e-8)
            else:
                return squared_errors.mean()
        elif self.reduction == 'sum':
            return squared_errors.sum()
        elif self.reduction == 'batch_mean':
            # Keep batch dimension
            return squared_errors.mean(dim=list(range(1, squared_errors.dim())))
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class RMSELoss(MSELoss):
    """
    Root Mean Squared Error loss (wrapper around MSE with sqrt).
    Note: For training, MSE is usually preferred as RMSE can have gradient issues near zero.
    """
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        mse = super().forward(*args, **kwargs)
        return torch.sqrt(mse + 1e-8)


class MAELoss(nn.Module):
    """
    Mean Absolute Error loss with proper shape handling.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, 
                y_pred: torch.Tensor, 
                y_true: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate MAE loss."""
        # Calculate absolute errors
        abs_errors = torch.abs(y_pred - y_true)
        
        # Apply mask
        if mask is not None:
            if mask.dim() < abs_errors.dim():
                for _ in range(abs_errors.dim() - mask.dim()):
                    mask = mask.unsqueeze(-1)
            abs_errors = abs_errors * mask
        
        # Apply sample weights
        if sample_weight is not None:
            if sample_weight.dim() < abs_errors.dim():
                for _ in range(abs_errors.dim() - sample_weight.dim()):
                    sample_weight = sample_weight.unsqueeze(-1)
            abs_errors = abs_errors * sample_weight
        
        # Reduce
        if self.reduction == 'none':
            return abs_errors
        elif self.reduction == 'mean':
            if mask is not None:
                return abs_errors.sum() / (mask.sum() + 1e-8)
            else:
                return abs_errors.mean()
        elif self.reduction == 'sum':
            return abs_errors.sum()
        elif self.reduction == 'batch_mean':
            return abs_errors.mean(dim=list(range(1, abs_errors.dim())))
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class HuberLoss(nn.Module):
    """
    Huber loss with proper shape handling for TFT outputs.
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
    
    def forward(self, 
                y_pred: torch.Tensor, 
                y_true: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate Huber loss."""
        # Flatten for HuberLoss computation if needed
        original_shape = y_pred.shape
        if y_pred.dim() > 2:
            batch_size = y_pred.size(0)
            y_pred = y_pred.reshape(batch_size, -1)
            y_true = y_true.reshape(batch_size, -1)
        
        # Calculate Huber loss
        loss = self.huber(y_pred, y_true)
        
        # Reshape back
        if len(original_shape) > 2:
            loss = loss.reshape(original_shape)
        
        # Apply mask
        if mask is not None:
            if mask.dim() < loss.dim():
                for _ in range(loss.dim() - mask.dim()):
                    mask = mask.unsqueeze(-1)
            loss = loss * mask
        
        # Apply sample weights
        if sample_weight is not None:
            if sample_weight.dim() < loss.dim():
                for _ in range(loss.dim() - sample_weight.dim()):
                    sample_weight = sample_weight.unsqueeze(-1)
            loss = loss * sample_weight
        
        # Reduce
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            if mask is not None:
                return loss.sum() / (mask.sum() + 1e-8)
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'batch_mean':
            return loss.mean(dim=list(range(1, loss.dim())))
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class TweedieLoss(nn.Module):
    """
    Tweedie loss for non-negative continuous targets.
    Simplified to work with pre-scaled data from TFT dataset.
    """
    
    def __init__(self, p: float = 1.5, reduction: str = 'mean'):
        """
        Args:
            p: Power parameter (1 < p < 2 for compound Poisson-Gamma)
            reduction: How to reduce the loss
        """
        super().__init__()
        assert 1 < p < 2, "p must be between 1 and 2 for Tweedie loss"
        self.p = p
        self.reduction = reduction
    
    def forward(self,
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Tweedie loss.
        
        Note: Assumes data is already properly scaled by the dataset.
        For unscaled data, use inverse_transform first.
        """
        # Ensure predictions are positive
        y_pred = F.softplus(y_pred) + 1e-8
        
        # Tweedie loss formula
        loss = -y_true * torch.pow(y_pred, 1 - self.p) / (1 - self.p) + \
               torch.pow(y_pred, 2 - self.p) / (2 - self.p)
        
        # Apply mask
        if mask is not None:
            if mask.dim() < loss.dim():
                for _ in range(loss.dim() - mask.dim()):
                    mask = mask.unsqueeze(-1)
            loss = loss * mask
        
        # Apply sample weights
        if sample_weight is not None:
            if sample_weight.dim() < loss.dim():
                for _ in range(loss.dim() - sample_weight.dim()):
                    sample_weight = sample_weight.unsqueeze(-1)
            loss = loss * sample_weight
        
        # Reduce
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            if mask is not None:
                return loss.sum() / (mask.sum() + 1e-8)
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'batch_mean':
            return loss.mean(dim=list(range(1, loss.dim())))
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class CombinedLoss(nn.Module):
    """
    Combine multiple losses with learnable or fixed weights.
    Useful for multi-task learning or balancing different objectives.
    """
    
    def __init__(self, 
                 losses: List[nn.Module],
                 weights: Optional[List[float]] = None,
                 learnable_weights: bool = False):
        """
        Args:
            losses: List of loss modules
            weights: Initial weights for each loss
            learnable_weights: Whether weights should be learnable parameters
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0] * len(losses)
        
        if learnable_weights:
            # Log-space weights for numerical stability
            self.log_weights = nn.Parameter(torch.tensor(weights).log())
        else:
            self.register_buffer('weights', torch.tensor(weights))
            self.log_weights = None
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate combined loss."""
        total_loss = 0
        
        if self.log_weights is not None:
            weights = F.softmax(self.log_weights, dim=0) * len(self.losses)
        else:
            weights = self.weights
        
        for weight, loss_fn in zip(weights, self.losses):
            total_loss = total_loss + weight * loss_fn(y_pred, y_true, **kwargs)
        
        return total_loss


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that automatically balances multiple loss components
    based on their relative magnitudes using exponential moving average.
    """
    
    def __init__(self,
                 losses: List[nn.Module],
                 ema_decay: float = 0.99,
                 warmup_steps: int = 100):
        """
        Args:
            losses: List of loss modules
            ema_decay: Decay factor for exponential moving average
            warmup_steps: Number of steps before adaptation starts
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        
        # Buffers for EMA tracking
        self.register_buffer('loss_emas', torch.ones(len(losses)))
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('weights', torch.ones(len(losses)) / len(losses))
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate adaptive loss."""
        # Calculate individual losses
        individual_losses = []
        for loss_fn in self.losses:
            loss_val = loss_fn(y_pred, y_true, **kwargs)
            individual_losses.append(loss_val)
        
        losses_tensor = torch.stack(individual_losses)
        
        # Update EMAs and weights
        with torch.no_grad():
            self.step_count += 1
            
            if self.step_count > self.warmup_steps:
                # Update EMAs
                self.loss_emas = self.ema_decay * self.loss_emas + \
                                (1 - self.ema_decay) * losses_tensor.detach()
                
                # Calculate weights (inverse of EMAs, normalized)
                self.weights = 1.0 / (self.loss_emas + 1e-8)
                self.weights = self.weights / self.weights.sum()
            else:
                # During warmup, use equal weights
                self.weights = torch.ones_like(self.weights) / len(self.losses)
        
        # Apply weights
        weighted_loss = (self.weights * losses_tensor).sum()
        
        return weighted_loss


# Utility function for TFT training
#def get_tft_loss(loss_type: str = 'quantile', **kwargs) -> nn.Module:
#    """
#    Factory function to get appropriate loss for TFT models.
#    
#    Args:
#        loss_type: One of 'quantile', 'mse', 'mae', 'huber', 'tweedie'
#        **kwargs: Additional arguments for the loss function
#    
#    Returns:
#        Loss module
#    """
#    loss_map = {
#        'quantile': QuantileLoss,
#        'mse': MSELoss,
#        'rmse': RMSELoss,
#        'mae': MAELoss,
#        'huber': HuberLoss,
#        'tweedie': TweedieLoss,
#    }
#    
#    if loss_type not in loss_map:
#        raise ValueError(f"Unknown loss type: {loss_type}")
#    
#    return loss_map[loss_type](**kwargs)

