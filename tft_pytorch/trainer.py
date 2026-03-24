#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TFTTrainer:
    """
    Complete training pipeline for Temporal Fusion Transformer models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 train_adapter,  # TFTDataAdapter or TCNDataAdapter
                 val_adapter,

                 # Loss configuration
                 loss_type: str = 'quantile',
                 loss_params: Optional[Dict] = None,
                 
                 # Optimizer configuration
                 optimizer_type: str = 'adam',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 momentum: float = 0.9,  # For SGD
                 
                 # Scheduler configuration
                 scheduler_type: Optional[str] = 'reduce_on_plateau',
                 scheduler_factor: float = 0.5,
                 scheduler_patience: int = 10,
                 scheduler_t0: int = 10,  # For cosine annealing
                 scheduler_t_mult: int = 2,  # For cosine annealing
                 
                 # Training options
                 enable_gradient_clipping: bool = True,
                 max_grad_norm: float = 1.0,
                 enable_train_sample_weighting: bool = False,
                 enable_val_sample_weighting: bool = False,
                 
                 # Mixed precision training
                 enable_mixed_precision: bool = False,
                 
                 # Checkpointing
                 save_path: str = './checkpoints',
                 save_every: int = 5,
                 
                 # Other
                 device: Optional[str] = None):
        """
        Initialize TFT trainer with explicit parameters.
        
        Args:
            model: TFT or TFTEncoderOnly model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            adapter: Data adapter (TFTDataAdapter or TCNDataAdapter)
            
            loss_type: Type of loss function ('quantile', 'mse', 'mae', 'huber', 'tweedie', 'combined', 'adaptive')
            loss_params: Additional parameters for the loss function
            
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            momentum: Momentum for SGD optimizer
            
            scheduler_type: Type of LR scheduler ('reduce_on_plateau', 'cosine', None)
            scheduler_factor: Factor for ReduceLROnPlateau
            scheduler_patience: Patience for ReduceLROnPlateau
            scheduler_t0: T_0 for CosineAnnealingWarmRestarts
            scheduler_t_mult: T_mult for CosineAnnealingWarmRestarts
            
            enable_gradient_clipping: Whether to apply gradient clipping
            max_grad_norm: Maximum gradient norm for clipping
            enable_train_sample_weighting: Whether to use sample weights in loss calculation in train loop
            enable_val_sample_weighting: Whether to use sample weights in loss calculation in val loop
            
            save_path: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            
            device: Device to use (if None, uses model's device)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_adapter = train_adapter
        self.val_adapter = val_adapter

        # Training configuration
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        self.scheduler_type = scheduler_type
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_t0 = scheduler_t0
        self.scheduler_t_mult = scheduler_t_mult
        
        self.enable_gradient_clipping = enable_gradient_clipping
        self.max_grad_norm = max_grad_norm
        self.enable_train_sample_weighting = enable_train_sample_weighting
        self.enable_val_sample_weighting = enable_val_sample_weighting
        
        # Mixed precision training setup
        self.enable_mixed_precision = enable_mixed_precision
        self.scaler = torch.amp.GradScaler(model.device) if enable_mixed_precision else None
        
        self.save_every = save_every
        
        # Setup paths
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = device or model.device
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        
        # Setup loss function
        self.setup_loss()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {}
        
    def setup_loss(self):
        """Setup loss function based on configuration."""
        # Import loss functions
        from .losses import (
            QuantileLoss, MSELoss, MAELoss, HuberLoss, 
            TweedieLoss, CombinedLoss, AdaptiveLoss
        )
        
        if self.loss_type == 'quantile':
            self.criterion = QuantileLoss(
                quantiles=self.loss_params.get('quantiles', [0.1, 0.5, 0.9]),
                reduction='mean'
            )
        elif self.loss_type == 'mse':
            self.criterion = MSELoss(reduction='mean')
        elif self.loss_type == 'mae':
            self.criterion = MAELoss(reduction='mean')
        elif self.loss_type == 'huber':
            self.criterion = HuberLoss(
                delta=self.loss_params.get('delta', 1.0),
                reduction='mean'
            )
        elif self.loss_type == 'tweedie':
            self.criterion = TweedieLoss(
                p=self.loss_params.get('p', 1.5),
                reduction='mean'
            )
        elif self.loss_type == 'combined':
            # Use multiple losses
            losses = []
            for loss_spec in self.loss_params.get('losses', []):
                loss_fn = self._get_loss_function(loss_spec['type'], **loss_spec.get('params', {}))
                losses.append(loss_fn)
            self.criterion = CombinedLoss(
                losses=losses,
                weights=self.loss_params.get('weights'),
                learnable_weights=self.loss_params.get('learnable_weights', False)
            )
        elif self.loss_type == 'adaptive':
            # Adaptive multi-loss
            losses = []
            for loss_spec in self.loss_params.get('losses', []):
                loss_fn = self._get_loss_function(loss_spec['type'], **loss_spec.get('params', {}))
                losses.append(loss_fn)
            self.criterion = AdaptiveLoss(
                losses=losses,
                ema_decay=self.loss_params.get('ema_decay', 0.99),
                warmup_steps=self.loss_params.get('warmup_steps', 100)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _get_loss_function(self, loss_type: str, **kwargs):
        """Helper to get loss function by type."""
        from .losses import (
            QuantileLoss, MSELoss, MAELoss, HuberLoss, TweedieLoss
        )
        
        loss_map = {
            'quantile': QuantileLoss,
            'mse': MSELoss,
            'mae': MAELoss,
            'huber': HuberLoss,
            'tweedie': TweedieLoss,
        }
        
        if loss_type not in loss_map:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss_map[loss_type](**kwargs)
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Get optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        # Setup scheduler
        if self.scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.scheduler_t0,
                T_mult=self.scheduler_t_mult
            )
        else:
            self.scheduler = None
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Adapt batch for model
            model_inputs = self.train_adapter.adapt_for_tft(batch)
            
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=self.enable_mixed_precision):
                # Forward pass
                outputs = self.model(static_categorical = model_inputs.get('static_categorical', None),
                                     static_continuous = model_inputs.get('static_continuous', None),
                                     historical_categorical = model_inputs.get('historical_categorical', None),
                                     historical_continuous = model_inputs.get('historical_continuous', None),
                                     future_categorical = model_inputs.get('future_categorical', None),
                                     future_continuous = model_inputs.get('future_continuous', None),
                                     padding_mask = model_inputs.get('padding_mask', None)
                                    )

                # Calculate loss
                predictions = outputs['predictions']
                targets = model_inputs['future_targets']

                #print(f"predictions shape: {predictions.shape}, targets shape: {targets.shape}")

                # Get mask for future timesteps
                if 'mask' in batch:
                    future_mask = batch['mask'][:, -self.model.prediction_steps:]
                    #print(f"mask shape: {future_mask.shape}")
                else:
                    future_mask = None

                # Prepare loss kwargs
                loss_kwargs = {
                    'mask': future_mask
                }

                # Add sample weights if enabled
                #if self.enable_sample_weighting and 'recency_weight' in batch:
                #    loss_kwargs['sample_weight'] = batch['recency_weight']

                # Combine entity and recency weights
                if self.enable_train_sample_weighting:
                    batch_size = predictions.shape[0]
                    combined_weight = torch.ones(batch_size, dtype=torch.float32).to(self.device)

                    # Apply entity weight
                    if 'entity_weight' in batch:
                        combined_weight = combined_weight * batch['entity_weight']

                    # Apply recency weight
                    if 'recency_weight' in batch:
                        combined_weight = combined_weight * batch['recency_weight']

                    # Reshape for broadcasting
                    if predictions.dim() == 3:
                        # [batch] -> [batch, 1, 1] for [batch, time, features]
                        combined_weight = combined_weight.unsqueeze(1).unsqueeze(2)
                    elif predictions.dim() == 2:
                        # [batch] -> [batch, 1] for [batch, features]
                        combined_weight = combined_weight.unsqueeze(1)

                    loss_kwargs['sample_weight'] = combined_weight

                # Special handling for TweedieLoss
                if self.loss_type == 'tweedie':
                    # Inverse transform both predictions and targets
                    window_indices = batch['window_idx'].tolist()

                    predictions_original = self.train_adapter.dataset.inverse_transform_predictions(
                        predictions, window_indices, target_col=self.train_adapter.dataset.target_cols[0]
                    )
                    targets_original = self.train_adapter.dataset.inverse_transform_predictions(
                        targets, window_indices, target_col=self.train_adapter.dataset.target_cols[0]
                    )

                    # Ensure non-negative (clip at 0 for safety)
                    predictions_original = torch.clamp(predictions_original, min=0)
                    targets_original = torch.clamp(targets_original, min=0)

                    # Calculate loss with mask and weights
                    loss = self.criterion(predictions_original, targets_original, **loss_kwargs)
                else:
                    # Compute loss
                    loss = self.criterion(predictions, targets, **loss_kwargs)
                    
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.enable_mixed_precision:
                # Scale loss and backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping if enabled (with unscaling)
                if self.enable_gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard backward pass
                loss.backward()

                # Gradient clipping if enabled
                if self.enable_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Additional metrics
        all_predictions = []
        all_targets = []
        all_masks = []
        
        for batch in self.val_loader:
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Adapt batch
            model_inputs = self.val_adapter.adapt_for_tft(batch)
            
            # Forward pass with mixed precision if enabled
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=self.enable_mixed_precision):
                outputs = self.model(static_categorical = model_inputs.get('static_categorical', None),
                                     static_continuous = model_inputs.get('static_continuous', None),
                                     historical_categorical = model_inputs.get('historical_categorical', None),
                                     historical_continuous = model_inputs.get('historical_continuous', None),
                                     future_categorical = model_inputs.get('future_categorical', None),
                                     future_continuous = model_inputs.get('future_continuous', None),
                                     padding_mask = model_inputs.get('padding_mask', None)
                                    )

                # Calculate loss
                predictions = outputs['predictions']
                targets = model_inputs['future_targets']

                if 'mask' in batch:
                    future_mask = batch['mask'][:, -self.model.prediction_steps:]
                else:
                    future_mask = None

                loss_kwargs = {'mask': future_mask}

                # Combine entity and recency weights
                combined_weight = None
                if self.enable_val_sample_weighting:
                    batch_size = predictions.shape[0]
                    combined_weight = torch.ones(batch_size, dtype=torch.float32).to(self.device)

                    # Apply entity weight
                    if 'entity_weight' in batch:
                        entity_weight = batch['entity_weight']
                        combined_weight = combined_weight * entity_weight

                    # Apply recency weight
                    if 'recency_weight' in batch:
                        recency_weight = batch['recency_weight']
                        combined_weight = combined_weight * recency_weight

                    # Store original shape for metrics
                    combined_weight_1d = combined_weight.clone()

                    # Reshape for broadcasting in loss calculation
                    if predictions.dim() == 3:
                        combined_weight = combined_weight.unsqueeze(1).unsqueeze(2)
                    elif predictions.dim() == 2:
                        combined_weight = combined_weight.unsqueeze(1)

                    loss_kwargs['sample_weight'] = combined_weight

                # Special handling for TweedieLoss
                if self.loss_type == 'tweedie':
                    # Inverse transform both predictions and targets
                    window_indices = batch['window_idx'].tolist()

                    predictions_original = self.val_adapter.dataset.inverse_transform_predictions(
                        predictions, window_indices, target_col=self.val_adapter.dataset.target_cols[0]
                    )
                    targets_original = self.val_adapter.dataset.inverse_transform_predictions(
                        targets, window_indices, target_col=self.val_adapter.dataset.target_cols[0]
                    )

                    # Ensure non-negative (clip at 0 for safety)
                    predictions_original = torch.clamp(predictions_original, min=0)
                    targets_original = torch.clamp(targets_original, min=0)

                    # Calculate loss with mask and weights
                    loss = self.criterion(predictions_original, targets_original, **loss_kwargs)
                else:
                    loss = self.criterion(predictions, targets, **loss_kwargs)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store for metrics calculation
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            if future_mask is not None:
                all_masks.append(future_mask.cpu())
            
        avg_loss = total_loss / num_batches
        
        # Calculate additional metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['val_loss'] = avg_loss
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions: torch.Tensor, 
                         targets: torch.Tensor) -> Dict:
        """Calculate evaluation metrics."""
        import torch.nn.functional as F
        
        metrics = {}
        
        # For quantile predictions, use median (usually index 1 for [0.1, 0.5, 0.9])
        if predictions.dim() == 3 and predictions.size(-1) > 1:
            # Use median quantile for point estimates
            median_idx = predictions.size(-1) // 2
            point_predictions = predictions[..., median_idx]
        else:
            point_predictions = predictions.squeeze(-1)
        
        if targets.dim() == 3:
            targets = targets.squeeze(-1)
        
        # MSE
        mse = F.mse_loss(point_predictions, targets).item()
        metrics['mse'] = mse
        metrics['rmse'] = np.sqrt(mse)
        
        # MAE
        mae = F.l1_loss(point_predictions, targets).item()
        metrics['mae'] = mae
        
        # MAPE (if targets are not too close to zero)
        mask = torch.abs(targets) > 0.1
        if mask.any():
            mape = (torch.abs((targets - point_predictions) / targets) * mask).sum() / mask.sum()
            metrics['mape'] = mape.item() * 100
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None, 
            'best_val_loss': self.best_val_loss,
            'training_params': {
                'loss_type': self.loss_type,
                'optimizer_type': self.optimizer_type,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'enable_gradient_clipping': self.enable_gradient_clipping,
                'max_grad_norm': self.max_grad_norm,
                'enable_train_sample_weighting': self.enable_train_sample_weighting,
                'enable_val_sample_weighting': self.enable_val_sample_weighting,
                'enable_mixed_precision': self.enable_mixed_precision
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_path / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.save_path / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            
            # Also save just the model for easier inference
            model_only_path = self.save_path / 'best_model_weights.pt'
            torch.save(self.model.state_dict(), model_only_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available and mixed precision is enabled
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.metrics_history = checkpoint.get('metrics_history', {})
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int, patience: int = 20):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            logger.info(f"\nEpoch {self.epoch}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, metrics = self.validate()
            self.val_losses.append(val_loss)
            logger.info(f"Val Loss: {val_loss:.4f}, Metrics: {metrics}")
            
            # Store metrics
            for key, value in metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if self.epoch % self.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {self.epoch} epochs")
                break
        
        logger.info("Training completed!")


class TFTInference:
    """
    Inference pipeline for trained TFT models.
    """
    
    def __init__(self, 
                 model_path: str,
                 model: nn.Module,
                 adapter,
                 device: str = 'cuda'):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to saved model checkpoint
            model: Initialized model architecture (must match training)
            adapter: Data adapter (must match training setup)
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.model = model
        self.adapter = adapter
        
        # Load weights
        self.load_model_weights()
    
    def load_model_weights(self):
        """Load trained model weights."""
        if self.model_path.suffix == '.pt':
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume it's just the state dict
                self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model from {self.model_path}")
    
    @torch.no_grad()
    def predict_batch(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a dataloader.
        
        Returns:
            predictions: Model predictions
            targets: Actual targets (if available)
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_window_indices = []
        
        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Adapt batch
            model_inputs = self.adapter.adapt_for_tft(batch)
            
            # Forward pass
            outputs = self.model(static_categorical = model_inputs.get('static_categorical', None),
                                 static_continuous = model_inputs.get('static_continuous', None),
                                 historical_categorical = model_inputs.get('historical_categorical', None),
                                 historical_continuous = model_inputs.get('historical_continuous', None),
                                 future_categorical = model_inputs.get('future_categorical', None),
                                 future_continuous = model_inputs.get('future_continuous', None),
                                 padding_mask = model_inputs.get('padding_mask', None)
                                )
            predictions = outputs['predictions']
            
            # Store predictions and indices
            all_predictions.append(predictions.cpu())
            if 'window_idx' in batch:
                all_window_indices.extend(batch['window_idx'].cpu().tolist())
            
            if 'future_targets' in model_inputs and model_inputs['future_targets'] is not None:
                all_targets.append(model_inputs['future_targets'].cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        predictions_np = all_predictions.numpy()
        
        # Process targets if available
        targets_np = None
        if all_targets:
            all_targets = torch.cat(all_targets, dim=0)
            targets_np = all_targets.numpy()
        
        return predictions_np, targets_np


class TFTInferenceWithTracking(TFTInference):
    """
    Enhanced inference pipeline that tracks entity IDs and timestamps.
    """
    
    @torch.no_grad()
    def predict_with_metadata(self, dataloader) -> pd.DataFrame:
        """
        Run inference and return a DataFrame with predictions, actuals, entity IDs, and timestamps.
        
        Returns:
            DataFrame with columns: entity_id, timestamp, prediction_*, actual_*, ...
        """
        self.model.eval()
        
        all_results = []
        
        for batch in dataloader:
            batch_gpu = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            
            # Adapt batch
            model_inputs = self.adapter.adapt_for_tft(batch_gpu)
            
            # Forward pass
            outputs = self.model(
                static_categorical=model_inputs.get('static_categorical', None),
                static_continuous=model_inputs.get('static_continuous', None),
                historical_categorical=model_inputs.get('historical_categorical', None),
                historical_continuous=model_inputs.get('historical_continuous', None),
                future_categorical=model_inputs.get('future_categorical', None),
                future_continuous=model_inputs.get('future_continuous', None),
                padding_mask=model_inputs.get('padding_mask', None)
            )
            
            predictions = outputs['predictions'].cpu().numpy()
            
            # Get actuals if available
            actuals = None
            if 'future_targets' in model_inputs and model_inputs['future_targets'] is not None:
                actuals = model_inputs['future_targets'].cpu().numpy()
            
            # Get metadata from batch
            entity_ids = batch['entity_id']  # List of entity IDs
            window_indices = batch['window_idx'].cpu().numpy()
            
            # Process each sample in batch
            for i in range(len(entity_ids)):
                entity_id = entity_ids[i]
                window_idx = window_indices[i]
                
                # Get timestamps for this window
                future_timestamps = self.adapter.dataset.get_future_timestamps(window_idx)
                
                # Inverse transform predictions if needed
                pred_sample = predictions[i]  # Shape: [prediction_steps, num_outputs]
                
                if self.adapter.dataset.scaling_method != 'none':
                    # Inverse transform for each target
                    pred_original = np.zeros_like(pred_sample)
                    for target_idx, target_col in enumerate(self.adapter.dataset.target_cols):
                        if target_idx < pred_sample.shape[-1]:
                            pred_original[:, target_idx] = self._inverse_transform_single(
                                pred_sample[:, target_idx],
                                window_idx,
                                target_col
                            )
                else:
                    pred_original = pred_sample
                
                # Process actuals if available
                actual_original = None
                if actuals is not None:
                    actual_sample = actuals[i]
                    if self.adapter.dataset.scaling_method != 'none':
                        actual_original = np.zeros_like(actual_sample)
                        for target_idx, target_col in enumerate(self.adapter.dataset.target_cols):
                            if target_idx < actual_sample.shape[-1]:
                                actual_original[:, target_idx] = self._inverse_transform_single(
                                    actual_sample[:, target_idx],
                                    window_idx,
                                    target_col
                                )
                    else:
                        actual_original = actual_sample
                
                # Create records for each timestamp
                for t, timestamp in enumerate(future_timestamps):
                    record = {
                        'entity_id': entity_id,
                        'timestamp': timestamp,
                        'window_idx': window_idx,
                        'horizon': t + 1  # 1-indexed horizon
                    }
                    
                    # Add predictions (handle multiple outputs/quantiles)
                    if pred_original.ndim == 2:
                        if pred_original.shape[1] == 3:  # Assuming quantiles
                            record['pred_q10'] = pred_original[t, 0]
                            record['pred_q50'] = pred_original[t, 1]
                            record['pred_q90'] = pred_original[t, 2]
                        else:
                            for q in range(pred_original.shape[1]):
                                record[f'pred_{q}'] = pred_original[t, q]
                    else:
                        record['prediction'] = pred_original[t]
                    
                    # Add actuals if available
                    if actual_original is not None:
                        if actual_original.ndim == 2:
                            for target_idx, target_col in enumerate(self.adapter.dataset.target_cols):
                                if target_idx < actual_original.shape[1]:
                                    record[f'actual_{target_col}'] = actual_original[t, target_idx]
                        else:
                            record['actual'] = actual_original[t]
                    
                    all_results.append(record)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Sort by entity and timestamp
        results_df = results_df.sort_values(['entity_id', 'timestamp'])
        
        return results_df
    
    def _inverse_transform_single(self, values, window_idx, target_col):
        """
        Inverse transform a single target column.
        """
        dataset = self.adapter.dataset
        col_idx = dataset.feature_to_idx.get(target_col)
        
        if col_idx is None:
            return values
        
        if dataset.scaling_method == 'mean':
            scale = dataset.scaler_params[window_idx, col_idx]
            return values * scale
        elif dataset.scaling_method == 'standard':
            mean = dataset.scaler_params[window_idx, col_idx, 0]
            std = dataset.scaler_params[window_idx, col_idx, 1]
            return values * std + mean
        
        return values
    
