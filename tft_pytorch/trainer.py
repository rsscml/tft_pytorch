"""
Training and inference pipelines for Temporal Fusion Transformer models.

Classes
-------
TFTTrainer
    End-to-end training loop with configurable loss, optimizer, LR scheduler,
    gradient clipping, mixed-precision, sample weighting, and checkpointing.
TFTInference
    Lightweight inference wrapper that loads a checkpoint and runs batch prediction.
TFTInferenceWithTracking
    Extended inference that returns a tidy DataFrame with entity IDs, timestamps,
    predictions (with quantile columns), and actuals.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

from .losses import (
    AdaptiveLoss,
    CombinedLoss,
    HuberLoss,
    MAELoss,
    MSELoss,
    QuantileLoss,
    TweedieLoss,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TFTTrainer:
    """
    Complete training pipeline for :class:`~tft_pytorch.models.TemporalFusionTransformer`
    (and compatible encoder-only / TCN) models.

    Parameters
    ----------
    model : nn.Module
        TFT or TFTEncoderOnly model.
    train_loader, val_loader : DataLoader
        Must be created with the matching adapter's ``collate_fn``.
    train_adapter, val_adapter : TFTDataAdapter
        Used to convert collated batches to model inputs.
    loss_type : str
        One of ``'quantile'``, ``'mse'``, ``'mae'``, ``'huber'``, ``'tweedie'``,
        ``'combined'``, ``'adaptive'``.
    loss_params : dict, optional
        Extra keyword arguments forwarded to the loss constructor.
        For ``'quantile'``: ``{'quantiles': [0.1, 0.5, 0.9]}``.
        For ``'huber'``:    ``{'delta': 1.0}``.
        For ``'tweedie'``:  ``{'p': 1.5}``.
        For ``'combined'`` / ``'adaptive'``:
          ``{'losses': [{'type': 'quantile', 'params': {...}}, ...], 'weights': [...]}``
    optimizer_type : str
        ``'adam'`` | ``'adamw'`` | ``'sgd'``
    learning_rate : float
    weight_decay : float
    momentum : float
        Only used with SGD.
    scheduler_type : str or None
        ``'reduce_on_plateau'`` | ``'cosine'`` | ``None``
    scheduler_factor, scheduler_patience : float / int
        Parameters for ReduceLROnPlateau.
    scheduler_t0, scheduler_t_mult : int
        Parameters for CosineAnnealingWarmRestarts.
    enable_gradient_clipping : bool
    max_grad_norm : float
    enable_train_sample_weighting : bool
        Multiply per-sample loss by ``entity_weight * recency_weight`` during training.
    enable_val_sample_weighting : bool
        Same as above but for validation.
    enable_mixed_precision : bool
        Use ``torch.amp`` FP16 training (GPU only).
    save_path : str
        Directory for checkpoints.
    save_every : int
        Save a checkpoint every N epochs.
    device : str, optional
        Falls back to the model's own device attribute.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_adapter,
        val_adapter,
        loss_type: str = 'quantile',
        loss_params: Optional[Dict] = None,
        optimizer_type: str = 'adam',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        momentum: float = 0.9,
        scheduler_type: Optional[str] = 'reduce_on_plateau',
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        scheduler_t0: int = 10,
        scheduler_t_mult: int = 2,
        enable_gradient_clipping: bool = True,
        max_grad_norm: float = 1.0,
        enable_train_sample_weighting: bool = False,
        enable_val_sample_weighting: bool = False,
        enable_mixed_precision: bool = False,
        save_path: str = './checkpoints',
        save_every: int = 5,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_adapter = train_adapter
        self.val_adapter = val_adapter

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
        self.enable_mixed_precision = enable_mixed_precision
        self.save_every = save_every
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.device = device or getattr(model, 'device', 'cpu')

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.metrics_history: Dict[str, List] = {}

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler(self.device) if enable_mixed_precision else None

        self._setup_loss()
        self._setup_optimizer()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_loss(self):
        lp = self.loss_params

        if self.loss_type == 'quantile':
            self.criterion = QuantileLoss(quantiles=lp.get('quantiles', [0.1, 0.5, 0.9]))
        elif self.loss_type == 'mse':
            self.criterion = MSELoss()
        elif self.loss_type == 'mae':
            self.criterion = MAELoss()
        elif self.loss_type == 'huber':
            self.criterion = HuberLoss(delta=lp.get('delta', 1.0))
        elif self.loss_type == 'tweedie':
            self.criterion = TweedieLoss(p=lp.get('p', 1.5))
        elif self.loss_type == 'combined':
            losses = [self._build_single_loss(s['type'], **s.get('params', {}))
                      for s in lp.get('losses', [])]
            self.criterion = CombinedLoss(losses, weights=lp.get('weights'),
                                          learnable_weights=lp.get('learnable_weights', False))
        elif self.loss_type == 'adaptive':
            losses = [self._build_single_loss(s['type'], **s.get('params', {}))
                      for s in lp.get('losses', [])]
            self.criterion = AdaptiveLoss(losses, ema_decay=lp.get('ema_decay', 0.99),
                                          warmup_steps=lp.get('warmup_steps', 100))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _build_single_loss(self, loss_type: str, **kwargs) -> nn.Module:
        mapping = {
            'quantile': QuantileLoss,
            'mse': MSELoss,
            'mae': MAELoss,
            'huber': HuberLoss,
            'tweedie': TweedieLoss,
        }
        if loss_type not in mapping:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return mapping[loss_type](**kwargs)

    def _setup_optimizer(self):
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                       momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        if self.scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                               factor=self.scheduler_factor,
                                               patience=self.scheduler_patience)
        elif self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer,
                                                         T_0=self.scheduler_t0,
                                                         T_mult=self.scheduler_t_mult)
        else:
            self.scheduler = None

    # ------------------------------------------------------------------
    # Per-batch helpers
    # ------------------------------------------------------------------

    def _build_sample_weight(self, predictions: torch.Tensor, batch: Dict) -> Optional[torch.Tensor]:
        bs = predictions.shape[0]
        w = torch.ones(bs, dtype=torch.float32, device=self.device)
        if 'entity_weight' in batch:
            w = w * batch['entity_weight'].to(self.device)
        if 'recency_weight' in batch:
            w = w * batch['recency_weight'].to(self.device)
        # Reshape for broadcasting
        if predictions.dim() == 3:
            w = w.unsqueeze(1).unsqueeze(2)
        elif predictions.dim() == 2:
            w = w.unsqueeze(1)
        return w

    def _forward_and_loss(self, model_inputs: Dict, batch: Dict,
                          use_sample_weighting: bool) -> torch.Tensor:
        outputs = self.model(
            static_categorical=model_inputs.get('static_categorical'),
            static_continuous=model_inputs.get('static_continuous'),
            historical_categorical=model_inputs.get('historical_categorical'),
            historical_continuous=model_inputs.get('historical_continuous'),
            future_categorical=model_inputs.get('future_categorical'),
            future_continuous=model_inputs.get('future_continuous'),
            padding_mask=model_inputs.get('padding_mask'),
        )

        predictions = outputs['predictions']
        targets = model_inputs['future_targets']

        future_mask = batch['mask'][:, -self.model.prediction_steps:] if 'mask' in batch else None
        loss_kwargs = {'mask': future_mask}

        if use_sample_weighting:
            loss_kwargs['sample_weight'] = self._build_sample_weight(predictions, batch)

        if self.loss_type == 'tweedie':
            win_idx = batch['window_idx'].tolist()
            dataset = self.train_adapter.dataset
            tgt_col = dataset.target_cols[0]
            predictions = torch.clamp(dataset.inverse_transform_predictions(predictions, win_idx, tgt_col), min=0)
            targets = torch.clamp(dataset.inverse_transform_predictions(targets, win_idx, tgt_col), min=0)

        return self.criterion(predictions, targets, **loss_kwargs)

    # ------------------------------------------------------------------
    # Training / validation epochs
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """Run one training epoch and return average loss."""
        self.model.train()
        total, count = 0.0, 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            model_inputs = self.train_adapter.adapt_for_tft(batch)

            with torch.amp.autocast(device_type=self.device, dtype=torch.float16,
                                    enabled=self.enable_mixed_precision):
                loss = self._forward_and_loss(model_inputs, batch, self.enable_train_sample_weighting)

            self.optimizer.zero_grad()
            if self.enable_mixed_precision:
                self.scaler.scale(loss).backward()
                if self.enable_gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.enable_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total += loss.item()
            count += 1

            if batch_idx % 100 == 0:
                logger.info(f"  Batch {batch_idx}/{len(self.train_loader)}  loss={loss.item():.4f}")

        return total / count

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """Run validation and return (avg_loss, metrics_dict)."""
        self.model.eval()
        total, count = 0.0, 0
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            model_inputs = self.val_adapter.adapt_for_tft(batch)

            with torch.amp.autocast(device_type=self.device, dtype=torch.float16,
                                    enabled=self.enable_mixed_precision):
                outputs = self.model(
                    static_categorical=model_inputs.get('static_categorical'),
                    static_continuous=model_inputs.get('static_continuous'),
                    historical_categorical=model_inputs.get('historical_categorical'),
                    historical_continuous=model_inputs.get('historical_continuous'),
                    future_categorical=model_inputs.get('future_categorical'),
                    future_continuous=model_inputs.get('future_continuous'),
                    padding_mask=model_inputs.get('padding_mask'),
                )
                predictions = outputs['predictions']
                targets = model_inputs['future_targets']
                future_mask = batch['mask'][:, -self.model.prediction_steps:] if 'mask' in batch else None
                loss_kwargs = {'mask': future_mask}
                if self.enable_val_sample_weighting:
                    loss_kwargs['sample_weight'] = self._build_sample_weight(predictions, batch)

                if self.loss_type == 'tweedie':
                    win_idx = batch['window_idx'].tolist()
                    dataset = self.val_adapter.dataset
                    tgt_col = dataset.target_cols[0]
                    predictions = torch.clamp(
                        dataset.inverse_transform_predictions(predictions, win_idx, tgt_col), min=0)
                    targets = torch.clamp(
                        dataset.inverse_transform_predictions(targets, win_idx, tgt_col), min=0)

                loss = self.criterion(predictions, targets, **loss_kwargs)

            total += loss.item()
            count += 1
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())

        avg_loss = total / count
        metrics = self.calculate_metrics(torch.cat(all_preds), torch.cat(all_targets))
        metrics['val_loss'] = avg_loss
        return avg_loss, metrics

    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Compute MSE, RMSE, MAE, and MAPE from point predictions."""
        import torch.nn.functional as F

        if predictions.dim() == 3 and predictions.size(-1) > 1:
            p = predictions[..., predictions.size(-1) // 2]
        else:
            p = predictions.squeeze(-1)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)

        mse = F.mse_loss(p, targets).item()
        mae = F.l1_loss(p, targets).item()
        metrics = {'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': mae}

        mask = torch.abs(targets) > 0.1
        if mask.any():
            mape = (torch.abs((targets - p) / targets) * mask).sum() / mask.sum()
            metrics['mape'] = float(mape.item() * 100)

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_best: bool = False):
        """Save training state to disk."""
        ckpt = {
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
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
        }
        path = self.save_path / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint → {path}")

        if is_best:
            best_path = self.save_path / 'best_model.pt'
            torch.save(ckpt, best_path)
            torch.save(self.model.state_dict(), self.save_path / 'best_model_weights.pt')
            logger.info(f"Saved best model → {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Restore training state from a checkpoint file."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if self.scaler and ckpt.get('scaler_state_dict'):
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.epoch = ckpt['epoch']
        self.best_val_loss = ckpt['best_val_loss']
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses = ckpt.get('val_losses', [])
        self.metrics_history = ckpt.get('metrics_history', {})
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int, patience: int = 20):
        """
        Full training loop with early stopping.

        Parameters
        ----------
        num_epochs : int
            Maximum number of epochs.
        patience : int
            Stop training after this many epochs without improvement.
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            logger.info(f"\nEpoch {self.epoch}/{num_epochs}")

            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            logger.info(f"Train Loss: {train_loss:.4f}")

            val_loss, metrics = self.validate()
            self.val_losses.append(val_loss)
            logger.info(f"Val Loss: {val_loss:.4f}  Metrics: {metrics}")

            for k, v in metrics.items():
                self.metrics_history.setdefault(k, []).append(v)

            # LR scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"New best val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1

            if self.epoch % self.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {self.epoch} epochs")
                break

        logger.info("Training completed!")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TFTInference:
    """
    Minimal inference wrapper.

    Loads model weights from a ``.pt`` checkpoint and provides :meth:`predict_batch`
    for running on a DataLoader.

    Parameters
    ----------
    model_path : str
        Path to a ``best_model.pt`` or ``best_model_weights.pt`` file.
    model : nn.Module
        An initialised model with the *same architecture* as during training.
    adapter : TFTDataAdapter
        The adapter used to convert batches into model inputs.
    device : str
    """

    def __init__(self, model_path: str, model: nn.Module, adapter, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.model = model
        self.adapter = adapter
        self._load_weights()

    def _load_weights(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()
        logger.info(f"Loaded model weights from {self.model_path}")

    @torch.no_grad()
    def predict_batch(self, dataloader: DataLoader) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run inference on a DataLoader.

        Returns
        -------
        predictions : np.ndarray  [N, prediction_steps, num_outputs]
        targets     : np.ndarray or None
        """
        self.model.eval()
        all_preds, all_targets = [], []

        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            mi = self.adapter.adapt_for_tft(batch)
            outputs = self.model(
                static_categorical=mi.get('static_categorical'),
                static_continuous=mi.get('static_continuous'),
                historical_categorical=mi.get('historical_categorical'),
                historical_continuous=mi.get('historical_continuous'),
                future_categorical=mi.get('future_categorical'),
                future_continuous=mi.get('future_continuous'),
                padding_mask=mi.get('padding_mask'),
            )
            all_preds.append(outputs['predictions'].cpu())
            if mi.get('future_targets') is not None:
                all_targets.append(mi['future_targets'].cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy() if all_targets else None
        return preds, targets


class TFTInferenceWithTracking(TFTInference):
    """
    Enhanced inference pipeline that returns a DataFrame with predictions,
    actuals, entity IDs, and timestamps.

    Inherits from :class:`TFTInference`.
    """

    @torch.no_grad()
    def predict_with_metadata(self, dataloader: DataLoader) -> pd.DataFrame:
        """
        Run inference and return a tidy DataFrame.

        Columns
        -------
        ``entity_id``, ``timestamp``, ``window_idx``, ``horizon`` (1-indexed),
        ``pred_q10`` / ``pred_q50`` / ``pred_q90`` (for 3-quantile models) or
        ``pred_0`` … ``pred_N``, ``actual_<target_col>`` (when targets available).

        Returns
        -------
        pd.DataFrame  sorted by (entity_id, timestamp)
        """
        self.model.eval()
        records = []

        for batch in dataloader:
            batch_gpu = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            mi = self.adapter.adapt_for_tft(batch_gpu)

            outputs = self.model(
                static_categorical=mi.get('static_categorical'),
                static_continuous=mi.get('static_continuous'),
                historical_categorical=mi.get('historical_categorical'),
                historical_continuous=mi.get('historical_continuous'),
                future_categorical=mi.get('future_categorical'),
                future_continuous=mi.get('future_continuous'),
                padding_mask=mi.get('padding_mask'),
            )

            preds = outputs['predictions'].cpu().numpy()
            actuals = mi['future_targets'].cpu().numpy() if mi.get('future_targets') is not None else None
            entity_ids = batch['entity_id']
            window_indices = batch['window_idx'].cpu().numpy()

            for i in range(len(entity_ids)):
                eid = entity_ids[i]
                widx = int(window_indices[i])
                future_ts = self.adapter.dataset.get_future_timestamps(widx)

                pred = self._maybe_inverse(preds[i], widx)
                act = self._maybe_inverse(actuals[i], widx) if actuals is not None else None

                for t, ts in enumerate(future_ts):
                    rec: Dict = {
                        'entity_id': eid,
                        'timestamp': ts,
                        'window_idx': widx,
                        'horizon': t + 1,
                    }
                    if pred.ndim == 2:
                        if pred.shape[1] == 3:
                            rec.update({'pred_q10': pred[t, 0], 'pred_q50': pred[t, 1], 'pred_q90': pred[t, 2]})
                        else:
                            for q in range(pred.shape[1]):
                                rec[f'pred_{q}'] = pred[t, q]
                    else:
                        rec['prediction'] = pred[t]

                    if act is not None:
                        if act.ndim == 2:
                            for qi, tgt_col in enumerate(self.adapter.dataset.target_cols):
                                if qi < act.shape[1]:
                                    rec[f'actual_{tgt_col}'] = act[t, qi]
                        else:
                            rec['actual'] = act[t]

                    records.append(rec)

        df = pd.DataFrame(records)
        return df.sort_values(['entity_id', 'timestamp']).reset_index(drop=True)

    def _maybe_inverse(self, values: np.ndarray, window_idx: int) -> np.ndarray:
        """Inverse-transform predictions / actuals if scaling is enabled."""
        dataset = self.adapter.dataset
        if dataset.scaling_method == 'none':
            return values
        out = np.zeros_like(values)
        for ti, tgt_col in enumerate(dataset.target_cols):
            cidx = dataset.feature_to_idx.get(tgt_col)
            if cidx is None:
                out[:, ti] = values[:, ti] if values.ndim == 2 else values
                continue
            v = values[:, ti] if values.ndim == 2 else values
            if dataset.scaling_method == 'mean':
                out[:, ti] = v * dataset.scaler_params[window_idx, cidx]
            else:
                mean = dataset.scaler_params[window_idx, cidx, 0]
                std = dataset.scaler_params[window_idx, cidx, 1]
                out[:, ti] = v * std + mean
        return out
