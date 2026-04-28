#!/usr/bin/env python
# coding: utf-8

"""
patchtst
========

PyTorch implementation of **PatchTST** ("A Time Series is Worth 64 Words:
Long-term Forecasting with Transformers", Nie et al., ICLR 2023) that plugs
directly into the ``tft_pytorch`` pipeline (``OptimizedTFTDataset`` ->
``TFTDataAdapter`` -> ``TFTTrainer``).

Design notes
------------
* **Channel-independent (CI)** backbone, as in the paper: every numeric
  channel is projected and attended-to with shared weights, which is both
  memory-efficient and a strong inductive bias for long-horizon forecasting.
* **Reversible Instance Normalization (RevIN)** is built in and is aware of
  the padding mask produced by ``OptimizedTFTDataset`` -- padded timesteps
  are excluded from the per-channel statistics.
* **Forward interface mirrors ``TemporalFusionTransformer``**, so the
  existing ``TFTTrainer`` / ``TFTInference`` / ``TFTInferenceWithTracking``
  work with zero code changes. Inputs the paper does not use (static,
  categorical, future) are accepted and silently ignored.
* **Output format** is ``{'predictions': [B, prediction_steps, num_outputs]}``,
  identical to the TFT output, so ``QuantileLoss``, ``MSELoss`` etc. work
  unchanged. Setting ``num_outputs = len(quantiles)`` enables probabilistic
  forecasting for a single-target setup.

The backbone is faithful to the reference implementation at
https://github.com/yuqinie98/PatchTST (supervised track), with the additions
noted above for adapter compatibility.
"""

from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reversible Instance Normalization (RevIN)
# ---------------------------------------------------------------------------
class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Kim et al., "Reversible Instance Normalization for Accurate Time-Series
    Forecasting against Distribution Shift" (ICLR 2022). Used by PatchTST to
    de-correlate the forecast from slow-moving distributional drift.

    Args:
        num_channels: Number of independent channels (C).
        eps: Variance floor for numerical stability.
        affine: If ``True``, learn per-channel affine parameters.
        subtract_last: If ``True``, use the last observed value instead of the
            mean as the centering statistic. The paper's ``subtract_last``
            ablation; useful when the series is highly trended.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if affine:
            # Per-channel affine parameters [C]
            self.affine_weight = nn.Parameter(torch.ones(num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))

        # Populated during ``normalize`` and reused during ``denormalize``.
        self._mean: Optional[torch.Tensor] = None
        self._stdev: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mode: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply or reverse the normalization.

        Args:
            x: Tensor of shape ``[B, T, C]`` in ``norm`` mode, or
                ``[B, T_out, C]`` in ``denorm`` mode.
            mode: Either ``'norm'`` or ``'denorm'``.
            mask: Optional ``[B, T]`` validity mask (``1`` = real,
                ``0`` = padding). Used only in ``norm`` mode to exclude
                padded steps from the statistics.
        """
        if mode == 'norm':
            self._compute_statistics(x, mask)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown RevIN mode: {mode!r}")

    def _compute_statistics(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> None:
        # x: [B, T, C]; mask: [B, T] or None
        if self.subtract_last:
            # Use the last *valid* observation per sample.
            if mask is not None:
                # Find the index of the last valid step per sample.
                # argmax on reversed mask gives the first 1 from the right.
                reversed_mask = mask.flip(dims=[1])
                # If a row is entirely padded, clamp to 0 to avoid OOB.
                has_any = (mask.sum(dim=1) > 0)
                last_idx = (mask.size(1) - 1) - reversed_mask.argmax(dim=1)
                last_idx = torch.where(has_any, last_idx, torch.zeros_like(last_idx))
                # Gather: [B, 1, C]
                batch_idx = torch.arange(x.size(0), device=x.device)
                last = x[batch_idx, last_idx].unsqueeze(1)
            else:
                last = x[:, -1:, :]
            self._mean = last  # [B, 1, C]
        else:
            if mask is not None:
                m = mask.unsqueeze(-1)  # [B, T, 1]
                denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
                self._mean = (x * m).sum(dim=1, keepdim=True) / denom
            else:
                self._mean = x.mean(dim=1, keepdim=True)

        if mask is not None:
            m = mask.unsqueeze(-1)  # [B, T, 1]
            denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
            var = ((x - self._mean) ** 2 * m).sum(dim=1, keepdim=True) / denom
        else:
            var = x.var(dim=1, keepdim=True, unbiased=False)
        self._stdev = torch.sqrt(var + self.eps)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_out, C_sub] where C_sub <= num_channels (target subset).
        if self.affine:
            # Use only the slice of affine params matching the output channels.
            C_out = x.size(-1)
            x = (x - self.affine_bias[:C_out]) / (self.affine_weight[:C_out] + self.eps)
        C_out = x.size(-1)
        # Broadcast only the matching channels of the stored statistics.
        return x * self._stdev[..., :C_out] + self._mean[..., :C_out]


# ---------------------------------------------------------------------------
# Transformer encoder block used inside PatchTST
# ---------------------------------------------------------------------------
class PatchTSTEncoderLayer(nn.Module):
    """Single encoder block: multi-head self-attention + position-wise FFN.

    Supports both ``pre_norm`` and ``post_norm`` residual arrangements. The
    paper's default is ``post_norm``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = 'gelu',
        pre_norm: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.pre_norm = pre_norm

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)

        if activation == 'gelu':
            act_layer: nn.Module = nn.GELU()
        elif activation == 'relu':
            act_layer = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation!r}")

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*, N_patches, d_model]
        if self.pre_norm:
            h = self.norm_attn(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.dropout_attn(attn_out)

            h = self.norm_ff(x)
            x = x + self.dropout_ff(self.ff(h))
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.norm_attn(x + self.dropout_attn(attn_out))
            x = self.norm_ff(x + self.dropout_ff(self.ff(x)))
        return x


class PatchTSTEncoder(nn.Module):
    """Stack of ``PatchTSTEncoderLayer`` blocks."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = 'gelu',
        pre_norm: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PatchTSTEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Flatten + Linear prediction head
# ---------------------------------------------------------------------------
class FlattenHead(nn.Module):
    """Flatten patch tokens along the patch axis and project to the horizon.

    Produces ``[B, C, prediction_steps * num_outputs]`` which the parent
    module reshapes into ``[B, prediction_steps, C, num_outputs]``.
    """

    def __init__(
        self,
        num_patches: int,
        d_model: int,
        prediction_steps: int,
        num_outputs: int = 1,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.prediction_steps = prediction_steps
        self.num_outputs = num_outputs

        self.flatten = nn.Flatten(start_dim=-2)  # [..., N_patches * d_model]
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(num_patches * d_model, prediction_steps * num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N_patches, d_model]
        x = self.flatten(x)           # [B, C, N_patches * d_model]
        x = self.dropout(x)
        x = self.linear(x)            # [B, C, prediction_steps * num_outputs]
        return x


# ---------------------------------------------------------------------------
# Full PatchTST backbone (patching + encoder + head)
# ---------------------------------------------------------------------------
class PatchTSTBackbone(nn.Module):
    """Core PatchTST body: patching, projection, transformer encoder, head.

    Works on a ``[B, T, C]`` tensor and produces
    ``[B, prediction_steps, C, num_outputs]``.
    """

    def __init__(
        self,
        historical_steps: int,
        prediction_steps: int,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = 'end',
        d_model: int = 128,
        n_heads: int = 16,
        num_encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = 'gelu',
        pre_norm: bool = False,
        head_dropout: float = 0.0,
        num_outputs: int = 1,
    ):
        super().__init__()
        self.historical_steps = historical_steps
        self.prediction_steps = prediction_steps
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.d_model = d_model
        self.num_outputs = num_outputs

        # How many patches will we get? Match the reference repo's arithmetic.
        num_patches = max(1, (historical_steps - patch_len) // stride + 1)
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            num_patches += 1
        else:
            self.padding_layer = None
        if historical_steps < patch_len and padding_patch != 'end':
            raise ValueError(
                f"historical_steps ({historical_steps}) < patch_len ({patch_len}) "
                "and padding_patch is disabled. Enable padding or shrink patch_len."
            )
        self.num_patches = num_patches

        # Patch embedding: each patch of length ``patch_len`` -> ``d_model``.
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder = PatchTSTEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            pre_norm=pre_norm,
        )

        self.head = FlattenHead(
            num_patches=num_patches,
            d_model=d_model,
            prediction_steps=prediction_steps,
            num_outputs=num_outputs,
            head_dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, C]`` multivariate series.

        Returns:
            ``[B, prediction_steps, C, num_outputs]``
        """
        B, T, C = x.shape
        # Channel-independent: move channel to batch axis.
        # [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        # Replication-pad the end so patches tile the sequence.
        if self.padding_layer is not None:
            x = self.padding_layer(x)  # [B, C, T + stride]

        # Unfold into patches: [B, C, N_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Fold channels into batch for shared-weight processing.
        # [B, C, N_patches, patch_len] -> [B*C, N_patches, patch_len]
        x = x.reshape(B * C, self.num_patches, self.patch_len)

        # Patch embedding + positional embedding.
        x = self.patch_projection(x)                       # [B*C, N, d_model]
        x = self.embedding_dropout(x + self.pos_embedding) # broadcast on batch

        # Transformer encoder.
        x = self.encoder(x)                                # [B*C, N, d_model]

        # Reshape back to [B, C, N, d_model] for the head.
        x = x.reshape(B, C, self.num_patches, self.d_model)

        # Head: [B, C, prediction_steps * num_outputs]
        x = self.head(x)

        # Reshape into [B, C, prediction_steps, num_outputs]
        x = x.reshape(B, C, self.prediction_steps, self.num_outputs)

        # Rearrange into [B, prediction_steps, C, num_outputs]
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


# ---------------------------------------------------------------------------
# Top-level model: TFT-compatible forward interface
# ---------------------------------------------------------------------------
class PatchTST(nn.Module):
    """PatchTST model wired for ``tft_pytorch``.

    The ``forward`` signature matches :class:`TemporalFusionTransformer` so
    that :class:`TFTTrainer` and :class:`TFTInference` can be used without
    modification. Static, categorical, and future inputs are accepted but
    ignored - PatchTST is a pure historical-numeric model.

    Args:
        num_historical_continuous: Number of continuous history channels
            produced by :meth:`TFTDataAdapter.adapt_for_tft`. Must include
            the ``num_targets`` target channels (which appear first in the
            list) plus any unknown/known numeric features.
        num_targets: Number of target channels to predict (the first
            ``num_targets`` entries of ``historical_continuous``).
        historical_steps: Look-back length ``T``.
        prediction_steps: Forecast horizon ``H``.
        channel_mode: ``'target_only'`` to feed only the target series into
            the backbone, or ``'all_numeric'`` to use every numeric channel
            in ``historical_continuous`` (channel-independent). In either
            mode only the first ``num_targets`` channels are returned.
        num_outputs: Number of outputs per target per horizon step. Set to
            ``len(quantiles)`` for probabilistic forecasting with
            :class:`QuantileLoss`; set to ``1`` for point forecasting.
        patch_len, stride, padding_patch: Patching configuration.
        d_model, n_heads, num_encoder_layers, d_ff: Transformer sizing.
        dropout, attn_dropout, head_dropout: Regularization.
        activation: ``'gelu'`` (default) or ``'relu'``.
        pre_norm: If ``True``, use pre-norm residual layout instead of
            post-norm (paper default is post-norm).
        use_revin, revin_affine, revin_eps, subtract_last: RevIN config.
        device: Target device.

    Forward inputs:
        ``historical_continuous`` is the only required argument. It must be a
        list of ``[B, historical_steps]`` tensors -- exactly what
        :meth:`TFTDataAdapter.adapt_for_tft` produces. The list must contain
        at least ``num_historical_continuous`` tensors when
        ``channel_mode='all_numeric'``, or at least ``num_targets`` when
        ``channel_mode='target_only'``.

    Forward output:
        ``dict`` with:
            - ``'predictions'``: ``[B, prediction_steps, num_outputs]`` when
              ``num_targets == 1``, else
              ``[B, prediction_steps, num_targets * num_outputs]``. This
              matches the convention used by
              :class:`TemporalFusionTransformer`.
    """

    def __init__(
        self,
        # Required input / output sizing
        num_historical_continuous: int,
        num_targets: int = 1,
        historical_steps: int = 96,
        prediction_steps: int = 24,

        # Channel handling
        channel_mode: str = 'target_only',

        # Output
        num_outputs: int = 1,

        # Patching
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = 'end',

        # Transformer
        d_model: int = 128,
        n_heads: int = 16,
        num_encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = 'gelu',
        pre_norm: bool = False,

        # Head
        head_dropout: float = 0.0,

        # RevIN
        use_revin: bool = True,
        revin_affine: bool = True,
        revin_eps: float = 1e-5,
        subtract_last: bool = False,

        # Device
        device: str = 'cpu',
    ):
        super().__init__()
        if num_targets < 1:
            raise ValueError("num_targets must be >= 1")
        if num_targets > num_historical_continuous:
            raise ValueError(
                f"num_targets ({num_targets}) cannot exceed "
                f"num_historical_continuous ({num_historical_continuous})"
            )
        if channel_mode not in ('target_only', 'all_numeric'):
            raise ValueError(
                f"channel_mode must be 'target_only' or 'all_numeric', "
                f"got {channel_mode!r}"
            )

        self.device = device
        self.num_historical_continuous = num_historical_continuous
        self.num_targets = num_targets
        self.historical_steps = historical_steps
        self.prediction_steps = prediction_steps
        self.channel_mode = channel_mode
        self.num_outputs = num_outputs

        # How many channels go into the backbone?
        if channel_mode == 'target_only':
            self.num_input_channels = num_targets
        else:
            self.num_input_channels = num_historical_continuous

        # RevIN
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(
                num_channels=self.num_input_channels,
                eps=revin_eps,
                affine=revin_affine,
                subtract_last=subtract_last,
            )

        # Core backbone
        self.backbone = PatchTSTBackbone(
            historical_steps=historical_steps,
            prediction_steps=prediction_steps,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            pre_norm=pre_norm,
            head_dropout=head_dropout,
            num_outputs=num_outputs,
        )

        self.to(device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        static_categorical: Optional[List[torch.Tensor]] = None,
        static_continuous: Optional[List[torch.Tensor]] = None,
        historical_categorical: Optional[List[torch.Tensor]] = None,
        historical_continuous: Optional[List[torch.Tensor]] = None,
        future_categorical: Optional[List[torch.Tensor]] = None,
        future_continuous: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """TFT-compatible forward. See class docstring for details."""
        if historical_continuous is None or len(historical_continuous) == 0:
            raise ValueError(
                "PatchTST requires 'historical_continuous' input. "
                "Ensure the OptimizedTFTDataset exposes at least one target "
                "channel via TFTDataAdapter.adapt_for_tft."
            )

        # --- 1. Select channels -----------------------------------------
        if self.channel_mode == 'target_only':
            needed = self.num_targets
        else:
            needed = self.num_historical_continuous

        if len(historical_continuous) < needed:
            raise ValueError(
                f"historical_continuous has {len(historical_continuous)} "
                f"tensors but PatchTST expects at least {needed} "
                f"(channel_mode='{self.channel_mode}')."
            )
        channels = historical_continuous[:needed]

        # Each entry: [B, T]  ->  stack to [B, T, C]
        x = torch.stack(channels, dim=-1)

        # --- 2. Resolve padding mask -----------------------------------
        # ``padding_mask`` from the adapter has shape [B, 1, seq_len] with
        # 1 = "mask out". We need [B, T_hist] with 1 = valid for RevIN.
        revin_mask: Optional[torch.Tensor] = None
        if padding_mask is not None:
            pm = padding_mask
            if pm.dim() == 3:
                # [B, 1, seq_len] -> [B, seq_len]
                pm = pm.squeeze(1)
            # If the adapter passed a full-window mask, keep only history.
            if pm.size(-1) >= self.historical_steps:
                pm = pm[:, : self.historical_steps]
            # Convert "1 = mask" -> "1 = valid"
            revin_mask = (1.0 - pm).to(x.dtype)

        # --- 3. RevIN normalize ----------------------------------------
        if self.use_revin:
            x = self.revin(x, mode='norm', mask=revin_mask)

        # --- 4. Backbone -----------------------------------------------
        # [B, T, C] -> [B, prediction_steps, C, num_outputs]
        y = self.backbone(x)

        # --- 5. Keep only target channels ------------------------------
        # If we fed extra exogenous channels, drop them here.
        y = y[:, :, : self.num_targets, :]

        # --- 6. RevIN denormalize (target channels only) ---------------
        if self.use_revin:
            # Merge the num_outputs axis for denorm, then split again.
            B = y.size(0)
            H = self.prediction_steps
            T_tgt = self.num_targets
            O = self.num_outputs
            # [B, H, T_tgt, O] -> [B, H, T_tgt] for each output independently.
            # RevIN stats were stored per-channel; we apply them per-channel
            # for every output slice.
            y_out = torch.empty_like(y)
            for o in range(O):
                y_out[..., o] = self.revin(y[..., o], mode='denorm')
            y = y_out

        # --- 7. Collapse to [B, H, num_targets * num_outputs] ----------
        B, H, T_tgt, O = y.shape
        predictions = y.reshape(B, H, T_tgt * O)

        return {'predictions': predictions}

    # ------------------------------------------------------------------
    # Convenience APIs matching the rest of the library
    # ------------------------------------------------------------------
    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Return only the ``'predictions'`` tensor."""
        return self.forward(*args, **kwargs)['predictions']


# ---------------------------------------------------------------------------
# Convenience factory from an ``OptimizedTFTDataset`` / ``TFTDataAdapter``
# ---------------------------------------------------------------------------
def create_patchtst_from_dataset(
    dataset,
    num_outputs: int = 1,
    channel_mode: str = 'target_only',
    device: str = 'cpu',
    **patchtst_kwargs,
) -> PatchTST:
    """Instantiate a :class:`PatchTST` sized for an existing dataset.

    The helper reads ``historical_steps``, ``prediction_steps``, and the
    numeric-feature layout directly from the dataset, so the user only has
    to pick hyper-parameters.

    Args:
        dataset: An ``OptimizedTFTDataset`` (or any object exposing the same
            attributes: ``historical_steps``, ``prediction_steps``,
            ``target_cols``, ``temporal_unknown_numeric_cols``,
            ``temporal_known_numeric_cols``).
        num_outputs: Output dimension (e.g. number of quantiles).
        channel_mode: See :class:`PatchTST`.
        device: Target device.
        **patchtst_kwargs: Any extra arguments forwarded to
            :class:`PatchTST` (e.g. ``patch_len``, ``d_model``).

    Returns:
        A ready-to-train ``PatchTST`` instance.
    """
    num_targets = len(dataset.target_cols)
    num_historical_continuous = (
        num_targets
        + len(dataset.temporal_unknown_numeric_cols)
        + len(dataset.temporal_known_numeric_cols)
    )

    return PatchTST(
        num_historical_continuous=num_historical_continuous,
        num_targets=num_targets,
        historical_steps=dataset.historical_steps,
        prediction_steps=dataset.prediction_steps,
        channel_mode=channel_mode,
        num_outputs=num_outputs,
        device=device,
        **patchtst_kwargs,
    )


# ===========================================================================
#                              PatchTSTPlus
# ===========================================================================
# Extended PatchTST with full TFT-style feature support: static features
# (numeric + categorical), historical categoricals, future-known features
# (numeric + categorical). Built for global demand forecasting workloads
# where entity identity and forward-looking signals are non-negotiable.
# ===========================================================================


def _safe_embedding_lookup(x: torch.Tensor, embed_layer: nn.Embedding) -> torch.Tensor:
    """Embed categorical indices with safe handling of -1 / overflow.

    Convention (matches :class:`TemporalFusionTransformer`): vocabulary size
    is ``K + 2`` where the last index ``vocab_size - 1`` is reserved as the
    unknown / padding slot. Values of ``-1`` (the dataset's
    ``categorical_padding_value``) and any out-of-range index are remapped
    to that slot before lookup.
    """
    vocab_size = embed_layer.num_embeddings
    x_safe = x.clone()
    x_safe[x_safe == -1] = vocab_size - 1
    x_safe = torch.clamp(x_safe, min=0, max=vocab_size - 1)
    return embed_layer(x_safe)


class CategoricalEmbeddingBank(nn.Module):
    """Holds a bank of ``nn.Embedding`` layers and concatenates their outputs.

    Args:
        embedding_specs: List of ``(vocab_size, embed_dim)`` tuples, one per
            categorical variable. Order matters and is preserved in the
            concatenation.
    """

    def __init__(self, embedding_specs: List[Tuple[int, int]]):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embed_dim) for vocab_size, embed_dim in embedding_specs]
        )
        self.embed_dims = [d for _, d in embedding_specs]
        self.total_embed_dim = int(sum(self.embed_dims))
        self.num_vars = len(embedding_specs)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Embed each input and concatenate along the last dimension.

        Args:
            inputs: ``num_vars`` LongTensors of arbitrary (matching) shape.

        Returns:
            Tensor with the same leading shape and a final dimension of
            size ``total_embed_dim``.
        """
        if len(inputs) != self.num_vars:
            raise ValueError(
                f"CategoricalEmbeddingBank expected {self.num_vars} inputs, "
                f"got {len(inputs)}"
            )
        embedded = [_safe_embedding_lookup(x, e) for x, e in zip(inputs, self.embeddings)]
        return torch.cat(embedded, dim=-1)


class StaticContextEncoder(nn.Module):
    """Encode static (per-entity) features into a single ``d_model`` vector.

    Concatenates static numerics and embedded static categoricals, then
    passes them through a 2-layer MLP. The resulting vector conditions
    every patch token in the encoder, giving the channel-independent
    backbone access to entity identity.
    """

    def __init__(
        self,
        num_static_numeric: int,
        static_cat_specs: List[Tuple[int, int]],
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_static_numeric = num_static_numeric

        if static_cat_specs:
            self.cat_bank: Optional[CategoricalEmbeddingBank] = CategoricalEmbeddingBank(
                static_cat_specs
            )
            cat_total = self.cat_bank.total_embed_dim
        else:
            self.cat_bank = None
            cat_total = 0

        input_dim = num_static_numeric + cat_total
        self.input_dim = input_dim

        if input_dim == 0:
            self.mlp: Optional[nn.Sequential] = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )

    def forward(
        self,
        static_numeric_list: List[torch.Tensor],
        static_categorical_list: List[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.mlp is None:
            return None

        parts: List[torch.Tensor] = []

        if self.num_static_numeric > 0:
            if not static_numeric_list:
                raise ValueError(
                    f"StaticContextEncoder configured for {self.num_static_numeric} "
                    "numeric features but received an empty list."
                )
            # Each item is [B, 1]; concat to [B, num_static_numeric].
            parts.append(torch.cat(static_numeric_list, dim=-1))

        if self.cat_bank is not None:
            if not static_categorical_list:
                raise ValueError(
                    "StaticContextEncoder configured with categorical embeddings "
                    "but received an empty static_categorical_list."
                )
            # Each input is [B] -> embedded to [B, embed_dim] -> concat to [B, sum_embed].
            parts.append(self.cat_bank(static_categorical_list))

        x = torch.cat(parts, dim=-1)  # [B, input_dim]
        return self.mlp(x)              # [B, d_model]


class TemporalCategoricalEncoder(nn.Module):
    """Embed a list of per-timestep categoricals into a temporal context tensor.

    Output shape ``[B, T, d_model]`` is later pooled per patch (or per
    horizon step) and added into the encoder / head.
    """

    def __init__(
        self,
        cat_specs: List[Tuple[int, int]],
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_vars = len(cat_specs)

        if cat_specs:
            self.cat_bank: Optional[CategoricalEmbeddingBank] = CategoricalEmbeddingBank(cat_specs)
            self.projection: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.cat_bank.total_embed_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.cat_bank = None
            self.projection = None

    def forward(self, cat_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.cat_bank is None:
            return None
        if not cat_list:
            raise ValueError(
                "TemporalCategoricalEncoder configured with categoricals but "
                "received an empty cat_list."
            )
        embedded = self.cat_bank(cat_list)  # [B, T, total_embed]
        return self.projection(embedded)    # [B, T, d_model]


class FutureFeatureEncoder(nn.Module):
    """Encode future-known features (numeric + categorical) per timestep.

    Produces ``[B, prediction_steps, d_model]`` which the head fuses with
    the encoder output. This is how PatchTSTPlus learns to use planned
    promotions, upcoming holidays, weather forecasts, etc.
    """

    def __init__(
        self,
        num_future_numeric: int,
        future_cat_specs: List[Tuple[int, int]],
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_future_numeric = num_future_numeric

        if future_cat_specs:
            self.cat_bank: Optional[CategoricalEmbeddingBank] = CategoricalEmbeddingBank(
                future_cat_specs
            )
            cat_total = self.cat_bank.total_embed_dim
        else:
            self.cat_bank = None
            cat_total = 0

        input_dim = num_future_numeric + cat_total
        self.input_dim = input_dim

        if input_dim == 0:
            self.mlp: Optional[nn.Sequential] = None
            self.output_dim = 0
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.output_dim = d_model

    def forward(
        self,
        future_numeric_list: List[torch.Tensor],
        future_categorical_list: List[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.mlp is None:
            return None

        parts: List[torch.Tensor] = []

        if self.num_future_numeric > 0:
            if not future_numeric_list:
                raise ValueError(
                    f"FutureFeatureEncoder configured for {self.num_future_numeric} "
                    "numeric features but received an empty list."
                )
            # Each [B, T_pred] -> stack to [B, T_pred, n_future_num].
            parts.append(torch.stack(future_numeric_list, dim=-1))

        if self.cat_bank is not None:
            if not future_categorical_list:
                raise ValueError(
                    "FutureFeatureEncoder configured with categoricals but "
                    "received an empty future_categorical_list."
                )
            # Each [B, T_pred] -> embed to [B, T_pred, embed_dim] -> concat.
            parts.append(self.cat_bank(future_categorical_list))

        x = torch.cat(parts, dim=-1)  # [B, T_pred, input_dim]
        return self.mlp(x)             # [B, T_pred, d_model]


class PatchTSTPlusHead(nn.Module):
    """Prediction head fusing encoder output with future-known features.

    The encoder output ``[B, C, num_patches, d_model]`` is flattened to
    ``[B, C, num_patches * d_model]``. Future features ``[B, T_pred, d_future]``
    are flattened to ``[B, T_pred * d_future]`` and tiled across channels,
    giving a head input of ``[B, C, num_patches * d_model + T_pred * d_future]``.
    A single linear projection produces ``[B, C, T_pred * num_outputs]``.

    Doing the fusion at the head (rather than via a cross-attention decoder)
    keeps the parameter count and latency competitive with vanilla PatchTST
    while still letting the model learn rich interactions between the
    encoded history and forward-looking signals.
    """

    def __init__(
        self,
        num_patches: int,
        d_model: int,
        prediction_steps: int,
        num_outputs: int = 1,
        future_dim: int = 0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.prediction_steps = prediction_steps
        self.num_outputs = num_outputs
        self.future_dim = future_dim

        self.encoder_flat_dim = num_patches * d_model
        self.future_flat_dim = future_dim * prediction_steps
        self.head_input_dim = self.encoder_flat_dim + self.future_flat_dim

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(self.head_input_dim, prediction_steps * num_outputs)

    def forward(
        self,
        encoder_output: torch.Tensor,
        future_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # encoder_output: [B, C, num_patches, d_model]
        # future_features: [B, T_pred, future_dim] or None
        x = self.flatten(encoder_output)  # [B, C, num_patches * d_model]

        if self.future_dim > 0:
            if future_features is None:
                raise RuntimeError(
                    "PatchTSTPlusHead was configured with future_dim > 0 but "
                    "received future_features=None at runtime."
                )
            B, C = x.size(0), x.size(1)
            future_flat = future_features.reshape(B, -1)             # [B, T_pred * d_future]
            future_tiled = future_flat.unsqueeze(1).expand(B, C, -1) # [B, C, T_pred * d_future]
            x = torch.cat([x, future_tiled], dim=-1)

        x = self.dropout(x)
        x = self.linear(x)  # [B, C, prediction_steps * num_outputs]
        return x


class PatchTSTPlus(nn.Module):
    """PatchTST with full TFT-style feature support.

    Extends :class:`PatchTST` with three principled mechanisms that preserve
    the channel-independent backbone:

    1. **Static context** - Numeric and categorical static features are
       fused into a single ``d_model`` vector and added (broadcast) to every
       patch token before the encoder. This is how the model learns "which
       series am I predicting" - critical for global models that share
       weights across thousands of entities.
    2. **Historical categorical conditioning** - Each historical
       categorical is embedded, the per-timestep embeddings are mean (or
       last-value) pooled per patch, projected to ``d_model``, and added to
       all numeric channels' patch tokens. Captures temporal categorical
       signals like day-of-week or past promotion state.
    3. **Future-known fusion at the head** - Future numeric and categorical
       features are encoded per horizon step and concatenated to the
       flattened encoder output before the prediction projection. Lets the
       model exploit planned promotions, upcoming holidays, weather
       forecasts, and the like.

    The model accepts the exact same ``forward`` keyword arguments as
    :class:`TemporalFusionTransformer`, so :class:`TFTTrainer`,
    :class:`TFTInference`, and :class:`TFTInferenceWithTracking` work
    without modification.

    Args:
        num_historical_continuous: Total number of continuous channels in
            ``historical_continuous`` (targets + unknown numeric + known
            numeric, in the order produced by
            :meth:`TFTDataAdapter.adapt_for_tft`).
        num_targets: Number of target channels (the first ``num_targets``
            entries of ``historical_continuous``).
        num_static_continuous: Number of static numeric features.
        num_future_continuous: Number of future-known numeric features.
        categorical_embedding_dims: Dictionary mapping
            ``'static_cat_{i}'``, ``'historical_cat_{i}'``,
            ``'future_cat_{i}'`` to ``(vocab_size, embed_dim)`` tuples.
            The same format produced by
            :func:`tft_pytorch.create_uniform_embedding_dims`.
        num_static_categorical: Number of static categorical features.
        num_historical_categorical: Number of historical categorical
            features (= ``len(temporal_unknown_categorical_cols) +
            len(temporal_known_categorical_cols)``).
        num_future_categorical: Number of future-known categorical
            features (= ``len(temporal_known_categorical_cols)``).
        historical_steps, prediction_steps: Window dimensions.
        channel_mode: ``'target_only'`` or ``'all_numeric'`` (see
            :class:`PatchTST`).
        num_outputs: Output dimension per target per horizon step.
        patch_len, stride, padding_patch: Patching configuration.
        d_model, n_heads, num_encoder_layers, d_ff: Transformer sizing.
        dropout, attn_dropout, head_dropout: Regularization.
        activation: ``'gelu'`` or ``'relu'``.
        pre_norm: Use pre-norm residual layout if ``True``.
        use_revin, revin_affine, revin_eps, subtract_last: RevIN config.
        cat_pool: How to pool per-timestep categorical embeddings into
            per-patch representations: ``'mean'`` (default) or ``'last'``.
        device: Target device.

    Forward output:
        ``dict`` with ``'predictions'`` of shape
        ``[B, prediction_steps, num_targets * num_outputs]``.
    """

    def __init__(
        self,
        # Numeric input sizing
        num_historical_continuous: int,
        num_targets: int = 1,
        num_static_continuous: int = 0,
        num_future_continuous: int = 0,

        # Categorical input sizing + embeddings
        categorical_embedding_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        num_static_categorical: int = 0,
        num_historical_categorical: int = 0,
        num_future_categorical: int = 0,

        # Time
        historical_steps: int = 96,
        prediction_steps: int = 24,

        # Channel handling
        channel_mode: str = 'target_only',

        # Output
        num_outputs: int = 1,

        # Patching
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = 'end',

        # Transformer
        d_model: int = 128,
        n_heads: int = 16,
        num_encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = 'gelu',
        pre_norm: bool = False,

        # Head
        head_dropout: float = 0.0,

        # RevIN
        use_revin: bool = True,
        revin_affine: bool = True,
        revin_eps: float = 1e-5,
        subtract_last: bool = False,

        # Categorical pooling for historical context
        cat_pool: str = 'mean',

        # Device
        device: str = 'cpu',
    ):
        super().__init__()

        # ---- Validation ------------------------------------------------
        if num_targets < 1:
            raise ValueError("num_targets must be >= 1")
        if num_targets > num_historical_continuous:
            raise ValueError(
                f"num_targets ({num_targets}) cannot exceed "
                f"num_historical_continuous ({num_historical_continuous})"
            )
        if channel_mode not in ('target_only', 'all_numeric'):
            raise ValueError(
                f"channel_mode must be 'target_only' or 'all_numeric', "
                f"got {channel_mode!r}"
            )
        if cat_pool not in ('mean', 'last'):
            raise ValueError(f"cat_pool must be 'mean' or 'last', got {cat_pool!r}")

        # ---- Parse categorical embedding specs -------------------------
        ed = categorical_embedding_dims or {}

        def _collect(prefix: str, n: int) -> List[Tuple[int, int]]:
            specs: List[Tuple[int, int]] = []
            for i in range(n):
                key = f"{prefix}_cat_{i}"
                if key not in ed:
                    raise ValueError(
                        f"categorical_embedding_dims is missing required key "
                        f"{key!r}. Use create_uniform_embedding_dims(dataset, "
                        f"hidden_layer_size=...) to build the dict automatically."
                    )
                specs.append(tuple(ed[key]))
            return specs

        static_cat_specs = _collect('static', num_static_categorical)
        historical_cat_specs = _collect('historical', num_historical_categorical)
        future_cat_specs = _collect('future', num_future_categorical)

        # ---- Save attrs ------------------------------------------------
        self.device = device
        self.num_historical_continuous = num_historical_continuous
        self.num_targets = num_targets
        self.num_static_continuous = num_static_continuous
        self.num_future_continuous = num_future_continuous
        self.num_static_categorical = num_static_categorical
        self.num_historical_categorical = num_historical_categorical
        self.num_future_categorical = num_future_categorical
        self.historical_steps = historical_steps
        self.prediction_steps = prediction_steps
        self.num_outputs = num_outputs
        self.channel_mode = channel_mode
        self.cat_pool = cat_pool
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # ---- Channel count for the backbone ---------------------------
        if channel_mode == 'target_only':
            self.num_input_channels = num_targets
        else:
            self.num_input_channels = num_historical_continuous

        # ---- RevIN (operates on numeric channels) ---------------------
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(
                num_channels=self.num_input_channels,
                eps=revin_eps,
                affine=revin_affine,
                subtract_last=subtract_last,
            )

        # ---- Static encoder -------------------------------------------
        self.static_encoder = StaticContextEncoder(
            num_static_numeric=num_static_continuous,
            static_cat_specs=static_cat_specs,
            d_model=d_model,
            dropout=dropout,
        )

        # ---- Historical categorical encoder ---------------------------
        self.hist_cat_encoder = TemporalCategoricalEncoder(
            cat_specs=historical_cat_specs,
            d_model=d_model,
            dropout=dropout,
        )

        # ---- Patching geometry (mirror PatchTSTBackbone) --------------
        num_patches = max(1, (historical_steps - patch_len) // stride + 1)
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            num_patches += 1
        else:
            self.padding_layer = None
            if historical_steps < patch_len:
                raise ValueError(
                    f"historical_steps ({historical_steps}) < patch_len "
                    f"({patch_len}) and padding_patch is disabled."
                )
        self.num_patches = num_patches

        # ---- Patch embedding + positional embedding -------------------
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)

        # ---- Transformer encoder --------------------------------------
        self.encoder = PatchTSTEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            pre_norm=pre_norm,
        )

        # ---- Future feature encoder -----------------------------------
        self.future_encoder = FutureFeatureEncoder(
            num_future_numeric=num_future_continuous,
            future_cat_specs=future_cat_specs,
            d_model=d_model,
            dropout=dropout,
        )
        future_dim = self.future_encoder.output_dim

        # ---- Prediction head ------------------------------------------
        self.head = PatchTSTPlusHead(
            num_patches=num_patches,
            d_model=d_model,
            prediction_steps=prediction_steps,
            num_outputs=num_outputs,
            future_dim=future_dim,
            head_dropout=head_dropout,
        )

        self.to(device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pool_temporal_per_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a ``[B, T_hist, D]`` tensor onto the patch grid.

        Applies the same end-padding + unfold geometry used for numeric
        channels, so per-patch categorical context aligns 1:1 with each
        numeric patch token. Returns ``[B, num_patches, D]``.
        """
        B, T, D = x.shape
        # [B, T, D] -> [B, D, T] for ReplicationPad1d / unfold along time.
        x = x.permute(0, 2, 1)
        if self.padding_layer is not None:
            x = self.padding_layer(x)  # [B, D, T + stride]
        # [B, D, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if self.cat_pool == 'mean':
            x = x.mean(dim=-1)
        else:  # 'last'
            x = x[..., -1]
        # [B, D, num_patches] -> [B, num_patches, D]
        return x.permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        static_categorical: Optional[List[torch.Tensor]] = None,
        static_continuous: Optional[List[torch.Tensor]] = None,
        historical_categorical: Optional[List[torch.Tensor]] = None,
        historical_continuous: Optional[List[torch.Tensor]] = None,
        future_categorical: Optional[List[torch.Tensor]] = None,
        future_continuous: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """TFT-compatible forward. See class docstring for details."""

        # ---- 1. Required: historical_continuous -----------------------
        if historical_continuous is None or len(historical_continuous) == 0:
            raise ValueError(
                "PatchTSTPlus requires 'historical_continuous'. Use "
                "TFTDataAdapter.adapt_for_tft to produce it from your dataset."
            )

        # ---- 2. Select numeric channels for the backbone -------------
        if self.channel_mode == 'target_only':
            needed = self.num_targets
        else:
            needed = self.num_historical_continuous

        if len(historical_continuous) < needed:
            raise ValueError(
                f"historical_continuous has {len(historical_continuous)} tensors "
                f"but PatchTSTPlus expects at least {needed} "
                f"(channel_mode='{self.channel_mode}')."
            )

        channels = historical_continuous[:needed]
        x = torch.stack(channels, dim=-1)  # [B, T_hist, C]
        B, _, C = x.shape

        # ---- 3. Resolve padding mask for RevIN -----------------------
        revin_mask: Optional[torch.Tensor] = None
        if padding_mask is not None:
            pm = padding_mask
            if pm.dim() == 3:
                pm = pm.squeeze(1)
            if pm.size(-1) >= self.historical_steps:
                pm = pm[:, : self.historical_steps]
            revin_mask = (1.0 - pm).to(x.dtype)

        # ---- 4. RevIN normalize --------------------------------------
        if self.use_revin:
            x = self.revin(x, mode='norm', mask=revin_mask)

        # ---- 5. Patching ---------------------------------------------
        # [B, T_hist, C] -> [B, C, T_hist]
        x = x.permute(0, 2, 1)
        if self.padding_layer is not None:
            x = self.padding_layer(x)
        # Unfold: [B, C, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Fold channels into batch: [B*C, num_patches, patch_len]
        x = x.reshape(B * C, self.num_patches, self.patch_len)

        # ---- 6. Patch embedding + positional embedding ---------------
        x = self.patch_projection(x)             # [B*C, N, d_model]
        x = x + self.pos_embedding               # broadcast over batch axis

        # ---- 7. Static context conditioning --------------------------
        if self.static_encoder.mlp is not None:
            static_ctx = self.static_encoder(
                static_continuous or [],
                static_categorical or [],
            )  # [B, d_model]
            # Add to every patch token, broadcast over channels and patches.
            # [B, d_model] -> [B, C, d_model] -> [B*C, 1, d_model]
            static_expanded = (
                static_ctx.unsqueeze(1)
                .expand(B, C, self.d_model)
                .reshape(B * C, 1, self.d_model)
            )
            x = x + static_expanded

        # ---- 8. Historical categorical conditioning ------------------
        if self.hist_cat_encoder.cat_bank is not None:
            hist_cat_emb = self.hist_cat_encoder(historical_categorical or [])
            # hist_cat_emb: [B, T_hist, d_model]
            hist_cat_per_patch = self._pool_temporal_per_patch(hist_cat_emb)
            # [B, num_patches, d_model] -> [B, C, num_patches, d_model] -> [B*C, num_patches, d_model]
            hist_cat_tiled = (
                hist_cat_per_patch.unsqueeze(1)
                .expand(B, C, self.num_patches, self.d_model)
                .reshape(B * C, self.num_patches, self.d_model)
            )
            x = x + hist_cat_tiled

        # ---- 9. Embedding dropout ------------------------------------
        x = self.embedding_dropout(x)

        # ---- 10. Transformer encoder ---------------------------------
        x = self.encoder(x)                        # [B*C, N, d_model]
        x = x.reshape(B, C, self.num_patches, self.d_model)

        # ---- 11. Future feature encoding -----------------------------
        future_feat: Optional[torch.Tensor] = None
        if self.future_encoder.mlp is not None:
            future_feat = self.future_encoder(
                future_continuous or [],
                future_categorical or [],
            )  # [B, T_pred, d_model]

        # ---- 12. Head with future fusion -----------------------------
        y = self.head(x, future_feat)              # [B, C, T_pred * num_outputs]
        y = y.reshape(B, C, self.prediction_steps, self.num_outputs)

        # ---- 13. Keep only target channels ---------------------------
        y = y[:, : self.num_targets, :, :]         # [B, T_tgt, T_pred, num_outputs]

        # ---- 14. Permute for RevIN denorm ----------------------------
        y = y.permute(0, 2, 1, 3).contiguous()     # [B, T_pred, T_tgt, num_outputs]

        # ---- 15. RevIN denormalize per output ------------------------
        if self.use_revin:
            y_out = torch.empty_like(y)
            for o in range(self.num_outputs):
                y_out[..., o] = self.revin(y[..., o], mode='denorm')
            y = y_out

        # ---- 16. Collapse to [B, T_pred, T_tgt * num_outputs] --------
        B_, H, T_tgt, O = y.shape
        predictions = y.reshape(B_, H, T_tgt * O)

        return {'predictions': predictions}

    # ------------------------------------------------------------------
    # Convenience APIs
    # ------------------------------------------------------------------
    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Return only the ``'predictions'`` tensor."""
        return self.forward(*args, **kwargs)['predictions']


# ---------------------------------------------------------------------------
# Convenience factory for PatchTSTPlus
# ---------------------------------------------------------------------------
def create_patchtst_plus_from_dataset(
    dataset,
    categorical_embedding_dims: Optional[Dict[str, Tuple[int, int]]] = None,
    num_outputs: int = 1,
    channel_mode: str = 'target_only',
    device: str = 'cpu',
    **patchtst_kwargs,
) -> 'PatchTSTPlus':
    """Build a :class:`PatchTSTPlus` sized for an existing dataset.

    Reads ``historical_steps``, ``prediction_steps``, and the full feature
    layout (numeric + categorical, static + temporal, known + unknown)
    directly from the dataset, so the user only picks hyper-parameters.

    If ``categorical_embedding_dims`` is not supplied, it is built from the
    dataset using :func:`create_uniform_embedding_dims` with the model's
    ``d_model`` as the embedding width.

    Args:
        dataset: An ``OptimizedTFTDataset``.
        categorical_embedding_dims: Optional pre-built embedding dim dict.
        num_outputs: Output dimension (e.g. number of quantiles).
        channel_mode: See :class:`PatchTST`.
        device: Target device.
        **patchtst_kwargs: Extra arguments forwarded to
            :class:`PatchTSTPlus` (``patch_len``, ``d_model``, etc.).
    """
    # Numeric counts
    num_targets = len(dataset.target_cols)
    num_historical_continuous = (
        num_targets
        + len(dataset.temporal_unknown_numeric_cols)
        + len(dataset.temporal_known_numeric_cols)
    )
    num_static_continuous = len(dataset.static_numeric_cols)
    num_future_continuous = len(dataset.temporal_known_numeric_cols)

    # Categorical counts -- mirror exactly what TFTDataAdapter.adapt_for_tft emits.
    num_static_categorical = len(dataset.static_categorical_cols)
    num_historical_categorical = (
        len(dataset.temporal_unknown_categorical_cols)
        + len(dataset.temporal_known_categorical_cols)
    )
    num_future_categorical = len(dataset.temporal_known_categorical_cols)

    # Auto-build embedding dims if missing.
    if categorical_embedding_dims is None:
        from .dataset import create_uniform_embedding_dims  # lazy import
        d_model = patchtst_kwargs.get('d_model', 128)
        categorical_embedding_dims = create_uniform_embedding_dims(
            dataset, hidden_layer_size=d_model
        )

    return PatchTSTPlus(
        num_historical_continuous=num_historical_continuous,
        num_targets=num_targets,
        num_static_continuous=num_static_continuous,
        num_future_continuous=num_future_continuous,
        categorical_embedding_dims=categorical_embedding_dims,
        num_static_categorical=num_static_categorical,
        num_historical_categorical=num_historical_categorical,
        num_future_categorical=num_future_categorical,
        historical_steps=dataset.historical_steps,
        prediction_steps=dataset.prediction_steps,
        channel_mode=channel_mode,
        num_outputs=num_outputs,
        device=device,
        **patchtst_kwargs,
    )


# ===========================================================================
#                    Classification variants
# ===========================================================================
# PatchTSTClassifier       -- paper-faithful, historical-numeric only
# PatchTSTPlusClassifier   -- full TFT-style feature support
#
# Both adapt the channel-independent transformer backbone to classification
# via a padding-aware mean-pooling head. Static, historical-categorical,
# and future-known features enter through the same three mechanisms as in
# PatchTSTPlus, preserving the architectural rationale.
# ===========================================================================


class PatchTSTClassificationHead(nn.Module):
    """Padding-aware mean-pooling classification head.

    Pools encoder output over patches (ignoring patches that fall entirely
    inside padded regions), then over channels, then runs a linear
    classifier. Optionally concatenates a pooled future-feature context
    vector before the classifier.

    Args:
        d_model: Encoder token dimension.
        num_classes: Number of output classes.
        future_dim: Dimension of the optional future-feature context
            vector (``0`` to disable).
        head_dropout: Dropout applied before the classifier linear.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        future_dim: int = 0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.future_dim = future_dim
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model + future_dim, num_classes)

    def forward(
        self,
        encoder_output: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        future_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the head.

        Args:
            encoder_output: ``[B, C, num_patches, d_model]``.
            patch_mask: ``[B, num_patches]`` with 1 for valid patches, 0
                for patches that are entirely inside the padded region.
                Passing ``None`` treats every patch as valid.
            future_context: Optional ``[B, future_dim]`` vector to
                concatenate before the classifier.

        Returns:
            Logits of shape ``[B, num_classes]``.
        """
        # encoder_output: [B, C, N, D]
        if patch_mask is not None:
            # [B, 1, N, 1] -> broadcast over C and D
            mask = patch_mask.unsqueeze(1).unsqueeze(-1).to(encoder_output.dtype)
            weighted = encoder_output * mask
            denom = mask.sum(dim=2).clamp(min=1.0)                  # [B, 1, 1]
            pooled_over_patches = weighted.sum(dim=2) / denom        # [B, C, D]
        else:
            pooled_over_patches = encoder_output.mean(dim=2)         # [B, C, D]

        pooled = pooled_over_patches.mean(dim=1)                     # [B, D]

        if self.future_dim > 0:
            if future_context is None:
                raise RuntimeError(
                    "Classification head configured with future_dim > 0 "
                    "but received future_context=None at runtime."
                )
            pooled = torch.cat([pooled, future_context], dim=-1)

        return self.linear(self.dropout(pooled))


def _derive_patch_mask(
    padding_mask: Optional[torch.Tensor],
    historical_steps: int,
    padding_layer: Optional[nn.Module],
    patch_len: int,
    stride: int,
) -> Optional[torch.Tensor]:
    """Convert a per-step padding mask into a per-patch validity mask.

    A patch is treated as *valid* if it contains at least one non-padded
    timestep. Uses the same replication-padding + unfold geometry as the
    numeric patching so the result aligns 1:1 with the encoder's patch
    tokens.

    Args:
        padding_mask: ``[B, 1, total_steps]`` or ``[B, total_steps]`` with
            1 = padded, 0 = valid, or ``None``.
        historical_steps: Size of the historical window the model consumes.
        padding_layer: Replication-pad module applied to the raw series
            before unfolding, or ``None`` when ``padding_patch`` is off.
        patch_len, stride: Patch geometry.

    Returns:
        ``[B, num_patches]`` validity mask (1 = valid, 0 = all-padded), or
        ``None`` if ``padding_mask`` was ``None``.
    """
    if padding_mask is None:
        return None
    pm = padding_mask
    if pm.dim() == 3:
        pm = pm.squeeze(1)
    if pm.size(-1) >= historical_steps:
        pm = pm[:, :historical_steps]
    # [B, T_hist] -> [B, 1, T_hist], flip to validity (1 = valid).
    valid = (1.0 - pm).unsqueeze(1).to(torch.float32)
    if padding_layer is not None:
        valid = padding_layer(valid)
    # [B, 1, num_patches, patch_len]
    valid = valid.unfold(dimension=-1, size=patch_len, step=stride)
    patch_valid = (valid.sum(dim=-1) > 0).to(torch.float32).squeeze(1)
    return patch_valid


class PatchTSTClassifier(nn.Module):
    """PatchTST adapted for time series classification (paper-faithful).

    Uses historical numeric channels only -- no static, categorical, or
    future-known features. Good for standard UCR/UEA-style benchmarks,
    classification from raw sensor / price / demand streams, and as a
    baseline to compare :class:`PatchTSTPlusClassifier` against.

    Args:
        num_historical_continuous: Number of channels in
            ``historical_continuous`` (targets + any exog numerics in
            ``'all_numeric'`` mode).
        num_targets: Number of target channels (the first ``num_targets``
            entries of ``historical_continuous``). Used for channel
            selection when ``channel_mode='target_only'``.
        num_classes: Number of output classes.
        historical_steps: Lookback window length.
        channel_mode: ``'target_only'`` or ``'all_numeric'`` (see
            :class:`PatchTST`). For classification, ``'all_numeric'`` is
            the usual choice -- exogenous series often carry strong class
            signal.
        patch_len, stride, padding_patch: Patching configuration.
        d_model, n_heads, num_encoder_layers, d_ff: Transformer sizing.
        dropout, attn_dropout, head_dropout: Regularization.
        activation: ``'gelu'`` or ``'relu'``.
        pre_norm: Pre-norm residual layout when ``True``.
        use_revin: Apply RevIN normalization to inputs. For classification
            this is sometimes **harmful** (mean / variance may *be* part
            of the class signal); benchmark both.
        revin_affine, revin_eps, subtract_last: RevIN configuration.
        device: Target device.

    Forward output:
        ``dict`` with:
            * ``'logits'`` -- ``[B, num_classes]``
            * ``'predictions'`` -- alias of logits, for API symmetry with
              :class:`PatchTST`.
    """

    def __init__(
        self,
        num_historical_continuous: int,
        num_classes: int,
        num_targets: int = 1,
        historical_steps: int = 96,
        channel_mode: str = 'all_numeric',
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = 'end',
        d_model: int = 128,
        n_heads: int = 16,
        num_encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = 'gelu',
        pre_norm: bool = False,
        head_dropout: float = 0.0,
        use_revin: bool = True,
        revin_affine: bool = True,
        revin_eps: float = 1e-5,
        subtract_last: bool = False,
        device: str = 'cpu',
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2 (use BCE with num_classes=2 for "
                f"binary), got {num_classes}."
            )
        if num_targets < 1:
            raise ValueError("num_targets must be >= 1")
        if num_targets > num_historical_continuous:
            raise ValueError(
                f"num_targets ({num_targets}) cannot exceed "
                f"num_historical_continuous ({num_historical_continuous})"
            )
        if channel_mode not in ('target_only', 'all_numeric'):
            raise ValueError(
                f"channel_mode must be 'target_only' or 'all_numeric', "
                f"got {channel_mode!r}"
            )

        self.device = device
        self.num_historical_continuous = num_historical_continuous
        self.num_targets = num_targets
        self.num_classes = num_classes
        self.historical_steps = historical_steps
        self.channel_mode = channel_mode
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.use_revin = use_revin

        # Channel count for the backbone
        if channel_mode == 'target_only':
            self.num_input_channels = num_targets
        else:
            self.num_input_channels = num_historical_continuous

        # RevIN
        if use_revin:
            self.revin = RevIN(
                num_channels=self.num_input_channels,
                eps=revin_eps,
                affine=revin_affine,
                subtract_last=subtract_last,
            )

        # Patching geometry (mirror PatchTSTBackbone)
        num_patches = max(1, (historical_steps - patch_len) // stride + 1)
        if padding_patch == 'end':
            self.padding_layer: Optional[nn.Module] = nn.ReplicationPad1d((0, stride))
            num_patches += 1
        else:
            self.padding_layer = None
            if historical_steps < patch_len:
                raise ValueError(
                    f"historical_steps ({historical_steps}) < patch_len "
                    f"({patch_len}) and padding_patch is disabled."
                )
        self.num_patches = num_patches

        # Patch embedding + positional embedding
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = PatchTSTEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            pre_norm=pre_norm,
        )

        # Classification head (no future features in vanilla classifier)
        self.head = PatchTSTClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            future_dim=0,
            head_dropout=head_dropout,
        )

        self.to(device)

    def forward(
        self,
        static_categorical: Optional[List[torch.Tensor]] = None,
        static_continuous: Optional[List[torch.Tensor]] = None,
        historical_categorical: Optional[List[torch.Tensor]] = None,
        historical_continuous: Optional[List[torch.Tensor]] = None,
        future_categorical: Optional[List[torch.Tensor]] = None,
        future_continuous: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """TFT-compatible forward. Non-numeric / future inputs are ignored."""
        if historical_continuous is None or len(historical_continuous) == 0:
            raise ValueError(
                "PatchTSTClassifier requires 'historical_continuous'. "
                "Use TFTDataAdapter.adapt_for_tft to produce it."
            )

        if self.channel_mode == 'target_only':
            needed = self.num_targets
        else:
            needed = self.num_historical_continuous

        if len(historical_continuous) < needed:
            raise ValueError(
                f"historical_continuous has {len(historical_continuous)} tensors "
                f"but PatchTSTClassifier expects at least {needed} "
                f"(channel_mode='{self.channel_mode}')."
            )

        x = torch.stack(historical_continuous[:needed], dim=-1)  # [B, T_hist, C]
        B, _, C = x.shape

        # RevIN normalize (no denorm in classification)
        if self.use_revin:
            revin_mask: Optional[torch.Tensor] = None
            if padding_mask is not None:
                pm = padding_mask
                if pm.dim() == 3:
                    pm = pm.squeeze(1)
                if pm.size(-1) >= self.historical_steps:
                    pm = pm[:, : self.historical_steps]
                revin_mask = (1.0 - pm).to(x.dtype)
            x = self.revin(x, mode='norm', mask=revin_mask)

        # Patching
        x = x.permute(0, 2, 1)  # [B, C, T_hist]
        if self.padding_layer is not None:
            x = self.padding_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, C, num_patches, patch_len] -> [B*C, num_patches, patch_len]
        x = x.reshape(B * C, self.num_patches, self.patch_len)

        # Patch embedding + positional + dropout
        x = self.patch_projection(x) + self.pos_embedding
        x = self.embedding_dropout(x)

        # Encoder
        x = self.encoder(x)
        x = x.reshape(B, C, self.num_patches, self.d_model)

        # Derive per-patch validity mask
        patch_mask = _derive_patch_mask(
            padding_mask,
            self.historical_steps,
            self.padding_layer,
            self.patch_len,
            self.stride,
        )

        # Classify
        logits = self.head(x, patch_mask=patch_mask)

        return {'logits': logits, 'predictions': logits}

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """Return the argmax class index."""
        return self.forward(*args, **kwargs)['logits'].argmax(dim=-1)

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Return class probabilities (softmax over the last dim)."""
        return torch.softmax(self.forward(*args, **kwargs)['logits'], dim=-1)


class PatchTSTPlusClassifier(nn.Module):
    """PatchTST classifier with full TFT-style feature support.

    Same three-mechanism feature integration as :class:`PatchTSTPlus`:

    1. Static numerics + categoricals -> single ``d_model`` vector added
       to every patch token as a global bias (entity identity).
    2. Historical categoricals -> per-timestep embedding, pooled per
       patch, added to numeric patch tokens.
    3. Future-known features -> per-step encoding, then **mean-pooled
       over the horizon** and concatenated to the pooled classification
       representation. Pooling (rather than flattening) keeps the final
       linear's input size independent of ``prediction_steps``.

    The forward signature matches :class:`TemporalFusionTransformer`, so
    the same dataloader / adapter plumbing works end to end -- you only
    swap the training loop to use classification losses (see
    ``INTEGRATION.md``).

    Args mirror :class:`PatchTSTPlus` with these differences:

    * ``num_outputs`` is replaced by ``num_classes``.
    * ``prediction_steps`` becomes optional; it is only consulted by
      ``create_patchtst_plus_classifier_from_dataset`` for API symmetry.
    * ``future_pool``: how to collapse future features along time before
      fusing into the classifier -- ``'mean'`` (default) or ``'last'``.

    Forward output:
        ``dict`` with ``'logits'`` and an alias ``'predictions'`` of
        shape ``[B, num_classes]``.
    """

    def __init__(
        self,
        # Numeric sizing
        num_historical_continuous: int,
        num_classes: int,
        num_targets: int = 1,
        num_static_continuous: int = 0,
        num_future_continuous: int = 0,

        # Categorical sizing + embeddings
        categorical_embedding_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        num_static_categorical: int = 0,
        num_historical_categorical: int = 0,
        num_future_categorical: int = 0,

        # Time
        historical_steps: int = 96,

        # Channel handling
        channel_mode: str = 'all_numeric',

        # Patching
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = 'end',

        # Transformer
        d_model: int = 128,
        n_heads: int = 16,
        num_encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = 'gelu',
        pre_norm: bool = False,

        # Head
        head_dropout: float = 0.0,

        # RevIN
        use_revin: bool = True,
        revin_affine: bool = True,
        revin_eps: float = 1e-5,
        subtract_last: bool = False,

        # Categorical pooling
        cat_pool: str = 'mean',
        future_pool: str = 'mean',

        # Device
        device: str = 'cpu',
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {num_classes}."
            )
        if num_targets < 1:
            raise ValueError("num_targets must be >= 1")
        if num_targets > num_historical_continuous:
            raise ValueError(
                f"num_targets ({num_targets}) cannot exceed "
                f"num_historical_continuous ({num_historical_continuous})"
            )
        if channel_mode not in ('target_only', 'all_numeric'):
            raise ValueError(
                f"channel_mode must be 'target_only' or 'all_numeric', "
                f"got {channel_mode!r}"
            )
        if cat_pool not in ('mean', 'last'):
            raise ValueError(f"cat_pool must be 'mean' or 'last', got {cat_pool!r}")
        if future_pool not in ('mean', 'last'):
            raise ValueError(f"future_pool must be 'mean' or 'last', got {future_pool!r}")

        # ---- Parse categorical embedding specs (same as PatchTSTPlus) --
        ed = categorical_embedding_dims or {}

        def _collect(prefix: str, n: int) -> List[Tuple[int, int]]:
            specs: List[Tuple[int, int]] = []
            for i in range(n):
                key = f"{prefix}_cat_{i}"
                if key not in ed:
                    raise ValueError(
                        f"categorical_embedding_dims is missing required key "
                        f"{key!r}. Use create_uniform_embedding_dims(dataset, "
                        f"hidden_layer_size=...) to build the dict automatically."
                    )
                specs.append(tuple(ed[key]))
            return specs

        static_cat_specs = _collect('static', num_static_categorical)
        historical_cat_specs = _collect('historical', num_historical_categorical)
        future_cat_specs = _collect('future', num_future_categorical)

        # ---- Save attrs ------------------------------------------------
        self.device = device
        self.num_historical_continuous = num_historical_continuous
        self.num_targets = num_targets
        self.num_static_continuous = num_static_continuous
        self.num_future_continuous = num_future_continuous
        self.num_static_categorical = num_static_categorical
        self.num_historical_categorical = num_historical_categorical
        self.num_future_categorical = num_future_categorical
        self.historical_steps = historical_steps
        self.num_classes = num_classes
        self.channel_mode = channel_mode
        self.cat_pool = cat_pool
        self.future_pool = future_pool
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.use_revin = use_revin

        # ---- Channel count --------------------------------------------
        if channel_mode == 'target_only':
            self.num_input_channels = num_targets
        else:
            self.num_input_channels = num_historical_continuous

        # ---- RevIN -----------------------------------------------------
        if use_revin:
            self.revin = RevIN(
                num_channels=self.num_input_channels,
                eps=revin_eps,
                affine=revin_affine,
                subtract_last=subtract_last,
            )

        # ---- Static / historical-cat / future encoders ----------------
        self.static_encoder = StaticContextEncoder(
            num_static_numeric=num_static_continuous,
            static_cat_specs=static_cat_specs,
            d_model=d_model,
            dropout=dropout,
        )
        self.hist_cat_encoder = TemporalCategoricalEncoder(
            cat_specs=historical_cat_specs,
            d_model=d_model,
            dropout=dropout,
        )
        self.future_encoder = FutureFeatureEncoder(
            num_future_numeric=num_future_continuous,
            future_cat_specs=future_cat_specs,
            d_model=d_model,
            dropout=dropout,
        )

        # ---- Patching geometry ----------------------------------------
        num_patches = max(1, (historical_steps - patch_len) // stride + 1)
        if padding_patch == 'end':
            self.padding_layer: Optional[nn.Module] = nn.ReplicationPad1d((0, stride))
            num_patches += 1
        else:
            self.padding_layer = None
            if historical_steps < patch_len:
                raise ValueError(
                    f"historical_steps ({historical_steps}) < patch_len "
                    f"({patch_len}) and padding_patch is disabled."
                )
        self.num_patches = num_patches

        # ---- Patch embedding + positional embedding -------------------
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)

        # ---- Transformer encoder --------------------------------------
        self.encoder = PatchTSTEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            pre_norm=pre_norm,
        )

        # ---- Classification head --------------------------------------
        future_dim = self.future_encoder.output_dim
        self.head = PatchTSTClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            future_dim=future_dim,
            head_dropout=head_dropout,
        )

        self.to(device)

    def _pool_temporal_per_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Pool [B, T_hist, D] onto patches -> [B, num_patches, D] (same as Plus)."""
        B, T, D = x.shape
        x = x.permute(0, 2, 1)
        if self.padding_layer is not None:
            x = self.padding_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if self.cat_pool == 'mean':
            x = x.mean(dim=-1)
        else:
            x = x[..., -1]
        return x.permute(0, 2, 1)

    def forward(
        self,
        static_categorical: Optional[List[torch.Tensor]] = None,
        static_continuous: Optional[List[torch.Tensor]] = None,
        historical_categorical: Optional[List[torch.Tensor]] = None,
        historical_continuous: Optional[List[torch.Tensor]] = None,
        future_categorical: Optional[List[torch.Tensor]] = None,
        future_continuous: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """TFT-compatible forward. See class docstring."""
        if historical_continuous is None or len(historical_continuous) == 0:
            raise ValueError(
                "PatchTSTPlusClassifier requires 'historical_continuous'. "
                "Use TFTDataAdapter.adapt_for_tft to produce it."
            )

        if self.channel_mode == 'target_only':
            needed = self.num_targets
        else:
            needed = self.num_historical_continuous
        if len(historical_continuous) < needed:
            raise ValueError(
                f"historical_continuous has {len(historical_continuous)} tensors "
                f"but PatchTSTPlusClassifier expects at least {needed}."
            )

        x = torch.stack(historical_continuous[:needed], dim=-1)  # [B, T_hist, C]
        B, _, C = x.shape

        # RevIN
        if self.use_revin:
            revin_mask: Optional[torch.Tensor] = None
            if padding_mask is not None:
                pm = padding_mask
                if pm.dim() == 3:
                    pm = pm.squeeze(1)
                if pm.size(-1) >= self.historical_steps:
                    pm = pm[:, : self.historical_steps]
                revin_mask = (1.0 - pm).to(x.dtype)
            x = self.revin(x, mode='norm', mask=revin_mask)

        # Patching
        x = x.permute(0, 2, 1)
        if self.padding_layer is not None:
            x = self.padding_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(B * C, self.num_patches, self.patch_len)

        # Patch embedding + positional
        x = self.patch_projection(x) + self.pos_embedding

        # Static context bias
        if self.static_encoder.mlp is not None:
            static_ctx = self.static_encoder(
                static_continuous or [],
                static_categorical or [],
            )
            static_expanded = (
                static_ctx.unsqueeze(1)
                .expand(B, C, self.d_model)
                .reshape(B * C, 1, self.d_model)
            )
            x = x + static_expanded

        # Historical categorical bias
        if self.hist_cat_encoder.cat_bank is not None:
            hist_cat_emb = self.hist_cat_encoder(historical_categorical or [])
            hist_cat_per_patch = self._pool_temporal_per_patch(hist_cat_emb)
            hist_cat_tiled = (
                hist_cat_per_patch.unsqueeze(1)
                .expand(B, C, self.num_patches, self.d_model)
                .reshape(B * C, self.num_patches, self.d_model)
            )
            x = x + hist_cat_tiled

        # Embedding dropout, encoder
        x = self.embedding_dropout(x)
        x = self.encoder(x)
        x = x.reshape(B, C, self.num_patches, self.d_model)

        # Future features -> pooled context vector
        future_context: Optional[torch.Tensor] = None
        if self.future_encoder.mlp is not None:
            future_feat = self.future_encoder(
                future_continuous or [],
                future_categorical or [],
            )  # [B, T_pred, d_model]
            if self.future_pool == 'mean':
                future_context = future_feat.mean(dim=1)
            else:
                future_context = future_feat[:, -1, :]

        # Derive per-patch validity mask
        patch_mask = _derive_patch_mask(
            padding_mask,
            self.historical_steps,
            self.padding_layer,
            self.patch_len,
            self.stride,
        )

        logits = self.head(x, patch_mask=patch_mask, future_context=future_context)
        return {'logits': logits, 'predictions': logits}

    def predict(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)['logits'].argmax(dim=-1)

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        return torch.softmax(self.forward(*args, **kwargs)['logits'], dim=-1)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def create_patchtst_classifier_from_dataset(
    dataset,
    num_classes: int,
    channel_mode: str = 'all_numeric',
    device: str = 'cpu',
    **patchtst_kwargs,
) -> PatchTSTClassifier:
    """Build :class:`PatchTSTClassifier` sized for an existing dataset."""
    num_targets = len(dataset.target_cols)
    num_historical_continuous = (
        num_targets
        + len(dataset.temporal_unknown_numeric_cols)
        + len(dataset.temporal_known_numeric_cols)
    )
    return PatchTSTClassifier(
        num_historical_continuous=num_historical_continuous,
        num_classes=num_classes,
        num_targets=num_targets,
        historical_steps=dataset.historical_steps,
        channel_mode=channel_mode,
        device=device,
        **patchtst_kwargs,
    )


def create_patchtst_plus_classifier_from_dataset(
    dataset,
    num_classes: int,
    categorical_embedding_dims: Optional[Dict[str, Tuple[int, int]]] = None,
    channel_mode: str = 'all_numeric',
    device: str = 'cpu',
    **patchtst_kwargs,
) -> PatchTSTPlusClassifier:
    """Build :class:`PatchTSTPlusClassifier` sized for an existing dataset."""
    num_targets = len(dataset.target_cols)
    num_historical_continuous = (
        num_targets
        + len(dataset.temporal_unknown_numeric_cols)
        + len(dataset.temporal_known_numeric_cols)
    )
    num_static_continuous = len(dataset.static_numeric_cols)
    num_future_continuous = len(dataset.temporal_known_numeric_cols)
    num_static_categorical = len(dataset.static_categorical_cols)
    num_historical_categorical = (
        len(dataset.temporal_unknown_categorical_cols)
        + len(dataset.temporal_known_categorical_cols)
    )
    num_future_categorical = len(dataset.temporal_known_categorical_cols)

    if categorical_embedding_dims is None:
        from .dataset import create_uniform_embedding_dims  # lazy import
        d_model = patchtst_kwargs.get('d_model', 128)
        categorical_embedding_dims = create_uniform_embedding_dims(
            dataset, hidden_layer_size=d_model
        )

    return PatchTSTPlusClassifier(
        num_historical_continuous=num_historical_continuous,
        num_classes=num_classes,
        num_targets=num_targets,
        num_static_continuous=num_static_continuous,
        num_future_continuous=num_future_continuous,
        categorical_embedding_dims=categorical_embedding_dims,
        num_static_categorical=num_static_categorical,
        num_historical_categorical=num_historical_categorical,
        num_future_categorical=num_future_categorical,
        historical_steps=dataset.historical_steps,
        channel_mode=channel_mode,
        device=device,
        **patchtst_kwargs,
    )
