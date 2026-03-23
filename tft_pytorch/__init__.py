"""
tft_pytorch
===========

A PyTorch library for the Temporal Fusion Transformer (TFT) family of models,
featuring a memory-efficient dataset/dataloader pipeline, multiple loss functions,
and complete training & inference utilities.

Quick-start
-----------
>>> from tft_pytorch import (
...     OptimizedTFTDataset,
...     create_tft_dataloader,
...     create_uniform_embedding_dims,
...     TemporalFusionTransformer,
...     TFTTrainer,
...     TFTInferenceWithTracking,
...     QuantileLoss,
... )
"""

from .dataset import (
    OptimizedTFTDataset,
    TFTDataAdapter,
    TCNDataAdapter,
    create_tft_dataloader,
    create_tcn_dataloader,
    create_uniform_embedding_dims,
    inverse_transform_predictions,
)

from .models import (
    TemporalFusionTransformer,
    TFTEncoderOnly,
    # Core building blocks
    apply_time_distributed,
    scaled_dot_product_attention,
    TFTMultiHeadAttention,
    TFTLinearLayer,
    TFTApplyMLP,
    TFTApplyGatingLayer,
    TFTAddAndNormLayer,
    TFTGRNLayer,
    VariableSelectionStatic,
    VariableSelectionTemporal,
    StaticContexts,
    LSTMLayer,
    StaticEnrichmentLayer,
    AttentionLayer,
    AttentionStack,
    FinalGatingLayer,
)

from .losses import (
    QuantileLoss,
    MSELoss,
    RMSELoss,
    MAELoss,
    HuberLoss,
    TweedieLoss,
    CombinedLoss,
    AdaptiveLoss,
)

from .trainer import (
    TFTTrainer,
    TFTInference,
    TFTInferenceWithTracking,
)

__version__ = '0.1.1'
__all__ = [
    # dataset
    'OptimizedTFTDataset',
    'TFTDataAdapter',
    'TCNDataAdapter',
    'create_tft_dataloader',
    'create_tcn_dataloader',
    'create_uniform_embedding_dims',
    'inverse_transform_predictions',
    # models
    'TemporalFusionTransformer',
    'TFTEncoderOnly',
    'apply_time_distributed',
    'scaled_dot_product_attention',
    'TFTMultiHeadAttention',
    'TFTLinearLayer',
    'TFTApplyMLP',
    'TFTApplyGatingLayer',
    'TFTAddAndNormLayer',
    'TFTGRNLayer',
    'VariableSelectionStatic',
    'VariableSelectionTemporal',
    'StaticContexts',
    'LSTMLayer',
    'StaticEnrichmentLayer',
    'AttentionLayer',
    'AttentionStack',
    'FinalGatingLayer',
    # losses
    'QuantileLoss',
    'MSELoss',
    'RMSELoss',
    'MAELoss',
    'HuberLoss',
    'TweedieLoss',
    'CombinedLoss',
    'AdaptiveLoss',
    # training
    'TFTTrainer',
    'TFTInference',
    'TFTInferenceWithTracking',
]
