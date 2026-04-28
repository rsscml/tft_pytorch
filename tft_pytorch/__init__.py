"""
tft_pytorch
===========

A PyTorch library for the Temporal Fusion Transformer (TFT) family of models,
featuring a memory-efficient dataset/dataloader pipeline, multiple loss functions,
and complete training & inference utilities.

Includes four PatchTST variants that plug into the same dataset pipeline
without modification:

* ``PatchTST``                 - paper-faithful forecasting, historical numeric only
* ``PatchTSTPlus``             - forecasting with full TFT-style features
                                 (static + categoricals + future-known)
* ``PatchTSTClassifier``       - paper-faithful classification, historical numeric only
* ``PatchTSTPlusClassifier``   - classification with full TFT-style features

The forecasting models work with the existing ``TFTTrainer``. The
classifiers use a short custom training loop with standard PyTorch losses
(``nn.CrossEntropyLoss``, ``nn.BCEWithLogitsLoss``) -- see
``INTEGRATION.md``.

Quick-start
-----------
>>> from tft_pytorch import (
...     OptimizedTFTDataset,
...     create_tft_dataloader,
...     create_uniform_embedding_dims,
...     TemporalFusionTransformer,
...     PatchTST,
...     PatchTSTPlus,
...     PatchTSTClassifier,
...     PatchTSTPlusClassifier,
...     create_patchtst_from_dataset,
...     create_patchtst_plus_from_dataset,
...     create_patchtst_classifier_from_dataset,
...     create_patchtst_plus_classifier_from_dataset,
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

from .patchtst import (
    # Vanilla forecasting (paper-faithful)
    PatchTST,
    PatchTSTBackbone,
    PatchTSTEncoder,
    PatchTSTEncoderLayer,
    FlattenHead,
    RevIN,
    create_patchtst_from_dataset,
    # Extended forecasting (full TFT-style features)
    PatchTSTPlus,
    PatchTSTPlusHead,
    StaticContextEncoder,
    TemporalCategoricalEncoder,
    FutureFeatureEncoder,
    CategoricalEmbeddingBank,
    create_patchtst_plus_from_dataset,
    # Classification variants
    PatchTSTClassifier,
    PatchTSTPlusClassifier,
    PatchTSTClassificationHead,
    create_patchtst_classifier_from_dataset,
    create_patchtst_plus_classifier_from_dataset,
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

from .interpretation import (
    TFTInterpreter, InterpretationResult,
    historical_feature_names, future_feature_names, static_feature_names,
)

__version__ = '0.3.0'
__all__ = [
    # dataset
    'OptimizedTFTDataset',
    'TFTDataAdapter',
    'TCNDataAdapter',
    'create_tft_dataloader',
    'create_tcn_dataloader',
    'create_uniform_embedding_dims',
    'inverse_transform_predictions',
    # models (TFT)
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
    # models (PatchTST family)
    'PatchTST',
    'PatchTSTBackbone',
    'PatchTSTEncoder',
    'PatchTSTEncoderLayer',
    'FlattenHead',
    'RevIN',
    'create_patchtst_from_dataset',
    'PatchTSTPlus',
    'PatchTSTPlusHead',
    'StaticContextEncoder',
    'TemporalCategoricalEncoder',
    'FutureFeatureEncoder',
    'CategoricalEmbeddingBank',
    'create_patchtst_plus_from_dataset',
    'PatchTSTClassifier',
    'PatchTSTPlusClassifier',
    'PatchTSTClassificationHead',
    'create_patchtst_classifier_from_dataset',
    'create_patchtst_plus_classifier_from_dataset',
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
