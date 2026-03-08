# tft_pytorch

A PyTorch library for the **Temporal Fusion Transformer (TFT)** family, providing
a production-ready dataset pipeline, configurable loss functions, and complete
training & inference utilities.

---

## Table of Contents

1. [Installation](#installation)
2. [Package Structure](#package-structure)
3. [Core Concepts](#core-concepts)
4. [Quick Start](#quick-start)
5. [Dataset & DataLoader](#dataset--dataloader)
   - [Feature Configuration](#feature-configuration)
   - [Scaling Strategies](#scaling-strategies)
   - [Padding Strategies](#padding-strategies)
   - [Saving & Loading Encoders and Scalers](#saving--loading-encoders-and-scalers)
   - [Embedding Dimensions Helper](#embedding-dimensions-helper)
6. [Models](#models)
   - [TemporalFusionTransformer](#temporalfusiontransformer)
   - [TFTEncoderOnly](#tftencoderonly)
7. [Loss Functions](#loss-functions)
8. [Training](#training)
   - [TFTTrainer Parameters](#tfttrainer-parameters)
   - [Early Stopping & Checkpoints](#early-stopping--checkpoints)
   - [Sample Weighting](#sample-weighting)
   - [Mixed-Precision Training](#mixed-precision-training)
9. [Inference](#inference)
   - [Simple Batch Prediction](#simple-batch-prediction)
   - [Prediction with Metadata](#prediction-with-metadata)
10. [End-to-End Example](#end-to-end-example)
11. [Advanced Usage](#advanced-usage)
    - [Entity-Level Scaling](#entity-level-scaling)
    - [Cold-Start / Inference-Only Entities](#cold-start--inference-only-entities)
    - [Encoder-Only Model Example](#encoder-only-model-example)
    - [Custom Preprocessing](#custom-preprocessing)
    - [TCN Adapter](#tcn-adapter)

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # editable install from repo root
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.13, NumPy, Pandas, scikit-learn, joblib.

---

## Package Structure

```
tft_pytorch/
├── __init__.py        # public API surface
├── dataset.py         # OptimizedTFTDataset, TFTDataAdapter, TCNDataAdapter, helpers
├── models.py          # TemporalFusionTransformer, TFTEncoderOnly, all sub-layers
├── losses.py          # QuantileLoss, MSELoss, MAELoss, HuberLoss, TweedieLoss,
│                      # CombinedLoss, AdaptiveLoss
└── trainer.py         # TFTTrainer, TFTInference, TFTInferenceWithTracking
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Entity** | A single time series (e.g. one store, one product, one sensor). |
| **Window** | A fixed-length slice `[historical_steps + prediction_steps]` extracted from an entity's series. |
| **Historical** | The look-back portion of a window (encoder inputs). |
| **Future / Known** | The forecast portion (decoder inputs). Only features that are *known in advance* (e.g. calendar variables) are available here. |
| **Static** | Features that do not change over time for an entity (e.g. category, location). |
| **Scaling** | Numeric features are scaled *per-window* or *per-entity* before being fed to the model. Scaler parameters travel with each batch so predictions can be un-scaled later. |

---

## Quick Start

```python
import pandas as pd
import torch
from tft_pytorch import (
    OptimizedTFTDataset,
    create_tft_dataloader,
    create_uniform_embedding_dims,
    TemporalFusionTransformer,
    TFTTrainer,
    QuantileLoss,
)

# ------ 1. Describe your features ------
features_config = {
    "entity_col":    "store_id",
    "time_index_col": "date",
    "target_col":    "sales",

    # known in the future (e.g. promotions, calendar)
    "temporal_known_numeric_col_list":      ["price", "temperature"],
    "temporal_known_categorical_col_list":  ["day_of_week", "is_holiday"],

    # only historical (the target itself, other unknowns)
    "temporal_unknown_numeric_col_list":    [],
    "temporal_unknown_categorical_col_list": [],

    # static per entity
    "static_numeric_col_list":      ["store_size"],
    "static_categorical_col_list":  ["region"],
}

# ------ 2. Create datasets ------
train_dataset = OptimizedTFTDataset(
    data_source="train.csv",
    features_config=features_config,
    historical_steps=30,
    prediction_steps=7,
    scaling_method="standard",
    mode="train",
    encoders_path="./artifacts/encoders",
    scaler_path="./artifacts/scalers.pkl",
    scaling_strategy="entity_level",
)

val_dataset = OptimizedTFTDataset(
    data_source="val.csv",
    features_config=features_config,
    historical_steps=30,
    prediction_steps=7,
    scaling_method="standard",
    mode="val",
    encoders_path="./artifacts/encoders",   # load from training
    scaler_path="./artifacts/scalers.pkl",  # load from training
    scaling_strategy="entity_level",
)

# ------ 3. DataLoaders ------
train_loader, train_adapter = create_tft_dataloader(train_dataset, batch_size=64, shuffle=True)
val_loader,   val_adapter   = create_tft_dataloader(val_dataset,   batch_size=64, shuffle=False)

# ------ 4. Model ------
embedding_dims = create_uniform_embedding_dims(train_dataset, hidden_layer_size=128)

model = TemporalFusionTransformer(
    hidden_layer_size=128,
    num_attention_heads=4,
    num_lstm_layers=1,
    num_attention_layers=1,
    dropout_rate=0.1,
    num_static_categorical=len(train_dataset.static_categorical_cols),
    num_static_continuous=len(train_dataset.static_numeric_cols),
    num_historical_categorical=len(train_dataset.temporal_unknown_categorical_cols)
                               + len(train_dataset.temporal_known_categorical_cols),
    num_historical_continuous=len(train_dataset.target_cols)
                              + len(train_dataset.temporal_unknown_numeric_cols)
                              + len(train_dataset.temporal_known_numeric_cols),
    num_future_categorical=len(train_dataset.temporal_known_categorical_cols),
    num_future_continuous=len(train_dataset.temporal_known_numeric_cols),
    categorical_embedding_dims=embedding_dims,
    historical_steps=30,
    prediction_steps=7,
    num_outputs=3,           # [q10, q50, q90]
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# ------ 5. Train ------
trainer = TFTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    train_adapter=train_adapter,
    val_adapter=val_adapter,
    loss_type="quantile",
    loss_params={"quantiles": [0.1, 0.5, 0.9]},
    optimizer_type="adam",
    learning_rate=1e-3,
    scheduler_type="reduce_on_plateau",
    save_path="./checkpoints",
)
trainer.train(num_epochs=50, patience=10)
```

---

## Dataset & DataLoader

### Feature Configuration

Pass a `features_config` dict with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `entity_col` | str | Column that uniquely identifies each series |
| `time_index_col` | str | Datetime or sortable timestamp column |
| `target_col` | str or list[str] | Column(s) to forecast |
| `temporal_known_numeric_col_list` | list[str] | Numeric features known in the future |
| `temporal_known_categorical_col_list` | list[str] | Categorical features known in the future |
| `temporal_unknown_numeric_col_list` | list[str] | Numeric features only available historically |
| `temporal_unknown_categorical_col_list` | list[str] | Categorical features only available historically |
| `static_numeric_col_list` | list[str] | Time-invariant numeric features |
| `static_categorical_col_list` | list[str] | Time-invariant categorical features |
| `wt_col` | str, optional | Per-entity sample weight column |

### Scaling Strategies

| `scaling_strategy` | `scaling_method` | Description |
|-------------------|-----------------|-------------|
| `'per_window'` | `'standard'` | Zero-mean unit-variance, computed from *each window's* historical portion |
| `'per_window'` | `'mean'` | Divide by mean absolute value of the historical portion |
| `'entity_level'` | `'standard'` | One scaler fitted across all non-overlapping historical windows per entity |
| `'entity_level'` | `'mean'` | Same but mean-scaling |
| any | `'none'` | No scaling |

> **Tip:** Use `scaling_strategy='entity_level'` for datasets with many short series.
> It is more stable and reduces memory fragmentation compared to per-window scaling.

#### Memory Footprint

Scaler parameters are stored as compact NumPy arrays (not sklearn objects):

| Method | Storage per window |
|--------|-------------------|
| `'standard'` | `n_features × 2 × 4 bytes` |
| `'mean'` | `n_features × 4 bytes` |

For 500 k windows × 20 features: ~80 MB (standard) vs ~40 MB (mean).

### Padding Strategies

Short series (fewer rows than `historical_steps + prediction_steps`) can still
produce windows via left-padding.  Set `enable_padding=True` (default) and
choose a `padding_strategy`:

| Strategy | Numeric pad value |
|----------|------------------|
| `'zero'` | 0.0 |
| `'mean'` | Mean of the entity's available data |
| `'forward_fill'` | First available value (used for left-padding) |
| `'intelligent'` | Feature-type heuristics (mean for targets/prices, 0.5 for binary) |

Categorical columns are *always* padded with `-1` regardless of strategy.

Control the minimum usable window size with:

```python
OptimizedTFTDataset(..., min_historical_steps=10)
```

### Saving & Loading Encoders and Scalers

**Train split** — fit and save automatically:

```python
train_ds = OptimizedTFTDataset(
    ...,
    mode="train",
    encoders_path="./artifacts/encoders",
    scaler_path="./artifacts/scalers.pkl",   # entity_level only
    scaling_strategy="entity_level",
)
```

**Val / test split** — load automatically:

```python
val_ds = OptimizedTFTDataset(
    ...,
    mode="val",
    encoders_path="./artifacts/encoders",    # same path
    scaler_path="./artifacts/scalers.pkl",   # same path
    scaling_strategy="entity_level",
)
```

### Embedding Dimensions Helper

```python
from tft_pytorch import create_uniform_embedding_dims

# Returns dict like {"static_cat_0": (vocab_size, hidden_size), ...}
emb_dims = create_uniform_embedding_dims(train_dataset, hidden_layer_size=128)
```

Pass this directly to `TemporalFusionTransformer(categorical_embedding_dims=emb_dims)`.

---

## Models

### TemporalFusionTransformer

Full encoder-decoder TFT for **multi-horizon, multi-quantile** time-series forecasting.

```python
from tft_pytorch import TemporalFusionTransformer

model = TemporalFusionTransformer(
    hidden_layer_size=160,
    num_attention_heads=4,
    num_lstm_layers=1,
    num_attention_layers=1,
    dropout_rate=0.1,

    # Counts must match your dataset
    num_static_categorical=2,
    num_static_continuous=1,
    num_historical_categorical=3,   # unknown_cat + known_cat
    num_historical_continuous=5,    # targets + unknown_num + known_num (hist portion)
    num_future_categorical=1,       # known_cat only
    num_future_continuous=2,        # known_num only

    categorical_embedding_dims=emb_dims,

    historical_steps=30,
    prediction_steps=7,
    num_outputs=3,      # number of quantiles
    device="cpu",
)

outputs = model(
    static_categorical=...,      # list of [B] tensors
    static_continuous=...,       # list of [B, 1] tensors
    historical_categorical=...,  # list of [B, historical_steps] tensors
    historical_continuous=...,   # list of [B, historical_steps] tensors
    future_categorical=...,      # list of [B, prediction_steps] tensors
    future_continuous=...,       # list of [B, prediction_steps] tensors
    padding_mask=...,            # [B, 1, total_steps] optional
)

predictions = outputs["predictions"]  # [B, prediction_steps, num_quantiles]
```

**Output dict keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `predictions` | `[B, T_pred, Q]` | Quantile forecasts |
| `temporal_features` | `[B, T_total, H]` | Post-LSTM features |
| `attention_output` | `[B, T_total, H]` | Post-attention features |
| `static_weights` | `[B, n_static]` | Variable selection weights (static) |
| `historical_weights` | `[B, T_hist, n_hist]` | Variable selection weights (historical) |
| `future_weights` | `[B, T_pred, n_future]` | Variable selection weights (future) |

### TFTEncoderOnly

Encoder-only variant for tasks **without future inputs** — e.g. predicting
a scalar return from a rolling historical window, or classifying a regime.

```python
from tft_pytorch import TFTEncoderOnly

model = TFTEncoderOnly(
    hidden_layer_size=128,
    num_attention_heads=4,
    num_lstm_layers=2,
    num_attention_layers=1,
    dropout_rate=0.1,

    num_static_categorical=2,
    num_static_continuous=1,
    num_historical_categorical=2,
    num_historical_continuous=10,

    categorical_embedding_dims=emb_dims,
    historical_steps=30,

    output_size=1,             # regression: 1 scalar
    output_type="regression",  # or "classification"
    # num_classes=3,           # required for classification

    temporal_aggregation="attention",  # 'mean' | 'max' | 'last' | 'attention'
    device="cpu",
)

outputs = model(
    historical_continuous=...,
    historical_categorical=...,
    static_continuous=...,
    static_categorical=...,
)

scalar_pred = outputs["output"]  # [B, output_size]
```

For **classification**:

```python
model = TFTEncoderOnly(..., output_type="classification", num_classes=3)
outputs = model(...)
print(outputs["predictions"])     # [B]  — argmax class indices
print(outputs["probabilities"])   # [B, 3]
```

---

## Loss Functions

All losses accept optional `mask` (1=valid, 0=ignore) and `sample_weight` tensors.

```python
from tft_pytorch import QuantileLoss, MSELoss, HuberLoss, TweedieLoss, CombinedLoss, AdaptiveLoss

# Quantile loss  (default quantiles = [0.1, 0.25, 0.5, 0.75, 0.9])
criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
loss = criterion(predictions, targets, mask=future_mask)

# MSE
criterion = MSELoss()

# MAE
from tft_pytorch import MAELoss
criterion = MAELoss()

# Huber (smooth L1)
criterion = HuberLoss(delta=1.0)

# Tweedie  (good for non-negative count-like targets, e.g. sales)
# Note: trainer will inverse-transform before computing this loss
criterion = TweedieLoss(p=1.5)   # p in (1, 2)

# Combine two losses with fixed weights
from tft_pytorch import CombinedLoss
criterion = CombinedLoss(
    losses=[QuantileLoss([0.1, 0.5, 0.9]), MAELoss()],
    weights=[0.8, 0.2],
)

# Automatically balance losses by their running magnitudes
criterion = AdaptiveLoss(
    losses=[QuantileLoss([0.1, 0.5, 0.9]), MAELoss()],
    ema_decay=0.99,
    warmup_steps=200,
)
```

---

## Training

### TFTTrainer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_type` | `'quantile'` | Loss function type |
| `loss_params` | `{}` | Extra args for the loss (e.g. `{'quantiles': [0.1,0.5,0.9]}`) |
| `optimizer_type` | `'adam'` | `'adam'` \| `'adamw'` \| `'sgd'` |
| `learning_rate` | `1e-3` | Initial learning rate |
| `weight_decay` | `1e-5` | L2 regularisation |
| `momentum` | `0.9` | Only for SGD |
| `scheduler_type` | `'reduce_on_plateau'` | `'reduce_on_plateau'` \| `'cosine'` \| `None` |
| `scheduler_factor` | `0.5` | LR reduction factor (ReduceLROnPlateau) |
| `scheduler_patience` | `10` | Epochs with no improvement before LR reduction |
| `scheduler_t0` | `10` | Restart period (CosineAnnealingWarmRestarts) |
| `scheduler_t_mult` | `2` | Period multiplier after each restart |
| `enable_gradient_clipping` | `True` | Clip gradient norm |
| `max_grad_norm` | `1.0` | Max gradient norm |
| `enable_train_sample_weighting` | `False` | Multiply loss by entity × recency weight during train |
| `enable_val_sample_weighting` | `False` | Same for validation |
| `enable_mixed_precision` | `False` | FP16 training via `torch.amp` |
| `save_path` | `'./checkpoints'` | Checkpoint directory |
| `save_every` | `5` | Save every N epochs |

### Early Stopping & Checkpoints

```python
trainer.train(num_epochs=100, patience=15)
# Saves 'best_model.pt' and 'best_model_weights.pt' to save_path.
# Also saves 'checkpoint_epoch_N.pt' every save_every epochs.
```

Resume from a checkpoint:

```python
trainer.load_checkpoint("./checkpoints/checkpoint_epoch_20.pt")
trainer.train(num_epochs=100, patience=15)
```

### Sample Weighting

When `enable_train_sample_weighting=True`, the trainer looks for:
- `entity_weight` — a per-entity importance score (set via `wt_col` in `features_config`)
- `recency_weight` — exponential weight based on window recency (set via `recency_alpha` in the dataset)

Both are multiplied together and applied to the per-sample loss.

```python
# In the dataset
train_ds = OptimizedTFTDataset(
    ...,
    recency_alpha=0.5,   # recent windows get higher weight
    # features_config with wt_col="entity_importance"
)

# In the trainer
trainer = TFTTrainer(..., enable_train_sample_weighting=True)
```

### Mixed-Precision Training

```python
trainer = TFTTrainer(
    ...,
    enable_mixed_precision=True,   # requires CUDA
)
```

---

## Inference

### Simple Batch Prediction

```python
from tft_pytorch import TFTInference, create_tft_dataloader

test_ds = OptimizedTFTDataset(..., mode="test", ...)
test_loader, test_adapter = create_tft_dataloader(test_ds, batch_size=128, shuffle=False)

inference = TFTInference(
    model_path="./checkpoints/best_model.pt",
    model=model,         # same architecture as training
    adapter=test_adapter,
    device="cpu",
)

predictions, actuals = inference.predict_batch(test_loader)
# predictions: np.ndarray  [N, prediction_steps, num_quantiles]
# actuals:     np.ndarray  [N, prediction_steps, 1]  (or None if unavailable)
```

### Prediction with Metadata

```python
from tft_pytorch import TFTInferenceWithTracking

inference = TFTInferenceWithTracking(
    model_path="./checkpoints/best_model.pt",
    model=model,
    adapter=test_adapter,
)

results_df = inference.predict_with_metadata(test_loader)
print(results_df.head())
#   entity_id  timestamp  window_idx  horizon  pred_q10  pred_q50  pred_q90  actual_sales
```

The returned DataFrame has:
- One row per `(entity, forecast horizon)` combination
- Predictions are automatically **inverse-transformed** to the original scale
- `pred_q10`, `pred_q50`, `pred_q90` for 3-quantile models (or `pred_0..N` for others)
- `actual_<target_col>` when ground truth is available

---

## End-to-End Example

```python
import torch, pandas as pd
from tft_pytorch import (
    OptimizedTFTDataset, create_tft_dataloader, create_uniform_embedding_dims,
    TemporalFusionTransformer, TFTTrainer, TFTInferenceWithTracking
)

FEATURES = {
    "entity_col": "store",
    "time_index_col": "date",
    "target_col": "sales",
    "temporal_known_numeric_col_list": ["price"],
    "temporal_known_categorical_col_list": ["weekday"],
    "static_categorical_col_list": ["category"],
}
H, F = 30, 7   # look-back, forecast horizon
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Datasets
train_ds = OptimizedTFTDataset("train.csv", FEATURES, H, F, mode="train",
                                scaling_strategy="entity_level", scaling_method="standard",
                                encoders_path="./artifacts/enc",
                                scaler_path="./artifacts/scalers.pkl")
val_ds   = OptimizedTFTDataset("val.csv",   FEATURES, H, F, mode="val",
                                scaling_strategy="entity_level", scaling_method="standard",
                                encoders_path="./artifacts/enc",
                                scaler_path="./artifacts/scalers.pkl")
test_ds  = OptimizedTFTDataset("test.csv",  FEATURES, H, F, mode="test",
                                scaling_strategy="entity_level", scaling_method="standard",
                                encoders_path="./artifacts/enc",
                                scaler_path="./artifacts/scalers.pkl")

train_dl, train_adp = create_tft_dataloader(train_ds, 64, shuffle=True)
val_dl,   val_adp   = create_tft_dataloader(val_ds,   64, shuffle=False)
test_dl,  test_adp  = create_tft_dataloader(test_ds,  128, shuffle=False)

# Model
emb = create_uniform_embedding_dims(train_ds, hidden_layer_size=128)
model = TemporalFusionTransformer(
    hidden_layer_size=128, num_attention_heads=4,
    num_static_categorical=1, num_historical_categorical=2,
    num_historical_continuous=2, num_future_categorical=1,
    num_future_continuous=1, categorical_embedding_dims=emb,
    historical_steps=H, prediction_steps=F, num_outputs=3, device=DEVICE,
)

# Train
TFTTrainer(model, train_dl, val_dl, train_adp, val_adp,
           loss_type="quantile", loss_params={"quantiles": [0.1, 0.5, 0.9]},
           save_path="./ckpt").train(num_epochs=50, patience=10)

# Inference
results = TFTInferenceWithTracking("./ckpt/best_model.pt", model, test_adp) \
            .predict_with_metadata(test_dl)
results.to_csv("forecasts.csv", index=False)
```

---

## Advanced Usage

### Entity-Level Scaling

```python
train_ds = OptimizedTFTDataset(
    ...,
    scaling_strategy="entity_level",   # one scaler per entity (not per window)
    scaling_method="standard",
    scaler_path="./artifacts/scalers.pkl",  # auto-saved after training
    mode="train",
)

# Val/test: scalers loaded automatically from scaler_path
```

### Cold-Start / Inference-Only Entities

For brand-new entities at inference time that have fewer rows than `prediction_steps`:

```python
test_ds = OptimizedTFTDataset(
    ...,
    mode="test",
    allow_inference_only_entities=True,  # accept any-length series
    cold_start_scaler_cols=["category"], # use these static cols to pick best scaler
)
```

### Encoder-Only Model Example

Predict a stock's average return over the next 3 candles from 30 historical candles:

```python
from tft_pytorch import TFTEncoderOnly

model = TFTEncoderOnly(
    hidden_layer_size=128,
    num_historical_continuous=10,   # OHLCV + technical indicators
    num_historical_categorical=2,   # day_of_week, session_type
    num_static_categorical=2,       # ticker_id, sector
    num_static_continuous=1,        # log_market_cap
    categorical_embedding_dims=emb,
    historical_steps=30,
    output_size=1,
    output_type="regression",
    temporal_aggregation="attention",
    device="cpu",
)
```

### Custom Preprocessing

```python
def my_preprocessing(df):
    df["log_sales"] = df["sales"].clip(lower=1).apply(np.log)
    df["sales"] = df["log_sales"]
    return df

ds = OptimizedTFTDataset(..., preprocessing_fn=my_preprocessing)
```

### TCN Adapter

Use the dataset with a Temporal Convolutional Network (or any model expecting a
single concatenated feature tensor):

```python
from tft_pytorch import create_tcn_dataloader

loader, adapter = create_tcn_dataloader(dataset, batch_size=64)

for batch in loader:
    tcn_batch = adapter.adapt_for_tcn(batch)
    # tcn_batch["numeric_features"]      [B, T, n_numeric]
    # tcn_batch["categorical_features"]  [B, T, n_categorical]  ← raw indices
    # tcn_batch["targets"]               [B, T_pred, n_targets]
    outputs = my_tcn_model(tcn_batch["numeric_features"])
```

---

## License

MIT
