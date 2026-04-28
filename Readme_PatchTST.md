# Integrating PatchTST family into `tft_pytorch`

This guide adds four PatchTST variants to your library:

| Class                         | Task           | Features supported                                                     |
|-------------------------------|----------------|------------------------------------------------------------------------|
| `PatchTST`                    | Forecasting    | Historical numeric only                                                |
| `PatchTSTPlus`                | Forecasting    | + static + historical categoricals + future-known (num + cat)          |
| `PatchTSTClassifier`          | Classification | Historical numeric only                                                |
| `PatchTSTPlusClassifier`      | Classification | + static + historical categoricals + future-known (num + cat)          |

All four plug into `OptimizedTFTDataset`, `TFTDataAdapter`, and
`create_uniform_embedding_dims` **without modifying any of those files**.
The forecasting models additionally work with the existing `TFTTrainer`
unchanged. The classifiers use a short custom training loop (see §5).

---

## 1. Files to add / change

### 1.1 Add `tft_pytorch/patchtst.py`

Drop the provided `patchtst.py` into your package directory.

### 1.2 Replace `tft_pytorch/__init__.py`

Use the provided `__init__.py`. It re-exports all four models plus their
supporting modules and factories, and bumps `__version__` to `0.3.0`.

### 1.3 Bump `setup.py`

```python
version="0.3.0",
```

No new dependencies.

---

## 2. Why no trainer / adapter / dataset changes?

All four models share the forward signature of `TemporalFusionTransformer`:

```python
static_categorical, static_continuous,
historical_categorical, historical_continuous,
future_categorical, future_continuous,
padding_mask
```

The forecasting models return `{'predictions': [B, T_pred, num_outputs]}`
-- `TFTTrainer`, `TFTInference`, and `TFTInferenceWithTracking` consume
this directly.

The classifiers return `{'logits': [B, num_classes], 'predictions': ...}`
where `'predictions'` aliases `'logits'` for API symmetry. The shape is
fundamentally different from forecasting, so `TFTTrainer` would need
meaningful changes to work -- a small custom loop (§5) is cleaner.

---

## 3. Quick start -- forecasting with `PatchTSTPlus`

```python
from tft_pytorch import (
    OptimizedTFTDataset, create_tft_dataloader,
    create_patchtst_plus_from_dataset,
    TFTTrainer, TFTInferenceWithTracking,
)

features_config = {
    'entity_col':        'sku_store_id',
    'time_index_col':    'date',
    'target_col':        'units_sold',
    'static_numeric_col_list':            ['store_size', 'latitude'],
    'static_categorical_col_list':        ['store_type', 'region', 'segment'],
    'temporal_unknown_numeric_col_list':  ['competitor_price', 'foot_traffic'],
    'temporal_unknown_categorical_col_list': ['stock_status'],
    'temporal_known_numeric_col_list':    ['planned_price', 'temperature_forecast'],
    'temporal_known_categorical_col_list':['day_of_week', 'month', 'holiday', 'planned_promotion'],
    'wt_col':            'importance',
}

train_ds = OptimizedTFTDataset(data_source=train_df, features_config=features_config,
                                historical_steps=96, prediction_steps=28,
                                scaling_method='standard', mode='train')
val_ds   = OptimizedTFTDataset(data_source=val_df, features_config=features_config,
                                historical_steps=96, prediction_steps=28,
                                scaling_method='standard', mode='val')

train_loader, train_adapter = create_tft_dataloader(train_ds, batch_size=128)
val_loader,   val_adapter   = create_tft_dataloader(val_ds,   batch_size=128, shuffle=False)

model = create_patchtst_plus_from_dataset(
    train_ds,
    num_outputs=3, channel_mode='all_numeric',
    patch_len=16, stride=8,
    d_model=128, n_heads=8, num_encoder_layers=3, d_ff=256,
    dropout=0.2, use_revin=True, cat_pool='mean',
    device='cuda',
)

trainer = TFTTrainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    train_adapter=train_adapter, val_adapter=val_adapter,
    loss_type='quantile', loss_params={'quantiles': [0.1, 0.5, 0.9]},
    learning_rate=1e-4, scheduler_type='reduce_on_plateau',
    enable_train_sample_weighting=True,
    save_path='./checkpoints/patchtst_plus',
)
trainer.train(num_epochs=80, patience=12)
```

---

## 4. Architecture summary (all four variants)

Shared across variants:

| Component            | Used by all              | Role                                                     |
|----------------------|--------------------------|----------------------------------------------------------|
| RevIN                | All (optional)           | Padding-aware per-channel instance normalization         |
| Patching             | All                      | Unfold history into overlapping patches                  |
| Transformer encoder  | All                      | Channel-independent, shared weights across channels      |

Feature injection mechanisms (Plus variants only):

| Feature                               | Mechanism                                                                                      |
|---------------------------------------|------------------------------------------------------------------------------------------------|
| Static (num + cat)                    | MLP -> single `d_model` vector, added to every patch token (global bias)                       |
| Historical categorical                | Per-timestep embedding -> linear -> pooled per patch -> added to numeric patch tokens          |
| Future-known (num + cat)              | Per-step encoder -> either concat in forecasting head (Plus) or mean-pool + concat (Classifier)|

Heads:

| Model                    | Head                                                                                                      |
|--------------------------|-----------------------------------------------------------------------------------------------------------|
| `PatchTST`               | Flatten -> linear -> horizon * num_outputs                                                                |
| `PatchTSTPlus`           | Flatten encoder + flatten future features -> linear -> horizon * num_outputs                              |
| `PatchTSTClassifier`     | **Padding-aware** mean-pool over patches and channels -> linear -> num_classes                            |
| `PatchTSTPlusClassifier` | Padding-aware mean-pool over patches and channels + pooled future context -> linear -> num_classes        |

---

## 5. Quick start -- classification with `PatchTSTPlusClassifier`

### 5.1 Preparing labels

The dataset pipeline produces `future_targets` of shape
`[B, prediction_steps, num_targets]`. For classification you typically
derive a single label per window. Common recipes:

```python
# A) Threshold on the horizon sum (e.g. "will total sales drop below X")
labels = (future_targets.sum(dim=1).squeeze(-1) < threshold).long()

# B) Match against a stored-per-window label column (recommended: add a
# separate label lookup keyed by (entity_id, time_index) and join in
# your training loop)

# C) Synthesize from the future_categorical inputs themselves
# e.g. "will a promotion be active during horizon"
labels = (future_categorical[1].sum(dim=1) > 0).long()
```

Whichever you pick, compute labels in the same batch step as the model
forward pass, or pre-compute them and attach to the dataset as an
additional field.

### 5.2 Custom training loop

The existing `TFTTrainer` assumes forecasting-shaped outputs. For
classification use a short loop -- it's ~30 lines and gives you full
control over metrics and label derivation:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tft_pytorch import (
    OptimizedTFTDataset, create_tft_dataloader,
    create_patchtst_plus_classifier_from_dataset,
)

# Dataset / dataloaders -- unchanged from forecasting setup
train_ds = OptimizedTFTDataset(data_source=train_df, features_config=features_config,
                                historical_steps=96, prediction_steps=28,
                                scaling_method='standard', mode='train')
train_loader, train_adapter = create_tft_dataloader(train_ds, batch_size=128)

model = create_patchtst_plus_classifier_from_dataset(
    train_ds,
    num_classes=2,                    # e.g. binary: will demand drop >30%?
    channel_mode='all_numeric',
    d_model=128, n_heads=8, num_encoder_layers=3, d_ff=256,
    dropout=0.2, head_dropout=0.1,
    use_revin=True,                   # see note below
    device='cuda',
).to('cuda')

# Class weights for imbalance (compute from training data once)
class_weights = torch.tensor([0.2, 1.5]).to('cuda')  # e.g. 80/20 split
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in train_loader:
        # TFTDataAdapter returns the same dict used for TFT/PatchTSTPlus
        model_inputs = train_adapter.adapt_for_tft(batch)
        labels = derive_labels(model_inputs['future_targets']).to('cuda')  # your recipe

        # Move all inputs to device (trainer does this for you normally)
        model_inputs = move_to_device(model_inputs, 'cuda')

        logits = model(
            static_categorical=model_inputs.get('static_categorical'),
            static_continuous=model_inputs.get('static_continuous'),
            historical_categorical=model_inputs.get('historical_categorical'),
            historical_continuous=model_inputs.get('historical_continuous'),
            future_categorical=model_inputs.get('future_categorical'),
            future_continuous=model_inputs.get('future_continuous'),
            padding_mask=model_inputs.get('padding_mask'),
        )['logits']

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_n += labels.size(0)

    scheduler.step()
    print(f'epoch {epoch+1}  loss: {total_loss/total_n:.4f}  acc: {total_correct/total_n:.2%}')
```

### 5.3 Inference helpers

Both classifiers expose convenience methods:

```python
model.eval()
with torch.no_grad():
    cls = model.predict(**model_inputs)          # [B] long tensor
    proba = model.predict_proba(**model_inputs)  # [B, num_classes], sums to 1
```

Or manually with full control:

```python
logits = model(**model_inputs)['logits']
proba = torch.softmax(logits, dim=-1)
```

### 5.4 A note on RevIN for classification

RevIN removes each channel's mean and std before the encoder sees them,
and in forecasting that's unambiguously helpful (you're predicting
deviations). For classification, **the mean and std may be part of the
class signal** -- a high-volume SKU is qualitatively different from a
low-volume one, and you probably don't want to normalize that away.

Recommended workflow: benchmark both `use_revin=True` and `use_revin=False`
and pick the one that validates better. The toggle is free.

---

## 6. Output shapes reference

| Model                    | Output key        | Shape                                  |
|--------------------------|-------------------|----------------------------------------|
| `PatchTST`               | `predictions`     | `[B, prediction_steps, num_outputs]`   |
| `PatchTSTPlus`           | `predictions`     | `[B, prediction_steps, num_outputs]`   |
| `PatchTSTClassifier`     | `logits`          | `[B, num_classes]`                     |
| `PatchTSTPlusClassifier` | `logits`          | `[B, num_classes]`                     |

For multi-target forecasting: `num_outputs` means output-dim per target
and the final axis has size `num_targets * num_outputs`.

For the classifiers, `'predictions'` is an alias of `'logits'` so that
anything generic iterating over the model's output dict finds both keys.

---

## 7. Loss compatibility

### Forecasting

| Loss            | Works | Recommended `num_outputs`             |
|-----------------|:-----:|---------------------------------------|
| `QuantileLoss`  | Yes   | `len(quantiles)` (single target)      |
| `MSELoss`       | Yes   | `1` (single) or `num_targets` (multi) |
| `RMSELoss`      | Yes   | same as MSE                           |
| `MAELoss`       | Yes   | same as MSE                           |
| `HuberLoss`     | Yes   | same as MSE                           |
| `TweedieLoss`   | Yes   | `1` (non-negative targets only)       |
| `CombinedLoss`  | Yes   | same as inner losses                  |
| `AdaptiveLoss`  | Yes   | same as inner losses                  |

### Classification

Use standard PyTorch losses directly on the `'logits'` output:

| Loss                        | When to use                                               |
|-----------------------------|-----------------------------------------------------------|
| `nn.CrossEntropyLoss`       | Standard multi-class (single label per window)            |
| `nn.CrossEntropyLoss(weight=...)` | Imbalanced multi-class -- pass per-class weights    |
| `nn.BCEWithLogitsLoss`      | Binary or multi-label (independent labels per window)     |
| Focal loss (implement-your-own) | Extreme imbalance where class weights aren't enough   |

---

## 8. Tested scenarios

**PatchTST forecasting** (13 tests): basic construction, forward shape,
all loss integrations, channel modes, padding mask, sample weighting,
full TFT kwargs, literal trainer simulation, mixed precision, state_dict
roundtrip, factory, multi-target, train/eval modes.

**PatchTSTPlus forecasting** (14 tests: A-N): full feature set, -1
padding, feature subsets, multi-target, literal trainer simulation, all
losses, state_dict, mixed precision, padding mask, factory, **gradient
flow audit** (all 6 pathways receive gradients), **sensitivity audit**
(perturbing each feature type independently shifts predictions).

**PatchTSTClassifier + PatchTSTPlusClassifier** (new tests):
1. Binary and multi-class forward shapes
2. Literal training step with `CrossEntropyLoss`, class weighting
3. `BCEWithLogitsLoss` for multi-label
4. Padding-aware pooling -- all-padded patches excluded from the pool
5. Feature subsets (no static, no future, no categoricals)
6. `state_dict` save/load roundtrip
7. Mixed-precision autocast
8. `use_revin=False` path
9. `predict()` and `predict_proba()` helpers
10. `num_classes < 2` rejected with clear error
11. **Gradient flow audit** through all 6 input pathways
12. **Sensitivity audit** -- perturbing each feature type shifts logits
13. End-to-end 5-epoch learning sanity check (accuracy rises from 37.5%
    to 75% on a fake task where label correlates with a static feature)
14. Regression: forecasting models unaffected by classifier additions

---

## 9. When to choose which variant

| Situation                                                     | Pick                         |
|---------------------------------------------------------------|------------------------------|
| Forecasting benchmarks / ablations / paper baseline           | `PatchTST`                   |
| Production demand forecasting with rich TFT-style features    | `PatchTSTPlus`               |
| UCR/UEA classification, sensor-stream classification          | `PatchTSTClassifier`         |
| Retail classification with static + promo + calendar features | `PatchTSTPlusClassifier`     |

All four share the dataset pipeline, so you can mix and match in the
same codebase without duplicating data-prep code.

---

## 10. Suggested `README.md` addition

> ### PatchTST family
>
> Four channel-independent transformer variants that share the
> `OptimizedTFTDataset` plumbing with `TemporalFusionTransformer`.
> `PatchTST` and `PatchTSTPlus` work with `TFTTrainer` unchanged;
> `PatchTSTClassifier` and `PatchTSTPlusClassifier` use a short custom
> training loop with standard PyTorch classification losses.
>
> ```python
> from tft_pytorch import (
>     create_patchtst_from_dataset,                # forecasting, numeric-only
>     create_patchtst_plus_from_dataset,           # forecasting, full features
>     create_patchtst_classifier_from_dataset,     # classification, numeric-only
>     create_patchtst_plus_classifier_from_dataset,# classification, full features
> )
>
> model = create_patchtst_plus_classifier_from_dataset(
>     train_ds, num_classes=2, channel_mode='all_numeric',
> )
> ```
>
> The Plus variants add three principled mechanisms to vanilla PatchTST:
> a static-context vector added to every patch token, patch-aligned
> pooling of historical categoricals, and future-known features fused
> into the head. The channel-independent backbone and padding-aware RevIN
> are preserved across all four.
