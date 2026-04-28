# TFT Interpretation Module

Documentation for `tft_pytorch/interpretation.py` — a module that turns a
trained Temporal Fusion Transformer into an **explainable** forecaster by
extracting and aggregating the two interpretability signals the architecture
was designed to expose: variable selection (VSN) weights and self-attention
weights.

---

## Table of contents

1. [What this module does](#what-this-module-does)
2. [Why TFT is interpretable](#why-tft-is-interpretable)
3. [Installation and wiring](#installation-and-wiring)
4. [Quick start](#quick-start)
5. [Design](#design)
   - [Forward hooks, not model edits](#forward-hooks-not-model-edits)
   - [Feature-name resolution](#feature-name-resolution)
   - [Long-form result container](#long-form-result-container)
   - [The horizon × past-lag attention view](#the-horizon--past-lag-attention-view)
6. [Detailed usage](#detailed-usage)
   - [Global feature importance](#global-feature-importance)
   - [Per-sample explanations](#per-sample-explanations)
   - [Persistent temporal patterns](#persistent-temporal-patterns)
   - [Plotting](#plotting)
   - [Exporting](#exporting)
   - [Combining with predictions and uncertainty](#combining-with-predictions-and-uncertainty)
7. [Special notes and gotchas](#special-notes-and-gotchas)
8. [API reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## What this module does

`TFTInterpreter` runs a trained TFT model in `eval()` mode over a dataloader
and harvests, for every sample:

- The **variable selection weights** for static, historical, and future
  feature groups (already returned by your model's forward pass).
- The **multi-head self-attention weights** that the model uses to decide
  which past timesteps to pay attention to when forecasting each horizon
  (currently discarded by `AttentionLayer.forward` — captured here via
  forward hooks).

Everything is collected into an `InterpretationResult` object containing
tidy long-form pandas DataFrames, with helpers for the common aggregations
(per-feature importance, attention by horizon, persistent temporal pattern)
and matplotlib plotting helpers for the standard four-panel TFT
interpretability view.

It is purely additive: no existing files in `tft_pytorch` need to change, and
it works on already-trained checkpoints with no retraining.

---

## Why TFT is interpretable

The TFT paper (Lim et al., 2021) builds two explainability mechanisms
directly into the architecture:

1. **Variable Selection Networks** — for each input group (static,
   historical, future), a small softmax network outputs a per-feature weight.
   These weights are intrinsic to the forward pass: the model literally
   multiplies feature embeddings by them before integrating. So the weights
   are not a post-hoc attribution; they are the model's actual reliance.

2. **Interpretable multi-head self-attention** — the decoder attends over
   the full historical-plus-future sequence. The attention weights from a
   future query position to past key positions tell you which historical
   timesteps the model used when producing that horizon's forecast.

Together these two signals answer:

- *"Which features did the model use?"* → VSN weights.
- *"Which past timesteps did it look at?"* → attention weights.

A handful of other interpretability methods (Integrated Gradients, SHAP,
input perturbation) can also be applied to a TFT, but they are external —
they treat the model as a black box. This module only deals with the two
internal signals.

---

## Installation and wiring

The file `interpretation.py` lives next to `models.py` inside the
`tft_pytorch` package:

```
tft_pytorch/
├── __init__.py
├── dataset.py
├── models.py
├── losses.py
├── trainer.py
└── interpretation.py        ← new
```

Add to `tft_pytorch/__init__.py`:

```python
from .interpretation import (
    TFTInterpreter,
    InterpretationResult,
    historical_feature_names,
    future_feature_names,
    static_feature_names,
)
```

And update `__all__` accordingly. No new third-party dependencies are
introduced. `matplotlib` is imported lazily inside the plotting helpers, so
you only need it if you actually call `plot_*`.

---

## Quick start

```python
import torch
from tft_pytorch import (
    OptimizedTFTDataset, create_tft_dataloader, TemporalFusionTransformer,
)
from tft_pytorch.interpretation import TFTInterpreter

# 1. Re-build the model architecture and load trained weights
model = TemporalFusionTransformer(...)   # same args as training
ckpt = torch.load("./checkpoints/best_model.pt", map_location="cpu",
                  weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 2. Build a test dataloader and adapter (same as for inference)
test_ds = OptimizedTFTDataset("test.csv", FEATURES, H, F, mode="test", ...)
test_loader, test_adapter = create_tft_dataloader(test_ds, batch_size=64,
                                                  shuffle=False)

# 3. Interpret
with TFTInterpreter(model, test_adapter) as interp:
    result = interp.interpret(test_loader, max_batches=10)

# 4. Inspect
print(result.feature_importance("historical").head())
print(result.attention_by_horizon().round(3))
result.plot_feature_importance("historical", top_k=10)
```

---

## Design

### Forward hooks, not model edits

In your `models.py`, line 819:

```python
attn_out, _ = self.mha(x, x, x, attn_mask, padding_mask)
```

The `_` is the attention weight tensor — the most interpretation-relevant
quantity in the whole model. It's computed (correctly) by
`TFTMultiHeadAttention` and immediately thrown away. The stub
`get_attention_weights` near line 1370 acknowledges this gap.

Two ways to fix it:

| Option | Pros | Cons |
| --- | --- | --- |
| Modify `AttentionLayer` and propagate `attn` up through `AttentionStack` and the main `forward` | Cleaner data flow; weights become a regular output | Requires touching three classes; existing callers' code changes; trained checkpoints still work but the change is intrusive |
| Register a `forward_hook` on every `TFTMultiHeadAttention` instance | Zero changes to existing files; works on any trained checkpoint; can be enabled/disabled at will | Slightly less obvious than a normal return value; needs explicit cleanup (handled by `with`) |

This module takes the second approach. `_AttentionCapture` registers a
hook on every MHA module at construction time. The hook intercepts the
`(out, attn)` tuple and stashes `attn` in a list. Each batch, the list is
read, then cleared.

The hook stays alive for the lifetime of the `TFTInterpreter`, so you must
either use the context manager or call `.close()` explicitly. Forgetting
this is harmless during a single Python session but can leak memory if you
construct many interpreters in a loop.

### Feature-name resolution

The model returns VSN weights as raw tensors:

| Output key | Shape | Meaning |
| --- | --- | --- |
| `static_weights` | `[B, n_static]` | weight per static feature |
| `historical_weights` | `[B, T_hist, n_hist]` | weight per (timestep, historical feature) |
| `future_weights` | `[B, T_pred, n_future]` | weight per (horizon, future feature) |

But the columns are anonymous — column `k` of `historical_weights` could be
anything. The mapping back to feature names depends on the order the
adapter assembles the input lists, and *that* order depends on
`_prepare_temporal_inputs` in `models.py` and `adapt_for_tft` in
`dataset.py`.

I traced both paths and locked the orderings into three public functions:

```python
static_feature_names(adapter)
# = static_categorical_cols + static_numeric_cols

historical_feature_names(adapter)
# = temporal_unknown_categorical_cols
# + temporal_known_categorical_cols
# + target_cols
# + temporal_unknown_numeric_cols
# + temporal_known_numeric_cols

future_feature_names(adapter)
# = temporal_known_categorical_cols + temporal_known_numeric_cols
```

These are exported so you can verify them against your `features_config`,
and they're the **single point of change** if you ever rearrange the
adapter or `_prepare_*_inputs`.

### Long-form result container

`InterpretationResult` holds four pandas DataFrames in long format:

| DataFrame | Columns |
| --- | --- |
| `static_weights_df` | `sample_id, entity_id, window_idx, feature, weight` |
| `historical_weights_df` | `sample_id, entity_id, window_idx, time_step, feature, weight` |
| `future_weights_df` | `sample_id, entity_id, window_idx, horizon, feature, weight` |
| `attention_df` | `sample_id, entity_id, window_idx, layer, head, query_pos, key_pos, weight` |

Plus `predictions` (numpy `[N, prediction_steps, num_quantiles]`),
`metadata_df`, and the scalars `historical_steps` and `prediction_steps`.

Long-form was a deliberate choice over per-sample dicts or wide tensors:

- pandas `groupby + agg` covers most aggregations the user will actually
  want, without me writing a method for each.
- Joining with the user's own metadata (entity attributes, evaluation
  metrics, business labels) is a one-liner.
- Long format survives serialization to CSV cleanly.
- The convenience helpers (`feature_importance`, `attention_by_horizon`,
  `temporal_importance`, `persistent_temporal_pattern`) are thin pivots
  over these DataFrames — easy to read, easy to extend.

### The horizon × past-lag attention view

`attention_by_horizon()` filters the full attention tensor to one specific
slice: rows where `query_pos >= historical_steps` (the future / decoder
positions) and columns where `key_pos < historical_steps` (the past /
encoder positions). It then maps:

- `query_pos → horizon = query_pos - historical_steps + 1` (1-indexed)
- `key_pos → lag = historical_steps - key_pos` (1 = most recent past step)

The result is a `prediction_steps × historical_steps` matrix that answers:
**"On average, how much weight did the model put on past lag L when
forecasting horizon H?"** — the standard TFT interpretability heatmap from
the paper.

Cells excluded from this view (decoder attending to other decoder
positions, or to itself, or encoder attending to anything) are *not* lost —
they are still in `attention_df`. You can build other slices by hand if you
care about, for example, how horizon 7 attends to horizon 3 in the
prediction window.

---

## Detailed usage

### Global feature importance

`feature_importance(scope, agg)` is the headline aggregation. It returns
one row per feature, sorted descending by importance, where importance is
the chosen aggregation of the feature's VSN weight across all samples (and
all timesteps, for temporal scopes).

```python
result.feature_importance(scope="historical", agg="mean")
#       feature  importance
# 0       sales    0.412
# 1       price    0.214
# 2  is_holiday    0.155

result.feature_importance(scope="future", agg="mean")
#     feature  importance
# 0     price    0.388
# 1   weekday    0.301

result.feature_importance(scope="static", agg="mean")
#       feature  importance
# 0      region    0.578
# 1  store_size    0.422
```

`agg` can be `"mean"`, `"median"`, or `"max"`. Mean is the right default.
Median is useful when a few samples have wildly skewed weights. Max
highlights features that are *occasionally* dominant even if usually
quiet — useful for spotting context-specific drivers.

### Per-sample explanations

For a single forecast — say, "why did the model predict X for store 47
window 1203?" — slice the long-form DataFrames by `sample_id`:

```python
sid = 0  # find the sample_id you care about in result.metadata_df

# Static features (constant per entity)
print(result.static_weights_df.query("sample_id == @sid")
       .sort_values("weight", ascending=False))

# Top-3 historical features at each historical step
hist_one = result.historical_weights_df.query("sample_id == @sid")
print(hist_one.sort_values(["time_step", "weight"], ascending=[True, False])
              .groupby("time_step").head(3))

# Attention this sample paid from each horizon to each past step
attn_one = (
    result.attention_df
    .query("sample_id == @sid "
           "and query_pos >= @result.historical_steps "
           "and key_pos < @result.historical_steps")
    .groupby(["query_pos", "key_pos"], as_index=False)["weight"].mean()
    .pivot(index="query_pos", columns="key_pos", values="weight")
)
print(attn_one.round(3))
```

You can also map `sample_id` back to entity / window via `metadata_df`:

```python
result.metadata_df.head()
#    sample_id entity_id  window_idx
# 0          0   store_1        1203
# 1          1   store_1        1204
# ...
```

### Persistent temporal patterns

Section 6.3 of the TFT paper introduces the **persistent temporal
pattern**: the average attention weight each past lag receives, averaged
over all horizons and all samples. Recurring peaks at lags 7, 14, 21
indicate the model has discovered weekly seasonality without being told
about it.

```python
ptp = result.persistent_temporal_pattern()
print(ptp.sort_values(ascending=False).head(5))
# lag
# 7     0.184    ← weekly seasonality
# 1     0.122
# 14    0.094
# 8     0.061
# 21    0.058
```

Compare across entity subsets to check whether different entities exhibit
different seasonal patterns:

```python
weekend_samples = result.metadata_df.query("entity_id in @weekend_stores")["sample_id"]
weekend_ptp = (
    result.attention_df
    .query("sample_id in @weekend_samples "
           "and query_pos >= @result.historical_steps "
           "and key_pos < @result.historical_steps")
    .assign(lag=lambda d: result.historical_steps - d["key_pos"])
    .groupby("lag")["weight"].mean()
)
```

### Plotting

Four matplotlib helpers, all returning the `Axes` for further customisation:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
result.plot_feature_importance("historical", top_k=10, ax=axes[0, 0])
result.plot_feature_importance("future", top_k=10, ax=axes[0, 1])
result.plot_attention_heatmap(ax=axes[1, 0])
result.plot_persistent_temporal_pattern(ax=axes[1, 1])
plt.tight_layout()
plt.savefig("tft_interpretation.png", dpi=120)
```

Individually:

| Helper | Shows |
| --- | --- |
| `plot_feature_importance(scope, top_k)` | Horizontal bar chart of mean VSN weight per feature |
| `plot_temporal_importance(scope, features, top_k)` | Line plot of mean VSN weight per timestep, one line per feature |
| `plot_attention_heatmap(layer, head)` | Heatmap of horizon (rows) × past-lag (cols) attention |
| `plot_persistent_temporal_pattern(layer, head)` | Line plot of mean attention by past lag, averaged over horizons |

`head` defaults to `"mean"` (averaged across heads). Pass an integer to
inspect one head specifically, or `"max"` to highlight the strongest
attender.

### Exporting

```python
paths = result.to_csvs("./interpretation_artifacts/")
# paths = {
#   "static_weights":     ".../static_weights.csv",
#   "historical_weights": ".../historical_weights.csv",
#   "future_weights":     ".../future_weights.csv",
#   "attention":          ".../attention.csv",
#   "metadata":           ".../metadata.csv",
#   "predictions":        ".../predictions.npy",
# }
```

The CSVs can be slurped into a notebook, a BI tool, or downstream analysis
without any TFT-specific code.

### Combining with predictions and uncertainty

If you trained with `QuantileLoss` (the default in `TFTTrainer`), the third
axis of `result.predictions` indexes the quantiles. To pair interpretation
with forecast uncertainty:

```python
# Assuming quantiles were [0.1, 0.5, 0.9]
preds = result.predictions          # [N, T_pred, 3]
median = preds[:, :, 1]
width = preds[:, :, 2] - preds[:, :, 0]   # P90 − P10 interval width

# Find the sample with the widest uncertainty at horizon 1
sid_uncertain = int(np.argmax(width[:, 0]))
# Now inspect that sample's VSN and attention to see whether it relied on
# unusual features or attended to atypical lags
```

This is one of the more useful patterns in practice: when uncertainty is
unusually high for a specific forecast, the interpretation usually shows
either an unusual VSN weight distribution or an attention pattern that
differs from the persistent temporal pattern. Both are actionable signals.

---

## Special notes and gotchas

**1. The decoder also attends to itself.** A TFT decoder position can
attend, causally, to earlier decoder positions and to the same position
(the diagonal). `attention_by_horizon` deliberately excludes these cells so
that the matrix matches "which past lags did horizon H use". If you want
the full picture (e.g. "did horizon 3 use horizon 1's intermediate
representation"), filter `attention_df` directly without the past-key
constraint.

**2. Static weights repeat across windows of the same entity.** In TFT the
static weight tensor is computed per-window from the static input
embedding. If the same entity has many windows in your evaluation set,
each contributes a row to `static_weights_df`. The values will be very
close (the static input is the same), but not bit-for-bit identical
because dropout is off but other stochastic elements may have been
present at training time. When you call `feature_importance("static")`
you'll be averaging over window-instances, which is what you want.

**3. VSN weights are reliance, not causality.** A high VSN weight on
"price" means the model leaned on it during the forward pass. It does *not*
mean price *causes* sales, nor that ablating price would degrade
predictions by that proportion (the model could partially recover via
correlated features). If you want causal claims, you need an intervention
study; if you want ablation-based importance, you need permutation
importance. Both are out of scope here.

**4. Memory: attention scales as `n_heads × B × T² × n_layers`.** For a
typical configuration (4 heads, batch 64, T=37, 1 attention layer) that is
~340K floats per batch — fine. For 8 heads, batch 256, T=120, 4 layers
that is ~118M floats per batch in the GPU intermediate plus a much larger
long-form DataFrame after collection. Use `max_attention_samples` to cap
how many samples per batch contribute attention rows; VSN weights and
predictions are still kept for every sample.

```python
TFTInterpreter(model, adapter, max_attention_samples=128)
```

If you want VSN weights only and no attention at all:

```python
TFTInterpreter(model, adapter, capture_attention=False)
```

**5. The hooks must be cleaned up.** Use `with TFTInterpreter(...) as
interp:` whenever possible. If you don't, call `interp.close()` explicitly
when you're done. The hooks themselves don't leak much memory but they
keep references that may surprise you later.

**6. Models without certain feature groups.** If your model has no static
features, `outputs["static_weights"]` is not produced by the forward pass
at all. The interpreter handles this with `.get()` and produces an empty
`static_weights_df` rather than crashing. Same for the other groups. Empty
DataFrames have the right column names, so downstream `groupby` calls keep
working.

**7. Encoder-only models are not supported in this version.**
`TFTEncoderOnly` has no future weights and uses temporal aggregation
(mean/max/last/attention) instead of a multi-horizon decoder. Most of the
attention-by-horizon machinery doesn't apply. If you want interpretability
for the encoder-only variant, the VSN halves of this module would still
work; the attention half would need a different pivot. Open question
whether to add it — flag if useful.

**8. The model must be in `eval()` mode.** The interpreter calls
`model.eval()` once at the top of `interpret()`. If you toggle the model
back to train mode in the same Python session and re-run, you'll get
results polluted by dropout. Don't do that.

**9. Determinism.** Provided dropout is off (model in eval mode), no other
sampling occurs in the forward pass — VSN weights and attention weights
will be exactly reproducible across runs given the same input and weights.

**10. Multi-target models.** When `target_cols` has more than one entry,
each target appears as its own historical feature in `historical_weights`,
in the same order as `target_cols`. Predictions remain
`[N, T_pred, num_quantiles]` regardless. If you trained a multi-output
model and predictions have a different shape, the existing aggregations
still work but you may need to adjust the prediction-side analysis.

---

## API reference

### `class TFTInterpreter(model, adapter, device=None, capture_attention=True, max_attention_samples=None)`

Wrapper around a trained model + adapter that runs inference and
collects interpretation data.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `nn.Module` | — | Trained `TemporalFusionTransformer` (weights loaded). |
| `adapter` | `TFTDataAdapter` | — | Same adapter type used for inference. |
| `device` | `str` or `torch.device` | model's device | Device to run on. |
| `capture_attention` | `bool` | `True` | If `False`, skip hook registration. |
| `max_attention_samples` | `int \| None` | `None` | Per-batch cap on attention rows kept (memory control). |

**Methods**

- `interpret(dataloader, max_batches=None) → InterpretationResult`
  Run inference and return collected interpretation data.
- `close()` Remove the forward hooks. Idempotent.
- `__enter__` / `__exit__` Context-manager protocol calls `close()` on exit.

### `class InterpretationResult`

Dataclass holding the long-form interpretation tables.

**Attributes**: `static_weights_df`, `historical_weights_df`,
`future_weights_df`, `attention_df`, `predictions`, `metadata_df`,
`historical_steps`, `prediction_steps`.

**Methods**

| Method | Returns |
| --- | --- |
| `feature_importance(scope, agg="mean")` | DataFrame `[feature, importance]` sorted desc. `scope ∈ {"static","historical","future"}`. `agg ∈ {"mean","median","max"}`. |
| `temporal_importance(scope, features=None)` | Wide DataFrame indexed by `time_step`/`horizon`, columns are features. |
| `attention_by_horizon(layer=None, head="mean")` | Wide DataFrame `horizon × lag` of mean attention. |
| `persistent_temporal_pattern(layer=None, head="mean")` | Series indexed by lag of mean attention across horizons. |
| `to_csvs(directory)` | Write all DataFrames + predictions to disk; returns dict of paths. |
| `plot_feature_importance(scope, top_k=15, ax=None)` | matplotlib `Axes`. |
| `plot_temporal_importance(scope, features=None, top_k=5, ax=None)` | matplotlib `Axes`. |
| `plot_attention_heatmap(layer=None, head="mean", ax=None, cmap="viridis")` | matplotlib `Axes`. |
| `plot_persistent_temporal_pattern(layer=None, head="mean", ax=None)` | matplotlib `Axes`. |

### Module-level functions

- `static_feature_names(adapter) → list[str]`
- `historical_feature_names(adapter) → list[str]`
- `future_feature_names(adapter) → list[str]`

These mirror the column ordering of the corresponding VSN weight tensors.
Use them whenever you need to reason about which weight column maps to
which feature in your config.

---

## Troubleshooting

**`ImportError: Plotting helpers require matplotlib.`**
Install matplotlib: `pip install matplotlib`. The core module itself does
not need it; only the `plot_*` methods do.

**`AssertionError` or shape mismatch when reading weights.**
The most likely cause is that the `adapter` you passed to the interpreter
has a different feature configuration than the model was trained with. The
interpreter resolves names from the adapter; the model produces weight
tensors sized by training. Re-build the adapter from the dataset that
matches your trained model.

**Empty `attention_df`.**
Either you constructed the interpreter with `capture_attention=False`, or
the model has no `TFTMultiHeadAttention` instances (the encoder-only
variant doesn't always use it depending on `temporal_aggregation`). Check
`interp._capture.num_layers` to confirm hooks were registered.

**Empty `static_weights_df` / `historical_weights_df` / `future_weights_df`.**
The corresponding feature group is empty in your config (e.g. you have no
static features). This is expected behaviour, not a bug.

**Out-of-memory when running on a large evaluation set.**
Use `max_attention_samples` to cap attention rows per batch, or set
`capture_attention=False` if you only want VSN-based importance. As a last
resort, run `interpret()` on a sample of batches via `max_batches`.

**Hooks still attached after I'm done.**
You forgot to call `close()` or use the context manager. Either is fine
to fix the issue; if you've already lost the reference, restarting the
Python session is the simplest reset.

---

## License

Same license as the parent package (MIT).
