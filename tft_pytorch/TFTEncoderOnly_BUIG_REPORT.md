# TFTEncoderOnly Fix Proposals

## Context

This report summarizes the proposed fixes for the `TFTEncoderOnly` implementation in the fresh `models.py` version. The earlier lazy-initialization issue appears fixed: `TFTLinearLayer`, `TFTApplyGatingLayer`, and `TFTAddAndNormLayer` now create trainable submodules eagerly in `__init__` rather than replacing modules during `forward()`.

The remaining issues discussed here are specific to `TFTEncoderOnly`:

1. Incorrect static-context handling when the model has historical variables but no static variables.
2. Unsafe use of bare `.squeeze()` on static categorical inputs.
3. Example constructors missing `categorical_embedding_dims` after the embedding fallback was changed to hard-fail.
4. Recommended guard to enforce the model's current assumption that categorical embedding dimensions equal `hidden_layer_size`.

---

## Executive Summary

| # | Area | Severity | Current behavior | Proposed fix |
|---|---|---:|---|---|
| 1 | Historical VSN static-context handling | High | Historical VSN is built with `static_context=True` even when there are no static variables. | Set `static_context` based on whether static variables exist. |
| 2 | Static categorical preprocessing | Medium | Uses bare `.squeeze()`, which can remove the batch dimension for `batch_size=1`. | Use shape-specific `squeeze(-1)` and validate expected shapes. |
| 3 | Example constructors | Medium | Examples declare categorical variables but omit `categorical_embedding_dims`, causing the new hard-fail path to raise `KeyError`. | Add complete embedding specs to each example. |
| 4 | Uniform embedding dimension contract | Medium | VSNs assume all variable representations have width `hidden_layer_size`. | Add an explicit validation guard for categorical embedding dims. |

---

## Background: Uniform Categorical Embedding Dimension Assumption

The current implementation assumes that every variable representation passed into a variable selection network has width `hidden_layer_size`.

This is true for continuous variables because each continuous input is projected using `nn.Linear(1, hidden_layer_size)`.

It is true for categorical variables only if each categorical embedding uses:

```python
embedding_dim == hidden_layer_size
```

If this invariant is maintained, the current VSN design is acceptable. If not, the more general solution would be to pass explicit `input_size` values into each VSN and GRN. Since your actual usage uses uniform categorical embedding dimensions equal to `hidden_layer_size`, the lighter guard-based approach is sufficient.

---

# Fix 1: Correct static-context handling in `TFTEncoderOnly`

## Problem

`TFTEncoderOnly.__init__` currently creates the historical variable selection network with `static_context=True` unconditionally:

```python
self.historical_variable_selection = VariableSelectionTemporal(
    hidden_layer_size=hidden_layer_size,
    static_context=True,
    dropout_rate=dropout_rate,
    device=device
)
```

This is wrong when the model has no static variables. In that case, `forward()` may call the historical VSN without passing a context tuple:

```python
historical_features, historical_weights = self.historical_variable_selection(
    historical_inputs
)
```

But the VSN was constructed to expect:

```python
(historical_inputs, context)
```

This leads to an unpacking/runtime error in historical-only configurations.

## Proposed patch

Add a `has_static_inputs` flag in `TFTEncoderOnly.__init__`:

```python
has_static_inputs = (num_static_categorical + num_static_continuous) > 0
self.has_static_inputs = has_static_inputs
```

Then use it consistently.

### Replace static VSN construction with:

```python
# Variable Selection Networks
if has_static_inputs:
    self.static_variable_selection = VariableSelectionStatic(
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        device=device
    )
    self.static_variable_selection.build_layers(
        num_vars=num_static_categorical + num_static_continuous
    )
else:
    self.static_variable_selection = None
```

### Replace historical VSN construction with:

```python
if num_historical_categorical + num_historical_continuous > 0:
    self.historical_variable_selection = VariableSelectionTemporal(
        hidden_layer_size=hidden_layer_size,
        static_context=has_static_inputs,
        dropout_rate=dropout_rate,
        device=device
    )
    self.historical_variable_selection.build_layers(
        num_vars=num_historical_categorical + num_historical_continuous
    )
else:
    self.historical_variable_selection = None
```

### Replace the historical VSN branch in `forward()` with:

```python
if self.historical_variable_selection.static_context:
    historical_features, historical_weights = self.historical_variable_selection(
        (historical_inputs, stat_vec)
    )
else:
    historical_features, historical_weights = self.historical_variable_selection(
        historical_inputs
    )
```

## Why this works

The `VariableSelectionTemporal` instance and the `forward()` call now agree on whether a static context vector is expected.

Historical-only models will build:

```python
static_context=False
```

and static + historical models will build:

```python
static_context=True
```

---

# Fix 2: Make static categorical preprocessing batch-size safe

## Problem

`TFTEncoderOnly._prepare_static_inputs()` currently uses:

```python
embedded = self._embed_categorical(cat_var.squeeze(), f"static_cat_{i}")
```

Bare `.squeeze()` removes all singleton dimensions. For `batch_size=1`, an input shaped `[1, 1]` becomes a scalar, losing the batch dimension. This can later break concatenation or variable selection.

## Proposed patch

Replace `_prepare_static_inputs()` with the following safer version:

```python
def _prepare_static_inputs(
    self,
    static_categorical: List[torch.Tensor],
    static_continuous: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Prepare static inputs."""
    static_inputs = []

    # Process categorical variables
    for i, cat_var in enumerate(static_categorical):
        # Accept [B] or [B, 1], but preserve batch dimension.
        if cat_var.dim() == 2 and cat_var.shape[-1] == 1:
            cat_var = cat_var.squeeze(-1)
        elif cat_var.dim() != 1:
            raise ValueError(
                f"static_categorical[{i}] must have shape [B] or [B, 1], "
                f"got {tuple(cat_var.shape)}"
            )

        embedded = self._embed_categorical(cat_var.long(), f"static_cat_{i}")
        static_inputs.append(embedded)

    # Process continuous variables
    for i, cont_var in enumerate(static_continuous):
        if cont_var.dim() == 1:
            cont_var = cont_var.unsqueeze(-1)
        elif cont_var.dim() != 2 or cont_var.shape[-1] != 1:
            raise ValueError(
                f"static_continuous[{i}] must have shape [B] or [B, 1], "
                f"got {tuple(cont_var.shape)}"
            )

        transformed = self.static_continuous_transforms[i](cont_var)
        static_inputs.append(transformed)

    return static_inputs
```

## Why this works

This preserves the batch dimension for `batch_size=1` while still accepting both common input shapes:

```text
[B]
[B, 1]
```

It also validates malformed static input shapes early.

---

# Fix 3: Harden historical input preprocessing

This is not strictly required for the original bug, but it makes the encoder-only model much safer and easier to debug.

## Proposed patch

Replace `_prepare_historical_inputs()` with:

```python
def _prepare_historical_inputs(
    self,
    categorical_vars: List[torch.Tensor],
    continuous_vars: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Prepare historical temporal inputs."""
    temporal_inputs = []

    # Process categorical variables
    for i, cat_var in enumerate(categorical_vars):
        # Accept [B, T] or [B, T, 1].
        if cat_var.dim() == 3 and cat_var.shape[-1] == 1:
            cat_var = cat_var.squeeze(-1)
        elif cat_var.dim() != 2:
            raise ValueError(
                f"historical_categorical[{i}] must have shape [B, T] or [B, T, 1], "
                f"got {tuple(cat_var.shape)}"
            )

        batch_size, time_steps = cat_var.shape
        cat_var_flat = cat_var.reshape(-1).long()

        embedded = self._embed_categorical(cat_var_flat, f"historical_cat_{i}")

        embed_dim = embedded.shape[-1]
        embedded = embedded.reshape(batch_size, time_steps, embed_dim)
        temporal_inputs.append(embedded)

    # Process continuous variables
    for i, cont_var in enumerate(continuous_vars):
        if cont_var.dim() == 2:
            cont_var = cont_var.unsqueeze(-1)
        elif cont_var.dim() != 3 or cont_var.shape[-1] != 1:
            raise ValueError(
                f"historical_continuous[{i}] must have shape [B, T] or [B, T, 1], "
                f"got {tuple(cont_var.shape)}"
            )

        transformed = apply_time_distributed(
            self.historical_continuous_transforms[i],
            cont_var
        )
        temporal_inputs.append(transformed)

    return temporal_inputs
```

## Why this works

This makes temporal input handling explicit and robust for both:

```text
[B, T]
[B, T, 1]
```

It also ensures categorical tensors are cast to `long` before embedding lookup.

---

# Fix 4: Update example constructors with `categorical_embedding_dims`

## Problem

The model now correctly raises an error if a categorical variable is used without a registered embedding. Therefore, example functions that declare categorical variables must also pass `categorical_embedding_dims`.

## Stock prediction example patch

Use a local `hidden_layer_size` variable so the embedding dimensions stay synchronized:

```python
def create_stock_prediction_example():
    """
    Example: Predict average return over next 3 candles using past 30 candles.
    """
    hidden_layer_size = 128

    model = TFTEncoderOnly(
        hidden_layer_size=hidden_layer_size,
        num_attention_heads=4,
        num_lstm_layers=2,
        num_attention_layers=1,
        dropout_rate=0.1,

        # Static features
        num_static_categorical=2,   # ticker_id, sector
        num_static_continuous=1,    # log_market_cap

        # Historical features
        num_historical_categorical=2,  # day_of_week, is_market_open
        num_historical_continuous=10,  # open, high, low, close, volume, RSI, etc.

        categorical_embedding_dims={
            "static_cat_0": (500, hidden_layer_size),      # ticker_id
            "static_cat_1": (11, hidden_layer_size),       # sector
            "historical_cat_0": (7, hidden_layer_size),    # day_of_week
            "historical_cat_1": (2, hidden_layer_size),    # is_market_open
        },

        historical_steps=30,
        output_size=1,
        output_type='regression',
        temporal_aggregation='attention',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return model
```

## Classification example patch

```python
def create_classification_example():
    """
    Example: Classify future direction using historical features.
    """
    hidden_layer_size = 128

    model = TFTEncoderOnly(
        hidden_layer_size=hidden_layer_size,
        num_attention_heads=4,
        num_lstm_layers=1,
        num_attention_layers=1,
        dropout_rate=0.2,

        num_static_categorical=0,
        num_static_continuous=0,
        num_historical_categorical=1,
        num_historical_continuous=5,

        categorical_embedding_dims={
            "historical_cat_0": (4, hidden_layer_size),
        },

        historical_steps=50,
        output_type='classification',
        num_classes=3,
        temporal_aggregation='last',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return model
```

Adjust the vocabulary sizes to match your actual encodings.

---

# Fix 5: Add a guard for the uniform embedding-dimension contract

## Problem

The current VSN implementation assumes each input variable representation has width `hidden_layer_size`. That assumption is valid only when every categorical embedding dimension equals `hidden_layer_size`.

## Proposed patch

Add this after creating `self.categorical_embeddings` in `TFTEncoderOnly.__init__`:

```python
if categorical_embedding_dims:
    for var_name, (_, embed_dim) in categorical_embedding_dims.items():
        if embed_dim != hidden_layer_size:
            raise ValueError(
                f"{var_name} has embedding_dim={embed_dim}, but this "
                f"TFTEncoderOnly implementation requires all categorical "
                f"embedding dimensions to equal hidden_layer_size={hidden_layer_size}."
            )
```

Optionally add the same guard to the full `TemporalFusionTransformer` class as well.

## Why this matters

Without the guard, a future configuration such as:

```python
categorical_embedding_dims={
    "historical_cat_0": (7, 32),
}
```

with:

```python
hidden_layer_size = 128
```

would break VSN shape assumptions.

---

# Minimal validation tests

Run the following tests after applying the patches.

## Test 1: Historical-only, no static variables

```python
import torch

model = TFTEncoderOnly(
    hidden_layer_size=16,
    num_attention_heads=4,
    num_static_categorical=0,
    num_static_continuous=0,
    num_historical_categorical=0,
    num_historical_continuous=3,
    historical_steps=10,
    output_type="regression",
)

out = model(
    historical_continuous=[torch.randn(2, 10) for _ in range(3)]
)

assert out["output"].shape == (2, 1)
```

## Test 2: Historical categorical, no static variables

```python
import torch

model = TFTEncoderOnly(
    hidden_layer_size=16,
    num_attention_heads=4,
    num_static_categorical=0,
    num_static_continuous=0,
    num_historical_categorical=1,
    num_historical_continuous=2,
    categorical_embedding_dims={
        "historical_cat_0": (7, 16),
    },
    historical_steps=10,
    output_type="classification",
    num_classes=3,
)

out = model(
    historical_categorical=[torch.randint(0, 7, (2, 10))],
    historical_continuous=[torch.randn(2, 10) for _ in range(2)]
)

assert out["logits"].shape == (2, 3)
```

## Test 3: Batch-size-1 static categorical

```python
import torch

model = TFTEncoderOnly(
    hidden_layer_size=16,
    num_attention_heads=4,
    num_static_categorical=1,
    num_static_continuous=0,
    num_historical_categorical=0,
    num_historical_continuous=2,
    categorical_embedding_dims={
        "static_cat_0": (5, 16),
    },
    historical_steps=10,
)

out = model(
    static_categorical=[torch.randint(0, 5, (1, 1))],
    historical_continuous=[torch.randn(1, 10) for _ in range(2)]
)

assert out["output"].shape == (1, 1)
```

## Test 4: No parameter drift after forward

```python
import torch

model = TFTEncoderOnly(
    hidden_layer_size=16,
    num_attention_heads=4,
    num_static_categorical=1,
    num_static_continuous=1,
    num_historical_categorical=1,
    num_historical_continuous=2,
    categorical_embedding_dims={
        "static_cat_0": (5, 16),
        "historical_cat_0": (7, 16),
    },
    historical_steps=10,
)

n_params_before = sum(p.numel() for p in model.parameters())
n_tensors_before = len(list(model.parameters()))

optimizer = torch.optim.Adam(model.parameters())
opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

_ = model(
    static_categorical=[torch.randint(0, 5, (2, 1))],
    static_continuous=[torch.randn(2, 1)],
    historical_categorical=[torch.randint(0, 7, (2, 10))],
    historical_continuous=[torch.randn(2, 10) for _ in range(2)],
)

n_params_after = sum(p.numel() for p in model.parameters())
n_tensors_after = len(list(model.parameters()))
missed = [p for p in model.parameters() if id(p) not in opt_param_ids]

assert n_params_before == n_params_after
assert n_tensors_before == n_tensors_after
assert len(missed) == 0
```

---

# Recommended implementation order

1. Add `has_static_inputs` in `TFTEncoderOnly.__init__`.
2. Use `has_static_inputs` when constructing `VariableSelectionTemporal`.
3. Update the historical VSN call in `forward()` to use `self.historical_variable_selection.static_context`.
4. Replace bare `.squeeze()` in `_prepare_static_inputs()`.
5. Harden `_prepare_historical_inputs()`.
6. Add the categorical embedding dimension guard.
7. Update example constructors with `categorical_embedding_dims`.
8. Run the four validation tests above.

---

# Final assessment

The lazy-initialization issue is resolved in the current version. The remaining `TFTEncoderOnly` problems are conventional shape/configuration issues rather than optimizer-orphaning bugs.

The most important fix is the `static_context` correction. Without it, historical-only encoder models can fail even though the rest of the architecture is valid.

The second most important fix is replacing bare `.squeeze()`, because it creates a fragile batch-size-1 edge case.

With these changes and the uniform embedding-dimension guard, `TFTEncoderOnly` should be much safer for both static + historical and historical-only use cases.
