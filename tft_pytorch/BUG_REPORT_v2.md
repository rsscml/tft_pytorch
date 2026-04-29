# TFT Codebase: Bug Report v2 — Lazy-Build Foot-Guns Run Deeper

A follow-up to `BUG_REPORT.md`. Two of the three originally identified bugs
are now fixed. One remains. **Two additional instances of the same root
cause were found at a deeper layer of the module hierarchy** — they are
critical and almost certainly the largest remaining contributor to
under-trained weights in the model.

A latent fourth issue and a couple of cosmetic items are noted at the end.

---

## Executive summary

| # | Bug | Location | Severity | Status |
|---|---|---|---|---|
| 1 | `AttentionStack.attn_layers` plain list | `models.py` line 838 | High | **Fixed** |
| 2 | VSN inner GRNs built lazily in `forward` | `models.py` lines 1019, 1034, 1047 (TFT init); 1631, 1643 (EncoderOnly init) | Critical | **Fixed at parent call sites** |
| 3 | `TFTAddAndNormLayer.layer_norm` built lazily | `models.py` lines 357–378 | Mild | Open |
| **A** | **`TFTLinearLayer` placeholder `nn.Linear(0,0)` pattern** | **`models.py` lines 217–256** | **Critical** | **Open** |
| **B** | **`TFTApplyGatingLayer` placeholder `nn.Linear(0,0)` pattern** | **`models.py` lines 305–354** | **Critical** | **Open** |
| C | `_embed_categorical` builds `nn.Embedding` in `forward` fallback | `models.py` lines 1140–1156, 1786–1802 | Latent | Open |

After applying the remaining fixes — A, B, 3, and optionally C — the model
**must be retrained from scratch**. Existing checkpoints contain random
values for the affected weights and cannot be salvaged.

### Confirmed: the trainer does not work around this

`TFTTrainer.__init__` (in `trainer.py`) calls `setup_loss()` and then
`setup_optimizer()` (line 147), which captures `self.model.parameters()`
*before* any forward pass occurs. The four `self.model(...)` call sites
in `trainer.py` (lines 283, 432, 746, 802) all live inside
`train_epoch`, `validate`, or the `TFTInference` predict methods — none
of them is reachable from `TFTTrainer.__init__`. So the optimizer
captures placeholder/empty parameters, exactly as described above. A
dummy forward in user code before constructing the trainer would mask
the symptom but is fragile (e.g. a fresh-session `load_checkpoint` with
`strict=True` would still fail on shape mismatch). The structural fixes
below are the durable solution.

---

## Background reminder: why this is happening

`optim.Adam(model.parameters(), …)` consumes the parameter iterator at
construction time and stores concrete tensor references in
`optimizer.param_groups[i]['params']`. When a module is swapped for a new
one inside `forward()` — whether via the `if x is None: build_layers()`
guard pattern or the `nn.Linear(0, 0)` placeholder pattern — the new
module's parameters are properly registered as children (so they appear
in subsequent `model.parameters()` calls and in `state_dict()`) and they
do receive gradients during `backward()`. **But the optimizer was given
a snapshot before any of this happened, and it does not re-query.**
`optimizer.step()` only updates the parameters captured at construction
time.

The placeholder pattern is a more subtle variant of the same bug. An
`nn.Linear(0, 0)` is a fully-formed `nn.Module` with weight shape `(0, 0)`
and bias shape `(0,)` — zero learnable scalars. It registers as a child
and the optimizer dutifully captures references to those zero-element
tensors. Then `forward()` does `self.layer = nn.Linear(real_in, real_out)`
and the new module replaces the placeholder. The optimizer is now updating
empty tensors; the real weights are orphaned at Kaiming initialisation.

---

## Previously reported bugs — current status

### Bug 1 — `AttentionStack.attn_layers` (FIXED)

`models.py` line 838 now reads:

```python
self.attn_layers = nn.ModuleList([
    AttentionLayer(hidden_layer_size, device, n_head, dropout_rate)
    for _ in range(num_layers)
])
```

`AttentionLayer` instances are properly registered. Verified.

### Bug 2 — VSN lazy `build_layers` (FIXED at the parent level)

`TemporalFusionTransformer.__init__` (lines 1011–1049) and
`TFTEncoderOnly.__init__` (lines 1623–1645) now eagerly call
`build_layers(num_vars=…)` immediately after constructing each VSN.
The `grn_flat` and `grn_vars` of every VSN are visible to the optimizer
at the moment it is constructed. Verified.

The `if self.grn_flat is None: self.build_layers(len(inputs))` guards
inside `VariableSelectionStatic.forward` (line 523) and
`VariableSelectionTemporal.forward` (lines 655, 664) are now unreachable
dead code. They are harmless but worth removing for clarity (see the
"Cosmetic cleanup" section).

---

## Bug 3 — `TFTAddAndNormLayer.layer_norm` (MILD, OPEN)

### Location

`models.py` lines 357–378:

```python
class TFTAddAndNormLayer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.layer_norm = None  # We'll build dynamically based on last dimension

    def forward(self, inputs):
        skip, gating_layer = inputs
        out = skip + gating_layer
        if self.layer_norm is None:
            norm_shape = out.shape[-1]
            self.layer_norm = nn.LayerNorm(norm_shape).to(self.device)
        out = self.layer_norm(out)
        return out
```

### What's wrong

Same lazy pattern as Bug 2 — `LayerNorm` is built on the first forward
call, after the optimizer has been constructed. The learnable
`weight` (γ) and `bias` (β) are never updated.

`TFTAddAndNormLayer` is used inside every `TFTGRNLayer` (line 434),
`LSTMLayer` (line 720), `AttentionLayer` (line 815), and `FinalGatingLayer`
(line 872). So every `LayerNorm` in the model is affected.

### Why this is mild

`nn.LayerNorm` initialises with `weight=1.0` and `bias=0.0`, so an
untrained `LayerNorm` is just plain mean-variance normalisation —
`out = (x − mean) / std`. That's still a sensible operation. The model
simply loses the learned scale and shift per layer, which subsequent
layers can partially compensate for. Real cost, but small.

### Recommended fix

Add a `normalized_shape` argument to `__init__` and build eagerly. Update
each call site to pass it. The shape is always known at construction
time at every call site.

```python
class TFTAddAndNormLayer(nn.Module):
    def __init__(self, device, normalized_shape):
        super().__init__()
        self.device = device
        self.layer_norm = nn.LayerNorm(normalized_shape).to(device)

    def forward(self, inputs):
        skip, gating_layer = inputs
        return self.layer_norm(skip + gating_layer)
```

Update the five call sites:

| Call site | Approx. line | Pass |
|---|---|---|
| `TFTGRNLayer.__init__` | 434 | `normalized_shape=self.output_size` |
| `LSTMLayer.__init__` | 720 | `normalized_shape=hidden_layer_size` |
| `AttentionLayer.__init__` | 815 | `normalized_shape=hidden_layer_size` |
| `FinalGatingLayer.__init__` | 872 | `normalized_shape=hidden_layer_size` |

This is a class-signature change; if anything outside `models.py`
constructs `TFTAddAndNormLayer` directly it will need updating too.

---

## Bug A — `TFTLinearLayer` placeholder pattern (CRITICAL, OPEN, NEW)

### Location

`models.py` lines 217–256:

```python
class TFTLinearLayer(nn.Module):
    def __init__(self, hidden_layer_size, device, activation=None,
                 use_time_distributed=False, use_bias=True):
        super().__init__()
        self.use_time_distributed = use_time_distributed
        self.device = device
        self.layer = nn.Linear(in_features=0, out_features=0,
                               bias=use_bias).to(self.device)
        # Will set in_features dynamically at forward if desired (or pass it in the init).
        # For simplicity, we assume you know the in_features at construction time in real usage.
        self.hidden_layer_size = hidden_layer_size
        self.activation_fn = get_activation_fn(activation)

    def forward(self, x):
        if self.layer.in_features == 0:
            in_features = x.shape[-1]
            self.layer = nn.Linear(in_features, self.hidden_layer_size,
                                   bias=self.layer.bias is not None).to(self.device)
        if self.use_time_distributed:
            out = apply_time_distributed(self.layer, x)
        else:
            out = self.layer(x)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out
```

### What's wrong

`__init__` registers an `nn.Linear(0, 0)` placeholder. The optimizer
captures references to its zero-element `weight` and `bias` tensors.
On the first forward call, `self.layer = nn.Linear(in_features, hidden_layer_size, …)`
replaces the placeholder. The new Linear is registered as a child via
`nn.Module.__setattr__` — it appears in subsequent `model.parameters()`
calls and in `state_dict()` — but the optimizer's `param_groups` still
hold references to the original (now-orphaned, zero-element) tensors.
`optimizer.step()` updates those, accomplishing nothing. The real
weight matrix stays at Kaiming initialisation forever.

### Where this hurts

`TFTLinearLayer` is used inside every `TFTGRNLayer` for `hidden_1`,
`hidden_2`, and (when present) `context_layer` — see lines 407–422.
That covers:

- The single GRN inside each VSN's `grn_flat` and every `grn_vars[i]`
  (so the user's Bug 2 fix made `grn_flat` and `grn_vars` *visible* to
  the optimizer, but the actual MLP weights *inside* them are still
  orphaned via Bug A).
- All four GRNs in `StaticContexts` (lines 562–592).
- The GRN in `StaticEnrichmentLayer`.
- The final GRN in `AttentionStack.grn_final` (line 840).

In short: **most of the actual learnable matrices in the model**.

### Recommended fix

`TFTLinearLayer` is only ever instantiated inside `TFTGRNLayer`, at three
places: `hidden_1`, `hidden_2`, and `context_layer`. The actual input
dimension to each of these is *not* always `hidden_layer_size` — for
`hidden_1` of any GRN with an explicit `output_size` (i.e. `grn_flat`
inside both VSN classes), the input shape is `hidden_layer_size *
output_size`. The GRN's existing `skip_layer` already encodes this:

```python
input_size = hidden_layer_size*self.output_size if output_size is not None else hidden_layer_size
self.skip_layer = nn.Linear(input_size, self.output_size)
```

So a default of `hidden_layer_size` would be silently wrong for `grn_flat`
and would crash with a shape mismatch on the first forward. The honest
fix is to make `input_size` a **required** argument and pass the correct
value at every call site:

```python
class TFTLinearLayer(nn.Module):
    def __init__(self, hidden_layer_size, input_size, device,
                 activation=None, use_time_distributed=False, use_bias=True):
        super().__init__()
        self.use_time_distributed = use_time_distributed
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.layer = nn.Linear(input_size, hidden_layer_size,
                               bias=use_bias).to(self.device)
        self.activation_fn = get_activation_fn(activation)

    def forward(self, x):
        if self.use_time_distributed:
            out = apply_time_distributed(self.layer, x)
        else:
            out = self.layer(x)
        return self.activation_fn(out) if self.activation_fn is not None else out
```

Update the three call sites in `TFTGRNLayer.__init__` to compute the
GRN's input size once (mirroring `skip_layer`) and pass the correct
value to each role:

| Role | `input_size` value | Reason |
|---|---|---|
| `hidden_1` | `hidden_layer_size * output_size` if `output_size is not None` else `hidden_layer_size` | Receives the same input as `skip_layer` |
| `hidden_2` | `hidden_layer_size` | Consumes `hidden_1`'s output |
| `context_layer` | `hidden_layer_size` | Every context vector in this codebase comes from a `StaticContexts` GRN with `output_size=None`, so its dim is `hidden_layer_size` |

Concretely:

```python
# In TFTGRNLayer.__init__, replace the input_size / sub-module block with:

grn_input_size = (hidden_layer_size * self.output_size
                  if output_size is not None
                  else hidden_layer_size)

self.skip_layer = nn.Linear(grn_input_size, self.output_size)

self.hidden_1 = TFTLinearLayer(
    hidden_layer_size=hidden_layer_size,
    input_size=grn_input_size,
    activation=None,
    use_time_distributed=use_time_distributed,
    device=self.device,
)

self.hidden_2 = TFTLinearLayer(
    hidden_layer_size=hidden_layer_size,
    input_size=hidden_layer_size,
    activation=None,
    use_time_distributed=use_time_distributed,
    device=self.device,
)

if additional_context:
    self.context_layer = TFTLinearLayer(
        hidden_layer_size=hidden_layer_size,
        input_size=hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed,
        device=self.device,
        use_bias=False,
    )
else:
    self.context_layer = None
```

---

## Bug B — `TFTApplyGatingLayer` placeholder pattern (CRITICAL, OPEN, NEW)

### Location

`models.py` lines 305–354:

```python
class TFTApplyGatingLayer(nn.Module):
    def __init__(self, hidden_layer_size, device, dropout_rate=0.0,
                 use_time_distributed=True, activation=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_time_distributed = use_time_distributed
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.activation_fc = nn.Linear(in_features=0, out_features=0).to(self.device)
        self.gated_fc      = nn.Linear(in_features=0, out_features=0).to(self.device)
        ...

    def forward(self, x):
        if self.activation_fc.in_features == 0:
            in_features = x.shape[-1]
            self.activation_fc = nn.Linear(in_features, self.hidden_layer_size).to(self.device)
            self.gated_fc      = nn.Linear(in_features, self.hidden_layer_size).to(self.device)
        ...
```

### What's wrong

Identical pattern to Bug A, applied twice per gate. Every gate weight
in the model is orphaned.

`TFTApplyGatingLayer` is used as `self.gate` inside:

- Every `TFTGRNLayer` (line 427) — so every GRN's gating mechanism.
- `LSTMLayer` (line 714) — the post-LSTM gate.
- `AttentionLayer` (line 815-ish, inside the layer's `gate`).
- `FinalGatingLayer` (line 867) — the gate after the attention stack.

### Recommended fix

Same approach as Bug A — make `input_size` a **required** argument and
pass the correct value at every call site. Note that the gate's
`hidden_layer_size` parameter is the *output* dimension, which differs
from the input dim in the `TFTGRNLayer` case (where input =
`hidden_layer_size`, output = `output_size`).

```python
class TFTApplyGatingLayer(nn.Module):
    def __init__(self, hidden_layer_size, input_size, device,
                 dropout_rate=0.0, use_time_distributed=True, activation=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_time_distributed = use_time_distributed
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.activation_fc = nn.Linear(input_size, hidden_layer_size).to(self.device)
        self.gated_fc      = nn.Linear(input_size, hidden_layer_size).to(self.device)

        if activation is None:
            self.activation_fn = None
        elif activation.lower() == "elu":
            self.activation_fn = F.elu
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(self, x):
        x = self.dropout(x)
        if self.use_time_distributed:
            a = apply_time_distributed(self.activation_fc, x)
            g = apply_time_distributed(self.gated_fc, x)
        else:
            a = self.activation_fc(x)
            g = self.gated_fc(x)
        if self.activation_fn is not None:
            a = self.activation_fn(a)
        g = torch.sigmoid(g)
        return a * g, g
```

Update the four call sites:

| Call site | `input_size` value | `hidden_layer_size` (= output) |
|---|---|---|
| `TFTGRNLayer` (line 427) | the GRN's `hidden_layer_size` | `self.output_size` |
| `LSTMLayer` (line 714) | `hidden_layer_size` | `hidden_layer_size` |
| `AttentionLayer` (line 810) | `hidden_layer_size` | `hidden_layer_size` |
| `FinalGatingLayer` (line 867) | `hidden_layer_size` | `hidden_layer_size` |

The `TFTGRNLayer` row is the trap: a default of `hidden_layer_size` (the
gate's own param, which equals `output_size` in this case) would have
silently built `Linear(output_size, output_size)` instead of
`Linear(hidden_layer_size, output_size)`, and crashed at runtime when
the gate received `hidden_2`'s output of dimension `hidden_layer_size`.

---

## Bug C — `_embed_categorical` lazy fallback (LATENT, OPEN)

### Location

`models.py` lines 1140–1156 (in `TemporalFusionTransformer`) and
1786–1802 (in `TFTEncoderOnly`):

```python
else:
    # Handle missing embedding gracefully
    max_idx = x.max().item() if x.numel() > 0 else 1
    vocab_size = max(max_idx + 2, 10)
    embed_layer = nn.Embedding(
        vocab_size,
        self.hidden_layer_size,
        padding_idx=vocab_size - 1
    ).to(self.device)
    self.categorical_embeddings[var_name] = embed_layer
    return self._embed_categorical(x, var_name)
```

### What's wrong

If a categorical variable name is not present in
`self.categorical_embeddings`, this fallback creates an `nn.Embedding`
inside `forward` and adds it to the `ModuleDict`. The new embedding's
parameters won't be in the optimizer for the same reason as Bugs 2/A/B.

### Why this is "latent"

Under correct usage, `categorical_embedding_dims` covers every
categorical variable name and the else-branch never fires. It only
becomes an active bug if the dict is incomplete, in which case you'd
quietly get an embedding that never trains.

### Recommended fix

Pick one of:

1. **Hard-fail** (recommended): replace the else-branch with
   `raise KeyError(f"No embedding registered for categorical variable {var_name!r}; add it to categorical_embedding_dims")`.
   Forces the configuration error to surface immediately rather than
   silently producing an untrained embedding.

2. **Remove the branch entirely** and let the natural `KeyError` from
   the missing dict lookup propagate. Equivalent effect, less code.

Apply the same change in both `TemporalFusionTransformer` and
`TFTEncoderOnly`.

### Related — delete the `_embed_categorical_buggy` method

`models.py` line 1743 defines `_embed_categorical_buggy` which has an
even worse version of the pattern: it builds a one-shot embedding inside
`forward` and **doesn't add it to the dict**, so a fresh randomly
initialised embedding is produced on *every single call*. It appears
unused; deleting it removes a footgun for future readers.

---

## Cosmetic cleanup (nice to have)

After applying the fixes above, the following dead code paths can be
removed for clarity:

- `VariableSelectionStatic.forward` line 523:
  `if self.grn_flat is None: self.build_layers(num_vars=len(inputs))`
- `VariableSelectionTemporal.forward` lines 655 and 664:
  same guard pattern.

These are now unreachable because the parent classes call
`build_layers(...)` eagerly in `__init__`. Leaving them in place is
harmless but invites confusion.

---

## Verification you can run after fixing

The original report's verification snippet is the right tool. After
applying Bugs A, B, and 3:

```python
import torch
from tft_pytorch import TemporalFusionTransformer

model = TemporalFusionTransformer(...)  # your usual constructor args

# Snapshot before any forward
n_before = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters())
opt_param_ids = {id(p) for g in opt.param_groups for p in g['params']}
n_in_opt = sum(p.numel() for g in opt.param_groups for p in g['params'])

# Trigger any lazy paths that might still exist
batch = next(iter(your_train_loader))
model_inputs = your_adapter.adapt_for_tft(batch)
model.eval()
with torch.no_grad():
    _ = model(**{k: v for k, v in model_inputs.items() if k in {
        "static_categorical","static_continuous","historical_categorical",
        "historical_continuous","future_categorical","future_continuous",
        "padding_mask"
    }})

n_after = sum(p.numel() for p in model.parameters())
missed_tensors = sum(1 for p in model.parameters() if id(p) not in opt_param_ids)

print(f"Params before forward: {n_before}")
print(f"Params captured by optimizer: {n_in_opt}")
print(f"Params after forward:  {n_after}")
print(f"Param tensors missed by optimizer: {missed_tensors}")
```

After the fixes, all four numbers should agree:
`n_before == n_in_opt == n_after`, and `missed_tensors == 0`. Any
nonzero "missed" count means another instance of the same pattern is
hiding somewhere; treat it as a regression.

A sharper variant — **and the test that would have caught Bugs A and B
in the original report** — checks parameter *tensor counts* in addition
to *element counts*, since the placeholder `nn.Linear(0, 0)` contributes
0 elements but 2 tensors:

```python
n_tensors_before = len(list(model.parameters()))
n_tensors_in_opt  = sum(1 for g in opt.param_groups for p in g['params'])
# ... forward ...
n_tensors_after  = len(list(model.parameters()))

assert n_tensors_before == n_tensors_in_opt == n_tensors_after, \
    f"Tensor count drift: before={n_tensors_before}, opt={n_tensors_in_opt}, after={n_tensors_after}"
```

A drift between `n_tensors_before` and `n_tensors_after` is the
unambiguous fingerprint of any lazy-build pattern, including the
placeholder variant where element counts can be misleading.

---

## Action plan

The fixes are independent and can be applied in any order, but they all
require a retrain to take effect, so apply them together.

1. **Apply Bugs A and B fixes (critical).** Add `input_size` arg to
   both classes, default to `hidden_layer_size`, build eagerly, drop the
   lazy guard. No call-site changes required.

2. **Apply Bug 3 fix (mild).** Add `normalized_shape` arg to
   `TFTAddAndNormLayer`, build eagerly, update the four call sites.
   Skip if you want to minimise diff size.

3. **Apply Bug C fix (latent).** Replace the silent fallback with a
   hard error. Delete `_embed_categorical_buggy`.

4. **Cosmetic cleanup.** Remove the now-dead lazy guards in
   `VariableSelectionStatic.forward` and `VariableSelectionTemporal.forward`.

5. **Run the verification snippet.** Confirm
   `missed_tensors == 0` and `n_tensors_before == n_tensors_in_opt == n_tensors_after`.

6. **Retrain from scratch.** Existing checkpoints contain random values
   for the GRN MLPs, the gates, and the LayerNorms. There is nothing
   to recover.

7. **Re-run interpretation and validate.** Feature-importance ordering
   from `result.feature_importance(...)` and the persistent temporal
   pattern from `result.persistent_temporal_pattern()` should now
   reflect actual learned structure. Forecast accuracy should also
   improve, this time meaningfully — the previous Bug 2 fix unlocked
   the VSN's *outer* selection structure but the inner GRN MLPs and
   gates were still random; fixing A and B unlocks the rest.

---

## How big a change should you expect after retraining?

- The Bug 1 fix (already applied) restored attention. Probably already
  visible in any retraining you've done since.
- The Bug 2 fix (already applied) made VSN output weights start
  reflecting input feature semantics — but the GRNs feeding into them
  were still random, so the improvement is partial.
- The Bug A and B fixes will be the largest single jump in model quality
  you've seen from a bug fix in this codebase. Every GRN and every gate
  starts learning. Static contexts become meaningful, the static-enrichment
  layer actually enriches, the post-attention GRN actually transforms.
- Bug 3 contributes a small additional bump from learned LayerNorm
  scale/shift.

If after all of this the interpretation outputs still look like noise,
the issue is no longer parameter registration — look at data
preprocessing, loss scaling, or under-training.
