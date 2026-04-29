# TFT Codebase: Bug Report and Recommended Fixes

A consolidated record of three parameter-registration bugs found in
`tft_pytorch/models.py`. They share a common root cause — parameters that
exist on the model but are invisible to the optimizer at the moment it
is constructed — but the locations and severities differ. One is already
fixed; two remain.

---

## Executive summary

| # | Bug | Location | Severity | Status |
|---|---|---|---|---|
| 1 | `AttentionStack.attn_layers` stored in a plain Python list | `models.py` line 837 | High | **Fixed** (`nn.ModuleList`) |
| 2 | VSN inner layers (`grn_flat`, `grn_vars`) built lazily inside `forward()` | `models.py` lines 487–516 (Static), 605–686 (Temporal); used at lines 1011–1040 (TFT) and 1616+ (EncoderOnly) | **Critical** | Open |
| 3 | `TFTAddAndNormLayer.layer_norm` built lazily inside `forward()` | `models.py` line 357–378 | Mild | Open |

Bugs 1 and 2 mean the affected parameters were **never updated by the
optimizer during training** and remained at random initialisation. Bug 3
has the same cause but a much milder consequence because `LayerNorm`
defaults (gamma=1, beta=0) make an untrained `LayerNorm` behave as
standard normalisation — losing only the learned scale and shift.

After applying the remaining fixes, the model **must be retrained from
scratch**. Existing checkpoints contain random values for the affected
parameters and cannot be salvaged.

---

## Background: why this is happening

PyTorch's `nn.Module.__setattr__` registers a child module in
`self._modules` only when the assigned value is itself an `nn.Module` (or
an `nn.ModuleList`/`nn.ModuleDict`/`nn.ParameterList`). Two patterns
escape this:

1. **Plain Python containers holding modules.** A `list`, `tuple`, or
   `dict` is stored as an ordinary attribute. Its contents — even if they
   are `nn.Module` instances — are not registered as submodules. They
   don't show up in `model.modules()`, `model.parameters()`, or
   `model.state_dict()`. *(Bug 1.)*

2. **Modules created after the optimizer was built.** `optim.Adam(model.parameters())`
   exhausts the parameter iterator at construction time and stores
   concrete tensor references. New parameters created later — for
   example, in a lazy-build pattern triggered by `forward()` — do appear
   in subsequent `model.parameters()` calls and do receive gradients
   during `backward()`, but the optimizer was given a snapshot and does
   not re-query. `optimizer.step()` only updates the parameters captured
   at construction. *(Bugs 2 and 3.)*

Both patterns produce the same end result: parameters that exist, receive
gradients, and look fine in `state_dict()` — but are never updated.

The empirical demonstrations live in
`test_lazy_bug.py` (the optimizer-misses-lazy-params behaviour) and
`test_ckpt_roundtrip.py` (the saved-checkpoint mismatch on reload).

---

## Bug 1 — `AttentionStack.attn_layers` as a plain list (FIXED)

### Location

`models.py`, line 837 (originally):

```python
self.attn_layers = [AttentionLayer(hidden_layer_size, device, n_head, dropout_rate)
                    for _ in range(num_layers)]
```

### What was wrong

A plain Python list, not an `nn.ModuleList`. The `AttentionLayer` instances
inside it were not registered as children of `AttentionStack`, so:

- `model.modules()` did not yield them or their `TFTMultiHeadAttention` children.
- `model.parameters()` did not include their weights — `Q`, `K`, `V`, output
  projection of every attention head.
- `model.state_dict()` did not include them, so saved checkpoints were
  *missing* these keys entirely.
- The optimizer never saw them; they stayed at random initialisation.

### Why the model still produced reasonable forecasts

Two residual connections route around the broken attention output:

- `AttentionLayer.forward` (line 824): `attn_out = self.add_norm([attn_out, x])`
- `FinalGatingLayer.forward` (line 881): `out = self.add_norm([attn_out, temporal_features])`

So even when attention produced garbage, the LSTM/VSN-derived features
flowed through unchanged. The well-trained components carried the model.

### The fix (already applied)

```python
self.attn_layers = nn.ModuleList([
    AttentionLayer(hidden_layer_size, device, n_head, dropout_rate)
    for _ in range(num_layers)
])
```

### Side note on the interpretation module

The patched `interpretation.py` includes a recursive walker
(`_find_attention_modules`) that descends into plain-Python containers
when looking for attention modules. That walker is now redundant given
the `nn.ModuleList` fix; the interpreter could revert to the simpler
`for module in model.modules():` loop. Leaving the walker in place is
harmless and adds robustness against any future regression.

---

## Bug 2 — Lazy VSN construction (CRITICAL, OPEN)

### Location

`models.py`:

- `VariableSelectionStatic.__init__` (line 487) leaves `self.grn_flat = None`
  and `self.grn_vars = nn.ModuleList()` (empty) at construction time.
- `VariableSelectionStatic.build_layers` (line 497) creates `grn_flat` and
  populates `grn_vars` — but it's only called from inside
  `VariableSelectionStatic.forward` (line 524), guarded by
  `if self.grn_flat is None`.
- `VariableSelectionTemporal` (line 605) follows the same pattern, with
  `build_layers` called from `forward` at lines 656 and 665.

These three VSN modules are constructed in `TemporalFusionTransformer.__init__`
(lines 1012, 1022, 1033). The same pattern repeats in
`TFTEncoderOnly.__init__` around lines 1616 and 1626.

### What's wrong

The construction order in `TFTTrainer.__init__`:

1. Receive a freshly-built `model`. At this point each VSN's `grn_flat`
   is `None` and `grn_vars` is an empty `nn.ModuleList`.
2. Call `setup_optimizer()` (line 147), which executes
   `optim.Adam(self.model.parameters(), …)`. The optimizer captures
   only the eagerly-built parameters: embeddings, LSTM, static-context
   modules, eager parts of GRNs, output layers. **None of the VSN's
   inner-GRN parameters exist yet, so none are captured.**
3. The first `train_epoch()` call (line 283) runs `self.model(...)`. Inside
   each VSN, `forward` triggers `build_layers(num_vars=len(inputs))`,
   which creates the inner GRN layers. They are properly registered as
   children of the VSN, so they show up in subsequent
   `model.parameters()` calls — but the optimizer was already constructed
   with the old, smaller parameter list and does not re-query.
4. Backward pass: gradients flow correctly to the lazy parameters.
5. `optimizer.step()`: only applies updates to the parameters it was
   given at construction time. The lazy GRN weights remain at random
   initialisation through every epoch.

### Empirical confirmation

`test_lazy_bug.py` reproduces this with a minimal model. Running it shows:

```
Parameters before any forward: 2 (5 elements)
Parameters captured by optimizer: 2
Parameters after first forward: 10 (104 elements)
Optimizer is missing 8 parameter tensors (99 individual learnable scalars)

After loss.backward() + optimizer.step():
  head.weight changed:                   True   (eagerly built)
  vsn.grn_flat.weight changed:           False  (lazy — never updated)
  vsn.grn_vars[0].weight changed:        False  (lazy — never updated)

Gradients on lazy params:
  vsn.grn_flat.weight.grad nonzero:      True   (gradient flowed, just never applied)
```

`test_ckpt_roundtrip.py` confirms the second consequence: `state_dict()`
*does* capture the lazy keys after the first forward, so saved checkpoints
contain them. But trying to load such a checkpoint into a fresh model
(no forward yet) fails with:

```
RuntimeError: Error(s) in loading state_dict for ParentModel:
  Unexpected key(s) in state_dict: "vsn.grn_flat.weight", "vsn.grn_flat.bias",
  "vsn.grn_vars.0.weight", "vsn.grn_vars.0.bias", ...
```

This implies the user has been running interpretation in the same Python
session as training (where lazy layers were already built from earlier
forward passes). A fresh-session resume would have hit this error.

### Impact

VSN's `grn_flat` produces the softmax weights over input features; that
output is what `TemporalFusionTransformer.forward` returns as
`static_weights`, `historical_weights`, and `future_weights`. With
`grn_flat` at random Kaiming initialisation, the softmax is a deterministic
function of the input but a random function of feature semantics. The
weights vary per sample, but they don't reflect "this feature is
important" — they reflect "a random projection of this input happens to
score higher".

`grn_vars[i]` is the per-feature transformation. With random weights, the
LSTM receives a random — but deterministic — projection of each feature.
The LSTM is trained to handle whatever input it sees, so the model still
learns. Random projections preserve a lot of information (Johnson–
Lindenstrauss territory), which is why forecast quality is reduced rather
than catastrophic.

For interpretation specifically: any conclusions drawn from VSN weights
in the current model are unreliable. The numbers are real, but their
mapping to "feature importance" is broken.

### Recommended fix

Pre-build the VSN internals immediately after construction in both
`TemporalFusionTransformer.__init__` and `TFTEncoderOnly.__init__`. The
variable counts are already passed as constructor arguments, so the
information is available.

In `TemporalFusionTransformer.__init__` around lines 1011–1040:

```python
if num_static_categorical + num_static_continuous > 0:
    self.static_variable_selection = VariableSelectionStatic(
        hidden_layer_size=hidden_layer_size,
        dropout_rate=dropout_rate,
        device=device,
    )
    # NEW — eagerly build so the optimizer captures these params
    self.static_variable_selection.build_layers(
        num_vars=num_static_categorical + num_static_continuous
    )
else:
    self.static_variable_selection = None

if num_historical_categorical + num_historical_continuous > 0:
    self.historical_variable_selection = VariableSelectionTemporal(
        hidden_layer_size=hidden_layer_size,
        static_context=True,
        dropout_rate=dropout_rate,
        device=device,
    )
    # NEW
    self.historical_variable_selection.build_layers(
        num_vars=num_historical_categorical + num_historical_continuous
    )
else:
    self.historical_variable_selection = None

if num_future_categorical + num_future_continuous > 0:
    self.future_variable_selection = VariableSelectionTemporal(
        hidden_layer_size=hidden_layer_size,
        static_context=True,
        dropout_rate=dropout_rate,
        device=device,
    )
    # NEW
    self.future_variable_selection.build_layers(
        num_vars=num_future_categorical + num_future_continuous
    )
else:
    self.future_variable_selection = None
```

Apply the analogous additions in `TFTEncoderOnly.__init__` around lines
1616 and 1626 (no `future_variable_selection` there).

The `if self.grn_flat is None: self.build_layers(...)` guards inside the
VSN `forward` methods become dead code after this fix. Leaving them in
place is harmless; removing them is cleaner. If you remove them, run the
existing test suite first to make sure nothing else relies on the lazy
behaviour.

### Optional: a more principled rewrite

If you prefer to eliminate the lazy pattern entirely (cleaner long-term),
add `num_vars` as a constructor argument to `VariableSelectionStatic` and
`VariableSelectionTemporal`, do all the building in `__init__`, and drop
both `build_layers` and the `if self.grn_flat is None` guards. This is
more invasive — it changes the VSN class signatures — but it removes the
foot-gun for good and matches standard PyTorch idioms.

---

## Bug 3 — Lazy `TFTAddAndNormLayer.layer_norm` (MILD, OPEN)

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

Same lazy pattern as Bug 2: `layer_norm` is created during the first
`forward()` call, after the optimizer has already been constructed. The
`LayerNorm`'s `weight` (gamma) and `bias` (beta) are never updated.

`TFTAddAndNormLayer` is used pervasively — inside every `TFTGRNLayer`
(line 434), `LSTMLayer` (line 720), `AttentionLayer` (line 815), and
`FinalGatingLayer` (line 871). Plus indirectly via the VSN's internal
GRNs. So this affects most `LayerNorm` operations in the model.

### Why this is mild

PyTorch initialises `LayerNorm` with `weight=1.0` and `bias=0.0` by
default, so an untrained `LayerNorm` operates as plain
mean–variance normalisation:

```
out = (x − mean) / std × 1.0 + 0.0
```

That's still a sensible operation, just without the learned scale and
shift the model could otherwise have learned. Subsequent layers can
compensate during training because their inputs are at least normalised.
The forecast-quality cost is real but small.

### Recommended fix

Add a `normalized_shape` parameter to `TFTAddAndNormLayer.__init__` and
have callers pass it explicitly. The dimension is known at construction
time at every call site (always either `hidden_layer_size` or, in
`TFTGRNLayer`, the GRN's `output_size`).

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

Then update each call site to pass the shape:

| Call site | Line | Pass |
|---|---|---|
| `TFTGRNLayer.__init__` | 434 | `normalized_shape=self.output_size` |
| `LSTMLayer.__init__` | 720 | `normalized_shape=hidden_layer_size` |
| `AttentionLayer.__init__` | 815 | `normalized_shape=hidden_layer_size` |
| `FinalGatingLayer.__init__` | 871 | `normalized_shape=hidden_layer_size` |

This is a class-signature change, so you'll need to update tests and any
external code that constructs `TFTAddAndNormLayer` directly. If that's
a concern, an alternative minimal fix is to add a one-line eager-build
inside each containing class's `__init__` that calls a public
`build_layer_norm(normalized_shape)` method — same idea as the Bug 2 fix.

---

## Verification you can run on your installation

These can confirm or refute each finding on your actual codebase, in case
something has shifted between the files I inspected and your live code.

### Confirm Bug 2 (VSN lazy build)

```python
import torch
from tft_pytorch import TemporalFusionTransformer

model = TemporalFusionTransformer(...)  # your usual args

# Before any forward
n_before = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters())
n_in_opt = sum(p.numel() for g in opt.param_groups for p in g['params'])

# Trigger lazy build
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
print(f"Params before forward: {n_before}")
print(f"Params captured by optimizer: {n_in_opt}")
print(f"Params after forward: {n_after}")
print(f"Lazy/missed by optimizer: {n_after - n_in_opt}")
```

A non-zero "missed" count confirms the bug is present in your build.

### Confirm Bug 3 (lazy `LayerNorm`)

```python
# Run the same forward pass as above, then:
ln_params = [(name, p) for name, p in model.named_parameters()
             if 'layer_norm' in name]
print(f"LayerNorm params after forward: {len(ln_params)}")
# Compare against n_in_opt: any LayerNorm params present here that
# weren't in the optimizer's capture are affected.
```

### Confirm checkpoint round-trip would fail

```python
# Run training for one epoch, save the checkpoint. Then in a fresh
# Python process or after deleting and reconstructing `model`:
fresh = TemporalFusionTransformer(...)  # same args
ckpt = torch.load("checkpoints/best_model.pt", weights_only=False)
fresh.load_state_dict(ckpt['model_state_dict'])  # strict=True is default
# Expect: RuntimeError: Unexpected key(s) in state_dict: ...
```

If this raises, the bug is present and your existing inference flow only
works because lazy layers happen to already be built on the same `model`
object that ran training.

---

## Action plan

The three fixes are independent and can be applied in any order, but
since they all require a retrain to take effect, it's most efficient to
apply them together.

**Step 1 — Apply Bug 2 fix (critical).**
Add the `build_layers(...)` calls in `TemporalFusionTransformer.__init__`
and `TFTEncoderOnly.__init__`. Optionally remove the now-dead lazy guards
in the VSN `forward` methods.

**Step 2 — Apply Bug 3 fix (mild) if you have the appetite.**
Either the class-signature change or the per-call-site eager-build
workaround. Skip this if you'd rather minimise diff size — the behavioural
impact is small.

**Step 3 — Confirm with the verification snippets above.**
After fixing, the "Lazy/missed by optimizer" count from the Bug 2
verification should be zero, and the checkpoint round-trip into a fresh
model should succeed without strict-mode errors.

**Step 4 — Retrain from scratch.**
The existing checkpoints contain random VSN parameters and (depending on
how recently you fixed Bug 1) random or absent attention parameters.
There is nothing to recover. Train fresh.

**Step 5 — Re-run interpretation and validate.**
Compare `result.feature_importance("historical")` and
`result.persistent_temporal_pattern()` between the old (random VSN) and
new (trained VSN) checkpoints. The differences should be substantial:

- Feature importance ordering will change and stabilise across runs.
- The persistent temporal pattern will move from approximately uniform
  (or arbitrarily peaked) to having interpretable structure — peaks at
  multiples of your data's natural seasonality (e.g., lag 7 for daily
  data with weekly periodicity).
- Forecast accuracy should improve modestly, mostly via the now-functional
  attention head and learned VSN feature reliance.

If after retraining the interpretation outputs still look like noise,
something else is wrong — most likely a data preprocessing issue or an
under-trained model rather than another registration bug, since the
verification snippets would catch any further instances.

---

## Files referenced in this report

| File | Purpose |
|---|---|
| `interpretation.py` | The interpretation module (drop-in to `tft_pytorch/`) |
| `INTERPRETATION.md` | Detailed usage and design documentation |
| `example_interpretation.py` | End-to-end usage example |
| `test_interpretation.py` | Smoke test for the long-form converters and aggregations |
| `test_walker.py` | Verifies that `_find_attention_modules` handles plain-list nesting |
| `test_lazy_bug.py` | Empirically demonstrates the optimizer-misses-lazy-params behaviour |
| `test_ckpt_roundtrip.py` | Empirically demonstrates the `load_state_dict` failure mode |
