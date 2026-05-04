"""
Example: per-entity interpretation with TFTInterpreter.

Builds on the existing global-aggregation example by showing how to scope
every plot, every aggregation, and the underlying long-form DataFrames to
a single entity or a chosen subset.

Three patterns, in order of how often you'll reach for them:

1. Single-shot scoping with ``entity_ids=`` on existing methods.
2. Build a *narrowed* result with ``.for_entity()`` / ``.filter()`` and reuse it.
3. Faceted comparison with ``plot_*_by_entity`` (one subplot per entity).
"""
import matplotlib.pyplot as plt
import torch

from tft_pytorch import (
    OptimizedTFTDataset,
    create_tft_dataloader,
    TemporalFusionTransformer,
)
from tft_pytorch.interpretation import TFTInterpreter


# --- Assume model + dataloader/adapter already set up (see global example) ---
# model = TemporalFusionTransformer(...); model.load_state_dict(...); model.eval()
# test_loader, test_adapter = create_tft_dataloader(test_ds, batch_size=64,
#                                                   shuffle=False)


with TFTInterpreter(
    model,
    test_adapter,
    capture_attention=True,
    max_attention_samples=128,
) as interp:
    result = interp.interpret(test_loader, max_batches=10)


# ---------------------------------------------------------------------------
# 1. SINGLE-SHOT SCOPING
# ---------------------------------------------------------------------------
# Every aggregation and plot method now takes an `entity_ids=` kwarg.
# Pass a single id or a list — the title gets a small suffix automatically.

print("Top features for store_42 (historical):")
print(result.feature_importance(scope="historical", entity_ids="store_42").head())

# A single plot, scoped to one entity:
fig, ax = plt.subplots(figsize=(7, 4))
result.plot_feature_importance(
    scope="historical", top_k=10, entity_ids="store_42", ax=ax,
)
fig.tight_layout()
fig.savefig("fi_store_42.png", dpi=120)

# A heatmap for a small set of entities (averaged across them):
fig, ax = plt.subplots(figsize=(8, 4))
result.plot_attention_heatmap(
    entity_ids=["store_42", "store_99", "store_100"], ax=ax,
)
fig.tight_layout()
fig.savefig("attn_three_stores.png", dpi=120)


# ---------------------------------------------------------------------------
# 2. NARROWED RESULT — REUSE FOR MULTIPLE PLOTS / EXPORTS
# ---------------------------------------------------------------------------
# .for_entity() / .for_entities() / .filter() return a brand-new
# InterpretationResult with the same API. Sample_ids are preserved (not
# renumbered), so you can still cross-reference back to the original eval
# run. `predictions` is sliced positionally to remain aligned with metadata.

s42 = result.for_entity("store_42")
print(f"store_42: {len(s42.metadata_df)} samples, predictions shape {s42.predictions.shape}")

# Now every existing method operates on this single entity:
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
s42.plot_feature_importance("historical", top_k=10, ax=axes[0, 0])
s42.plot_feature_importance("future", top_k=10, ax=axes[0, 1])
s42.plot_attention_heatmap(ax=axes[1, 0])
s42.plot_persistent_temporal_pattern(ax=axes[1, 1])
fig.suptitle("Interpretation: store_42", fontsize=14)
fig.tight_layout()
fig.savefig("interp_store_42.png", dpi=120)

# Persist per-entity artifacts to disk:
s42.to_csvs("./interp_artifacts/store_42/")

# Filters compose; you can also filter by sample_id or window_idx:
recent_window = result.filter(entity_ids=["store_42", "store_99"], window_idx=[5, 6])

# And inspect a single forecast (sample-level):
one_forecast = result.filter(sample_ids=[0])
print(one_forecast.historical_weights_df.head())


# ---------------------------------------------------------------------------
# 3. FACETED COMPARISON: plot_*_by_entity
# ---------------------------------------------------------------------------
# When you want to *compare* entities side by side, the by_entity plots draw
# one panel per entity in a grid. If `entity_ids` is omitted they use every
# entity present — sensible for 2–8 entities, but pick a subset for more.

top_entities = ["store_42", "store_99", "store_100", "store_117"]

# Feature importance, side by side:
fig = result.plot_feature_importance_by_entity(
    entity_ids=top_entities, scope="historical", top_k=8, ncols=2,
)
fig.savefig("fi_by_entity.png", dpi=120, bbox_inches="tight")

# Attention heatmap with shared colour scale (so darker = darker across panels):
fig = result.plot_attention_heatmap_by_entity(
    entity_ids=top_entities, ncols=2, shared_scale=True,
)
fig.savefig("attn_by_entity.png", dpi=120, bbox_inches="tight")

# Temporal importance — same top-k features used for every panel so the
# legend is comparable across entities:
fig = result.plot_temporal_importance_by_entity(
    entity_ids=top_entities, scope="historical", top_k=4, ncols=2,
)
fig.savefig("temporal_by_entity.png", dpi=120, bbox_inches="tight")

# Persistent pattern — comparing seasonality strength across entities is
# often where the real insight lives (e.g. "store_42 has a sharp lag-7 peak,
# store_99 doesn't"):
fig = result.plot_persistent_temporal_pattern_by_entity(
    entity_ids=top_entities, ncols=2,
)
fig.savefig("persistent_by_entity.png", dpi=120, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Iterating over all entities
# ---------------------------------------------------------------------------
# Useful when you want to dump per-entity reports to disk:
for eid in result.entities:
    sub = result.for_entity(eid)
    if sub.metadata_df.empty:
        continue
    sub.to_csvs(f"./interp_artifacts/{eid}/")
