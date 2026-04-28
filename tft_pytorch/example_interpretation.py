"""
Example: using TFTInterpreter to explain forecasts.

Drops into your existing pipeline right after you've trained a model
and have a test dataloader + adapter.
"""
import torch
import matplotlib.pyplot as plt

from tft_pytorch import (
    OptimizedTFTDataset,
    create_tft_dataloader,
    TemporalFusionTransformer,
)
from tft_pytorch.interpretation import TFTInterpreter


# --- Assume model is already loaded with trained weights ---
# model = TemporalFusionTransformer(...)
# checkpoint = torch.load("./checkpoints/best_model.pt", map_location="cpu",
#                         weights_only=False)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# --- And a dataloader / adapter for your evaluation data ---
# test_ds = OptimizedTFTDataset("test.csv", FEATURES, H, F, mode="test", ...)
# test_loader, test_adapter = create_tft_dataloader(test_ds, batch_size=64,
#                                                   shuffle=False)


# --- Run the interpreter ----------------------------------------------------
# Use as a context manager so attention hooks are cleaned up on exit.
# `max_attention_samples` keeps memory bounded — attention is [n_head, B, T, T]
# and can balloon for large evaluation sets. VSN weights and predictions are
# kept for every sample regardless.
with TFTInterpreter(
    model,
    test_adapter,
    capture_attention=True,
    max_attention_samples=128,  # keep ~128 samples worth of attention
) as interp:
    result = interp.interpret(test_loader, max_batches=10)


# --- Inspect tidy long-form data --------------------------------------------
print(result.feature_importance(scope="historical").head())
#       feature  importance
# 0       sales    0.412
# 1       price    0.214
# 2  is_holiday    0.155
# ...

print(result.feature_importance(scope="future").head())
#     feature  importance
# 0     price    0.388
# 1   weekday    0.301
# ...

print(result.feature_importance(scope="static").head())

# Persistent attention pattern across all forecast horizons
ptp = result.persistent_temporal_pattern()
print("Top 3 lags by mean attention:")
print(ptp.sort_values(ascending=False).head(3))
# lag
# 7    0.184    <- weekly seasonality
# 1    0.122
# 14   0.094

# Attention as a horizon × past-lag matrix
heatmap = result.attention_by_horizon(head="mean")
print(heatmap.round(3))


# --- Visualise --------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
result.plot_feature_importance("historical", top_k=10, ax=axes[0, 0])
result.plot_feature_importance("future", top_k=10, ax=axes[0, 1])
result.plot_attention_heatmap(ax=axes[1, 0])
result.plot_persistent_temporal_pattern(ax=axes[1, 1])
plt.tight_layout()
plt.savefig("tft_interpretation.png", dpi=120)


# --- Save the raw long-form data for downstream analysis --------------------
paths = result.to_csvs("./interpretation_artifacts/")
# Now you have:
#   static_weights.csv      one row per (sample, feature)
#   historical_weights.csv  one row per (sample, time_step, feature)
#   future_weights.csv      one row per (sample, horizon, feature)
#   attention.csv           one row per (sample, layer, head, q_pos, k_pos)
#   metadata.csv            one row per sample (entity_id, window_idx, ...)
#   predictions.npy         [N, prediction_steps, num_quantiles]


# --- Explain ONE specific forecast ------------------------------------------
# For per-sample explanations, run on a single batch and slice by sample_id:
sample_id = 0
print("\n--- Explanation for sample_id=0 ---")

print("Static features (constant per entity):")
print(result.static_weights_df
      .query("sample_id == @sample_id")
      .sort_values("weight", ascending=False))

print("\nHistorical: top-3 features at each historical step:")
hist_one = result.historical_weights_df.query("sample_id == @sample_id")
print(
    hist_one.sort_values(["time_step", "weight"], ascending=[True, False])
    .groupby("time_step")
    .head(3)
)

print("\nAttention from each horizon to the past (mean over heads):")
attn_one = (
    result.attention_df
    .query("sample_id == @sample_id and query_pos >= @result.historical_steps "
           "and key_pos < @result.historical_steps")
    .groupby(["query_pos", "key_pos"], as_index=False)["weight"].mean()
    .pivot(index="query_pos", columns="key_pos", values="weight")
)
print(attn_one.round(3))
