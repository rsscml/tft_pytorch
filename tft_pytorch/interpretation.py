"""
tft_pytorch.interpretation
==========================

Interpretation utilities for the Temporal Fusion Transformer.

The TFT exposes two complementary interpretability signals:

1. **Variable Selection Network (VSN) weights** — for each variable group
   (static, historical, future), a softmax-weight per feature telling you how
   much the model "leaned on" that feature. The base model already returns
   these as ``static_weights``, ``historical_weights``, ``future_weights``
   in its forward output.

2. **Multi-head self-attention weights** — for each query timestep, a
   distribution over key timesteps, telling you which past steps the model
   attended to when producing each forecast horizon. The base model
   *computes* these inside ``TFTMultiHeadAttention`` but discards them in
   ``AttentionLayer``. This module captures them with a forward hook,
   without touching the model code.

Quick start
-----------
>>> from tft_pytorch.interpretation import TFTInterpreter
>>> with TFTInterpreter(model, adapter) as interp:
...     result = interp.interpret(test_loader, max_batches=4)
>>> result.feature_importance("historical").head()
>>> result.attention_by_horizon().head()
>>> result.plot_feature_importance("historical", top_k=10)
>>> result.plot_attention_heatmap()
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

# Imported lazily inside the module to avoid circular imports at package
# load time; we only need the type for isinstance checks on hooks.
from .models import TFTMultiHeadAttention

__all__ = [
    "TFTInterpreter",
    "InterpretationResult",
    "historical_feature_names",
    "future_feature_names",
    "static_feature_names",
]


# ---------------------------------------------------------------------------
# Feature-name resolution
# ---------------------------------------------------------------------------
#
# The VSN concatenates inputs in a fixed order. We replicate that order here
# so each weight column can be mapped back to the human-readable feature
# name. The orderings below match _prepare_temporal_inputs and
# _prepare_static_inputs in models.py and the assembly in
# TFTDataAdapter.adapt_for_tft.

def static_feature_names(adapter) -> List[str]:
    """Return the ordered list of static feature names matching
    ``static_weights`` columns.

    Order: static_categorical, then static_continuous.
    """
    return list(adapter.static_categorical_cols) + list(adapter.static_numeric_cols)


def historical_feature_names(adapter) -> List[str]:
    """Return the ordered list of historical feature names matching
    ``historical_weights`` columns.

    Order:
      1. temporal_unknown_categorical
      2. temporal_known_categorical
      3. targets (used as historical lag features)
      4. temporal_unknown_numeric
      5. temporal_known_numeric
    """
    return (
        list(adapter.temporal_unknown_categorical_cols)
        + list(adapter.temporal_known_categorical_cols)
        + list(adapter.target_cols)
        + list(adapter.temporal_unknown_numeric_cols)
        + list(adapter.temporal_known_numeric_cols)
    )


def future_feature_names(adapter) -> List[str]:
    """Return the ordered list of future feature names matching
    ``future_weights`` columns.

    Order: temporal_known_categorical, then temporal_known_numeric.
    """
    return (
        list(adapter.temporal_known_categorical_cols)
        + list(adapter.temporal_known_numeric_cols)
    )


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------

class _AttentionCapture:
    """Registers forward hooks on every TFTMultiHeadAttention module in a
    model and stores the attention weights from the most recent forward
    pass. One entry per attention layer, in network order.

    Hook output for ``TFTMultiHeadAttention.forward`` is the tuple
    ``(out, attn)`` where ``attn`` has shape ``[n_head, B, T, T]``.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._buffers: List[torch.Tensor] = []
        # Register in module-iteration order so layer index is stable.
        for module in model.modules():
            if isinstance(module, TFTMultiHeadAttention):
                self._handles.append(
                    module.register_forward_hook(self._make_hook(len(self._handles)))
                )
        # Track how many attention layers we discovered. If zero, callers
        # will get an empty attention DataFrame rather than a crash.
        self.num_layers = len(self._handles)

    def _make_hook(self, layer_idx: int):
        # We can't rely on layer_idx alone because the same forward pass
        # writes all layers in order; instead we just append per call and
        # the consumer slices by call order.
        def _hook(module, inputs, output):
            # output is (out, attn); attn shape: [n_head, B, T, T]
            if isinstance(output, tuple) and len(output) == 2:
                attn = output[1].detach()
                self._buffers.append(attn)
        return _hook

    def reset(self) -> None:
        """Drop all captured tensors. Call between batches."""
        self._buffers.clear()

    def pop_stack(self) -> List[torch.Tensor]:
        """Return captured attentions and reset the buffer."""
        out = self._buffers
        self._buffers = []
        return out

    def remove(self) -> None:
        """Remove all hooks. After this the capture object is inert."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._buffers.clear()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InterpretationResult:
    """Tidy long-form interpretation data for one or many forecasts.

    Each ``*_df`` is a long-format DataFrame so you can pivot, group, plot,
    or join with your own metadata freely.

    Attributes
    ----------
    static_weights_df : columns ``[sample_id, entity_id, window_idx, feature, weight]``
    historical_weights_df : columns ``[sample_id, entity_id, window_idx, time_step, feature, weight]``
        ``time_step`` is 0..historical_steps-1 (0 = oldest).
    future_weights_df : columns ``[sample_id, entity_id, window_idx, horizon, feature, weight]``
        ``horizon`` is 1..prediction_steps.
    attention_df : columns ``[sample_id, entity_id, window_idx, layer, head, query_pos, key_pos, weight]``
        ``query_pos`` and ``key_pos`` index the *full* sequence (length
        ``historical_steps + prediction_steps``), where positions
        ``< historical_steps`` are past and positions ``>= historical_steps``
        are future. May be empty if attention capture was disabled.
    predictions : ndarray ``[N, prediction_steps, num_quantiles]``
    metadata_df : DataFrame with at least ``[sample_id, entity_id, window_idx]``
    historical_steps, prediction_steps : ints from the model
    """

    static_weights_df: pd.DataFrame
    historical_weights_df: pd.DataFrame
    future_weights_df: pd.DataFrame
    attention_df: pd.DataFrame
    predictions: np.ndarray
    metadata_df: pd.DataFrame
    historical_steps: int
    prediction_steps: int

    # ------------------------------------------------------------------ aggregation

    def feature_importance(
        self,
        scope: str = "historical",
        agg: str = "mean",
    ) -> pd.DataFrame:
        """Aggregate VSN weights into a per-feature importance score.

        Parameters
        ----------
        scope : ``"static" | "historical" | "future"``
        agg : ``"mean" | "median" | "max"``

        Returns
        -------
        DataFrame with columns ``[feature, importance]`` sorted descending.
        For temporal scopes, weights are averaged over both samples and
        timesteps before aggregation.
        """
        if scope == "static":
            df = self.static_weights_df
        elif scope == "historical":
            df = self.historical_weights_df
        elif scope == "future":
            df = self.future_weights_df
        else:
            raise ValueError(f"scope must be 'static', 'historical', or 'future'; got {scope!r}")

        if df.empty:
            return pd.DataFrame(columns=["feature", "importance"])

        if agg == "mean":
            out = df.groupby("feature", as_index=False)["weight"].mean()
        elif agg == "median":
            out = df.groupby("feature", as_index=False)["weight"].median()
        elif agg == "max":
            out = df.groupby("feature", as_index=False)["weight"].max()
        else:
            raise ValueError(f"agg must be 'mean', 'median', or 'max'; got {agg!r}")
        out = out.rename(columns={"weight": "importance"})
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    def temporal_importance(
        self,
        scope: str = "historical",
        features: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Mean VSN weight for each (time_step or horizon, feature) pair,
        averaged over samples. Useful for line plots showing when each
        feature mattered.

        Parameters
        ----------
        scope : ``"historical" | "future"``
        features : optional whitelist of feature names

        Returns
        -------
        Wide-format DataFrame indexed by time_step (or horizon),
        columns = feature names, values = mean weight.
        """
        if scope == "historical":
            df = self.historical_weights_df
            time_col = "time_step"
        elif scope == "future":
            df = self.future_weights_df
            time_col = "horizon"
        else:
            raise ValueError("scope must be 'historical' or 'future' for temporal_importance")

        if df.empty:
            return pd.DataFrame()

        if features is not None:
            df = df[df["feature"].isin(features)]

        wide = (
            df.groupby([time_col, "feature"], as_index=False)["weight"]
            .mean()
            .pivot(index=time_col, columns="feature", values="weight")
        )
        return wide

    def attention_by_horizon(
        self,
        layer: Optional[int] = None,
        head: Optional[int | str] = "mean",
    ) -> pd.DataFrame:
        """Pivot attention into a (horizon × past-lag) matrix averaged over
        samples. This is the classic "which past timesteps did the model
        look at when forecasting horizon h" view from the TFT paper.

        Parameters
        ----------
        layer : optional layer index. ``None`` averages across all layers.
        head : int (specific head), ``"mean"`` (average across heads), or
            ``"max"`` (max across heads). Default ``"mean"``.

        Returns
        -------
        DataFrame indexed by horizon (1..prediction_steps), columns are
        past lag offsets. Lag is the negative offset from the forecast
        origin: lag=1 is the most recent historical step, lag=H is the
        oldest. Values are mean attention weight.
        """
        if self.attention_df.empty:
            return pd.DataFrame()

        df = self.attention_df
        # Restrict to future-query × past-key cells.
        H = self.historical_steps
        df = df[(df["query_pos"] >= H) & (df["key_pos"] < H)]

        if layer is not None:
            df = df[df["layer"] == layer]

        if head == "mean":
            df = df.groupby(
                ["query_pos", "key_pos"], as_index=False
            )["weight"].mean()
        elif head == "max":
            df = df.groupby(
                ["query_pos", "key_pos"], as_index=False
            )["weight"].max()
        elif isinstance(head, int):
            df = df[df["head"] == head]
            df = df.groupby(
                ["query_pos", "key_pos"], as_index=False
            )["weight"].mean()
        else:
            raise ValueError("head must be int, 'mean', or 'max'")

        # Convert query_pos -> horizon (1-indexed), key_pos -> past lag (1 = most recent).
        df = df.assign(
            horizon=df["query_pos"] - H + 1,
            lag=H - df["key_pos"],
        )
        wide = df.pivot_table(
            index="horizon", columns="lag", values="weight", aggfunc="mean"
        )
        # Sort lags so column 1 (most recent) is first.
        wide = wide.reindex(sorted(wide.columns), axis=1)
        return wide

    def persistent_temporal_pattern(
        self,
        layer: Optional[int] = None,
        head: Optional[int | str] = "mean",
    ) -> pd.Series:
        """Mean attention received by each past lag, averaged over horizons
        and samples. This is the "persistent temporal pattern" plot from
        Lim et al. — large peaks indicate recurring lags the model relies on
        (e.g. weekly seasonality at lag 7).

        Returns a Series indexed by lag (1 = most recent past step).
        """
        wide = self.attention_by_horizon(layer=layer, head=head)
        if wide.empty:
            return pd.Series(dtype=float, name="mean_attention")
        return wide.mean(axis=0).rename("mean_attention")

    # ------------------------------------------------------------------ I/O

    def to_csvs(self, directory: str) -> Dict[str, str]:
        """Write all DataFrames to a directory. Returns a dict of paths."""
        import os
        os.makedirs(directory, exist_ok=True)
        paths = {}
        for name, df in (
            ("static_weights", self.static_weights_df),
            ("historical_weights", self.historical_weights_df),
            ("future_weights", self.future_weights_df),
            ("attention", self.attention_df),
            ("metadata", self.metadata_df),
        ):
            path = os.path.join(directory, f"{name}.csv")
            df.to_csv(path, index=False)
            paths[name] = path
        np.save(os.path.join(directory, "predictions.npy"), self.predictions)
        paths["predictions"] = os.path.join(directory, "predictions.npy")
        return paths

    # ------------------------------------------------------------------ plotting (optional)

    def plot_feature_importance(
        self,
        scope: str = "historical",
        top_k: int = 15,
        ax=None,
    ):
        """Bar chart of mean VSN weight per feature."""
        plt = _import_matplotlib()
        imp = self.feature_importance(scope=scope).head(top_k)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(imp))))
        # Plot smallest-at-bottom by reversing.
        ax.barh(imp["feature"][::-1], imp["importance"][::-1])
        ax.set_xlabel("Mean variable selection weight")
        ax.set_title(f"{scope.capitalize()} feature importance (top {len(imp)})")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        return ax

    def plot_attention_heatmap(
        self,
        layer: Optional[int] = None,
        head: Optional[int | str] = "mean",
        ax=None,
        cmap: str = "viridis",
    ):
        """Heatmap of mean attention from each forecast horizon to each past lag."""
        plt = _import_matplotlib()
        wide = self.attention_by_horizon(layer=layer, head=head)
        if wide.empty:
            raise ValueError(
                "No attention data to plot. Did you initialize TFTInterpreter "
                "with capture_attention=True?"
            )
        if ax is None:
            _, ax = plt.subplots(figsize=(max(6, 0.25 * wide.shape[1]),
                                          max(3, 0.4 * wide.shape[0])))
        im = ax.imshow(wide.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(wide.shape[1]))
        ax.set_xticklabels(wide.columns)
        ax.set_yticks(range(wide.shape[0]))
        ax.set_yticklabels(wide.index)
        ax.set_xlabel("Past lag (1 = most recent)")
        ax.set_ylabel("Forecast horizon")
        ax.set_title("Mean attention: horizon × past lag")
        ax.figure.colorbar(im, ax=ax, label="attention weight")
        return ax

    def plot_temporal_importance(
        self,
        scope: str = "historical",
        features: Optional[Sequence[str]] = None,
        top_k: int = 5,
        ax=None,
    ):
        """Line chart of mean VSN weight per timestep for the top-k features."""
        plt = _import_matplotlib()
        if features is None:
            top = self.feature_importance(scope=scope).head(top_k)["feature"].tolist()
            features = top
        wide = self.temporal_importance(scope=scope, features=features)
        if wide.empty:
            raise ValueError("No data to plot for the chosen scope.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        for feat in wide.columns:
            ax.plot(wide.index, wide[feat], marker="o", markersize=3, label=feat)
        ax.set_xlabel("time_step" if scope == "historical" else "horizon")
        ax.set_ylabel("mean variable selection weight")
        ax.set_title(f"{scope.capitalize()} temporal importance")
        ax.legend(loc="best", fontsize=8)
        ax.grid(linestyle=":", alpha=0.4)
        return ax

    def plot_persistent_temporal_pattern(self, ax=None, **kwargs):
        """Line plot of persistent temporal attention pattern across all
        horizons and samples. Useful for spotting seasonality."""
        plt = _import_matplotlib()
        s = self.persistent_temporal_pattern(**kwargs)
        if s.empty:
            raise ValueError("No attention data to plot.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        ax.plot(s.index, s.values, marker="o", markersize=3)
        ax.set_xlabel("Past lag (1 = most recent)")
        ax.set_ylabel("Mean attention weight")
        ax.set_title("Persistent temporal pattern")
        ax.grid(linestyle=":", alpha=0.4)
        return ax


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ImportError as e:
        raise ImportError(
            "Plotting helpers require matplotlib. Install with `pip install matplotlib`."
        ) from e


# ---------------------------------------------------------------------------
# Main interpreter
# ---------------------------------------------------------------------------

class TFTInterpreter:
    """Run a trained TFT in eval mode and harvest its interpretability signals.

    This wraps an already-trained model and a ``TFTDataAdapter`` (the same
    one you use for inference). On construction, it registers forward hooks
    on every ``TFTMultiHeadAttention`` to capture attention weights, which
    the base model otherwise discards. Always call ``close()`` when done,
    or use it as a context manager.

    Parameters
    ----------
    model : nn.Module
        Loaded ``TemporalFusionTransformer`` (typically loaded via
        ``TFTInference.load_model_weights`` first).
    adapter : TFTDataAdapter
    device : str | torch.device, optional
        Defaults to the model's current device.
    capture_attention : bool, default True
        If False, skip hook registration (saves a bit of memory if you only
        want VSN weights).
    max_attention_samples : int | None, default None
        If set, only the first N samples per batch contribute attention
        rows to the long-form DataFrame. Use this when you have many
        samples and don't want a multi-GB attention table — VSN weights
        and predictions are still kept for every sample.

    Example
    -------
    >>> with TFTInterpreter(model, test_adapter) as interp:
    ...     result = interp.interpret(test_loader, max_batches=2)
    >>> result.feature_importance("historical")
    """

    def __init__(
        self,
        model: nn.Module,
        adapter,
        device: Optional[str | torch.device] = None,
        capture_attention: bool = True,
        max_attention_samples: Optional[int] = None,
    ):
        self.model = model
        self.adapter = adapter
        self.device = torch.device(
            device if device is not None else next(model.parameters()).device
        )
        self.capture_attention = capture_attention
        self.max_attention_samples = max_attention_samples

        self._capture: Optional[_AttentionCapture] = None
        if capture_attention:
            self._capture = _AttentionCapture(model)

        # Resolve feature names once.
        self._static_names = static_feature_names(adapter)
        self._hist_names = historical_feature_names(adapter)
        self._fut_names = future_feature_names(adapter)

        self.historical_steps = adapter.historical_steps
        self.prediction_steps = adapter.prediction_steps

    # ----- context-manager / cleanup -----
    def __enter__(self) -> "TFTInterpreter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Remove forward hooks. Safe to call more than once."""
        if self._capture is not None:
            self._capture.remove()
            self._capture = None

    # ----- main entry point -----

    @torch.no_grad()
    def interpret(
        self,
        dataloader,
        max_batches: Optional[int] = None,
    ) -> InterpretationResult:
        """Run inference over ``dataloader`` and collect interpretation data.

        Parameters
        ----------
        dataloader : iterable of batches (your standard TFT loader)
        max_batches : optional cap on number of batches (handy for spot-checking)

        Returns
        -------
        InterpretationResult
        """
        self.model.eval()

        static_rows: List[pd.DataFrame] = []
        hist_rows: List[pd.DataFrame] = []
        fut_rows: List[pd.DataFrame] = []
        attn_rows: List[pd.DataFrame] = []
        meta_rows: List[Dict[str, Any]] = []
        all_preds: List[np.ndarray] = []

        sample_offset = 0  # global sample id across batches

        for b_idx, batch in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break

            batch_gpu = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            model_inputs = self.adapter.adapt_for_tft(batch_gpu)

            if self._capture is not None:
                self._capture.reset()

            outputs = self.model(
                static_categorical=model_inputs.get("static_categorical"),
                static_continuous=model_inputs.get("static_continuous"),
                historical_categorical=model_inputs.get("historical_categorical"),
                historical_continuous=model_inputs.get("historical_continuous"),
                future_categorical=model_inputs.get("future_categorical"),
                future_continuous=model_inputs.get("future_continuous"),
                padding_mask=model_inputs.get("padding_mask"),
            )

            preds = outputs["predictions"].detach().cpu().numpy()
            all_preds.append(preds)
            B = preds.shape[0]

            entity_ids = batch.get("entity_id", [None] * B)
            window_idx = (
                batch["window_idx"].cpu().tolist() if "window_idx" in batch else [None] * B
            )
            sample_ids = list(range(sample_offset, sample_offset + B))
            sample_offset += B

            # Per-sample metadata rows.
            for sid, eid, wid in zip(sample_ids, entity_ids, window_idx):
                meta_rows.append(
                    {"sample_id": sid, "entity_id": eid, "window_idx": wid}
                )

            # ---- static weights: [B, n_static] ----
            sw = outputs.get("static_weights")
            if sw is not None and self._static_names:
                sw_np = sw.detach().cpu().numpy()
                static_rows.append(_static_to_long(
                    sw_np, sample_ids, entity_ids, window_idx, self._static_names
                ))

            # ---- historical weights: [B, T_hist, n_hist] ----
            hw = outputs.get("historical_weights")
            if hw is not None and self._hist_names:
                hw_np = hw.detach().cpu().numpy()
                hist_rows.append(_temporal_to_long(
                    hw_np, sample_ids, entity_ids, window_idx,
                    self._hist_names, time_col="time_step", time_offset=0,
                ))

            # ---- future weights: [B, T_pred, n_future] ----
            fw = outputs.get("future_weights")
            if fw is not None and self._fut_names:
                fw_np = fw.detach().cpu().numpy()
                fut_rows.append(_temporal_to_long(
                    fw_np, sample_ids, entity_ids, window_idx,
                    self._fut_names, time_col="horizon", time_offset=1,
                ))

            # ---- attention: list of [n_head, B, T, T] per layer ----
            if self._capture is not None:
                attn_stack = self._capture.pop_stack()
                if attn_stack:
                    n_keep = (
                        B if self.max_attention_samples is None
                        else min(B, self.max_attention_samples)
                    )
                    for layer_idx, attn in enumerate(attn_stack):
                        a_np = attn[:, :n_keep].cpu().numpy()  # [n_head, n_keep, T, T]
                        attn_rows.append(_attention_to_long(
                            a_np, sample_ids[:n_keep],
                            entity_ids[:n_keep], window_idx[:n_keep],
                            layer_idx,
                        ))

        return InterpretationResult(
            static_weights_df=_concat_or_empty(
                static_rows, ["sample_id", "entity_id", "window_idx", "feature", "weight"]
            ),
            historical_weights_df=_concat_or_empty(
                hist_rows,
                ["sample_id", "entity_id", "window_idx", "time_step", "feature", "weight"],
            ),
            future_weights_df=_concat_or_empty(
                fut_rows,
                ["sample_id", "entity_id", "window_idx", "horizon", "feature", "weight"],
            ),
            attention_df=_concat_or_empty(
                attn_rows,
                ["sample_id", "entity_id", "window_idx", "layer", "head",
                 "query_pos", "key_pos", "weight"],
            ),
            predictions=(
                np.concatenate(all_preds, axis=0)
                if all_preds else np.empty((0, self.prediction_steps, 0))
            ),
            metadata_df=pd.DataFrame(meta_rows),
            historical_steps=self.historical_steps,
            prediction_steps=self.prediction_steps,
        )


# ---------------------------------------------------------------------------
# Internals: tensor → long-form helpers
# ---------------------------------------------------------------------------

def _static_to_long(
    arr: np.ndarray,
    sample_ids: Sequence[int],
    entity_ids: Sequence[Any],
    window_idx: Sequence[Any],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    # arr: [B, n_features]
    B, n = arr.shape
    if n != len(feature_names):
        # Defensive: name list should match the dimension exactly.
        feature_names = [f"static_{i}" for i in range(n)]
    return pd.DataFrame({
        "sample_id": np.repeat(sample_ids, n),
        "entity_id": np.repeat(entity_ids, n),
        "window_idx": np.repeat(window_idx, n),
        "feature": np.tile(feature_names, B),
        "weight": arr.reshape(-1),
    })


def _temporal_to_long(
    arr: np.ndarray,
    sample_ids: Sequence[int],
    entity_ids: Sequence[Any],
    window_idx: Sequence[Any],
    feature_names: Sequence[str],
    time_col: str,
    time_offset: int,
) -> pd.DataFrame:
    # arr: [B, T, n_features]
    B, T, n = arr.shape
    if n != len(feature_names):
        feature_names = [f"feat_{i}" for i in range(n)]
    # Build index arrays the same shape as arr, then flatten consistently.
    sample_arr = np.repeat(np.array(sample_ids), T * n)
    entity_arr = np.repeat(np.array(entity_ids, dtype=object), T * n)
    window_arr = np.repeat(np.array(window_idx, dtype=object), T * n)
    time_arr = np.tile(np.repeat(np.arange(time_offset, time_offset + T), n), B)
    feat_arr = np.tile(np.tile(np.array(feature_names, dtype=object), T), B)
    return pd.DataFrame({
        "sample_id": sample_arr,
        "entity_id": entity_arr,
        "window_idx": window_arr,
        time_col: time_arr,
        "feature": feat_arr,
        "weight": arr.reshape(-1),
    })


def _attention_to_long(
    arr: np.ndarray,
    sample_ids: Sequence[int],
    entity_ids: Sequence[Any],
    window_idx: Sequence[Any],
    layer: int,
) -> pd.DataFrame:
    # arr: [n_head, B, T, T]
    n_head, B, Tq, Tk = arr.shape
    total = n_head * B * Tq * Tk
    # Build index axes via meshgrid for clarity; fast enough for the
    # batch sizes involved in interpretation runs.
    head_arr, sample_axis, q_arr, k_arr = np.meshgrid(
        np.arange(n_head),
        np.arange(B),
        np.arange(Tq),
        np.arange(Tk),
        indexing="ij",
    )
    return pd.DataFrame({
        "sample_id": np.array(sample_ids, dtype=object)[sample_axis.reshape(-1)],
        "entity_id": np.array(entity_ids, dtype=object)[sample_axis.reshape(-1)],
        "window_idx": np.array(window_idx, dtype=object)[sample_axis.reshape(-1)],
        "layer": np.full(total, layer, dtype=int),
        "head": head_arr.reshape(-1),
        "query_pos": q_arr.reshape(-1),
        "key_pos": k_arr.reshape(-1),
        "weight": arr.reshape(-1),
    })


def _concat_or_empty(frames: List[pd.DataFrame], cols: Sequence[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(cols))
    return pd.concat(frames, ignore_index=True)
