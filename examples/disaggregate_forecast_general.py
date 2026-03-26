"""
Forecast Disaggregation: Group-level → Item-level

Splits a group-level forecast back to item level using trailing
historical demand proportions.

All column names are configurable via DisaggConfig — nothing is hardcoded.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DisaggConfig:
    """
    Central configuration for all column names used in disaggregation.

    Parameters
    ----------
    group_key_cols : list[str]
        Columns that define the aggregated forecast grain.
        e.g. ["Plant", "Material_Group", "Channel"]
    item_col : str
        The column to disaggregate INTO (the finer grain).
        e.g. "Material"
    time_col : str
        The datetime / period column.
        e.g. "YearMonth"
    historical_qty_col : str
        Target column in the historical data.
        e.g. "Customer Demand Qty"
    forecast_qty_col : str
        Forecasted quantity column in the forecast data.
        e.g. "Forecast Qty"
    output_qty_col : str
        Name for the disaggregated quantity in the output.
    proportion_col : str
        Name for the proportion column in the output.
    output_key_col : str | None
        If provided, a concatenated key column is added to the output.
    output_key_parts : list[str] | None
        Columns to concatenate for output_key_col.
    output_key_sep : str
        Separator used when building output_key_col.
    """
    group_key_cols: List[str]
    item_col: str
    time_col: str
    historical_qty_col: str
    forecast_qty_col: str
    output_qty_col: str = "Disaggregated Qty"
    proportion_col: str = "proportion"
    output_key_col: Optional[str] = None
    output_key_parts: Optional[List[str]] = None
    output_key_sep: str = "_"


def compute_proportions(
    historical_df: pd.DataFrame,
    config: DisaggConfig,
    cutoff_date: str,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Compute each item's demand share within its group over the
    lookback window ending just before cutoff_date.

    Returns
    -------
    DataFrame with columns: [*group_key_cols, item_col, proportion_col]
    """
    c = config
    df = historical_df.copy()
    df[c.time_col] = pd.to_datetime(df[c.time_col])
    cutoff = pd.Timestamp(cutoff_date)

    window_start = cutoff - pd.DateOffset(months=lookback_months)
    mask = (df[c.time_col] >= window_start) & (df[c.time_col] < cutoff)
    window_df = df.loc[mask]

    if window_df.empty:
        raise ValueError(
            f"No historical data in [{window_start.date()}, {cutoff.date()}). "
            f"Check cutoff_date and lookback_months."
        )

    # Item-level totals within each group
    item_group_cols = c.group_key_cols + [c.item_col]
    item_totals = (
        window_df
        .groupby(item_group_cols, as_index=False)[c.historical_qty_col]
        .sum()
        .rename(columns={c.historical_qty_col: "_item_demand"})
    )

    # Group-level totals
    group_totals = (
        item_totals
        .groupby(c.group_key_cols, as_index=False)["_item_demand"]
        .sum()
        .rename(columns={"_item_demand": "_group_demand"})
    )

    # Merge and compute proportion
    proportions = item_totals.merge(group_totals, on=c.group_key_cols, how="left")
    proportions[c.proportion_col] = np.where(
        proportions["_group_demand"] > 0,
        proportions["_item_demand"] / proportions["_group_demand"],
        0,
    )

    # Sanity check
    group_sums = proportions.groupby(c.group_key_cols)[c.proportion_col].sum()
    bad = group_sums[~np.isclose(group_sums, 1.0, atol=1e-6)]
    if not bad.empty:
        print(f"WARNING: {len(bad)} group(s) have proportions not summing to 1.")

    return proportions[item_group_cols + [c.proportion_col]]


def disaggregate_forecast(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    config: DisaggConfig,
    cutoff_date: str,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Disaggregate a group-level forecast to item level.

    Parameters
    ----------
    forecast_df : DataFrame
        Must include columns [*config.group_key_cols, config.time_col,
        config.forecast_qty_col].
    historical_df : DataFrame
        Must include columns [*config.group_key_cols, config.item_col,
        config.time_col, config.historical_qty_col].
    config : DisaggConfig
        Column name configuration.
    cutoff_date : str
        First forecast period (e.g. "2025-01-01").
    lookback_months : int
        Trailing months for proportion calculation.

    Returns
    -------
    DataFrame at item level with proportions and disaggregated quantities.
    """
    c = config

    # ── Step 1: Proportions ──
    proportions = compute_proportions(
        historical_df, config, cutoff_date, lookback_months
    )
    print(f"Computed proportions for {len(proportions)} item-within-group combinations.")

    # ── Step 2: Join to forecast ──
    forecast = forecast_df.copy()
    forecast[c.time_col] = pd.to_datetime(forecast[c.time_col])

    result = forecast.merge(proportions, on=c.group_key_cols, how="left")

    # ── Step 3: Flag unmatched groups ──
    unmatched = result[c.item_col].isna()
    if unmatched.any():
        n_unmatched = (
            result.loc[unmatched, c.group_key_cols]
            .drop_duplicates().shape[0]
        )
        print(
            f"WARNING: {n_unmatched} forecast group(s) had NO matching items "
            f"in the lookback window. These rows will have 0 disaggregated qty."
        )

    # ── Step 4: Disaggregated quantity ──
    result[c.output_qty_col] = (
        result[c.forecast_qty_col] * result[c.proportion_col].fillna(0)
    )

    # ── Step 5: Optional concatenated key ──
    if c.output_key_col and c.output_key_parts:
        result[c.output_key_col] = (
            result[c.output_key_parts]
            .astype(str)
            .agg(c.output_key_sep.join, axis=1)
        )

    # ── Step 6: Order columns and sort ──
    leading = []
    if c.output_key_col and c.output_key_col in result.columns:
        leading = [c.output_key_col]

    item_group_cols = c.group_key_cols + [c.item_col]
    core_cols = (
        leading
        + item_group_cols
        + [c.time_col, c.forecast_qty_col, c.proportion_col, c.output_qty_col]
    )
    extra = [col for col in result.columns if col not in core_cols]
    result = result[core_cols + extra]

    sort_cols = item_group_cols + [c.time_col]
    result = result.sort_values(sort_cols).reset_index(drop=True)

    # ── Summary ──
    agg_total = forecast[c.forecast_qty_col].sum()
    disagg_total = result[c.output_qty_col].sum()
    print(f"\nAggregated forecast total : {agg_total:,.2f}")
    print(f"Disaggregated forecast total: {disagg_total:,.2f}")
    print(f"Difference (leakage)       : {agg_total - disagg_total:,.2f}")

    return result


# ─────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":

    np.random.seed(42)

    # ── Synthetic historical data ──
    records = []
    plants = ["P1"]
    channels = ["C1"]
    material_groups = {"MG1": ["M1", "M2", "M3"], "MG2": ["M4", "M5"]}
    months = pd.date_range("2023-01-01", "2024-12-01", freq="MS")

    for plant in plants:
        for channel in channels:
            for mg, materials in material_groups.items():
                for mat in materials:
                    for m in months:
                        qty = max(0, np.random.poisson(lam=50 if mat in ["M1", "M4"] else 10))
                        records.append({
                            "Plant": plant,
                            "Material": mat,
                            "Channel": channel,
                            "Material_Group": mg,
                            "YearMonth": m,
                            "Customer Demand Qty": qty,
                            "key": f"{plant}_{mat}_{channel}",
                        })
    hist_df = pd.DataFrame(records)

    # ── Synthetic forecast at group level ──
    cutoff = "2025-01-01"
    forecast_months = pd.date_range("2025-01-01", "2025-06-01", freq="MS")
    fc_records = []
    for plant in plants:
        for channel in channels:
            for mg in material_groups:
                for m in forecast_months:
                    fc_records.append({
                        "Plant": plant,
                        "Material_Group": mg,
                        "Channel": channel,
                        "YearMonth": m,
                        "Forecast Qty": np.random.randint(100, 300),
                    })
    fc_df = pd.DataFrame(fc_records)

    # ── Configure — this is the ONLY place column names appear ──
    cfg = DisaggConfig(
        group_key_cols=["Plant", "Material_Group", "Channel"],
        item_col="Material",
        time_col="YearMonth",
        historical_qty_col="Customer Demand Qty",
        forecast_qty_col="Forecast Qty",
        output_qty_col="Disaggregated Qty",
        proportion_col="proportion",
        output_key_col="key",
        output_key_parts=["Plant", "Material", "Channel"],
        output_key_sep="_",
    )

    # ── Run ──
    print("=" * 60)
    print("  DEMO: Generalized Forecast Disaggregation")
    print("=" * 60)

    result = disaggregate_forecast(
        forecast_df=fc_df,
        historical_df=hist_df,
        config=cfg,
        cutoff_date=cutoff,
        lookback_months=6,
    )

    print("\n── Proportions ──")
    props = compute_proportions(hist_df, cfg, cutoff, lookback_months=6)
    print(props.to_string(index=False))

    print("\n── Disaggregated forecast (first 15 rows) ──")
    print(result.head(15).to_string(index=False))

    print("\n── Verify group totals match ──")
    check = result.groupby(
        cfg.group_key_cols + [cfg.time_col]
    )[cfg.output_qty_col].sum().reset_index()
    merged = check.merge(fc_df, on=cfg.group_key_cols + [cfg.time_col])
    merged["match"] = np.isclose(merged[cfg.output_qty_col], merged[cfg.forecast_qty_col])
    print(f"All group totals match: {merged['match'].all()}")
