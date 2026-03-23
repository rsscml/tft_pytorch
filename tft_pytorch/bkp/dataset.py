"""
PyTorch Dataset and DataLoader utilities for Temporal Fusion Transformer models.

Key classes
-----------
OptimizedTFTDataset
    Memory-efficient, window-based dataset with per-window or entity-level
    scaling, automatic categorical encoding, and padding support.
TFTDataAdapter
    Bridges the dataset output format to the TFT model's input format.
TCNDataAdapter
    Bridges the dataset output format to TCN-style model inputs.

Utility helpers
---------------
create_tft_dataloader(dataset, ...) -> (DataLoader, TFTDataAdapter)
create_tcn_dataloader(dataset, ...) -> (DataLoader, TCNDataAdapter)
create_uniform_embedding_dims(dataset, hidden_layer_size) -> dict
inverse_transform_predictions(predictions, batch, dataset, target_idx) -> Tensor
"""

import json
import multiprocessing as mp
import pickle
import platform
import threading
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# OS-aware parallel backend selection
# ---------------------------------------------------------------------------

def get_optimal_backend() -> str:
    """Return the best Joblib backend for the current operating system."""
    system = platform.system().lower()
    if system == 'windows':
        return 'loky'
    elif system in ('linux', 'darwin'):
        return 'threading'
    return 'loky'


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class OptimizedTFTDataset(Dataset):
    """
    Memory-efficient TFT Dataset with window-specific or entity-level scaling.

    Key design decisions
    --------------------
    * Scaler *parameters* (not objects) are stored in compact numpy arrays,
      saving up to 80 % memory vs. storing sklearn scaler objects.
    * Window creation respects series length: full windows, padded windows, and
      ultra-short (inference-only) series are all handled.
    * Categorical encoders are fitted once on training data and saved/loaded
      for validation / test splits.

    Parameters
    ----------
    data_source : DataFrame, file path, or list of file paths
    features_config : dict
        Required keys: ``entity_col``, ``time_index_col``, ``target_col``
        Optional keys: ``static_numeric_col_list``, ``static_categorical_col_list``,
        ``temporal_known_numeric_col_list``, ``temporal_unknown_numeric_col_list``,
        ``temporal_known_categorical_col_list``, ``temporal_unknown_categorical_col_list``,
        ``wt_col``
    historical_steps : int
    prediction_steps : int
    stride : int
        Step between consecutive windows.
    enable_padding : bool
        Allow creating windows for short series by left-padding.
    padding_strategy : str
        ``'zero'``, ``'mean'``, ``'forward_fill'``, or ``'intelligent'``
    categorical_padding_value : int
        Value used to left-pad categorical sequences (always ``-1``).
    min_historical_steps : int, optional
        Minimum non-padded historical steps required.
    allow_inference_only_entities : bool
        Accept entities with fewer than ``prediction_steps`` rows in val/test.
    scaler_path : str, optional
        Path to save (train) or load (val/test) entity-level scalers.
    scaling_strategy : str
        ``'per_window'`` or ``'entity_level'``.
    scaling_method : str
        ``'standard'``, ``'mean'``, or ``'none'``.
    mean_scaler_epsilon : float
        Epsilon for mean scaling to avoid division by zero.
    cold_start_scaler_cols : list of str, optional
        Categorical columns used to assign scalers to brand-new entities.
    recency_alpha : float
        Exponential recency weight (0 = disabled).
    n_jobs : int
        Parallel workers (``-1`` = all CPUs).
    max_series : int, optional
        Cap the number of series loaded.
    mode : str
        ``'train'``, ``'val'``, or ``'test'``.
    encoders_path : str, optional
        Directory for persisting/loading LabelEncoders.
    fit_encoders : bool, optional
        Override automatic encoder-fit behaviour.
    preprocessing_fn : callable, optional
        Applied to every entity DataFrame after loading.
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame, List[Union[str, Path]]],
        features_config: Dict[str, Any],
        historical_steps: int = 30,
        prediction_steps: int = 1,
        stride: int = 1,
        enable_padding: bool = True,
        padding_strategy: str = 'zero',
        categorical_padding_value: int = -1,
        min_historical_steps: Optional[int] = None,
        allow_inference_only_entities: bool = False,
        scaler_path: Optional[str] = None,
        scaling_strategy: str = 'per_window',
        scaling_method: str = 'standard',
        mean_scaler_epsilon: float = 1.0,
        cold_start_scaler_cols: Optional[List[str]] = None,
        recency_alpha: float = 0.0,
        n_jobs: int = -1,
        max_series: Optional[int] = None,
        mode: str = 'train',
        encoders_path=None,
        fit_encoders=None,
        preprocessing_fn: Optional[Callable] = None,
    ):
        self.mode = mode
        self.encoders_path = Path(encoders_path) if encoders_path else Path('./.encoders')

        # Encoder fit behaviour
        if fit_encoders is not None:
            self.should_fit_encoders = fit_encoders
        else:
            self.should_fit_encoders = (mode == 'train')

        self._parse_features_config(features_config)

        # Window parameters
        self.historical_steps = historical_steps
        self.prediction_steps = prediction_steps
        self.stride = stride
        self.total_steps = historical_steps + prediction_steps

        # Padding
        self.enable_padding = enable_padding
        self.padding_strategy = padding_strategy
        self.categorical_padding_value = categorical_padding_value
        self.entity_padding_values: Dict = {}

        if min_historical_steps is None:
            self.min_historical_steps = max(1, historical_steps // 3)
        else:
            self.min_historical_steps = min_historical_steps

        self.allow_inference_only_entities = allow_inference_only_entities

        if allow_inference_only_entities and mode in ('test', 'val'):
            self.min_series_length = 0
        elif enable_padding:
            self.min_series_length = self.min_historical_steps + prediction_steps
        else:
            self.min_series_length = historical_steps + prediction_steps

        # Scaling
        self.scaler_path = scaler_path
        self.imported_scalers = None
        self.scaling_strategy = scaling_strategy
        self.scaling_method = scaling_method
        self.cold_start_scaler_cols = cold_start_scaler_cols or self.static_categorical_cols
        self.mean_scaler_epsilon = mean_scaler_epsilon

        self.max_series = max_series
        self.preprocessing_fn = preprocessing_fn

        if self.mode in ('val', 'test') and self.scaler_path and self.scaling_strategy == 'entity_level' and self.scaling_method != 'none':
            self.imported_scalers = self.load_entity_scalers(self.scaler_path)
            print(f"Loaded scalers for {self.imported_scalers['n_entities']} entities from {self.scaler_path}")
        elif self.mode == 'train' and self.scaler_path and self.scaling_strategy == 'entity_level':
            print(f"Will save scalers to {self.scaler_path} after fitting")

        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.recency_alpha = recency_alpha

        # Storage
        self.series_data: Dict = {}
        self.windows: List[Dict] = []
        self.scaler_params = None

        # Load → preprocess → encode → window/scale → cache
        self._load_data(data_source)

        if self.preprocessing_fn:
            self._apply_preprocessing()

        if self.enable_padding:
            self._compute_entity_padding_values()

        if self.should_fit_encoders:
            self._fit_and_save_encoders()
        else:
            self._load_encoders()

        self._create_windows_and_fit_scalers()

        if self.mode == 'train' and self.scaler_path and self.scaling_strategy == 'entity_level':
            self.save_entity_scalers(self.scaler_path)

        self._preprocess_and_cache_data()
        self._report_dataset_stats()

    # ------------------------------------------------------------------
    # Feature config
    # ------------------------------------------------------------------

    def _parse_features_config(self, config: Dict[str, Any]):
        self.entity_col = config.get('entity_col')
        self.time_col = config.get('time_index_col')

        target = config.get('target_col')
        self.target_cols = [target] if isinstance(target, str) else (target or [])

        self.static_numeric_cols = config.get('static_numeric_col_list', [])
        self.static_categorical_cols = config.get('static_categorical_col_list', [])
        self.temporal_known_numeric_cols = config.get('temporal_known_numeric_col_list', [])
        self.temporal_known_categorical_cols = config.get('temporal_known_categorical_col_list', [])
        self.temporal_unknown_numeric_cols = config.get('temporal_unknown_numeric_col_list', [])
        self.temporal_unknown_categorical_cols = config.get('temporal_unknown_categorical_col_list', [])
        self.weight_col = config.get('wt_col')

        self.categorical_cols = (
            self.static_categorical_cols
            + self.temporal_known_categorical_cols
            + self.temporal_unknown_categorical_cols
        )
        self.numeric_cols = (
            self.target_cols
            + self.static_numeric_cols
            + self.temporal_known_numeric_cols
            + self.temporal_unknown_numeric_cols
        )
        self.all_unknown_features = list(set(self.target_cols + self.temporal_unknown_numeric_cols))
        self.all_known_features = list(set(self.temporal_known_numeric_cols))
        self.feature_to_idx = {f: i for i, f in enumerate(self.numeric_cols)}

        print("\nFeature Configuration:")
        print(f"  Targets: {len(self.target_cols)}")
        print(f"  Static: {len(self.static_numeric_cols)} numeric, {len(self.static_categorical_cols)} categorical")
        print(f"  Known temporal: {len(self.temporal_known_numeric_cols)} numeric, {len(self.temporal_known_categorical_cols)} categorical")
        print(f"  Unknown temporal: {len(self.temporal_unknown_numeric_cols)} numeric, {len(self.temporal_unknown_categorical_cols)} categorical")
        print(f"  Total numeric features: {len(self.numeric_cols)}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, data_source):
        print("Loading data...")
        if isinstance(data_source, pd.DataFrame):
            self._load_from_dataframe(data_source)
        elif isinstance(data_source, (str, Path)):
            self._load_from_dataframe(pd.read_csv(data_source))
        elif isinstance(data_source, list):
            self._load_from_file_list(data_source)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

    def _load_from_dataframe(self, df: pd.DataFrame):
        if self.entity_col and self.entity_col in df.columns:
            entities = df[self.entity_col].unique()
            if self.max_series:
                entities = entities[:self.max_series]
            n_entities = len(entities)
            print(f"Processing {n_entities} entities...")

            def _process(entity_id, df_full, entity_col, time_col, min_len):
                edf = df_full[df_full[entity_col] == entity_id].copy()
                if time_col and time_col in edf.columns:
                    edf = edf.sort_values(time_col).set_index(time_col)
                if len(edf) >= min_len:
                    return str(entity_id), edf
                return None, None

            if self.n_jobs != 1 and n_entities > 50:
                backend = get_optimal_backend()
                print(f"  Using parallel processing with '{backend}' backend...")
                batch_size = max(10, n_entities // (self.n_jobs * 4))

                def _batch(batch_entities):
                    return [r for e in batch_entities
                            for r in [_process(e, df, self.entity_col, self.time_col, self.min_series_length)]
                            if r[0] is not None]

                batches = [entities[i:i+batch_size] for i in range(0, n_entities, batch_size)]
                results = Parallel(n_jobs=self.n_jobs, backend=backend)(delayed(_batch)(b) for b in batches)
                self.series_data = {eid: edf for batch in results for eid, edf in batch}
            else:
                print("  Using sequential processing...")
                self.series_data = {}
                for entity_id in entities:
                    eid, edf = _process(entity_id, df, self.entity_col, self.time_col, self.min_series_length)
                    if eid is not None:
                        self.series_data[eid] = edf
        else:
            if self.time_col and self.time_col in df.columns:
                df = df.sort_values(self.time_col).set_index(self.time_col)
            if len(df) >= self.min_series_length:
                self.series_data['series_0'] = df

        n_loaded = len(self.series_data)
        print(f"Loaded {n_loaded} valid series")

    def _load_from_file_list(self, file_paths: List[Union[str, Path]]):
        if self.max_series:
            file_paths = file_paths[:self.max_series]
        for path in file_paths:
            df = pd.read_csv(path)
            if self.time_col and self.time_col in df.columns:
                df = df.sort_values(self.time_col).set_index(self.time_col)
            if len(df) >= self.min_series_length:
                eid = Path(path).stem if isinstance(path, (str, Path)) else f'series_{path}'
                self.series_data[eid] = df
        print(f"Loaded {len(self.series_data)} series")

    def _apply_preprocessing(self):
        print("Applying preprocessing...")
        processed = {}
        for eid, df in self.series_data.items():
            try:
                pdf = self.preprocessing_fn(df.copy())
                if len(pdf) >= self.min_series_length:
                    processed[eid] = pdf
            except Exception as e:
                print(f"Warning: Failed to preprocess series {eid}: {e}")
        self.series_data = processed
        print(f"Preprocessing complete. {len(self.series_data)} series remaining.")

    # ------------------------------------------------------------------
    # Padding value computation
    # ------------------------------------------------------------------

    def _compute_entity_padding_values(self):
        print("Computing padding values for each entity...")
        for eid, df in self.series_data.items():
            pv = {}
            for col in self.numeric_cols:
                if col not in df.columns:
                    continue
                values = df[col].dropna().values
                if len(values) == 0:
                    pv[col] = 0.0
                elif self.padding_strategy == 'mean':
                    pv[col] = float(np.mean(values))
                elif self.padding_strategy == 'forward_fill':
                    pv[col] = float(values[0])
                elif self.padding_strategy == 'intelligent':
                    if col in self.target_cols:
                        pv[col] = float(np.mean(values))
                    elif any(k in col.lower() for k in ('price', 'cost', 'temperature')):
                        pv[col] = float(np.mean(values))
                    elif np.all(np.isin(values[~np.isnan(values)], [0, 1])):
                        pv[col] = 0.5
                    else:
                        pv[col] = float(np.mean(values))
                else:  # 'zero'
                    pv[col] = 0.0
            self.entity_padding_values[eid] = pv
        print(f"Computed padding values for {len(self.entity_padding_values)} entities")

    # ------------------------------------------------------------------
    # Window creation & scaler fitting
    # ------------------------------------------------------------------

    def _create_windows_and_fit_scalers(self):
        print("Creating windows and fitting scalers...")
        self.entity_scaling_windows: Dict = defaultdict(list)

        for eid, df in self.series_data.items():
            series_len = len(df)
            last_selected_end = -float('inf')

            # Ultra-short series (fewer rows than prediction_steps)
            if series_len < self.prediction_steps and self.allow_inference_only_entities:
                widx = len(self.windows)
                window = {
                    'entity_id': eid,
                    'start_idx': 0,
                    'end_idx': series_len,
                    'historical_end': 0,
                    'padding_steps': self.historical_steps,
                    'future_data_len': series_len,
                    'future_padding_steps': self.prediction_steps - series_len,
                    'is_ultra_short': True,
                }
                if self.scaling_strategy == 'entity_level':
                    self.entity_scaling_windows[eid].append(widx)
                self.windows.append(window)
                continue

            # Full-length windows
            if series_len >= self.total_steps:
                n_windows = (series_len - self.total_steps) // self.stride + 1
                last_regular_start = -1

                for i in range(n_windows):
                    start_idx = i * self.stride
                    end_idx = start_idx + self.total_steps
                    widx = len(self.windows)
                    last_regular_start = start_idx
                    window = {
                        'entity_id': eid,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'historical_end': start_idx + self.historical_steps,
                        'padding_steps': 0,
                    }
                    if self.scaling_strategy == 'entity_level' and start_idx >= last_selected_end:
                        self.entity_scaling_windows[eid].append(widx)
                        last_selected_end = end_idx
                    self.windows.append(window)

                last_possible_start = series_len - self.total_steps
                if last_possible_start > last_regular_start:
                    widx = len(self.windows)
                    window = {
                        'entity_id': eid,
                        'start_idx': last_possible_start,
                        'end_idx': series_len,
                        'historical_end': last_possible_start + self.historical_steps,
                        'padding_steps': 0,
                    }
                    if self.scaling_strategy == 'entity_level' and last_possible_start >= last_selected_end:
                        self.entity_scaling_windows[eid].append(widx)
                    self.windows.append(window)

            # Padded windows for short series
            elif series_len >= self.min_series_length and self.enable_padding:
                actual_historical = series_len - self.prediction_steps
                widx = len(self.windows)
                window = {
                    'entity_id': eid,
                    'start_idx': 0,
                    'end_idx': series_len,
                    'historical_end': actual_historical,
                    'padding_steps': self.historical_steps - actual_historical,
                }
                if self.scaling_strategy == 'entity_level':
                    self.entity_scaling_windows[eid].append(widx)
                self.windows.append(window)

        n_windows = len(self.windows)
        print(f"Created {n_windows} windows")

        if self.scaling_method == 'none':
            self.scaler_params = None
            print("No scaling will be applied")
            return

        n_features = len(self.numeric_cols)
        if self.scaling_method == 'mean':
            self.scaler_params = np.zeros((n_windows, n_features), dtype=np.float32)
        else:
            self.scaler_params = np.zeros((n_windows, n_features, 2), dtype=np.float32)
        print(f"Allocated scaler array: {self.scaler_params.shape}")

        if self.scaling_strategy == 'entity_level':
            self._fit_entity_level_scalers_fast()
        else:
            self._fit_per_window_scalers(n_windows)

        print("Scaler fitting complete")

    def _fit_per_window_scalers(self, n_windows: int):
        if self.n_jobs != 1 and n_windows > 100:
            backend = get_optimal_backend()
            batch_size = max(100, n_windows // (self.n_jobs * 10))

            def _batch(start, end):
                return [(i, self._fit_scalers_for_window_optimized(self.windows[i], self.series_data[self.windows[i]['entity_id']]))
                        for i in range(start, min(end, n_windows))]

            results = Parallel(n_jobs=self.n_jobs, backend=backend)(
                delayed(_batch)(i, i + batch_size) for i in range(0, n_windows, batch_size)
            )
            for batch in results:
                for idx, params in batch:
                    self.scaler_params[idx] = params
        else:
            for widx, window in enumerate(self.windows):
                df = self.series_data[window['entity_id']]
                self.scaler_params[widx] = self._fit_scalers_for_window_optimized(window, df)

    def _fit_scalers_for_window_optimized(self, window: Dict, df: pd.DataFrame) -> np.ndarray:
        historical_end = window['historical_end']
        n_features = len(self.numeric_cols)

        if self.scaling_method == 'mean':
            params = np.ones(n_features, dtype=np.float32) * self.mean_scaler_epsilon
            for col in self.numeric_cols:
                if col in df.columns:
                    cidx = self.feature_to_idx[col]
                    data = df.iloc[:historical_end][col].dropna().values
                    if len(data) > 0:
                        params[cidx] = np.mean(np.abs(data)) + self.mean_scaler_epsilon
            return params

        else:  # standard
            params = np.zeros((n_features, 2), dtype=np.float32)
            params[:, 1] = 1.0
            for col in self.numeric_cols:
                if col in df.columns:
                    cidx = self.feature_to_idx[col]
                    data = df.iloc[:historical_end][col].dropna().values
                    if len(data) > 0:
                        params[cidx, 0] = np.mean(data)
                        std = np.std(data, ddof=1)
                        params[cidx, 1] = std if std >= 1e-7 else 1.0
            return params

    def _fit_entity_level_scalers_fast(self):
        n_entities = len(self.entity_scaling_windows)
        print(f"Fitting entity-level scalers for {n_entities} entities...")

        imported_entity_scalers: Dict = {}
        entities_to_fit = set(self.entity_scaling_windows.keys())
        entities_using_imported: set = set()

        if self.imported_scalers and self.mode in ('val', 'test'):
            imported_entity_scalers = self.imported_scalers.get('entity_scalers', {})
            entities_using_imported = entities_to_fit & set(imported_entity_scalers)
            entities_to_fit -= entities_using_imported
            if entities_using_imported:
                print(f"  Reusing scalers for {len(entities_using_imported)} entities")
            if entities_to_fit:
                print(f"  Fitting fresh scalers for {len(entities_to_fit)} new entities")

        # Map entity → all window indices
        entity_to_all_windows: Dict[str, List[int]] = defaultdict(list)
        for widx, window in enumerate(self.windows):
            entity_to_all_windows[window['entity_id']].append(widx)

        # Assign imported scalers to ALL windows of that entity
        for eid in entities_using_imported:
            scaler = imported_entity_scalers[eid]
            for widx in entity_to_all_windows[eid]:
                self.scaler_params[widx] = scaler.copy()

        if not entities_to_fit:
            print("Entity-level scaling complete (all from imports)")
            return

        # Extract historical data for entities that need fresh scalers
        entity_data_for_scaling: Dict = {}
        for eid in entities_to_fit:
            df = self.series_data[eid]
            historical_values: Dict[str, List] = {c: [] for c in self.numeric_cols if c in df.columns}
            for widx in self.entity_scaling_windows[eid]:
                window = self.windows[widx]
                sl = slice(None, window['historical_end']) if window['padding_steps'] > 0 \
                    else slice(window['start_idx'], window['historical_end'])
                for col in self.numeric_cols:
                    if col in df.columns:
                        historical_values[col].append(df.iloc[sl][col].values)
            entity_data_for_scaling[eid] = {
                c: np.concatenate(v) for c, v in historical_values.items() if v
            }

        def _fit_entity_scaler(eid, hist_vals, numeric_cols, feature_to_idx, scaling_method, epsilon):
            n = len(numeric_cols)
            if scaling_method == 'mean':
                params = np.ones(n, dtype=np.float32) * epsilon
                for col in numeric_cols:
                    if col in hist_vals:
                        data = hist_vals[col][~np.isnan(hist_vals[col])]
                        if len(data) > 0:
                            params[feature_to_idx[col]] = np.mean(np.abs(data)) + epsilon
            else:
                params = np.zeros((n, 2), dtype=np.float32)
                params[:, 1] = 1.0
                for col in numeric_cols:
                    if col in hist_vals:
                        data = hist_vals[col][~np.isnan(hist_vals[col])]
                        if len(data) > 0:
                            cidx = feature_to_idx[col]
                            params[cidx, 0] = np.mean(data)
                            std = np.std(data, ddof=1)
                            params[cidx, 1] = std if std >= 1e-7 else 1.0
            return eid, params

        items = list(entity_data_for_scaling.items())
        if self.n_jobs != 1 and len(items) > 10:
            backend = get_optimal_backend()
            results = Parallel(n_jobs=self.n_jobs, backend=backend)(
                delayed(_fit_entity_scaler)(eid, hv, self.numeric_cols, self.feature_to_idx,
                                           self.scaling_method, self.mean_scaler_epsilon)
                for eid, hv in items
            )
        else:
            results = [_fit_entity_scaler(eid, hv, self.numeric_cols, self.feature_to_idx,
                                          self.scaling_method, self.mean_scaler_epsilon)
                       for eid, hv in items]

        entity_scalers = dict(results)
        for widx, window in enumerate(self.windows):
            eid = window['entity_id']
            if eid in entity_scalers:
                self.scaler_params[widx] = entity_scalers[eid]

        print(f"Entity-level scaling complete — reused: {len(entities_using_imported)}, fresh: {len(entities_to_fit)}")

        # Handle ultra-short inference-only entities
        if self.mode in ('test', 'val') and self.allow_inference_only_entities:
            ultra_short = [(widx, w['entity_id']) for widx, w in enumerate(self.windows)
                           if w.get('is_ultra_short', False)]
            if ultra_short:
                self._assign_category_scalers_to_ultra_short(ultra_short)

    def _assign_category_scalers_to_ultra_short(self, ultra_short_entities):
        category_cols = self.cold_start_scaler_cols or self.static_categorical_cols
        if not category_cols:
            self._assign_global_scalers_to_ultra_short(ultra_short_entities)
            return

        print(f"Using columns for category-based scaler matching: {category_cols}")

        def _get_category_key(eid):
            edf = self.series_data.get(eid)
            key = []
            for col in category_cols:
                if edf is not None and col in edf.columns:
                    val = edf[col].iloc[0]
                    if col in self.categorical_encoders:
                        enc = self.categorical_encoders[col]
                        key.append(int(enc.transform([val])[0]) if pd.notna(val) and val in enc.classes_ else 'unknown')
                    else:
                        key.append(val if pd.notna(val) else 'unknown')
                else:
                    key.append('unknown')
            return tuple(key)

        # Build library from normal windows
        cat_scalers: Dict = defaultdict(list)
        for widx, window in enumerate(self.windows):
            if not window.get('is_ultra_short', False):
                cat_scalers[_get_category_key(window['entity_id'])].append(self.scaler_params[widx])

        avg_cat_scalers = {k: np.mean(v, axis=0) for k, v in cat_scalers.items() if v}

        if avg_cat_scalers:
            fallback = np.mean(list(avg_cat_scalers.values()), axis=0)
        elif self.scaling_method == 'mean':
            fallback = np.ones(len(self.numeric_cols)) * self.mean_scaler_epsilon
        else:
            fallback = np.zeros((len(self.numeric_cols), 2))
            fallback[:, 1] = 1.0

        stats: Dict[str, int] = defaultdict(int)
        for widx, eid in ultra_short_entities:
            key = _get_category_key(eid)
            if key in avg_cat_scalers:
                self.scaler_params[widx] = avg_cat_scalers[key].copy()
                stats['exact_match'] += 1
            else:
                # Partial match on first column
                if len(category_cols) > 1:
                    matches = [s for k, s in avg_cat_scalers.items() if k[0] == key[0]]
                    if matches:
                        self.scaler_params[widx] = np.mean(matches, axis=0)
                        stats['partial_match'] += 1
                        continue
                self.scaler_params[widx] = fallback.copy()
                stats['fallback'] += 1

        print(f"Ultra-short entity scaler assignment: {dict(stats)}")

    def _assign_global_scalers_to_ultra_short(self, ultra_short_entities):
        normal_scalers = [self.scaler_params[widx]
                          for widx, window in enumerate(self.windows)
                          if not window.get('is_ultra_short', False)]
        if normal_scalers:
            global_scaler = np.mean(normal_scalers, axis=0)
            for widx, eid in ultra_short_entities:
                self.scaler_params[widx] = global_scaler.copy()

    # ------------------------------------------------------------------
    # Scaler persistence
    # ------------------------------------------------------------------

    def export_entity_scalers(self) -> Optional[Dict]:
        if self.scaling_method == 'none' or self.scaling_strategy != 'entity_level':
            return None
        entity_scalers = {
            eid: self.scaler_params[widxs[0]].copy()
            for eid, widxs in self.entity_scaling_windows.items() if widxs
        }
        return {
            'scaling_method': self.scaling_method,
            'mean_scaler_epsilon': self.mean_scaler_epsilon,
            'feature_to_idx': self.feature_to_idx.copy(),
            'numeric_cols': self.numeric_cols.copy(),
            'entity_scalers': entity_scalers,
            'n_entities': len(entity_scalers),
        }

    def save_entity_scalers(self, filepath: str):
        d = self.export_entity_scalers()
        if d:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'wb') as f:
                pickle.dump(d, f)
            print(f"Saved scalers for {d['n_entities']} entities to {p}")

    @staticmethod
    def load_entity_scalers(filepath: str) -> Dict:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Encoder management
    # ------------------------------------------------------------------

    def _fit_and_save_encoders(self):
        print(f"Fitting encoders on {self.mode} data...")
        self.categorical_encoders: Dict = {}
        self.encoder_metadata: Dict = {}

        all_cat_cols = (self.static_categorical_cols
                        + self.temporal_known_categorical_cols
                        + self.temporal_unknown_categorical_cols)

        for col in all_cat_cols:
            all_values: set = set()
            for edf in self.series_data.values():
                if col in edf.columns:
                    if col in self.static_categorical_cols:
                        v = edf[col].iloc[0]
                        if pd.notna(v):
                            all_values.add(v)
                    else:
                        all_values.update(edf[col].dropna().unique())

            if all_values:
                enc = LabelEncoder()
                enc.fit(list(all_values))
                self.categorical_encoders[col] = enc
                self.encoder_metadata[col] = {
                    'n_classes': len(enc.classes_),
                    'fitted_on_mode': self.mode,
                    'fitted_on_date': pd.Timestamp.now().isoformat(),
                    'sample_values': enc.classes_[:10].tolist(),
                }
                print(f"  {col}: {len(enc.classes_)} unique values")

        self._save_encoders()
        print(f"Encoders saved to {self.encoders_path}")

    def _save_encoders(self):
        self.encoders_path.mkdir(parents=True, exist_ok=True)
        with open(self.encoders_path / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(self.categorical_encoders, f)
        with open(self.encoders_path / 'encoder_metadata.json', 'w') as f:
            json.dump(self.encoder_metadata, f, indent=2)
        with open(self.encoders_path / 'feature_config.json', 'w') as f:
            json.dump({
                'static_categorical_cols': self.static_categorical_cols,
                'temporal_known_categorical_cols': self.temporal_known_categorical_cols,
                'temporal_unknown_categorical_cols': self.temporal_unknown_categorical_cols,
            }, f, indent=2)

    def _load_encoders(self):
        enc_file = self.encoders_path / 'label_encoders.pkl'
        if not enc_file.exists():
            raise FileNotFoundError(
                f"No encoders found at {enc_file}. "
                "Train the model first or provide correct encoders_path."
            )
        print(f"Loading encoders from {self.encoders_path}...")
        with open(enc_file, 'rb') as f:
            self.categorical_encoders = pickle.load(f)
        meta_file = self.encoders_path / 'encoder_metadata.json'
        if meta_file.exists():
            with open(meta_file) as f:
                self.encoder_metadata = json.load(f)
        cfg_file = self.encoders_path / 'feature_config.json'
        if cfg_file.exists():
            with open(cfg_file) as f:
                self._validate_feature_config(json.load(f))
        print(f"Loaded {len(self.categorical_encoders)} encoders")
        self._report_unknown_categories()

    def _validate_feature_config(self, saved_config):
        current = {
            'static_categorical_cols': self.static_categorical_cols,
            'temporal_known_categorical_cols': self.temporal_known_categorical_cols,
            'temporal_unknown_categorical_cols': self.temporal_unknown_categorical_cols,
        }
        mismatches = [k for k in current if set(current[k]) != set(saved_config.get(k, []))]
        if mismatches:
            print(f"WARNING: Feature config mismatch for: {mismatches}")

    def _report_unknown_categories(self):
        print("\nChecking for unknown categories...")
        unknown_stats: Dict = {}
        for col, enc in self.categorical_encoders.items():
            known = set(enc.classes_)
            unknown: set = set()
            for edf in self.series_data.values():
                if col not in edf.columns:
                    continue
                if col in self.static_categorical_cols:
                    v = edf[col].iloc[0]
                    if pd.notna(v) and v not in known:
                        unknown.add(v)
                else:
                    for v in edf[col].dropna().unique():
                        if v not in known:
                            unknown.add(v)
            if unknown:
                unknown_stats[col] = {'count': len(unknown), 'samples': list(unknown)[:5]}

        if unknown_stats:
            print("  Found unknown categories (will be encoded as -1):")
            for col, s in unknown_stats.items():
                print(f"    {col}: {s['count']} unknown — examples: {s['samples']}")
        else:
            print("  No unknown categories found")

    def _safe_encode_categorical(self, values, col_name: str, encoder):
        encoded = np.full(len(values), -1, dtype=np.int32)
        for i, val in enumerate(values):
            if pd.notna(val) and val in encoder.classes_:
                encoded[i] = encoder.transform([val])[0]
        return encoded

    # ------------------------------------------------------------------
    # Data caching
    # ------------------------------------------------------------------

    def _preprocess_and_cache_data(self):
        print("Preprocessing and caching data for fast access...")
        self.cached_series_data: Dict = {}
        self.cached_categorical_data: Dict = {}
        self.cached_static_data: Dict = {}
        self.unknown_category_counts: Dict = {}
        entities_list = list(self.series_data.keys())

        def _process_entity(eid, df, numeric_cols, static_numeric_cols, static_categorical_cols,
                            temporal_known_categorical_cols, temporal_unknown_categorical_cols,
                            categorical_encoders):
            numeric_data = {col: df[col].fillna(0).values.astype(np.float32)
                            for col in numeric_cols if col in df.columns}

            categorical_data: Dict = {}
            unknown_counts: Dict = {}
            for col in temporal_known_categorical_cols + temporal_unknown_categorical_cols:
                if col in df.columns and col in categorical_encoders:
                    enc = categorical_encoders[col]
                    values = df[col].values
                    encoded = np.full(len(values), -1, dtype=np.int32)
                    for i, val in enumerate(values):
                        if pd.notna(val) and val in enc.classes_:
                            encoded[i] = enc.transform([val])[0]
                    categorical_data[col] = encoded
                    n_unk = int(np.sum(encoded == -1))
                    if n_unk > 0:
                        unknown_counts[col] = n_unk

            static_data: Dict = {}
            for col in static_numeric_cols:
                if col in df.columns:
                    v = df[col].iloc[0]
                    static_data[col] = float(v) if pd.notna(v) else 0.0
            for col in static_categorical_cols:
                if col in df.columns and col in categorical_encoders:
                    v = df[col].iloc[0]
                    enc = categorical_encoders[col]
                    if pd.notna(v) and v in enc.classes_:
                        static_data[col] = int(enc.transform([v])[0])
                    else:
                        static_data[col] = -1
                        unknown_counts[col] = unknown_counts.get(col, 0) + 1

            return eid, numeric_data, categorical_data, static_data, unknown_counts

        if self.n_jobs != 1 and len(entities_list) > 50:
            backend = get_optimal_backend()
            results = Parallel(n_jobs=self.n_jobs, backend=backend)(
                delayed(_process_entity)(
                    eid, self.series_data[eid],
                    self.numeric_cols, self.static_numeric_cols,
                    self.static_categorical_cols,
                    self.temporal_known_categorical_cols,
                    self.temporal_unknown_categorical_cols,
                    self.categorical_encoders,
                ) for eid in entities_list
            )
        else:
            results = [_process_entity(
                eid, self.series_data[eid],
                self.numeric_cols, self.static_numeric_cols,
                self.static_categorical_cols,
                self.temporal_known_categorical_cols,
                self.temporal_unknown_categorical_cols,
                self.categorical_encoders,
            ) for eid in entities_list]

        for eid, nd, cd, sd, unk in results:
            self.cached_series_data[eid] = nd
            self.cached_categorical_data[eid] = cd
            self.cached_static_data[eid] = sd
            for col, cnt in unk.items():
                self.unknown_category_counts[col] = self.unknown_category_counts.get(col, 0) + cnt

        if self.unknown_category_counts:
            print("\nUnknown category statistics:")
            for col, cnt in self.unknown_category_counts.items():
                print(f"  {col}: {cnt} unknown values encoded as -1")

        self._print_encoding_summary()
        print("Data preprocessing complete!")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        eid = window['entity_id']
        entity_padding = self.entity_padding_values.get(eid, {})

        numeric_data = self.cached_series_data[eid]
        categorical_data = self.cached_categorical_data[eid]
        static_data = self.cached_static_data[eid]

        start_idx = window['start_idx']
        end_idx = window['end_idx']
        historical_end = window['historical_end']
        padding_steps = window['padding_steps']

        sample: Dict = {
            'entity_id': eid,
            'window_idx': idx,
            'mask': torch.ones(self.total_steps, dtype=torch.float32),
        }

        if self.weight_col:
            df = self.series_data[eid]
            if self.weight_col in df.columns:
                sample['entity_weight'] = torch.tensor(float(df[self.weight_col].iloc[0]), dtype=torch.float32)
            else:
                sample['entity_weight'] = torch.tensor(1.0, dtype=torch.float32)

        if self.recency_alpha > 0:
            rel = window['end_idx'] / len(self.series_data[eid])
            sample['recency_weight'] = torch.tensor(float(np.exp(self.recency_alpha * rel)), dtype=torch.float32)
        else:
            sample['recency_weight'] = torch.tensor(1.0, dtype=torch.float32)

        # Scaling
        scaled_numeric_data: Dict = {}
        if self.scaling_method != 'none' and self.scaler_params is not None:
            if self.scaling_method == 'mean':
                scales = self.scaler_params[idx]
                for col, data in numeric_data.items():
                    if col in self.feature_to_idx:
                        scale = scales[self.feature_to_idx[col]]
                        scaled_numeric_data[col] = data / scale if scale > 0 else data
            else:
                params = self.scaler_params[idx]
                for col, data in numeric_data.items():
                    if col in self.feature_to_idx:
                        cidx = self.feature_to_idx[col]
                        mean, std = params[cidx, 0], params[cidx, 1]
                        scaled_numeric_data[col] = (data - mean) / std if std > 0 else data
        else:
            scaled_numeric_data = numeric_data

        # Store scaler params for inverse transform
        if self.scaling_method != 'none' and self.scaler_params is not None and self.target_cols:
            n_t = len(self.target_cols)
            if self.scaling_method == 'mean':
                tsp = np.zeros(n_t, dtype=np.float32)
                for i, col in enumerate(self.target_cols):
                    if col in self.feature_to_idx:
                        tsp[i] = self.scaler_params[idx, self.feature_to_idx[col]]
                sample['target_scale'] = torch.tensor(tsp, dtype=torch.float32)
            else:
                tsp = np.zeros((n_t, 2), dtype=np.float32)
                for i, col in enumerate(self.target_cols):
                    if col in self.feature_to_idx:
                        cidx = self.feature_to_idx[col]
                        tsp[i, 0] = self.scaler_params[idx, cidx, 0]
                        tsp[i, 1] = self.scaler_params[idx, cidx, 1]
                sample['target_mean'] = torch.tensor(tsp[:, 0], dtype=torch.float32)
                sample['target_std'] = torch.tensor(tsp[:, 1], dtype=torch.float32)

        # Static numeric
        if self.static_numeric_cols:
            sv = [float(static_data[c]) if c in static_data and not pd.isna(static_data[c]) else 0.0
                  for c in self.static_numeric_cols]
            if sv:
                sample['static_numeric'] = torch.tensor(sv, dtype=torch.float32)

        # Static categorical
        if self.static_categorical_cols:
            sv = [int(static_data[c]) if c in static_data and not pd.isna(static_data[c]) else -1
                  for c in self.static_categorical_cols]
            if sv:
                sample['static_categorical'] = torch.tensor(sv, dtype=torch.long)

        # Ultra-short series
        if window.get('is_ultra_short', False):
            sample = self._build_ultra_short_sample(sample, window, scaled_numeric_data,
                                                     categorical_data, entity_padding, idx)
        else:
            sample = self._build_regular_sample(sample, window, scaled_numeric_data,
                                                 categorical_data, entity_padding,
                                                 start_idx, end_idx, historical_end, padding_steps)

        sample['time_index'] = torch.arange(self.total_steps)
        return sample

    def _build_ultra_short_sample(self, sample, window, scaled_numeric_data,
                                   categorical_data, entity_padding, idx):
        future_data_len = window.get('future_data_len', window['end_idx'])
        future_padding_steps = self.prediction_steps - future_data_len

        mask = torch.zeros(self.total_steps, dtype=torch.float32)
        mask[self.historical_steps:self.historical_steps + future_data_len] = 1.0
        sample['mask'] = mask

        def _scaled_pad_value(col):
            pv = entity_padding.get(col, 0.0)
            if self.scaling_method == 'mean' and col in self.feature_to_idx:
                sc = self.scaler_params[idx, self.feature_to_idx[col]]
                return pv / sc if sc > 0 else pv
            elif self.scaling_method == 'standard' and col in self.feature_to_idx:
                cidx = self.feature_to_idx[col]
                m, s = self.scaler_params[idx, cidx, 0], self.scaler_params[idx, cidx, 1]
                return (pv - m) / s if s > 0 else pv
            return pv

        # Historical targets (all padded)
        if self.target_cols:
            ht = []
            for col in self.target_cols:
                ht.append(np.full(self.historical_steps, _scaled_pad_value(col), dtype=np.float32))
            sample['historical_targets'] = torch.tensor(
                np.stack(ht, axis=1) if len(ht) > 1 else ht[0].reshape(-1, 1), dtype=torch.float32)

        # Historical unknown numeric (padded)
        if self.temporal_unknown_numeric_cols:
            un = [np.full(self.historical_steps, entity_padding.get(c, 0.0), dtype=np.float32)
                  for c in self.temporal_unknown_numeric_cols]
            sample['temporal_unknown_numeric'] = torch.tensor(np.stack(un, axis=1), dtype=torch.float32)

        # Historical unknown categorical (padded)
        if self.temporal_unknown_categorical_cols:
            uc = [np.full(self.historical_steps, self.categorical_padding_value, dtype=np.int32)
                  for _ in self.temporal_unknown_categorical_cols]
            sample['temporal_unknown_categorical'] = torch.tensor(np.stack(uc, axis=1), dtype=torch.long)

        # Known numeric (hist padded + future real/padded)
        if self.temporal_known_numeric_cols:
            kn = []
            for col in self.temporal_known_numeric_cols:
                hist_part = np.full(self.historical_steps, entity_padding.get(col, 0.0), dtype=np.float32)
                if col in scaled_numeric_data and len(scaled_numeric_data[col]) > 0:
                    actual = scaled_numeric_data[col][:future_data_len]
                    if future_padding_steps > 0:
                        last = actual[-1] if len(actual) > 0 else entity_padding.get(col, 0.0)
                        future_part = np.concatenate([actual, np.full(future_padding_steps, last, dtype=np.float32)])
                    else:
                        future_part = actual[:self.prediction_steps]
                else:
                    future_part = np.full(self.prediction_steps, entity_padding.get(col, 0.0), dtype=np.float32)
                kn.append(np.concatenate([hist_part, future_part]))
            sample['temporal_known_numeric'] = torch.tensor(np.stack(kn, axis=1), dtype=torch.float32)

        # Known categorical (hist padded + future real/padded)
        if self.temporal_known_categorical_cols:
            kc = []
            for col in self.temporal_known_categorical_cols:
                hist_part = np.full(self.historical_steps, self.categorical_padding_value, dtype=np.int32)
                if col in categorical_data and len(categorical_data[col]) > 0:
                    actual = categorical_data[col][:future_data_len]
                    if future_padding_steps > 0:
                        future_part = np.concatenate([actual, np.full(future_padding_steps, self.categorical_padding_value, dtype=np.int32)])
                    else:
                        future_part = actual[:self.prediction_steps]
                else:
                    future_part = np.full(self.prediction_steps, self.categorical_padding_value, dtype=np.int32)
                kc.append(np.concatenate([hist_part, future_part]))
            sample['temporal_known_categorical'] = torch.tensor(np.stack(kc, axis=1), dtype=torch.long)

        # Future targets
        if self.target_cols:
            ft = []
            for col in self.target_cols:
                pv = entity_padding.get(col, 0.0)
                if col in scaled_numeric_data and len(scaled_numeric_data[col]) > 0:
                    actual = scaled_numeric_data[col][:future_data_len]
                    if future_padding_steps > 0:
                        last = actual[-1] if len(actual) > 0 else pv
                        fd = np.concatenate([actual, np.full(future_padding_steps, last, dtype=np.float32)])
                    else:
                        fd = actual[:self.prediction_steps]
                else:
                    fd = np.full(self.prediction_steps, pv, dtype=np.float32)
                ft.append(fd)
            sample['future_targets'] = torch.tensor(
                np.stack(ft, axis=1) if len(ft) > 1 else ft[0].reshape(-1, 1), dtype=torch.float32)

        return sample

    def _build_regular_sample(self, sample, window, scaled_numeric_data, categorical_data,
                               entity_padding, start_idx, end_idx, historical_end, padding_steps):
        if padding_steps > 0:
            sample['mask'][:padding_steps] = 0

        def _pad_num(arr, target_len, col):
            if len(arr) >= target_len:
                return arr
            pv = entity_padding.get(col, 0.0)
            return np.concatenate([np.full(target_len - len(arr), pv, dtype=arr.dtype), arr])

        def _pad_cat(arr, target_len):
            if len(arr) >= target_len:
                return arr
            return np.concatenate([np.full(target_len - len(arr), self.categorical_padding_value, dtype=arr.dtype), arr])

        hs = slice(0 if padding_steps > 0 else start_idx, historical_end)
        fs = slice(historical_end, end_idx)

        def _get(col, data_dict, sl):
            d = data_dict[col][start_idx:end_idx][sl.start - start_idx:sl.stop - start_idx] \
                if sl.start >= start_idx else data_dict[col][sl]
            return data_dict[col][sl]

        # Temporal known numeric
        if self.temporal_known_numeric_cols:
            kn = []
            for col in self.temporal_known_numeric_cols:
                if col in scaled_numeric_data:
                    full = scaled_numeric_data[col][start_idx:end_idx]
                    if padding_steps > 0:
                        full = _pad_num(full, self.total_steps, col)
                    kn.append(full)
            if kn:
                sample['temporal_known_numeric'] = torch.tensor(np.stack(kn, axis=1), dtype=torch.float32)

        # Temporal known categorical
        if self.temporal_known_categorical_cols:
            kc = []
            for col in self.temporal_known_categorical_cols:
                if col in categorical_data:
                    full = categorical_data[col][start_idx:end_idx]
                    if padding_steps > 0:
                        full = _pad_cat(full, self.total_steps)
                    kc.append(full)
            if kc:
                sample['temporal_known_categorical'] = torch.tensor(np.stack(kc, axis=1), dtype=torch.long)

        # Temporal unknown numeric
        if self.temporal_unknown_numeric_cols:
            un = []
            for col in self.temporal_unknown_numeric_cols:
                if col in scaled_numeric_data:
                    h_start = 0 if padding_steps > 0 else start_idx
                    hd = scaled_numeric_data[col][h_start:historical_end]
                    if padding_steps > 0:
                        hd = _pad_num(hd, self.historical_steps, col)
                    un.append(hd)
            if un:
                sample['temporal_unknown_numeric'] = torch.tensor(np.stack(un, axis=1), dtype=torch.float32)

        # Temporal unknown categorical
        if self.temporal_unknown_categorical_cols:
            uc = []
            for col in self.temporal_unknown_categorical_cols:
                if col in categorical_data:
                    h_start = 0 if padding_steps > 0 else start_idx
                    hd = categorical_data[col][h_start:historical_end]
                    if padding_steps > 0:
                        hd = _pad_cat(hd, self.historical_steps)
                    uc.append(hd)
            if uc:
                sample['temporal_unknown_categorical'] = torch.tensor(np.stack(uc, axis=1), dtype=torch.long)

        # Targets
        if self.target_cols:
            h_start = 0 if padding_steps > 0 else start_idx
            ht = []
            for col in self.target_cols:
                if col in scaled_numeric_data:
                    hd = scaled_numeric_data[col][h_start:historical_end]
                    if padding_steps > 0:
                        hd = _pad_num(hd, self.historical_steps, col)
                    ht.append(hd)
            if ht:
                sample['historical_targets'] = torch.tensor(
                    np.stack(ht, axis=1) if len(ht) > 1 else ht[0].reshape(-1, 1), dtype=torch.float32)

            ft = []
            for col in self.target_cols:
                if col in scaled_numeric_data:
                    ft.append(scaled_numeric_data[col][historical_end:end_idx])
            if ft:
                sample['future_targets'] = torch.tensor(
                    np.stack(ft, axis=1) if len(ft) > 1 else ft[0].reshape(-1, 1), dtype=torch.float32)

        return sample

    # ------------------------------------------------------------------
    # Inverse transform & utilities
    # ------------------------------------------------------------------

    def inverse_transform_predictions(
        self,
        predictions: torch.Tensor,
        window_indices: List[int],
        target_col: Optional[str] = None,
    ) -> torch.Tensor:
        """Inverse-transform scaled predictions back to the original scale."""
        if self.scaling_method == 'none' or self.scaler_params is None:
            return predictions
        target_col = target_col or self.target_cols[0]
        cidx = self.feature_to_idx.get(target_col)
        if cidx is None:
            return predictions
        pnp = predictions.detach().cpu().numpy()
        out = np.zeros_like(pnp)
        for i, widx in enumerate(window_indices):
            if self.scaling_method == 'mean':
                out[i] = pnp[i] * self.scaler_params[widx, cidx]
            else:
                mean = self.scaler_params[widx, cidx, 0]
                std = self.scaler_params[widx, cidx, 1]
                out[i] = pnp[i] * std + mean
        return torch.FloatTensor(out)

    def get_encoder_mappings(self) -> Dict:
        """Return a dict of {col: {classes: [...], mapping: {val: idx}}} for all categoricals."""
        mappings: Dict = {}
        if hasattr(self, 'categorical_encoders'):
            for col, enc in self.categorical_encoders.items():
                mappings[col] = {
                    'classes': enc.classes_.tolist(),
                    'mapping': {v: i for i, v in enumerate(enc.classes_)},
                }
        return mappings

    def inverse_transform_categorical(self, encoded_values, feature_name: str):
        """Convert encoded categorical values back to original labels."""
        if feature_name not in self.categorical_encoders:
            raise ValueError(f"No encoder found for feature '{feature_name}'")
        enc = self.categorical_encoders[feature_name]
        if torch.is_tensor(encoded_values):
            encoded_values = encoded_values.cpu().numpy()
        shape = encoded_values.shape
        flat = encoded_values.flatten()
        result = []
        for v in flat:
            if v == -1:
                result.append(None)
            else:
                try:
                    result.append(enc.inverse_transform([v])[0])
                except Exception:
                    result.append(None)
        return np.array(result).reshape(shape)

    def get_window_timestamps(self, window_idx: int) -> pd.DatetimeIndex:
        """Return the full timestamp range for a given window."""
        w = self.windows[window_idx]
        df = self.series_data[w['entity_id']]
        idx = df.index[w['start_idx']:w['end_idx']]
        if hasattr(idx, 'to_timestamp'):
            idx = idx.to_timestamp()
        return pd.DatetimeIndex(idx)

    def get_future_timestamps(self, window_idx: int) -> pd.DatetimeIndex:
        """Return only the future/prediction timestamps for a window."""
        w = self.windows[window_idx]
        df = self.series_data[w['entity_id']]
        if w.get('is_ultra_short', False):
            return pd.DatetimeIndex(df.index[:w['future_data_len']])
        idx = df.index[w['historical_end']:w['end_idx']]
        if hasattr(idx, 'to_timestamp'):
            idx = idx.to_timestamp()
        return pd.DatetimeIndex(idx)

    def get_window_info(self, window_idx: int) -> Dict:
        """Return metadata dict for a given window."""
        w = self.windows[window_idx]
        return {
            'entity_id': w['entity_id'],
            'window_idx': window_idx,
            'start_idx': w['start_idx'],
            'end_idx': w['end_idx'],
            'historical_end': w['historical_end'],
            'padding_steps': w['padding_steps'],
            'timestamps': self.get_window_timestamps(window_idx),
            'future_timestamps': self.get_future_timestamps(window_idx),
        }

    def get_dataset_statistics(self) -> Dict:
        """Return a comprehensive statistics dictionary."""
        series_lengths = [len(df) for df in self.series_data.values()]
        stats: Dict = {
            'n_series': len(self.series_data),
            'n_windows': len(self.windows),
            'n_padded_windows': sum(1 for w in self.windows if w['padding_steps'] > 0),
            'features': {
                'n_targets': len(self.target_cols),
                'n_static_numeric': len(self.static_numeric_cols),
                'n_static_categorical': len(self.static_categorical_cols),
                'n_temporal_known_numeric': len(self.temporal_known_numeric_cols),
                'n_temporal_known_categorical': len(self.temporal_known_categorical_cols),
                'n_temporal_unknown_numeric': len(self.temporal_unknown_numeric_cols),
                'n_temporal_unknown_categorical': len(self.temporal_unknown_categorical_cols),
            },
            'window_params': {
                'historical_steps': self.historical_steps,
                'prediction_steps': self.prediction_steps,
                'stride': self.stride,
                'padding_enabled': self.enable_padding,
            },
            'scaling': {
                'method': self.scaling_method,
                'memory_mb': self.scaler_params.nbytes / (1024**2) if self.scaler_params is not None else 0,
            },
        }
        if series_lengths:
            stats['series_lengths'] = {
                'min': min(series_lengths),
                'max': max(series_lengths),
                'mean': float(np.mean(series_lengths)),
                'median': float(np.median(series_lengths)),
            }
        return stats

    # ------------------------------------------------------------------
    # Reporting / debug helpers
    # ------------------------------------------------------------------

    def _print_encoding_summary(self):
        if not self.cached_static_data:
            return
        print("\n=== Encoding Summary ===")
        if self.static_categorical_cols:
            print("Static Categorical Encodings (first 3 entities):")
            for i, (eid, sd) in enumerate(list(self.cached_static_data.items())[:3]):
                vals = [f"{c}={sd[c]}" for c in self.static_categorical_cols if c in sd]
                print(f"  Entity {eid}: {', '.join(vals)}")

    def inspect_padding_values(self, n_entities: int = 3):
        """Debug helper: print computed padding values for the first N entities."""
        print("\n=== Padding Values Inspection ===")
        for i, (eid, pv) in enumerate(self.entity_padding_values.items()):
            if i >= n_entities:
                break
            print(f"\nEntity: {eid}")
            for col, val in list(pv.items())[:5]:
                print(f"  {col}: {val:.4f}")

    def _report_dataset_stats(self):
        n_windows = len(self.windows)
        n_padded = sum(1 for w in self.windows if w['padding_steps'] > 0)
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Series:  {len(self.series_data)}")
        print(f"Windows: {n_windows:,}  (padded: {n_padded:,})")
        print(f"Features: {len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical")
        if self.scaler_params is not None:
            mb = self.scaler_params.nbytes / (1024 ** 2)
            print(f"Scaler params: {mb:.1f} MB  shape={self.scaler_params.shape}")


# ---------------------------------------------------------------------------
# Adapter: TFT model format
# ---------------------------------------------------------------------------

class TFTDataAdapter:
    """
    Bridges :class:`OptimizedTFTDataset` output to the ``TemporalFusionTransformer``
    model's input format.

    The dataset returns stacked feature tensors; the model expects separate lists,
    one tensor per feature.  This adapter handles the split and also provides the
    ``collate_fn`` for :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, dataset: OptimizedTFTDataset):
        self.dataset = dataset
        self.static_numeric_cols = dataset.static_numeric_cols
        self.static_categorical_cols = dataset.static_categorical_cols
        self.temporal_known_numeric_cols = dataset.temporal_known_numeric_cols
        self.temporal_known_categorical_cols = dataset.temporal_known_categorical_cols
        self.temporal_unknown_numeric_cols = dataset.temporal_unknown_numeric_cols
        self.temporal_unknown_categorical_cols = dataset.temporal_unknown_categorical_cols
        self.target_cols = dataset.target_cols
        self.historical_steps = dataset.historical_steps
        self.prediction_steps = dataset.prediction_steps

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        collated: Dict = {}
        for key in batch[0].keys():
            if key == 'entity_id':
                collated[key] = [s[key] for s in batch]
            elif key == 'window_idx':
                collated[key] = torch.tensor([s[key] for s in batch])
            else:
                collated[key] = torch.stack([s[key] for s in batch])
        return collated

    def adapt_for_tft(self, batch: Dict[str, torch.Tensor]) -> Dict:
        inputs: Dict = {}

        # Static categorical
        if 'static_categorical' in batch and self.static_categorical_cols:
            inputs['static_categorical'] = [batch['static_categorical'][:, i]
                                             for i in range(len(self.static_categorical_cols))]

        # Static continuous
        if 'static_numeric' in batch and self.static_numeric_cols:
            inputs['static_continuous'] = [batch['static_numeric'][:, i:i+1]
                                            for i in range(len(self.static_numeric_cols))]

        # Historical categorical  (unknown ++ known[:hist])
        hist_cat = []
        if 'temporal_unknown_categorical' in batch and self.temporal_unknown_categorical_cols:
            hist_cat += [batch['temporal_unknown_categorical'][:, :, i]
                         for i in range(len(self.temporal_unknown_categorical_cols))]
        if 'temporal_known_categorical' in batch and self.temporal_known_categorical_cols:
            hist_cat += [batch['temporal_known_categorical'][:, :self.historical_steps, i]
                         for i in range(len(self.temporal_known_categorical_cols))]
        if hist_cat:
            inputs['historical_categorical'] = hist_cat

        # Historical continuous  (targets ++ unknown ++ known[:hist])
        hist_cont = []
        if 'historical_targets' in batch:
            t = batch['historical_targets']
            if t.dim() == 3:
                hist_cont += [t[:, :, i] for i in range(t.shape[2])]
            else:
                hist_cont.append(t)
        if 'temporal_unknown_numeric' in batch and self.temporal_unknown_numeric_cols:
            hist_cont += [batch['temporal_unknown_numeric'][:, :, i]
                          for i in range(len(self.temporal_unknown_numeric_cols))]
        if 'temporal_known_numeric' in batch and self.temporal_known_numeric_cols:
            hist_cont += [batch['temporal_known_numeric'][:, :self.historical_steps, i]
                          for i in range(len(self.temporal_known_numeric_cols))]
        if hist_cont:
            inputs['historical_continuous'] = hist_cont

        # Future categorical
        if 'temporal_known_categorical' in batch and self.temporal_known_categorical_cols:
            inputs['future_categorical'] = [batch['temporal_known_categorical'][:, self.historical_steps:, i]
                                             for i in range(len(self.temporal_known_categorical_cols))]

        # Future continuous
        if 'temporal_known_numeric' in batch and self.temporal_known_numeric_cols:
            inputs['future_continuous'] = [batch['temporal_known_numeric'][:, self.historical_steps:, i]
                                            for i in range(len(self.temporal_known_numeric_cols))]

        # Padding mask
        if 'mask' in batch:
            inputs['padding_mask'] = self._create_padding_mask(batch['mask'])

        # Pass-through scaler params
        for key in ('target_scale', 'target_mean', 'target_std'):
            if key in batch:
                inputs[key] = batch[key]

        if 'entity_weight' in batch:
            inputs['entity_weight'] = batch['entity_weight']

        inputs['historical_targets'] = batch.get('historical_targets')
        inputs['future_targets'] = batch.get('future_targets')

        for key in ('time_index', 'window_idx', 'entity_id', 'recency_weight'):
            if key in batch:
                inputs[key] = batch[key]

        return inputs

    def adapt_for_encoder_only(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """Adapt batch for :class:`~tft_pytorch.models.TFTEncoderOnly`."""
        inputs: Dict = {}

        if 'static_categorical' in batch and self.static_categorical_cols:
            inputs['static_categorical'] = [batch['static_categorical'][:, i]
                                             for i in range(len(self.static_categorical_cols))]

        if 'static_numeric' in batch and self.static_numeric_cols:
            inputs['static_continuous'] = [batch['static_numeric'][:, i:i+1]
                                            for i in range(len(self.static_numeric_cols))]

        hist_cat = []
        if 'temporal_unknown_categorical' in batch and self.temporal_unknown_categorical_cols:
            hist_cat += [batch['temporal_unknown_categorical'][:, :, i]
                         for i in range(len(self.temporal_unknown_categorical_cols))]
        if 'temporal_known_categorical' in batch and self.temporal_known_categorical_cols:
            hist_cat += [batch['temporal_known_categorical'][:, :self.historical_steps, i]
                         for i in range(len(self.temporal_known_categorical_cols))]
        if hist_cat:
            inputs['historical_categorical'] = hist_cat

        hist_cont = []
        if 'temporal_unknown_numeric' in batch and self.temporal_unknown_numeric_cols:
            hist_cont += [batch['temporal_unknown_numeric'][:, :, i]
                          for i in range(len(self.temporal_unknown_numeric_cols))]
        if 'temporal_known_numeric' in batch and self.temporal_known_numeric_cols:
            hist_cont += [batch['temporal_known_numeric'][:, :self.historical_steps, i]
                          for i in range(len(self.temporal_known_numeric_cols))]
        if hist_cont:
            inputs['historical_continuous'] = hist_cont

        if 'mask' in batch:
            hist_mask = batch['mask'][:, :self.historical_steps]
            inputs['padding_mask'] = self._create_padding_mask(hist_mask)

        if 'future_targets' in batch and batch['future_targets'] is not None:
            t = batch['future_targets'][:, 0, :]
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            inputs['target'] = t

        for key in ('target_scale', 'target_mean', 'target_std',
                    'entity_weight', 'recency_weight', 'window_idx', 'entity_id'):
            if key in batch:
                inputs[key] = batch[key]

        return inputs

    def _create_padding_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """1 → masked (padded), 0 → attend (real).  Shape: [B, 1, T]."""
        return (1.0 - mask).unsqueeze(1)


# ---------------------------------------------------------------------------
# Adapter: TCN model format
# ---------------------------------------------------------------------------

class TCNDataAdapter:
    """
    Adapter that converts :class:`OptimizedTFTDataset` outputs to a single
    concatenated feature tensor expected by TCN-style models.

    The TCN model receives:
    * ``numeric_features``   – [batch, timesteps, n_numeric]
    * ``categorical_features`` – [batch, timesteps, n_categorical] (raw indices)
    * ``targets``, ``mask``

    The TCN model itself is responsible for embedding the categorical indices.
    """

    def __init__(self, dataset: OptimizedTFTDataset):
        self.dataset = dataset
        self.static_numeric_cols = dataset.static_numeric_cols
        self.static_categorical_cols = dataset.static_categorical_cols
        self.temporal_known_numeric_cols = dataset.temporal_known_numeric_cols
        self.temporal_known_categorical_cols = dataset.temporal_known_categorical_cols
        self.temporal_unknown_numeric_cols = dataset.temporal_unknown_numeric_cols
        self.temporal_unknown_categorical_cols = dataset.temporal_unknown_categorical_cols
        self.target_cols = dataset.target_cols
        self.historical_steps = dataset.historical_steps
        self.prediction_steps = dataset.prediction_steps

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        collated: Dict = {}
        for key in batch[0].keys():
            if key == 'entity_id':
                collated[key] = [s[key] for s in batch]
            elif key == 'window_idx':
                collated[key] = torch.tensor([s[key] for s in batch])
            else:
                collated[key] = torch.stack([s[key] for s in batch])
        return collated

    def adapt_for_tcn(self, batch: Dict[str, torch.Tensor], encoder_only: bool = False) -> Dict:
        bs = batch['mask'].shape[0]
        timesteps = self.historical_steps if encoder_only else (self.historical_steps + self.prediction_steps)
        num_feats = []
        cat_feats = []

        if 'historical_targets' in batch:
            ht = batch['historical_targets']
            if ht.dim() == 2:
                ht = ht.unsqueeze(-1)
            if not encoder_only:
                ht = torch.cat([ht, torch.zeros(bs, self.prediction_steps, ht.shape[-1])], dim=1)
            num_feats.append(ht)

        if 'temporal_unknown_numeric' in batch and self.temporal_unknown_numeric_cols:
            un = batch['temporal_unknown_numeric']
            if not encoder_only:
                un = torch.cat([un, torch.zeros(bs, self.prediction_steps, un.shape[-1])], dim=1)
            num_feats.append(un)

        if 'temporal_unknown_categorical' in batch and self.temporal_unknown_categorical_cols:
            uc = batch['temporal_unknown_categorical']
            if not encoder_only:
                pad = torch.full((bs, self.prediction_steps, uc.shape[-1]), -1)
                uc = torch.cat([uc, pad], dim=1)
            cat_feats.append(uc)

        if 'temporal_known_numeric' in batch and self.temporal_known_numeric_cols:
            kn = batch['temporal_known_numeric']
            if encoder_only:
                kn = kn[:, :self.historical_steps, :]
            num_feats.append(kn)

        if 'temporal_known_categorical' in batch and self.temporal_known_categorical_cols:
            kc = batch['temporal_known_categorical']
            if encoder_only:
                kc = kc[:, :self.historical_steps, :]
            cat_feats.append(kc)

        if 'static_numeric' in batch and self.static_numeric_cols:
            sn = batch['static_numeric'].unsqueeze(1).expand(-1, timesteps, -1)
            num_feats.append(sn)

        if 'static_categorical' in batch and self.static_categorical_cols:
            sc = batch['static_categorical'].unsqueeze(1).expand(-1, timesteps, -1)
            cat_feats.append(sc)

        output: Dict = {
            'numeric_features': torch.cat(num_feats, dim=-1) if num_feats else torch.zeros(bs, timesteps, 1),
            'categorical_features': torch.cat(cat_feats, dim=-1) if cat_feats else torch.zeros(bs, timesteps, 0, dtype=torch.long),
            'mask': batch['mask'][:, :timesteps] if encoder_only else batch['mask'],
        }
        if 'future_targets' in batch:
            output['targets'] = batch['future_targets']
        output['entity_id'] = batch.get('entity_id')
        output['window_idx'] = batch.get('window_idx')
        return output


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_tft_dataloader(
    dataset: OptimizedTFTDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, TFTDataAdapter]:
    """
    Create a :class:`~torch.utils.data.DataLoader` paired with a
    :class:`TFTDataAdapter`.

    Returns
    -------
    (dataloader, adapter)
    """
    adapter = TFTDataAdapter(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=adapter.collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    return loader, adapter


def create_tcn_dataloader(
    dataset: OptimizedTFTDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, TCNDataAdapter]:
    """Create a DataLoader with a :class:`TCNDataAdapter`."""
    adapter = TCNDataAdapter(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=adapter.collate_fn,
    )
    return loader, adapter


def create_uniform_embedding_dims(
    dataset: OptimizedTFTDataset,
    hidden_layer_size: int = 160,
) -> Dict[str, Tuple[int, int]]:
    """
    Build the ``categorical_embedding_dims`` dict required by both TFT model
    constructors from an :class:`OptimizedTFTDataset`.

    Each value is a ``(vocab_size, embedding_dim)`` tuple, where
    *vocab_size* = number of known classes + 2 (padding/unknown tokens) and
    *embedding_dim* = ``hidden_layer_size``.

    Returns
    -------
    dict  e.g. ``{"static_cat_0": (12, 160), "historical_cat_0": (8, 160), ...}``
    """
    dims: Dict = {}
    mappings = dataset.get_encoder_mappings()

    for i, col in enumerate(dataset.static_categorical_cols):
        if col in mappings:
            dims[f"static_cat_{i}"] = (len(mappings[col]['classes']) + 2, hidden_layer_size)

    all_hist_cat = dataset.temporal_unknown_categorical_cols + dataset.temporal_known_categorical_cols
    for i, col in enumerate(all_hist_cat):
        if col in mappings:
            dims[f"historical_cat_{i}"] = (len(mappings[col]['classes']) + 2, hidden_layer_size)

    for i, col in enumerate(dataset.temporal_known_categorical_cols):
        if col in mappings:
            dims[f"future_cat_{i}"] = (len(mappings[col]['classes']) + 2, hidden_layer_size)

    return dims


def inverse_transform_predictions(
    predictions: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    dataset: OptimizedTFTDataset,
    target_idx: int = 0,
) -> torch.Tensor:
    """
    Convenience function to inverse-transform model predictions using the
    scaler parameters stored in the batch dictionary.

    Parameters
    ----------
    predictions : Tensor [batch, ...]
    batch       : The collated batch dict from the DataLoader
    dataset     : The dataset instance (provides scaling_method)
    target_idx  : Which target to inverse-transform (when multi-target)
    """
    if dataset.scaling_method == 'none':
        return predictions

    pnp = predictions.detach().cpu().numpy()

    if dataset.scaling_method == 'mean' and 'target_scale' in batch:
        scale = batch['target_scale'][:, target_idx].cpu().numpy()
        while scale.ndim < pnp.ndim:
            scale = scale[..., np.newaxis]
        return torch.tensor(pnp * scale)

    if dataset.scaling_method == 'standard' and 'target_mean' in batch and 'target_std' in batch:
        mean = batch['target_mean'][:, target_idx].cpu().numpy()
        std = batch['target_std'][:, target_idx].cpu().numpy()
        while mean.ndim < pnp.ndim:
            mean = mean[..., np.newaxis]
            std = std[..., np.newaxis]
        return torch.tensor(pnp * std + mean)

    return predictions
