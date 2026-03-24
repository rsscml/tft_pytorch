#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import multiprocessing as mp
import pickle
import json
from collections import defaultdict
import queue
import threading
import warnings
warnings.filterwarnings('ignore')


# # OptimizedTFTDataset Documentation
# 
# ## Overview
# 
# `OptimizedTFTDataset` is a PyTorch Dataset implementation specifically designed for Temporal Fusion Transformer (TFT) models. It efficiently handles multiple time series with different lengths, creates sliding windows for training, and manages feature preprocessing with production-ready encoder consistency across train/validation/test splits.
# 
# ## Core Features & Capabilities
# 
# - **Window-specific scaling without data leakage**: Each window's scalers are fitted using only historically available data
# - **Memory-efficient storage**: Stores scaler parameters as numpy arrays instead of objects (97% memory reduction)
# - **Variable-length series support**: Handles series of different lengths with automatic padding
# - **Production-ready encoding**: Consistent categorical encoding across train/val/test datasets
# - **Parallel processing**: Multi-core support for window creation and scaler fitting
# - **Unknown category handling**: Gracefully handles new categories in validation/test data
# 
# ## Supported Data Sources
# 
# ```python
# # 1. Pandas DataFrame
# dataset = OptimizedTFTDataset(data_source=df, ...)
# 
# # 2. CSV file path
# dataset = OptimizedTFTDataset(data_source='data.csv', ...)
# 
# # 3. List of CSV files (one per entity)
# dataset = OptimizedTFTDataset(data_source=['entity1.csv', 'entity2.csv'], ...)
# ```
# 
# ## Constructor Parameters
# 
# | Parameter | Type | Default | Description |
# |-----------|------|---------|-------------|
# | **data_source** | DataFrame/str/List | - | Input data source |
# | **features_config** | Dict | - | Feature configuration dictionary |
# | **historical_steps** | int | 30 | Number of historical timesteps |
# | **prediction_steps** | int | 1 | Number of future timesteps to predict |
# | **stride** | int | 1 | Step size between consecutive windows |
# | **enable_padding** | bool | True | Allow padding for short series |
# | **padding_strategy** | str | zero | strategy used for padding (zero, mean, forward_fill, intelligent) |
# | **categorical_padding_value** | int | -1 | Always -1 for categorical variables padding |
# | **min_historical_steps** | int | historical_steps//3 | Minimum non-padded historical steps |
# | **scaler_path** | str | None | Filepath of the stored numeric scalers
# | **scaling_strategy** | str | 'per_window' | scale numeric variables at 'per_window' or at 'entity_level' |
# | **scaling_method** | str | 'standard' | 'standard', 'mean', or 'none' |
# | **mean_scaler_epsilon** | float | 1.0 | Epsilon for mean scaling |
# | **n_jobs** | int | -1 | Number of parallel jobs (-1 = all CPUs) |
# | **max_series** | int | None | Maximum number of series to load |
# | **mode** | str | 'train' | 'train', 'val', or 'test' |
# | **encoders_path** | str | './.encoders' | Path to save/load encoders |
# | **fit_encoders** | bool | None | Explicitly control encoder fitting |
# | **preprocessing_fn** | Callable | None | Optional preprocessing function |
# 
# ## Feature Configuration Dictionary
# 
# The `features_config` dictionary explicitly defines all feature types:
# 
# ```python
# features_config = {
#     # Required identifiers
#     "entity_col": "store_id",           # Column with entity/series identifiers
#     "time_index_col": "date",           # Datetime column
#     
#     # Target (what we're predicting)
#     "target_col": "sales",              # Can be string or list of strings
#     
#     # Static features (constant per entity)
#     "static_numeric_col_list": ["store_size", "latitude"],
#     "static_categorical_col_list": ["store_type", "region"],
#     
#     # Temporal known features (values known in future)
#     "temporal_known_numeric_col_list": ["temperature", "price"],
#     "temporal_known_categorical_col_list": ["day_of_week", "holiday"],
#     
#     # Temporal unknown features (only historical values available)
#     "temporal_unknown_numeric_col_list": ["competitor_price", "foot_traffic"],
#     "temporal_unknown_categorical_col_list": ["stock_status", "promotion"],
#     
#     # Optional weight column
#     "wt_col": "importance"              # Entity weights for weighted training
# }
# ```
# 
# ## Scaling Methods
# 
# | Method | Formula | Memory per Feature | Use Case |
# |--------|---------|-------------------|----------|
# | **standard** | `(x - mean) / std` | 8 bytes | Normally distributed features |
# | **mean** | `x / (mean(abs(x)) + ε)` | 4 bytes | Intermittent/sparse time series |
# | **none** | `x` | 0 bytes | Pre-scaled data |
# 
# ## Output Format
# 
# Each sample from the dataset returns a dictionary with the following keys:
# 
# ```python
# {
#     # Always present
#     'entity_id': str,                          # Series identifier
#     'window_idx': int,                         # Window index in dataset
#     'mask': torch.FloatTensor,                 # [total_steps] - Binary mask (0=padded, 1=real)
#     'time_index': torch.LongTensor,            # [total_steps] - Time position indices
#     
#     # Feature tensors (if applicable)
#     'static_numeric': torch.FloatTensor,       # [n_static_numeric]
#     'static_categorical': torch.LongTensor,    # [n_static_categorical]
#     
#     # Full window (historical + future)
#     'temporal_known_numeric': torch.FloatTensor,      # [total_steps, n_known_numeric]
#     'temporal_known_categorical': torch.LongTensor,   # [total_steps, n_known_categorical]
#     
#     # Historical only
#     'temporal_unknown_numeric': torch.FloatTensor,    # [historical_steps, n_unknown_numeric]
#     'temporal_unknown_categorical': torch.LongTensor, # [historical_steps, n_unknown_categorical]
#     
#     # Targets
#     'historical_targets': torch.FloatTensor,    # [historical_steps, n_targets]
#     'future_targets': torch.FloatTensor,        # [prediction_steps, n_targets]
#     
#     # Optional
#     'entity_weight': torch.FloatTensor          # [1]
# }
# ```
# 
# ## Key Methods
# 
# | Method | Description |
# |--------|-------------|
# | `__len__()` | Returns total number of windows |
# | `__getitem__(idx)` | Returns preprocessed sample dictionary |
# | `inverse_transform_predictions()` | Convert scaled predictions back to original scale |
# | `inverse_transform_categorical()` | Convert encoded categories back to original values |
# | `get_encoder_mappings()` | Get encoding mappings for all categorical features |
# | `get_dataset_statistics()` | Get comprehensive dataset statistics |
# 
# ## Key Design Principles
# 
# ### Scaling vs Encoding Strategy
# 
# The dataset implements two distinct preprocessing strategies:
# 
# - **Window-specific scaling**: Numeric features are scaled per window using only historically available data up to that window's endpoint. This prevents data leakage and ensures valid backtesting. Each window stores its own scaler parameters (mean/std or scale factor).
# 
# - **Global encoding**: Categorical features use global encoders fitted once on the entire training dataset and saved to disk. These same encoders are loaded and reused for validation, test, and inference datasets, ensuring consistent category mappings across all data splits.
# 
# This dual approach balances accuracy (no data leakage in scaling) with consistency (same categorical encodings everywhere).
# 
# ### Padding for Variable-Length Series
# 
# Short time series are handled through intelligent padding:
# 
# ```python
# # For a series with only 20 timesteps when historical_steps=30:
# # - Actual historical data: 20 - prediction_steps
# # - Padding needed: 30 - actual_historical
# # - The series is left-padded using the padding strategy (default is 0 padding). Other strategies are: mean, forward_fill, intelligent
# # - A binary mask indicates real (1) vs padded (0) timesteps
# ```
# 
# This is how forward_fill padding strategy works (forward_fill can be misleading; effectively first available value is used to back fill):
# 
# #### Current "forward_fill" implementation:
# ```python
# padding_val = values[0]  # Takes FIRST value
# padding = np.full(padding_steps, padding_val)
# return np.concatenate([padding, arr])  # Puts it at the BEGINNING
# 
# #### Example:
# #### Original series: [10, 12, 15, 18, 20]
# #### Need 3 padding steps
# #### Result: [10, 10, 10, 10, 12, 15, 18, 20]
# ####          ^^^^^^^^^^^^^ padding using first value
# ```
# The zero padding, mean padding and intelligent i.e., feature/logic-based custom padding are straightforward.
# 
# The `min_historical_steps` parameter ensures a minimum amount of real data is present. Series shorter than this threshold are excluded entirely.
# 
# ### Temporal Train/Validation/Test Splitting
# 
# For time series, random splitting would cause data leakage. Use temporal splitting instead:
# 
# ```python
# from torch.utils.data import Subset
# 
# # Create temporal splits - DO NOT SHUFFLE!
# dataset = OptimizedTFTDataset(data_source=df, ...)
# n_windows = len(dataset)
# 
# train_size = int(0.7 * n_windows)
# val_size = int(0.15 * n_windows)
# 
# # Sequential splits preserve temporal order
# train_dataset = Subset(dataset, range(0, train_size))
# val_dataset = Subset(dataset, range(train_size, train_size + val_size))
# test_dataset = Subset(dataset, range(train_size + val_size, n_windows))
# 
# # Important: shuffle=False for validation/test to maintain temporal order
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# ```
# 
# ### Performance & Robustness Considerations
# 
# **Data Leakage Prevention:**
# - Scalers never see future data - each window's scaler only uses data up to that point in time
# - Unknown categorical values in test/inference are encoded as -1, allowing the model to learn to handle unseen categories
# - Temporal splitting ensures test data comes after training data
# 
# **Speed Optimizations:**
# - Parallel processing for window creation and scaler fitting (use `n_jobs=-1`)
# - Numpy arrays for fast data access instead of repeated DataFrame operations
# - Scaler parameters stored efficiently (4-8 bytes per feature per window)
# - Preprocessing done once and stored in memory
# 
# **Robustness Features:**
# - Handles missing data (NaN → 0 for numeric, -1 for categorical)
# - Variable-length series with automatic padding
# - Graceful handling of unknown categories in production
# - Validation of feature configuration between train and test
# 
# **Memory Efficiency:**
# - 97% memory reduction compared to storing scikit-learn scaler objects
# - For 500k windows with 20 features: ~80MB (standard scaling) vs ~3GB (object storage)
# 
# ## Usage Examples
# 
# ### Basic Training Setup
# 
# ```python
# # Prepare data
# train_df = pd.read_csv('train_data.csv')
# val_df = pd.read_csv('val_data.csv')
# 
# # Configure features
# features_config = {
#     "entity_col": "store_id",
#     "time_index_col": "date",
#     "target_col": "sales",
#     "temporal_known_numeric_col_list": ["temperature", "price"],
#     "temporal_known_categorical_col_list": ["day_of_week"],
#     # ... other features
# }
# 
# # Create datasets with consistent encoding
# train_dataset = OptimizedTFTDataset(
#     data_source=train_df,
#     features_config=features_config,
#     historical_steps=60,
#     prediction_steps=14,
#     mode='train',
#     encoders_path='./model_encoders'
# )
# 
# val_dataset = OptimizedTFTDataset(
#     data_source=val_df,
#     features_config=features_config,
#     historical_steps=60,
#     prediction_steps=14,
#     mode='val',
#     encoders_path='./model_encoders'  # Uses encoders from training
# )
# 
# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# ```
# 
# ### Inference on New Data
# 
# ```python
# # Load new data
# new_data = pd.read_csv('new_data.csv')
# 
# # Create inference dataset (uses saved encoders)
# inference_dataset = OptimizedTFTDataset(
#     data_source=new_data,
#     features_config=features_config,
#     mode='test',
#     encoders_path='./model_encoders',
#     fit_encoders=False  # Must use existing encoders
# )
# 
# # Get predictions and inverse transform
# dataloader = DataLoader(inference_dataset, batch_size=32)
# for batch in dataloader:
#     predictions = model(batch)
#     
#     # Convert back to original scale
#     original_scale = inference_dataset.inverse_transform_predictions(
#         predictions,
#         batch['window_idx'].tolist(),
#         target_col='sales'
#     )
# ```
# 
# ### Custom Preprocessing
# 
# ```python
# def preprocess_series(df):
#     # Remove outliers, fill missing values, etc.
#     df['sales'] = df['sales'].clip(lower=0, upper=df['sales'].quantile(0.99))
#     return df
# 
# dataset = OptimizedTFTDataset(
#     data_source=df,
#     features_config=features_config,
#     preprocessing_fn=preprocess_series,
#     # ... other parameters
# )
# ```

# ## Scaling Strategies in OptimizedTFTDataset
# 
# ### Overview
# The dataset supports two scaling strategies for numeric features, controlled by the `scaling_strategy` parameter:
# 
# #### 1. Per-Window Scaling (`scaling_strategy='per_window'`)
# **Default strategy** - Each window gets its own scaler fitted only on its historical data.
# 
# **Advantages:**
# - ✅ **Zero data leakage** - Future data never influences past scalers
# - ✅ **Adapts to local patterns** - Handles non-stationary series well  
# - ✅ **Theoretically pure** - Valid for rigorous backtesting
# 
# **Disadvantages:**
# - ❌ **Distribution shift** - Same raw value scaled differently across windows
# - ❌ **Training instability** - TFT struggles with inconsistent feature scales
# - ❌ **Slower convergence** - Model needs 2-3x more epochs
# 
# **Best for:**
# - Research requiring strict no-leakage guarantee
# - Highly non-stationary series with regime changes
# - When relative patterns matter more than absolute values
# 
# #### 2. Entity-Level Scaling (`scaling_strategy='entity_level'`)
# **Optimized strategy** - One scaler per entity using non-overlapping historical windows.
# 
# **Advantages:**
# - ✅ **Stable training** - Consistent feature distributions within entities
# - ✅ **Fast convergence** - TFT trains much better
# - ✅ **Minimal leakage** - Uses only historical portions of non-overlapping windows
# - ✅ **Efficient** - 10x faster than per-window scaling
# 
# **Disadvantages:**
# - ❌ **Slight statistical leakage** - Multiple historical windows influence the scaler
# - ❌ **Less adaptive** - Single scaler for entire entity
# 
# **Best for:**
# - Production systems with TFT models
# - Datasets with many windows per entity (>10)
# - When training stability is prioritized
# 
# ### Implementation Details
# 
# #### How Entity-Level Scaling Prevents Leakage
# ```python
# # Windows separated by total_steps have ZERO overlap
# Window 1: [0--------------30][30---40]  # Historical | Future
# Window 2:                     [40-------------70][70---80]
#           └─ No overlap ─┘
# ```
# Only non-overlapping windows are used to fit the entity scaler, ensuring no data point is used for both scaling and prediction.
# 
# #### Memory Efficiency
# Both strategies store only scaler parameters (not objects):
# - **Standard scaling**: 8 bytes/feature/window (mean + std)
# - **Mean scaling**: 4 bytes/feature/window (scale factor only)
# - **Memory savings**: ~97% vs storing scikit-learn objects
# 
# ### Usage Examples
# 
# ```python
# # Per-window scaling (default)
# dataset = OptimizedTFTDataset(
#     data_source=df,
#     scaling_strategy='per_window',
#     scaling_method='standard',
#     historical_steps=30,
#     prediction_steps=10,
#     stride=1
# )
# 
# # Entity-level scaling (recommended for TFT)
# dataset = OptimizedTFTDataset(
#     data_source=df,
#     scaling_strategy='entity_level',
#     scaling_method='standard',
#     historical_steps=30,
#     prediction_steps=10,
#     stride=1  # Can safely use stride=1 with entity-level!
# )
# ```
# 
# ### Decision Guide
# 
# | Criterion | Per-Window | Entity-Level |
# |-----------|------------|--------------|
# | **Data Leakage** | None | Minimal (~statistical only) |
# | **Training Stability** | Poor | Excellent |
# | **Convergence Speed** | Slow | Fast |
# | **Memory Usage** | Higher | Lower |
# | **Computation Speed** | Slower | 10x faster |
# | **Short Entities (<5 windows)** | ✓ Better | Less reliable |
# | **Many Windows per Entity (>10)** | Inefficient | ✓ Optimal |
# 
# ### Recommendations
# 
# 1. **For TFT models**: Use `entity_level` - the model architecture assumes consistent feature distributions
# 2. **For strict backtesting**: Use `per_window` if zero leakage is critical
# 3. **For production**: Use `entity_level` with `stride ≥ historical_steps` to minimize leakage
# 4. **For non-stationary series**: Consider `per_window` or use `scaling_method='mean'` for robustness
# 
# ### Technical Notes
# 
# - **Padded windows**: Scalers are fitted only on actual data, never on padded values
# - **Short entities**: Entities with < `min_historical_steps + prediction_steps` timesteps are excluded
# - **Unknown features**: Only historical data is used for scaling target and unknown temporal features
# - **Parallel processing**: Both strategies support multi-core processing via `n_jobs` parameter
# 
# 
# ## `scaler_path` Parameter Documentation
# 
# ### Overview
# The `scaler_path` parameter enables consistent scaling across train/validation/test splits by saving and reusing entity-level scalers. This ensures that entities appearing in multiple datasets use the same scaling parameters learned during training.
# 
# ### Parameter Details
# - **Type**: `Optional[str]` - Path to a pickle file
# - **Default**: `None` - Fresh scaling for all entities
# - **Requirement**: Only works with `scaling_strategy='entity_level'`
# 
# ### Mode-Specific Behavior
# 
# #### Training Mode (`mode='train'`)
# - **Fits fresh scalers** for all entities using training data
# - **If `scaler_path` provided**: Automatically saves fitted scalers to the specified path after fitting
# - **If `scaler_path` is None**: No saving occurs (backward compatible)
# 
# ```python
# train_dataset = OptimizedTFTDataset(
#     data_source=train_df,
#     mode='train',
#     scaling_strategy='entity_level',
#     scaler_path='./scalers/train_scalers.pkl',  # Will save here
#     # ... other parameters
# )
# 
# # With scaler reuse
# val_dataset = OptimizedTFTDataset(
#     data_source=val_df,
#     mode='val',
#     scaling_strategy='entity_level',
#     scaler_path='./scalers/train_scalers.pkl',  # Will load and reuse
#     fit_encoders=False,  # Also reuse encoders from training
#     # ... other parameters
# )
# 
# # Without scaler reuse (original behavior)
# test_dataset = OptimizedTFTDataset(
#     data_source=test_df,
#     mode='test',
#     scaling_strategy='entity_level',
#     scaler_path=None,  # Fresh scaling for all entities
#     # ... other parameters
# )
# ```    

# In[ ]:


# Determine OS to inform Joblib Parallel's backend selection
import platform

def get_optimal_backend():
    """Determine the optimal Joblib backend based on the OS."""
    system = platform.system().lower()
    
    if system == 'windows':
        return 'loky'  # Best for Windows
    elif system in ['linux', 'darwin']:  # darwin = macOS
        return 'threading'  # Better for Unix-like systems
    else:
        return 'loky'  # Default fallback
    


# In[ ]:


# Pytorch Dataset class for training & inferencing with Temporal Fusion Transformer model


class OptimizedTFTDataset(Dataset):
    """
    Memory-efficient TFT Dataset with window-specific scaling.
    
    Key optimizations:
    - Stores only scaler parameters (not objects) in numpy arrays
    - Window-specific scaling without data leakage
    - Support for variable-length series with padding
    - Explicit feature configuration
    
    Memory usage for 500k windows with 20 features:
    - Standard scaling (mean + std): ~80 MB
    - Mean scaling: ~40 MB
    """
    
    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame, List[Union[str, Path]]],
        features_config: Dict[str, Any],
        
        # Window parameters
        historical_steps: int = 30,
        prediction_steps: int = 1,
        stride: int = 1,
        
        # Padding configuration
        enable_padding: bool = True,
        padding_strategy: str = 'zero',  # NEW: 'zero', 'mean', 'forward_fill', 'intelligent'
        categorical_padding_value: int = -1,    # NEW: Always -1 for categorical 
        min_historical_steps: Optional[int] = None,
        allow_inference_only_entities: bool = False,
        
        # Scaling options
        scaler_path: Optional[str] = None, # filepath where scalers dictionary is to be saved or retrieved from
        scaling_strategy: str = 'per_window', # 'per_window' or 'entity_level'
        scaling_method: str = 'standard',  # 'standard', 'mean', or 'none'
        mean_scaler_epsilon: float = 1.0,
        
        # static categorical columns to use for cold-start entities scaler assignment
        cold_start_scaler_cols: Optional[List[str]] = None,
        
        # recency weighting
        recency_alpha: float = 0.0,  # NEW: 0 means no recency weighting
        
        # Performance
        n_jobs: int = -1,
        
        # Other
        max_series: Optional[int] = None,
        mode: str = 'train',
        encoders_path=None,  # Path to save/load encoders
        fit_encoders=None,  # Explicitly control encoder fitting
        preprocessing_fn: Optional[Callable] = None
    ):
        """
        Initialize TFT dataset with optimized window-specific scaling.
        
        Args:
            data_source: Input data (DataFrame, file path, or list of files)
            features_config: Dictionary specifying feature types
            historical_steps: Number of historical timesteps
            prediction_steps: Number of future timesteps to predict
            stride: Step size between consecutive windows
            enable_padding: Allow padding for short series
            padding_strategy: One of 'zero', 'mean', 'forward_fill' or 'intelligent' (feature specific logic for padding)
            categorical_padding_value: Always -1 for categorical variables
            min_historical_steps: Minimum non-padded historical steps required
            scaling_method: 'standard' (mean+std), 'mean', or 'none'
            mean_scaler_epsilon: Epsilon for mean scaling (avoid division by zero)
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            max_series: Maximum number of series to load
            mode: 'train', 'val', or 'test'
            preprocessing_fn: Optional preprocessing function
            
            features_config: Dictionary with keys:
                - entity_col: Column containing unique identifiers
                - target_col: Column(s) to predict
                - time_index_col: Datetime column
                - static_numeric_col_list: Static numeric features
                - static_categorical_col_list: Static categorical features
                - temporal_known_numeric_col_list: Known future numeric features
                - temporal_unknown_numeric_col_list: Unknown future numeric features
                - temporal_known_categorical_col_list: Known future categorical features
                - temporal_unknown_categorical_col_list: Unknown future categorical features
                - wt_col: Optional weight column for entity weighting
        """
        
        self.mode = mode
        self.encoders_path = Path(encoders_path) if encoders_path else Path('./.encoders')
        
        # Determine whether to fit or load encoders
        if fit_encoders is not None:
            self.should_fit_encoders = fit_encoders
        else:
            # Auto-decide: fit only for training mode
            self.should_fit_encoders = (mode == 'train')
        
        # Parse feature configuration
        self._parse_features_config(features_config)
        
        # Window parameters
        self.historical_steps = historical_steps
        self.prediction_steps = prediction_steps
        self.stride = stride
        self.total_steps = historical_steps + prediction_steps
        
        # Padding configuration
        self.enable_padding = enable_padding
        # Padding instance variables:
        self.padding_strategy = padding_strategy
        self.categorical_padding_value = categorical_padding_value
        # Initialize storage for padding values
        self.entity_padding_values = {}  # NEW: Store per-entity padding values
        # min historical steps
        if min_historical_steps is None:
            self.min_historical_steps = max(1, historical_steps // 3)
        else:
            self.min_historical_steps = min_historical_steps
        
        # Determine minimum series length
        self.allow_inference_only_entities = allow_inference_only_entities
        
        if allow_inference_only_entities and mode in ['test', 'val']:
            self.min_series_length = 0  # Accept any entity
        elif enable_padding:
            self.min_series_length = self.min_historical_steps + prediction_steps
        else:
            self.min_series_length = historical_steps + prediction_steps
        
        # Other parameters - NEW: Setup scaler import path based on mode
        self.scaler_path = scaler_path
        self.imported_scalers = None
        self.scaling_strategy = scaling_strategy
        self.scaling_method = scaling_method
        
        # for cold-start entities scale assignment
        if cold_start_scaler_cols:
            self.cold_start_scaler_cols = cold_start_scaler_cols
        else:
            self.cold_start_scaler_cols = self.static_categorical_cols
            
        self.mean_scaler_epsilon = mean_scaler_epsilon
        self.max_series = max_series
        self.preprocessing_fn = preprocessing_fn
        
        # Auto-determine scaler behavior based on mode
        if self.mode in ['val', 'test'] and self.scaler_path and self.scaling_strategy == 'entity_level' and self.scaling_method != 'none':
            # Load scalers for val/test
            self.imported_scalers = self.load_entity_scalers(self.scaler_path)
            print(f"Loaded scalers for {self.imported_scalers['n_entities']} entities from {self.scaler_path}")
        elif self.mode == 'train' and self.scaler_path and self.scaling_strategy == 'entity_level':
            # For training, we'll save after fitting
            print(f"Will save scalers to {self.scaler_path} after fitting")
        
        # Performance settings
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
        # recency weighting
        self.recency_alpha = recency_alpha
        
        # Initialize storage
        self.series_data = {}  # Raw unscaled data
        self.windows = []  # Window definitions
        
        # Optimized scaler storage - just numpy arrays!
        self.scaler_params = None  # Will be initialized based on scaling method
        
        # Load data
        self._load_data(data_source)
            
        # Apply preprocessing if provided
        if self.preprocessing_fn:
            self._apply_preprocessing()
        
        # NEW: Compute padding values BEFORE encoding
        if self.enable_padding:
            self._compute_entity_padding_values()
            
        # Handle encoders based on mode
        if self.should_fit_encoders:
            self._fit_and_save_encoders()
        else:
            self._load_encoders()
            
        # Create windows and fit scalers efficiently
        self._create_windows_and_fit_scalers()
        
        # NEW: Auto-save scalers after training
        if self.mode == 'train' and self.scaler_path and self.scaling_strategy == 'entity_level':
            self.save_entity_scalers(self.scaler_path)
            
        # Preprocess and cache data for fast access
        self._preprocess_and_cache_data()
            
        self._report_dataset_stats()
    
    def _parse_features_config(self, config: Dict[str, Any]):
        """Parse and validate feature configuration."""
        
        # Required columns
        self.entity_col = config.get('entity_col')
        self.time_col = config.get('time_index_col')
        
        # Target column(s)
        target = config.get('target_col')
        self.target_cols = [target] if isinstance(target, str) else (target or [])
        
        # Static features
        self.static_numeric_cols = config.get('static_numeric_col_list', [])
        self.static_categorical_cols = config.get('static_categorical_col_list', [])
        
        # Temporal known features (available in future)
        self.temporal_known_numeric_cols = config.get('temporal_known_numeric_col_list', [])
        self.temporal_known_categorical_cols = config.get('temporal_known_categorical_col_list', [])
        
        # Temporal unknown features (only historical)
        self.temporal_unknown_numeric_cols = config.get('temporal_unknown_numeric_col_list', [])
        self.temporal_unknown_categorical_cols = config.get('temporal_unknown_categorical_col_list', [])
        
        # Weight column
        self.weight_col = config.get('wt_col')
        
        # Combine categories
        self.categorical_cols = (self.static_categorical_cols + 
                                 self.temporal_known_categorical_cols + 
                                 self.temporal_unknown_categorical_cols)
        
        self.numeric_cols = (self.target_cols + 
                            self.static_numeric_cols + 
                            self.temporal_known_numeric_cols + 
                            self.temporal_unknown_numeric_cols)
        
        # Features that are unknown in future (need careful scaling)
        self.all_unknown_features = list(set(self.target_cols + 
                                            self.temporal_unknown_numeric_cols))
        
        # Features that are known in future
        self.all_known_features = list(set(self.temporal_known_numeric_cols))
        
        # Create feature to index mapping for efficient array access
        self.feature_to_idx = {feat: i for i, feat in enumerate(self.numeric_cols)}
        
        print("\nFeature Configuration:")
        print(f"  Targets: {len(self.target_cols)}")
        print(f"  Static: {len(self.static_numeric_cols)} numeric, {len(self.static_categorical_cols)} categorical")
        print(f"  Known temporal: {len(self.temporal_known_numeric_cols)} numeric, {len(self.temporal_known_categorical_cols)} categorical")
        print(f"  Unknown temporal: {len(self.temporal_unknown_numeric_cols)} numeric, {len(self.temporal_unknown_categorical_cols)} categorical")
        print(f"  Total numeric features: {len(self.numeric_cols)}")
    
    def _load_data(self, data_source):
        """Load data from various sources."""
        print(f"Loading data...")
        
        if isinstance(data_source, pd.DataFrame):
            self._load_from_dataframe(data_source)
        elif isinstance(data_source, (str, Path)):
            df = pd.read_csv(data_source)
            self._load_from_dataframe(df)
        elif isinstance(data_source, list):
            self._load_from_file_list(data_source)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")
    
    def _load_from_dataframe(self, df: pd.DataFrame):
        """Load data from a DataFrame with parallel processing for multiple entities."""

        if self.entity_col and self.entity_col in df.columns:
            entities = df[self.entity_col].unique()

            if self.max_series:
                entities = entities[:self.max_series]

            n_entities = len(entities)
            print(f"Processing {n_entities} entities...")

            # Define standalone function for parallel processing
            def process_single_entity(entity_id, df_full, entity_col, time_col, min_series_length):
                """Process a single entity's data."""
                # Filter for this entity
                entity_data = df_full[df_full[entity_col] == entity_id].copy()

                # Sort by time and set index if time column exists
                if time_col and time_col in entity_data.columns:
                    entity_data = entity_data.sort_values(time_col)
                    entity_data = entity_data.set_index(time_col)

                # Check length requirement
                if len(entity_data) >= min_series_length:
                    return str(entity_id), entity_data
                else:
                    return None, None

            # Decide whether to use parallel processing
            if self.n_jobs != 1 and n_entities > 50:  # Parallel for >50 entities
                backend = get_optimal_backend()
                print(f"  Using parallel processing with '{backend}' backend...")

                # Process in batches to manage memory
                batch_size = max(10, n_entities // (self.n_jobs * 4))

                def process_batch(batch_entities):
                    """Process a batch of entities."""
                    results = []
                    for entity_id in batch_entities:
                        result = process_single_entity(
                            entity_id, df, self.entity_col, 
                            self.time_col, self.min_series_length
                        )
                        if result[0] is not None:  # Valid entity
                            results.append(result)
                    return results

                # Create batches
                entity_batches = [entities[i:i+batch_size] 
                                for i in range(0, n_entities, batch_size)]

                # Parallel processing
                batch_results = Parallel(n_jobs=self.n_jobs, backend=backend)(
                    delayed(process_batch)(batch) for batch in entity_batches
                )

                # Flatten results and build series_data dictionary
                self.series_data = {}
                for batch in batch_results:
                    for entity_id, entity_data in batch:
                        self.series_data[entity_id] = entity_data

            else:
                # Sequential processing for small datasets
                print(f"  Using sequential processing...")
                self.series_data = {}

                for entity_id in entities:
                    entity_str, entity_data = process_single_entity(
                        entity_id, df, self.entity_col, 
                        self.time_col, self.min_series_length
                    )
                    if entity_str is not None:
                        self.series_data[entity_str] = entity_data

        else:
            # Single series (no entity column)
            if self.time_col and self.time_col in df.columns:
                df = df.sort_values(self.time_col)
                df = df.set_index(self.time_col)

            if len(df) >= self.min_series_length:
                self.series_data['series_0'] = df
            else:
                self.series_data = {}

        print(f"Loaded {len(self.series_data)} valid series from {len(entities) if self.entity_col else 1} total")
    
    def _load_from_dataframe_orig(self, df: pd.DataFrame):
        """Load data from a DataFrame."""
        if self.entity_col and self.entity_col in df.columns:
            entities = df[self.entity_col].unique()
            
            if self.max_series:
                entities = entities[:self.max_series]
            
            for entity_id in entities:
                entity_data = df[df[self.entity_col] == entity_id].copy()
                
                if self.time_col and self.time_col in entity_data.columns:
                    entity_data = entity_data.sort_values(self.time_col)
                    entity_data = entity_data.set_index(self.time_col)
                
                if len(entity_data) >= self.min_series_length:
                    self.series_data[str(entity_id)] = entity_data
        else:
            # Single series
            if self.time_col and self.time_col in df.columns:
                df = df.sort_values(self.time_col)
                df = df.set_index(self.time_col)
            
            if len(df) >= self.min_series_length:
                self.series_data['series_0'] = df
        
        print(f"Loaded {len(self.series_data)} series")
    
    def _load_from_file_list(self, file_paths: List[Union[str, Path]]):
        """Load data from multiple files."""
        if self.max_series:
            file_paths = file_paths[:self.max_series]
        
        for path in file_paths:
            df = pd.read_csv(path)
            
            if self.time_col and self.time_col in df.columns:
                df = df.sort_values(self.time_col)
                df = df.set_index(self.time_col)
            
            if len(df) >= self.min_series_length:
                entity_id = Path(path).stem if isinstance(path, (str, Path)) else f'series_{path}'
                self.series_data[entity_id] = df
        
        print(f"Loaded {len(self.series_data)} series")
    
    def _apply_preprocessing(self):
        """Apply preprocessing function to all series."""
        print("Applying preprocessing...")
        
        processed_data = {}
        for entity_id, df in self.series_data.items():
            try:
                processed_df = self.preprocessing_fn(df.copy())
                if len(processed_df) >= self.min_series_length:
                    processed_data[entity_id] = processed_df
            except Exception as e:
                print(f"Warning: Failed to preprocess series {entity_id}: {e}")
        
        self.series_data = processed_data
        print(f"Preprocessing complete. {len(self.series_data)} series remaining.")
    
    def _create_windows_and_fit_scalers(self):
        """
        Create windows and fit scalers efficiently.
        Store only scaler parameters, not objects!
        """
        print(f"Creating windows and fitting scalers...")
        
        # NEW: Track which windows to use for entity-level scaling
        self.entity_scaling_windows = defaultdict(list)  # entity_id -> list of window indices
        
        # First, create all windows
        for entity_id, df in self.series_data.items():
            series_len = len(df)
            
            # Track last window end for this entity (for non-overlapping selection)
            last_selected_end = -float('inf')
            
            # NEW: Case 0 - Ultra-short series (less than prediction_steps)
            if series_len < self.prediction_steps and self.allow_inference_only_entities:
                window_idx = len(self.windows)

                # For inference: treat all available data as FUTURE, not historical
                # The historical window will be completely padded
                window = {
                    'entity_id': entity_id,
                    'start_idx': 0,  
                    'end_idx': series_len,  # Actual data ends here
                    'historical_end': 0,  # No historical data - all is future
                    'padding_steps': self.historical_steps,  # Full historical padding
                    'future_data_len': series_len,  # Actual future data available
                    'future_padding_steps': self.prediction_steps - series_len,  # Future padding needed
                    'is_ultra_short': True
                }
                if self.scaling_strategy == 'entity_level':
                    self.entity_scaling_windows[entity_id].append(window_idx)

                self.windows.append(window)
                continue  # Skip to next entity
            
            # Case 1: Series is long enough for standard windows
            if series_len >= self.total_steps:
                n_windows = (series_len - self.total_steps) // self.stride + 1
                last_regular_start = -1  # Track the last regular window's start

                for i in range(n_windows):
                    start_idx = i * self.stride
                    end_idx = start_idx + self.total_steps
                    window_idx = len(self.windows)
                    last_regular_start = start_idx

                    window = {
                        'entity_id': entity_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'historical_end': start_idx + self.historical_steps,
                        'padding_steps': 0
                    }

                    # NEW: Check if this window should be used for entity-level scaling
                    if self.scaling_strategy == 'entity_level' and start_idx >= last_selected_end:
                        self.entity_scaling_windows[entity_id].append(window_idx)
                        last_selected_end = end_idx

                    self.windows.append(window)
            
                # This captures the most recent data
                last_possible_start = series_len - self.total_steps
                
                # Only add if it's different from the last regular window
                # (avoid duplicate if stride=1 or if it happens to align)
                if last_possible_start > last_regular_start:
                    window_idx = len(self.windows)

                    window = {
                        'entity_id': entity_id,
                        'start_idx': last_possible_start,
                        'end_idx': series_len,
                        'historical_end': last_possible_start + self.historical_steps,
                        'padding_steps': 0
                    }

                    # Check if this window should be used for entity-level scaling
                    if self.scaling_strategy == 'entity_level' and last_possible_start >= last_selected_end:
                        self.entity_scaling_windows[entity_id].append(window_idx)

                    self.windows.append(window)
            
            # Case 2: Short series that needs padding
            elif series_len >= self.min_series_length and self.enable_padding:
                actual_historical = series_len - self.prediction_steps
                padding_needed = self.historical_steps - actual_historical
                window_idx = len(self.windows)

                window = {
                    'entity_id': entity_id,
                    'start_idx': 0,
                    'end_idx': series_len,
                    'historical_end': actual_historical,
                    'padding_steps': padding_needed
                }

                # NEW: Padded windows are always used for entity scaling (only one per entity)
                if self.scaling_strategy == 'entity_level':
                    self.entity_scaling_windows[entity_id].append(window_idx)

                self.windows.append(window)
        
        n_windows = len(self.windows)
        n_features = len(self.numeric_cols)
        
        print(f"Created {n_windows} windows")
        
        if self.scaling_strategy == 'entity_level':
            n_scaling_windows = sum(len(indices) for indices in self.entity_scaling_windows.values())
            print(f"  Pre-identified {n_scaling_windows} non-overlapping windows for entity scaling")
        
        # Initialize scaler parameter storage based on scaling method
        if self.scaling_method == 'none':
            self.scaler_params = None
            print("No scaling will be applied")
            return
        elif self.scaling_method == 'mean':
            # Only need to store scale values (1 parameter per feature)
            self.scaler_params = np.zeros((n_windows, n_features), dtype=np.float32)
            print(f"Allocated {n_windows} x {n_features} array for mean scaling")
        elif self.scaling_method == 'standard':
            # Need to store mean and std (2 parameters per feature)
            self.scaler_params = np.zeros((n_windows, n_features, 2), dtype=np.float32)
            print(f"Allocated {n_windows} x {n_features} x 2 array for standard scaling")
        
        # Apply scaling strategy
        if self.scaling_strategy == 'entity_level':
            self._fit_entity_level_scalers_fast()
        else:
            # Fit scalers for each window (parallel processing)
            if self.n_jobs != 1 and n_windows > 100:
                backend = get_optimal_backend()
                print(f"Fitting scalers in parallel using '{backend}' backend...")

                # Use threading and batch
                batch_size = max(100, n_windows // (self.n_jobs * 10))

                def process_batch(start, end):
                    results = []
                    for i in range(start, min(end, n_windows)):
                        window = self.windows[i]
                        df = self.series_data[window['entity_id']]
                        params = self._fit_scalers_for_window_optimized(window, df)
                        results.append((i, params))
                    return results

                batched_results = Parallel(n_jobs=self.n_jobs, backend=backend)(delayed(process_batch)(i, i + batch_size)
                    for i in range(0, n_windows, batch_size))

                # Flatten results
                for batch in batched_results:
                    for idx, params in batch:
                        self.scaler_params[idx] = params
            else:
                # Sequential processing
                print("Fitting scalers sequentially...")
                for window_idx, window in enumerate(self.windows):
                    df = self.series_data[window['entity_id']]
                    params = self._fit_scalers_for_window_optimized(window, df)

                    if self.scaling_method == 'mean':
                        self.scaler_params[window_idx] = params
                    else:  # standard
                        self.scaler_params[window_idx] = params
        
        print("Scaler fitting complete")
        
    def _compute_entity_padding_values(self):
        """
        Compute appropriate padding values for each entity and feature.
        Call this AFTER loading data but BEFORE preprocessing.
        """
        print("Computing padding values for each entity...")

        for entity_id, df in self.series_data.items():
            padding_values = {}

            for col in self.numeric_cols:
                if col not in df.columns:
                    continue

                values = df[col].dropna().values

                if len(values) == 0:
                    # No valid data, default to 0
                    padding_val = 0.0
                else:
                    if self.padding_strategy == 'mean':
                        padding_val = np.mean(values)

                    elif self.padding_strategy == 'forward_fill':
                        # Use first available value (since we pad left)
                        padding_val = values[0]

                    elif self.padding_strategy == 'intelligent':
                        # Feature-specific logic - some examples given, customize as needed
                        if col in self.target_cols:
                            # For targets like sales, use mean
                            padding_val = np.mean(values)

                        elif 'price' in col.lower() or 'cost' in col.lower():
                            # For prices, use mean
                            padding_val = np.mean(values)

                        elif 'temperature' in col.lower():
                            # For temperature, use seasonal average if possible
                            padding_val = np.mean(values)

                        elif values.min() >= 0 and values.max() <= 1:
                            # Likely probability or percentage, use mean
                            padding_val = np.mean(values)

                        elif np.all(np.isin(values[~np.isnan(values)], [0, 1])):
                            # Binary feature, use 0.5 for neutral
                            padding_val = 0.5

                        else:
                            # Default to mean
                            padding_val = np.mean(values)

                    else:  # 'zero' strategy (not recommended)
                        padding_val = 0.0

                padding_values[col] = padding_val

            self.entity_padding_values[entity_id] = padding_values

        print(f"Computed padding values for {len(self.entity_padding_values)} entities")
    
    def _fit_scalers_for_window_optimized(self, window: Dict, df: pd.DataFrame) -> np.ndarray:
        """
        Fit scalers for a window and return just the parameters.
        
        Returns:
            numpy array of scaler parameters
        """
        historical_end = window['historical_end']
        n_features = len(self.numeric_cols)
        
        if self.scaling_method == 'mean':
            params = np.ones(n_features, dtype=np.float32) * self.mean_scaler_epsilon
            
            for col in self.numeric_cols:
                if col in df.columns:
                    col_idx = self.feature_to_idx[col]
                    
                    # Determine data range based on feature type
                    if col in self.all_unknown_features:
                        # Use only historical data for unknown features
                        data = df.iloc[:historical_end][col].dropna().values
                    else:
                        # For known features, could use more, but use historical for consistency
                        data = df.iloc[:historical_end][col].dropna().values
                    
                    if len(data) > 0:
                        # Mean scaling: scale = mean(abs(data)) + epsilon
                        scale = np.mean(np.abs(data)) + self.mean_scaler_epsilon
                        params[col_idx] = scale
            
            return params
        
        elif self.scaling_method == 'standard':
            params = np.zeros((n_features, 2), dtype=np.float32)
            # Initialize with default values (mean=0, std=1)
            params[:, 1] = 1.0
            
            for col in self.numeric_cols:
                if col in df.columns:
                    col_idx = self.feature_to_idx[col]
                    
                    if col in self.all_unknown_features:
                        data = df.iloc[:historical_end][col].dropna().values
                    else:
                        data = df.iloc[:historical_end][col].dropna().values
                    
                    if len(data) > 0:
                        mean = np.mean(data)
                        std = np.std(data, ddof=1)
                        if std < 1e-7:  # Avoid division by zero
                            std = 1.0
                        params[col_idx, 0] = mean
                        params[col_idx, 1] = std
            
            return params
        
    def _fit_entity_level_scalers_fast(self):
        """
        Fixed entity-level scaling that avoids passing large data structures to workers.
        Modified to support importing scalers for val/test mode.
        """
        n_features = len(self.numeric_cols)
        n_entities = len(self.entity_scaling_windows)

        print(f"Fitting entity-level scalers for {n_entities} entities...")

        # NEW: Check for imported scalers
        imported_entity_scalers = {}
        entities_to_fit = set(self.entity_scaling_windows.keys())
        entities_using_imported = set()

        if self.imported_scalers and self.mode in ['val', 'test']:
            imported_entity_scalers = self.imported_scalers.get('entity_scalers', {})
            # Identify which entities we have scalers for
            entities_using_imported = entities_to_fit & set(imported_entity_scalers.keys())
            entities_to_fit = entities_to_fit - entities_using_imported

            if entities_using_imported:
                print(f"  Reusing scalers for {len(entities_using_imported)} entities from training")
            if entities_to_fit:
                print(f"  Fitting fresh scalers for {len(entities_to_fit)} new entities")

        # FIXED: Assign imported scalers to ALL windows of those entities, not just non-overlapping ones
        if entities_using_imported:
            # Create a mapping of entity_id to ALL its window indices
            entity_to_all_windows = {}
            for window_idx, window in enumerate(self.windows):
                entity_id = window['entity_id']
                if entity_id not in entity_to_all_windows:
                    entity_to_all_windows[entity_id] = []
                entity_to_all_windows[entity_id].append(window_idx)

            # Now assign imported scalers to ALL windows of imported entities
            for entity_id in entities_using_imported:
                if entity_id in entity_to_all_windows:
                    imported_scaler = imported_entity_scalers[entity_id]
                    for window_idx in entity_to_all_windows[entity_id]:
                        self.scaler_params[window_idx] = imported_scaler.copy()
                    #print(f"    Assigned scaler to {len(entity_to_all_windows[entity_id])} windows for entity {entity_id}")

        # If all entities used imported scalers, we're done
        if not entities_to_fit:
            print(f"Entity-level scaling complete (all from imports)")
            return

        # For remaining entities, use the original optimized code
        # Extract only the necessary data for entities that need fitting
        entity_data_for_scaling = {}

        print(f"  Extracting historical data for scaling...")
        for entity_id in entities_to_fit:  # Only process entities needing fresh scalers
            df = self.series_data[entity_id]
            window_indices = self.entity_scaling_windows[entity_id]

            # Collect just the numeric values we need (as numpy arrays, not DataFrames)
            historical_values = {col: [] for col in self.numeric_cols if col in df.columns}

            for idx in window_indices:
                window = self.windows[idx]

                if window['padding_steps'] > 0:
                    hist_slice = slice(None, window['historical_end'])
                else:
                    hist_slice = slice(window['start_idx'], window['historical_end'])

                for col in self.numeric_cols:
                    if col in df.columns:
                        values = df.iloc[hist_slice][col].values
                        historical_values[col].append(values)

            # Concatenate all historical values for this entity
            concatenated_values = {}
            for col, values_list in historical_values.items():
                if values_list:
                    concatenated_values[col] = np.concatenate(values_list)

            entity_data_for_scaling[entity_id] = concatenated_values

        # Now fit scalers using the extracted numpy arrays (original parallel code)
        def fit_entity_scaler_from_arrays(entity_id, historical_values, 
                                         numeric_cols, feature_to_idx,
                                         scaling_method, mean_scaler_epsilon):
            """Fit scaler from pre-extracted numpy arrays."""
            n_features = len(numeric_cols)

            if scaling_method == 'mean':
                params = np.ones(n_features, dtype=np.float32) * mean_scaler_epsilon

                for col in numeric_cols:
                    if col in historical_values:
                        col_idx = feature_to_idx[col]
                        data = historical_values[col]
                        # Remove NaN values
                        data = data[~np.isnan(data)]
                        if len(data) > 0:
                            params[col_idx] = np.mean(np.abs(data)) + mean_scaler_epsilon

                return entity_id, params

            elif scaling_method == 'standard':
                params = np.zeros((n_features, 2), dtype=np.float32)
                params[:, 1] = 1.0

                for col in numeric_cols:
                    if col in historical_values:
                        col_idx = feature_to_idx[col]
                        data = historical_values[col]
                        # Remove NaN values
                        data = data[~np.isnan(data)]
                        if len(data) > 0:
                            params[col_idx, 0] = np.mean(data)
                            params[col_idx, 1] = np.std(data, ddof=1)
                            if params[col_idx, 1] < 1e-7:
                                params[col_idx, 1] = 1.0

                return entity_id, params

        # Process with lightweight numpy arrays instead of DataFrames
        entity_items = list(entity_data_for_scaling.items())

        if self.n_jobs != 1 and len(entity_items) > 10:
            backend = get_optimal_backend()
            print(f"  Fitting scalers in parallel using {backend} backend...")

            # Now we're only passing numpy arrays, not DataFrames
            results = Parallel(n_jobs=self.n_jobs, backend=backend)(
                delayed(fit_entity_scaler_from_arrays)(
                    entity_id,
                    historical_values,
                    self.numeric_cols,
                    self.feature_to_idx,
                    self.scaling_method,
                    self.mean_scaler_epsilon
                )
                for entity_id, historical_values in entity_items
            )
        else:
            print(f"  Fitting scalers sequentially...")
            results = []
            for entity_id, historical_values in entity_items:
                result = fit_entity_scaler_from_arrays(
                    entity_id,
                    historical_values,
                    self.numeric_cols,
                    self.feature_to_idx,
                    self.scaling_method,
                    self.mean_scaler_epsilon
                )
                results.append(result)

        # Create entity scalers dictionary
        entity_scalers = dict(results)

        # FIXED: Assign scalers to ALL windows of each entity, not just non-overlapping ones
        print(f"  Assigning scalers to windows...")
        for window_idx, window in enumerate(self.windows):
            entity_id = window['entity_id']
            if entity_id in entity_scalers:
                self.scaler_params[window_idx] = entity_scalers[entity_id]

        print(f"Entity-level scaling complete")
        if entities_using_imported:
            print(f"  - Reused: {len(entities_using_imported)} entities")
        if entities_to_fit:
            print(f"  - Fresh: {len(entities_to_fit)} entities")
            
        # Scaler assignment for ultra short series which won't have a proper scale
        # After fitting scalers for normal entities, handle ultra-short
        if self.mode in ['test', 'val'] and self.allow_inference_only_entities:
            # Identify ultra-short entities
            ultra_short_entities = []
            for window_idx, window in enumerate(self.windows):
                if window.get('is_ultra_short', False):
                    ultra_short_entities.append((window_idx, window['entity_id']))
            
            print(f"no. of ultra short entities: {len(ultra_short_entities)}")
            if ultra_short_entities:
                # Use category-based scalers if static features available
                self._assign_category_scalers_to_ultra_short(ultra_short_entities)
    
    def _assign_category_scalers_to_ultra_short(self, ultra_short_entities):
        """
        Assign category-specific scalers based on user-defined static features.
        """
        
        # Determine which columns to use for categorization
        if self.cold_start_scaler_cols:
            category_cols = self.cold_start_scaler_cols
        elif self.static_categorical_cols:
            category_cols = self.static_categorical_cols
        else:
            print("No categorical columns available for category-based scaling. Using global strategy.")
            self._assign_global_scalers_to_ultra_short(ultra_short_entities)
            return

        print(f"Using columns for category-based scaler matching: {category_cols}")

        # Build scaler library from normal entities
        category_scalers = defaultdict(list)

        for window_idx, window in enumerate(self.windows):
            if not window.get('is_ultra_short', False):
                entity_id = window['entity_id']

                # Get static data from original DataFrame
                entity_df = self.series_data.get(entity_id)
                if entity_df is None:
                    continue

                # Extract static categorical values
                category_values = []
                for col in category_cols:
                    if col in entity_df.columns:
                        # Static features should be constant, so take first value
                        val = entity_df[col].iloc[0]
                        # Encode if we have encoder
                        if col in self.categorical_encoders:
                            encoder = self.categorical_encoders[col]
                            if pd.notna(val) and val in encoder.classes_:
                                encoded_val = encoder.transform([val])[0]
                            else:
                                encoded_val = 'unknown'
                        else:
                            encoded_val = val if pd.notna(val) else 'unknown'
                        category_values.append(encoded_val)
                    else:
                        category_values.append('unknown')

                category_key = tuple(category_values)
                category_scalers[category_key].append(self.scaler_params[window_idx])

        # Calculate average scaler per category
        avg_category_scalers = {}
        for category, scalers_list in category_scalers.items():
            if scalers_list:
                avg_category_scalers[category] = np.mean(np.array(scalers_list), axis=0)

        # Create fallback scaler
        if avg_category_scalers:
            fallback_scaler = np.mean(list(avg_category_scalers.values()), axis=0)
        else:
            # No normal entities to learn from - use defaults
            if self.scaling_method == 'mean':
                fallback_scaler = np.ones(len(self.numeric_cols)) * self.mean_scaler_epsilon
            elif self.scaling_method == 'standard':
                fallback_scaler = np.zeros((len(self.numeric_cols), 2))
                fallback_scaler[:, 0] = 0.0  # mean = 0
                fallback_scaler[:, 1] = 1.0  # std = 1
        
        # Track assignment statistics
        assignment_stats = defaultdict(int)

        # Assign scalers to ultra-short entities
        for window_idx, entity_id in ultra_short_entities:
            # Get static data from original DataFrame
            entity_df = self.series_data.get(entity_id)

            if entity_df is not None:
                # Extract category values for this entity
                category_values = []
                for col in category_cols:
                    if col in entity_df.columns:
                        val = entity_df[col].iloc[0] if len(entity_df) > 0 else None
                        # Encode if we have encoder
                        if col in self.categorical_encoders:
                            encoder = self.categorical_encoders[col]
                            if pd.notna(val) and val in encoder.classes_:
                                encoded_val = encoder.transform([val])[0]
                            else:
                                encoded_val = 'unknown'
                        else:
                            encoded_val = val if pd.notna(val) else 'unknown'
                        category_values.append(encoded_val)
                    else:
                        category_values.append('unknown')

                category_key = tuple(category_values)
            else:
                category_key = tuple(['unknown'] * len(category_cols))

            if category_key in avg_category_scalers:
                self.scaler_params[window_idx] = avg_category_scalers[category_key].copy()
                assignment_stats['exact_match'] += 1
            else:
                # Try partial match if configured for multiple columns
                if len(category_cols) > 1:
                    # Simplified partial matching - just use first column
                    first_col_key = (category_key[0],)

                    # Find all scalers that match on first column
                    matching_scalers = []
                    for full_key, scaler in avg_category_scalers.items():
                        if full_key[0] == category_key[0]:
                            matching_scalers.append(scaler)

                    if matching_scalers:
                        self.scaler_params[window_idx] = np.mean(np.array(matching_scalers), axis=0)
                        assignment_stats['partial_match'] += 1
                    else:
                        self.scaler_params[window_idx] = fallback_scaler.copy()
                        assignment_stats['fallback'] += 1
                else:
                    self.scaler_params[window_idx] = fallback_scaler.copy()
                    assignment_stats['fallback'] += 1

        # Report statistics
        print(f"Ultra-short entity scaler assignment statistics:")
        for key, count in sorted(assignment_stats.items()):
            print(f"  {key}: {count}")
    
    def _assign_global_scalers_to_ultra_short(self, ultra_short_entities):
        """Assign global average scalers to ultra-short entities."""

        # Compute global scalers from all normal entities
        normal_scalers = []
        for window_idx, window in enumerate(self.windows):
            if not window.get('is_ultra_short', False):
                normal_scalers.append(self.scaler_params[window_idx])

        if normal_scalers:
            # Calculate mean scaler across all normal entities
            global_scaler = np.mean(normal_scalers, axis=0)

            # Assign to ultra-short entities
            for window_idx, entity_id in ultra_short_entities:
                self.scaler_params[window_idx] = global_scaler.copy()
                #print(f"Assigned global scaler to ultra-short entity {entity_id}")
    
    def _assign_category_scalers_to_ultra_short_orig(self, ultra_short_entities):
        """Assign category-specific scalers based on static features."""

        # Group entities by static categorical features
        category_scalers = defaultdict(list)

        # Build category mappings for normal entities
        for window_idx, window in enumerate(self.windows):
            if not window.get('is_ultra_short', False):
                entity_id = window['entity_id']
                # Get static data from original DataFrame
                entity_df = self.series_data.get(entity_id)
                
                if entity_df is None:
                    continue

                # Extract static categorical values
                category_values = []
                for col in self.cold_start_scaler_cols:
                    if col in entity_df.columns:
                        # Static features should be constant, so take first value
                        val = entity_df[col].iloc[0]
                        # Encode if we have encoder
                        if col in self.categorical_encoders:
                            encoder = self.categorical_encoders[col]
                            if pd.notna(val) and val in encoder.classes_:
                                encoded_val = encoder.transform([val])[0]
                            else:
                                encoded_val = 'unknown'
                        else:
                            encoded_val = val if pd.notna(val) else 'unknown'
                        category_values.append(encoded_val)
                    else:
                        category_values.append('unknown')

                category_key = tuple(category_values)

                category_scalers[category_key].append(self.scaler_params[window_idx])

        # Calculate average scaler per category
        avg_category_scalers = {}
        for category, scalers_list in category_scalers.items():
            if scalers_list:
                # Handle different scaling methods
                if self.scaling_method == 'mean':
                    # Mean scaling: 2D array [n_windows, n_features]
                    avg_category_scalers[category] = np.mean(scalers_list, axis=0)

                elif self.scaling_method == 'standard':
                    # Standard scaling: 3D array [n_windows, n_features, 2]
                    # Average mean and std separately
                    avg_category_scalers[category] = np.mean(scalers_list, axis=0)

        # Calculate fallback scaler (global average)
        if avg_category_scalers:
            fallback_scaler = np.mean(list(avg_category_scalers.values()), axis=0)
        else:
            # No normal entities to learn from - use defaults
            if self.scaling_method == 'mean':
                fallback_scaler = np.ones(len(self.numeric_cols)) * self.mean_scaler_epsilon
            elif self.scaling_method == 'standard':
                fallback_scaler = np.zeros((len(self.numeric_cols), 2))
                fallback_scaler[:, 0] = 0.0  # mean = 0
                fallback_scaler[:, 1] = 1.0  # std = 1

        # Assign to ultra-short entities based on their category
        for window_idx, entity_id in ultra_short_entities:
            # Get static data from original DataFrame
            entity_df = self.series_data.get(entity_id)
                
            if entity_df is None:
                continue
                
            # Extract static categorical values
            category_values = []
            for col in self.cold_start_scaler_cols:
                if col in entity_df.columns:
                    # Static features should be constant, so take first value
                    val = entity_df[col].iloc[0]
                    # Encode if we have encoder
                    if col in self.categorical_encoders:
                        encoder = self.categorical_encoders[col]
                        if pd.notna(val) and val in encoder.classes_:
                            encoded_val = encoder.transform([val])[0]
                        else:
                            encoded_val = 'unknown'
                    else:
                        encoded_val = val if pd.notna(val) else 'unknown'
                    category_values.append(encoded_val)
                else:
                    category_values.append('unknown')

            category_key = tuple(category_values)
            
            if category_key in avg_category_scalers:
                # Found matching category
                self.scaler_params[window_idx] = avg_category_scalers[category_key].copy()
                print(f"Assigned category {category_key} scaler to {entity_id}")
            else:
                # No matching category - use fallback
                self.scaler_params[window_idx] = fallback_scaler.copy()
                print(f"Assigned fallback scaler to {entity_id}")
            
    def export_entity_scalers(self) -> Optional[Dict]:
        """Export entity-level scalers for reuse."""
        if self.scaling_method == 'none' or self.scaling_strategy != 'entity_level':
            return None

        entity_scalers = {}
        for entity_id, window_indices in self.entity_scaling_windows.items():
            if window_indices:
                entity_scalers[entity_id] = self.scaler_params[window_indices[0]].copy()

        return {
            'scaling_method': self.scaling_method,
            'mean_scaler_epsilon': self.mean_scaler_epsilon,
            'feature_to_idx': self.feature_to_idx.copy(),
            'numeric_cols': self.numeric_cols.copy(),
            'entity_scalers': entity_scalers,
            'n_entities': len(entity_scalers)
        }

    def save_entity_scalers(self, filepath: str):
        """Save entity scalers to disk."""
        import pickle
        from pathlib import Path

        scalers_dict = self.export_entity_scalers()
        if scalers_dict:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(scalers_dict, f)
            print(f"Saved scalers for {scalers_dict['n_entities']} entities to {filepath}")

    @staticmethod
    def load_entity_scalers(filepath: str) -> Dict:
        """Load entity scalers from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _fit_and_save_encoders(self):
        """
        Fit encoders on training data and save them for future use.
        Only called during training mode.
        """
        print(f"Fitting encoders on {self.mode} data...")
        
        # Fit global encoders for all categorical features
        self.categorical_encoders = {}
        self.encoder_metadata = {}  # Store additional info about encoders
        
        all_categorical_cols = (
            self.static_categorical_cols + 
            self.temporal_known_categorical_cols + 
            self.temporal_unknown_categorical_cols
        )
        
        for col in all_categorical_cols:
            print(f"  Fitting encoder for {col}...")
            
            # Collect all unique values
            all_values = set()
            
            for entity_id, df in self.series_data.items():
                if col in df.columns:
                    if col in self.static_categorical_cols:
                        val = df[col].iloc[0]
                        if pd.notna(val):
                            all_values.add(val)
                    else:
                        unique_vals = df[col].dropna().unique()
                        all_values.update(unique_vals)
            
            if all_values:
                # Fit encoder with all unique values from training data
                encoder = LabelEncoder()
                encoder.fit(list(all_values))
                self.categorical_encoders[col] = encoder
                
                # Store metadata
                self.encoder_metadata[col] = {
                    'n_classes': len(encoder.classes_),
                    'fitted_on_mode': self.mode,
                    'fitted_on_date': pd.Timestamp.now().isoformat(),
                    'sample_values': encoder.classes_[:10].tolist()
                }
                
                print(f"    -> {len(encoder.classes_)} unique values")
        
        # Save encoders and metadata
        self._save_encoders()
        print(f"Encoders saved to {self.encoders_path}")
    
    def _save_encoders(self):
        """Save fitted encoders and metadata to disk."""
        self.encoders_path.mkdir(parents=True, exist_ok=True)
        
        # Save encoders as pickle
        encoders_file = self.encoders_path / 'label_encoders.pkl'
        with open(encoders_file, 'wb') as f:
            pickle.dump(self.categorical_encoders, f)
        
        # Save metadata as JSON for human readability
        metadata_file = self.encoders_path / 'encoder_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.encoder_metadata, f, indent=2)
        
        # Save feature configuration for validation
        config_file = self.encoders_path / 'feature_config.json'
        feature_config = {
            'static_categorical_cols': self.static_categorical_cols,
            'temporal_known_categorical_cols': self.temporal_known_categorical_cols,
            'temporal_unknown_categorical_cols': self.temporal_unknown_categorical_cols,
        }
        with open(config_file, 'w') as f:
            json.dump(feature_config, f, indent=2)
    
    def _load_encoders(self):
        """Load pre-fitted encoders for validation/test/inference."""
        encoders_file = self.encoders_path / 'label_encoders.pkl'
        metadata_file = self.encoders_path / 'encoder_metadata.json'
        config_file = self.encoders_path / 'feature_config.json'
        
        if not encoders_file.exists():
            raise FileNotFoundError(
                f"No encoders found at {encoders_file}. "
                f"Please train the model first or provide correct encoders_path."
            )
        
        print(f"Loading encoders from {self.encoders_path}...")
        
        # Load encoders
        with open(encoders_file, 'rb') as f:
            self.categorical_encoders = pickle.load(f)
        
        # Load metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.encoder_metadata = json.load(f)
        
        # Validate feature configuration
        if config_file.exists():
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            
            # Check for consistency
            self._validate_feature_config(saved_config)
        
        print(f"Loaded {len(self.categorical_encoders)} encoders")
        
        # Report unknown categories in current data
        self._report_unknown_categories()
    
    def _validate_feature_config(self, saved_config):
        """Validate that current feature config matches saved config."""
        current_config = {
            'static_categorical_cols': self.static_categorical_cols,
            'temporal_known_categorical_cols': self.temporal_known_categorical_cols,
            'temporal_unknown_categorical_cols': self.temporal_unknown_categorical_cols,
        }
        
        mismatches = []
        for key in current_config:
            if set(current_config[key]) != set(saved_config.get(key, [])):
                mismatches.append(key)
        
        if mismatches:
            print(f"WARNING: Feature configuration mismatch for: {mismatches}")
            print("This may lead to encoding errors. Please verify your feature config.")
    
    def _report_unknown_categories(self):
        """Report categories in current data that weren't seen during training."""
        print("\nChecking for unknown categories...")
        
        unknown_stats = {}
        
        for col in self.categorical_encoders:
            encoder = self.categorical_encoders[col]
            known_classes = set(encoder.classes_)
            unknown_values = set()
            
            for entity_id, df in self.series_data.items():
                if col in df.columns:
                    if col in self.static_categorical_cols:
                        val = df[col].iloc[0]
                        if pd.notna(val) and val not in known_classes:
                            unknown_values.add(val)
                    else:
                        unique_vals = df[col].dropna().unique()
                        for val in unique_vals:
                            if val not in known_classes:
                                unknown_values.add(val)
            
            if unknown_values:
                unknown_stats[col] = {
                    'count': len(unknown_values),
                    'samples': list(unknown_values)[:5]
                }
        
        if unknown_stats:
            print("  Found unknown categories (will be encoded as -1):")
            for col, stats in unknown_stats.items():
                print(f"    {col}: {stats['count']} unknown values")
                print(f"      Examples: {stats['samples']}")
        else:
            print("  No unknown categories found")
    
    def _safe_encode_categorical(self, values, col_name, encoder):
        """
        Safely encode categorical values, handling unknown categories.
        
        Args:
            values: Array of values to encode
            col_name: Name of the column (for logging)
            encoder: Fitted LabelEncoder
            
        Returns:
            Encoded values with -1 for unknown categories
        """
        encoded = np.full(len(values), -1, dtype=np.int32)
        
        for i, val in enumerate(values):
            if pd.notna(val):
                if val in encoder.classes_:
                    encoded[i] = encoder.transform([val])[0]
                else:
                    # Unknown category - keep as -1
                    # Could optionally log this for monitoring
                    encoded[i] = -1
        
        return encoded
    
    def _preprocess_and_cache_data(self):
        """
        Parallelized preprocessing that safely handles unknown categories.
        Maintains exact functional equivalence with original version.
        """
        print("Preprocessing and caching data for fast access...")

        # Initialize storage (same as original)
        self.cached_series_data = {}
        self.cached_categorical_data = {}
        self.cached_static_data = {}

        # Track unknown categories for monitoring
        self.unknown_category_counts = {}

        # Get entities list - maintains original order
        entities_list = list(self.series_data.keys())

        # Determine if we should parallelize
        should_parallelize = self.n_jobs != 1 and len(entities_list) > 50

        if should_parallelize:
            backend = get_optimal_backend()
            print(f"Using parallel processing with '{backend}' backend...")

            # Create a standalone function that doesn't rely on self
            def process_single_entity(entity_id, df, numeric_cols, static_numeric_cols, 
                                     static_categorical_cols, temporal_known_categorical_cols,
                                     temporal_unknown_categorical_cols, categorical_encoders):
                """Standalone function for parallel processing."""

                # Process numeric columns
                numeric_data = {}
                for col in numeric_cols:
                    if col in df.columns:
                        numeric_data[col] = df[col].fillna(0).values.astype(np.float32)

                # Process temporal categorical columns
                categorical_data = {}
                unknown_counts_local = {}
                temporal_categorical_cols = (
                    temporal_known_categorical_cols + 
                    temporal_unknown_categorical_cols
                )

                for col in temporal_categorical_cols:
                    if col in df.columns and col in categorical_encoders:
                        encoder = categorical_encoders[col]
                        values = df[col].values

                        # Inline safe encoding (replaces self._safe_encode_categorical)
                        encoded = np.full(len(values), -1, dtype=np.int32)
                        for i, val in enumerate(values):
                            if pd.notna(val) and val in encoder.classes_:
                                encoded[i] = encoder.transform([val])[0]

                        categorical_data[col] = encoded

                        # Track unknowns
                        n_unknowns = np.sum(encoded == -1)
                        if n_unknowns > 0:
                            unknown_counts_local[col] = n_unknowns

                # Process static features
                static_data = {}

                # Static numeric
                for col in static_numeric_cols:
                    if col in df.columns:
                        val = df[col].iloc[0]
                        static_data[col] = float(val) if pd.notna(val) else 0.0

                # Static categorical
                for col in static_categorical_cols:
                    if col in df.columns and col in categorical_encoders:
                        val = df[col].iloc[0]
                        encoder = categorical_encoders[col]

                        if pd.notna(val) and val in encoder.classes_:
                            static_data[col] = int(encoder.transform([val])[0])
                        else:
                            static_data[col] = -1
                            if col not in unknown_counts_local:
                                unknown_counts_local[col] = 0
                            unknown_counts_local[col] += 1

                return entity_id, numeric_data, categorical_data, static_data, unknown_counts_local

            # Run parallel processing
            results = Parallel(n_jobs=self.n_jobs, backend=backend)(
                delayed(process_single_entity)(
                    entity_id, 
                    self.series_data[entity_id],
                    self.numeric_cols,
                    self.static_numeric_cols,
                    self.static_categorical_cols,
                    self.temporal_known_categorical_cols,
                    self.temporal_unknown_categorical_cols,
                    self.categorical_encoders
                ) for entity_id in entities_list
            )

            # Unpack results IN ORDER
            for entity_id, numeric_data, categorical_data, static_data, unknown_counts_local in results:
                self.cached_series_data[entity_id] = numeric_data
                self.cached_categorical_data[entity_id] = categorical_data
                self.cached_static_data[entity_id] = static_data

                # Aggregate unknown counts
                for col, count in unknown_counts_local.items():
                    if col not in self.unknown_category_counts:
                        self.unknown_category_counts[col] = 0
                    self.unknown_category_counts[col] += count

        else:
            # Sequential processing (original logic preserved exactly)
            print("Using sequential processing...")

            for entity_id in entities_list:
                df = self.series_data[entity_id]

                # Process numeric columns
                numeric_data = {}
                for col in self.numeric_cols:
                    if col in df.columns:
                        numeric_data[col] = df[col].fillna(0).values.astype(np.float32)

                # Process temporal categorical columns
                categorical_data = {}
                temporal_categorical_cols = (
                    self.temporal_known_categorical_cols + 
                    self.temporal_unknown_categorical_cols
                )

                for col in temporal_categorical_cols:
                    if col in df.columns and col in self.categorical_encoders:
                        encoder = self.categorical_encoders[col]
                        values = df[col].values
                        encoded = self._safe_encode_categorical(values, col, encoder)
                        categorical_data[col] = encoded

                        # Track unknowns
                        n_unknowns = np.sum(encoded == -1)
                        if n_unknowns > 0:
                            if col not in self.unknown_category_counts:
                                self.unknown_category_counts[col] = 0
                            self.unknown_category_counts[col] += n_unknowns

                # Process static features
                static_data = {}

                # Static numeric
                for col in self.static_numeric_cols:
                    if col in df.columns:
                        val = df[col].iloc[0]
                        static_data[col] = float(val) if pd.notna(val) else 0.0

                # Static categorical
                for col in self.static_categorical_cols:
                    if col in df.columns and col in self.categorical_encoders:
                        val = df[col].iloc[0]
                        encoder = self.categorical_encoders[col]

                        if pd.notna(val) and val in encoder.classes_:
                            static_data[col] = int(encoder.transform([val])[0])
                        else:
                            static_data[col] = -1

                            # Track unknown
                            if col not in self.unknown_category_counts:
                                self.unknown_category_counts[col] = 0
                            self.unknown_category_counts[col] += 1

                self.cached_series_data[entity_id] = numeric_data
                self.cached_categorical_data[entity_id] = categorical_data
                self.cached_static_data[entity_id] = static_data

        # Report unknown category statistics (same as original)
        if self.unknown_category_counts:
            print("\nUnknown category statistics:")
            for col, count in self.unknown_category_counts.items():
                print(f"  {col}: {count} unknown values encoded as -1")

        self._print_encoding_summary()
        print("Data preprocessing complete!")

    
    def get_encoder_mappings(self):
        """
        Get the mapping of original values to encoded values for all categorical features.
        Useful for debugging and understanding the encodings.
        """
        mappings = {}
        
        if hasattr(self, 'categorical_encoders'):
            for col, encoder in self.categorical_encoders.items():
                mappings[col] = {
                    'classes': encoder.classes_.tolist(),
                    'mapping': {val: i for i, val in enumerate(encoder.classes_)}
                }
        
        return mappings
    
    def inverse_transform_categorical(self, encoded_values, feature_name):
        """
        Convert encoded categorical values back to original values.
        
        Args:
            encoded_values: Encoded integer values (tensor or numpy array)
            feature_name: Name of the categorical feature
            
        Returns:
            Original categorical values
        """
        if feature_name not in self.categorical_encoders:
            raise ValueError(f"No encoder found for feature '{feature_name}'")
        
        encoder = self.categorical_encoders[feature_name]
        
        # Handle tensor input
        if torch.is_tensor(encoded_values):
            encoded_values = encoded_values.cpu().numpy()
        
        # Flatten if needed
        original_shape = encoded_values.shape
        encoded_flat = encoded_values.flatten()
        
        # Inverse transform
        original_values = []
        for val in encoded_flat:
            if val == -1:
                original_values.append(None)  # Unknown category
            else:
                try:
                    original_values.append(encoder.inverse_transform([val])[0])
                except:
                    original_values.append(None)
        
        # Reshape back
        return np.array(original_values).reshape(original_shape)
    
    def _print_encoding_summary(self):
        """Print summary of encodings for debugging."""
        if not self.cached_static_data:
            return
            
        print("\n=== Encoding Summary ===")
        
        # Check static categorical encodings
        if self.static_categorical_cols:
            print("Static Categorical Encodings (first 3 entities):")
            for i, (entity_id, static_data) in enumerate(list(self.cached_static_data.items())[:3]):
                values = []
                for col in self.static_categorical_cols:
                    if col in static_data:
                        values.append(f"{col}={static_data[col]}")
                print(f"  Entity {entity_id}: {', '.join(values)}")
        
        # Check temporal categorical encodings
        temporal_cats = self.temporal_known_categorical_cols + self.temporal_unknown_categorical_cols
        if temporal_cats and self.cached_categorical_data:
            print("\nTemporal Categorical Encodings (first entity, first 10 timesteps):")
            entity_id = list(self.cached_categorical_data.keys())[0]
            cat_data = self.cached_categorical_data[entity_id]
            
            for col in temporal_cats[:3]:  # Show first 3 temporal categorical features
                if col in cat_data:
                    values = cat_data[col][:10]  # First 10 timesteps
                    print(f"  {col}: {values}")
    
    def inspect_padding_values(self, n_entities=3):
        """
        Debug method to inspect computed padding values.
        """
        print("\n=== Padding Values Inspection ===")
        for i, (entity_id, padding_vals) in enumerate(self.entity_padding_values.items()):
            if i >= n_entities:
                break
            print(f"\nEntity: {entity_id}")
            for feature, value in list(padding_vals.items())[:5]:  # Show first 5 features
                print(f"  {feature}: {value:.4f}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Optimized version of __getitem__ that uses cached numpy arrays.
        """
        window = self.windows[idx]
        entity_id = window['entity_id']
        
        # Get entity-specific padding values
        entity_padding = self.entity_padding_values.get(entity_id, {})
        
        # Use cached numpy arrays instead of DataFrame
        numeric_data = self.cached_series_data[entity_id]
        categorical_data = self.cached_categorical_data[entity_id]
        static_data = self.cached_static_data[entity_id]
        
        # Get window parameters
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        historical_end = window['historical_end']
        padding_steps = window['padding_steps']
        
        # Build sample dictionary
        sample = {
            'entity_id': entity_id,
            'window_idx': idx,
            'mask': torch.ones(self.total_steps, dtype=torch.float32)
        }
        
        # Entity weight
        if self.weight_col:
            df = self.series_data[entity_id]
            if self.weight_col in df.columns:
                # Entity weight should be constant per entity
                entity_weight = df[self.weight_col].iloc[0]
                sample['entity_weight'] = torch.tensor(entity_weight, dtype=torch.float32)
            else:
                sample['entity_weight'] = torch.tensor(1.0, dtype=torch.float32)
            
        # Calculate recency weight for this window
        if self.recency_alpha > 0:
            # Get the relative position of this window's end point
            entity_length = len(self.series_data[entity_id])
            window_end_position = window['end_idx']
            relative_position = window_end_position / entity_length  # 0 to 1

            # Exponential weighting: more recent windows get higher weight
            recency_weight = np.exp(self.recency_alpha * relative_position)
            sample['recency_weight'] = torch.tensor(recency_weight, dtype=torch.float32)
        else:
            # No recency weighting
            sample['recency_weight'] = torch.tensor(1.0, dtype=torch.float32)
                
        # Apply scaling using vectorized operations
        scaled_numeric_data = {}
        if self.scaling_method != 'none' and self.scaler_params is not None:
            if self.scaling_method == 'mean':
                scales = self.scaler_params[idx]
                for col, data in numeric_data.items():
                    if col in self.feature_to_idx:
                        col_idx = self.feature_to_idx[col]
                        scale = scales[col_idx]
                        if scale > 0:
                            # Scale the entire array at once
                            scaled_numeric_data[col] = data / scale
                        else:
                            scaled_numeric_data[col] = data
            elif self.scaling_method == 'standard':
                params = self.scaler_params[idx]
                for col, data in numeric_data.items():
                    if col in self.feature_to_idx:
                        col_idx = self.feature_to_idx[col]
                        mean = params[col_idx, 0]
                        std = params[col_idx, 1]
                        if std > 0:
                            scaled_numeric_data[col] = (data - mean) / std
                        else:
                            scaled_numeric_data[col] = data
        else:
            scaled_numeric_data = numeric_data
            
        # Extract and store target scaler parameters
        if self.scaling_method != 'none' and self.scaler_params is not None and self.target_cols:
            n_targets = len(self.target_cols)

            if self.scaling_method == 'mean':
                # For mean scaling: store scale values
                target_scaler_params = np.zeros(n_targets, dtype=np.float32)
                for i, col in enumerate(self.target_cols):
                    if col in self.feature_to_idx:
                        col_idx = self.feature_to_idx[col]
                        target_scaler_params[i] = self.scaler_params[idx, col_idx]
                sample['target_scale'] = torch.tensor(target_scaler_params, dtype=torch.float32)

            elif self.scaling_method == 'standard':
                # For standard scaling: store mean and std
                target_scaler_params = np.zeros((n_targets, 2), dtype=np.float32)
                for i, col in enumerate(self.target_cols):
                    if col in self.feature_to_idx:
                        col_idx = self.feature_to_idx[col]
                        target_scaler_params[i, 0] = self.scaler_params[idx, col_idx, 0]  # mean
                        target_scaler_params[i, 1] = self.scaler_params[idx, col_idx, 1]  # std
                sample['target_mean'] = torch.tensor(target_scaler_params[:, 0], dtype=torch.float32)
                sample['target_std'] = torch.tensor(target_scaler_params[:, 1], dtype=torch.float32)
       
        # Static features (already cached)
        if self.static_numeric_cols:
            static_numeric = []
            for col in self.static_numeric_cols:
                if col in static_data:
                    value = static_data[col]
                    static_numeric.append(float(value) if not pd.isna(value) else 0.0)
            if static_numeric:
                sample['static_numeric'] = torch.tensor(static_numeric, dtype=torch.float32)
        
        if self.static_categorical_cols:
            static_categorical = []
            for col in self.static_categorical_cols:
                if col in static_data:
                    value = static_data[col]
                    static_categorical.append(int(value) if not pd.isna(value) else -1)
            if static_categorical:
                sample['static_categorical'] = torch.tensor(static_categorical, dtype=torch.long)
        
        
        # Check if Ultra-Short Series ( length < self.prediction_steps) & handle accordingly
       
        if window.get('is_ultra_short', False):
            # ULTRA-SHORT SERIES HANDLING
            future_data_len = window.get('future_data_len', window['end_idx'])
            future_padding_steps = self.prediction_steps - future_data_len

            # Create mask: historical all 0 (padded), future real data 1, future padding 0
            mask = torch.zeros(self.total_steps, dtype=torch.float32)
            mask[self.historical_steps:self.historical_steps + future_data_len] = 1.0
            sample['mask'] = mask

            # Helper function for numeric padding
            def pad_numeric_ultra_short(col_name, is_target=False):
                pad_value = entity_padding.get(col_name, 0.0)
                if is_target or col_name in self.all_unknown_features:
                    # For targets/unknowns: use scaled data
                    data_dict = scaled_numeric_data
                else:
                    # For known features: use scaled data
                    data_dict = scaled_numeric_data

                if col_name in data_dict and len(data_dict[col_name]) > 0:
                    return data_dict[col_name][:future_data_len], pad_value
                else:
                    return np.array([]), pad_value

            # Historical Features (all padded for ultra-short)
            # Historical targets (padded)
            if self.target_cols:
                historical_targets = []
                for col in self.target_cols:
                    pad_value = entity_padding.get(col, 0.0)
                    # Apply scaling to padding value
                    if self.scaling_method == 'mean' and col in self.feature_to_idx:
                        col_idx = self.feature_to_idx[col]
                        scale = self.scaler_params[idx, col_idx]
                        if scale > 0:
                            pad_value = pad_value / scale
                    elif self.scaling_method == 'standard' and col in self.feature_to_idx:
                        col_idx = self.feature_to_idx[col]
                        mean = self.scaler_params[idx, col_idx, 0]
                        std = self.scaler_params[idx, col_idx, 1]
                        if std > 0:
                            pad_value = (pad_value - mean) / std

                    hist_data = np.full(self.historical_steps, pad_value, dtype=np.float32)
                    historical_targets.append(hist_data)

                sample['historical_targets'] = torch.tensor(
                    np.stack(historical_targets, axis=1) if len(historical_targets) > 1
                    else historical_targets[0].reshape(-1, 1),
                    dtype=torch.float32
                )

            # Historical unknown numeric (padded)
            if self.temporal_unknown_numeric_cols:
                unknown_numeric = []
                for col in self.temporal_unknown_numeric_cols:
                    actual_data, pad_value = pad_numeric_ultra_short(col)
                    hist_data = np.full(self.historical_steps, pad_value, dtype=np.float32)
                    unknown_numeric.append(hist_data)

                sample['temporal_unknown_numeric'] = torch.tensor(
                    np.stack(unknown_numeric, axis=1), dtype=torch.float32
                )

            # Historical unknown categorical (padded)
            if self.temporal_unknown_categorical_cols:
                unknown_categorical = []
                for col in self.temporal_unknown_categorical_cols:
                    hist_data = np.full(self.historical_steps, self.categorical_padding_value, dtype=np.int32)
                    unknown_categorical.append(hist_data)

                sample['temporal_unknown_categorical'] = torch.tensor(
                    np.stack(unknown_categorical, axis=1), dtype=torch.long
                )

            # Known Features (historical padded + future with actual data)
            # Temporal known numeric
            if self.temporal_known_numeric_cols:
                known_numeric = []
                for col in self.temporal_known_numeric_cols:
                    actual_data, pad_value = pad_numeric_ultra_short(col)

                    # Historical part (padded)
                    hist_part = np.full(self.historical_steps, pad_value, dtype=np.float32)

                    # Future part (actual + padding)
                    if len(actual_data) > 0:
                        if future_padding_steps > 0:
                            # Use last actual value for padding
                            last_val = actual_data[-1] if len(actual_data) > 0 else pad_value
                            future_padding = np.full(future_padding_steps, last_val, dtype=np.float32)
                            future_part = np.concatenate([actual_data, future_padding])
                        else:
                            future_part = actual_data[:self.prediction_steps]
                    else:
                        future_part = np.full(self.prediction_steps, pad_value, dtype=np.float32)

                    full_data = np.concatenate([hist_part, future_part])
                    known_numeric.append(full_data)

                sample['temporal_known_numeric'] = torch.tensor(
                    np.stack(known_numeric, axis=1), dtype=torch.float32
                )

            # Temporal known categorical
            if self.temporal_known_categorical_cols:
                known_categorical = []
                for col in self.temporal_known_categorical_cols:
                    # Historical part (padded)
                    hist_part = np.full(self.historical_steps, self.categorical_padding_value, dtype=np.int32)

                    # Future part (actual + padding)
                    if col in categorical_data and len(categorical_data[col]) > 0:
                        actual_future = categorical_data[col][:future_data_len]
                        if future_padding_steps > 0:
                            future_padding = np.full(future_padding_steps, self.categorical_padding_value, dtype=np.int32)
                            future_part = np.concatenate([actual_future, future_padding])
                        else:
                            future_part = actual_future[:self.prediction_steps]
                    else:
                        future_part = np.full(self.prediction_steps, self.categorical_padding_value, dtype=np.int32)

                    full_data = np.concatenate([hist_part, future_part])
                    known_categorical.append(full_data)

                sample['temporal_known_categorical'] = torch.tensor(
                    np.stack(known_categorical, axis=1), dtype=torch.long
                )

            # Future Targets (actual data + padding)
            if self.target_cols:
                future_targets = []
                for col in self.target_cols:
                    actual_data, pad_value = pad_numeric_ultra_short(col, is_target=True)

                    if len(actual_data) > 0:
                        if future_padding_steps > 0:
                            # Use last actual value for padding
                            last_val = actual_data[-1] if len(actual_data) > 0 else pad_value
                            future_padding = np.full(future_padding_steps, last_val, dtype=np.float32)
                            future_data = np.concatenate([actual_data, future_padding])
                        else:
                            future_data = actual_data[:self.prediction_steps]
                    else:
                        future_data = np.full(self.prediction_steps, pad_value, dtype=np.float32)

                    future_targets.append(future_data)

                sample['future_targets'] = torch.tensor(
                    np.stack(future_targets, axis=1) if len(future_targets) > 1
                    else future_targets[0].reshape(-1, 1),
                    dtype=torch.float32
                )
            
        
        else:
            # Regular series handling with length >= self.prediction_steps
            # Get the window slices
            
            # Handle padding for mask
            if padding_steps > 0:
                sample['mask'][:padding_steps] = 0

            # Helper function for fast padding
            def pad_numeric(arr, target_length, feature_name):
                """Pad numeric array with feature-specific values."""
                if len(arr) >= target_length:
                    return arr

                # Get padding value for this specific feature
                pad_value = entity_padding.get(feature_name, 0.0)

                # Create left padding
                padding = np.full(target_length - len(arr), pad_value, dtype=arr.dtype)
                return np.concatenate([padding, arr])

            def pad_categorical(arr, target_length):
                """Pad categorical array with -1."""
                if len(arr) >= target_length:
                    return arr

                # Always use -1 for categorical padding
                padding = np.full(target_length - len(arr), self.categorical_padding_value, dtype=arr.dtype)
                return np.concatenate([padding, arr])
        
            if padding_steps > 0:
                hist_slice_start = 0
                hist_slice_end = historical_end
                future_slice_start = historical_end
                future_slice_end = end_idx
            else:
                hist_slice_start = start_idx
                hist_slice_end = historical_end
                future_slice_start = historical_end
                future_slice_end = end_idx

            # Temporal known numeric features
            if self.temporal_known_numeric_cols:
                known_numeric = []
                for col in self.temporal_known_numeric_cols:
                    if col in scaled_numeric_data:
                        full_data = scaled_numeric_data[col][start_idx:end_idx]
                        if padding_steps > 0:
                            full_data = pad_numeric(full_data, self.total_steps, col)
                        known_numeric.append(full_data)
                if known_numeric:
                    sample['temporal_known_numeric'] = torch.tensor(
                        np.stack(known_numeric, axis=1), dtype=torch.float32
                    )

            # Temporal known categorical features
            if self.temporal_known_categorical_cols:
                known_categorical = []
                for col in self.temporal_known_categorical_cols:
                    if col in categorical_data:
                        full_data = categorical_data[col][start_idx:end_idx]
                        if padding_steps > 0:
                            full_data = pad_categorical(full_data, self.total_steps)
                        known_categorical.append(full_data)
                if known_categorical:
                    sample['temporal_known_categorical'] = torch.tensor(
                        np.stack(known_categorical, axis=1), dtype=torch.long
                    )

            # Temporal unknown numeric features (historical only)
            if self.temporal_unknown_numeric_cols:
                unknown_numeric = []
                for col in self.temporal_unknown_numeric_cols:
                    if col in scaled_numeric_data:
                        hist_data = scaled_numeric_data[col][hist_slice_start:hist_slice_end]
                        if padding_steps > 0:
                            hist_data = pad_numeric(hist_data, self.historical_steps, col)
                        unknown_numeric.append(hist_data)
                if unknown_numeric:
                    sample['temporal_unknown_numeric'] = torch.tensor(
                        np.stack(unknown_numeric, axis=1), dtype=torch.float32
                    )

            # Temporal unknown categorical features (historical only)
            if self.temporal_unknown_categorical_cols:
                unknown_categorical = []
                for col in self.temporal_unknown_categorical_cols:
                    if col in categorical_data:
                        hist_data = categorical_data[col][hist_slice_start:hist_slice_end]
                        if padding_steps > 0:
                            hist_data = pad_categorical(hist_data, self.historical_steps, col)
                        unknown_categorical.append(hist_data)
                if unknown_categorical:
                    sample['temporal_unknown_categorical'] = torch.tensor(
                        np.stack(unknown_categorical, axis=1), dtype=torch.long
                    )

            # Targets
            if self.target_cols:
                # Historical targets
                historical_targets = []
                for col in self.target_cols:
                    if col in scaled_numeric_data:
                        hist_data = scaled_numeric_data[col][hist_slice_start:hist_slice_end]
                        if padding_steps > 0:
                            hist_data = pad_numeric(hist_data, self.historical_steps, col)
                        historical_targets.append(hist_data)

                if historical_targets:
                    if len(historical_targets) > 1:
                        sample['historical_targets'] = torch.tensor(
                            np.stack(historical_targets, axis=1), dtype=torch.float32
                        )
                    else:
                        sample['historical_targets'] = torch.tensor(
                            historical_targets[0].reshape(-1, 1), dtype=torch.float32
                        )

                # Future targets
                future_targets = []
                for col in self.target_cols:
                    if col in scaled_numeric_data:
                        future_data = scaled_numeric_data[col][future_slice_start:future_slice_end]
                        future_targets.append(future_data)
                
                if future_targets:
                    if len(future_targets) > 1:
                        sample['future_targets'] = torch.tensor(
                            np.stack(future_targets, axis=1), dtype=torch.float32
                        )
                    else:
                        sample['future_targets'] = torch.tensor(
                            future_targets[0].reshape(-1, 1), dtype=torch.float32
                        )
        
        # Add Time index
        sample['time_index'] = torch.arange(self.total_steps)
        
        return sample
    
    def inverse_transform_predictions(self, predictions: torch.Tensor, 
                                     window_indices: List[int], 
                                     target_col: Optional[str] = None) -> torch.Tensor:
        """
        Inverse transform predictions using stored scaler parameters.
        
        Args:
            predictions: Scaled predictions [batch_size, prediction_steps, n_targets]
            window_indices: Window index for each batch item
            target_col: Target column name (uses first if not specified)
        
        Returns:
            Inverse transformed predictions
        """
        if self.scaling_method == 'none' or self.scaler_params is None:
            return predictions
        
        target_col = target_col or self.target_cols[0]
        col_idx = self.feature_to_idx.get(target_col)
        
        if col_idx is None:
            return predictions
        
        predictions_np = predictions.detach().cpu().numpy()
        inverse_transformed = np.zeros_like(predictions_np)
        
        for i, window_idx in enumerate(window_indices):
            if self.scaling_method == 'mean':
                scale = self.scaler_params[window_idx, col_idx]
                inverse_transformed[i] = predictions_np[i] * scale
            
            elif self.scaling_method == 'standard':
                mean = self.scaler_params[window_idx, col_idx, 0]
                std = self.scaler_params[window_idx, col_idx, 1]
                inverse_transformed[i] = predictions_np[i] * std + mean
        
        return torch.FloatTensor(inverse_transformed)
    
    def get_window_timestamps(self, window_idx: int) -> pd.DatetimeIndex:
        """
        Get the actual timestamps for a given window index.

        Args:
            window_idx: Index of the window

        Returns:
            DatetimeIndex of timestamps for the window
        """
        window = self.windows[window_idx]
        entity_id = window['entity_id']
        df = self.series_data[entity_id]
        
        start_idx = window['start_idx']
        end_idx = window['end_idx']

        # Get the time index
        if hasattr(df.index, 'to_timestamp'):
            timestamps = df.index[start_idx:end_idx].to_timestamp()
        else:
            timestamps = df.index[start_idx:end_idx]

        return pd.DatetimeIndex(timestamps)

    def get_future_timestamps(self, window_idx: int) -> pd.DatetimeIndex:
        """
        Get only the future/prediction timestamps for a window.

        Args:
            window_idx: Index of the window

        Returns:
            DatetimeIndex of future timestamps
        """
        window = self.windows[window_idx]
        entity_id = window['entity_id']
        df = self.series_data[entity_id]
        
        if window.get('is_ultra_short', False):
            # For ultra-short, data IS in the future period
            # Return actual timestamps from the data
            return pd.DatetimeIndex(df.index[:window['future_data_len']])

        future_start = window['historical_end']
        end_idx = window['end_idx']

        # Get the future time index
        if hasattr(df.index, 'to_timestamp'):
            timestamps = df.index[future_start:end_idx].to_timestamp()
        else:
            timestamps = df.index[future_start:end_idx]

        return pd.DatetimeIndex(timestamps)

    def get_window_info(self, window_idx: int) -> Dict:
        """
        Get comprehensive information about a window.

        Args:
            window_idx: Index of the window

        Returns:
            Dictionary with window metadata
        """
        window = self.windows[window_idx]
        entity_id = window['entity_id']

        return {
            'entity_id': entity_id,
            'window_idx': window_idx,
            'start_idx': window['start_idx'],
            'end_idx': window['end_idx'],
            'historical_end': window['historical_end'],
            'padding_steps': window['padding_steps'],
            'timestamps': self.get_window_timestamps(window_idx),
            'future_timestamps': self.get_future_timestamps(window_idx)
        }
    
    def _report_dataset_stats(self):
        """Report dataset statistics including memory usage."""
        n_windows = len(self.windows)
        n_padded = sum(1 for w in self.windows if w['padding_steps'] > 0)
        
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Series: {len(self.series_data)}")
        print(f"Windows: {n_windows:,}")
        print(f"Padded windows: {n_padded:,}")
        print(f"Features: {len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical")
        
        # Memory usage
        if self.scaler_params is not None:
            memory_mb = self.scaler_params.nbytes / (1024**2)
            print(f"\nMemory Usage:")
            print(f"  Scaler parameters: {memory_mb:.1f} MB")
            print(f"  Array shape: {self.scaler_params.shape}")
            
            # Compare to object storage
            if self.scaling_method == 'mean':
                object_memory_mb = n_windows * len(self.numeric_cols) * 300 / (1024**2)
            else:  # standard
                object_memory_mb = n_windows * len(self.numeric_cols) * 400 / (1024**2)
            
            #print(f"  If stored as objects: ~{object_memory_mb:.1f} MB")
            #print(f"  Memory savings: {(1 - memory_mb/object_memory_mb)*100:.1f}%")
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        stats = {
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
                'padding_enabled': self.enable_padding
            },
            'scaling': {
                'method': self.scaling_method,
                'memory_mb': self.scaler_params.nbytes / (1024**2) if self.scaler_params is not None else 0
            }
        }
        
        # Series length statistics
        series_lengths = [len(df) for df in self.series_data.values()]
        if series_lengths:
            stats['series_lengths'] = {
                'min': min(series_lengths),
                'max': max(series_lengths),
                'mean': np.mean(series_lengths),
                'median': np.median(series_lengths)
            }
        
        return stats


# In[ ]:


# Utility function to create the 'categorical_embedding_dims' parameter of the TFT models.

def create_uniform_embedding_dims(dataset, hidden_layer_size=160):
    """Create embeddings with uniform hidden_layer_size."""
    embedding_dims = {}
    encoder_mappings = dataset.get_encoder_mappings()
    
    # Static categorical
    for i, col_name in enumerate(dataset.static_categorical_cols):
        if col_name in encoder_mappings:
            vocab_size = len(encoder_mappings[col_name]['classes']) + 2  # +2 for padding/unknown
            embedding_dims[f"static_cat_{i}"] = (vocab_size, hidden_layer_size)
    
    # Historical categorical  
    all_historical_cat = (dataset.temporal_unknown_categorical_cols + 
                          dataset.temporal_known_categorical_cols)
    for i, col_name in enumerate(all_historical_cat):
        if col_name in encoder_mappings:
            vocab_size = len(encoder_mappings[col_name]['classes']) + 2
            embedding_dims[f"historical_cat_{i}"] = (vocab_size, hidden_layer_size)
    
    # Future categorical
    for i, col_name in enumerate(dataset.temporal_known_categorical_cols):
        if col_name in encoder_mappings:
            vocab_size = len(encoder_mappings[col_name]['classes']) + 2
            embedding_dims[f"future_cat_{i}"] = (vocab_size, hidden_layer_size)
    
    return embedding_dims


# 
# ## TFTDataAdapter: Dataset-Model Bridge
# 
# ### Purpose
# The `TFTDataAdapter` bridges the format mismatch between `OptimizedTFTDataset` outputs and TFT model inputs:
# - **Dataset returns:** Stacked feature tensors (e.g., all static features in one tensor `[batch, n_features]`)
# - **Model expects:** Lists of individual feature tensors (e.g., `[tensor1, tensor2, ...]` where each is `[batch, 1]`)
# 
# ### How It Works
# 1. **Initialization:** Reads feature configuration from the dataset to understand feature types and counts
# 2. **Collation:** Custom `collate_fn` properly batches samples from the dataset
# 3. **Transformation:** Splits stacked tensors into lists and separates temporal features into historical/future portions
# 
# ### Key Methods
# - `collate_fn(batch)`: Custom collation for DataLoader
# - `adapt_for_tft(batch)`: Transforms batch for full TemporalFusionTransformer
# - `adapt_for_encoder_only(batch)`: Transforms batch for TFTEncoderOnly model
# 
# ### Usage
# ```python
# # Create dataset and adapter
# dataset = OptimizedTFTDataset(data_source=df, features_config=config, ...)
# dataloader, adapter = create_tft_dataloader(dataset, batch_size=32)
# 
# # Training loop
# for batch in dataloader:
#     # Transform batch format - now includes all necessary data
#     model_inputs = adapter.adapt_for_tft(batch)
#     
#     # Model forward pass - padding mask is now properly used
#     outputs = model(**model_inputs)
#     
#     # Calculate loss using targets from model_inputs
#     loss = model.quantile_loss(
#         outputs['predictions'], 
#         model_inputs['future_targets'],  # Now available from adapter
#         quantiles=[0.1, 0.5, 0.9]
#     )
#     
#     # For evaluation, inverse transform predictions
#     if evaluating:
#         original_scale_preds = inverse_transform_predictions(
#             outputs['predictions'], 
#             batch,  # Original batch has scaler params
#             dataset
#         )
# ```
# 
# ### Feature Mapping Example
# ```
# Dataset Output                    →  Model Input
# static_numeric [B, 3]             →  static_continuous: [tensor[B,1], tensor[B,1], tensor[B,1]]
# temporal_known [B, T_all, 2]     →  historical_continuous: [tensor[B,T_hist], tensor[B,T_hist]]
#                                   →  future_continuous: [tensor[B,T_pred], tensor[B,T_pred]]
# ```
# 
# The adapter handles all feature types (static/temporal, numeric/categorical, known/unknown) automatically based on the dataset's configuration.
# 

# In[ ]:


class TFTDataAdapter:
    """
    Adapter to bridge the gap between OptimizedTFTDataset outputs and TFT model inputs.
    
    The dataset returns stacked features (all features in one tensor),
    while the model expects lists of tensors (one tensor per feature).
    """
    
    def __init__(self, dataset):
        """
        Initialize the adapter with dataset metadata.
        
        Args:
            dataset: OptimizedTFTDataset instance to get feature configuration
        """
        self.dataset = dataset
        
        # Store feature lists for proper ordering
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
        """
        Custom collate function for DataLoader that maintains the structure
        needed for the adapter.
        
        Args:
            batch: List of sample dictionaries from the dataset
            
        Returns:
            Batched dictionary with all tensors properly stacked
        """
        # Initialize output dictionary
        collated = {}
        
        # Get all keys from first sample
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['entity_id']:
                # Keep as list of strings
                collated[key] = [sample[key] for sample in batch]
            elif key in ['window_idx']:
                # Convert to tensor of indices
                collated[key] = torch.tensor([sample[key] for sample in batch])
            else:
                # Stack tensors
                if key in batch[0]:
                    collated[key] = torch.stack([sample[key] for sample in batch])
        
        return collated
    
    def adapt_for_tft(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Adapt batched dataset output to TFT model input format.
        
        Args:
            batch: Collated batch from DataLoader
            
        Returns:
            Dictionary with lists of tensors for each feature type
        """
        model_inputs = {}
        
        # 1. Static categorical features
        if 'static_categorical' in batch and len(self.static_categorical_cols) > 0:
            # Split [batch_size, n_features] -> List of [batch_size] tensors
            static_categorical = []
            for i in range(len(self.static_categorical_cols)):
                static_categorical.append(batch['static_categorical'][:, i])
            model_inputs['static_categorical'] = static_categorical
        
        # 2. Static continuous features
        if 'static_numeric' in batch and len(self.static_numeric_cols) > 0:
            # Split [batch_size, n_features] -> List of [batch_size, 1] tensors
            static_continuous = []
            for i in range(len(self.static_numeric_cols)):
                static_continuous.append(batch['static_numeric'][:, i:i+1])
            model_inputs['static_continuous'] = static_continuous
        
        # 3. Historical categorical features (for both models)
        historical_categorical = []
        
        # Add temporal unknown categorical (historical only)
        if 'temporal_unknown_categorical' in batch and len(self.temporal_unknown_categorical_cols) > 0:
            # Shape: [batch_size, historical_steps, n_features]
            for i in range(len(self.temporal_unknown_categorical_cols)):
                historical_categorical.append(batch['temporal_unknown_categorical'][:, :, i])
        
        # Add temporal known categorical (historical portion)
        if 'temporal_known_categorical' in batch and len(self.temporal_known_categorical_cols) > 0:
            # Shape: [batch_size, total_steps, n_features]
            # Extract historical portion
            for i in range(len(self.temporal_known_categorical_cols)):
                historical_categorical.append(
                    batch['temporal_known_categorical'][:, :self.historical_steps, i]
                )
        
        if historical_categorical:
            model_inputs['historical_categorical'] = historical_categorical
        
        # 4. Historical continuous features (for both models)
        historical_continuous = []
        
        # Add targets as historical features
        if 'historical_targets' in batch:
            # Shape: [batch_size, historical_steps, n_targets]
            if batch['historical_targets'].dim() == 3:
                for i in range(batch['historical_targets'].shape[2]):
                    historical_continuous.append(batch['historical_targets'][:, :, i])
            else:
                # Single target
                historical_continuous.append(batch['historical_targets'])
        
        # Add temporal unknown numeric (historical only)
        if 'temporal_unknown_numeric' in batch and len(self.temporal_unknown_numeric_cols) > 0:
            # Shape: [batch_size, historical_steps, n_features]
            for i in range(len(self.temporal_unknown_numeric_cols)):
                historical_continuous.append(batch['temporal_unknown_numeric'][:, :, i])
        
        # Add temporal known numeric (historical portion)
        if 'temporal_known_numeric' in batch and len(self.temporal_known_numeric_cols) > 0:
            # Shape: [batch_size, total_steps, n_features]
            for i in range(len(self.temporal_known_numeric_cols)):
                historical_continuous.append(
                    batch['temporal_known_numeric'][:, :self.historical_steps, i]
                )
        
        if historical_continuous:
            model_inputs['historical_continuous'] = historical_continuous
        
        # 5. Future categorical features (for full TFT model)
        if 'temporal_known_categorical' in batch and len(self.temporal_known_categorical_cols) > 0:
            future_categorical = []
            # Extract future portion
            for i in range(len(self.temporal_known_categorical_cols)):
                future_categorical.append(
                    batch['temporal_known_categorical'][:, self.historical_steps:, i]
                )
            model_inputs['future_categorical'] = future_categorical
        
        # 6. Future continuous features (for full TFT model)
        if 'temporal_known_numeric' in batch and len(self.temporal_known_numeric_cols) > 0:
            future_continuous = []
            # Extract future portion
            for i in range(len(self.temporal_known_numeric_cols)):
                future_continuous.append(
                    batch['temporal_known_numeric'][:, self.historical_steps:, i]
                )
            model_inputs['future_continuous'] = future_continuous
        
        # 7. NEW: Add padding mask (convert to proper shape for attention)
        if 'mask' in batch:
            # Create padding mask for attention layers
            # Shape: [batch_size, 1, seq_len] for broadcasting in attention
            model_inputs['padding_mask'] = self._create_padding_mask(batch['mask'])
            
        # 8. NEW: Pass through scaler parameters for inverse transformation
        for key in ['target_scale', 'target_mean', 'target_std']:
            if key in batch:
                model_inputs[key] = batch[key]
        
        # 9. NEW: Pass through entity weights if present
        if 'entity_weight' in batch:
            model_inputs['entity_weight'] = batch['entity_weight']
            
        # 10. NEW: Pass through targets separately (not as features)
        model_inputs['historical_targets'] = batch.get('historical_targets')
        model_inputs['future_targets'] = batch.get('future_targets')

        # 11. NEW: Pass through time index if needed
        if 'time_index' in batch:
            model_inputs['time_index'] = batch['time_index']

        # 12. NEW: Pass through window indices for inverse transformation
        if 'window_idx' in batch:
            model_inputs['window_idx'] = batch['window_idx']
        
        return model_inputs
    
    def _create_padding_mask(self, mask):
        """
        Create padding mask for attention mechanism.

        Args:
            mask: [batch_size, seq_len] binary mask (1=real, 0=padded)

        Returns:
            padding_mask: [batch_size, 1, seq_len] mask for attention (1=masked, 0=attend)
        """
        # Convert: 1 (real) -> 0 (attend), 0 (padded) -> 1 (mask)
        # This follows the convention where 1 means "mask out" in attention
        padding_mask = 1.0 - mask
        return padding_mask.unsqueeze(1)
    
    def adapt_for_encoder_only_orig(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Adapt batched dataset output specifically for TFTEncoderOnly model.
        
        Args:
            batch: Collated batch from DataLoader
            
        Returns:
            Dictionary with lists of tensors for encoder-only model
        """
        # Use the same adaptation but only return relevant keys
        full_inputs = self.adapt_for_tft(batch)
        
        encoder_inputs = {}
        
        # Encoder-only model uses these keys
        for key in ['historical_continuous', 'historical_categorical', 
                    'static_continuous', 'static_categorical']:
            if key in full_inputs:
                encoder_inputs[key] = full_inputs[key]
        
        # For encoder-only, we only need historical mask
        if 'padding_mask' in encoder_inputs:
            # Trim to historical portion only
            encoder_inputs['padding_mask'] = encoder_inputs['padding_mask'][:, :, :self.historical_steps]
        
        return encoder_inputs
    
    def adapt_for_encoder_only(self, batch: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """
        Adapt batched dataset output for TFTEncoderOnly model.
        Simply excludes historical_targets from historical_continuous.
        """
        model_inputs = {}
        
        # 1. Static categorical features (unchanged)
        if 'static_categorical' in batch and len(self.static_categorical_cols) > 0:
            static_categorical = []
            for i in range(len(self.static_categorical_cols)):
                static_categorical.append(batch['static_categorical'][:, i])
            model_inputs['static_categorical'] = static_categorical
        
        # 2. Static continuous features (unchanged)
        if 'static_numeric' in batch and len(self.static_numeric_cols) > 0:
            static_continuous = []
            for i in range(len(self.static_numeric_cols)):
                static_continuous.append(batch['static_numeric'][:, i:i+1])
            model_inputs['static_continuous'] = static_continuous
        
        # 3. Historical categorical features
        historical_categorical = []
        
        if 'temporal_unknown_categorical' in batch and len(self.temporal_unknown_categorical_cols) > 0:
            for i in range(len(self.temporal_unknown_categorical_cols)):
                historical_categorical.append(batch['temporal_unknown_categorical'][:, :, i])
        
        if 'temporal_known_categorical' in batch and len(self.temporal_known_categorical_cols) > 0:
            for i in range(len(self.temporal_known_categorical_cols)):
                historical_categorical.append(
                    batch['temporal_known_categorical'][:, :self.historical_steps, i]
                )
        
        if historical_categorical:
            model_inputs['historical_categorical'] = historical_categorical
        
        # 4. Historical continuous features - SKIP historical_targets!
        historical_continuous = []
        
        # DON'T add historical_targets - the derived target shouldn't be an input feature
        
        # Add temporal unknown numeric (now includes original sales as a feature)
        if 'temporal_unknown_numeric' in batch and len(self.temporal_unknown_numeric_cols) > 0:
            for i in range(len(self.temporal_unknown_numeric_cols)):
                historical_continuous.append(batch['temporal_unknown_numeric'][:, :, i])
        
        # Add temporal known numeric (historical portion)
        if 'temporal_known_numeric' in batch and len(self.temporal_known_numeric_cols) > 0:
            for i in range(len(self.temporal_known_numeric_cols)):
                historical_continuous.append(
                    batch['temporal_known_numeric'][:, :self.historical_steps, i]
                )
        
        if historical_continuous:
            model_inputs['historical_continuous'] = historical_continuous
        
        # 5. Padding mask (historical portion only)
        if 'mask' in batch:
            historical_mask = batch['mask'][:, :self.historical_steps]
            model_inputs['padding_mask'] = self._create_padding_mask(historical_mask)
        
        # 6. Target - use future_targets which contains the derived target
        if 'future_targets' in batch:
            # Take just the first timestep since derived target should be constant
            # Shape: [batch_size, prediction_steps, n_targets] -> [batch_size, n_targets]
            target = batch['future_targets'][:, 0, :]
            if target.dim() == 1:
                target = target.unsqueeze(-1)
            model_inputs['target'] = target
        
        # 7. Pass through metadata
        for key in ['target_scale', 'target_mean', 'target_std', 
                    'entity_weight', 'recency_weight', 'window_idx', 'entity_id']:
            if key in batch:
                model_inputs[key] = batch[key]
        
        return model_inputs


def create_tft_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False, pin_memory=True):
    """
    Create a DataLoader with proper collation for TFT models.
    
    Args:
        dataset: OptimizedTFTDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (dataloader, adapter)
    """
    adapter = TFTDataAdapter(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=adapter.collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory
    )
    
    return dataloader, adapter


def inverse_transform_predictions(predictions, batch, dataset, target_idx=0):
    """
    Helper function to inverse transform model predictions using scaler parameters.
    
    Args:
        predictions: Model predictions [batch_size, ...]
        batch: Original batch dictionary containing scaler parameters
        dataset: OptimizedTFTDataset instance (for scaling_method info)
        target_idx: Index of target if multiple targets (default 0)
    
    Returns:
        Inverse transformed predictions in original scale
    """
    if dataset.scaling_method == 'none':
        return predictions
    
    predictions_np = predictions.detach().cpu().numpy()
    
    if dataset.scaling_method == 'mean':
        if 'target_scale' in batch:
            scale = batch['target_scale'][:, target_idx].cpu().numpy()
            # Reshape scale for broadcasting if needed
            while scale.ndim < predictions_np.ndim:
                scale = scale[..., np.newaxis]
            return torch.tensor(predictions_np * scale)
    
    elif dataset.scaling_method == 'standard':
        if 'target_mean' in batch and 'target_std' in batch:
            mean = batch['target_mean'][:, target_idx].cpu().numpy()
            std = batch['target_std'][:, target_idx].cpu().numpy()
            # Reshape for broadcasting
            while mean.ndim < predictions_np.ndim:
                mean = mean[..., np.newaxis]
                std = std[..., np.newaxis]
            return torch.tensor(predictions_np * std + mean)
    
    return predictions


# ## TCNDataAdapter: Dataset-TCN Bridge
# 
# ### Purpose
# The `TCNDataAdapter` transforms `OptimizedTFTDataset` outputs into TCN-compatible format:
# - **Dataset returns:** Separate tensors for different feature types
# - **TCN expects:** Concatenated features ready for temporal convolutions
# - **Key principle:** Adapter only reorganizes data; embeddings are handled by the model
# 
# ### How It Works
# 1. **Feature Separation:** Separates numeric and categorical features
# 2. **Static Broadcasting:** Repeats static features across all timesteps
# 3. **Temporal Alignment:** Handles padding for unknown future features
# 4. **Encoder Support:** Configurable for full sequence or historical-only
# 
# ### Output Format
# The adapter returns:
# - `numeric_features`: All continuous features `[batch, timesteps, n_numeric]`
# - `categorical_features`: Raw categorical indices `[batch, timesteps, n_categorical]`
# - `targets`: Future targets for loss calculation
# - `mask`: Temporal mask for valid timesteps
# 
# ### Usage
# ```python
# # Create dataset and TCN adapter
# dataset = OptimizedTFTDataset(data_source=df, features_config=config, ...)
# dataloader, tcn_adapter = create_tcn_dataloader(dataset, batch_size=32)
# 
# # Get feature dimensions for model initialization
# feature_info = tcn_adapter.get_feature_info()
# n_numeric = feature_info['n_numeric_features']        # For input layer
# n_categorical = feature_info['n_categorical_features']  # For embedding layers
# 
# # Training loop
# for batch in dataloader:
#     # Full sequence (historical + future known)
#     tcn_input = tcn_adapter.adapt_for_tcn(batch, encoder_only=False)
#     
#     # Encoder-only (historical only)
#     tcn_input = tcn_adapter.adapt_for_tcn(batch, encoder_only=True)
#     
#     # Model handles embeddings internally
#     output = tcn_model(
#         tcn_input['numeric_features'],     # Continuous features
#         tcn_input['categorical_features']  # Categorical indices (to be embedded)
#     )
#     loss = criterion(output, tcn_input['targets'])
#     

# In[ ]:


class TCNDataAdapter:
    """
    Adapter to bridge OptimizedTFTDataset outputs to TCN model inputs.
    
    TCN models expect a single concatenated feature tensor [batch, timesteps, features]
    where all feature types are combined into one channel dimension.
    
    Note: This adapter only reorganizes data. Categorical embeddings should be 
    handled by the TCN model itself.
    """
    
    def __init__(self, dataset):
        """
        Initialize the TCN adapter.
        
        Args:
            dataset: OptimizedTFTDataset instance
        """
        self.dataset = dataset
        
        # Store feature lists
        self.static_numeric_cols = dataset.static_numeric_cols
        self.static_categorical_cols = dataset.static_categorical_cols
        self.temporal_known_numeric_cols = dataset.temporal_known_numeric_cols
        self.temporal_known_categorical_cols = dataset.temporal_known_categorical_cols
        self.temporal_unknown_numeric_cols = dataset.temporal_unknown_numeric_cols
        self.temporal_unknown_categorical_cols = dataset.temporal_unknown_categorical_cols
        self.target_cols = dataset.target_cols
        
        self.historical_steps = dataset.historical_steps
        self.prediction_steps = dataset.prediction_steps
        
        # Calculate feature counts
        self.n_numeric_features = self._calculate_numeric_features()
        self.n_categorical_features = self._calculate_categorical_features()
        self.n_total_features = self.n_numeric_features + self.n_categorical_features
    
    def _calculate_numeric_features(self) -> int:
        """Calculate total numeric feature count."""
        return (len(self.static_numeric_cols) + 
                len(self.temporal_known_numeric_cols) + 
                len(self.temporal_unknown_numeric_cols) + 
                len(self.target_cols))
    
    def _calculate_categorical_features(self) -> int:
        """Calculate total categorical feature count."""
        return (len(self.static_categorical_cols) + 
                len(self.temporal_known_categorical_cols) + 
                len(self.temporal_unknown_categorical_cols))
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader.
        """
        collated = {}
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['entity_id']:
                collated[key] = [sample[key] for sample in batch]
            elif key in ['window_idx']:
                collated[key] = torch.tensor([sample[key] for sample in batch])
            else:
                if key in batch[0]:
                    collated[key] = torch.stack([sample[key] for sample in batch])
        
        return collated
    
    def adapt_for_tcn(self, batch: Dict[str, torch.Tensor], 
                      encoder_only: bool = False) -> Dict[str, torch.Tensor]:
        """
        Adapt batch for TCN models.
        
        Args:
            batch: Collated batch from DataLoader
            encoder_only: If True, only return historical features
            
        Returns:
            Dictionary with:
                - 'numeric_features': [batch, timesteps, n_numeric] tensor of continuous features
                - 'categorical_features': [batch, timesteps, n_categorical] tensor of categorical indices
                - 'targets': Future targets for prediction
                - 'mask': Temporal mask
                
        Note: The TCN model should handle embedding of categorical features internally.
        """
        batch_size = batch['mask'].shape[0]
        timesteps = self.historical_steps if encoder_only else (self.historical_steps + self.prediction_steps)
        
        numeric_features = []
        categorical_features = []
        
        # 1. Process historical targets as numeric features
        if 'historical_targets' in batch:
            if batch['historical_targets'].dim() == 3:
                hist_targets = batch['historical_targets']
            else:
                hist_targets = batch['historical_targets'].unsqueeze(-1)
            
            if not encoder_only and self.prediction_steps > 0:
                # Pad for future timesteps
                padding = torch.zeros(batch_size, self.prediction_steps, hist_targets.shape[-1])
                hist_targets = torch.cat([hist_targets, padding], dim=1)
            
            numeric_features.append(hist_targets)
        
        # 2. Process temporal unknown numeric features
        if 'temporal_unknown_numeric' in batch and len(self.temporal_unknown_numeric_cols) > 0:
            unknown_numeric = batch['temporal_unknown_numeric']
            
            if not encoder_only and self.prediction_steps > 0:
                # Pad with zeros for future timesteps
                padding = torch.zeros(batch_size, self.prediction_steps, unknown_numeric.shape[-1])
                unknown_numeric = torch.cat([unknown_numeric, padding], dim=1)
            
            numeric_features.append(unknown_numeric)
        
        # 3. Process temporal unknown categorical features
        if 'temporal_unknown_categorical' in batch and len(self.temporal_unknown_categorical_cols) > 0:
            unknown_cat = batch['temporal_unknown_categorical']
            
            if not encoder_only and self.prediction_steps > 0:
                # Pad with -1 (missing indicator) for future
                padding = torch.full((batch_size, self.prediction_steps, unknown_cat.shape[-1]), -1)
                unknown_cat = torch.cat([unknown_cat, padding], dim=1)
            
            categorical_features.append(unknown_cat)
        
        # 4. Process temporal known numeric features
        if 'temporal_known_numeric' in batch and len(self.temporal_known_numeric_cols) > 0:
            if encoder_only:
                known_numeric = batch['temporal_known_numeric'][:, :self.historical_steps, :]
            else:
                known_numeric = batch['temporal_known_numeric']
            
            numeric_features.append(known_numeric)
        
        # 5. Process temporal known categorical features
        if 'temporal_known_categorical' in batch and len(self.temporal_known_categorical_cols) > 0:
            known_cat = batch['temporal_known_categorical']
            
            if encoder_only:
                known_cat = known_cat[:, :self.historical_steps, :]
            
            categorical_features.append(known_cat)
        
        # 6. Process static numeric features (broadcast across time)
        if 'static_numeric' in batch and len(self.static_numeric_cols) > 0:
            static_numeric = batch['static_numeric']
            # Broadcast: [batch, features] -> [batch, timesteps, features]
            static_numeric = static_numeric.unsqueeze(1).expand(-1, timesteps, -1)
            numeric_features.append(static_numeric)
        
        # 7. Process static categorical features (broadcast across time)
        if 'static_categorical' in batch and len(self.static_categorical_cols) > 0:
            static_cat = batch['static_categorical']
            # Broadcast: [batch, features] -> [batch, timesteps, features]
            static_cat = static_cat.unsqueeze(1).expand(-1, timesteps, -1)
            categorical_features.append(static_cat)
        
        # Concatenate features by type
        output = {}
        
        if numeric_features:
            output['numeric_features'] = torch.cat(numeric_features, dim=-1)
        else:
            output['numeric_features'] = torch.zeros(batch_size, timesteps, 1)
        
        if categorical_features:
            output['categorical_features'] = torch.cat(categorical_features, dim=-1)
        else:
            # No categorical features - provide empty tensor
            output['categorical_features'] = torch.zeros(batch_size, timesteps, 0, dtype=torch.long)
        
        # Add mask and targets
        output['mask'] = batch['mask'][:, :timesteps] if encoder_only else batch['mask']
        
        if 'future_targets' in batch:
            output['targets'] = batch['future_targets']
        
        # Add metadata
        output['entity_id'] = batch.get('entity_id', None)
        output['window_idx'] = batch.get('window_idx', None)
        
        return output
    
    def get_feature_info(self) -> Dict:
        """Get information about the feature composition."""
        info = {
            'n_numeric_features': self.n_numeric_features,
            'n_categorical_features': self.n_categorical_features,
            'n_total_raw_features': self.n_total_features,
            'feature_breakdown': {
                'numeric': {
                    'targets': len(self.target_cols),
                    'static': len(self.static_numeric_cols),
                    'temporal_known': len(self.temporal_known_numeric_cols),
                    'temporal_unknown': len(self.temporal_unknown_numeric_cols),
                },
                'categorical': {
                    'static': len(self.static_categorical_cols),
                    'temporal_known': len(self.temporal_known_categorical_cols),
                    'temporal_unknown': len(self.temporal_unknown_categorical_cols),
                }
            },
            'categorical_columns': {
                'static': self.static_categorical_cols,
                'temporal_known': self.temporal_known_categorical_cols,
                'temporal_unknown': self.temporal_unknown_categorical_cols,
            }
        }
        return info


def create_tcn_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader with TCN adapter.
    
    Args:
        dataset: OptimizedTFTDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (dataloader, adapter)
    """
    adapter = TCNDataAdapter(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=adapter.collate_fn
    )
    
    return dataloader, adapter


# ### Usage Example
# 
# """
# Usage example for the optimized TFT Dataset implementation.
# Demonstrates memory efficiency and proper window-specific scaling.
# """
# 
# 
# def create_large_scale_data():
#     """
#     Create a large dataset to demonstrate memory efficiency.
#     """
#     np.random.seed(42)
#     
#     # Create 100 series with 1000 timesteps each
#     n_series = 100
#     n_timesteps = 1000
#     
#     data_list = []
#     
#     for series_id in range(n_series):
#         dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
#         
#         # Different series have different patterns
#         if series_id < 30:
#             # Stable series
#             base = 100
#             trend = 0
#         elif series_id < 60:
#             # Growing series
#             base = 50
#             trend = 0.1
#         else:
#             # Series with regime change
#             base = 100
#             trend = 0.05
#         
#         # Generate sales with trend
#         t = np.arange(n_timesteps)
#         sales = base + trend * t + 10 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 5, n_timesteps)
#         
#         # Add regime change for some series
#         if series_id >= 60 and n_timesteps > 500:
#             sales[500:] += 50  # Jump at day 500
#         
#         series_data = pd.DataFrame({
#             'entity_id': f'series_{series_id:03d}',
#             'date': dates,
#             
#             # Target
#             'sales': np.maximum(sales, 0),
#             
#             # Static features
#             'store_size': 1000 + series_id * 10,
#             'location': series_id % 5,
#             
#             # Known temporal features (we know these in future)
#             'day_of_week': dates.dayofweek,
#             'month': dates.month,
#             'temperature': 20 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, n_timesteps),
#             'holiday': np.random.choice([0, 1], n_timesteps, p=[0.95, 0.05]),
#             
#             # Unknown temporal features (only historical)
#             'competitor_price': 100 + np.random.normal(0, 5, n_timesteps),
#             'foot_traffic': 1000 + np.random.normal(0, 100, n_timesteps),
#             'promotion': np.random.choice([0, 1], n_timesteps, p=[0.8, 0.2]),
#             
#             # Weight
#             'importance': 1.0 + (series_id % 3) * 0.5
#         })
#         
#         data_list.append(series_data)
#     
#     df = pd.concat(data_list, ignore_index=True)
#     
#     print(f"Created dataset:")
#     print(f"  Total rows: {len(df):,}")
#     print(f"  Series: {n_series}")
#     print(f"  Timesteps per series: {n_timesteps}")
#     print(f"  Features: {len(df.columns) - 2}")  # Minus entity_id and date
#     
#     return df
# 
# 
# def demonstrate_memory_efficiency():
#     """
#     Demonstrate the memory efficiency of the optimized implementation.
#     """
#     print("="*80)
#     print("MEMORY EFFICIENCY DEMONSTRATION")
#     print("="*80)
#     
#     # Create large dataset
#     df = create_large_scale_data()
#     
#     # Define features
#     features_config = {
#         "entity_col": "entity_id",
#         "target_col": "sales",
#         "time_index_col": "date",
#         
#         # Static features
#         "static_numeric_col_list": ["store_size"],
#         "static_categorical_col_list": ["location"],
#         
#         # Known temporal (available in future)
#         "temporal_known_numeric_col_list": ["temperature"],
#         "temporal_known_categorical_col_list": ["day_of_week", "month", "holiday"],
#         
#         # Unknown temporal (only historical)
#         "temporal_unknown_numeric_col_list": ["competitor_price", "foot_traffic"],
#         "temporal_unknown_categorical_col_list": ["promotion"],
#         
#         # Weight
#         "wt_col": "importance"
#     }
#     
#     # Create dataset with optimized scaling
#     dataset = OptimizedTFTDataset(
#         data_source=df,
#         features_config=features_config,
#         
#         # Window configuration
#         historical_steps=60,
#         prediction_steps=14,
#         stride=1,  # Create many windows to show memory efficiency
#         
#         # Scaling
#         scaling_method='standard',  # Uses mean + std
#         
#         # Performance
#         n_jobs=4,
#         use_cache=False,  # Disable for demonstration
#         
#         mode='train'
#     )
#     
#     # The statistics will show actual memory usage
#     return dataset
# 
# 
# def verify_no_data_leakage(dataset, n_checks=5):
#     """
#     Verify that scalers are properly window-specific without leakage.
#     """
#     print("\n" + "="*80)
#     print("VERIFYING NO DATA LEAKAGE")
#     print("="*80)
#     
#     # Check random windows
#     window_indices = np.random.choice(len(dataset.windows), min(n_checks, len(dataset.windows)), replace=False)
#     
#     for idx in window_indices:
#         window = dataset.windows[idx]
#         entity_id = window['entity_id']
#         df = dataset.series_data[entity_id]
#         
#         # Get stored scaler parameters
#         if dataset.scaling_method == 'standard':
#             target_idx = dataset.feature_to_idx.get('sales', 0)
#             stored_mean = dataset.scaler_params[idx, target_idx, 0]
#             stored_std = dataset.scaler_params[idx, target_idx, 1]
#             
#             # Calculate what the parameters should be
#             historical_data = df.iloc[:window['historical_end']]['sales'].dropna().values
#             expected_mean = np.mean(historical_data)
#             expected_std = np.std(historical_data, ddof=1)
#             
#             print(f"\nWindow {idx} (Entity: {entity_id}):")
#             print(f"  Historical period: [0:{window['historical_end']}]")
#             print(f"  Stored params: mean={stored_mean:.2f}, std={stored_std:.2f}")
#             print(f"  Expected params: mean={expected_mean:.2f}, std={expected_std:.2f}")
#             
#             if abs(stored_mean - expected_mean) < 1e-5 and abs(stored_std - expected_std) < 1e-5:
#                 print(f"  ✓ No leakage - parameters match!")
#             else:
#                 print(f"  ⚠ Mismatch detected!")
# 
# 
# def demonstrate_dataloader_usage(dataset):
#     """
#     Show how to use the dataset with PyTorch DataLoader.
#     """
#     print("\n" + "="*80)
#     print("DATALOADER USAGE")
#     print("="*80)
#     
#     # Create DataLoader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=True,
#         num_workers=0
#     )
#     
#     # Get one batch
#     for batch_idx, batch in enumerate(dataloader):
#         if batch_idx == 0:
#             print("\nBatch information:")
#             print(f"  Batch size: {len(batch['entity_id'])}")
#             
#             # Show available tensors
#             print("\nAvailable tensors:")
#             for key in sorted(batch.keys()):
#                 if key not in ['entity_id', 'window_idx']:
#                     tensor = batch[key]
#                     print(f"  {key}: shape {tuple(tensor.shape)}")
#             
#             # Demonstrate inverse transform
#             if 'future_targets' in batch:
#                 # Simulate predictions
#                 fake_predictions = batch['future_targets'] * 1.1
#                 
#                 # Get window indices
#                 window_indices = batch['window_idx'].tolist()
#                 
#                 # Inverse transform
#                 original_scale = dataset.inverse_transform_predictions(
#                     fake_predictions,
#                     window_indices,
#                     target_col='sales'
#                 )
#                 
#                 print("\nInverse transform example:")
#                 print(f"  Scaled prediction: {fake_predictions[0, 0, 0]:.2f}")
#                 print(f"  Original scale: {original_scale[0, 0, 0]:.2f}")
#             
#             break
# 
# 
# def compare_memory_usage():
#     """
#     Compare memory usage between object storage and value storage.
#     """
#     print("\n" + "="*80)
#     print("MEMORY COMPARISON: OBJECTS VS VALUES")
#     print("="*80)
#     
#     n_windows = 100_000
#     n_features = 20
#     
#     print(f"\nScenario: {n_windows:,} windows with {n_features} features")
#     print("-"*60)
#     
#     # Object storage (old approach)
#     object_size_per_scaler = 300  # bytes (conservative estimate)
#     object_memory = n_windows * n_features * object_size_per_scaler
#     object_memory_mb = object_memory / (1024**2)
#     
#     print(f"\nOld approach (storing scaler objects):")
#     print(f"  Memory per scaler: ~{object_size_per_scaler} bytes")
#     print(f"  Total memory: {object_memory_mb:.1f} MB")
#     
#     # Value storage (optimized approach)
#     # Standard scaling: 2 float32 per feature
#     value_memory = n_windows * n_features * 2 * 4  # 2 params * 4 bytes
#     value_memory_mb = value_memory / (1024**2)
#     
#     print(f"\nOptimized approach (storing values):")
#     print(f"  Memory per scaler: 8 bytes (2 float32)")
#     print(f"  Total memory: {value_memory_mb:.1f} MB")
#     
#     print(f"\nMemory savings: {(1 - value_memory/object_memory)*100:.1f}%")
#     print(f"Reduction factor: {object_memory/value_memory:.1f}x")
# 
# 
# def main():
#     """
#     Main demonstration of the optimized TFT dataset.
#     """
#     print("="*80)
#     print("OPTIMIZED TFT DATASET DEMONSTRATION")
#     print("="*80)
#     print("\nKey Features:")
#     print("  ✓ Stores only scaler parameters (not objects)")
#     print("  ✓ Window-specific scaling without data leakage")
#     print("  ✓ Memory efficient (40-80 MB for 500k windows)")
#     print("  ✓ Support for padding and variable-length series")
#     print("  ✓ Explicit feature configuration")
#     
#     # Create and test dataset
#     dataset = demonstrate_memory_efficiency()
#     
#     # Verify no data leakage
#     verify_no_data_leakage(dataset)
#     
#     # Show DataLoader usage
#     demonstrate_dataloader_usage(dataset)
#     
#     # Compare memory usage
#     compare_memory_usage()
#     
#     print("\n" + "="*80)
#     print("SUCCESS!")
#     print("="*80)
#     print("\nThe optimized implementation provides:")
#     print("  • 97% memory reduction compared to object storage")
#     print("  • Window-specific scaling with no data leakage")
#     print("  • Efficient enough for millions of windows")
#     print("  • Production-ready performance")
#     
#     # Show actual memory for a large dataset
#     stats = dataset.get_dataset_statistics()
#     print(f"\nActual dataset created:")
#     print(f"  Windows: {stats['n_windows']:,}")
#     print(f"  Scaler memory: {stats['scaling']['memory_mb']:.1f} MB")
#     
#     if stats['n_windows'] > 0:
#         mb_per_million = stats['scaling']['memory_mb'] / (stats['n_windows'] / 1_000_000)
#         print(f"  Memory per million windows: {mb_per_million:.1f} MB")
# 
# 
# if __name__ == "__main__":
#     main()



