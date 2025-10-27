"""
TimeRAG-based Retrieval for In-Context Learning

Implements clustering-based candidate pool construction and DTW-based similarity retrieval.
Based on the TimeRAG paper's approach for efficient time series retrieval.

Key idea:
1. Cluster all training samples using K-means on flattened time series
2. Select centroid-nearest sample from each cluster as representative
3. Use DTW to find most similar samples from this representative pool
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw_ndim
from tqdm import tqdm

try:
    from .sensor_transformation import aggregate_window_features, get_user_window_data
    from . import config
    # Import helper functions to avoid circular dependency
    import importlib
    _example_selection = None
except ImportError:
    from sensor_transformation import aggregate_window_features, get_user_window_data
    import config
    _example_selection = None


def _get_statistical_features():
    """Get statistical features for the current target."""
    if config.DEFAULT_TARGET == 'compass':
        import json
        from pathlib import Path
        config_path = Path(__file__).parent.parent / 'config' / 'globem_use_cols.json'
        with open(config_path, 'r') as f:
            cols_config = json.load(f)
        return list(cols_config['compass']['feature_set']['statistical'].keys())
    return None


def _extract_time_series_from_window_data(
    window_data, stat_features=None
):
    """
    Extract time series from already-filtered window data.
    
    Args:
        window_data: Already filtered and sorted DataFrame
        stat_features: List of statistical feature columns to extract
        
    Returns:
        numpy array of shape (n_days, n_features) or None if insufficient data
    """
    if stat_features is None or window_data is None or len(window_data) == 0:
        return None
    
    import pandas as pd
    
    # Extract feature values for each day
    time_series_list = []
    
    for _, row in window_data.iterrows():
        day_features = []
        for feat_col in stat_features:
            if feat_col in row:
                val = row[feat_col]
                # Convert to float, handle NaN
                if pd.isna(val):
                    day_features.append(0.0)
                else:
                    day_features.append(float(val))
            else:
                day_features.append(0.0)
        
        time_series_list.append(day_features)
    
    if not time_series_list:
        return None
    
    # Convert to numpy array: shape (n_days, n_features)
    time_series = np.array(time_series_list)
    
    return time_series


def _extract_time_series_from_raw_data(
    feat_df, user_id, ema_date, cols,
    window_days=28, stat_features=None
):
    """
    Extract actual 28-day time series from raw feat_df.
    
    Args:
        feat_df: Raw feature dataframe
        user_id: User ID
        ema_date: EMA date
        cols: Column configuration
        window_days: Number of days to look back (default 28)
        stat_features: List of statistical feature columns to extract
        
    Returns:
        numpy array of shape (n_days, n_features) where n_days <= window_days
        Returns None if insufficient data
    """
    # Use shared utility to get window data
    window_data = get_user_window_data(feat_df, user_id, ema_date, cols, window_days)
    
    if window_data is None or len(window_data) < window_days * 0.5:
        return None
    
    return _extract_time_series_from_window_data(window_data, stat_features)


def build_retrieval_candidate_pool_timerag(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    pool_size: Optional[int] = None, random_state: Optional[int] = None
) -> Optional[List[Dict]]:
    """
    Build candidate pool using TimeRAG's clustering-based approach.
    
    Steps:
    1. Extract time series for all trainset samples (28 days x 15 features)
    2. Flatten time series to vectors (28*15 = 420 dimensions)
    3. Apply K-means clustering (K = pool_size)
    4. Select sample nearest to each cluster centroid
    
    Args:
        feat_df: Feature DataFrame
        lab_df: Label DataFrame (trainset only - excluding testset)
        cols: Column configuration
        pool_size: Number of representative samples (default: config.TIMERAG_POOL_SIZE)
        random_state: Random seed
        
    Returns:
        List of candidate dicts with 'time_series', 'sample', and metadata
    """
    if pool_size is None:
        pool_size = config.TIMERAG_POOL_SIZE
    
    print(f"\n[Building TimeRAG Candidate Pool...]")
    print(f"  Method: K-means clustering + centroid-nearest selection")
    print(f"  Trainset size: {len(lab_df)} samples")
    print(f"  Target pool size: {pool_size}")
    
    # Get statistical features
    stat_features = _get_statistical_features()
    if stat_features is None:
        print("  [ERROR] No statistical features found")
        return None
    
    print(f"  Extracting time series for all trainset samples...")
    
    # Step 1: Extract time series for all samples
    all_time_series = []
    all_samples = []
    
    for idx in tqdm(lab_df.index, desc="  Extracting time series", leave=False):
        row = lab_df.loc[idx]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # Extract window data
        window_data = get_user_window_data(feat_df, user_id, ema_date, cols, window_days=28)
        
        if window_data is None or len(window_data) < 14:
            continue
        
        # Extract time series
        time_series = _extract_time_series_from_window_data(window_data, stat_features)
        
        if time_series is not None and time_series.shape[0] > 0:
            # Compute aggregated features for the sample dict
            agg_feats = aggregate_window_features(
                feat_df, user_id, ema_date, cols,
                window_days=28,
                mode='statistics',
                use_immediate_window=False,
                immediate_window_days=0,
                adaptive_window=False,
                precomputed_window_data=window_data
            )
            
            if agg_feats is not None:
                labels = row[cols['labels']].to_dict()
                sample = {
                    'aggregated_features': agg_feats,
                    'labels': labels,
                    'user_id': user_id,
                    'ema_date': ema_date
                }
                
                all_time_series.append(time_series)
                all_samples.append({
                    'time_series': time_series,
                    'sample': sample,
                    'user_id': user_id,
                    'ema_date': ema_date
                })
    
    print(f"  Extracted {len(all_time_series)} valid time series")
    
    if len(all_time_series) < pool_size:
        print(f"  [WARNING] Only {len(all_time_series)} samples, less than target {pool_size}")
        print(f"            Returning all samples")
        return all_samples
    
    # Step 2: Flatten time series to vectors
    print(f"  Flattening time series...")
    flattened_series = []
    for ts in all_time_series:
        # Pad or truncate to fixed length if needed
        # Target shape: (28, n_features)
        n_days, n_features = ts.shape
        
        if n_days < 28:
            # Pad with zeros
            padded = np.pad(ts, ((0, 28 - n_days), (0, 0)), mode='constant', constant_values=0)
        elif n_days > 28:
            # Truncate to last 28 days
            padded = ts[-28:, :]
        else:
            padded = ts
        
        # Flatten to 1D vector
        flattened = padded.flatten()  # Shape: (28 * n_features,)
        flattened_series.append(flattened)
    
    flattened_array = np.array(flattened_series)  # Shape: (n_samples, 28*n_features)
    print(f"  Flattened shape: {flattened_array.shape}")
    
    # Step 3: Normalize features
    print(f"  Normalizing features...")
    scaler = StandardScaler()
    normalized_array = scaler.fit_transform(flattened_array)
    
    # Handle NaN values
    normalized_array = np.nan_to_num(normalized_array, nan=0.0)
    
    # Step 4: Apply K-means clustering
    print(f"  Applying K-means clustering (K={pool_size})...")
    kmeans = KMeans(
        n_clusters=pool_size,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    cluster_labels = kmeans.fit_predict(normalized_array)
    centroids = kmeans.cluster_centers_
    
    print(f"  Clustering complete")
    
    # Step 5: Select sample nearest to each centroid
    print(f"  Selecting representative samples (nearest to centroids)...")
    representative_candidates = []
    
    for cluster_id in range(pool_size):
        # Get samples in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Find sample nearest to centroid
        centroid = centroids[cluster_id]
        cluster_samples = normalized_array[cluster_indices]
        
        # Compute distances to centroid
        distances = np.linalg.norm(cluster_samples - centroid, axis=1)
        nearest_idx = cluster_indices[np.argmin(distances)]
        
        # Add to candidate pool
        representative_candidates.append(all_samples[nearest_idx])
    
    print(f"  [OK] Built TimeRAG candidate pool: {len(representative_candidates)} representatives")
    print(f"       Average cluster size: {len(all_time_series) / pool_size:.1f} samples per cluster\n")
    
    return representative_candidates


def sample_from_timerag_pool_dtw(
    feat_df: pd.DataFrame, cols: Dict,
    n_samples: int, target_sample: Dict, 
    candidates: List[Dict]
) -> Optional[List[Dict]]:
    """
    DTW-based retrieval from TimeRAG candidate pool.
    
    Uses multi-dimensional DTW (dtaidistance.dtw_ndim) for faster computation.
    
    Args:
        feat_df: Feature DataFrame
        cols: Column configuration
        n_samples: Number of samples to retrieve
        target_sample: Target sample dict
        candidates: TimeRAG candidate pool (cluster representatives)
        
    Returns:
        List of selected examples
    """
    if not candidates or len(candidates) < n_samples:
        print(f"Warning: Only {len(candidates)} candidates available")
        return [c['sample'] for c in candidates] if candidates else None
    
    # Get statistical features
    stat_features = _get_statistical_features()
    
    # Extract target time series
    target_ts = _extract_time_series_from_raw_data(
        feat_df, target_sample['user_id'], target_sample['ema_date'], cols,
        window_days=28, stat_features=stat_features
    )
    
    if target_ts is None:
        print("Warning: Could not extract target time series")
        return None
    
    # Compute DTW distances using dtaidistance (multi-dimensional)
    distances = []
    
    for candidate in candidates:
        cand_ts = candidate['time_series']
        
        # Pad sequences to same length
        max_len = max(target_ts.shape[0], cand_ts.shape[0])
        target_padded = np.pad(target_ts, ((0, max_len - target_ts.shape[0]), (0, 0)), 
                               mode='constant', constant_values=0)
        cand_padded = np.pad(cand_ts, ((0, max_len - cand_ts.shape[0]), (0, 0)),
                            mode='constant', constant_values=0)
        
        # Normalize (avoid data leakage - use candidate stats)
        cand_mean = np.nanmean(cand_padded, axis=0, keepdims=True)
        cand_std = np.nanstd(cand_padded, axis=0, keepdims=True)
        cand_std[cand_std < 1e-6] = 1.0  # Avoid division by zero
        
        target_norm = (target_padded - cand_mean) / cand_std
        cand_norm = (cand_padded - cand_mean) / cand_std
        
        # Replace NaN with 0
        target_norm = np.nan_to_num(target_norm, nan=0.0)
        cand_norm = np.nan_to_num(cand_norm, nan=0.0)
        
        # Compute multi-dimensional DTW distance
        try:
            distance = dtw_ndim.distance(target_norm, cand_norm)
        except Exception as e:
            print(f"Warning: DTW calculation failed: {e}")
            distance = 1e6
        
        distances.append(distance)
    
    # Select top-k nearest samples
    distances = np.array(distances)
    top_k_indices = np.argsort(distances)[:n_samples]
    
    selected_examples = [candidates[i]['sample'] for i in top_k_indices]
    
    return selected_examples

