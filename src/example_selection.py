"""
In-Context Learning Example Selection Module

Handles selection of ICL examples based on 4 strategies:
1. Cross-User Random: Random sampling from other users
2. Cross-User Retrieval: DTW-based similarity from other users (TimeRAG)
3. Personal-Recent: Most recent samples from target user
4. Hybrid-Blend: Mix of personal recent + cross-user (retrieval or random)
"""

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw_ndim
from tqdm import tqdm

try:
    from .sensor_transformation import aggregate_window_features, check_missing_ratio, get_user_window_data
    from .timerag_retrieval import (
        build_retrieval_candidate_pool_timerag, sample_from_timerag_pool_dtw,
        build_retrieval_candidate_pool_timerag_ces, sample_from_timerag_pool_dtw_ces
    )
    from . import config
except ImportError:
    from sensor_transformation import aggregate_window_features, check_missing_ratio, get_user_window_data
    from timerag_retrieval import (
        build_retrieval_candidate_pool_timerag, sample_from_timerag_pool_dtw,
        build_retrieval_candidate_pool_timerag_ces, sample_from_timerag_pool_dtw_ces
    )
    import config


def _get_statistical_features() -> Optional[List[str]]:
    """Get statistical features for the current target."""
    if config.DEFAULT_TARGET == 'compass':
        config_path = Path(__file__).parent.parent / 'config' / 'globem_use_cols.json'
        with open(config_path, 'r') as f:
            cols_config = json.load(f)
        return list(cols_config['compass']['feature_set']['statistical'].keys())
    return None


def build_retrieval_candidate_pool(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    max_pool_size: Optional[int] = None, random_state: Optional[int] = None,
    dataset: str = 'globem', target_sample_date: Optional[pd.Timestamp] = None
) -> Optional[List[Dict]]:
    """
    Build candidate pool for DTW retrieval using TimeRAG clustering approach.
    
    This function now uses TimeRAG's clustering-based selection instead of random sampling,
    providing better representation of the data distribution.
    
    Args:
        feat_df: Feature DataFrame
        lab_df: Label DataFrame (trainset only - excluding testset)
        cols: Column configuration
        max_pool_size: Maximum number of candidates (default: config.TIMERAG_POOL_SIZE)
        random_state: Random seed
        dataset: Dataset type ('globem' or 'ces')
        target_sample_date: Target sample date (for CES quarterly chunking)
        
    Returns:
        List of candidate dicts with 'time_series', 'sample', and metadata
    """
    if max_pool_size is None:
        max_pool_size = config.TIMERAG_POOL_SIZE
    
    # Use dataset-specific TimeRAG implementation
    if dataset == 'ces':
        return build_retrieval_candidate_pool_timerag_ces(
            feat_df, lab_df, cols,
            target_sample_date=target_sample_date,
            random_state=random_state,
            min_samples_threshold=config.CES_TIMERAG_MIN_SAMPLES_THRESHOLD,
            min_k=config.CES_TIMERAG_MIN_K,
            max_k_per_chunk=config.CES_TIMERAG_MAX_K_PER_CHUNK,
            max_raw_samples_threshold=config.CES_TIMERAG_MAX_RAW_SAMPLES
        )
    else:
        # GLOBEM
        return build_retrieval_candidate_pool_timerag(
            feat_df, lab_df, cols,
            pool_size=max_pool_size,
            random_state=random_state
        )


def select_icl_examples(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    target_user_id: str, target_ema_date: pd.Timestamp,
    n_shot: int = 4, strategy: str = 'cross_random', use_dtw: bool = False,
    random_state: Optional[int] = None, target_sample: Optional[Dict] = None,
    retrieval_candidates: Optional[List[Dict]] = None, dataset: str = 'globem'
) -> Optional[List[Dict]]:
    """
    Select in-context learning examples based on specified strategy.
    
    Args:
        feat_df: Feature DataFrame
        lab_df: Label DataFrame
        cols: Column configuration
        target_user_id: Target user ID
        target_ema_date: Target EMA date
        n_shot: Number of examples to select
        strategy: 'cross_random', 'cross_retrieval', 'personal_recent', or 'hybrid_blend'
        use_dtw: Whether to use DTW for hybrid (if False, uses random for cross-user part)
        random_state: Random seed for reproducibility
        target_sample: Target sample dict (required for retrieval-based selection)
        retrieval_candidates: Pre-built candidate pool (for cross_retrieval, avoids rebuilding)
        dataset: Dataset type ('globem' or 'ces')
    
    Returns:
        List of example dictionaries, or None if insufficient data
    """
    if n_shot < 1:
        return []
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    examples = []
    
    if strategy == 'cross_random':
        # Cross-User Random: Random sampling from other users (before target date)
        other_users_lab = lab_df[
            (lab_df[cols['user_id']] != target_user_id) & 
            (lab_df[cols['date']] < target_ema_date)
        ]
        
        if len(other_users_lab) < n_shot:
            print(f"Warning: Not enough data from other users (before target date)")
            return None
        
        examples = _sample_cross_random(
            feat_df, other_users_lab, cols, n_shot, random_state, dataset=dataset
        )
    
    elif strategy == 'cross_retrieval':
        # Cross-User Retrieval: DTW-based similarity from precomputed candidates
        if retrieval_candidates is None or len(retrieval_candidates) == 0:
            print(f"Warning: retrieval_candidates required for cross_retrieval strategy")
            return None
        
        if target_sample is None:
            print(f"Warning: target_sample required for retrieval strategy")
            return None
        
        # Filter out target user from candidates AND samples after target date
        other_user_candidates = [
            c for c in retrieval_candidates 
            if c['user_id'] != target_user_id and c['ema_date'] < target_ema_date
        ]
        
        if len(other_user_candidates) < n_shot:
            print(f"Warning: Not enough candidates from other users (before target date)")
            return None
        
        # Use dataset-specific DTW sampling
        if dataset == 'ces':
            examples = sample_from_timerag_pool_dtw_ces(
                feat_df, cols, n_shot, target_sample, other_user_candidates,
                diversity_factor=config.RETRIEVAL_DIVERSITY_FACTOR
            )
        else:
            examples = sample_from_timerag_pool_dtw(
                feat_df, cols, n_shot, target_sample, other_user_candidates,
                diversity_factor=config.RETRIEVAL_DIVERSITY_FACTOR
            )
    
    elif strategy == 'personal_recent':
        # Personal-Recent: Most recent samples from target user
        personal_lab = lab_df[
            (lab_df[cols['user_id']] == target_user_id) & 
            (lab_df[cols['date']] < target_ema_date)
        ]
        
        if len(personal_lab) < n_shot:
            print(f"Warning: Not enough historical data for user {target_user_id}")
            return None
        
        examples = _sample_personal_recent(
            feat_df, personal_lab, cols, n_shot, target_ema_date, dataset=dataset
        )
    
    elif strategy == 'hybrid_blend':
        # Hybrid-Blend: Mix personal recent + cross-user
        n_personal = n_shot // 2
        n_cross = n_shot - n_personal
        
        # Get personal recent samples
        personal_lab = lab_df[
            (lab_df[cols['user_id']] == target_user_id) & 
            (lab_df[cols['date']] < target_ema_date)
        ]
        
        if len(personal_lab) < n_personal:
            print(f"Warning: Not enough personal historical data for user {target_user_id}")
            return None
        
        personal_examples = _sample_personal_recent(
            feat_df, personal_lab, cols, n_personal, target_ema_date, dataset=dataset
        )
        
        # Get cross-user samples (DTW or random based on use_dtw flag)
        other_users_lab = lab_df[
            (lab_df[cols['user_id']] != target_user_id) & 
            (lab_df[cols['date']] < target_ema_date)
        ]
        
        if len(other_users_lab) < n_cross:
            print(f"Warning: Not enough cross-user data (before target date)")
            return None
        
        if use_dtw and target_sample is not None and retrieval_candidates is not None:
            # Use prebuilt pool for DTW (filter by user and date)
            other_user_candidates = [
                c for c in retrieval_candidates 
                if c['user_id'] != target_user_id and c['ema_date'] < target_ema_date
            ]
            # Use dataset-specific DTW sampling
            if dataset == 'ces':
                cross_examples = sample_from_timerag_pool_dtw_ces(
                    feat_df, cols, n_cross, target_sample, other_user_candidates,
                    diversity_factor=config.RETRIEVAL_DIVERSITY_FACTOR
                )
            else:
                cross_examples = sample_from_timerag_pool_dtw(
                    feat_df, cols, n_cross, target_sample, other_user_candidates,
                    diversity_factor=config.RETRIEVAL_DIVERSITY_FACTOR
                )
        else:
            cross_examples = _sample_cross_random(
                feat_df, other_users_lab, cols, n_cross, random_state, dataset=dataset
            )
        
        if personal_examples is None or cross_examples is None:
            return None
        
        examples = personal_examples + cross_examples
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return examples


def _sample_cross_random(
    feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict,
    n_samples: int, random_state: Optional[int] = None, dataset: str = 'globem'
) -> Optional[List[Dict]]:
    """
    Random sampling from other users with stratified label distribution.
    
    Ensures balanced representation of labels.
    For GLOBEM: anxiety/depression (0_0, 0_1, 1_0, 1_1)
    For CES: anxiety/depression/stress (0_0_0, 0_0_1, ..., 1_1_1)
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()
    
    # Create stratification key from anxiety and depression labels
    label_cols = cols['labels']
    lab_pool_copy = lab_pool.copy()
    
    # Combine anxiety and depression labels: e.g., "0_0", "0_1", "1_0", "1_1"
    if len(label_cols) >= 2:
        # Assume first two labels are anxiety and depression
        strat_key = lab_pool_copy[label_cols[0]].astype(str) + "_" + lab_pool_copy[label_cols[1]].astype(str)
        lab_pool_copy['_strat_key'] = strat_key
        
        # Calculate samples per stratification group
        class_counts = lab_pool_copy['_strat_key'].value_counts()
        classes = class_counts.index.tolist()
        
        # Distribute samples proportionally, but ensure each class gets at least 1 if possible
        samples_per_class = {}
        if n_samples >= len(classes):
            # Give at least 1 to each class, then distribute remaining proportionally
            base_per_class = n_samples // len(classes)
            remainder = n_samples % len(classes)
            
            for cls in classes:
                samples_per_class[cls] = base_per_class
            
            # Distribute remainder to most frequent classes
            for i, cls in enumerate(class_counts.index[:remainder]):
                samples_per_class[cls] += 1
        else:
            # If n_samples < number of classes, sample from largest classes
            largest_classes = class_counts.nlargest(n_samples).index.tolist()
            for cls in largest_classes:
                samples_per_class[cls] = 1
    else:
        # Fallback: no stratification
        samples_per_class = {'all': n_samples}
        lab_pool_copy['_strat_key'] = 'all'
    
    # Sample from each stratification group
    examples = []
    max_attempts_per_class = 50
    
    for cls, n_class_samples in samples_per_class.items():
        if n_class_samples <= 0:
            continue
        
        cls_indices = lab_pool_copy[lab_pool_copy['_strat_key'] == cls].index.tolist()
        
        if len(cls_indices) == 0:
            continue
        
        # Shuffle and try to collect valid samples
        rng.shuffle(cls_indices)
        collected_for_class = 0
        attempts = 0
        
        for idx in cls_indices:
            if collected_for_class >= n_class_samples:
                break
            if attempts >= max_attempts_per_class:
                break
            
            row = lab_pool.loc[idx]
            user_id = row[cols['user_id']]
            ema_date = row[cols['date']]
            
            # Aggregate features (different logic for CES vs GLOBEM)
            if dataset == 'ces':
                # For CES, aggregated_features is already in feat_df
                feat_row = feat_df[
                    (feat_df[cols['user_id']] == user_id) & 
                    (feat_df[cols['date']] == ema_date)
                ]
                if len(feat_row) == 0:
                    attempts += 1
                    continue
                agg_feats = feat_row.iloc[0].to_dict()
            else:
                # For GLOBEM, compute on-the-fly
                agg_feats = aggregate_window_features(
                    feat_df, user_id, ema_date, cols,
                    window_days=config.AGGREGATION_WINDOW_DAYS,
                    mode=config.DEFAULT_AGGREGATION_MODE,
                    use_immediate_window=config.USE_IMMEDIATE_WINDOW,
                    immediate_window_days=config.IMMEDIATE_WINDOW_DAYS,
                    adaptive_window=config.USE_ADAPTIVE_WINDOW
                )
            
            if agg_feats is not None and (dataset == 'ces' or check_missing_ratio(agg_feats)):
                labels = row[cols['labels']].to_dict()
                examples.append({
                    'aggregated_features': agg_feats,
                    'labels': labels,
                    'user_id': user_id,
                    'ema_date': ema_date,
                    '_label_strat': cls  # For debugging
                })
                collected_for_class += 1
            
            attempts += 1
    
    # Shuffle final examples to mix label groups
    if examples:
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        examples = [examples[i] for i in indices]
    
    # Only warn if we got less than 75% of requested samples
    if len(examples) < n_samples * 0.75:
        print(f"Warning: Only collected {len(examples)}/{n_samples} valid examples (stratified)")
    
    return examples if examples else None


def _sample_from_prebuilt_pool_dtw(
    feat_df: pd.DataFrame, cols: Dict,
    n_samples: int, target_sample: Dict, 
    candidates: List[Dict]
) -> Optional[List[Dict]]:
    """
    DTW-based retrieval from prebuilt candidate pool.
    
    Uses multi-dimensional DTW (dtaidistance.dtw_ndim) for faster computation.
    
    Args:
        feat_df: Feature DataFrame
        cols: Column configuration
        n_samples: Number of samples to retrieve
        target_sample: Target sample dict
        candidates: Prebuilt candidate pool with time_series already extracted
        
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


def _sample_cross_retrieval(
    feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict,
    n_samples: int, target_sample: Dict, random_state: Optional[int] = None,
    max_pool_size: int = 50  # Reduced from 500 for faster computation
) -> Optional[List[Dict]]:
    """
    [DEPRECATED] DTW-based retrieval from other users.
    Use build_retrieval_candidate_pool() + _sample_from_prebuilt_pool_dtw() instead.
    
    Uses DTW distance on actual 28-day time series for each feature.
    """
    t0 = time.time()
    
    # Build candidate pool with actual time series
    print(f"  [DTW] Building candidate pool (max_size={max_pool_size})...", end=" ", flush=True)
    candidates = _build_candidate_pool_dtw(
        feat_df, lab_pool, cols, max_pool_size, random_state
    )
    print(f"Done. {len(candidates) if candidates else 0} candidates in {time.time()-t0:.2f}s")
    
    if not candidates:
        return None
    
    if len(candidates) < n_samples:
        print(f"Warning: Only {len(candidates)} valid candidates for retrieval")
        return [c['sample'] for c in candidates]
    
    # Extract target time series from raw data (actual 28-day time series)
    t1 = time.time()
    
    # Get statistical features for compass mode
    stat_features = _get_statistical_features()
    
    target_ts = _extract_time_series_from_raw_data(
        feat_df, target_sample['user_id'], target_sample['ema_date'], cols,
        window_days=28, stat_features=stat_features
    )
    
    if target_ts is None:
        print("Warning: Could not extract target time series")
        return None
    
    # print(f"  [DTW] Target time series shape: {target_ts.shape}")
    # print(f"  [DTW] Computing DTW distances for {len(candidates)} candidates x {target_ts.shape[1]} features...")
    
    # Compute DTW distances
    distances = []
    dtw_times = []
    
    for candidate in candidates:
        cand_ts = candidate['time_series']
        
        # Compute DTW for each feature independently and sum
        total_distance = 0.0
        valid_features = 0
        
        t_dtw = time.time()
        for feat_idx in range(min(target_ts.shape[1], cand_ts.shape[1])):
            target_feat = target_ts[:, feat_idx]
            cand_feat = cand_ts[:, feat_idx]
            
            # Check if feature has any variation
            cand_std = np.nanstd(cand_feat)
            target_std = np.nanstd(target_feat)
            
            # Skip if both are constant (std=0)
            if cand_std < 1e-6 and target_std < 1e-6:
                continue
            
            # Normalize both series
            # Use candidate's stats for normalization to avoid data leakage
            cand_mean = np.nanmean(cand_feat)
            
            if cand_std > 1e-6:
                target_feat_norm = (target_feat - cand_mean) / cand_std
                cand_feat_norm = (cand_feat - cand_mean) / cand_std
            else:
                # If candidate has no variation, just center
                target_feat_norm = target_feat - cand_mean
                cand_feat_norm = cand_feat - cand_mean
            
            # Replace NaN with 0
            target_feat_norm = np.nan_to_num(target_feat_norm, nan=0.0)
            cand_feat_norm = np.nan_to_num(cand_feat_norm, nan=0.0)
            
            # Reshape to ensure 1-D arrays for fastdtw
            target_feat_norm = target_feat_norm.reshape(-1)
            cand_feat_norm = cand_feat_norm.reshape(-1)
            
            # Compute DTW distance
            # Use radius=1 for faster approximation (allows warping of +/- 1 day)
            distance, _ = fastdtw(target_feat_norm, cand_feat_norm, dist=1)
            total_distance += distance
            valid_features += 1
        
        dtw_times.append(time.time() - t_dtw)
        
        # Average distance across features
        if valid_features > 0:
            avg_distance = total_distance / valid_features
        else:
            # If no valid features, assign a large but not infinite distance
            avg_distance = 1e6
        
        distances.append(avg_distance)
    
    avg_dtw_time = np.mean(dtw_times)
    print(f"  [DTW] Completed in {time.time()-t1:.2f}s (avg {avg_dtw_time:.3f}s per candidate)")
    
    # Select top-k nearest samples
    distances = np.array(distances)
    top_k_indices = np.argsort(distances)[:n_samples]
    
    selected_examples = [candidates[i]['sample'] for i in top_k_indices]
    
    print(f"  [DTW] Total time: {time.time()-t0:.2f}s")
    
    return selected_examples


def _sample_personal_recent(
    feat_df: pd.DataFrame, personal_lab: pd.DataFrame, cols: Dict,
    n_samples: int, target_date: pd.Timestamp, dataset: str = 'globem'
) -> Optional[List[Dict]]:
    """Select most recent samples from personal history."""
    # Sort by date descending
    sorted_lab = personal_lab.sort_values(by=cols['date'], ascending=False)
    
    # Need to search more to account for samples with missing data
    max_search = min(len(sorted_lab), n_samples * 3)  # Search up to 3x to find valid samples
    
    examples = []
    for idx in sorted_lab.head(max_search).index:
        row = sorted_lab.loc[idx]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # Aggregate features (different logic for CES vs GLOBEM)
        if dataset == 'ces':
            # For CES, aggregated_features is already in feat_df
            feat_row = feat_df[
                (feat_df[cols['user_id']] == user_id) & 
                (feat_df[cols['date']] == ema_date)
            ]
            if len(feat_row) == 0:
                continue
            agg_feats = feat_row.iloc[0].to_dict()
        else:
            # For GLOBEM, compute on-the-fly
            agg_feats = aggregate_window_features(
                feat_df, user_id, ema_date, cols,
                window_days=config.AGGREGATION_WINDOW_DAYS,
                mode=config.DEFAULT_AGGREGATION_MODE,
                use_immediate_window=config.USE_IMMEDIATE_WINDOW,
                immediate_window_days=config.IMMEDIATE_WINDOW_DAYS,
                adaptive_window=config.USE_ADAPTIVE_WINDOW
            )
        
        if agg_feats is not None and (dataset == 'ces' or check_missing_ratio(agg_feats)):
            labels = row[cols['labels']].to_dict()
            examples.append({
                'aggregated_features': agg_feats,
                'labels': labels,
                'user_id': user_id,
                'ema_date': ema_date
            })
        
        if len(examples) >= n_samples:
            break
    
    # Only warn if we got less than 75% of requested samples
    if len(examples) < n_samples * 0.75:
        print(f"Warning: Only collected {len(examples)}/{n_samples} recent personal examples")
    
    return examples if examples else None


def _build_candidate_pool_dtw(
    feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict,
    max_pool_size: int = 500, random_state: Optional[int] = None
) -> List[Dict]:
    """
    Build candidate pool with time series for DTW calculation.
    
    Returns list of dicts with 'time_series' (numpy array) and 'sample' (dict).
    """
    # Limit pool size for efficiency
    if len(lab_pool) > max_pool_size:
        if random_state is not None:
            lab_pool = lab_pool.sample(n=max_pool_size, random_state=random_state)
        else:
            lab_pool = lab_pool.sample(n=max_pool_size)
    
    candidates = []
    
    # Get statistical features for compass mode
    stat_features = _get_statistical_features()
    
    for idx in lab_pool.index:
        row = lab_pool.loc[idx]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # **OPTIMIZATION**: Extract window data only ONCE and reuse it
        window_data = get_user_window_data(feat_df, user_id, ema_date, cols, window_days=28)
        
        if window_data is None or len(window_data) < 14:  # Need at least 50% of 28 days
            continue
        
        # Extract time series from window data (no additional filtering)
        time_series = _extract_time_series_from_window_data(window_data, stat_features)
        
        # Check if valid time series
        if time_series is not None and time_series.shape[0] > 0:
            # Compute aggregated features using the SAME precomputed window data
            # This avoids redundant filtering!
            agg_feats = aggregate_window_features(
                feat_df, user_id, ema_date, cols,
                window_days=28,
                mode='statistics',
                use_immediate_window=False,
                immediate_window_days=0,
                adaptive_window=False,
                precomputed_window_data=window_data  # <-- KEY OPTIMIZATION!
            )
            
            if agg_feats is not None:
                labels = row[cols['labels']].to_dict()
                sample = {
                    'aggregated_features': agg_feats,
                    'labels': labels,
                    'user_id': user_id,
                    'ema_date': ema_date
                }
                
                candidates.append({
                    'time_series': time_series,
                    'sample': sample
                })
    
    return candidates


def _extract_time_series_from_window_data(
    window_data: pd.DataFrame, stat_features: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Extract time series from already-filtered window data.
    
    Args:
        window_data: Already filtered and sorted DataFrame (from get_user_window_data)
        stat_features: List of statistical feature columns to extract
        
    Returns:
        numpy array of shape (n_days, n_features) or None if insufficient data
    """
    if stat_features is None or window_data is None or len(window_data) == 0:
        return None
    
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
    feat_df: pd.DataFrame, user_id: str, ema_date: pd.Timestamp, cols: Dict,
    window_days: int = 28, stat_features: Optional[List[str]] = None
) -> Optional[np.ndarray]:
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


def _extract_time_series(agg_feats: Dict, cols: Dict) -> Optional[np.ndarray]:
    """
    Extract 28-day time series for statistical features.
    
    Returns:
        numpy array of shape (28, n_features) or None if not available
    """
    mode = agg_feats.get('aggregation_mode', 'unknown')
    
    if mode == 'statistics':
        # For statistics mode, we need to extract the 28-day raw values
        # However, statistics mode only has aggregated statistics (mean, std, etc.)
        # We need the raw time series, so this won't work directly
        
        # For now, we'll use the statistical features themselves
        # This is a simplified approach - ideally we'd store the raw time series
        features_dict = agg_feats.get('features', {})
        
        if not features_dict:
            return None
        
        # Extract statistical values and create a pseudo time series
        # by repeating the mean value 28 times (not ideal but workable)
        feature_values = []
        for feat_name, feat_stats in features_dict.items():
            if isinstance(feat_stats, dict):
                mean_val = feat_stats.get('mean', 0.0)
                mean_val = mean_val if mean_val is not None else 0.0
                feature_values.append(mean_val)
            elif isinstance(feat_stats, (int, float)):
                feature_values.append(feat_stats if feat_stats is not None else 0.0)
        
        if not feature_values:
            return None
        
        # Create a 28 x n_features array by repeating the mean
        # This is a simplified approach - in reality we'd want actual time series
        time_series = np.tile(feature_values, (28, 1))
        
        return time_series
    
    elif mode == 'compass':
        # For compass mode, extract statistical features
        stat_feats = agg_feats.get('statistical_features', {})
        
        if not stat_feats:
            return None
        
        feature_values = []
        for feat_name, stats in stat_feats.items():
            mean_val = stats.get('mean', 0.0)
            mean_val = mean_val if mean_val is not None else 0.0
            feature_values.append(mean_val)
        
        if not feature_values:
            return None
        
        # Create a 28 x n_features array
        time_series = np.tile(feature_values, (28, 1))
        
        return time_series
    
    else:
        return None
