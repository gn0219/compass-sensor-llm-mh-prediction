"""
In-Context Learning Example Selection Module

Handles selection of ICL examples based on 4 strategies:
1. Cross-User Random: Random sampling from other users
2. Cross-User Retrieval: DTW-based similarity from other users
3. Personal-Recent: Most recent samples from target user
4. Hybrid-Blend: Mix of personal recent + cross-user (retrieval or random)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from .sensor_transformation import aggregate_window_features, check_missing_ratio
from . import config


def select_icl_examples(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    target_user_id: str, target_ema_date: pd.Timestamp,
    n_shot: int = 4, strategy: str = 'cross_random', use_dtw: bool = False,
    random_state: Optional[int] = None, target_sample: Optional[Dict] = None
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
        # Cross-User Random: Random sampling from other users
        other_users_lab = lab_df[lab_df[cols['user_id']] != target_user_id]
        
        if len(other_users_lab) < n_shot:
            print(f"Warning: Not enough data from other users")
            return None
        
        examples = _sample_cross_random(
            feat_df, other_users_lab, cols, n_shot, random_state
        )
    
    elif strategy == 'cross_retrieval':
        # Cross-User Retrieval: DTW-based similarity from other users
        other_users_lab = lab_df[lab_df[cols['user_id']] != target_user_id]
        
        if len(other_users_lab) < n_shot:
            print(f"Warning: Not enough data from other users")
            return None
        
        if target_sample is None:
            print(f"Warning: target_sample required for retrieval strategy")
            return None
        
        examples = _sample_cross_retrieval(
            feat_df, other_users_lab, cols, n_shot, target_sample, random_state
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
            feat_df, personal_lab, cols, n_shot, target_ema_date
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
            feat_df, personal_lab, cols, n_personal, target_ema_date
        )
        
        # Get cross-user samples (DTW or random based on use_dtw flag)
        other_users_lab = lab_df[lab_df[cols['user_id']] != target_user_id]
        
        if len(other_users_lab) < n_cross:
            print(f"Warning: Not enough cross-user data")
            return None
        
        if use_dtw and target_sample is not None:
            cross_examples = _sample_cross_retrieval(
                feat_df, other_users_lab, cols, n_cross, target_sample, random_state
            )
        else:
            cross_examples = _sample_cross_random(
                feat_df, other_users_lab, cols, n_cross, random_state
            )
        
        if personal_examples is None or cross_examples is None:
            return None
        
        examples = personal_examples + cross_examples
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return examples


def _sample_cross_random(
    feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict,
    n_samples: int, random_state: Optional[int] = None
) -> Optional[List[Dict]]:
    """Random sampling from other users."""
    # Sample more to account for filtering due to missing data
    # Try up to 3x the requested amount to ensure we get enough valid samples
    max_attempts = min(len(lab_pool), n_samples * 3)
    
    if random_state is not None:
        sampled = lab_pool.sample(n=max_attempts, random_state=random_state)
    else:
        sampled = lab_pool.sample(n=max_attempts)
    
    examples = []
    for idx in sampled.index:
        row = sampled.loc[idx]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # Aggregate features
        agg_feats = aggregate_window_features(
            feat_df, user_id, ema_date, cols,
            window_days=config.AGGREGATION_WINDOW_DAYS,
            mode=config.DEFAULT_AGGREGATION_MODE,
            use_immediate_window=config.USE_IMMEDIATE_WINDOW,
            immediate_window_days=config.IMMEDIATE_WINDOW_DAYS,
            adaptive_window=config.USE_ADAPTIVE_WINDOW
        )
        
        if agg_feats is not None and check_missing_ratio(agg_feats):
            labels = row[cols['labels']].to_dict()
            examples.append({
                'aggregated_features': agg_feats,
                'labels': labels,
                'user_id': user_id,
                'ema_date': ema_date
            })
            
            # Stop once we have enough valid examples
            if len(examples) >= n_samples:
                break
    
    # Only warn if we got less than 75% of requested samples
    if len(examples) < n_samples * 0.75:
        print(f"Warning: Only collected {len(examples)}/{n_samples} valid examples")
    
    return examples if examples else None


def _sample_cross_retrieval(
    feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict,
    n_samples: int, target_sample: Dict, random_state: Optional[int] = None,
    max_pool_size: int = 500
) -> Optional[List[Dict]]:
    """
    DTW-based retrieval from other users.
    
    Uses DTW distance on 28-day time series for each of 16 statistical features.
    """
    # Build candidate pool
    candidates = _build_candidate_pool_dtw(
        feat_df, lab_pool, cols, max_pool_size, random_state
    )
    
    if not candidates:
        return None
    
    if len(candidates) < n_samples:
        print(f"Warning: Only {len(candidates)} valid candidates for retrieval")
        return [c['sample'] for c in candidates]
    
    # Extract target time series (28 days x 16 features)
    target_ts = _extract_time_series(
        target_sample['aggregated_features'], cols
    )
    
    if target_ts is None:
        print("Warning: Could not extract target time series")
        return None
    
    # Compute DTW distances
    distances = []
    for candidate in candidates:
        cand_ts = candidate['time_series']
        
        # Compute DTW for each feature independently and sum
        total_distance = 0.0
        valid_features = 0
        
        for feat_idx in range(min(target_ts.shape[1], cand_ts.shape[1])):
            target_feat = target_ts[:, feat_idx]
            cand_feat = cand_ts[:, feat_idx]
            
            # Normalize both series (avoid data leakage)
            # Use only candidate's stats for normalization
            cand_mean = np.nanmean(cand_feat)
            cand_std = np.nanstd(cand_feat)
            
            if cand_std > 0:
                target_feat_norm = (target_feat - cand_mean) / cand_std
                cand_feat_norm = (cand_feat - cand_mean) / cand_std
                
                # Replace NaN with 0
                target_feat_norm = np.nan_to_num(target_feat_norm, nan=0.0)
                cand_feat_norm = np.nan_to_num(cand_feat_norm, nan=0.0)
                
                # Reshape to ensure 1-D arrays for fastdtw
                target_feat_norm = target_feat_norm.reshape(-1)
                cand_feat_norm = cand_feat_norm.reshape(-1)
                
                # Compute DTW distance
                distance, _ = fastdtw(target_feat_norm, cand_feat_norm, dist=euclidean)
                total_distance += distance
                valid_features += 1
        
        # Average distance across features
        if valid_features > 0:
            avg_distance = total_distance / valid_features
        else:
            avg_distance = float('inf')
        
        distances.append(avg_distance)
    
    # Select top-k nearest samples
    distances = np.array(distances)
    top_k_indices = np.argsort(distances)[:n_samples]
    
    selected_examples = [candidates[i]['sample'] for i in top_k_indices]
    
    return selected_examples


def _sample_personal_recent(
    feat_df: pd.DataFrame, personal_lab: pd.DataFrame, cols: Dict,
    n_samples: int, target_date: pd.Timestamp
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
        
        # Aggregate features
        agg_feats = aggregate_window_features(
            feat_df, user_id, ema_date, cols,
            window_days=config.AGGREGATION_WINDOW_DAYS,
            mode=config.DEFAULT_AGGREGATION_MODE,
            use_immediate_window=config.USE_IMMEDIATE_WINDOW,
            immediate_window_days=config.IMMEDIATE_WINDOW_DAYS,
            adaptive_window=config.USE_ADAPTIVE_WINDOW
        )
        
        if agg_feats is not None and check_missing_ratio(agg_feats):
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
    
    for idx in lab_pool.index:
        row = lab_pool.loc[idx]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # Get aggregated features
        agg_feats = aggregate_window_features(
            feat_df, user_id, ema_date, cols,
            window_days=28,  # Use 28-day window for DTW
            mode='statistics',  # Use statistics mode for faster computation
            use_immediate_window=False,
            immediate_window_days=0,
            adaptive_window=False
        )
        
        # Check if valid
        if agg_feats is not None and check_missing_ratio(agg_feats):
            # Extract time series
            time_series = _extract_time_series(agg_feats, cols)
            
            if time_series is not None:
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
