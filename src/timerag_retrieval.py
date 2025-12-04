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
    from .sensor_transformation import (
        aggregate_window_features, get_user_window_data,
        _get_statistical_features, _extract_time_series_from_window_data, _extract_time_series_from_raw_data
    )
    from . import config
    # Import helper functions to avoid circular dependency
    import importlib
    _example_selection = None
except ImportError:
    from sensor_transformation import (
        aggregate_window_features, get_user_window_data,
        _get_statistical_features, _extract_time_series_from_window_data, _extract_time_series_from_raw_data
    )
    import config
    _example_selection = None


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
    stat_features = _get_statistical_features(cols)
    
    if stat_features is None or len(stat_features) == 0:
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
    candidates: List[Dict],
    diversity_factor: float = 2.0
) -> Optional[List[Dict]]:
    """
    DTW-based retrieval from TimeRAG candidate pool with label diversity.
    
    Strategy:
    1. Retrieve top-(n_samples * diversity_factor) most similar candidates by DTW
    2. Among these, select n_samples with balanced label distribution
    3. This ensures both similarity and label diversity
    
    Args:
        feat_df: Feature DataFrame
        cols: Column configuration
        n_samples: Number of samples to retrieve
        target_sample: Target sample dict
        candidates: TimeRAG candidate pool (cluster representatives)
        diversity_factor: Multiplier for initial retrieval (default 2.0)
        
    Returns:
        List of selected examples with balanced labels
    """
    if not candidates or len(candidates) < n_samples:
        print(f"Warning: Only {len(candidates)} candidates available")
        return [c['sample'] for c in candidates] if candidates else None
    
    # Get statistical features
    stat_features = _get_statistical_features(cols)
    
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
    
    # Step 1: Select top-(n_samples * diversity_factor) most similar candidates
    distances = np.array(distances)
    n_retrieve = min(len(candidates), int(n_samples * diversity_factor))
    top_k_indices = np.argsort(distances)[:n_retrieve]
    
    # Extract label information from retrieved candidates
    retrieved_samples = []
    for idx in top_k_indices:
        sample = candidates[idx]['sample']
        labels = sample['labels']
        
        # Extract anxiety and depression labels (assume these are first two)
        label_cols = cols['labels']
        if len(label_cols) >= 2:
            anx_label = str(int(labels.get(label_cols[0], 0)))
            dep_label = str(int(labels.get(label_cols[1], 0)))
            label_key = f"{anx_label}_{dep_label}"
        else:
            label_key = "unknown"
        
        retrieved_samples.append({
            'sample': sample,
            'distance': distances[idx],
            'label_key': label_key
        })
    
    # Step 2: Select n_samples with balanced label distribution
    # Group by label
    from collections import defaultdict
    label_groups = defaultdict(list)
    for item in retrieved_samples:
        label_groups[item['label_key']].append(item)
    
    # Determine how many samples to take from each group
    n_groups = len(label_groups)
    if n_groups == 0:
        return None
    
    # Strategy: Give each group roughly equal representation
    samples_per_group = {}
    if n_samples >= n_groups:
        base_per_group = n_samples // n_groups
        remainder = n_samples % n_groups
        
        # Sort groups by size (descending) for stable allocation
        sorted_groups = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]), reverse=True)
        
        for label_key in sorted_groups:
            samples_per_group[label_key] = base_per_group
        
        # Distribute remainder to groups with most samples available
        for i in range(remainder):
            samples_per_group[sorted_groups[i]] += 1
    else:
        # If n_samples < n_groups, select from largest groups only
        sorted_groups = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]), reverse=True)
        for i in range(n_samples):
            label_key = sorted_groups[i % n_groups]
            samples_per_group[label_key] = samples_per_group.get(label_key, 0) + 1
    
    # Select samples from each group (prioritize by DTW distance)
    selected_examples = []
    for label_key, n_needed in samples_per_group.items():
        group_items = label_groups[label_key]
        
        # Sort by distance (most similar first)
        group_items_sorted = sorted(group_items, key=lambda x: x['distance'])
        
        # Take top n_needed from this group
        for item in group_items_sorted[:n_needed]:
            selected_examples.append(item['sample'])
    
    # Shuffle to mix label groups
    np.random.shuffle(selected_examples)
    
    return selected_examples


# ============================================================================
# CES-SPECIFIC TIMERAG RETRIEVAL WITH QUARTERLY CHUNKING
# ============================================================================

def build_retrieval_candidate_pool_timerag_ces(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    target_sample_date: Optional[pd.Timestamp] = None,
    random_state: Optional[int] = None,
    min_samples_threshold: int = 20,
    min_k: int = 5,
    max_k_per_chunk: int = 30,
    max_raw_samples_threshold: int = 100
) -> Optional[List[Dict]]:
    """
    Build candidate pool for CES using TimeRAG's quarterly chunking approach.
    
    Strategy (as requested by user):
    1. Offline: Pre-build representative pools for each quarter
       - If quarter has < min_samples_threshold samples: keep all samples
       - Else: K-means clustering with adaptive K = max(min_k, min(sqrt(N), max_k))
       
    2. Online (when target arrives):
       - Pool 1: All representatives from quarters BEFORE target's quarter
       - Pool 2: Raw samples from target's quarter BEFORE target's date
                 (if > max_raw_samples_threshold, cluster them too)
       - Final pool: Pool 1 + Pool 2
       
    3. DTW ranking with user deduplication
    
    Args:
        feat_df: Feature DataFrame (CES aggregated data)
        lab_df: Label DataFrame (trainset only - excluding testset)
        cols: Column configuration
        target_sample_date: Target sample date (for online filtering)
        random_state: Random seed
        min_samples_threshold: Minimum samples to trigger clustering (default: 20)
        min_k: Minimum K for clustering (default: 5)
        max_k_per_chunk: Maximum K per chunk (default: 30)
        max_raw_samples_threshold: Max raw samples in current quarter before clustering (default: 100)
        
    Returns:
        List of candidate dicts with 'time_series', 'sample', and metadata
    """
    print("\n[Building TimeRAG Candidate Pool for CES (Quarterly Chunking)]")
    print(f"  Method: Quarterly chunking + adaptive K-means clustering")
    print(f"  Trainset size: {len(lab_df)} samples")
    if target_sample_date:
        print(f"  Target date: {target_sample_date.strftime('%Y-%m-%d')}")
    print(f"  Parameters: min_samples={min_samples_threshold}, min_k={min_k}, max_k={max_k_per_chunk}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Step 1: Divide data into quarterly chunks
    print(f"\n  [Step 1] Dividing data into quarterly chunks...")
    
    # Get date range
    all_dates = lab_df[cols['date']]
    min_date = all_dates.min()
    max_date = all_dates.max()
    
    print(f"    Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Create quarterly periods
    quarters = pd.period_range(start=min_date, end=max_date, freq='Q')
    print(f"    Number of quarters: {len(quarters)}")
    
    # Build representative pools for each quarter (OFFLINE)
    quarterly_pools = {}
    
    for quarter in quarters:
        quarter_start = quarter.start_time
        quarter_end = quarter.end_time
        
        # Get samples in this quarter
        quarter_mask = (lab_df[cols['date']] >= quarter_start) & (lab_df[cols['date']] <= quarter_end)
        quarter_lab = lab_df[quarter_mask]
        
        if len(quarter_lab) == 0:
            continue
        
        n_samples = len(quarter_lab)
        print(f"\n    Quarter {quarter}: {n_samples} samples")
        
        # Apply adaptive K logic
        if n_samples < min_samples_threshold:
            print(f"      -> Skip clustering (< {min_samples_threshold}), keep all {n_samples} samples")
            # Keep all samples as representatives
            quarter_representatives = _extract_ces_samples_as_candidates(
                feat_df, quarter_lab, cols
            )
            quarterly_pools[quarter] = quarter_representatives
        else:
            # Apply clustering
            k_t = max(min_k, min(int(np.sqrt(n_samples)), max_k_per_chunk))
            print(f"      -> Clustering into K={k_t} representatives")
            
            quarter_representatives = _cluster_ces_quarterly_chunk(
                feat_df, quarter_lab, cols, k_t, random_state
            )
            quarterly_pools[quarter] = quarter_representatives
    
    # Step 2: Combine pools based on target date (ONLINE)
    if target_sample_date is None:
        # No target: return all representatives
        print(f"\n  [No target date] Returning all quarterly representatives")
        all_candidates = []
        for quarter, candidates in quarterly_pools.items():
            all_candidates.extend(candidates)
        print(f"  [OK] Total candidates: {len(all_candidates)}\n")
        return all_candidates
    
    print(f"\n  [Step 2] Combining pools for target date {target_sample_date.strftime('%Y-%m-%d')}")
    
    # Determine target's quarter
    target_quarter = pd.Period(target_sample_date, freq='Q')
    print(f"    Target quarter: {target_quarter}")
    
    # Pool 1: All representatives from BEFORE target's quarter
    pool_representatives = []
    for quarter, candidates in quarterly_pools.items():
        if quarter < target_quarter:
            pool_representatives.extend(candidates)
    
    print(f"    Pool 1 (past quarters): {len(pool_representatives)} representatives")
    
    # Pool 2: Raw samples from target's quarter BEFORE target's date
    quarter_start = target_quarter.start_time
    quarter_end = min(target_sample_date, target_quarter.end_time)
    
    current_quarter_mask = (
        (lab_df[cols['date']] >= quarter_start) & 
        (lab_df[cols['date']] < target_sample_date)
    )
    current_quarter_lab = lab_df[current_quarter_mask]
    
    print(f"    Pool 2 (current quarter, before target): {len(current_quarter_lab)} samples")
    
    # Pool 2 compression
    if len(current_quarter_lab) > max_raw_samples_threshold:
        print(f"      -> Clustering current quarter (> {max_raw_samples_threshold} samples)")
        k_current = max(min_k, min(int(np.sqrt(len(current_quarter_lab))), max_k_per_chunk))
        print(f"         K={k_current}")
        pool_current = _cluster_ces_quarterly_chunk(
            feat_df, current_quarter_lab, cols, k_current, random_state
        )
    else:
        print(f"      -> Using all raw samples (≤ {max_raw_samples_threshold})")
        pool_current = _extract_ces_samples_as_candidates(
            feat_df, current_quarter_lab, cols
        )
    
    # Combine pools
    final_pool = pool_representatives + pool_current
    
    print(f"\n  [OK] Final candidate pool: {len(final_pool)} samples")
    print(f"       - From past quarters: {len(pool_representatives)}")
    print(f"       - From current quarter: {len(pool_current)}\n")
    
    return final_pool


def _extract_ces_samples_as_candidates(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    global_valid_features: Optional[List[str]] = None
) -> List[Dict]:
    """
    Extract CES samples as candidate pool (without clustering).
    
    For CES, time series are extracted from before1day~before28day columns.
    Uses globally valid features (common across all samples) to ensure consistent shape.
    
    Args:
        global_valid_features: Pre-determined global valid features. If None, will compute.
    """
    candidates = []
    
    # Get statistical feature columns
    stat_features = list(cols['feature_set']['statistical'].keys())
    
    # Step 1: Determine globally valid features (present in ALL samples)
    # This ensures consistent feature dimensions across iOS and Android users
    if global_valid_features is None:
        print("  [Determining globally valid features across all samples...]")
        global_valid_feats_set = None
        
        sample_count = 0
        for idx in lab_df.index:
            row = lab_df.loc[idx]
            user_id = row[cols['user_id']]
            ema_date = row[cols['date']]
            
            feat_row = feat_df[
                (feat_df[cols['user_id']] == user_id) & 
                (feat_df[cols['date']] == ema_date)
            ]
            
            if len(feat_row) == 0:
                continue
            
            feat_row = feat_row.iloc[0]
            
            # Find valid features for this sample
            sample_valid_features = set()
            for feat_col in stat_features:
                col_name = f"{feat_col}_before1day"
                if col_name in feat_row.index and pd.notna(feat_row[col_name]):
                    sample_valid_features.add(feat_col)
            
            # Intersect with global valid features
            if global_valid_feats_set is None:
                global_valid_feats_set = sample_valid_features
            else:
                global_valid_feats_set = global_valid_feats_set.intersection(sample_valid_features)
            
            sample_count += 1
            if sample_count >= 100:  # Sample first 100 to determine global features
                break
        
        if global_valid_feats_set is None or len(global_valid_feats_set) == 0:
            print("  [ERROR] No globally valid features found!")
            return []
        
        global_valid_features = sorted(list(global_valid_feats_set))  # Sort for consistency
        print(f"  [Found {len(global_valid_features)} globally valid features: {', '.join(global_valid_features[:5])}...]")
    else:
        print(f"  [Using pre-determined {len(global_valid_features)} globally valid features]")
    
    # Step 2: Extract time series using only global valid features
    for idx in lab_df.index:
        row = lab_df.loc[idx]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # Get corresponding feature row
        feat_row = feat_df[
            (feat_df[cols['user_id']] == user_id) & 
            (feat_df[cols['date']] == ema_date)
        ]
        
        if len(feat_row) == 0:
            continue
        
        feat_row = feat_row.iloc[0]
        
        # Extract time series using global valid features
        time_series = _extract_ces_time_series_from_row(feat_row, global_valid_features)
        
        if time_series is None:
            continue
        
        # Build sample dict
        labels = row[cols['labels']].to_dict()
        sample = {
            'aggregated_features': feat_row.to_dict(),
            'labels': labels,
            'user_id': user_id,
            'ema_date': ema_date
        }
        
        candidates.append({
            'time_series': time_series,
            'sample': sample,
            'user_id': user_id,
            'ema_date': ema_date
        })
    
    return candidates


def _cluster_ces_quarterly_chunk(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    k: int, random_state: Optional[int] = None
) -> List[Dict]:
    """
    Cluster a quarterly chunk of CES data using K-means.
    
    Returns K representative samples (nearest to centroids).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Extract all samples with time series
    all_candidates = _extract_ces_samples_as_candidates(feat_df, lab_df, cols)
    
    if len(all_candidates) == 0:
        return []
    
    if len(all_candidates) <= k:
        # If samples ≤ k, return all
        return all_candidates
    
    # Extract time series and flatten
    all_time_series = [c['time_series'] for c in all_candidates]
    
    # Pad/truncate to fixed length (28 days x n_features)
    n_features = all_time_series[0].shape[1]
    flattened_series = []
    
    for ts in all_time_series:
        n_days = ts.shape[0]
        if n_days < 28:
            padded = np.pad(ts, ((0, 28 - n_days), (0, 0)), mode='constant', constant_values=0)
        elif n_days > 28:
            padded = ts[-28:, :]
        else:
            padded = ts
        
        flattened = padded.flatten()
        flattened_series.append(flattened)
    
    flattened_array = np.array(flattened_series)
    
    # Normalize
    scaler = StandardScaler()
    normalized_array = scaler.fit_transform(flattened_array)
    normalized_array = np.nan_to_num(normalized_array, nan=0.0)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(normalized_array)
    centroids = kmeans.cluster_centers_
    
    # Select nearest sample to each centroid
    representatives = []
    for cluster_id in range(k):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        centroid = centroids[cluster_id]
        cluster_samples = normalized_array[cluster_indices]
        
        distances = np.linalg.norm(cluster_samples - centroid, axis=1)
        nearest_idx = cluster_indices[np.argmin(distances)]
        
        representatives.append(all_candidates[nearest_idx])
    
    return representatives


def _extract_ces_time_series_from_row(feat_row: pd.Series, stat_features: List[str]) -> Optional[np.ndarray]:
    """
    Extract time series from CES aggregated row.
    
    CES rows have columns like: feature_before1day, feature_before2day, ..., feature_before28day
    
    Returns:
        numpy array of shape (n_days, n_features) or None
    """
    # First, determine which features actually have before#day columns
    valid_features = []
    for feat_col in stat_features:
        col_name = f"{feat_col}_before1day"
        if col_name in feat_row.index:
            valid_features.append(feat_col)
    
    if len(valid_features) == 0:
        return None
    
    time_series_list = []
    
    # Try to extract up to 28 days
    for day_idx in range(1, 29):
        day_features = []
        has_any_data = False
        
        for feat_col in valid_features:
            col_name = f"{feat_col}_before{day_idx}day"
            val = feat_row.get(col_name, np.nan)
            
            if pd.notna(val):
                day_features.append(float(val))
                has_any_data = True
            else:
                day_features.append(0.0)
        
        if has_any_data:
            time_series_list.append(day_features)
    
    if len(time_series_list) < 14:  # Need at least 50% of 28 days
        return None
    
    # Convert to numpy array: shape (n_days, n_features)
    # Note: time_series_list is in reverse chronological order (before1day is most recent)
    # Reverse it to get chronological order
    time_series = np.array(list(reversed(time_series_list)))
    
    return time_series


def sample_from_timerag_pool_dtw_ces(
    feat_df: pd.DataFrame, cols: Dict,
    n_samples: int, target_sample: Dict, 
    candidates: List[Dict],
    diversity_factor: float = 2.0
) -> Optional[List[Dict]]:
    """
    DTW-based retrieval from TimeRAG candidate pool for CES data.
    
    Includes user deduplication as requested.
    
    Args:
        feat_df: Feature DataFrame
        cols: Column configuration
        n_samples: Number of samples to retrieve
        target_sample: Target sample dict
        candidates: TimeRAG candidate pool
        diversity_factor: Multiplier for initial retrieval (default 2.0)
        
    Returns:
        List of selected examples with balanced labels and deduplicated users
    """
    if not candidates or len(candidates) < n_samples:
        print(f"Warning: Only {len(candidates)} candidates available")
        return [c['sample'] for c in candidates] if candidates else None
    
    # Extract target time series using SAME features as candidates
    target_feat_row = feat_df[
        (feat_df[cols['user_id']] == target_sample['user_id']) & 
        (feat_df[cols['date']] == target_sample['ema_date'])
    ]
    
    if len(target_feat_row) == 0:
        print("Warning: Could not find target sample in feat_df")
        return None
    
    target_feat_row = target_feat_row.iloc[0]
    
    # Infer global_valid_features from candidates
    # All candidates should have the same feature dimensions
    if len(candidates) > 0:
        n_features = candidates[0]['time_series'].shape[1]
        stat_features = list(cols['feature_set']['statistical'].keys())
        
        # Determine which features are valid for target (same logic as candidates)
        target_valid_features = []
        for feat_col in stat_features:
            col_name = f"{feat_col}_before1day"
            if col_name in target_feat_row.index:
                target_valid_features.append(feat_col)
        
        # Use only the first n_features that are valid
        # This ensures consistency with candidates
        target_valid_features = sorted(target_valid_features)[:n_features]
        target_ts = _extract_ces_time_series_from_row(target_feat_row, target_valid_features)
    else:
        # Fallback: use all statistical features
        stat_features = list(cols['feature_set']['statistical'].keys())
        target_ts = _extract_ces_time_series_from_row(target_feat_row, stat_features)
    
    if target_ts is None:
        print("Warning: Could not extract target time series")
        return None
    
    # Compute DTW distances
    distances = []
    
    for candidate in candidates:
        cand_ts = candidate['time_series']
        
        # Pad sequences to same length (both time and feature dimensions)
        max_len = max(target_ts.shape[0], cand_ts.shape[0])
        max_feats = max(target_ts.shape[1], cand_ts.shape[1])
        
        target_padded = np.pad(target_ts, 
                               ((0, max_len - target_ts.shape[0]), (0, max_feats - target_ts.shape[1])), 
                               mode='constant', constant_values=0)
        cand_padded = np.pad(cand_ts, 
                            ((0, max_len - cand_ts.shape[0]), (0, max_feats - cand_ts.shape[1])),
                            mode='constant', constant_values=0)
        
        # Normalize
        cand_mean = np.nanmean(cand_padded, axis=0, keepdims=True)
        cand_std = np.nanstd(cand_padded, axis=0, keepdims=True)
        cand_std[cand_std < 1e-6] = 1.0
        
        target_norm = (target_padded - cand_mean) / cand_std
        cand_norm = (cand_padded - cand_mean) / cand_std
        
        target_norm = np.nan_to_num(target_norm, nan=0.0)
        cand_norm = np.nan_to_num(cand_norm, nan=0.0)
        
        # Compute DTW
        try:
            distance = dtw_ndim.distance(target_norm, cand_norm)
        except Exception as e:
            print(f"Warning: DTW calculation failed: {e}")
            distance = 1e6
        
        distances.append(distance)
    
    # Step 1: Retrieve top-(n_samples * diversity_factor) candidates
    distances = np.array(distances)
    n_retrieve = min(len(candidates), int(n_samples * diversity_factor))
    top_k_indices = np.argsort(distances)[:n_retrieve]
    
    # Step 2: User deduplication (as requested)
    retrieved_samples = []
    user_count = {}
    
    for idx in top_k_indices:
        sample = candidates[idx]['sample']
        user_id = sample['user_id']
        distance = distances[idx]
        labels = sample['labels']
        
        # Create label key
        label_cols = cols['labels']
        if len(label_cols) >= 2:
            anx_label = str(int(labels.get(label_cols[0], 0)))
            dep_label = str(int(labels.get(label_cols[1], 0)))
            label_key = f"{anx_label}_{dep_label}"
        else:
            label_key = "unknown"
        
        retrieved_samples.append({
            'sample': sample,
            'distance': distance,
            'label_key': label_key,
            'user_id': user_id
        })
        
        user_count[user_id] = user_count.get(user_id, 0) + 1
    
    # Step 3: Deduplicate users (keep closest sample per user)
    dedup_samples = {}
    for item in retrieved_samples:
        user_id = item['user_id']
        if user_id not in dedup_samples:
            dedup_samples[user_id] = item
        else:
            # Keep sample with smaller distance
            if item['distance'] < dedup_samples[user_id]['distance']:
                dedup_samples[user_id] = item
    
    dedup_list = list(dedup_samples.values())
    
    print(f"  User deduplication: {len(retrieved_samples)} -> {len(dedup_list)} samples")
    
    # Step 4: Select n_samples with balanced label distribution
    from collections import defaultdict
    label_groups = defaultdict(list)
    for item in dedup_list:
        label_groups[item['label_key']].append(item)
    
    n_groups = len(label_groups)
    if n_groups == 0:
        return None
    
    # Allocate samples per group
    samples_per_group = {}
    if n_samples >= n_groups:
        base_per_group = n_samples // n_groups
        remainder = n_samples % n_groups
        
        sorted_groups = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]), reverse=True)
        
        for label_key in sorted_groups:
            samples_per_group[label_key] = base_per_group
        
        for i in range(remainder):
            samples_per_group[sorted_groups[i]] += 1
    else:
        sorted_groups = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]), reverse=True)
        for i in range(n_samples):
            label_key = sorted_groups[i % n_groups]
            samples_per_group[label_key] = samples_per_group.get(label_key, 0) + 1
    
    # Select samples from each group
    selected_examples = []
    for label_key, n_needed in samples_per_group.items():
        group_items = label_groups[label_key]
        group_items_sorted = sorted(group_items, key=lambda x: x['distance'])
        
        for item in group_items_sorted[:n_needed]:
            selected_examples.append(item['sample'])
    
    # Shuffle
    np.random.shuffle(selected_examples)
    
    return selected_examples