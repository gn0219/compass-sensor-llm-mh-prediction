"""
In-Context Learning Example Selection Module

Handles selection of ICL examples based on different strategies:
- Source: Where to draw examples (personalized, generalized, hybrid)
- Selection: How to select examples (random, similarity, temporal, diversity)

Selection Strategies:
- random: Random sampling (baseline, works for all sources)
- similarity: Feature-space cosine similarity (works for all sources)
- temporal: Most recent samples (only for personalized source)
- diversity: K-means clustering + stratified sampling (only for generalized source)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from .sensor_transformation import aggregate_window_features, check_missing_ratio
from . import config


def select_icl_examples(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    target_user_id: str, target_ema_date: pd.Timestamp,
    n_shot: int = 5, source: str = 'hybrid', selection: str = 'random',
    random_state: Optional[int] = None, target_sample: Optional[Dict] = None,
    beta: float = 0.0
) -> Optional[List[Dict]]:
    """
    Select in-context learning examples based on specified strategy.
    Uses aggregation settings from config.py.
    
    Args:
        feat_df: Feature DataFrame
        lab_df: Label DataFrame
        cols: Column configuration
        target_user_id: Target user ID
        target_ema_date: Target EMA date
        n_shot: Number of examples to select
        source: 'personalized', 'generalized', or 'hybrid'
        selection: 'random', 'similarity', 'temporal', or 'diversity'
        random_state: Random seed for reproducibility
        target_sample: Target sample dict (required for similarity-based selection)
        beta: Label balance penalty for diversity selection (0.0=no penalty, 0.1-0.3=recommended)
    
    Returns:
        List of example dictionaries, or None if insufficient data
    """
    if n_shot < 1:
        return []
    
    # Validate selection method for source
    _validate_selection_for_source(source, selection)
    
    # Use config defaults for aggregation (agg_params=None will trigger defaults)
    agg_params = None
    
    examples = []
    
    if source == 'personalized':
        # Use only user's own historical data (before target date)
        personal_lab = lab_df[
            (lab_df[cols['user_id']] == target_user_id) & 
            (lab_df[cols['date']] < target_ema_date)
        ]
        
        if len(personal_lab) < n_shot:
            print(f"Warning: Not enough historical data for user {target_user_id} before {target_ema_date}")
            return None
        
        examples = _sample_from_pool(
            feat_df, personal_lab, cols, n_shot, selection, random_state,
            target_sample=target_sample, target_date=target_ema_date, beta=beta
        )
    
    elif source == 'generalized':
        # Use only other users' data
        general_lab = lab_df[lab_df[cols['user_id']] != target_user_id]
        
        if len(general_lab) < n_shot:
            print(f"Warning: Not enough data from other users")
            return None
        
        examples = _sample_from_pool(
            feat_df, general_lab, cols, n_shot, selection, random_state,
            target_sample=target_sample, beta=beta, agg_params=agg_params
        )
    
    elif source == 'hybrid':
        # Use half from personal, half from general
        n_personal = n_shot // 2
        n_general = n_shot - n_personal
        
        personal_lab = lab_df[
            (lab_df[cols['user_id']] == target_user_id) & 
            (lab_df[cols['date']] < target_ema_date)
        ]
        general_lab = lab_df[lab_df[cols['user_id']] != target_user_id]
        
        if len(personal_lab) < n_personal:
            print(f"Warning: Not enough historical data for user {target_user_id}")
            return None
        if len(general_lab) < n_general:
            print(f"Warning: Not enough data from other users")
            return None
        
        # Personal examples: use specified selection method
        personal_examples = _sample_from_pool(
            feat_df, personal_lab, cols, n_personal, selection, random_state,
            target_sample=target_sample, target_date=target_ema_date, beta=beta,
            agg_params=agg_params
        )
        
        general_examples = _sample_from_pool(
            feat_df, general_lab, cols, n_general, selection, random_state,
            # random_state + 1000 if random_state else None,
            target_sample=target_sample, beta=beta, agg_params=agg_params
        )
        
        examples = personal_examples + general_examples
    
    else:
        raise ValueError(f"Invalid source: {source}. Must be 'personalized', 'generalized', or 'hybrid'")
    
    return examples


def _validate_selection_for_source(source: str, selection: str):
    """Validate that selection method is compatible with source."""
    if selection == 'temporal' and source != 'personalized':
        raise ValueError(f"Temporal selection only works with 'personalized', got '{source}'")
    
    if selection == 'diversity' and source != 'generalized':
        raise ValueError(f"Diversity selection is designed for 'generalized', got '{source}'")


def _sample_from_pool(feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict, n_samples: int,
    selection: str = 'random', random_state: Optional[int] = None, target_sample: Optional[Dict] = None,
    target_date: Optional[pd.Timestamp] = None, beta: float = 0.0, agg_params: Dict = None) -> List[Dict]:
    """
    Sample n_samples from the label pool using specified selection strategy.
    
    Args:
        feat_df: Feature DataFrame
        lab_pool: Pool of label samples to choose from
        cols: Column configuration
        n_samples: Number of samples to select
        selection: Selection method ('random', 'similarity', 'temporal', 'diversity')
        random_state: Random seed
        target_sample: Target sample for similarity-based selection
        target_date: Target date for temporal selection
        beta: Label balance penalty for diversity selection
        agg_params: Aggregation parameters (window_days, aggregation_mode, etc.)
    
    Returns:
        List of example dictionaries with aggregated features and labels
    """
    if agg_params is None:
        agg_params = {
            'window_days': config.AGGREGATION_WINDOW_DAYS,
            'aggregation_mode': config.DEFAULT_AGGREGATION_MODE,
            'use_immediate_window': config.USE_IMMEDIATE_WINDOW,
            'immediate_window_days': config.IMMEDIATE_WINDOW_DAYS,
            'adaptive_window': config.USE_ADAPTIVE_WINDOW
        }
    
    if selection == 'random':
        return _random_selection(feat_df, lab_pool, cols, n_samples, random_state, agg_params)
    
    elif selection == 'similarity':
        return _similarity_selection(feat_df, lab_pool, cols, n_samples, random_state, 
                                     target_sample, metric='cosine', agg_params=agg_params)
    
    elif selection == 'temporal':
        return _temporal_selection(feat_df, lab_pool, cols, n_samples, random_state,
                                   target_date, most_recent=True, agg_params=agg_params)
    
    elif selection == 'diversity':
        return _diversity_selection(feat_df, lab_pool, cols, n_samples, random_state,
                                    n_clusters=min(n_samples, 5) if n_samples > 10 else n_samples, 
                                    beta=beta, agg_params=agg_params)
    
    else:
        raise ValueError(f"Invalid selection method: {selection}. Choose from: 'random', 'similarity', 'temporal', 'diversity'")


# ==================== Selection Strategy Implementations ====================

def _random_selection(feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict, n_samples: int, 
                      random_state: Optional[int] = None, agg_params: Dict = None, 
                      max_attempts: int = None) -> List[Dict]:
    """
    Random selection baseline - shuffle and select valid samples.
    
    Hyperparameters:
        - random_state: Seed for reproducibility
        - max_attempts: Maximum iterations to find valid samples (default: n_samples * 10)
        - agg_params: Aggregation parameters (window_days, aggregation_mode, etc.)
    """
    if max_attempts is None:
        max_attempts = n_samples * 10
    
    if agg_params is None:
        agg_params = {
            'window_days': config.AGGREGATION_WINDOW_DAYS,
            'aggregation_mode': config.DEFAULT_AGGREGATION_MODE,
            'use_immediate_window': config.USE_IMMEDIATE_WINDOW,
            'immediate_window_days': config.IMMEDIATE_WINDOW_DAYS,
            'adaptive_window': config.USE_ADAPTIVE_WINDOW
        }
    
    examples = []
    attempts = 0
    
    # Shuffle the pool
    if random_state is not None:
        lab_pool = lab_pool.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        lab_pool = lab_pool.sample(frac=1).reset_index(drop=True)
    
    idx = 0
    while len(examples) < n_samples and attempts < max_attempts:
        if idx >= len(lab_pool):
            break
        
        row = lab_pool.iloc[idx]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # Get aggregated features
        agg_feats = aggregate_window_features(
            feat_df, user_id, ema_date, cols,
            window_days=agg_params['window_days'],
            mode=agg_params['aggregation_mode'],
            use_immediate_window=agg_params['use_immediate_window'],
            immediate_window_days=agg_params['immediate_window_days'],
            adaptive_window=agg_params['adaptive_window']
        )
        
        # Check if valid (not too many missing values)
        if agg_feats is not None and check_missing_ratio(agg_feats):
            labels = row[cols['labels']].to_dict()
            examples.append({
                'aggregated_features': agg_feats,
                'labels': labels,
                'user_id': user_id,
                'ema_date': ema_date
            })
        
        idx += 1
        attempts += 1
    
    return examples


def _similarity_selection(feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict, n_samples: int,
                          random_state: Optional[int] = None, target_sample: Optional[Dict] = None,
                          metric: str = 'cosine', max_pool_size: int = 500, agg_params: Dict = None) -> List[Dict]:
    """
    Feature-space similarity-based selection using cosine similarity.
    
    Selects top-k most similar samples based on normalized sensor features.
    
    Hyperparameters:
        - metric: Similarity metric ('cosine' or 'euclidean')
        - max_pool_size: Maximum candidates to consider (for efficiency)
        - random_state: Seed for candidate sampling if pool is large
        - agg_params: Aggregation parameters (window_days, aggregation_mode, etc.)
    """
    if target_sample is None:
        raise ValueError("target_sample is required for similarity-based selection")
    
    if agg_params is None:
        agg_params = {
            'window_days': config.AGGREGATION_WINDOW_DAYS,
            'aggregation_mode': config.DEFAULT_AGGREGATION_MODE,
            'use_immediate_window': config.USE_IMMEDIATE_WINDOW,
            'immediate_window_days': config.IMMEDIATE_WINDOW_DAYS,
            'adaptive_window': config.USE_ADAPTIVE_WINDOW
        }
    
    # Extract target feature vector
    target_vector = _extract_feature_vector(target_sample['aggregated_features'], cols)
    
    # Build candidate pool with features
    candidates = _build_candidate_pool(feat_df, lab_pool, cols, max_pool_size, random_state, agg_params)
    
    if len(candidates) < n_samples:
        print(f"Warning: Only {len(candidates)} valid candidates for similarity selection")
        return [c['sample'] for c in candidates]
    
    # Extract feature vectors from candidates and pad to same length
    all_vectors = [c['features'] for c in candidates] + [target_vector]
    max_length = max(len(v) for v in all_vectors)
    
    # Pad candidate vectors to max length
    padded_candidates = []
    for vec in [c['features'] for c in candidates]:
        if len(vec) < max_length:
            padded = np.pad(vec, (0, max_length - len(vec)), mode='constant', constant_values=0)
        else:
            padded = vec
        padded_candidates.append(padded)
    
    # Pad target vector to max length
    if len(target_vector) < max_length:
        target_vector_padded = np.pad(target_vector, (0, max_length - len(target_vector)), mode='constant', constant_values=0)
    else:
        target_vector_padded = target_vector
    
    candidate_vectors = np.vstack(padded_candidates)
    
    scaler = StandardScaler()
    candidates_norm = scaler.fit_transform(candidate_vectors)
    target_norm = scaler.transform(target_vector_padded.reshape(1, -1))

    # Handle NaN from features with zero variance (constant features)
    # Replace NaN with 0 (these features don't contribute to similarity)
    candidates_norm = np.nan_to_num(candidates_norm, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    target_norm = np.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    
    # Compute similarity
    if metric == 'cosine':
        similarities = cosine_similarity(target_norm, candidates_norm)[0]
    elif metric == 'euclidean':
        # Convert distance to similarity (negative distance)
        distances = np.linalg.norm(candidates_norm - target_norm, axis=1)
        similarities = -distances
    else:
        raise ValueError(f"Invalid metric: {metric}. Choose 'cosine' or 'euclidean'")
    
    # Select top-k most similar
    top_indices = np.argsort(similarities)[-n_samples:][::-1]
    
    return [candidates[i]['sample'] for i in top_indices]


def _temporal_selection(feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict, n_samples: int,
                        random_state: Optional[int] = None, target_date: Optional[pd.Timestamp] = None,
                        most_recent: bool = True, max_attempts: int = None, agg_params: Dict = None) -> List[Dict]:
    """
    Temporal proximity-based selection - select most recent samples before target date.
    
    Only applicable for personalized source (user's own historical data).
    
    Hyperparameters:
        - most_recent: If True, select most recent; if False, select oldest
        - max_attempts: Maximum iterations to find valid samples
        - agg_params: Aggregation parameters (window_days, aggregation_mode, etc.)
    """
    if target_date is None:
        raise ValueError("target_date is required for temporal selection")
    
    if max_attempts is None:
        max_attempts = n_samples * 10
    
    if agg_params is None:
        agg_params = {
            'window_days': config.AGGREGATION_WINDOW_DAYS,
            'aggregation_mode': config.DEFAULT_AGGREGATION_MODE,
            'use_immediate_window': config.USE_IMMEDIATE_WINDOW,
            'immediate_window_days': config.IMMEDIATE_WINDOW_DAYS,
            'adaptive_window': config.USE_ADAPTIVE_WINDOW
        }
    
    # Sort by date (most recent first if most_recent=True)
    lab_pool_sorted = lab_pool.sort_values(by=cols['date'], ascending=not most_recent).reset_index(drop=True)
    
    examples = []
    attempts = 0
    idx = 0
    
    while len(examples) < n_samples and attempts < max_attempts:
        if idx >= len(lab_pool_sorted):
            break
        
        row = lab_pool_sorted.iloc[idx]
        user_id = row[cols['user_id']]
        ema_date = row[cols['date']]
        
        # Get aggregated features
        agg_feats = aggregate_window_features(
            feat_df, user_id, ema_date, cols,
            window_days=agg_params['window_days'],
            mode=agg_params['aggregation_mode'],
            use_immediate_window=agg_params['use_immediate_window'],
            immediate_window_days=agg_params['immediate_window_days'],
            adaptive_window=agg_params['adaptive_window']
        )
        
        # Check if valid
        if agg_feats is not None and check_missing_ratio(agg_feats):
            labels = row[cols['labels']].to_dict()
            examples.append({
                'aggregated_features': agg_feats,
                'labels': labels,
                'user_id': user_id,
                'ema_date': ema_date
            })
        
        idx += 1
        attempts += 1
    
    return examples


def _diversity_selection(feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict, n_samples: int,
                         random_state: Optional[int] = None, n_clusters: int = 5, beta: float = 0.0,
                         max_pool_size: int = 1000, agg_params: Dict = None) -> List[Dict]:
    """
    Diversity-based selection using K-means clustering with optional label balance penalty.
    
    Selection strategy (per cluster):
        score(x) = sim(x, centroid_C) - β * label_deviation(x, S)
    
    Where:
        - sim(x, centroid_C): Similarity to cluster centroid (representativeness)
        - label_deviation(x, S): How much label(x) deviates from ideal frequency in selected set S
        - β: Penalty weight (0 = pure representativeness, >0 = soft label balance)
    
    Hyperparameters:
        - n_clusters: Number of clusters (default: min(n_samples, 5) if n_samples > 10)
        - beta: Label balance penalty weight (default: 0.0)
                  - 0.0: Pure cluster representativeness (baseline)
                  - 0.1-0.3: Soft label balance
        - max_pool_size: Maximum candidates to cluster (for efficiency)
        - random_state: Seed for clustering reproducibility
        - agg_params: Aggregation parameters (window_days, aggregation_mode, etc.)
    """
    if agg_params is None:
        agg_params = {
            'window_days': config.AGGREGATION_WINDOW_DAYS,
            'aggregation_mode': config.DEFAULT_AGGREGATION_MODE,
            'use_immediate_window': config.USE_IMMEDIATE_WINDOW,
            'immediate_window_days': config.IMMEDIATE_WINDOW_DAYS,
            'adaptive_window': config.USE_ADAPTIVE_WINDOW
        }
    
    # Build candidate pool with features
    candidates = _build_candidate_pool(feat_df, lab_pool, cols, max_pool_size, random_state, agg_params)
    
    if len(candidates) < n_samples:
        print(f"Warning: Only {len(candidates)} valid candidates for diversity selection")
        return [c['sample'] for c in candidates]
    
    return _diversity_kmeans(candidates, n_samples, n_clusters, beta, random_state)


def _diversity_kmeans(candidates: List[Dict], n_samples: int, n_clusters: int, beta: float = 0.0,
                      random_state: Optional[int] = None) -> List[Dict]:
    """
    K-means clustering with centroid-based selection and optional label balance penalty.
    
    Selection strategy per cluster:
        score(x) = sim(x, centroid_C) - β * label_deviation(x, S)
    
    Where:
        - sim(x, centroid_C): Cosine similarity to cluster centroid (representativeness)
        - label_deviation(x, S): |freq(label(x), S) - ideal_freq| (label imbalance)
        - β: Penalty weight (0 = no penalty, >0 = soft label balance)
    
    Steps:
    1. Cluster candidates into n_clusters using K-means
    2. From each cluster, select samples that maximize: centroid similarity - β * label deviation
    3. Ensures both behavioral diversity (via clustering) and label balance (via penalty)
    """
    # Extract and normalize features
    candidate_vectors = np.vstack([c['features'] for c in candidates])
    scaler = StandardScaler()
    normalized = scaler.fit_transform(candidate_vectors)
    
    # Handle NaN from features with zero variance (constant features)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure n_clusters doesn't exceed candidates
    n_clusters = min(n_clusters, len(candidates))
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized)
    centroids = kmeans.cluster_centers_
    
    # Assign cluster labels and centroid similarities to candidates
    for i, candidate in enumerate(candidates):
        candidate['cluster'] = cluster_labels[i]
        candidate['normalized_features'] = normalized[i]
        # Compute similarity to cluster centroid
        centroid = centroids[cluster_labels[i]]
        candidate['centroid_similarity'] = cosine_similarity(
            normalized[i].reshape(1, -1), 
            centroid.reshape(1, -1)
        )[0, 0]
    
    # Determine samples per cluster (proportional allocation)
    samples_per_cluster = {}
    for cluster_id in range(n_clusters):
        samples_per_cluster[cluster_id] = n_samples // n_clusters
    
    # Distribute remaining samples
    remaining = n_samples - sum(samples_per_cluster.values())
    for i in range(remaining):
        samples_per_cluster[i % n_clusters] += 1
    
    # Select samples from each cluster using scoring function
    selected = []
    
    for cluster_id, n_cluster_samples in samples_per_cluster.items():
        cluster_candidates = [c for c in candidates if c['cluster'] == cluster_id]
        
        if len(cluster_candidates) == 0:
            continue
        
        # Track selected candidate IDs to avoid duplicates
        selected_ids = set()
        
        # Iteratively select samples based on score
        for _ in range(min(n_cluster_samples, len(cluster_candidates))):
            # Compute scores for all unselected candidates
            scores = []
            for idx, candidate in enumerate(cluster_candidates):
                candidate_id = (candidate['sample']['user_id'], candidate['sample']['ema_date'])
                
                if candidate_id in selected_ids:
                    continue  # Skip already selected
                
                # Compute label deviation penalty
                if beta > 0:
                    label_penalty = _compute_label_deviation(candidate['sample'], selected)
                else:
                    label_penalty = 0.0
                
                # Score: centroid similarity - beta * label deviation
                score = candidate['centroid_similarity'] - beta * label_penalty
                scores.append({'idx': idx, 'candidate': candidate, 'score': score, 'id': candidate_id})
            
            if not scores:
                break
            
            # Select candidate with highest score
            best = max(scores, key=lambda x: x['score'])
            selected.append(best['candidate']['sample'])
            selected_ids.add(best['id'])
    
    return selected[:n_samples]


def _compute_label_deviation(candidate_sample: Dict, selected_samples: List[Dict]) -> float:
    """
    Compute label deviation penalty for a candidate sample.
    
    Formula:
        deviation = |current_freq(label) - ideal_freq|
    
    Where:
        - current_freq(label): Proportion of label in selected samples + candidate
        - ideal_freq: 1 / num_unique_labels (e.g., 0.25 for 4 label combinations)
    
    Returns:
        Float in [0, 1] representing deviation from ideal label frequency
    """
    if not selected_samples:
        return 0.0  # No deviation when no samples selected yet
    
    # Extract label combination from candidate
    candidate_label = (
        candidate_sample['labels'].get('phq4_anxiety_EMA', 0),
        candidate_sample['labels'].get('phq4_depression_EMA', 0)
    )
    
    # Count label frequencies in selected samples
    label_counts = {}
    for sample in selected_samples:
        label = (
            sample['labels'].get('phq4_anxiety_EMA', 0),
            sample['labels'].get('phq4_depression_EMA', 0)
        )
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Compute frequency if we add this candidate
    total_count = len(selected_samples) + 1
    candidate_freq = (label_counts.get(candidate_label, 0) + 1) / total_count
    
    # Ideal frequency (uniform distribution over possible labels)
    # For binary anxiety × binary depression = 4 possible combinations
    ideal_freq = 0.25
    
    # Deviation: absolute difference from ideal
    deviation = abs(candidate_freq - ideal_freq)
    
    return deviation


# ==================== Helper Functions ====================

def _extract_feature_vector(agg_feats, cols: Dict, fill_strategy: str = 'zero') -> np.ndarray:
    """
    Extract numeric feature vector from aggregated features.
    
    Args:
        agg_feats: Aggregated features (Dict or DataFrame)
        cols: Column configuration
        fill_strategy: How to handle missing values ('zero', 'mean', 'median')
    
    Returns:
        1D numpy array of feature values
    """
    # Handle Dict format
    if isinstance(agg_feats, dict):
        mode = agg_feats.get('aggregation_mode', 'unknown')
        
        # COMPASS format
        if mode == 'compass':
            feature_values = []
            
            # Extract statistical features
            stat_feats = agg_feats.get('statistical_features', {})
            for feat_name, stats in stat_feats.items():
                for stat_name, stat_value in stats.items():
                    val = stat_value if stat_value is not None else np.nan
                    feature_values.append(val)
            
            # Extract structural features
            struct_feats = agg_feats.get('structural_features', {})
            for feat_name, struct in struct_feats.items():
                for struct_name, struct_value in struct.items():
                    if isinstance(struct_value, (int, float)):
                        val = struct_value if struct_value is not None else np.nan
                        feature_values.append(val)
                    # Skip string values like 'increasing', 'stable', etc.
            
            # Extract semantic features (flatten nested structure)
            semantic_feats = agg_feats.get('semantic_features', {})
            for feat_name, semantic_info in semantic_feats.items():
                for key, value in semantic_info.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                val = sub_value if sub_value is not None else np.nan
                                feature_values.append(val)
                    elif isinstance(value, (int, float)):
                        val = value if value is not None else np.nan
                        feature_values.append(val)
            
            # Extract temporal descriptors
            temporal_feats = agg_feats.get('temporal_descriptors', {})
            for feat_name, values_list in temporal_feats.items():
                if isinstance(values_list, list):
                    for v in values_list:
                        val = v if v is not None else np.nan
                        feature_values.append(val)
            
            feature_vector = np.array(feature_values, dtype=float)
        
        # ARRAY mode
        elif mode == 'array':
            feature_values = []
            for feat_name, feat_values_list in agg_feats['features'].items():
                values = [v if v is not None else np.nan for v in feat_values_list]
                feature_values.extend(values)
            feature_vector = np.array(feature_values, dtype=float)
        
        # STATISTICS mode or RAW mode
        elif mode == 'statistics' or mode == 'raw':
            feature_values = []
            features_dict = agg_feats.get('features', {})
            for feat_name, feat_stats in features_dict.items():
                if isinstance(feat_stats, dict):
                    for stat_name, stat_value in feat_stats.items():
                        if not stat_name.startswith('last_'):
                            val = stat_value if stat_value is not None else np.nan
                            feature_values.append(val)
                elif isinstance(feat_stats, (int, float)):
                    val = feat_stats if feat_stats is not None else np.nan
                    feature_values.append(val)
            
            # If no features, return a minimal vector
            if not feature_values:
                feature_values = [0.0]
            
            feature_vector = np.array(feature_values, dtype=float)
        
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")
    
    # Handle legacy DataFrame format
    elif isinstance(agg_feats, pd.DataFrame):
        feature_cols = [col for col in agg_feats.columns 
                       if col not in [cols.get('user_id'), cols.get('date')]]
        feature_vector = agg_feats[feature_cols].values.flatten()
    
    else:
        raise ValueError(f"Unsupported agg_feats type: {type(agg_feats)}")
    
    # Handle missing values
    if fill_strategy == 'zero':
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    elif fill_strategy == 'mean':
        mean_val = np.nanmean(feature_vector)
        feature_vector = np.nan_to_num(feature_vector, nan=mean_val if not np.isnan(mean_val) else 0.0)
    elif fill_strategy == 'median':
        median_val = np.nanmedian(feature_vector)
        feature_vector = np.nan_to_num(feature_vector, nan=median_val if not np.isnan(median_val) else 0.0)
    else:
        raise ValueError(f"Invalid fill_strategy: {fill_strategy}")
    
    return feature_vector


def _build_candidate_pool(feat_df: pd.DataFrame, lab_pool: pd.DataFrame, cols: Dict,
                          max_pool_size: int = 500, random_state: Optional[int] = None,
                          agg_params: Dict = None) -> List[Dict]:
    """
    Build candidate pool with aggregated features and feature vectors.
    
    Args:
        feat_df: Feature DataFrame
        lab_pool: Pool of label samples
        cols: Column configuration
        max_pool_size: Maximum pool size for efficiency
        random_state: Random seed
        agg_params: Aggregation parameters (window_days, aggregation_mode, etc.)
    
    Returns:
        List of dicts with 'features' (numpy array) and 'sample' (dict)
    """
    if agg_params is None:
        agg_params = {
            'window_days': config.AGGREGATION_WINDOW_DAYS,
            'aggregation_mode': config.DEFAULT_AGGREGATION_MODE,
            'use_immediate_window': config.USE_IMMEDIATE_WINDOW,
            'immediate_window_days': config.IMMEDIATE_WINDOW_DAYS,
            'adaptive_window': config.USE_ADAPTIVE_WINDOW
        }
    
    # Force 'statistics' mode for similarity calculation (faster and sufficient)
    # Use 28-day window for consistency
    similarity_agg_params = {
        'window_days': 28,
        'mode': 'statistics',
        'use_immediate_window': False,
        'immediate_window_days': 0,
        'adaptive_window': False
    }
    
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
        
        # Get aggregated features (use statistics mode for faster similarity calculation)
        agg_feats = aggregate_window_features(
            feat_df, user_id, ema_date, cols,
            window_days=similarity_agg_params['window_days'],
            mode=similarity_agg_params['mode'],
            use_immediate_window=similarity_agg_params['use_immediate_window'],
            immediate_window_days=similarity_agg_params['immediate_window_days'],
            adaptive_window=similarity_agg_params['adaptive_window']
        )
        
        # Check if valid
        if agg_feats is not None and check_missing_ratio(agg_feats):
            labels = row[cols['labels']].to_dict()
            feature_vector = _extract_feature_vector(agg_feats, cols)
            
            candidates.append({
                'features': feature_vector,
                'sample': {
                    'aggregated_features': agg_feats,
                    'labels': labels,
                    'user_id': user_id,
                    'ema_date': ema_date
                }
            })
    
    return candidates

