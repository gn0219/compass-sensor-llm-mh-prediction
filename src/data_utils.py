"""
Data Utility Functions

Functions for testset sampling, filtering, and other data operations.
Extracted from sensor_transformation.py for better code organization.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from datetime import timedelta

try:
    from . import config
except ImportError:
    import config


def binarize_labels(df: pd.DataFrame, labels: list, thresholds: Dict[str, int]) -> pd.DataFrame:
    """Binarize labels based on thresholds."""
    df = df.copy()
    for label in labels:
        if label in df.columns:
            df[label] = (df[label] > thresholds[label]).astype(int)
    return df

def load_ces_data(use_cols_path: str = './config/ces_use_cols.json') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load CES dataset with feature and label data - Returns AGGREGATED data."""
    aggregated_path = '../dataset/CES/aggregated_ces.csv'
    
    # Check if aggregated data already exists
    if os.path.exists(aggregated_path):
        print(f"Loading cached aggregated CES data from {aggregated_path}")
        agg_df = pd.read_csv(aggregated_path, low_memory=False)
        
        # Convert date column
        agg_df['date'] = pd.to_datetime(agg_df['date'])
        
        # Load column configuration
        with open(use_cols_path, 'r') as f:
            use_cols = json.load(f)
        cols = use_cols['compass']
        
        # Split into features and labels
        label_cols = [cols['user_id'], cols['date']] + cols['labels']
        lab_df = agg_df[label_cols].copy()
        
        # Feature columns: everything except pure label columns (but keep user_id, date, gender)
        feat_cols = [c for c in agg_df.columns if c not in cols['labels']]
        feat_df = agg_df[feat_cols].copy()
        
        print(f"  Loaded {len(feat_df)} samples with {len(feat_df.columns)-3} aggregated features")
        return feat_df, lab_df, cols
    
    print(f"Aggregated data not found. Building from raw CES data...")
    feat_df, lab_df, cols = _aggregate_ces_data(use_cols_path)
    
    # Save aggregated data
    print(f"Saving aggregated data to {aggregated_path}")
    # Merge features and labels for saving
    merged_df = feat_df.merge(lab_df, on=[cols['user_id'], cols['date']], how='inner')
    merged_df.to_csv(aggregated_path, index=False)
    
    return feat_df, lab_df, cols


def load_mentaliot_data(use_cols_path: str = './config/mentaliot_use_cols.json') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load MentalIoT dataset with feature and label data - Returns AGGREGATED data.
    
    MentalIoT data should already be aggregated via prepare_mentaliot_data.py.
    This function simply loads the pre-aggregated CSV files.
    
    Returns:
        (feat_df, lab_df, cols): Features, labels, and column configuration
    """
    aggregated_path = '../dataset/MentalIoT/aggregated_mentaliot.csv'
    
    # Check if aggregated data exists
    if not os.path.exists(aggregated_path):
        raise FileNotFoundError(
            f"Aggregated MentalIoT data not found at {aggregated_path}\n"
            f"Please run: python prepare_mentaliot_data.py"
        )
    
    print(f"Loading aggregated MentalIoT data from {aggregated_path}")
    agg_df = pd.read_csv(aggregated_path, low_memory=False)
    
    # Convert timestamp column (already converted when saved, just parse)
    agg_df['timestamp'] = pd.to_datetime(agg_df['timestamp'])
    
    # Load column configuration
    with open(use_cols_path, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols['compass']
    
    # Split into features and labels
    label_cols = [cols['user_id'], cols['date']] + cols['labels']
    lab_df = agg_df[label_cols].copy()
    
    # Feature columns: everything except pure label columns (but keep user_id, timestamp)
    feat_cols = [c for c in agg_df.columns if c not in cols['labels']]
    feat_df = agg_df[feat_cols].copy()
    
    print(f"  Loaded {len(feat_df)} samples from {feat_df[cols['user_id']].nunique()} users")
    print(f"  Features: {len(feat_df.columns)-2} (excluding uid, timestamp)")
    
    return feat_df, lab_df, cols


def load_dataset_testset(dataset_type: str = 'globem') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Universal testset loader for all datasets (GLOBEM, CES, MentalIoT).
    
    Loads pre-generated testset and trainset from CSV files.
    Eliminates need for runtime testset generation and ensures consistency.
    
    Args:
        dataset_type: 'globem', 'ces', or 'mentaliot'
    
    Returns:
        (feat_df, lab_df, test_df, train_df, cols)
        - feat_df: All features (for aggregation/retrieval)
        - lab_df: All labels (combined test + train)
        - test_df: Test samples only
        - train_df: Training samples (for ICL)
        - cols: Column configuration
    """
    dataset_config = config.DATASET_CONFIGS[dataset_type]
    
    print(f"\n[Loading {dataset_config['name']} testset...]")
    
    # Load testset and trainset
    test_df = pd.read_csv(dataset_config['testset_file'], low_memory=False)
    train_df = pd.read_csv(dataset_config['trainset_file'], low_memory=False)
    
    # Load column configuration
    with open(dataset_config['use_cols_path'], 'r') as f:
        use_cols = json.load(f)
    cols = use_cols.get('compass', use_cols)  # Handle both formats
    
    # Convert date columns
    date_col = cols['date']
    test_df[date_col] = pd.to_datetime(test_df[date_col])
    train_df[date_col] = pd.to_datetime(train_df[date_col])
    
    # Combine for full label df
    lab_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Load features based on dataset type
    if dataset_type == 'globem':
        # GLOBEM: Load daily features (for DTW/ML) and pre-aggregated (for prompts)
        if dataset_config.get('has_daily_features') and os.path.exists(dataset_config['daily_features_file']):
            # Load combined daily features (faster, recommended)
            print(f"  Loading daily features from {dataset_config['daily_features_file']}")
            feat_df = pd.read_csv(dataset_config['daily_features_file'], low_memory=False)
            feat_df[date_col] = pd.to_datetime(feat_df[date_col])
        else:
            # Fallback: Load raw features from individual institutions
            print(f"  Loading raw features from {dataset_config['base_path']}")
            feat_dfs = []
            
            if 'institution' in test_df.columns:
                # Multi-institution GLOBEM
                institutions = test_df['institution'].unique()
                for inst in institutions:
                    feat_path = f"{dataset_config['base_path']}/{inst}/FeatureData/rapids.csv"
                    if os.path.exists(feat_path):
                        inst_feat = pd.read_csv(feat_path, low_memory=False)
                        inst_feat['institution'] = inst
                        inst_feat[date_col] = pd.to_datetime(inst_feat[date_col])
                        feat_dfs.append(inst_feat)
                feat_df = pd.concat(feat_dfs, ignore_index=True) if feat_dfs else pd.DataFrame()
            else:
                # Single institution GLOBEM (legacy)
                feat_path = f"{dataset_config['base_path']}/FeatureData/rapids.csv"
                feat_df = pd.read_csv(feat_path, low_memory=False)
                feat_df[date_col] = pd.to_datetime(feat_df[date_col])
        
        # Also load pre-aggregated features for fast prompt generation
        print(f"  Loading pre-aggregated features from {dataset_config['aggregated_file']}")
        aggregated_feat_df = pd.read_csv(dataset_config['aggregated_file'], low_memory=False)
        aggregated_feat_df[date_col] = pd.to_datetime(aggregated_feat_df[date_col])
        
        # Store aggregated features in config for prompt generation
        config.GLOBEM_AGGREGATED_FEAT_DF = aggregated_feat_df
    elif dataset_config['has_pre_aggregated']:
        # CES/MentalIoT: Load pre-aggregated features
        print(f"  Loading pre-aggregated features from {dataset_config['aggregated_file']}")
        feat_df = pd.read_csv(dataset_config['aggregated_file'], low_memory=False)
        feat_df[date_col] = pd.to_datetime(feat_df[date_col])
    else:
        # Should not reach here
        raise ValueError(f"Unknown dataset configuration for {dataset_type}")
    
    print(f"  Testset: {len(test_df)} samples from {test_df[cols['user_id']].nunique()} users")
    print(f"  Trainset: {len(train_df)} samples")
    print(f"  Labels: {dataset_config['labels']}")
    
    return feat_df, lab_df, test_df, train_df, cols


def sample_mentaliot_testset(
    n_samples_per_user: int = 10,
    random_state: Optional[int] = None,
    use_cols_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Sample MentalIoT testset.
    
    Returns pre-sampled testset and trainset created by prepare_mentaliot_data.py.
    
    Args:
        n_samples_per_user: Number of samples per user (default: 10)
        random_state: Random seed (not used, provided for API consistency)
        use_cols_path: Path to column configuration
    
    Returns:
        (feat_df, lab_df, test_df, cols)
    """
    # Use config path if not specified
    if use_cols_path is None:
        use_cols_path = config.MENTALIOT_USE_COLS_PATH
    
    test_path = '../dataset/MentalIoT/mentaliot_testset.csv'
    train_path = '../dataset/MentalIoT/mentaliot_trainset.csv'
    
    if not os.path.exists(test_path) or not os.path.exists(train_path):
        raise FileNotFoundError(
            f"MentalIoT testset/trainset not found.\n"
            f"Please run: python prepare_mentaliot_data.py"
        )
    
    print(f"Loading MentalIoT testset from {test_path}")
    test_df = pd.read_csv(test_path, low_memory=False)
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    print(f"Loading MentalIoT trainset from {train_path}")
    train_df = pd.read_csv(train_path, low_memory=False)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    
    # Load column configuration
    with open(use_cols_path, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols['compass']
    
    # Combine into full feat_df and lab_df
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Split into features and labels
    label_cols = [cols['user_id'], cols['date']] + cols['labels']
    lab_df = full_df[label_cols].copy()
    
    # Rename label columns to match GLOBEM/CES format for pipeline compatibility
    # MentalIoT uses: phq2_result_binary, gad2_result_binary, stress_result_binary
    # Pipeline expects: phq4_anxiety_EMA, phq4_depression_EMA, stress
    lab_df = lab_df.rename(columns={
        'gad2_result_binary': 'phq4_anxiety_EMA',
        'phq2_result_binary': 'phq4_depression_EMA',
        'stress_result_binary': 'stress'
    })
    
    # Also rename in test_df and train_df for use in evaluation_runner
    test_df = test_df.rename(columns={
        'gad2_result_binary': 'phq4_anxiety_EMA',
        'phq2_result_binary': 'phq4_depression_EMA',
        'stress_result_binary': 'stress'
    })
    
    train_df = train_df.rename(columns={
        'gad2_result_binary': 'phq4_anxiety_EMA',
        'phq2_result_binary': 'phq4_depression_EMA',
        'stress_result_binary': 'stress'
    })
    
    # Also update cols['labels'] to match renamed columns
    cols['labels'] = ['phq4_anxiety_EMA', 'phq4_depression_EMA', 'stress']
    
    feat_cols = [c for c in full_df.columns if c not in ['phq2_result_binary', 'gad2_result_binary', 'stress_result_binary']]
    feat_df = full_df[feat_cols].copy()
    
    print(f"  Test set: {len(test_df)} samples from {test_df[cols['user_id']].nunique()} users")
    print(f"  Train set: {len(train_df)} samples")
    
    return feat_df, lab_df, test_df, train_df, cols


def _aggregate_ces_data(use_cols_path: str = './config/ces_use_cols.json') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Build aggregated CES data from raw sensing, steps, and EMA data."""
    import os
    from datetime import timedelta
    
    print("\n" + "="*80)
    print("BUILDING AGGREGATED CES DATASET")
    print("="*80)
    
    # Load paths
    demographic_path = '../dataset/CES/Demographics/demographics.csv'
    sensing_path = '../dataset/CES/Sensing/sensing.csv'
    steps_path = '../dataset/CES/Sensing/steps.csv'
    ema_path = '../dataset/CES/EMA/general_ema.csv'
    
    print(f"Loading raw data files...")
    demo_df = pd.read_csv(demographic_path)
    # Load sensing in chunks to handle large file
    print(f"  Loading sensing.csv (large file, may take time)...")
    sensing_df = pd.read_csv(sensing_path, low_memory=False)
    steps_df = pd.read_csv(steps_path)
    ema_df = pd.read_csv(ema_path)
    
    print(f"  Demographics: {len(demo_df)} users")
    print(f"  Sensing: {len(sensing_df)} rows")
    print(f"  Steps: {len(steps_df)} rows")
    print(f"  EMA: {len(ema_df)} rows")
    
    # Load column configuration
    with open(use_cols_path, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols['compass']

    # Convert date columns
    sensing_df['day'] = pd.to_datetime(sensing_df['day'], format='%Y%m%d', errors='coerce')
    steps_df['day'] = pd.to_datetime(steps_df['day'], format='%Y%m%d', errors='coerce')
    ema_df['day'] = pd.to_datetime(ema_df['day'], format='%Y%m%d', errors='coerce')
    
    # Merge sensing and steps
    print(f"\nMerging sensing and steps data...")
    feat_df = sensing_df.merge(steps_df, on=['uid', 'day'], how='outer')
    print(f"  Merged: {len(feat_df)} rows")
    
    # Process EMA labels
    print(f"\nProcessing EMA labels...")
    # Create derived labels: phq4_anxiety_EMA and phq4_depression_EMA
    ema_df['phq4_anxiety_EMA'] = ema_df['phq4-1'] + ema_df['phq4-2']
    ema_df['phq4_depression_EMA'] = ema_df['phq4-3'] + ema_df['phq4-4']
    
    # Keep only rows with valid labels
    ema_df = ema_df.dropna(subset=['phq4_anxiety_EMA', 'phq4_depression_EMA', 'stress'])
    print(f"  Valid EMA samples: {len(ema_df)}")
    
    # Binarize labels
    ema_df = binarize_labels(ema_df, ['phq4_anxiety_EMA', 'phq4_depression_EMA', 'stress'], cols['threshold'])
    
    # Rename user ID for readability (user1, user2, ...)
    print(f"\nRenaming user IDs to readable format...")
    unique_users = sorted(ema_df['uid'].unique())
    user_mapping = {uid: f"user{i+1}" for i, uid in enumerate(unique_users)}
    
    ema_df['uid'] = ema_df['uid'].map(user_mapping)
    feat_df['uid'] = feat_df['uid'].map(user_mapping)
    demo_df['uid'] = demo_df['uid'].map(user_mapping)
    
    print(f"  Mapped {len(unique_users)} users to user1...user{len(unique_users)}")
    
    # Build aggregated features
    print(f"\nBuilding aggregated features (28-day windows)...")
    print(f"  This may take a while...")
    
    agg_rows = []
    total_ema = len(ema_df)
    
    for count, (idx, ema_row) in enumerate(ema_df.iterrows(), 1):
        if count % 100 == 0 or count == total_ema:
            print(f"  Progress: {count}/{total_ema} ({count/total_ema*100:.1f}%)")
        
        user_id = ema_row['uid']
        ema_date = ema_row['day']
        
        # Get user's features for past 28 days
        start_date = ema_date - timedelta(days=28)
        user_feats = feat_df[
            (feat_df['uid'] == user_id) & 
            (feat_df['day'] >= start_date) & 
            (feat_df['day'] < ema_date)
        ].sort_values('day')
        
        # Must have full 28 days of data
        if len(user_feats) < 28:
            continue
        
        # Build aggregated row
        agg_row = _build_ces_aggregated_row(user_feats, user_id, ema_date, ema_row, demo_df, cols)
        
        if agg_row is not None:
            agg_rows.append(agg_row)
    
    print(f"\n  Built {len(agg_rows)} aggregated samples")
    
    # Convert to DataFrame
    agg_df = pd.DataFrame(agg_rows)
    
    # Split into features and labels
    label_cols = [cols['user_id'], cols['date']] + cols['labels']
    lab_df = agg_df[label_cols].copy()
    
    # Feature columns: everything except pure label columns
    feat_cols = [c for c in agg_df.columns if c not in cols['labels']]
    feat_df = agg_df[feat_cols].copy()
    
    print(f"\n  Final feature set: {len(feat_df.columns)-3} features")
    print(f"  Final samples: {len(feat_df)}")
    print("="*80 + "\n")
    
    return feat_df, lab_df, cols


def _build_ces_aggregated_row(user_feats: pd.DataFrame, user_id: str, ema_date: pd.Timestamp,
                               ema_row: pd.Series, demo_df: pd.DataFrame, cols: Dict) -> Optional[Dict]:
    """
    Build a single aggregated row for CES data.
    
    Aggregation includes:
    - Statistical: mean, std, min, max over 28 days
    - Structural: p2w_slope, r2w_slope (past 2 weeks, recent 2 weeks trend)
    - Semantic: weekday/weekend averages, ep1/2/3 patterns
    - Raw: before1day ~ before28day for TimeRAG retrieval
    """
    if len(user_feats) == 0:
        return None
    
    # Get gender from demographics
    user_demo = demo_df[demo_df['uid'] == user_id]
    gender = user_demo['gender'].iloc[0] if len(user_demo) > 0 else 'unknown'
    
    # Initialize aggregated row
    agg_row = {
        'uid': user_id,
        'date': ema_date,
        'gender': gender,
    }
    
    # Get feature list from config
    stat_features = cols['feature_set']['statistical']
    semantic_features = cols['feature_set']['semantic']
    
    # Process each statistical feature
    for feat_col, feat_name in stat_features.items():
        if feat_col not in user_feats.columns:
            continue
        
        values = user_feats[feat_col].values
        
        # Skip if too much missing data
        if np.sum(~np.isnan(values)) < len(values) * 0.5:
            continue
        
        # === STATISTICAL FEATURES ===
        agg_row[f"{feat_col}_28mean"] = np.nanmean(values)
        agg_row[f"{feat_col}_28std"] = np.nanstd(values)
        agg_row[f"{feat_col}_28min"] = np.nanmin(values)
        agg_row[f"{feat_col}_28max"] = np.nanmax(values)
        
        # === STRUCTURAL FEATURES ===
        # Calculate slopes for past 2 weeks and recent 2 weeks
        if len(values) >= 28:
            past_2weeks = values[:14]
            recent_2weeks = values[-14:]
            
            past_slope, _ = _calculate_normalized_slope_ces(past_2weeks)
            recent_slope, _ = _calculate_normalized_slope_ces(recent_2weeks)
            
            agg_row[f"{feat_col}_p2wslope"] = past_slope if not np.isnan(past_slope) else None
            agg_row[f"{feat_col}_r2wslope"] = recent_slope if not np.isnan(recent_slope) else None
        else:
            # If less than 28 days, compute single slope
            slope, _ = _calculate_normalized_slope_ces(values)
            agg_row[f"{feat_col}_p2wslope"] = None
            agg_row[f"{feat_col}_r2wslope"] = slope if not np.isnan(slope) else None
        
        # === RAW FEATURES (for TimeRAG retrieval) ===
        # Store last 28 days in reverse order (before1day, before2day, ...)
        for i in range(min(28, len(user_feats))):
            day_value = user_feats.iloc[-(i+1)][feat_col] if i < len(user_feats) else None
            agg_row[f"{feat_col}_before{i+1}day"] = day_value
    
    # === SEMANTIC FEATURES ===
    # Process semantic features: weekday/weekend and ep1/2/3 patterns
    
    # For sleep and ep_0 features: compute weekday/weekend
    for feat_col, feat_name in semantic_features.items():
        if feat_col not in user_feats.columns:
            continue
        
        # Identify feature type
        is_sleep = 'sleep' in feat_col
        is_ep0 = feat_col.endswith('_ep_0') or feat_col.endswith('_ep0')
        is_ep1 = feat_col.endswith('_ep_1') or feat_col.endswith('_ep1')
        is_ep2 = feat_col.endswith('_ep_2') or feat_col.endswith('_ep2')
        is_ep3 = feat_col.endswith('_ep_3') or feat_col.endswith('_ep3')
        
        # Get base feature name (without ep suffix)
        if is_ep0:
            base_col = feat_col.replace('_ep_0', '').replace('_ep0', '')
        elif is_ep1:
            base_col = feat_col.replace('_ep_1', '').replace('_ep1', '')
        elif is_ep2:
            base_col = feat_col.replace('_ep_2', '').replace('_ep2', '')
        elif is_ep3:
            base_col = feat_col.replace('_ep_3', '').replace('_ep3', '')
        else:
            base_col = feat_col
        
        # Compute weekday/weekend for sleep and ep_0 features
        if is_sleep or is_ep0:
            weekday_mask = user_feats['day'].dt.dayofweek < 5
            weekend_mask = user_feats['day'].dt.dayofweek >= 5
            
            weekday_values = user_feats.loc[weekday_mask, feat_col].values
            weekend_values = user_feats.loc[weekend_mask, feat_col].values
            
            if len(weekday_values) > 0:
                agg_row[f"{feat_col}_28weekday"] = np.nanmean(weekday_values)
            if len(weekend_values) > 0:
                agg_row[f"{feat_col}_28weekend"] = np.nanmean(weekend_values)
        
        # For ep1/2/3 features: compute 28-day mean and yesterday value
        if is_ep1 or is_ep2 or is_ep3:
            # 28-day mean
            agg_row[f"{feat_col}_28mean"] = np.nanmean(user_feats[feat_col].values)
            
            # Yesterday value (most recent day)
            if len(user_feats) > 0:
                yesterday_val = user_feats.iloc[-1][feat_col]
                agg_row[f"{feat_col}_yesterday"] = yesterday_val if pd.notna(yesterday_val) else None
    
    # Add labels
    for label in cols['labels']:
        if label in ema_row:
            agg_row[label] = ema_row[label]
        else:
            agg_row[label] = None
    
    return agg_row


def _calculate_normalized_slope_ces(values: np.ndarray) -> Tuple[float, str]:
    """
    Calculate normalized slope for CES data (same as GLOBEM implementation).
    
    Returns:
        Tuple of (normalized_slope, direction)
    """
    valid_mask = ~np.isnan(values)
    if np.sum(valid_mask) < 2:
        return np.nan, 'stable'
    
    y = values[valid_mask]
    mean_val = np.mean(y)
    
    if mean_val == 0:
        return np.nan, 'stable'
    
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    
    # Normalize by mean to make slope scale-independent
    normalized_slope = slope / abs(mean_val)
    
    # Determine direction
    if normalized_slope > 0.05:  # 5% increase per day
        direction = 'increasing'
    elif normalized_slope < -0.05:  # 5% decrease per day
        direction = 'decreasing'
    else:
        direction = 'stable'
    
    return normalized_slope, direction


def sample_ces_testset(
    n_users: int = 60,
    min_ema_per_user: int = 9,
    samples_per_user: int = 5,
    random_state: Optional[int] = None,
    use_cols_path: str = './config/ces_use_cols.json'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Sample test set from CES dataset with gender-balanced user selection.
    
    Args:
        n_users: Number of users to sample (default: 60)
        min_ema_per_user: Minimum number of EMA samples required (default: 9, to allow 4 ICL + 5 test)
        samples_per_user: Number of samples per user for testset (default: 5)
        random_state: Random seed for reproducibility
        use_cols_path: Path to column configuration
        
    Returns:
        Tuple of (feat_df_all, test_df, train_df, cols)
        - feat_df_all: Full feature dataframe (for ICL retrieval)
        - test_df: Test samples (n_users * samples_per_user rows)
        - train_df: Training samples (remaining samples for ICL)
        - cols: Column configuration
    """
    print("\n" + "="*80)
    print("CES TESTSET SAMPLING (GENDER-BALANCED)")
    print("="*80)
    print(f"  Target users: {n_users}")
    print(f"  Min EMA per user: {min_ema_per_user}")
    print(f"  Samples per user: {samples_per_user}")
    if random_state:
        print(f"  Random seed: {random_state}")
    print("="*80 + "\n")
    
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()
    
    # Load CES data (aggregated)
    feat_df, lab_df, cols = load_ces_data(use_cols_path)
    
    # Count EMAs per user
    user_ema_counts = lab_df.groupby(cols['user_id']).size()
    
    # Filter users with sufficient EMAs
    users_with_emas = user_ema_counts[user_ema_counts >= min_ema_per_user].index.tolist()
    
    print(f"Total users in dataset: {lab_df[cols['user_id']].nunique()}")
    print(f"Users with >= {min_ema_per_user} EMAs: {len(users_with_emas)}")
    
    if len(users_with_emas) < n_users:
        print(f"[WARNING] Only {len(users_with_emas)} eligible users, requested {n_users}")
        print(f"          Using all {len(users_with_emas)} eligible users")
        n_users = len(users_with_emas)
    
    # Get gender information
    user_genders = feat_df.groupby(cols['user_id'])['gender'].first()
    
    # Count gender distribution
    gender_counts = user_genders[user_genders.index.isin(users_with_emas)].value_counts()
    print(f"\nGender distribution in eligible users:")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count} ({count/len(users_with_emas)*100:.1f}%)")
    
    # Sample users proportionally by gender
    selected_users = []
    for gender, count in gender_counts.items():
        # Calculate proportional allocation
        gender_users = user_genders[
            (user_genders == gender) & (user_genders.index.isin(users_with_emas))
        ].index.tolist()
        
        # Proportional number of users to sample from this gender
        n_from_gender = int(n_users * (count / len(users_with_emas)))
        
        # Ensure we don't request more than available
        n_from_gender = min(n_from_gender, len(gender_users))
        
        # Sample
        sampled = rng.choice(gender_users, size=n_from_gender, replace=False)
        selected_users.extend(sampled.tolist())
    
    # If we haven't reached n_users yet (due to rounding), sample more
    if len(selected_users) < n_users:
        remaining_users = [u for u in users_with_emas if u not in selected_users]
        n_remaining = n_users - len(selected_users)
        if len(remaining_users) >= n_remaining:
            additional = rng.choice(remaining_users, size=n_remaining, replace=False)
            selected_users.extend(additional.tolist())
    
    print(f"\nSelected {len(selected_users)} users")
    
    # Verify gender balance
    selected_genders = user_genders[user_genders.index.isin(selected_users)].value_counts()
    print(f"Gender distribution in selected users:")
    for gender, count in selected_genders.items():
        print(f"  {gender}: {count} ({count/len(selected_users)*100:.1f}%)")
    
    # For each selected user, sample test samples with stratified sampling
    print(f"\nSampling {samples_per_user} test samples per user (from 5th EMA onwards)...")
    
    # Build candidate pool: for each user, collect samples from 5th EMA onwards
    candidate_pool = []
    
    for user_id in selected_users:
        user_labs = lab_df[lab_df[cols['user_id']] == user_id].sort_values(cols['date'])
        
        # Only consider samples from 5th EMA onwards (index 4+) to ensure at least 4 samples for ICL
        if len(user_labs) >= 5:
            eligible_samples = user_labs.iloc[4:]  # Skip first 4 samples
            
            # Add to candidate pool with stratification key
            for idx in eligible_samples.index:
                row = eligible_samples.loc[idx]
                # Create stratification key: anxiety_depression_stress (e.g., "0_1_1")
                strat_key = f"{row[cols['labels'][0]]}_{row[cols['labels'][1]]}_{row[cols['labels'][2]]}"
                candidate_pool.append({
                    'index': idx,
                    'user_id': user_id,
                    'strat_key': strat_key
                })
    
    # Group candidates by user and stratification key
    from collections import defaultdict
    user_strat_candidates = defaultdict(lambda: defaultdict(list))
    
    for candidate in candidate_pool:
        user_strat_candidates[candidate['user_id']][candidate['strat_key']].append(candidate['index'])
    
    # For each user, sample samples_per_user trying to maintain stratification
    test_indices = []
    
    for user_id in selected_users:
        user_candidates = user_strat_candidates[user_id]
        
        if not user_candidates:
            # No eligible samples (< 5 EMAs), skip this user
            print(f"  [Warning] User {user_id} has < 5 EMAs, skipping")
            continue
        
        # Get all strat keys for this user
        all_strat_keys = list(user_candidates.keys())
        
        # Try to sample proportionally from each strat group
        sampled = []
        
        if len(all_strat_keys) == 1:
            # Only one strat group, sample randomly
            indices = user_candidates[all_strat_keys[0]]
            n_to_sample = min(samples_per_user, len(indices))
            sampled = rng.choice(indices, size=n_to_sample, replace=False).tolist()
        else:
            # Multiple strat groups, try to sample proportionally
            total_available = sum(len(indices) for indices in user_candidates.values())
            
            for strat_key in all_strat_keys:
                indices = user_candidates[strat_key]
                # Proportional allocation
                proportion = len(indices) / total_available
                n_from_strat = max(1, int(samples_per_user * proportion))  # At least 1 from each group
                n_from_strat = min(n_from_strat, len(indices))
                
                sampled_from_strat = rng.choice(indices, size=n_from_strat, replace=False).tolist()
                sampled.extend(sampled_from_strat)
            
            # If we have too many, randomly remove some
            if len(sampled) > samples_per_user:
                sampled = rng.choice(sampled, size=samples_per_user, replace=False).tolist()
            # If we have too few, add more from largest group
            elif len(sampled) < samples_per_user:
                n_needed = samples_per_user - len(sampled)
                # Find largest group with remaining samples
                for strat_key in sorted(all_strat_keys, key=lambda k: len(user_candidates[k]), reverse=True):
                    remaining = [idx for idx in user_candidates[strat_key] if idx not in sampled]
                    if remaining:
                        additional = min(n_needed, len(remaining))
                        sampled.extend(rng.choice(remaining, size=additional, replace=False).tolist())
                        n_needed -= additional
                        if n_needed == 0:
                            break
        
        test_indices.extend(sampled)
    
    # Split into test and train
    test_df = lab_df.loc[test_indices].copy()
    train_df = lab_df[~lab_df.index.isin(test_indices)].copy()
    
    print(f"\nTest set: {len(test_df)} samples from {len([u for u in selected_users if u in test_df[cols['user_id']].values])} users")
    print(f"Train set: {len(train_df)} samples")
    
    # Print stratification distribution in test set
    if len(test_df) > 0:
        print(f"\nTest set stratification (Anxiety_Depression_Stress):")
        test_strat_keys = test_df.apply(
            lambda row: f"{row[cols['labels'][0]]}_{row[cols['labels'][1]]}_{row[cols['labels'][2]]}", 
            axis=1
        )
        strat_counts = test_strat_keys.value_counts().sort_index()
        for strat_key, count in strat_counts.items():
            print(f"  {strat_key}: {count} ({count/len(test_df)*100:.1f}%)")
    print(f"  (Train set used for ICL example selection)")
    print("="*80 + "\n")
    
    return feat_df, test_df, train_df, cols


def load_globem_data(institution: str = 'INS-W_2', target: str = 'compass',
                    use_cols_path: str = './config/globem_use_cols.json') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load GLOBEM dataset with feature and label data."""
    feat_path = f'../dataset/Globem/{institution}/FeatureData/rapids.csv'
    lab_path = f'../dataset/Globem/{institution}/SurveyData/ema.csv'
    
    feat_df = pd.read_csv(feat_path, low_memory=False)
    lab_df = pd.read_csv(lab_path)
    
    with open(use_cols_path, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols[target]
    
    # Collect all required columns
    feat_cols = [cols['user_id'], cols['date']]
    
    # Handle different feature_set structures
    if isinstance(cols['feature_set'], dict):
        # Compass format: statistical, structural, semantic, temporal_descriptor
        for category, features in cols['feature_set'].items():
            if isinstance(features, dict):
                feat_cols.extend(features.keys())
            elif features != "None":
                feat_cols.append(features)
    elif isinstance(cols['feature_set'], list):
        # Legacy format: simple list
        feat_cols.extend(cols['feature_set'])
    
    # Remove duplicates while preserving order
    feat_cols = list(dict.fromkeys(feat_cols))
    
    lab_cols = [cols['user_id'], cols['date']] + cols['labels']
    
    # Select available columns only
    available_feat_cols = [col for col in feat_cols if col in feat_df.columns]
    feat_df = feat_df[available_feat_cols].copy()
    lab_df = lab_df[lab_cols].copy()
    
    # Convert dates
    feat_df[cols['date']] = pd.to_datetime(feat_df[cols['date']])
    lab_df[cols['date']] = pd.to_datetime(lab_df[cols['date']])
    
    # Binarize labels
    lab_df = binarize_labels(lab_df, cols['labels'], cols['threshold'])
    
    print(f"Loaded {len(available_feat_cols)-2} feature columns for target '{target}'")
    
    return feat_df, lab_df, cols


def filter_testset_by_historical_labels(lab_df: pd.DataFrame, cols: Dict, 
                                         min_historical: int = 4) -> pd.DataFrame:
    """
    Filter label dataframe to only include samples with sufficient historical labels.
    
    This ensures that personalized ICL selection always has enough examples,
    enabling fair comparison across generalized, personalized, and hybrid strategies.
    
    Args:
        lab_df: Label DataFrame
        cols: Column configuration
        min_historical: Minimum number of historical labels required before the sample date
    
    Returns:
        Filtered DataFrame with only samples that have >= min_historical prior labels
    """
    print(f"\n[Filtering test set: requiring >= {min_historical} historical labels per user...]")
    
    user_id_col = cols['user_id']
    date_col = cols['date']
    
    # Sort by user and date
    lab_df_sorted = lab_df.sort_values([user_id_col, date_col]).reset_index(drop=True)
    
    valid_indices = []
    
    # Group by user
    for user_id, user_group in lab_df_sorted.groupby(user_id_col):
        user_dates = user_group[date_col].values
        user_indices = user_group.index.tolist()
        
        # For each sample, count how many prior labels exist
        for i, (idx, date) in enumerate(zip(user_indices, user_dates)):
            n_historical = i  # i = 0 means 0 prior samples
            
            if n_historical >= min_historical:
                valid_indices.append(idx)
    
    # Filter by valid indices
    filtered_df = lab_df_sorted.loc[valid_indices].copy()
    
    print(f"  Original samples: {len(lab_df)}")
    print(f"  Filtered samples: {len(filtered_df)}")
    print(f"  Unique users in filtered set: {filtered_df[user_id_col].nunique()}")
    print(f"  Reduction: {(1 - len(filtered_df)/len(lab_df))*100:.1f}%")
    
    if len(filtered_df) == 0:
        raise ValueError(f"No samples remain after filtering for {min_historical} historical labels")
    
    return filtered_df


def sample_multiinstitution_testset(
    institutions_config: Dict[str, int],
    min_ema_per_user: int = 10,
    samples_per_user: int = 3,
    random_state: Optional[int] = None,
    use_cols_path: str = './config/globem_use_cols.json',
    target: str = 'compass'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Sample testset from multiple institutions with specified number of users per institution.
    For each selected user, extract the last N EMA samples.
    
    Args:
        institutions_config: Dict mapping institution names to number of users to sample
                           e.g., {'INS-W_2': 65, 'INS-W_3': 28, 'INS-W_4': 45}
        min_ema_per_user: Minimum number of EMA samples required per user
        samples_per_user: Number of last EMA samples to use per user (default: 3)
        random_state: Random seed for reproducibility
        use_cols_path: Path to column configuration
        target: Target feature set name
    
    Returns:
        Tuple of (feat_df, lab_df, cols) with combined data from all institutions
    """
    print("\n" + "="*80)
    print("MULTI-INSTITUTION TESTSET SAMPLING")
    print("="*80)
    print(f"  Configuration:")
    for inst, n_users in institutions_config.items():
        print(f"    {inst}: {n_users} users")
    print(f"  Min EMA per user: {min_ema_per_user}")
    print(f"  Samples per user: {samples_per_user} (last EMAs)")
    if random_state:
        print(f"  Random seed: {random_state}")
    print("="*80 + "\n")
    
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()
    
    all_feat_dfs = []
    all_lab_dfs = []
    total_users_selected = 0
    total_samples_collected = 0
    
    # Load column configuration
    with open(use_cols_path, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols[target]
    
    # Process each institution
    for institution, n_users_target in institutions_config.items():
        print(f"\n[Processing {institution}...]")
        
        # Load institution data
        feat_path = f'../dataset/Globem/{institution}/FeatureData/rapids.csv'
        lab_path = f'../dataset/Globem/{institution}/SurveyData/ema.csv'
        
        feat_df_inst = pd.read_csv(feat_path, low_memory=False)
        lab_df_inst = pd.read_csv(lab_path)
        
        # Ensure correct column names (handle pid/uid variations)
        user_col = cols['user_id']
        # if user_col not in feat_df_inst.columns:
        #     # Try alternative column names
        #     if 'pid' in feat_df_inst.columns:
        #         feat_df_inst.rename(columns={'pid': user_col}, inplace=True)
        #     elif 'uid' in feat_df_inst.columns and user_col != 'uid':
        #         feat_df_inst.rename(columns={'uid': user_col}, inplace=True)
        
        # if user_col not in lab_df_inst.columns:
        #     # Try alternative column names
        #     if 'pid' in lab_df_inst.columns:
        #         lab_df_inst.rename(columns={'pid': user_col}, inplace=True)
        #     elif 'uid' in lab_df_inst.columns and user_col != 'uid':
        #         lab_df_inst.rename(columns={'uid': user_col}, inplace=True)
        
        # Convert dates
        feat_df_inst[cols['date']] = pd.to_datetime(feat_df_inst[cols['date']])
        lab_df_inst[cols['date']] = pd.to_datetime(lab_df_inst[cols['date']])
        
        # Binarize labels
        lab_df_inst = binarize_labels(lab_df_inst, cols['labels'], cols['threshold'])
        
        # Count EMAs per user
        user_ema_counts = lab_df_inst.groupby(user_col).size()
        
        # Filter users with sufficient EMAs
        users_with_emas = user_ema_counts[user_ema_counts >= min_ema_per_user].index.tolist()
        
        print(f"  Total users: {lab_df_inst[user_col].nunique()}")
        print(f"  Users with >= {min_ema_per_user} EMAs: {len(users_with_emas)}")
        
        # Further filter: check if users have sufficient sensor data for their last N samples
        print(f"  Checking sensor data availability for last {samples_per_user} EMA samples...")
        eligible_users = []
        
        for user_id in users_with_emas:
            user_labs = lab_df_inst[lab_df_inst[user_col] == user_id].sort_values(cols['date'])
            last_n_samples = user_labs.tail(samples_per_user)
            
            # Check if user has sensor features for these dates
            has_sufficient_data = True
            for _, sample in last_n_samples.iterrows():
                ema_date = sample[cols['date']]
                user_feat = feat_df_inst[
                    (feat_df_inst[user_col] == user_id) & 
                    (feat_df_inst[cols['date']] < ema_date)
                ]
                
                if len(user_feat) == 0:
                    has_sufficient_data = False
                    break
            
            if has_sufficient_data:
                eligible_users.append(user_id)
        
        print(f"  Eligible users (with sensor data): {len(eligible_users)}")
        
        if len(eligible_users) < n_users_target:
            print(f"  [WARNING] Only {len(eligible_users)} eligible users, requested {n_users_target}")
            print(f"            Using all {len(eligible_users)} eligible users")
            n_users_target = len(eligible_users)
        
        # Randomly select users from eligible pool
        selected_users = rng.choice(eligible_users, size=n_users_target, replace=False)
        print(f"  Selected {len(selected_users)} users: {selected_users[:5]}..." if len(selected_users) > 5 else f"  Selected {len(selected_users)} users: {selected_users}")
        
        # For each selected user, get last N EMA samples
        inst_testset_indices = []
        for user_id in selected_users:
            user_labs = lab_df_inst[lab_df_inst[user_col] == user_id].sort_values(cols['date'])
            last_n_samples = user_labs.tail(samples_per_user)
            inst_testset_indices.extend(last_n_samples.index.tolist())
        
        # Filter dataframes for testset
        lab_df_testset = lab_df_inst.loc[inst_testset_indices].copy()
        
        # Add institution column for tracking
        lab_df_testset['institution'] = institution
        feat_df_inst['institution'] = institution
        
        # Store full feature data and only testset labels
        all_feat_dfs.append(feat_df_inst)
        all_lab_dfs.append(lab_df_testset)
        
        total_users_selected += len(selected_users)
        total_samples_collected += len(lab_df_testset)
        
        print(f"  [OK] Collected {len(lab_df_testset)} samples from {len(selected_users)} users")
    
    # Combine all institutions
    combined_feat_df = pd.concat(all_feat_dfs, ignore_index=True)
    combined_lab_df = pd.concat(all_lab_dfs, ignore_index=True)
    
    # Select only required feature columns
    feat_cols = [cols['user_id'], cols['date'], 'institution']
    if isinstance(cols['feature_set'], dict):
        for category, features in cols['feature_set'].items():
            if isinstance(features, dict):
                feat_cols.extend(features.keys())
            elif features != "None":
                feat_cols.append(features)
    elif isinstance(cols['feature_set'], list):
        feat_cols.extend(cols['feature_set'])
    
    # Remove duplicates and select available columns
    feat_cols = list(dict.fromkeys(feat_cols))
    available_feat_cols = [col for col in feat_cols if col in combined_feat_df.columns]
    combined_feat_df = combined_feat_df[available_feat_cols].copy()
    
    print("\n" + "="*80)
    print("TESTSET SUMMARY")
    print("="*80)
    print(f"  Total users selected: {total_users_selected}")
    print(f"  Total testset samples: {total_samples_collected}")
    print(f"  Samples per institution:")
    for inst in institutions_config.keys():
        inst_count = len(combined_lab_df[combined_lab_df['institution'] == inst])
        inst_users = combined_lab_df[combined_lab_df['institution'] == inst][cols['user_id']].nunique()
        print(f"    {inst}: {inst_count} samples from {inst_users} users")
    print(f"  Note: ICL examples will be drawn from each user's historical data (before testset)")
    print("="*80 + "\n")
    
    return combined_feat_df, combined_lab_df, cols


def sample_batch_stratified(feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict, n_samples: int,
                            random_state: Optional[int] = None, max_attempts: int = 5000, dataset: str = 'globem') -> list[Dict]:
    """
    Sample a batch of data points with stratified sampling across ALL labels.
    Combines all labels into stratification groups (e.g., anxiety_depression: 0_0, 0_1, 1_0, 1_1).
    Uses aggregation settings from config.py.
    
    Args:
        feat_df: Feature dataframe
        lab_df: Label dataframe
        cols: Column configuration
        n_samples: Number of samples to collect
        random_state: Random seed for reproducibility
        max_attempts: Maximum sampling attempts
    
    Returns:
        List of sample dictionaries
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()
    
    # Create combined stratification key from ALL labels
    # Example: anxiety=0, depression=1 -> "0_1"
    label_cols = cols['labels']
    lab_df_copy = lab_df.copy()
    
    # Combine all label values with underscore
    strat_key = lab_df_copy[label_cols[0]].astype(str)
    for label in label_cols[1:]:
        strat_key = strat_key + "_" + lab_df_copy[label].astype(str)
    
    lab_df_copy['_stratification_key'] = strat_key
    
    # Calculate samples per stratification group
    class_counts = lab_df_copy['_stratification_key'].value_counts()
    classes = class_counts.index.tolist()
    
    # Handle case when n_samples < number of groups
    if n_samples < len(classes):
        # Sample only from the largest groups
        print(f"[Warning] n_samples ({n_samples}) < number of groups ({len(classes)})")
        print(f"   Sampling from {n_samples} largest groups only")
        largest_groups = class_counts.nlargest(n_samples).index.tolist()
        samples_per_class = {cls: 1 for cls in largest_groups}
    else:
        # Proportional allocation
        samples_per_class = {}
        for cls in classes:
            proportion = class_counts[cls] / len(lab_df_copy)
            samples_per_class[cls] = max(0, int(n_samples * proportion))
        
        # Distribute remaining samples to largest groups
        allocated = sum(samples_per_class.values())
        remaining = n_samples - allocated
        
        if remaining > 0:
            # Sort by frequency and allocate remaining
            sorted_classes = class_counts.index.tolist()
            for i in range(remaining):
                samples_per_class[sorted_classes[i % len(sorted_classes)]] += 1
        elif remaining < 0:
            # Remove from smallest groups (shouldn't happen but safe)
            sorted_by_allocated = sorted(samples_per_class.items(), key=lambda x: (x[1], -class_counts[x[0]]), reverse=True)
            for cls, count in sorted_by_allocated:
                if remaining >= 0:
                    break
                if count > 0:
                    remove = min(count, -remaining)
                    samples_per_class[cls] -= remove
                    remaining += remove
    
    # Print stratification info
    label_names = "_".join([l.replace('_EMA', '').replace('phq4_', '') for l in label_cols])
    print(f"Stratified sampling by all labels ({label_names}):")
    for cls, count in sorted(samples_per_class.items()):
        if count > 0:  # Only print groups with samples
            print(f"  {cls}: {count} samples ({count/n_samples*100:.1f}%)")
    
    # Sample from each stratification group
    collected_samples, attempted_indices = [], set()
    
    for cls, n_class_samples in samples_per_class.items():
        if n_class_samples <= 0:  # Skip groups with no samples
            continue
        
        cls_indices = lab_df_copy[lab_df_copy['_stratification_key'] == cls].index.tolist()
        
        if len(cls_indices) < n_class_samples:
            print(f"[Warning] Only {len(cls_indices)} samples for class {cls}, requested {n_class_samples}")
            n_class_samples = len(cls_indices)
        
        class_samples_collected, attempts = 0, 0
        
        while class_samples_collected < n_class_samples and attempts < max_attempts:
            idx = rng.choice(cls_indices)
            
            if idx in attempted_indices:
                attempts += 1
                continue
            
            attempted_indices.add(idx)
            
            sample_lab = lab_df.loc[[idx]]
            user_id = sample_lab.iloc[0][cols['user_id']]
            ema_date = sample_lab.iloc[0][cols['date']]
            
            # Get aggregated features (different for CES/MentalIoT vs GLOBEM)
            if dataset in ['ces', 'mentaliot']:
                # For CES/MentalIoT, aggregated_features is already in feat_df
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
                agg_feats = aggregate_window_features(feat_df, user_id, ema_date, cols)
                if agg_feats is None or not check_missing_ratio(agg_feats):
                    attempts += 1
                    continue
            
            labels = sample_lab[cols['labels']].iloc[0].to_dict()
            
            collected_samples.append({
                'aggregated_features': agg_feats, 'labels': labels,
                'user_id': user_id, 'ema_date': ema_date
            })
            
            class_samples_collected += 1
            attempts += 1
    
    if len(collected_samples) < n_samples * 0.75:
        raise ValueError(f"Could not collect enough valid samples. Requested: {n_samples}, Collected: {len(collected_samples)}")
    
    print(f"Successfully collected {len(collected_samples)} stratified samples")
    return collected_samples