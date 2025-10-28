"""
Data Utility Functions

Functions for testset sampling, filtering, and other data operations.
Extracted from sensor_transformation.py for better code organization.
"""

import json
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
                            random_state: Optional[int] = None, max_attempts: int = 5000) -> list[Dict]:
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