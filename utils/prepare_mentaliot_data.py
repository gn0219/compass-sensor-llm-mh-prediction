"""
Prepare MentalIoT dataset for mental health prediction.

This script:
1. Loads preprocessed_1st.csv
2. Aggregates IoT sensor data (aqara total usage)
3. Selects and organizes features into statistical/structural/semantic categories
4. Creates aggregated_mentaliot.csv for easy loading
5. Implements testset sampling (20 users × 10 samples, from 5th EMA onwards, stratified)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_mentaliot_raw(data_path: str = '../dataset/MentalIoT/preprocessed_1st.csv'):
    """Load raw MentalIoT data."""
    print(f"Loading MentalIoT data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples from {df['uid'].nunique()} users")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Total columns: {len(df.columns)}")
    return df


def aggregate_iot_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate IoT sensor data.
    
    Creates:
    - aqara_total: Total count of all aqara appliance activations in past hour
    """
    print("\nAggregating IoT sensor data...")
    
    # List of aqara appliance columns (before_60min)
    aqara_cols = [col for col in df.columns if col.startswith('aqara_') and col.endswith('_before_60min')]
    # Exclude deviation and comparison columns
    aqara_cols = [col for col in aqara_cols if '_deviation' not in col and '_comparison' not in col]
    
    print(f"  Found {len(aqara_cols)} aqara appliance types: {aqara_cols}")
    
    # Sum all aqara activations
    df['aqara_total'] = df[aqara_cols].fillna(0).sum(axis=1)
    
    print(f"  Created aqara_total (mean: {df['aqara_total'].mean():.2f})")
    
    return df


def create_aggregated_mentaliot(df: pd.DataFrame, use_cols_path: str = './config/mentaliot_use_cols.json') -> pd.DataFrame:
    """
    Create aggregated MentalIoT dataset.
    
    Unlike CES/GLOBEM, MentalIoT data is already aggregated with statistics (AVG, STD, KUR, etc.)
    across time windows (ImmediatePast_60, Yesterday{Dawn/Morning/Afternoon/Evening}).
    
    We simply select and rename columns for consistency with our pipeline.
    """
    print("\nCreating aggregated MentalIoT dataset...")
    
    # Load feature configuration
    with open(use_cols_path, 'r') as f:
        cols = json.load(f)['compass']
    
    # Start with basic columns
    agg_df = pd.DataFrame()
    agg_df['uid'] = df['uid']
    agg_df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Add labels (already binary)
    for label in cols['labels']:
        if label in df.columns:
            agg_df[label] = df[label]
    
    # Add statistical features (direct mapping)
    print("\n  Adding statistical features...")
    stat_features = cols['feature_set']['statistical']
    for col_name, feature_desc in stat_features.items():
        if col_name in df.columns:
            agg_df[col_name] = df[col_name]
            print(f"    [+] {col_name}: {feature_desc}")
        else:
            print(f"    [X] {col_name}: NOT FOUND in raw data")
    
    # Add structural features (entropy, kurtosis, trend indicators)
    print("\n  Adding structural features...")
    struct_features = cols['feature_set']['structural']
    for col_name, feature_desc in struct_features.items():
        if col_name in df.columns:
            agg_df[col_name] = df[col_name]
            print(f"    [+] {col_name}: {feature_desc}")
        else:
            print(f"    [X] {col_name}: NOT FOUND in raw data")
    
    # Add semantic features (time-of-day patterns, transitions)
    print("\n  Adding semantic features...")
    semantic_features = cols['feature_set']['semantic']
    for col_name, feature_desc in semantic_features.items():
        if col_name in df.columns:
            agg_df[col_name] = df[col_name]
            print(f"    [+] {col_name}: {feature_desc}")
        else:
            print(f"    [X] {col_name}: NOT FOUND in raw data")
    
    print(f"\n  Created aggregated dataset: {len(agg_df)} samples, {len(agg_df.columns)} features")
    return agg_df


def sample_mentaliot_testset(
    agg_df: pd.DataFrame,
    cols: dict,
    n_samples_per_user: int = 10,
    min_ema_per_user: int = 14,  # At least 14 to ensure 5th+ EMAs are testable
    random_state: int = 42
) -> tuple:
    """
    Sample MentalIoT testset.
    
    Strategy:
    - Use all 20 users (no user sampling needed)
    - For each user, sample 10 EMAs from the 5th EMA onwards
    - Apply stratified sampling based on (depression, anxiety, stress) labels
    - Total: 20 users × 10 samples = 200 samples
    
    Returns:
        (testset_df, trainset_df, cols_dict)
    """
    print("\n" + "="*80)
    print("MENTALIOT TESTSET SAMPLING")
    print("="*80)
    
    np.random.seed(random_state)
    
    # Group by user
    users = agg_df['uid'].unique()
    print(f"\nTotal users: {len(users)}")
    
    # Count EMAs per user
    user_ema_counts = agg_df.groupby('uid').size()
    valid_users = user_ema_counts[user_ema_counts >= min_ema_per_user].index.tolist()
    
    print(f"Users with >= {min_ema_per_user} EMAs: {len(valid_users)}")
    
    if len(valid_users) < 20:
        print(f"  WARNING: Only {len(valid_users)} users available (target: 20)")
    else:
        print(f"  All 20 users have sufficient EMAs")
    
    # Sample testset
    test_indices = []
    
    for user_id in valid_users:
        user_df = agg_df[agg_df['uid'] == user_id].sort_values('timestamp').reset_index(drop=True)
        
        # Only consider EMAs from index 4 onwards (5th EMA+)
        eligible_indices = user_df.index[4:].tolist()
        
        if len(eligible_indices) < n_samples_per_user:
            print(f"  WARNING: User {user_id} has only {len(eligible_indices)} eligible EMAs (need {n_samples_per_user})")
            # Take all available
            sampled_indices = eligible_indices
        else:
            # Stratified sampling based on labels
            label_cols = cols['labels']
            
            # Create stratification key
            user_df['strat_key'] = user_df.apply(
                lambda row: f"{int(row[label_cols[0]])}_{int(row[label_cols[1]])}_{int(row[label_cols[2]])}",
                axis=1
            )
            
            # Get eligible samples
            eligible_df = user_df.iloc[eligible_indices]
            
            # Group by stratification key
            strat_groups = eligible_df.groupby('strat_key')
            
            # Proportional sampling
            sampled_indices = []
            for strat_key, group in strat_groups:
                n_from_group = max(1, int(n_samples_per_user * len(group) / len(eligible_df)))
                n_from_group = min(n_from_group, len(group))
                
                group_sample = group.sample(n=n_from_group, random_state=random_state)
                sampled_indices.extend(group_sample.index.tolist())
            
            # If we have too many, randomly remove some
            if len(sampled_indices) > n_samples_per_user:
                sampled_indices = np.random.choice(sampled_indices, n_samples_per_user, replace=False).tolist()
            # If we have too few, add more from largest group
            elif len(sampled_indices) < n_samples_per_user:
                remaining = n_samples_per_user - len(sampled_indices)
                available = [idx for idx in eligible_indices if idx not in sampled_indices]
                if available:
                    additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                    sampled_indices.extend(additional)
        
        # Map back to original DataFrame indices
        original_indices = user_df.loc[sampled_indices, 'index'].tolist() if 'index' in user_df.columns else user_df.index[sampled_indices].tolist()
        test_indices.extend(agg_df[agg_df['uid'] == user_id].iloc[sampled_indices].index.tolist())
    
    # Split into test and train
    test_df = agg_df.loc[test_indices].copy().reset_index(drop=True)
    train_df = agg_df[~agg_df.index.isin(test_indices)].copy().reset_index(drop=True)
    
    print(f"\nTest set: {len(test_df)} samples from {test_df['uid'].nunique()} users")
    print(f"Train set: {len(train_df)} samples")
    
    # Print stratification distribution
    if len(test_df) > 0:
        label_cols = cols['labels']
        print(f"\nTest set stratification (Depression_Anxiety_Stress):")
        test_df['strat_key'] = test_df.apply(
            lambda row: f"{int(row[label_cols[0]])}_{int(row[label_cols[1]])}_{int(row[label_cols[2]])}",
            axis=1
        )
        strat_counts = test_df['strat_key'].value_counts().sort_index()
        for strat_key, count in strat_counts.items():
            print(f"  {strat_key}: {count} ({count/len(test_df)*100:.1f}%)")
        test_df = test_df.drop(columns=['strat_key'])
    
    print("="*80 + "\n")
    
    return test_df, train_df, cols


if __name__ == '__main__':
    # Paths
    raw_data_path = '../dataset/MentalIoT/preprocessed_1st.csv'
    output_path = '../dataset/MentalIoT/aggregated_mentaliot.csv'
    use_cols_path = './config/mentaliot_use_cols.json'
    
    print("="*80)
    print("MENTALIOT DATA PREPARATION")
    print("="*80)
    
    # Load raw data
    df = load_mentaliot_raw(raw_data_path)
    
    # Aggregate IoT sensors
    df = aggregate_iot_sensors(df)
    
    # Create aggregated dataset
    agg_df = create_aggregated_mentaliot(df, use_cols_path)
    
    # Save aggregated dataset
    print(f"\nSaving aggregated dataset to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(output_path, index=False)
    print(f"  [OK] Saved {len(agg_df)} samples")
    
    # Sample testset
    with open(use_cols_path, 'r') as f:
        cols = json.load(f)['compass']
    
    test_df, train_df, cols = sample_mentaliot_testset(agg_df, cols, random_state=42)
    
    # Save testset and trainset
    test_path = output_path.parent / 'mentaliot_testset.csv'
    train_path = output_path.parent / 'mentaliot_trainset.csv'
    
    test_df.to_csv(test_path, index=False)
    train_df.to_csv(train_path, index=False)
    
    print(f"  [OK] Saved testset: {test_path} ({len(test_df)} samples)")
    print(f"  [OK] Saved trainset: {train_path} ({len(train_df)} samples)")
    
    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print("="*80)

