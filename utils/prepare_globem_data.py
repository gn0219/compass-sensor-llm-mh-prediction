#!/usr/bin/env python3
"""
Prepare GLOBEM Dataset

1. Creates pre-sampled testset and trainset for GLOBEM dataset (multi-institution)
2. Pre-aggregates features for all samples (like CES and MentalIoT)

Similar to prepare_ces_data.py and prepare_mentaliot_data.py for consistency.

Usage:
    python prepare_globem_data.py [--seed SEED] [--skip-aggregation]
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
MULTI_INSTITUTION_CONFIG = {
    'INS-W_2': 65,
    'INS-W_3': 28,
    'INS-W_4': 45
}
MIN_EMA_PER_USER = 10
SAMPLES_PER_USER = 3  # Last N EMA samples per user
USE_COLS_PATH = './config/globem_use_cols.json'
TARGET = 'compass'
OUTPUT_DIR = '../dataset/Globem'


def binarize_labels(df, labels, thresholds):
    """Binarize labels based on thresholds."""
    df = df.copy()
    for label in labels:
        if label in df.columns:
            df[label] = (df[label] > thresholds[label]).astype(int)
    return df


def prepare_globem_testset(seed=42):
    """
    Prepare GLOBEM testset from multiple institutions.
    
    For each institution:
    - Sample N users with >= MIN_EMA_PER_USER EMAs
    - For each selected user, take the last SAMPLES_PER_USER EMAs as testset
    - Remaining EMAs become trainset (for ICL)
    """
    print("\n" + "="*80)
    print("GLOBEM TESTSET PREPARATION")
    print("="*80)
    print(f"Configuration:")
    for inst, n_users in MULTI_INSTITUTION_CONFIG.items():
        print(f"  {inst}: {n_users} users")
    print(f"  Min EMA per user: {MIN_EMA_PER_USER}")
    print(f"  Samples per user: {SAMPLES_PER_USER} (last EMAs)")
    print(f"  Random seed: {seed}")
    print("="*80 + "\n")
    
    rng = np.random.RandomState(seed)
    
    # Load column configuration
    with open(USE_COLS_PATH, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols[TARGET]
    
    all_test_dfs = []
    all_train_dfs = []
    
    # Process each institution
    for institution, n_users_target in MULTI_INSTITUTION_CONFIG.items():
        print(f"\n[Processing {institution}...]")
        
        # Load institution data
        feat_path = f'../dataset/Globem/{institution}/FeatureData/rapids.csv'
        lab_path = f'../dataset/Globem/{institution}/SurveyData/ema.csv'
        
        if not os.path.exists(feat_path) or not os.path.exists(lab_path):
            print(f"  [ERROR] Data files not found for {institution}")
            print(f"    Feature: {feat_path}")
            print(f"    Label: {lab_path}")
            continue
        
        feat_df = pd.read_csv(feat_path, low_memory=False)
        lab_df = pd.read_csv(lab_path)
        
        # Convert dates
        feat_df[cols['date']] = pd.to_datetime(feat_df[cols['date']])
        lab_df[cols['date']] = pd.to_datetime(lab_df[cols['date']])
        
        # Binarize labels
        lab_df = binarize_labels(lab_df, cols['labels'], cols['threshold'])
        
        # Count EMAs per user
        user_ema_counts = lab_df.groupby(cols['user_id']).size()
        
        # Filter users with sufficient EMAs
        users_with_emas = user_ema_counts[user_ema_counts >= MIN_EMA_PER_USER].index.tolist()
        
        print(f"  Total users: {lab_df[cols['user_id']].nunique()}")
        print(f"  Users with >= {MIN_EMA_PER_USER} EMAs: {len(users_with_emas)}")
        
        # Further filter: check if users have sensor data for their last N samples
        print(f"  Checking sensor data availability...")
        eligible_users = []
        
        for user_id in users_with_emas:
            user_labs = lab_df[lab_df[cols['user_id']] == user_id].sort_values(cols['date'])
            last_n_samples = user_labs.tail(SAMPLES_PER_USER)
            
            # Check if user has sensor features for these dates
            has_data = True
            for _, sample in last_n_samples.iterrows():
                ema_date = sample[cols['date']]
                user_feat = feat_df[
                    (feat_df[cols['user_id']] == user_id) & 
                    (feat_df[cols['date']] < ema_date)
                ]
                if len(user_feat) == 0:
                    has_data = False
                    break
            
            if has_data:
                eligible_users.append(user_id)
        
        print(f"  Eligible users (with sensor data): {len(eligible_users)}")
        
        if len(eligible_users) < n_users_target:
            print(f"  [WARNING] Only {len(eligible_users)} eligible, requested {n_users_target}")
            n_users_target = len(eligible_users)
        
        # Randomly select users
        selected_users = rng.choice(eligible_users, size=n_users_target, replace=False)
        print(f"  Selected {len(selected_users)} users")
        
        # For each selected user, split into test (last N) and train (rest)
        inst_test_indices = []
        inst_train_indices = []
        
        for user_id in selected_users:
            user_labs = lab_df[lab_df[cols['user_id']] == user_id].sort_values(cols['date'])
            
            # Last N samples -> testset
            test_samples = user_labs.tail(SAMPLES_PER_USER)
            inst_test_indices.extend(test_samples.index.tolist())
            
            # Rest -> trainset (for ICL)
            train_samples = user_labs.iloc[:-SAMPLES_PER_USER] if len(user_labs) > SAMPLES_PER_USER else pd.DataFrame()
            if len(train_samples) > 0:
                inst_train_indices.extend(train_samples.index.tolist())
        
        # Create test and train DataFrames
        test_df_inst = lab_df.loc[inst_test_indices].copy()
        train_df_inst = lab_df.loc[inst_train_indices].copy() if inst_train_indices else pd.DataFrame()
        
        # Add institution column
        test_df_inst['institution'] = institution
        if len(train_df_inst) > 0:
            train_df_inst['institution'] = institution
        
        all_test_dfs.append(test_df_inst)
        if len(train_df_inst) > 0:
            all_train_dfs.append(train_df_inst)
        
        print(f"  Test: {len(test_df_inst)} samples from {len(selected_users)} users")
        print(f"  Train: {len(train_df_inst)} samples")
    
    # Combine all institutions
    combined_test_df = pd.concat(all_test_dfs, ignore_index=True)
    combined_train_df = pd.concat(all_train_dfs, ignore_index=True) if all_train_dfs else pd.DataFrame()
    
    # Save to CSV
    test_output_path = os.path.join(OUTPUT_DIR, 'globem_testset.csv')
    train_output_path = os.path.join(OUTPUT_DIR, 'globem_trainset.csv')
    
    combined_test_df.to_csv(test_output_path, index=False)
    print(f"\n✅ Testset saved: {test_output_path}")
    print(f"   {len(combined_test_df)} samples from {combined_test_df[cols['user_id']].nunique()} users")
    
    if len(combined_train_df) > 0:
        combined_train_df.to_csv(train_output_path, index=False)
        print(f"✅ Trainset saved: {train_output_path}")
        print(f"   {len(combined_train_df)} samples")
    
    # Print distribution by institution
    print(f"\n📊 Distribution by institution:")
    for inst in MULTI_INSTITUTION_CONFIG.keys():
        inst_count = len(combined_test_df[combined_test_df['institution'] == inst])
        inst_users = combined_test_df[combined_test_df['institution'] == inst][cols['user_id']].nunique()
        print(f"  {inst}: {inst_count} samples from {inst_users} users")
    
    # Print label distribution
    print(f"\n📊 Label distribution in testset:")
    for label in cols['labels']:
        counts = combined_test_df[label].value_counts().sort_index()
        print(f"  {label}:")
        for val, count in counts.items():
            print(f"    {val}: {count} ({count/len(combined_test_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("GLOBEM TESTSET PREPARATION COMPLETE!")
    print("="*80 + "\n")
    
    return combined_test_df, combined_train_df


def aggregate_globem_features(test_df, train_df, seed=42):
    """
    Pre-aggregate features for all GLOBEM samples (testset + trainset).
    
    This eliminates the need for on-the-fly aggregation during evaluation,
    making GLOBEM consistent with CES and MentalIoT.
    
    Args:
        test_df: Testset dataframe (with labels)
        train_df: Trainset dataframe (with labels)
        seed: Random seed
    
    Returns:
        Aggregated dataframe combining test and train samples
    """
    print("\n" + "="*80)
    print("GLOBEM FEATURE AGGREGATION")
    print("="*80)
    print(f"Aggregating features for {len(test_df)} test + {len(train_df)} train samples")
    print("="*80 + "\n")
    
    # Import sensor transformation utilities
    sys.path.insert(0, './src')
    from sensor_transformation import aggregate_window_features
    from src import config
    
    # Load use_cols configuration
    with open(USE_COLS_PATH, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols['compass']
    
    # Load raw feature data for all institutions
    print("[1/4] Loading raw feature data from all institutions...")
    all_feat_dfs = []
    for institution in MULTI_INSTITUTION_CONFIG.keys():
        feat_path = f'../dataset/Globem/{institution}/FeatureData/rapids.csv'
        print(f"  Loading {institution}: {feat_path}")
        feat_df = pd.read_csv(feat_path, low_memory=False)
        feat_df['institution'] = institution
        all_feat_dfs.append(feat_df)
    
    combined_feat_df = pd.concat(all_feat_dfs, ignore_index=True)
    combined_feat_df['date'] = pd.to_datetime(combined_feat_df['date'])
    print(f"  ✓ Loaded {len(combined_feat_df)} feature rows from {len(MULTI_INSTITUTION_CONFIG)} institutions\n")
    
    # Combine test and train for aggregation
    combined_df = pd.concat([test_df, train_df], ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['is_testset'] = combined_df.index < len(test_df)
    
    print(f"[2/4] Total samples to aggregate: {len(combined_df)}")
    print(f"  Test: {len(test_df)}, Train: {len(train_df)}\n")
    
    # Aggregate features for each sample
    print("[3/4] Aggregating features (this may take a while)...")
    aggregated_rows = []
    failed_count = 0
    
    for idx, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Aggregating"):
        user_id = row['pid']
        ema_date = pd.to_datetime(row['date'])
        
        # Get aggregated features
        agg_feats = aggregate_window_features(
            combined_feat_df, user_id, ema_date, cols,
            window_days=config.AGGREGATION_WINDOW_DAYS,
            mode=config.DEFAULT_AGGREGATION_MODE,
            use_immediate_window=config.USE_IMMEDIATE_WINDOW,
            immediate_window_days=config.IMMEDIATE_WINDOW_DAYS,
            adaptive_window=config.USE_ADAPTIVE_WINDOW
        )
        
        if agg_feats is None:
            failed_count += 1
            continue
        
        # Create aggregated row - start with metadata and labels
        agg_row = {
            'pid': user_id,
            'date': ema_date,
            'institution': row['institution'],
            'is_testset': row['is_testset'],
            # Labels
            'phq4_EMA': row['phq4_EMA'],
            'phq4_anxiety_EMA': row['phq4_anxiety_EMA'],
            'phq4_depression_EMA': row['phq4_depression_EMA'],
            'pss4_EMA': row['pss4_EMA'],
            'positive_affect_EMA': row['positive_affect_EMA'],
            'negative_affect_EMA': row['negative_affect_EMA'],
        }
        
        # Add aggregated features (GLOBEM COMPASS format has statistical_features, structural_features, etc.)
        for key, value in agg_feats.items():
            if key in ['user_id', 'ema_date', 'aggregation_mode', 'window_days']:
                # Skip metadata
                continue
            
            if isinstance(value, dict):
                # Flatten nested features (statistical_features, structural_features, semantic_features, etc.)
                for feat_name, feat_data in value.items():
                    if isinstance(feat_data, dict):
                        # Further nested (e.g., statistical features have mean/std/min/max)
                        for sub_key, sub_value in feat_data.items():
                            column_name = f"{feat_name}_{sub_key}" if sub_key not in ['mean', 'std', 'min', 'max'] else f"{feat_name}_{sub_key}"
                            agg_row[column_name] = sub_value
                    else:
                        # Direct value
                        agg_row[feat_name] = feat_data
            else:
                # Direct feature
                agg_row[key] = value
        
        aggregated_rows.append(agg_row)
    
    print(f"\n  ✓ Successfully aggregated {len(aggregated_rows)}/{len(combined_df)} samples")
    if failed_count > 0:
        print(f"  ⚠ Failed to aggregate {failed_count} samples (insufficient historical data)")
    
    # Create aggregated dataframe
    aggregated_df = pd.DataFrame(aggregated_rows)
    
    # Save to CSV
    print("\n[4/4] Saving aggregated features...")
    output_path = os.path.join(OUTPUT_DIR, 'aggregated_globem.csv')
    aggregated_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")
    print(f"    Rows: {len(aggregated_df)}")
    print(f"    Columns: {len(aggregated_df.columns)}")
    
    # Print summary
    test_agg = aggregated_df[aggregated_df['is_testset']]
    train_agg = aggregated_df[~aggregated_df['is_testset']]
    print(f"\n  Test samples: {len(test_agg)}")
    print(f"  Train samples: {len(train_agg)}")
    
    print("\n" + "="*80)
    print("GLOBEM FEATURE AGGREGATION COMPLETE!")
    print("="*80 + "\n")
    
    return aggregated_df


def save_daily_features(seed=42):
    """
    Combine and save daily features from all institutions for ML and DTW.
    
    This creates a single daily_features_globem.csv file that contains:
    - All raw daily sensor features from all institutions
    - Can be used for machine learning and time series extraction (DTW)
    
    Similar to how CES has both aggregated features (28-day stats) and 
    daily features for flexible use.
    
    Args:
        seed: Random seed (not used but kept for consistency)
    
    Returns:
        Combined daily features dataframe
    """
    print("\n" + "="*80)
    print("GLOBEM DAILY FEATURES COMBINATION")
    print("="*80)
    print("Combining daily features from all institutions for ML and DTW")
    print("="*80 + "\n")
    
    # Load use_cols configuration
    with open(USE_COLS_PATH, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols['compass']
    
    # Load raw feature data for all institutions
    print("[1/2] Loading daily feature data from all institutions...")
    all_feat_dfs = []
    
    for institution in MULTI_INSTITUTION_CONFIG.keys():
        feat_path = f'../dataset/Globem/{institution}/FeatureData/rapids.csv'
        print(f"  Loading {institution}: {feat_path}")
        
        if not os.path.exists(feat_path):
            print(f"    [WARNING] File not found, skipping")
            continue
        
        feat_df = pd.read_csv(feat_path, low_memory=False)
        feat_df['institution'] = institution
        all_feat_dfs.append(feat_df)
        print(f"    ✓ {len(feat_df)} rows")
    
    if not all_feat_dfs:
        print("[ERROR] No feature data found!")
        return None
    
    # Combine all institutions
    combined_feat_df = pd.concat(all_feat_dfs, ignore_index=True)
    combined_feat_df['date'] = pd.to_datetime(combined_feat_df['date'])
    
    print(f"\n  ✓ Combined: {len(combined_feat_df)} rows from {len(all_feat_dfs)} institutions")
    print(f"  Date range: {combined_feat_df['date'].min()} to {combined_feat_df['date'].max()}")
    print(f"  Users: {combined_feat_df['pid'].nunique()}")
    print(f"  Columns: {len(combined_feat_df.columns)}")
    
    # Save to CSV
    print("\n[2/2] Saving daily features...")
    output_path = os.path.join(OUTPUT_DIR, 'daily_features_globem.csv')
    combined_feat_df.to_csv(output_path, index=False)
    
    print(f"  ✓ Saved: {output_path}")
    print(f"    Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    # Print summary by institution
    print(f"\n📊 Distribution by institution:")
    for inst in MULTI_INSTITUTION_CONFIG.keys():
        inst_count = len(combined_feat_df[combined_feat_df['institution'] == inst])
        inst_users = combined_feat_df[combined_feat_df['institution'] == inst]['pid'].nunique()
        if inst_count > 0:
            print(f"  {inst}: {inst_count} rows, {inst_users} users")
    
    print("\n" + "="*80)
    print("GLOBEM DAILY FEATURES COMBINATION COMPLETE!")
    print("="*80 + "\n")
    
    return combined_feat_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare GLOBEM testset/trainset and aggregate features')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-aggregation', action='store_true',
                       help='Skip feature aggregation (only generate testset/trainset)')
    parser.add_argument('--only-daily-features', action='store_true',
                       help='Only generate daily features (skip testset/aggregation)')
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(USE_COLS_PATH):
        print(f"[ERROR] Config file not found: {USE_COLS_PATH}")
        print(f"Please make sure you're running from compass-sensor-llm-mh-prediction/ directory")
        exit(1)
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Option 1: Only generate daily features
    if args.only_daily_features:
        save_daily_features(seed=args.seed)
        exit(0)
    
    # Step 1: Generate testset/trainset
    test_df, train_df = prepare_globem_testset(seed=args.seed)
    
    # Step 2: Save daily features (for ML and DTW)
    if not args.skip_aggregation:
        save_daily_features(seed=args.seed)
    
    # Step 3: Aggregate features (28-day statistics)
    if not args.skip_aggregation:
        aggregate_globem_features(test_df, train_df, seed=args.seed)
    else:
        print("\n⚠ Skipping feature processing (--skip-aggregation flag)")
        print("   You'll need to run without this flag to generate:")
        print("   - daily_features_globem.csv (for ML and DTW)")
        print("   - aggregated_globem.csv (for prompts)")

