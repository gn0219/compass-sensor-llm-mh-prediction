#!/usr/bin/env python3
"""
Prepare CES Testset

Creates pre-sampled testset and trainset for CES dataset.
Similar to prepare_globem_data.py and prepare_mentaliot_data.py.

Usage:
    python prepare_ces_data.py [--seed SEED]
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
N_USERS = 60  # Number of users to sample
MIN_EMA_PER_USER = 9  # Minimum EMAs required (to allow 4 ICL + 5 test)
SAMPLES_PER_USER = 5  # Number of test samples per user
USE_COLS_PATH = './config/ces_use_cols.json'
AGGREGATED_FILE = '../dataset/CES/aggregated_ces.csv'
OUTPUT_DIR = '../dataset/CES'


def prepare_ces_testset(seed=42):
    """
    Prepare CES testset with gender-balanced user selection.
    
    Strategy:
    - Select 60 users with >= 9 EMAs (gender-balanced)
    - For each user, sample 5 EMAs from 5th EMA onwards (stratified)
    - Total: 60 users × 5 samples = 300 samples
    - Remaining EMAs become trainset (for ICL)
    """
    n_users = N_USERS
    min_ema = MIN_EMA_PER_USER
    samples_per_user = SAMPLES_PER_USER
    
    print("\n" + "="*80)
    print("CES TESTSET PREPARATION (GENDER-BALANCED)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Target users: {n_users}")
    print(f"  Min EMA per user: {min_ema}")
    print(f"  Samples per user: {samples_per_user}")
    print(f"  Random seed: {seed}")
    print("="*80 + "\n")
    
    rng = np.random.RandomState(seed)
    
    # Load aggregated CES data
    print(f"Loading aggregated CES data from: {AGGREGATED_FILE}")
    if not os.path.exists(AGGREGATED_FILE):
        print(f"[ERROR] Aggregated file not found: {AGGREGATED_FILE}")
        print(f"Please ensure CES data has been aggregated first.")
        return None, None
    
    agg_df = pd.read_csv(AGGREGATED_FILE, low_memory=False)
    agg_df['date'] = pd.to_datetime(agg_df['date'])
    
    # Load column configuration
    with open(USE_COLS_PATH, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols['compass']
    
    print(f"  Loaded {len(agg_df)} samples from {agg_df['uid'].nunique()} users")
    
    # Count EMAs per user
    user_ema_counts = agg_df.groupby('uid').size()
    users_with_emas = user_ema_counts[user_ema_counts >= min_ema].index.tolist()
    
    print(f"\nTotal users in dataset: {agg_df['uid'].nunique()}")
    print(f"Users with >= {min_ema} EMAs: {len(users_with_emas)}")
    
    if len(users_with_emas) < n_users:
        print(f"[WARNING] Only {len(users_with_emas)} eligible users, requested {n_users}")
        n_users = len(users_with_emas)
    
    # Get gender information
    user_genders = agg_df.groupby('uid')['gender'].first()
    
    # Count gender distribution
    gender_counts = user_genders[user_genders.index.isin(users_with_emas)].value_counts()
    print(f"\nGender distribution in eligible users:")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count} ({count/len(users_with_emas)*100:.1f}%)")
    
    # Sample users proportionally by gender
    selected_users = []
    for gender, count in gender_counts.items():
        gender_users = user_genders[
            (user_genders == gender) & (user_genders.index.isin(users_with_emas))
        ].index.tolist()
        
        # Proportional allocation
        n_from_gender = int(n_users * (count / len(users_with_emas)))
        n_from_gender = min(n_from_gender, len(gender_users))
        
        sampled = rng.choice(gender_users, size=n_from_gender, replace=False)
        selected_users.extend(sampled.tolist())
    
    # Fill up to n_users if needed
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
    
    # For each user, sample test samples (from 5th EMA onwards, stratified)
    print(f"\nSampling {samples_per_user} test samples per user (from 5th EMA onwards)...")
    
    test_indices = []
    
    for user_id in selected_users:
        user_df = agg_df[agg_df['uid'] == user_id].sort_values('date')
        
        # Only consider samples from 5th EMA onwards
        if len(user_df) < 5:
            print(f"  [Warning] User {user_id} has < 5 EMAs, skipping")
            continue
        
        eligible_samples = user_df.iloc[4:]  # Skip first 4
        
        if len(eligible_samples) < samples_per_user:
            # Take all available
            sampled_indices = eligible_samples.index.tolist()
        else:
            # Stratified sampling based on (anxiety, depression, stress)
            label_cols = cols['labels']
            strat_key = eligible_samples.apply(
                lambda row: f"{int(row[label_cols[0]])}_{int(row[label_cols[1]])}_{int(row[label_cols[2]])}",
                axis=1
            )
            
            # Group by stratification
            sampled_indices = []
            for key, group in eligible_samples.groupby(strat_key):
                n_from_group = max(1, int(samples_per_user * len(group) / len(eligible_samples)))
                n_from_group = min(n_from_group, len(group))
                
                sample = rng.choice(group.index, size=n_from_group, replace=False)
                sampled_indices.extend(sample.tolist())
            
            # Adjust to exact count
            if len(sampled_indices) > samples_per_user:
                sampled_indices = rng.choice(sampled_indices, size=samples_per_user, replace=False).tolist()
            elif len(sampled_indices) < samples_per_user:
                remaining = samples_per_user - len(sampled_indices)
                available = [idx for idx in eligible_samples.index if idx not in sampled_indices]
                if available:
                    additional = rng.choice(available, size=min(remaining, len(available)), replace=False)
                    sampled_indices.extend(additional.tolist())
        
        test_indices.extend(sampled_indices)
    
    # Split into test and train
    test_df = agg_df.loc[test_indices].copy()
    train_df = agg_df[~agg_df.index.isin(test_indices)].copy()
    
    print(f"\nTest set: {len(test_df)} samples from {test_df['uid'].nunique()} users")
    print(f"Train set: {len(train_df)} samples")
    
    # Print stratification distribution
    if len(test_df) > 0:
        label_cols = cols['labels']
        print(f"\nTest set stratification (Anxiety_Depression_Stress):")
        test_strat = test_df.apply(
            lambda row: f"{int(row[label_cols[0]])}_{int(row[label_cols[1]])}_{int(row[label_cols[2]])}",
            axis=1
        )
        strat_counts = test_strat.value_counts().sort_index()
        for strat_key, count in strat_counts.items():
            print(f"  {strat_key}: {count} ({count/len(test_df)*100:.1f}%)")
    
    # Save to CSV
    test_output_path = os.path.join(OUTPUT_DIR, 'ces_testset.csv')
    train_output_path = os.path.join(OUTPUT_DIR, 'ces_trainset.csv')
    
    test_df.to_csv(test_output_path, index=False)
    print(f"\n✅ Testset saved: {test_output_path}")
    print(f"   {len(test_df)} samples from {test_df['uid'].nunique()} users")
    
    train_df.to_csv(train_output_path, index=False)
    print(f"✅ Trainset saved: {train_output_path}")
    print(f"   {len(train_df)} samples")
    
    # Print label distribution
    print(f"\n📊 Label distribution in testset:")
    for label in cols['labels']:
        counts = test_df[label].value_counts().sort_index()
        print(f"  {label}:")
        for val, count in counts.items():
            print(f"    {val}: {count} ({count/len(test_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("CES TESTSET PREPARATION COMPLETE!")
    print("="*80 + "\n")
    
    return test_df, train_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare CES testset/trainset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(USE_COLS_PATH):
        print(f"[ERROR] Config file not found: {USE_COLS_PATH}")
        print(f"Please make sure you're running from compass-sensor-llm-mh-prediction/ directory")
        exit(1)
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    prepare_ces_testset(seed=args.seed)

