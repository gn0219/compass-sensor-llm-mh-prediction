"""
In-Context Learning Example Selection Module

Handles selection of ICL examples based on different strategies:
- Personalization: Use user's own history
- Generalization: Use other users' data
- Hybrid: Mix of both
"""

import pandas as pd
from typing import Optional, List, Dict
from .sensor_transformation import aggregate_7day_features, check_missing_ratio


def select_icl_examples(
    feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
    target_user_id: str, target_ema_date: pd.Timestamp,
    n_shot: int = 5, source: str = 'hybrid', selection: str = 'random',
    random_state: Optional[int] = None
) -> Optional[List[Dict]]:
    """
    Select in-context learning examples based on specified strategy.
    """
    if n_shot < 1:
        return []
    
    examples = []
    
    if source == 'personalization':
        # Use only user's own historical data (before target date)
        personal_lab = lab_df[
            (lab_df[cols['user_id']] == target_user_id) & 
            (lab_df[cols['date']] < target_ema_date)
        ]
        
        if len(personal_lab) < n_shot:
            print(f"Warning: Not enough historical data for user {target_user_id} before {target_ema_date}")
            return None
        
        examples = _sample_from_pool(
            feat_df, personal_lab, cols, n_shot, selection, random_state
        )
    
    elif source == 'generalization':
        # Use only other users' data
        general_lab = lab_df[lab_df[cols['user_id']] != target_user_id]
        
        if len(general_lab) < n_shot:
            print(f"Warning: Not enough data from other users")
            return None
        
        examples = _sample_from_pool(
            feat_df, general_lab, cols, n_shot, selection, random_state
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
        
        personal_examples = _sample_from_pool(
            feat_df, personal_lab, cols, n_personal, selection, random_state
        )
        general_examples = _sample_from_pool(
            feat_df, general_lab, cols, n_general, selection, 
            random_state + 1000 if random_state else None
        )
        
        examples = personal_examples + general_examples
    
    else:
        raise ValueError(f"Invalid source: {source}. Must be 'personalization', 'generalization', or 'hybrid'")
    
    return examples


def _sample_from_pool(
    feat_df: pd.DataFrame,
    lab_pool: pd.DataFrame,
    cols: Dict,
    n_samples: int,
    selection: str = 'random',
    random_state: Optional[int] = None
) -> List[Dict]:
    """
    Sample n_samples from the label pool and aggregate their features.
    
    Args:
        feat_df: Feature DataFrame
        lab_pool: Pool of label samples to choose from
        cols: Column configuration
        n_samples: Number of samples to select
        selection: Selection method ('random' or 'similarity')
        random_state: Random seed
    
    Returns:
        List of example dictionaries with aggregated features and labels
    """
    examples = []
    attempts = 0
    max_attempts = n_samples * 10  # Prevent infinite loop
    
    if selection == 'random':
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
            
            # Get 7-day aggregated features
            agg_feats = aggregate_7day_features(feat_df, user_id, ema_date, cols)
            
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
    
    elif selection == 'similarity':
        # TODO: Implement similarity-based selection (e.g., KNN)
        raise NotImplementedError("Similarity-based selection not yet implemented")
    
    else:
        raise ValueError(f"Invalid selection method: {selection}")
    
    return examples

