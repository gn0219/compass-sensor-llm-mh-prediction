"""
Sensor Data Transformation Module

Handles loading, processing, and transforming sensor data for mental health prediction.
"""

import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from datetime import timedelta


def binarize_labels(df: pd.DataFrame, labels: List[str], thresholds: Dict[str, int]) -> pd.DataFrame:
    """Binarize labels based on thresholds."""
    df = df.copy()
    for label in labels:
        if label in df.columns:
            df[label] = (df[label] >= thresholds[label]).astype(int)
    return df


def load_globem_data(institution: str = 'INS-W_2', target: str = 'fctci',
                    use_cols_path: str = './config/use_cols.json') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load GLOBEM dataset with feature and label data."""
    feat_path = f'/home/iclab/compass/dataset/Globem/{institution}/FeatureData/rapids.csv'
    lab_path = f'/home/iclab/compass/dataset/Globem/{institution}/SurveyData/ema.csv'
    
    feat_df = pd.read_csv(feat_path, low_memory=False)
    lab_df = pd.read_csv(lab_path)
    
    with open(use_cols_path, 'r') as f:
        use_cols = json.load(f)
    cols = use_cols['globem'][target]
    
    # Select columns
    feat_cols = [cols['user_id'], cols['date']] + cols['feature_set']
    lab_cols = [cols['user_id'], cols['date']] + cols['labels']
    
    feat_df = feat_df[feat_cols].copy()
    lab_df = lab_df[lab_cols].copy()
    
    # Convert dates
    feat_df[cols['date']] = pd.to_datetime(feat_df[cols['date']])
    lab_df[cols['date']] = pd.to_datetime(lab_df[cols['date']])
    
    # Binarize labels
    lab_df = binarize_labels(lab_df, cols['labels'], cols['threshold'])
    
    return feat_df, lab_df, cols


def aggregate_7day_features(feat_df: pd.DataFrame, user_id: str, ema_date: pd.Timestamp,
                            cols: Dict, days: int = 7) -> Optional[pd.DataFrame]:
    """Aggregate sensor features from N days before EMA date."""
    start_date = ema_date - timedelta(days=days)
    
    user_feats = feat_df[
        (feat_df[cols['user_id']] == user_id) & 
        (feat_df[cols['date']] >= start_date) & 
        (feat_df[cols['date']] < ema_date)
    ].copy()
    
    if len(user_feats) == 0:
        return None
    
    aggregated = {cols['user_id']: user_id, cols['date']: ema_date}
    
    for feat in cols['feature_set']:
        if feat in user_feats.columns:
            for stat, func in [('mean', 'mean'), ('std', 'std'), ('min', 'min'), ('max', 'max')]:
                val = getattr(user_feats[feat], func)()
                aggregated[f"{feat}_{stat}"] = round(val, 2) if pd.notna(val) else None
    
    return pd.DataFrame([aggregated])


def check_missing_ratio(df: pd.DataFrame, threshold: float = 0.7) -> bool:
    """Check if sample has acceptable missing data ratio."""
    missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
    return missing_ratio < threshold


def _simplify_feature_name(feat_name: str) -> str:
    """Simplify feature names for readability."""
    replacements = {
        'f_loc:': 'Location - ', 'f_screen:': 'Screen - ', 'f_call:': 'Call - ',
        'f_blue:': 'Bluetooth - ', 'f_steps:': 'Activity - ', 'f_slp:': 'Sleep - ',
        'phone_locations_doryab_': '', 'phone_screen_rapids_': '', 'phone_calls_rapids_': '',
        'phone_bluetooth_doryab_': '', 'fitbit_steps_intraday_rapids_': '',
        'fitbit_sleep_intraday_rapids_': '', ':allday': '', '_': ' '
    }
    for old, new in replacements.items():
        feat_name = feat_name.replace(old, new)
    return feat_name.title()


def features_to_text(agg_feats: pd.DataFrame, cols: Dict, include_stats: bool = True) -> str:
    """Convert aggregated sensor features to natural language text."""
    text = ""
    
    if include_stats:
        feature_groups = {}
        for col in agg_feats.columns:
            if col not in [cols['user_id'], cols['date']]:
                for feat in cols['feature_set']:
                    if col.startswith(feat):
                        if feat not in feature_groups:
                            feature_groups[feat] = {}
                        stat_type = col.replace(feat + '_', '')
                        feature_groups[feat][stat_type] = agg_feats[col].iloc[0]
                        break
        
        for feat_name, stats in feature_groups.items():
            simple_name = _simplify_feature_name(feat_name)
            text += f"  - {simple_name}:\n"
            for stat, value in stats.items():
                if value is not None and pd.notna(value):
                    text += f"    * {stat}: {value:.2f}\n"
                else:
                    text += f"    * {stat}: missing\n"
    
    return text


def sample_to_prompt(sample: Dict, cols: Dict, format_type: str = 'structured',
                    include_labels: bool = False) -> str:
    """Convert a sample to prompt text."""
    prompt = f"User ID: {sample['user_id']}\n"
    prompt += f"Date: {sample['ema_date'].strftime('%Y-%m-%d')}\n"
    prompt += "Sensor Features (7-day aggregated statistics):\n"
    prompt += features_to_text(sample['aggregated_features'], cols)
    
    if include_labels:
        prompt += "\nLabels:\n"
        for label_name, label_value in sample['labels'].items():
            label_simple = label_name.replace('_EMA', '').replace('phq4_', '').title()
            status = "High Risk" if label_value == 1 else "Low Risk"
            prompt += f"  - {label_simple}: {status}\n"
    
    return prompt


def sample_input_data(feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
                     random_state: Optional[int] = None) -> Optional[Dict]:
    """Randomly sample an input data point for prediction."""
    sample_lab = lab_df.sample(1, random_state=random_state) if random_state is not None else lab_df.sample(1)
    
    user_id = sample_lab.iloc[0][cols['user_id']]
    ema_date = sample_lab.iloc[0][cols['date']]
    
    agg_feats = aggregate_7day_features(feat_df, user_id, ema_date, cols)
    
    if agg_feats is None or not check_missing_ratio(agg_feats):
        return None
    
    labels = sample_lab[cols['labels']].iloc[0].to_dict()
    
    return {
        'aggregated_features': agg_feats, 'labels': labels,
        'user_id': user_id, 'ema_date': ema_date
    }


def sample_batch_stratified(feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict, n_samples: int,
                            stratify_by: str = 'phq4_anxiety_EMA', random_state: Optional[int] = None,
                            max_attempts: int = 5000) -> List[Dict]:
    """Sample a batch of data points with stratified sampling."""
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()
    
    if stratify_by not in cols['labels']:
        raise ValueError(f"Stratification column '{stratify_by}' not found in labels")
    
    # Calculate samples per class
    class_counts = lab_df[stratify_by].value_counts()
    classes = class_counts.index.tolist()
    samples_per_class = {}
    
    for cls in classes:
        proportion = class_counts[cls] / len(lab_df)
        samples_per_class[cls] = max(1, int(n_samples * proportion))
    
    # Adjust to match n_samples
    total = sum(samples_per_class.values())
    if total != n_samples:
        largest_class = class_counts.idxmax()
        samples_per_class[largest_class] += (n_samples - total)
    
    print(f"Stratified sampling by '{stratify_by}':")
    for cls, count in samples_per_class.items():
        print(f"  Class {cls}: {count} samples ({count/n_samples*100:.1f}%)")
    
    # Sample from each class
    collected_samples, attempted_indices = [], set()
    
    for cls, n_class_samples in samples_per_class.items():
        cls_indices = lab_df[lab_df[stratify_by] == cls].index.tolist()
        
        if len(cls_indices) < n_class_samples:
            print(f"⚠️  Warning: Only {len(cls_indices)} samples for class {cls}, requested {n_class_samples}")
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
            
            agg_feats = aggregate_7day_features(feat_df, user_id, ema_date, cols)
            
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
    
    print(f"✅ Successfully collected {len(collected_samples)} stratified samples")
    return collected_samples


def get_data_statistics(lab_df: pd.DataFrame, cols: Dict) -> Dict:
    """Get statistics about the dataset."""
    stats = {
        'total_samples': len(lab_df),
        'unique_users': lab_df[cols['user_id']].nunique(),
        'label_distributions': {}
    }
    
    for label in cols['labels']:
        if label in lab_df.columns:
            dist = lab_df[label].value_counts().to_dict()
            stats['label_distributions'][label] = {
                'counts': dist,
                'proportions': {k: v/len(lab_df) for k, v in dist.items()}
            }
    
    return stats


def print_data_statistics(stats: Dict):
    """Pretty print dataset statistics."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"\nTotal Samples: {stats['total_samples']}")
    print(f"Unique Users: {stats['unique_users']}")
    
    print("\nLabel Distributions:")
    for label, dist_info in stats['label_distributions'].items():
        print(f"\n  {label}:")
        for cls, count in dist_info['counts'].items():
            proportion = dist_info['proportions'][cls]
            print(f"    Class {cls}: {count} ({proportion*100:.1f}%)")
    print("="*80 + "\n")