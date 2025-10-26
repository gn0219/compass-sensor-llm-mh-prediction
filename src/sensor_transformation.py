"""
Sensor Data Transformation Module

Handles loading, processing, and transforming sensor data for mental health prediction.
"""

import json
import warnings
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from datetime import timedelta
from scipy import stats as scipy_stats

try:
    from . import config
except ImportError:
    import config

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)


def binarize_labels(df: pd.DataFrame, labels: List[str], thresholds: Dict[str, int]) -> pd.DataFrame:
    """Binarize labels based on thresholds."""
    df = df.copy()
    for label in labels:
        if label in df.columns:
            df[label] = (df[label] > thresholds[label]).astype(int)
    return df


def load_globem_data(institution: str = 'INS-W_2', target: str = 'fctci',
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



def aggregate_window_features(feat_df: pd.DataFrame, user_id: str, ema_date: pd.Timestamp,
                             cols: Dict, window_days: int = None, mode: str = None,
                             use_immediate_window: bool = None, immediate_window_days: int = None,
                             adaptive_window: bool = None) -> Optional[Dict]:
    """
    Aggregate sensor features from window_days before EMA date.
    
    Args:
        feat_df: Feature dataframe
        user_id: User identifier
        ema_date: Target date for prediction
        cols: Column configuration
        window_days: Number of days for aggregation window (default: from config)
        mode: Aggregation mode ('array' or 'statistics') (default: from config)
        use_immediate_window: Whether to include immediate window statistics (default: from config)
        immediate_window_days: Number of days for immediate window (must be < window_days) (default: from config)
        adaptive_window: If True, use all available data when window_days exceeds available history (default: from config)
        
    Returns:
        Dictionary with aggregated features or None if insufficient data
    """
    # Get defaults from config if not specified
    if window_days is None:
        window_days = config.AGGREGATION_WINDOW_DAYS
    if mode is None:
        mode = config.DEFAULT_AGGREGATION_MODE
    if use_immediate_window is None:
        use_immediate_window = config.USE_IMMEDIATE_WINDOW
    if immediate_window_days is None:
        immediate_window_days = config.IMMEDIATE_WINDOW_DAYS
    if adaptive_window is None:
        adaptive_window = config.USE_ADAPTIVE_WINDOW
    
    start_date = ema_date - timedelta(days=window_days)
    
    user_feats = feat_df[
        (feat_df[cols['user_id']] == user_id) & 
        (feat_df[cols['date']] >= start_date) & 
        (feat_df[cols['date']] < ema_date)
    ].copy()
    
    if len(user_feats) == 0:
        return None
    
    # Sort by date for chronological ordering
    user_feats = user_feats.sort_values(cols['date'])
    
    
    # Adaptive window: adjust window_days based on actual available data
    actual_window_days = window_days
    if adaptive_window and len(user_feats) < window_days:
        # Calculate actual days covered
        min_date = user_feats[cols['date']].min()
        max_date = user_feats[cols['date']].max()
        actual_days = (max_date - min_date).days + 1
        
        # Use all available data (don't penalize early samples)
        actual_window_days = max(len(user_feats), actual_days)
        
        # Also adjust immediate_window if needed
        if use_immediate_window and immediate_window_days >= actual_window_days:
            # If immediate window is >= actual window, just use statistics mode without immediate
            use_immediate_window = False
    
    # Check if this is compass format (dict with statistical/semantic/etc)
    if isinstance(cols.get('feature_set'), dict) and 'statistical' in cols['feature_set']:
        # Compass format - use new aggregation
        return _aggregate_compass_features(
            feat_df, user_feats, user_id, ema_date, cols, actual_window_days
        )
    elif mode == 'array':
        return _aggregate_as_array(user_feats, user_id, ema_date, cols, actual_window_days)
    elif mode == 'statistics':
        return _aggregate_as_statistical_structural_semantical(
            user_feats, user_id, ema_date, cols, actual_window_days,
            use_immediate_window, immediate_window_days
        )
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def _aggregate_as_array(user_feats: pd.DataFrame, user_id: str, ema_date: pd.Timestamp,
                       cols: Dict, window_days: int) -> Dict:
    """Aggregate features as raw arrays (Option 1)."""
    result = {
        'user_id': user_id,
        'ema_date': ema_date,
        'aggregation_mode': 'array',
        'window_days': window_days,
        'features': {}
    }
    
    for feat in cols['feature_set']:
        if feat in user_feats.columns:
            # Get array of values, pad with None if needed
            values = user_feats[feat].tolist()
            
            # Pad with None if we don't have full window_days
            if len(values) < window_days:
                values = [None] * (window_days - len(values)) + values
            
            # Round non-null values
            values = [round(v, 2) if pd.notna(v) else None for v in values]
            
            result['features'][feat] = values
    
    return result



def _calculate_normalized_slope(values: np.ndarray) -> Tuple[float, str]:
    """
    Calculate normalized slope (slope / mean) to make it scale-independent.
    
    Returns:
        Tuple of (normalized_slope, direction) where direction is 'increasing', 'decreasing', or 'stable'
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



def _aggregate_compass_features(feat_df: pd.DataFrame, user_feats: pd.DataFrame, user_id: str, 
                               ema_date: pd.Timestamp, cols: Dict, window_days: int) -> Dict:
    """
    Aggregate features for compass format with statistical, structural, semantic, and temporal components.
    
    Args:
        feat_df: Full feature dataframe (needed for yesterday's data)
        user_feats: User's features within window
        user_id: User ID
        ema_date: EMA date
        cols: Column configuration with statistical, semantic, temporal_descriptor sections
        window_days: Aggregation window (default 28)
    """
    result = {
        'user_id': user_id,
        'ema_date': ema_date,
        'aggregation_mode': 'compass',
        'window_days': window_days,
        'statistical_features': {},
        'structural_features': {},
        'semantic_features': {},
        'temporal_descriptors': {}
    }
    
    feature_set = cols['feature_set']
    statistical_feats = feature_set.get('statistical', {})
    semantic_feats = feature_set.get('semantic', {})
    temporal_feats = feature_set.get('temporal_descriptor', {})
    
    # ========== 1. STATISTICAL & STRUCTURAL FEATURES ==========
    for feat_col, feat_name in statistical_feats.items():
        if feat_col not in user_feats.columns:
            continue
        
        values = user_feats[feat_col].values
        
        if len(values) < window_days * 0.5:  # Need at least 50% data
            continue
        
        # Statistical: mean, std, min, max over 28 days
        stats = {
            'mean': round(np.nanmean(values), 2),
            'std': round(np.nanstd(values), 2),
            'min': round(np.nanmin(values), 2),
            'max': round(np.nanmax(values), 2)
        }
        
        # Structural: slopes for past 2 weeks (28~15 days) and recent 2 weeks (14~1 days)
        # Past 2 weeks: days 28 to 15 (earlier period)
        if len(values) >= 28:
            past_2weeks = values[:14]  # First 14 days of 28-day window
            past_slope, past_dir = _calculate_normalized_slope(past_2weeks)
            
            # Recent 2 weeks: days 14 to 1
            recent_2weeks = values[-14:]  # Last 14 days
            recent_slope, recent_dir = _calculate_normalized_slope(recent_2weeks)
            
            structural = {
                'past_2weeks_slope': round(past_slope, 2) if not np.isnan(past_slope) else None,
                'past_2weeks_direction': past_dir,
                'recent_2weeks_slope': round(recent_slope, 2) if not np.isnan(recent_slope) else None,
                'recent_2weeks_direction': recent_dir
            }
        else:
            # If less than 28 days, just compute one slope
            slope, direction = _calculate_normalized_slope(values)
            structural = {
                'past_2weeks_slope': None,
                'past_2weeks_direction': 'stable',
                'recent_2weeks_slope': round(slope, 2) if not np.isnan(slope) else None,
                'recent_2weeks_direction': direction
            }
        
        result['statistical_features'][feat_name] = stats
        result['structural_features'][feat_name] = structural
    
    # ========== 2. SEMANTIC FEATURES ==========
    # Extract base feature names for semantic analysis
    semantic_groups = {}
    
    # Group semantic features by base name and type (weekday/weekend/morning/etc)
    for feat_col, feat_name in semantic_feats.items():
        if feat_col not in user_feats.columns:
            continue
        
        # Parse feature column: extract base and time period
        # e.g., "f_steps:fitbit_steps_intraday_rapids_sumsteps:weekday" -> base: steps, period: weekday
        parts = feat_col.split(':')
        if len(parts) >= 3:
            base = parts[1]  # e.g., "fitbit_steps_intraday_rapids_sumsteps"
            period = parts[2]  # e.g., "weekday", "morning"
            
            # Simplify base name
            if 'sleep' in base:
                base_name = 'Sleep duration'
            elif 'steps' in base or 'sumsteps' in base:
                base_name = 'Physical activity'
            elif 'locationentropy' in base:
                base_name = 'Location entropy'
            elif 'sumdurationunlock' in base:
                base_name = 'Phone usage'
            else:
                base_name = base
            
            if base_name not in semantic_groups:
                semantic_groups[base_name] = {}
            
            values = user_feats[feat_col].values
            mean_val = np.nanmean(values)
            
            semantic_groups[base_name][period] = round(mean_val, 2) if not np.isnan(mean_val) else None
    
    # Build semantic feature descriptions
    for base_name, periods in semantic_groups.items():
        semantic_info = {}
        
        # Pattern: weekday vs weekend
        if 'weekday' in periods and 'weekend' in periods:
            weekday = periods['weekday']
            weekend = periods['weekend']
            if weekday is not None and weekend is not None:
                diff = weekend - weekday
                semantic_info['pattern'] = {
                    'weekday': weekday,
                    'weekend': weekend,
                    'difference': round(diff, 2)
                }
        
        # Circadian: 28-day average for morning/afternoon/evening/night
        circadian_periods = ['morning', 'afternoon', 'evening', 'night']
        if all(p in periods for p in circadian_periods):
            circadian = {p: periods[p] for p in circadian_periods if periods[p] is not None}
            if circadian:
                semantic_info['circadian_28day'] = circadian
        
        # Yesterday transition: Get yesterday's data for morning/afternoon/evening/night
        yesterday = ema_date - timedelta(days=1)
        yesterday_data = feat_df[
            (feat_df[cols['user_id']] == user_id) &
            (feat_df[cols['date']] == yesterday)
        ]
        
        if len(yesterday_data) > 0:
            yesterday_values = {}
            for period in circadian_periods:
                # Find corresponding column in semantic_feats
                for feat_col in semantic_feats.keys():
                    if base_name.lower().replace(' ', '') in feat_col.lower() and f':{period}' in feat_col:
                        if feat_col in yesterday_data.columns:
                            val = yesterday_data[feat_col].iloc[0]
                            if pd.notna(val):
                                yesterday_values[period] = round(val, 2)
                        break
            
            if yesterday_values:
                semantic_info['yesterday_transition'] = yesterday_values
        
        if semantic_info:
            result['semantic_features'][base_name] = semantic_info
    
    # ========== 3. TEMPORAL DESCRIPTORS ==========
    # Get last 7 days as daily arrays
    last_7_days = user_feats.tail(7)
    
    for feat_col, feat_name in temporal_feats.items():
        if feat_col in last_7_days.columns:
            values = last_7_days[feat_col].tolist()
            # Round and handle None
            values = [round(v, 2) if pd.notna(v) else None for v in values]
            result['temporal_descriptors'][feat_name] = values
    
    return result


def check_missing_ratio(data, threshold: float = 0.7) -> bool:
    """
    Check if sample has acceptable missing data ratio.
    
    Args:
        data: Either pd.DataFrame (legacy) or Dict (new format)
        threshold: Maximum acceptable missing ratio
    
    Returns:
        True if missing ratio is acceptable
    """
    if isinstance(data, pd.DataFrame):
        # Legacy format
        missing_ratio = data.isna().sum().sum() / (data.shape[0] * data.shape[1])
    elif isinstance(data, dict):
        mode = data.get('aggregation_mode', 'unknown')
        
        if mode == 'compass':
            # Compass format: check statistical features
            stat_feats = data.get('statistical_features', {})
            if not stat_feats:
                return False
            
            total_values = 0
            missing_values = 0
            
            for feat_name, stats in stat_feats.items():
                for stat_name, stat_value in stats.items():
                    total_values += 1
                    if stat_value is None or (isinstance(stat_value, float) and np.isnan(stat_value)):
                        missing_values += 1
            
            if total_values == 0:
                return False
            missing_ratio = missing_values / total_values
            
        elif 'features' in data:
            # Array or statistics format
            total_values = 0
            missing_values = 0
            
            for feat_name, feat_data in data['features'].items():
                if isinstance(feat_data, list):
                    # Array mode
                    total_values += len(feat_data)
                    missing_values += sum(1 for v in feat_data if v is None)
                elif isinstance(feat_data, dict):
                    # Statistics mode
                    for stat_name, stat_value in feat_data.items():
                        if not stat_name.startswith('last_'):  # Skip raw arrays
                            total_values += 1
                            if stat_value is None:
                                missing_values += 1
            
            if total_values == 0:
                return False
            missing_ratio = missing_values / total_values
        else:
            raise ValueError(f"Unsupported dict format: {data.keys()}")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
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


def features_to_text(agg_feats, cols: Dict, include_stats: bool = True) -> str:
    """
    Convert aggregated sensor features to natural language text.
    
    Args:
        agg_feats: Either pd.DataFrame (legacy) or Dict (new format)
        cols: Column configuration
        include_stats: Whether to include statistics (legacy parameter)
    
    Returns:
        Formatted text representation of features
    """
    text = ""
    
    if not isinstance(agg_feats, dict):
        return text
    
    mode = agg_feats.get('aggregation_mode', 'unknown')
    
    # ========== COMPASS FORMAT ==========
    if mode == 'compass':
        window_days = agg_feats.get('window_days', 28)
        
        # 1. Statistical & Structural features
        statistical = agg_feats.get('statistical_features', {})
        structural = agg_feats.get('structural_features', {})
        
        for feat_name in statistical.keys():
            stats = statistical.get(feat_name, {})
            struct = structural.get(feat_name, {})
            
            text += f"{feat_name}\n"
            text += f"- {window_days} day mean: {stats.get('mean', 'N/A')}, "
            text += f"std: {stats.get('std', 'N/A')}, "
            text += f"min: {stats.get('min', 'N/A')}, "
            text += f"max: {stats.get('max', 'N/A')}\n"
            
            # Slopes
            past_dir = struct.get('past_2weeks_direction', 'stable')
            past_slope = struct.get('past_2weeks_slope', 'N/A')
            recent_dir = struct.get('recent_2weeks_direction', 'stable')
            recent_slope = struct.get('recent_2weeks_slope', 'N/A')
            text += f"- Past 2 weeks slope: ({past_dir}, {past_slope}), Recent 2 weeks slope: ({recent_dir}, {recent_slope})\n\n"
        
        # 2. Semantic features
        semantic = agg_feats.get('semantic_features', {})
        if semantic:
            text += "The following shows weekday/weekend patterns, 28-day circadian rhythms, "
            text += "and yesterday's transitions (morning/afternoon/evening/night) for some features.\n\n"
            
            for feat_name, sem_info in semantic.items():
                text += f"- {feat_name}\n"
                
                # Pattern (weekday vs weekend)
                if 'pattern' in sem_info:
                    pat = sem_info['pattern']
                    weekday = pat.get('weekday', 'N/A')
                    weekend = pat.get('weekend', 'N/A')
                    diff = pat.get('difference', 'N/A')
                    text += f"  - Weekday: {weekday}, Weekend: {weekend} (diff={diff})\n"
                
                # Circadian (28-day rhythms)
                if 'circadian_28day' in sem_info:
                    circ = sem_info['circadian_28day']
                    text += f"  - 28 day rhythms (mean):\n"
                    for period in ['morning', 'afternoon', 'evening', 'night']:
                        if period in circ:
                            text += f"    - {period}: {circ[period]}\n"
                
                # Yesterday transition
                if 'yesterday_transition' in sem_info:
                    trans = sem_info['yesterday_transition']
                    text += f"  - Yesterday transition:\n"
                    for period in ['morning', 'afternoon', 'evening', 'night']:
                        if period in trans:
                            text += f"    - {period}: {trans[period]}\n"
                
                text += "\n"
        
        # 3. Temporal descriptors
        temporal = agg_feats.get('temporal_descriptors', {})
        if temporal:
            text += "Here are some recent variables (last 7 days):\n"
            for feat_name, values in temporal.items():
                value_str = ', '.join([str(v) if v is not None else 'N/A' for v in values])
                text += f"- {feat_name}: [{value_str}]\n"
    
    # ========== ARRAY FORMAT ==========
    elif mode == 'array':
        window_days = agg_feats.get('window_days', 'N')
        for feat_name, values in agg_feats.get('features', {}).items():
            simple_name = _simplify_feature_name(feat_name)
            text += f"  - {simple_name}:\n"
            
            if values:
                value_str = ', '.join([str(v) if v is not None else 'missing' for v in values])
                text += f"    [{value_str}]\n"
            else:
                text += f"    [all missing]\n"
    
    # ========== STATISTICS FORMAT ==========
    elif mode == 'statistics':
        window_days = agg_feats.get('window_days', 'N')
        use_immediate = agg_feats.get('use_immediate_window', False)
        immediate_window_days = agg_feats.get('immediate_window_days', 7)
        
        if use_immediate:
            text += f"(Statistics over {window_days} days, with recent {immediate_window_days}-day window)\n\n"
        else:
            text += f"(Statistics over {window_days} days)\n\n"
        
        for feat_name, feat_stats in agg_feats.get('features', {}).items():
            simple_name = _simplify_feature_name(feat_name)
            text += f"  - {simple_name}:\n"
            
            for stat_name, stat_value in feat_stats.items():
                if stat_name.startswith('last_'):
                    if isinstance(stat_value, list):
                        value_str = ', '.join([str(v) if v is not None else 'missing' for v in stat_value])
                        text += f"    * {stat_name}: [{value_str}]\n"
                else:
                    if stat_value is not None and pd.notna(stat_value):
                        text += f"    * {stat_name}: {stat_value}\n"
                    else:
                        text += f"    * {stat_name}: missing\n"
    
    return text


def sample_to_prompt(sample: Dict, cols: Dict, format_type: str = 'structured',
                    include_labels: bool = False) -> str:
    """Convert a sample to prompt text."""
    prompt = f"User ID: {sample['user_id']}\n"
    prompt += f"Date: {sample['ema_date'].strftime('%Y-%m-%d')}\n"
    
    # Get window information from aggregated features
    agg_feats = sample['aggregated_features']
    if isinstance(agg_feats, dict):
        window_days = agg_feats.get('window_days', 7)
        mode = agg_feats.get('aggregation_mode', 'statistics')
    else:
        window_days = 7
        mode = 'legacy'
    
    prompt += "\nSensor Features:\n"
    
    prompt += features_to_text(agg_feats, cols)
    
    if include_labels:
        prompt += "\nLabels:\n"
        for label_name, label_value in sample['labels'].items():
            label_simple = label_name.replace('_EMA', '').replace('phq4_', '').title()
            status = "High Risk" if label_value == 1 else "Low Risk"
            prompt += f"  - {label_simple}: {status}\n"
    
    return prompt


def sample_input_data(feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict,
                     random_state: Optional[int] = None) -> Optional[Dict]:
    """
    Randomly sample an input data point for prediction.
    Uses aggregation settings from config.py.
    
    Args:
        feat_df: Feature dataframe
        lab_df: Label dataframe
        cols: Column configuration
        random_state: Random seed for reproducibility
    
    Returns:
        Sample dictionary or None if insufficient data
    """
    sample_lab = lab_df.sample(1, random_state=random_state) if random_state is not None else lab_df.sample(1)
    
    user_id = sample_lab.iloc[0][cols['user_id']]
    ema_date = sample_lab.iloc[0][cols['date']]
    
    agg_feats = aggregate_window_features(feat_df, user_id, ema_date, cols)
    
    if agg_feats is None or not check_missing_ratio(agg_feats):
        return None
    
    labels = sample_lab[cols['labels']].iloc[0].to_dict()
    
    return {
        'aggregated_features': agg_feats, 'labels': labels,
        'user_id': user_id, 'ema_date': ema_date
    }


def sample_batch_stratified(feat_df: pd.DataFrame, lab_df: pd.DataFrame, cols: Dict, n_samples: int,
                            random_state: Optional[int] = None, max_attempts: int = 5000) -> List[Dict]:
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
        print(f"⚠️  Warning: n_samples ({n_samples}) < number of groups ({len(classes)})")
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
    
    print(f"✅ Successfully collected {len(collected_samples)} stratified samples")
    return collected_samples


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
        # user_group already has original indices from lab_df_sorted
        user_dates = user_group[date_col].values
        user_indices = user_group.index.tolist()
        
        # For each sample, count how many prior labels exist
        for i, (idx, date) in enumerate(zip(user_indices, user_dates)):
            # Count samples before current date (i is the position in sorted user_group)
            n_historical = i  # i = 0 means 0 prior samples, i = 1 means 1 prior sample, etc.
            
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


# def get_data_statistics(lab_df: pd.DataFrame, cols: Dict) -> Dict:
#     """Get statistics about the dataset."""
#     stats = {
#         'total_samples': len(lab_df),
#         'unique_users': lab_df[cols['user_id']].nunique(),
#         'label_distributions': {}
#     }
    
#     for label in cols['labels']:
#         if label in lab_df.columns:
#             dist = lab_df[label].value_counts().to_dict()
#             stats['label_distributions'][label] = {
#                 'counts': dist,
#                 'proportions': {k: v/len(lab_df) for k, v in dist.items()}
#             }
    
#     return stats


# def print_data_statistics(stats: Dict):
#     """Pretty print dataset statistics."""
#     print("\n" + "="*80)
#     print("DATASET STATISTICS")
#     print("="*80)
#     print(f"\nTotal Samples: {stats['total_samples']}")
#     print(f"Unique Users: {stats['unique_users']}")
    
#     print("\nLabel Distributions:")
#     for label, dist_info in stats['label_distributions'].items():
#         print(f"\n  {label}:")
#         for cls, count in dist_info['counts'].items():
#             proportion = dist_info['proportions'][cls]
#             print(f"    Class {cls}: {count} ({proportion*100:.1f}%)")
#     print("="*80 + "\n")