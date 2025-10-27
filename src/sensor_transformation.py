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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)


def binarize_labels(df: pd.DataFrame, labels: List[str], thresholds: Dict[str, int]) -> pd.DataFrame:
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
        # For fctci and health-llm, return a minimal structure
        # The actual data will be extracted directly from feat_df in features_to_text_*
        return {
            'user_id': user_id,
            'ema_date': ema_date,
            'aggregation_mode': 'raw',  # Special mode for fctci/health-llm
            'window_days': actual_window_days,
            'features': {}  # Empty - will use feat_df directly
        }
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
    
    # Handle 'None' string from JSON config
    if temporal_feats == 'None':
        temporal_feats = None
    
    # ========== 1. STATISTICAL & STRUCTURAL FEATURES ==========
    for feat_col, feat_name in statistical_feats.items():
        if feat_col not in user_feats.columns:
            continue
        
        values = user_feats[feat_col].values
        
        if len(values) < window_days * 0.5:  # Need at least 50% data
            continue
        
        # Statistical: mean, std, min, max over 28 days
        stats = {
            'mean': round(np.nanmean(values), 1),
            'std': round(np.nanstd(values), 1),
            'min': round(np.nanmin(values), 1),
            'max': round(np.nanmax(values), 1)
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
            
            semantic_groups[base_name][period] = round(mean_val, 1) if not np.isnan(mean_val) else None
    
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
                # Match by checking if the feature is related to the base name
                for feat_col in semantic_feats.keys():
                    # Match based on keywords in base_name
                    match = False
                    if 'Physical activity' in base_name and 'steps' in feat_col.lower():
                        match = True
                    elif 'Sleep' in base_name and 'sleep' in feat_col.lower():
                        match = True
                    elif 'Location entropy' in base_name and 'locationentropy' in feat_col.lower():
                        match = True
                    elif 'Phone usage' in base_name and 'unlock' in feat_col.lower():
                        match = True
                    
                    if match and f':{period}' in feat_col:
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
    # Only process if temporal_feats is not None and not 'None' string
    if temporal_feats and temporal_feats != 'None':
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
        
        if mode == 'raw':
            # For fctci/health-llm: skip missing ratio check
            # Data will be extracted directly from feat_df
            return True
        
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


def features_to_text_fctci(feat_df: pd.DataFrame, user_id: str, ema_date: pd.Timestamp, 
                           cols: Dict, window_days: int = 28) -> str:
    """
    Convert sensor features to FCTCI markdown table format.
    
    Args:
        feat_df: Feature dataframe
        user_id: User ID
        ema_date: Target EMA date
        cols: Column configuration with feature_set list
        window_days: Number of days to include (default: 28)
    
    Returns:
        Markdown table string
    """
    # Feature name mapping for FCTCI format (matching the paper)
    FCTCI_FEATURE_NAMES = {
        'f_loc:phone_locations_doryab_totaldistance:allday': 'total_distance_traveled(meters)',
        'f_loc:phone_locations_doryab_timeathome:allday': 'time_at_home(minutes)',
        'f_loc:phone_locations_doryab_locationentropy:allday': 'location_entropy',
        'f_screen:phone_screen_rapids_sumdurationunlock:allday': 'screen_time(seconds)',
        'f_screen:phone_screen_rapids_avgdurationunlock:allday': 'avg_screen_duration(seconds)',
        'f_call:phone_calls_rapids_incoming_sumduration:allday': 'call_incoming_duration(seconds)',
        'f_call:phone_calls_rapids_outgoing_sumduration:allday': 'call_outgoing_duration(seconds)',
        'f_blue:phone_bluetooth_doryab_uniquedevicesothers:allday': 'bluetooth_devices',
        'f_steps:fitbit_steps_intraday_rapids_sumsteps:allday': 'steps',
        'f_steps:fitbit_steps_intraday_rapids_countepisodesedentarybout:allday': 'sedentary_episodes',
        'f_steps:fitbit_steps_intraday_rapids_sumdurationsedentarybout:allday': 'sedentary_duration(minutes)',
        'f_steps:fitbit_steps_intraday_rapids_countepisodeactivebout:allday': 'active_episodes',
        'f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout:allday': 'active_duration(minutes)',
        'f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:allday': 'sleep_duration(minutes)',
        'f_slp:fitbit_sleep_intraday_rapids_sumdurationawakeunifiedmain:allday': 'awake_duration(minutes)'
    }
    
    # Get feature columns
    if not isinstance(cols['feature_set'], list):
        raise ValueError("FCTCI format requires feature_set to be a list of feature names")
    
    feature_cols = cols['feature_set']
    
    # Get user data for the window
    start_date = ema_date - timedelta(days=window_days)
    user_data = feat_df[
        (feat_df[cols['user_id']] == user_id) & 
        (feat_df[cols['date']] >= start_date) & 
        (feat_df[cols['date']] < ema_date)
    ].sort_values(cols['date'])
    
    if len(user_data) == 0:
        return "No data available for this time window."
    
    # Build markdown table
    # Header: date | feature1 | feature2 | ...
    # Use readable feature names
    available_cols = [col for col in feature_cols if col in user_data.columns]
    readable_headers = ['date'] + [FCTCI_FEATURE_NAMES.get(col, col) for col in available_cols]
    header_row = '|'.join(readable_headers)
    
    # Data rows (no separator row as per the user's example)
    data_rows = []
    for _, row in user_data.iterrows():
        date_str = row[cols['date']].strftime('%Y-%m-%d')
        values = [date_str]
        for col in available_cols:
            val = row[col]
            if pd.isna(val):
                values.append('nan')
            else:
                # Format numbers appropriately
                if isinstance(val, (int, float)):
                    # Round to 2 decimals, but show as int if whole number
                    rounded = round(val, 2)
                    if rounded == int(rounded):
                        values.append(str(int(rounded)))
                    else:
                        values.append(str(rounded))
                else:
                    values.append(str(val))
        data_rows.append('|'.join(values) + '|')
    
    # Combine (no separator row)
    table = header_row + '|' + '\n' + '\n'.join(data_rows)
    return table


def features_to_text_healthllm(feat_df: pd.DataFrame, user_id: str, ema_date: pd.Timestamp,
                                cols: Dict, period_days: int = 14) -> str:
    """
    Convert sensor features to Health-LLM captioning format.
    
    Matches Health-LLM's actual format:
    "The recent 14-days sensor readings show: [Steps] is {avg_steps}. 
    [Sleep] efficiency, duration the user stayed in bed after waking up, 
    duration the user spent to fall asleep, duration the user stayed awake but still in bed, 
    duration the user spent to fall asleep are {eff}, {durafwake}, {dursleep}, {durawake}, 
    {durfall}, {durbed} mins in average"
    
    Note: The 14dhist features are already 14-day statistics, so we just read the value at ema_date.
    
    Args:
        feat_df: Feature dataframe
        user_id: User ID
        ema_date: Target EMA date
        cols: Column configuration with feature_set list (14dhist features)
        period_days: Number of days for statistics (default: 14)
    
    Returns:
        Health-LLM style text description
    """
    # Get feature columns
    if not isinstance(cols['feature_set'], list):
        raise ValueError("Health-LLM format requires feature_set to be a list of feature names")
    
    feature_cols = cols['feature_set']
    
    # Get user data for the specific EMA date
    # The 14dhist features already contain 14-day statistics
    user_data = feat_df[
        (feat_df[cols['user_id']] == user_id) & 
        (feat_df[cols['date']] == ema_date)
    ]
    
    if len(user_data) == 0:
        return f"No data available for the last {period_days} days."
    
    # Extract the row
    row = user_data.iloc[0]
    
    # Feature mapping - exact order from Health-LLM paper
    STEPS_FEATURE = 'f_steps:fitbit_steps_summary_rapids_avgsumsteps:14dhist'
    
    # Sleep features in the exact order they appear in Health-LLM's prompt
    SLEEP_FEATURES_ORDERED = [
        'f_slp:fitbit_sleep_summary_rapids_avgefficiencymain:14dhist',
        'f_slp:fitbit_sleep_summary_rapids_avgdurationafterwakeupmain:14dhist',
        'f_slp:fitbit_sleep_summary_rapids_avgdurationasleepmain:14dhist',
        'f_slp:fitbit_sleep_summary_rapids_avgdurationawakemain:14dhist',
        'f_slp:fitbit_sleep_summary_rapids_avgdurationtofallasleepmain:14dhist',
        'f_slp:fitbit_sleep_summary_rapids_avgdurationinbedmain:14dhist'
    ]
    
    # Build Health-LLM style text - exact format from paper
    result_parts = []
    
    # Get steps value
    steps_val = None
    if STEPS_FEATURE in row.index:
        steps_val = row[STEPS_FEATURE]
        if pd.notna(steps_val):
            steps_val = f"{steps_val:.2f}"
        else:
            steps_val = "N/A"
    else:
        steps_val = "N/A"
    
    # Get sleep values in order
    sleep_vals = []
    for feat in SLEEP_FEATURES_ORDERED:
        if feat in row.index:
            val = row[feat]
            if pd.notna(val):
                sleep_vals.append(f"{val:.2f}")
            else:
                sleep_vals.append("N/A")
        else:
            sleep_vals.append("N/A")
    
    # Build the exact Health-LLM format
    if len(sleep_vals) >= 6:
        result = (
            f"The recent {period_days}-days sensor readings show: "
            f"[Steps] is {steps_val}. "
            f"[Sleep] efficiency, duration the user stayed in bed after waking up, "
            f"duration the user spent to fall asleep, duration the user stayed awake but still in bed, "
            f"duration the user spent to fall asleep are "
            f"{sleep_vals[0]}, {sleep_vals[1]}, {sleep_vals[2]}, {sleep_vals[3]}, {sleep_vals[4]}, {sleep_vals[5]} "
            f"mins in average"
        )
    else:
        result = f"No data available for the last {period_days} days."
    
    return result


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
        
        text += "28 day summary features (P2W slope and R2W slope are calculated based on the past 2 weeks and recent 2 weeks trend):\n"
        for feat_name in statistical.keys():
            stats = statistical.get(feat_name, {})
            struct = structural.get(feat_name, {})
            
            # text += f"{feat_name}\n"
            # text += f"- {window_days} day mean: {stats.get('mean', 'N/A')}, "
            # text += f"std: {stats.get('std', 'N/A')}, "
            # text += f"min: {stats.get('min', 'N/A')}, "
            # text += f"max: {stats.get('max', 'N/A')}\n"
            

            text += f"{feat_name}: "
            text += f"mean={stats.get('mean', 'N/A')}, "
            text += f"sd={stats.get('std', 'N/A')}, "
            text += f"min={stats.get('min', 'N/A')}, "
            text += f"max={stats.get('max', 'N/A')}\n"

            # Slopes
            past_dir = struct.get('past_2weeks_direction', 'stable')
            past_slope = struct.get('past_2weeks_slope', 'N/A')
            recent_dir = struct.get('recent_2weeks_direction', 'stable')
            recent_slope = struct.get('recent_2weeks_slope', 'N/A')
            text += f"- P2W slope=({past_dir}, {past_slope}), R2W slope=({recent_dir}, {recent_slope})\n\n"
        
        # 2. Semantic features
        semantic = agg_feats.get('semantic_features', {})
        if semantic:
            text += "The following shows weekday/weekend patterns, 28-day circadian rhythms, "
            text += "and yesterday's transitions (morning/afternoon/evening/night).\n\n"
            
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
                            text += f"    - {period}={circ[period]}\n"
                
                # Yesterday transition
                if 'yesterday_transition' in sem_info:
                    trans = sem_info['yesterday_transition']
                    text += f"  - Yesterday transition:\n"
                    for period in ['morning', 'afternoon', 'evening', 'night']:
                        if period in trans:
                            text += f"    - {period}: {trans[period]}\n"
                
                text += "\n"
        
        # 3. Temporal descriptors (only if available)
        temporal = agg_feats.get('temporal_descriptors', {})
        if temporal and temporal != 'None':
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
    
    return text


def sample_to_prompt(sample: Dict, cols: Dict, format_type: str = 'structured',
                    include_labels: bool = False, feat_df: pd.DataFrame = None) -> str:
    """
    Convert a sample to prompt text.
    
    Args:
        sample: Sample dictionary with user_id, ema_date, aggregated_features, labels
        cols: Column configuration
        format_type: Format type ('structured'/'compass', 'fctci', 'health-llm')
        include_labels: Whether to include labels
        feat_df: Feature dataframe (required for fctci and health-llm formats)
    
    Returns:
        Formatted prompt text
    """
    prompt = f"User ID: {sample['user_id']}\n"
    prompt += f"Date: {sample['ema_date'].strftime('%Y-%m-%d')}\n"
    prompt += "\nSensor Features:\n"
    
    # Check format type
    if format_type in ['fctci', 'FCTCI']:
        if feat_df is None:
            raise ValueError("feat_df is required for fctci format")
        prompt += features_to_text_fctci(
            feat_df, sample['user_id'], sample['ema_date'], cols, window_days=28
        )
    elif format_type in ['health-llm', 'healthllm', 'health_llm']:
        if feat_df is None:
            raise ValueError("feat_df is required for health-llm format")
        prompt += features_to_text_healthllm(
            feat_df, sample['user_id'], sample['ema_date'], cols, period_days=14
        )
    else:
        # Default: compass/structured format
        agg_feats = sample['aggregated_features']
        if isinstance(agg_feats, dict):
            window_days = agg_feats.get('window_days', 7)
            mode = agg_feats.get('aggregation_mode', 'statistics')
        else:
            window_days = 7
            mode = 'legacy'
        
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
        
        # Convert dates
        feat_df_inst[cols['date']] = pd.to_datetime(feat_df_inst[cols['date']])
        lab_df_inst[cols['date']] = pd.to_datetime(lab_df_inst[cols['date']])
        
        # Binarize labels
        lab_df_inst = binarize_labels(lab_df_inst, cols['labels'], cols['threshold'])
        
        # Count EMAs per user
        user_ema_counts = lab_df_inst.groupby(cols['user_id']).size()
        
        # Filter users with sufficient EMAs
        users_with_emas = user_ema_counts[user_ema_counts >= min_ema_per_user].index.tolist()
        
        print(f"  Total users: {lab_df_inst[cols['user_id']].nunique()}")
        print(f"  Users with >= {min_ema_per_user} EMAs: {len(users_with_emas)}")
        
        # Further filter: check if users have sufficient sensor data for their last N samples
        print(f"  Checking sensor data availability for last {samples_per_user} EMA samples...")
        eligible_users = []
        
        for user_id in users_with_emas:
            user_labs = lab_df_inst[lab_df_inst[cols['user_id']] == user_id].sort_values(cols['date'])
            last_n_samples = user_labs.tail(samples_per_user)
            
            # Check if user has sensor features for these dates
            has_sufficient_data = True
            for _, sample in last_n_samples.iterrows():
                ema_date = sample[cols['date']]
                # Check if user has any sensor data around this EMA date
                user_feat = feat_df_inst[
                    (feat_df_inst[cols['user_id']] == user_id) & 
                    (feat_df_inst[cols['date']] < ema_date)
                ]
                
                # If user has no sensor data before this EMA, skip this user
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
            user_labs = lab_df_inst[lab_df_inst[cols['user_id']] == user_id].sort_values(cols['date'])
            
            # Get last N samples
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
    combined_lab_df = pd.concat(all_lab_dfs, ignore_index=True)  # Only testset samples (414 total)
    
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