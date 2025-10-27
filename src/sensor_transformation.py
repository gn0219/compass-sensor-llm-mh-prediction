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
    from .data_utils import binarize_labels, filter_testset_by_historical_labels, sample_multiinstitution_testset, load_globem_data
except ImportError:
    import config
    from data_utils import binarize_labels, filter_testset_by_historical_labels, sample_multiinstitution_testset

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)


def get_user_window_data(feat_df: pd.DataFrame, user_id: str, ema_date: pd.Timestamp,
                        cols: Dict, window_days: int = 28) -> Optional[pd.DataFrame]:
    """
    Extract user's window data (shared utility for aggregation and time series extraction).
    
    Args:
        feat_df: Feature DataFrame
        user_id: User ID
        ema_date: EMA date (end of window, not included)
        cols: Dictionary with column names
        window_days: Number of days to look back
        
    Returns:
        DataFrame with user's window data, sorted by date, or None if empty
    """
    start_date = ema_date - timedelta(days=window_days)
    
    user_feats = feat_df[
        (feat_df[cols['user_id']] == user_id) & 
        (feat_df[cols['date']] >= start_date) & 
        (feat_df[cols['date']] < ema_date)
    ].copy()
    
    if len(user_feats) == 0:
        return None
    
    # Sort by date for chronological ordering
    return user_feats.sort_values(cols['date'])


def aggregate_window_features(feat_df: pd.DataFrame, user_id: str, ema_date: pd.Timestamp,
                             cols: Dict, window_days: int = None, mode: str = None,
                             use_immediate_window: bool = None, immediate_window_days: int = None,
                             adaptive_window: bool = None,
                             precomputed_window_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
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
        precomputed_window_data: Optional pre-filtered window data (for performance optimization)
        
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
    
    # Use precomputed window data if available (optimization for DTW)
    if precomputed_window_data is not None:
        user_feats = precomputed_window_data
    else:
        # Use shared utility to get window data
        user_feats = get_user_window_data(feat_df, user_id, ema_date, cols, window_days)
    
    if user_feats is None:
        return None
    
    
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
                    if 'Physical activity' in base_name and 'steps' in feat_col.lower() and 'sumsteps' in feat_col.lower():
                        match = True
                    elif 'Sleep' in base_name and 'sleep' in feat_col.lower() and 'duration' in feat_col.lower():
                        match = True
                    elif 'Location entropy' in base_name and 'locationentropy' in feat_col.lower():
                        match = True
                    elif 'Phone usage' in base_name and ('unlock' in feat_col.lower() or 'screen' in feat_col.lower()):
                        match = True
                    
                    if match and f':{period}' in feat_col:
                        if feat_col in yesterday_data.columns:
                            val = yesterday_data[feat_col].iloc[0]
                            if pd.notna(val):
                                yesterday_values[period] = round(val, 1)
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
            text += "The following shows weekday/weekend patterns, 28-day circadian rhythms (mean), "
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
                    text += f"  - 28 day rhythms: "
                    for period in ['morning', 'afternoon', 'evening', 'night']:
                        if period in circ:
                            if period != 'night':
                                text += f"{period}={circ[period]}, "
                            else:
                                text += f"{period}={circ[period]}\n"
                
                # Yesterday transition
                if 'yesterday_transition' in sem_info:
                    trans = sem_info['yesterday_transition']
                    text += f"  - Yesterday transition: "
                    for period in ['morning', 'afternoon', 'evening', 'night']:
                        if period in trans:
                            if period != 'night':
                                text += f"{period}={trans[period]}, "
                            else:
                                text += f"{period}={trans[period]}"
        
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
                    include_labels: bool = False, feat_df: pd.DataFrame = None,
                    include_user_info: bool = True) -> str:
    """
    Convert a sample to prompt text.
    
    Args:
        sample: Sample dictionary with user_id, ema_date, aggregated_features, labels
        cols: Column configuration
        format_type: Format type ('structured'/'compass', 'fctci', 'health-llm')
        include_labels: Whether to include labels
        feat_df: Feature dataframe (required for fctci and health-llm formats)
        include_user_info: Whether to include User ID and Date (default True)
    
    Returns:
        Formatted prompt text
    """
    prompt = ""
    if include_user_info:
        prompt = f"User ID: {sample['user_id']}\n"
        prompt += f"Date: {sample['ema_date'].strftime('%Y-%m-%d')}\n"
        prompt += "\n"
    
    prompt += "Sensor Features:\n"
    
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
            # window_days = agg_feats.get('window_days', 7)
            mode = agg_feats.get('aggregation_mode', 'statistics')
        else:
            # window_days = 7
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