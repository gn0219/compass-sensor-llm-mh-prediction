"""
Configuration Constants

Central location for all configuration parameters used across the project.
Modify these values to change system behavior without touching code.
"""

# ============================================================================
# DATA PROCESSING
# ============================================================================

# Time window for feature aggregation
AGGREGATION_WINDOW_DAYS = 7

# Missing data threshold (samples with more missing data are excluded)
MISSING_RATIO_THRESHOLD = 0.7

# Label binarization thresholds
DEFAULT_THRESHOLDS = {
    'phq4_anxiety_EMA': 3,      # PHQ-4 anxiety subscale
    'phq4_depression_EMA': 3    # PHQ-4 depression subscale
}

# ============================================================================
# IN-CONTEXT LEARNING
# ============================================================================

# Default number of ICL examples
DEFAULT_N_SHOT = 5

# Default ICL source strategy
DEFAULT_ICL_SOURCE = 'hybrid'  # Options: 'personalization', 'generalization', 'hybrid'

# ICL example selection method
DEFAULT_SELECTION_METHOD = 'random'  # Options: 'random', 'similarity' (not implemented)

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Default model
DEFAULT_MODEL = 'gpt-5-nano'

# Supported models
SUPPORTED_MODELS = [
    'gpt-5-nano',          # OpenAI
    'claude-3-5-sonnet',   # Anthropic
    'gemini-2.5-pro',      # Google
    'llama-3.1-8b',        # Ollama (on-device)
    'mistral-7b',          # Ollama (on-device)
    'alpaca-7b',           # Ollama (on-device)
]

# Default temperature for generation
DEFAULT_TEMPERATURE = 1.0

# Default max tokens for completion
DEFAULT_MAX_TOKENS = 1000

# Default reasoning method
DEFAULT_REASONING_METHOD = 'cot'  # Options: 'direct', 'cot', 'tot', 'self_consistency'

# Self-consistency parameters
SELF_CONSISTENCY_N_SAMPLES = 5
SELF_CONSISTENCY_TEMPERATURE = 0.9

# ============================================================================
# EVALUATION
# ============================================================================

# Default batch size for evaluation
DEFAULT_BATCH_SIZE = 30

# Minimum samples for reliable metrics
MIN_SAMPLES_FOR_METRICS = 10

# Statistical significance test
DEFAULT_SIGNIFICANCE_TEST = 'mcnemar'  # Options: 'mcnemar', 'bootstrap'
DEFAULT_ALPHA = 0.05
N_BOOTSTRAP_SAMPLES = 1000

# ============================================================================
# FILE PATHS
# ============================================================================

# Default output directory
DEFAULT_OUTPUT_DIR = './results'

# Dataset paths (relative to project root)
GLOBEM_BASE_PATH = '../dataset/Globem'
DEFAULT_INSTITUTION = 'INS-W_2'
DEFAULT_TARGET = 'fctci'

# Configuration files
PROMPT_CONFIGS_PATH = '../config/prompt_configs.yaml'
USE_COLS_PATH = 'use_cols.json'

# ============================================================================
# EXPERIMENT NAMING
# ============================================================================

# Dataset name for result files
DATASET_NAME = 'globem'

# Sensor-to-text format
SENSOR_FORMAT = 'structured'  # Options: 'structured', 'narrative'

# Reasoning method abbreviations
REASONING_ABBREV = {
    'direct': 'direct',
    'cot': 'cot',
    'tot': 'tot',
    'self_consistency': 'sc'
}

# ICL source abbreviations
ICL_SOURCE_ABBREV = {
    'personalization': 'personalized',
    'generalization': 'generalized',
    'hybrid': 'hybrid'
}

# ============================================================================
# FEATURE NAMES MAPPING
# ============================================================================

# Mapping for cleaner feature name display
FEATURE_NAME_MAPPING = {
    'f_loc:phone_locations_doryab_': 'Location - ',
    'f_screen:phone_screen_rapids_': 'Screen - ',
    'f_call:phone_calls_rapids_': 'Call - ',
    'f_blue:phone_bluetooth_doryab_': 'Bluetooth - ',
    'f_steps:fitbit_steps_intraday_rapids_': 'Activity - ',
    'f_slp:fitbit_sleep_intraday_rapids_': 'Sleep - ',
    'phone_locations_doryab_': '',
    'phone_screen_rapids_': '',
    'phone_calls_rapids_': '',
    'phone_bluetooth_doryab_': '',
    'fitbit_steps_intraday_rapids_': '',
    'fitbit_sleep_intraday_rapids_': '',
    ':allday': '',
    '_': ' '
}

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Number of decimal places for metrics
METRIC_DECIMAL_PLACES = 4

# Number of decimal places for timing
TIMING_DECIMAL_PLACES = 3

# Number of decimal places for cost
COST_DECIMAL_PLACES = 6

# Console output width
CONSOLE_WIDTH = 80

