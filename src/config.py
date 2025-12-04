"""
Configuration Constants

Central location for all configuration parameters used across the project.
Modify these values to change system behavior without touching code.
"""

# ============================================================================
# FILE PATHS
# ============================================================================

# Default output directory
DEFAULT_OUTPUT_DIR = './results'

# Dataset paths (relative to project root)
# Note: Dataset paths now managed in DATASET_CONFIGS

# Dataset selection: 'globem', 'ces', or 'mentaliot'
DATASET_TYPE = 'mentaliot'  # Change to 'ces' or 'mentaliot' to use different datasets

# ============================================================================
# DATASET METADATA
# ============================================================================

# Centralized dataset configuration - eliminates hardcoding
DATASET_CONFIGS = {
    'globem': {
        'name': 'GLOBEM',
        'base_path': '../dataset/Globem',
        'testset_file': '../dataset/Globem/globem_testset.csv',
        'trainset_file': '../dataset/Globem/globem_trainset.csv',
        'aggregated_file': '../dataset/Globem/aggregated_globem.csv',
        'daily_features_file': '../dataset/Globem/daily_features_globem.csv',  # For ML and DTW
        'use_cols_path': './config/globem_use_cols.json',
        'labels': ['phq4_anxiety_EMA', 'phq4_depression_EMA'],  # NO stress
        'similarity_method': 'dtw',  # 'dtw' or 'cosine'
        'has_pre_aggregated': True,  # Features pre-aggregated
        'has_daily_features': True,  # Daily features available for DTW/ML
    },
    'ces': {
        'name': 'CES',
        'base_path': '../dataset/CES',
        'testset_file': '../dataset/CES/ces_testset.csv',
        'trainset_file': '../dataset/CES/ces_trainset.csv',
        'aggregated_file': '../dataset/CES/aggregated_ces.csv',
        'use_cols_path': './config/ces_use_cols.json',
        'labels': ['phq4_anxiety_EMA', 'phq4_depression_EMA', 'stress'],  # WITH stress
        'similarity_method': 'dtw',  # 'dtw' or 'cosine'
        'has_pre_aggregated': True,  # Features pre-computed
    },
    'mentaliot': {
        'name': 'MentalIoT',
        'base_path': '../dataset/MentalIoT',
        'testset_file': '../dataset/MentalIoT/mentaliot_testset.csv',
        'trainset_file': '../dataset/MentalIoT/mentaliot_trainset.csv',
        'aggregated_file': '../dataset/MentalIoT/aggregated_mentaliot.csv',
        'use_cols_path': './config/mentaliot_use_cols.json',
        'labels': ['phq2_result_binary', 'gad2_result_binary', 'stress_result_binary'],  # WITH stress
        'similarity_method': 'cosine',  # 'dtw' or 'cosine'
        'has_pre_aggregated': True,  # Features pre-computed
    }
}

# Get current dataset config
CURRENT_DATASET_CONFIG = DATASET_CONFIGS[DATASET_TYPE]

# Note: Testset configurations now in prepare_*_data.py scripts
# GLOBEM: prepare_globem_data.py
# CES: prepare_ces_data.py (if exists)
# MentalIoT: prepare_mentaliot_data.py

# CES TimeRAG quarterly chunking parameters
CES_TIMERAG_MIN_SAMPLES_THRESHOLD = 20  # Min samples to trigger clustering
CES_TIMERAG_MIN_K = 5  # Minimum K for clustering
CES_TIMERAG_MAX_K_PER_CHUNK = 30  # Maximum K per quarterly chunk
CES_TIMERAG_MAX_RAW_SAMPLES = 100  # Max raw samples in current quarter before clustering

# TimeRAG Retrieval Configuration
TIMERAG_POOL_SIZE = 300  # Number of representative samples for DTW candidate pool (via clustering)
RETRIEVAL_DIVERSITY_FACTOR = 2.0  # Multiplier for initial retrieval to ensure label diversity (e.g., 2.0 = retrieve 2x candidates, then select with balanced labels)

# Sensor data format and features
# Options: 
#   'compass' - Statistical/structural/semantic features with descriptive text
#   'fctci' - Markdown table format (From Classification to Clinical Insights)
#   'health-llm' - Statistical summaries (max, min, avg, median, std)
DEFAULT_TARGET = 'compass'

# Configuration files
PROMPT_CONFIGS_PATH = '../config/prompt_configs.yaml'
USE_COLS_PATH = './config/globem_use_cols.json'  # Default for GLOBEM
CES_USE_COLS_PATH = './config/ces_use_cols.json'  # For CES
MENTALIOT_USE_COLS_PATH = './config/mentaliot_use_cols.json'  # For MentalIoT

# ============================================================================
# DATA PROCESSING
# ============================================================================

# Time window for feature aggregation
AGGREGATION_WINDOW_DAYS = 28

# Immediate window for recent behavioral patterns
IMMEDIATE_WINDOW_DAYS = 7

# Whether to use immediate window by default
USE_IMMEDIATE_WINDOW = True

# Aggregation mode: 'array' or 'statistics'
# - 'array': Raw daily values as list [val1, val2, ..., valN]
# - 'statistics': Statistical summaries (mean, std, slope, etc.)
DEFAULT_AGGREGATION_MODE = 'statistics'

# Adaptive window for early samples (ICL examples)
# If True, use all available data when window_days exceeds available history
# Useful for personalized ICL where early samples have limited history
USE_ADAPTIVE_WINDOW = True

# Test set filtering: skip if you want all samples (not recommended for personalized ICL)
FILTER_TESTSET_BY_HISTORY = True

# Missing data threshold (samples with more missing data are excluded)
MISSING_RATIO_THRESHOLD = 0.7

# Label binarization thresholds
DEFAULT_THRESHOLDS = {
    'phq4_anxiety_EMA': 2,      # PHQ-4 anxiety subscale
    'phq4_depression_EMA': 2    # PHQ-4 depression subscale
}

# ============================================================================
# IN-CONTEXT LEARNING
# ============================================================================

# Default number of ICL examples
DEFAULT_N_SHOT = 4

# Note: ICL strategy specified via --strategy argument (cross_random, cross_retrieval, personal_recent, hybrid)

# Minimum historical labels required for test set samples
# This ensures personalized ICL always has sufficient examples
# For fair comparison across generalized, personalized, and hybrid strategies
MIN_HISTORICAL_LABELS = 4

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Default model
DEFAULT_MODEL = 'gpt-5-nano'

# Supported models
SUPPORTED_MODELS = [
    'gpt-5',
    'gpt-5-nano',          # OpenAI
    'claude-4.5-sonnet',   # OpenRouter (Anthropic)
    'gemini-2.5-pro',      # OpenRouter (Google)
    'gemini-2.5-flash',    # OpenRouter (Google)
    'llama-3.1-8b',        # Ollama (on-device)
    'llama-3.2-3b',        # Ollama (on-device)
    'mistral-7b',          # Ollama (on-device)
    'qwen3-4b',            # Ollama (on-device)
    'gemma2-9b',           # Ollama (on-device)
    'gpt-oss-20b',         # OpenRouter (OpenAI)
    'mistral-7b-instruct', # OpenRouter (Mistral)
]

# Default temperature for generation
DEFAULT_TEMPERATURE = 1.0

# Default max tokens for completion
DEFAULT_MAX_TOKENS = 6000

# Default reasoning method
DEFAULT_REASONING_METHOD = 'cot'  # Options: 'direct', 'cot', 'tot', 'self_consistency'

# Self-consistency parameters
SELF_CONSISTENCY_N_SAMPLES = 5
SELF_CONSISTENCY_TEMPERATURE = 0.9

# ============================================================================
# EVALUATION
# ============================================================================

# Stratified sampling for batch evaluation
# Now uses ALL labels for stratification (e.g., anxiety_depression: 0_0, 0_1, 1_0, 1_1)
USE_STRATIFIED_SAMPLING = True

# Minimum samples for reliable metrics
MIN_SAMPLES_FOR_METRICS = 10

# Statistical significance test
DEFAULT_SIGNIFICANCE_TEST = 'mcnemar'  # Options: 'mcnemar', 'bootstrap'
DEFAULT_ALPHA = 0.05
N_BOOTSTRAP_SAMPLES = 1000

# ============================================================================
# EXPERIMENT NAMING
# ============================================================================

# Dataset name for result files (set automatically based on DATASET_TYPE)
DATASET_NAME = DATASET_TYPE  # 'globem' or 'ces'

# ============================================================================
# FEATURE NAMES MAPPING
# ============================================================================

# Note: Feature name mapping is now handled in globem_use_cols.json

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
