# CES Dataset Integration Guide

## Overview

The CES (College Experience Study) dataset has been successfully integrated into the mental health prediction pipeline. This guide explains how to use CES data for LLM-based mental health prediction.

## Key Differences: CES vs GLOBEM

| Aspect | GLOBEM | CES |
|--------|--------|-----|
| **Data Format** | Raw sensor data (computed on-the-fly) | Pre-aggregated features |
| **Labels** | 2 labels (anxiety, depression) | 3 labels (anxiety, depression, stress) |
| **Time Periods** | morning/afternoon/evening/night | ep_0/ep_1/ep_2/ep_3 (12am-9am/9am-6pm/6pm-12am) |
| **User Sampling** | Multi-institution based | Gender-balanced sampling |
| **TimeRAG Retrieval** | Simple clustering | Quarterly chunking with adaptive K |
| **Test Set** | Multi-institution (138 users, 3 samples each) | 60 users, 5 samples each (300 total) |

## Setup

### 1. Data Preparation

CES data should be organized as follows:
```
dataset/CES/
├── Demographics/demographics.csv
├── Sensing/sensing.csv
├── Sensing/steps.csv
├── EMA/general_ema.csv
└── aggregated_ces.csv (generated automatically)
```

### 2. Configuration

Edit `src/config.py`:

```python
# Change dataset type
DATASET_TYPE = 'ces'  # Switch from 'globem' to 'ces'
```

Other CES-specific configurations:
- `CES_N_USERS = 60`: Number of test users
- `CES_MIN_EMA_PER_USER = 9`: Minimum EMAs required (4 ICL + 5 test)
- `CES_SAMPLES_PER_USER = 5`: Test samples per user

### 3. Feature Configuration

CES features are defined in `config/ces_use_cols.json`:

```json
{
  "compass": {
    "user_id": "uid",
    "user_info": "gender",
    "date": "date",
    "feature_set": {
      "statistical": {
        "loc_dist_ep_0": "Location - total distance",
        "unlock_duration_ep_0": "Screen - sum duration unlock",
        ...
      },
      "semantic": {
        "sleep_duration": "Sleep - weekday sleep duration",
        "step_ep_1": "Steps - number of steps 12am-9am",
        ...
      }
    },
    "labels": ["phq4_anxiety_EMA", "phq4_depression_EMA", "stress"],
    "threshold": {
      "phq4_anxiety_EMA": 2,
      "phq4_depression_EMA": 2,
      "stress": 2
    }
  }
}
```

## Usage

### Running Evaluation on CES

The pipeline automatically detects the dataset type from `config.DATASET_TYPE`:

```bash
# Make sure DATASET_TYPE = 'ces' in config.py

# Zero-shot evaluation
python run_evaluation.py --mode batch --n_samples 300 --strategy none --seed 42 --save-prompts-only

# Few-shot with cross-user random
python run_evaluation.py --mode batch --strategy cross_random --n_shot 4 --seed 42 --save-prompts-only

# Few-shot with cross-user retrieval (TimeRAG + quarterly chunking)
python run_evaluation.py --mode batch --strategy cross_retrieval --n_shot 4 --seed 42 --save-prompts-only

# Few-shot with personal recent
python run_evaluation.py --mode batch --strategy personal_recent --n_shot 4 --seed 42 --save-prompts-only

# Few-shot with hybrid blend
python run_evaluation.py --mode batch --strategy hybrid_blend --n_shot 4 --use-dtw --seed 42 --save-prompts-only
```

### Data Loading

The system automatically handles data loading:

1. **First Run**: Builds aggregated features from raw data (slow, ~30-60 minutes)
   - Loads raw sensing, steps, and EMA data
   - Creates phq4_anxiety_EMA and phq4_depression_EMA from phq4-1, phq4-2, phq4-3, phq4-4
   - Aggregates 28-day windows with statistical, structural, and semantic features
   - Saves to `dataset/CES/aggregated_ces.csv`

2. **Subsequent Runs**: Loads cached aggregated data (fast, ~10 seconds)

### Test Set Sampling

CES uses gender-balanced user sampling:

```python
from src.data_utils import sample_ces_testset

feat_df, test_df, train_df, cols = sample_ces_testset(
    n_users=60,
    min_ema_per_user=9,
    samples_per_user=5,
    random_state=42
)
```

Output:
- `feat_df`: Full feature DataFrame (for ICL retrieval)
- `test_df`: 300 test samples (60 users × 5 samples)
- `train_df`: Remaining samples for ICL
- `cols`: Column configuration

## Feature Aggregation

### Statistical Features
For each feature (e.g., `loc_dist_ep_0`):
- `loc_dist_ep_0_28mean`: Mean over 28 days
- `loc_dist_ep_0_28std`: Standard deviation
- `loc_dist_ep_0_28min`: Minimum value
- `loc_dist_ep_0_28max`: Maximum value

### Structural Features
- `loc_dist_ep_0_p2wslope`: Past 2 weeks trend (days 1-14)
- `loc_dist_ep_0_r2wslope`: Recent 2 weeks trend (days 15-28)

### Semantic Features

**Weekday/Weekend** (for sleep and ep_0 features):
- `sleep_duration_28weekday`: Average sleep on weekdays
- `sleep_duration_28weekend`: Average sleep on weekends

**Time-of-Day Patterns** (for ep_1/2/3 features):
- `step_ep_1_28mean`: Average steps in 12am-9am over 28 days
- `step_ep_1_yesterday`: Steps in 12am-9am yesterday
- `step_ep_2_28mean`: Average steps in 9am-6pm over 28 days
- `step_ep_2_yesterday`: Steps in 9am-6pm yesterday
- `step_ep_3_28mean`: Average steps in 6pm-12am over 28 days
- `step_ep_3_yesterday`: Steps in 6pm-12am yesterday

### Raw Features (for TimeRAG)
- `loc_dist_ep_0_before1day`: Value 1 day ago
- `loc_dist_ep_0_before2day`: Value 2 days ago
- ...
- `loc_dist_ep_0_before28day`: Value 28 days ago

## TimeRAG Retrieval with Quarterly Chunking

CES uses an advanced TimeRAG strategy with quarterly chunking:

### Offline Phase (Pre-computation)

1. **Divide data into quarters** (Q1 2017, Q2 2017, ...)
2. **For each quarter**:
   - If samples < 20: Keep all samples
   - Else: Cluster with adaptive K = max(5, min(√N, 30))

### Online Phase (Per test sample)

1. **Pool 1**: All representatives from past quarters
2. **Pool 2**: Raw samples from current quarter before target date
   - If > 100 samples: Cluster again
   - Else: Use all raw samples
3. **Final pool**: Pool 1 + Pool 2

### User Deduplication

After DTW ranking:
1. Retrieve top-(n_shot × 2) most similar samples
2. Deduplicate by user (keep closest per user)
3. Select n_shot samples with balanced labels

## Sensor-to-Text Transformation

CES features are converted to natural language:

```
28 day summary features (P2W slope and R2W slope are calculated based on the past 2 weeks and recent 2 weeks trend):
Location - total distance: mean=5234.5, sd=1234.2, min=1200.0, max=8900.0
- P2W slope=(increasing, 0.12), R2W slope=(stable, 0.02)

The following shows weekday/weekend patterns, 28-day time-of-day patterns (mean), and yesterday's transitions (12am-9am/9am-6pm/6pm-12am).

- Steps
  - Weekday: 8234.5, Weekend: 6123.4 (diff=-2111.1)
  - 28 day patterns: 12am-9am=234.5, 9am-6pm=5234.2, 6pm-12am=2765.8
  - Yesterday transition: 12am-9am=189.3, 9am-6pm=5678.1, 6pm-12am=2987.6
```

## Label Processing

CES computes composite labels:
- `phq4_anxiety_EMA = phq4-1 + phq4-2`
- `phq4_depression_EMA = phq4-3 + phq4-4`
- `stress = stress` (already provided)

Binary thresholds (> 2 = high risk):
- Anxiety > 2
- Depression > 2
- Stress > 2

## Code Structure

### New/Modified Files

**Core Data Processing**:
- `src/data_utils.py`:
  - `load_ces_data()`: Load aggregated CES data (with caching)
  - `_aggregate_ces_data()`: Build aggregated features from raw data
  - `_build_ces_aggregated_row()`: Aggregate single sample
  - `sample_ces_testset()`: Gender-balanced test set sampling

**Sensor Transformation**:
- `src/sensor_transformation.py`:
  - `features_to_text_ces()`: Convert CES aggregated features to text
  - `_get_slope_direction()`: Helper for slope interpretation

**TimeRAG Retrieval**:
- `src/timerag_retrieval.py`:
  - `build_retrieval_candidate_pool_timerag_ces()`: Quarterly chunking
  - `_extract_ces_samples_as_candidates()`: Extract samples without clustering
  - `_cluster_ces_quarterly_chunk()`: K-means clustering per quarter
  - `_extract_ces_time_series_from_row()`: Extract time series from aggregated row
  - `sample_from_timerag_pool_dtw_ces()`: DTW ranking with user deduplication

**Example Selection**:
- `src/example_selection.py`:
  - Updated `build_retrieval_candidate_pool()`: Support CES
  - Updated `select_icl_examples()`: Support CES
  - Updated `_sample_cross_random()`: Handle CES aggregated data
  - Updated `_sample_personal_recent()`: Handle CES aggregated data

**Configuration**:
- `src/config.py`: Added CES-specific settings
- `config/ces_use_cols.json`: CES feature definitions

## Switching Between Datasets

To switch between GLOBEM and CES, simply change:

```python
# src/config.py
DATASET_TYPE = 'globem'  # or 'ces'
```

The pipeline automatically:
- Uses correct data loading function
- Applies appropriate feature aggregation
- Selects correct TimeRAG strategy
- Formats prompts accordingly

## Troubleshooting

### Issue: "aggregated_ces.csv not found"
**Solution**: Run once to build aggregated data (takes 30-60 minutes). Subsequent runs will be fast.

### Issue: "Not enough users with >= 9 EMAs"
**Solution**: Reduce `CES_MIN_EMA_PER_USER` or `CES_N_USERS` in config.py

### Issue: "Missing features in aggregated data"
**Solution**: Check `config/ces_use_cols.json` - ensure feature names match CES column names

## Performance Notes

- **Data Aggregation**: ~30-60 minutes (first time only)
- **TimeRAG Quarterly Chunking**: ~5-10 minutes (per experiment)
- **Test Set Sampling**: ~10 seconds
- **Prompt Generation**: ~2-5 minutes (for 300 samples)

## Future Enhancements

Potential improvements for CES integration:
1. Incremental aggregation (update only new samples)
2. Parallel aggregation (use multiprocessing)
3. Feature importance analysis
4. Cross-dataset transfer learning (GLOBEM → CES)

---

**Last Updated**: October 28, 2025
**Version**: 1.0
**Author**: AI Assistant

