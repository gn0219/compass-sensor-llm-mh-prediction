"""
Evaluation Runner Module (refined)

- Removes duplication with small helpers (timeit, ICL helpers, prompt builder wrapper).
- Keeps the original flow & outputs; fixes None-return issues and positional-arg bugs.
"""

import os
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from . import config
from .data_utils import sample_batch_stratified
from .sensor_transformation import sample_input_data, check_missing_ratio, aggregate_window_features
from .example_selection import select_icl_examples, build_retrieval_candidate_pool
from .prompt_utils import build_prompt
from .reasoning import LLMReasoner
from .performance import generate_comprehensive_report, print_comprehensive_report
from .output_utils import (
    print_input_sample_info, print_icl_selection_info, print_prompt_building_info,
    print_prediction_results, print_timing_breakdown, print_batch_progress, print_batch_timing_summary,
)
from .prompt_manager import PromptManager


# ---------------------------
# Utilities (timing & setup)
# ---------------------------

def new_step_timings(keys=('loading', 'test_sampling', 'feature_engineering', 'icl_selection', 'prompt_building', 'llm_call', 'response_parsing')):
    """Create a fresh timings dict with list slots."""
    return {k: [] for k in keys}

@contextmanager
def timeit(step_timings: Dict[str, List[float]], key: str):
    """Append elapsed seconds for the given key, even if an exception happens."""
    t0 = time.time()
    try:
        yield
    finally:
        step_timings[key].append(time.time() - t0)

def append_zero(step_timings: Dict[str, List[float]], key: str):
    step_timings[key].append(0.0)

# ---------------------------
# ICL selection & prompt build
# ---------------------------

def select_icl(
    feat_df, lab_df, cols: Dict, input_sample: Dict,
    n_shot: int, strategy: str, use_dtw: bool, random_state: Optional[int],
    step_timings: Dict[str, List[float]], verbose: bool,
    retrieval_candidates: Optional[List[Dict]] = None, dataset: str = 'globem',
    lab_df_for_icl=None, full_lab_df_for_personal=None
):
    """
    Select ICL examples if needed, append timing, and return (icl_examples, icl_strategy).
    
    Args:
        strategy: 'cross_random', 'cross_retrieval', 'personal_recent', 'hybrid', or 'none'
        use_dtw: For hybrid, whether to use DTW for cross-user part
        retrieval_candidates: Prebuilt candidate pool for retrieval strategies
        dataset: Dataset type ('globem' or 'ces')
        lab_df_for_icl: Trainset for cross-user ICL
        full_lab_df_for_personal: Full data (train+test) for personal ICL
    """
    icl_examples = None
    icl_strategy = 'zero_shot'

    if n_shot > 0 and strategy != 'none':
        if verbose:
            msg = f"\n[ICL] Selecting {n_shot} examples (strategy: {strategy}"
            if strategy == 'hybrid':
                msg += f", cross-user: {'DTW' if use_dtw else 'random'}"
            msg += ")..."
            print(msg)
        with timeit(step_timings, 'icl_selection'):
            # Choose appropriate lab_df based on strategy
            icl_lab_df = full_lab_df_for_personal if strategy in ['personal_recent', 'hybrid'] and full_lab_df_for_personal is not None else lab_df_for_icl if lab_df_for_icl is not None else lab_df
            
            icl_examples = select_icl_examples(
                feat_df, icl_lab_df, cols,
                input_sample['user_id'], input_sample['ema_date'],
                n_shot=n_shot, strategy=strategy, use_dtw=use_dtw,
                random_state=random_state, target_sample=input_sample,
                retrieval_candidates=retrieval_candidates, dataset=dataset
            )
        if icl_examples is None:
            if verbose:
                print("  [WARNING]  Failed to select ICL examples, falling back to zero-shot")
            icl_strategy = 'zero_shot'
        else:
            if verbose:
                print(f"  [OK] Selected {len(icl_examples)} examples in {step_timings['icl_selection'][-1]:.2f}s")
            icl_strategy = strategy
    else:
        if verbose:
            print(f"\n[ICL] Using zero-shot (no ICL examples)")
        append_zero(step_timings, 'icl_selection')

    return icl_examples, icl_strategy


def build_prompt_with_timing(
    prompt_manager: PromptManager, input_sample: Dict, cols: Dict,
    icl_examples, icl_strategy: str, reasoning_method: str,
    step_timings: Dict[str, List[float]], verbose: bool, feat_df=None
) -> str:
    if verbose:
        print(f"\n[Prompt] Building prompt (reasoning: {reasoning_method})...")
    with timeit(step_timings, 'prompt_building'):
        prompt = build_prompt(prompt_manager, input_sample, cols, icl_examples, icl_strategy, reasoning_method, feat_df=feat_df, step_timings=step_timings)
    
    # Print feature engineering time separately if available
    if 'feature_engineering' in step_timings and len(step_timings['feature_engineering']) > 0:
        feat_time = step_timings['feature_engineering'][-1]
        prompt_time = step_timings['prompt_building'][-1]
        if verbose:
            print(f"  [OK] Feature engineering: {feat_time:.2f}s, Prompt assembly: {prompt_time - feat_time:.2f}s")
    else:
        print_prompt_building_info(len(prompt), step_timings['prompt_building'][-1], verbose)
    
    return prompt


# ---------------------------
# Predict 
# ---------------------------

def predict(all_predictions: List[Dict], all_step_timings: Dict[str, List[float]],
            user_id, ema_date, true_anxiety, true_depression, failed_count: int,
            reasoner: LLMReasoner, reasoning_method: str, prompt: str, *,
            true_stress=None, llm_seed: Optional[int] = None, sc_samples: Optional[int] = None,
            verbose: bool = True) -> Tuple[List[Dict], Dict[str, List[float]], int, Optional[Dict]]:
    """
    Prediction executor.
    Always returns (all_predictions, all_step_timings, failed_count, pred).
    
    Args:
        true_stress: True stress label (optional, for CES dataset)
    """
    with timeit(all_step_timings, 'llm_call'):
        if reasoning_method == 'self_consistency':
            n_sc = sc_samples or 5
            prediction, _ = reasoner.predict_with_self_consistency(
                prompt, n_samples=n_sc, seed=llm_seed
            )
        elif reasoning_method == 'self_feedback':
            # Self-feedback: iterative refinement
            prediction, iterations = reasoner.predict_with_self_feedback(
                prompt, max_iterations=3, temperature=0.7, seed=llm_seed
            )
            if verbose and iterations:
                print(f"  [Self-Feedback] Total iterations: {len(iterations)}")
        else:
            response_text, usage_info = reasoner.call_llm(prompt, seed=llm_seed)

    if reasoning_method in ['self_consistency', 'self_feedback']:       
        # Parsing is internal to self-consistency and self-feedback paths
        append_zero(all_step_timings, 'response_parsing')
    else:
        if not response_text:
            if verbose:
                print("  [WARNING]  LLM call failed, skipping")
            failed_count += 1
            append_zero(all_step_timings, 'response_parsing')
            return all_predictions, all_step_timings, failed_count, None

        with timeit(all_step_timings, 'response_parsing'):
            prediction = reasoner.parse_response(response_text)

    if not prediction:
        if verbose:
            print("  [WARNING]  Parse failed, skipping")
        failed_count += 1
        return all_predictions, all_step_timings, failed_count, None

    pred = {
        'user_id': user_id,
        'ema_date': ema_date,
        'true_anxiety': true_anxiety,
        'true_depression': true_depression,
        'pred_anxiety': prediction['Prediction']['Anxiety_binary'],
        'pred_depression': prediction['Prediction']['Depression_binary'],
        'prediction': prediction,
    }
    
    # Add stress if available (CES dataset)
    if true_stress is not None:
        pred['true_stress'] = true_stress
        pred['pred_stress'] = prediction['Prediction'].get('Stress_binary', 0)
    
    all_predictions.append(pred)

    if verbose:
        print(f"  User ID: {user_id}")
        print(f"  Date: {ema_date}")
        
        # Build prediction output dynamically
        targets = [
            ('Anx', pred['pred_anxiety'], true_anxiety),
            ('Dep', pred['pred_depression'], true_depression),
        ]
        if true_stress is not None:
            targets.append(('Stress', pred['pred_stress'], true_stress))
        
        pred_parts = [f"{name}: {pred_val} (true: {true_val})" 
                      for name, pred_val, true_val in targets]
        print(f"  ✓ {' | '.join(pred_parts)}")

    return all_predictions, all_step_timings, failed_count, pred

def records_to_report_items(records: List[Dict]) -> List[Dict]:
    """Convert record schema to generate_comprehensive_report's expected items."""
    items = []
    for r in records:
        item = {
            'user_id': r['user_id'],
            'ema_date': str(r['ema_date']),
            'labels': {
                'phq4_anxiety_EMA': r['true_anxiety'],
                'phq4_depression_EMA': r['true_depression'],
            },
            'prediction': {
                'Anxiety_binary': r['pred_anxiety'],
                'Depression_binary': r['pred_depression'],
            },
        }
        
        # Add stress if available (CES dataset)
        if 'true_stress' in r:
            item['labels']['stress'] = r['true_stress']
            item['prediction']['Stress_binary'] = r.get('pred_stress', 0)
        
        items.append(item)
    
    return items

def run_batch_evaluation(prompt_manager: PromptManager, reasoner: LLMReasoner,
                         feat_df, lab_df, cols: Dict, n_samples: int = 30, n_shot: int = 5, 
                         strategy: str = 'cross_random', use_dtw: bool = False, reasoning_method: str = 'cot', 
                         random_state: Optional[int] = 42, llm_seed: Optional[int] = None, 
                         collect_prompts: bool = False, verbose: bool = True,
                         initial_timings: Optional[Dict[str, float]] = None,
                         dataset: str = 'globem') -> Optional[Dict]:
    """Run batch evaluation on multiple samples.
    
    For pre-generated testsets (all datasets now), uses all samples from lab_df.
    For ad-hoc evaluation, samples n_samples from lab_df.
    """
    
    # Determine actual number of samples to use
    # If lab_df is small (<= n_samples * 2), assume it's a pre-selected testset and use all
    use_all_samples = len(lab_df) <= n_samples * 2
    actual_n_samples = len(lab_df) if use_all_samples else n_samples
    
    if verbose:
        print("\n" + "="*60 + f"\n🔬 BATCH EVALUATION ({actual_n_samples} samples)" + "\n" + "="*60)
        if use_all_samples:
            print(f"  [Using all pre-selected samples from testset]")
        print(f"  ICL Strategy: {strategy} | N-Shot: {n_shot} | Reasoning: {reasoning_method} | Model: {reasoner.model}")
        if random_state or llm_seed:
            print(f"   Seed: {random_state} | LLM Seed: {llm_seed}")
        print(f"  Config: {config.AGGREGATION_WINDOW_DAYS}d window | {config.DEFAULT_AGGREGATION_MODE} mode | Stratified: {config.USE_STRATIFIED_SAMPLING}")
        print("="*60 + "\n")

    all_step_timings = new_step_timings()
    
    # Add initial timings (loading, test_sampling) if provided
    if initial_timings:
        if 'loading' in initial_timings:
            all_step_timings['loading'].append(initial_timings['loading'])
        if 'test_sampling' in initial_timings:
            all_step_timings['test_sampling'].append(initial_timings['test_sampling'])
    
    collected_prompts, collected_metadata = ([], []) if collect_prompts else (None, None)
    
    # Load trainset for ICL examples (universal approach)
    # For personal_recent, we need full data (train+test) to access all historical samples
    # For cross-user strategies, we use trainset only to prevent future data leakage
    lab_df_for_icl = None
    full_lab_df_for_personal = None
    
    train_df_attr = f'{dataset.upper()}_TRAIN_DF'
    full_df_attr = f'{dataset.upper()}_FULL_LAB_DF'
    
    # For GLOBEM, use pre-aggregated features for prompt generation
    # (feat_df contains raw data for DTW)
    aggregated_feat_df = feat_df  # Default: same as feat_df
    if dataset == 'globem' and hasattr(config, 'GLOBEM_AGGREGATED_FEAT_DF'):
        aggregated_feat_df = config.GLOBEM_AGGREGATED_FEAT_DF
    
    if hasattr(config, train_df_attr):
        lab_df_for_icl = getattr(config, train_df_attr)
        if verbose:
            print(f"  [Using {dataset.upper()} train set for cross-user ICL: {len(lab_df_for_icl)} samples]")
    
    if strategy in ['personal_recent', 'hybrid'] and hasattr(config, full_df_attr):
        full_lab_df_for_personal = getattr(config, full_df_attr)
        if verbose:
            print(f"  [Using {dataset.upper()} full data for personal ICL: {len(full_lab_df_for_personal)} samples (train+test)]")
    
    # Build retrieval candidate pool ONCE for entire evaluation (if using retrieval)
    retrieval_candidates = None
    if strategy in ['cross_retrieval', 'hybrid'] and (strategy == 'cross_retrieval' or use_dtw):
        if verbose:
            print(f"\n[Retrieval Setup] Building candidate pool for {strategy}...")
        
        # Use trainset (ICL pool)
        trainset = lab_df_for_icl if lab_df_for_icl is not None else lab_df
        
        retrieval_candidates = build_retrieval_candidate_pool(
            feat_df, trainset, cols,
            max_pool_size=500,
            random_state=random_state,
            dataset=dataset
        )
        
        if verbose and retrieval_candidates:
            print(f"  [OK] Candidate pool ready: {len(retrieval_candidates)} candidates\n")
    
    # Sample input data
    if verbose:
        print("[Data] Sampling input data...")
    
    # If use_all_samples is True, use all testset samples (414)
    if use_all_samples:
        if verbose:
            print(f"  [Using {len(lab_df)} pre-selected testset samples]")
        input_samples = []
        for idx, row in tqdm(lab_df.iterrows(), total=len(lab_df), desc="Preparing samples", disable=not verbose):
            user_id = row[cols['user_id']]
            ema_date = row[cols['date']]
            
            # Get aggregated features from pre-aggregated file (all datasets now use this)
            feat_row = aggregated_feat_df[
                (aggregated_feat_df[cols['user_id']] == user_id) & 
                (aggregated_feat_df[cols['date']] == ema_date)
            ]
            if len(feat_row) > 0:
                agg_feats = feat_row.iloc[0].to_dict()
            else:
                # No data for this sample
                agg_feats = None
            
            # For pre-selected testset: include ALL samples even if agg_feats is None
            if agg_feats is None:
                # Create minimal placeholder for samples with no historical data
                agg_feats = {
                    'user_id': user_id,
                    'ema_date': ema_date,
                    'aggregation_mode': 'raw',
                    'window_days': 0,
                    'features': {}
                }
            
            labels = row[cols['labels']].to_dict()
            input_samples.append({
                'aggregated_features': agg_feats,
                'labels': labels,
                'user_id': user_id,
                'ema_date': ema_date
            })
    else:
        # Original sampling logic
        use_stratified = config.USE_STRATIFIED_SAMPLING
        if use_stratified:
            try:
                input_samples = sample_batch_stratified(
                    feat_df, lab_df, cols, n_samples,
                    random_state=random_state, dataset=dataset
                )
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Stratified sampling failed: {e}\n   Falling back to random sampling...")
                use_stratified = False
        
        if not use_stratified:
            # Non-stratified random sampling
            input_samples = []
            rng = np.random.RandomState(random_state) if random_state else np.random.RandomState()
            max_attempts = n_samples * 20  # Try up to 20x the requested samples
            attempts = 0
            
            while len(input_samples) < n_samples and attempts < max_attempts:
                # Randomly sample from label dataframe
                idx = rng.choice(len(lab_df))
                row = lab_df.iloc[idx]
                user_id = row[cols['user_id']]
                ema_date = row[cols['date']]
                
                # Get aggregated features from pre-aggregated file
                feat_row = feat_df[
                    (feat_df[cols['user_id']] == user_id) & 
                    (feat_df[cols['date']] == ema_date)
                ]
                if len(feat_row) > 0:
                    agg_feats = feat_row.iloc[0].to_dict()
                else:
                    agg_feats = None
                
                if agg_feats is not None and check_missing_ratio(agg_feats):
                    labels = row[cols['labels']].to_dict()
                    input_samples.append({
                        'aggregated_features': agg_feats,
                        'labels': labels,
                        'user_id': user_id,
                        'ema_date': ema_date
                    })
                
                attempts += 1
            
            if len(input_samples) < n_samples * 0.75:
                if verbose:
                    print(f"[WARNING]  Warning: Only collected {len(input_samples)}/{n_samples} samples")
            
            if len(input_samples) == 0:
                if verbose:
                    print("[ERROR] Could not collect any valid samples")
                return None

    if verbose:
        print(f"[OK] Collected {len(input_samples)} valid samples\n")

    # Loop
    all_predictions, failed_count = [], 0
    for i, input_sample in enumerate(tqdm(input_samples, desc="Generating prompts", disable=not verbose)):

        # ICL - Select appropriate data source based on strategy
        # Personal strategies: use full data (train+test) with date filtering
        # Cross-user strategies: use trainset only to prevent future leakage
        if strategy in ['personal_recent', 'hybrid']:
            icl_lab_df = full_lab_df_for_personal if full_lab_df_for_personal is not None else (lab_df_for_icl if lab_df_for_icl is not None else lab_df)
        else:
            icl_lab_df = lab_df_for_icl if lab_df_for_icl is not None else lab_df
        
        icl_examples, icl_strategy = select_icl(
            feat_df, icl_lab_df, cols, input_sample, n_shot, strategy, use_dtw,
            (random_state + i * 1000) if random_state else None,
            all_step_timings, verbose, retrieval_candidates, dataset,
            lab_df_for_icl, full_lab_df_for_personal
        )

        # Prompt
        prompt = build_prompt_with_timing(
            prompt_manager, input_sample, cols, icl_examples, icl_strategy,
            reasoning_method, all_step_timings, verbose, feat_df=feat_df
        )

        if collect_prompts:
            collected_prompts.append(prompt)
            
            # Filter out before#day columns from aggregated_features for CES data
            agg_feats = input_sample.get('aggregated_features', {})
            if isinstance(agg_feats, dict) and dataset == 'ces':
                # Exclude columns ending with before#day (e.g., loc_dist_ep_0_before1day)
                agg_feats_filtered = {
                    k: v for k, v in agg_feats.items()
                    if not (isinstance(k, str) and 'before' in k and k.split('_')[-1].startswith('before'))
                }
            else:
                agg_feats_filtered = agg_feats
            
            # Get labels dynamically based on dataset
            labels = input_sample['labels']
            anxiety_key = cols['labels'][0] if len(cols['labels']) > 0 else list(labels.keys())[0]
            depression_key = cols['labels'][1] if len(cols['labels']) > 1 else list(labels.keys())[1]
            stress_key = cols['labels'][2] if len(cols['labels']) > 2 else 'stress'
            
            true_anxiety = labels[anxiety_key]
            true_depression = labels[depression_key]
            true_stress = labels.get(stress_key, None)
            
            metadata_entry = {
                'user_id': input_sample['user_id'],
                'ema_date': str(input_sample['ema_date']),
                'true_anxiety': true_anxiety,
                'true_depression': true_depression,
            }
            
            # Add stress if available (before features to maintain label grouping)
            if true_stress is not None:
                metadata_entry['true_stress'] = true_stress
            
            # Add part of day information for MentalIoT
            if dataset == 'mentaliot':
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(str(input_sample['ema_date']).replace(' ', 'T'))
                    hour = dt.hour
                    # Categorize into 6 time periods matching feature windows
                    if 0 <= hour < 4:
                        part_of_day = "0-4h (night)"
                    elif 4 <= hour < 8:
                        part_of_day = "4-8h (early morning)"
                    elif 8 <= hour < 12:
                        part_of_day = "8-12h (late morning)"
                    elif 12 <= hour < 16:
                        part_of_day = "12-16h (afternoon)"
                    elif 16 <= hour < 20:
                        part_of_day = "16-20h (evening)"
                    else:
                        part_of_day = "20-24h (night)"
                    metadata_entry['part_of_day'] = part_of_day
                except:
                    pass
            
            # Add features last
            metadata_entry['aggregated_features'] = agg_feats_filtered
            
            collected_metadata.append(metadata_entry)

        # Predict (use dynamically extracted labels)
        all_predictions, all_step_timings, failed_count, last_pred = predict(
            all_predictions, all_step_timings,
            input_sample['user_id'], input_sample['ema_date'],
            true_anxiety, true_depression,
            failed_count, reasoner, reasoning_method, prompt,
            true_stress=true_stress,
            llm_seed=llm_seed, sc_samples=5, verbose=verbose,
        )

        if last_pred is not None and verbose:
            print_batch_progress(i, len(input_samples), input_sample, last_pred['prediction'], input_sample['labels'], verbose)

    if verbose:
        print(f"\n{'=' * 60}\n[DONE] Completed: {len(all_predictions)}/{n_samples} | Failed: {failed_count}\n{'=' * 60}\n")

    if not all_predictions:
        print("[ERROR] No successful predictions")
        return None

    usage = reasoner.get_usage_summary()
    eval_config = {'n_samples': n_samples, 'n_shot': n_shot, 'strategy': strategy, 'reasoning_method': reasoning_method}
    avg_timings = {step: float(np.mean(times)) if times else 0.0 for step, times in all_step_timings.items()}

    print_batch_timing_summary(all_step_timings, verbose)

    results_for_report = records_to_report_items(all_predictions)
    # Pass predictions to generate_comprehensive_report - for self_feedback, usage will be extracted from predictions
    report = generate_comprehensive_report(results_for_report, usage, eval_config, predictions=all_predictions)
    report['step_timings_avg'] = avg_timings
    report['step_timings_all'] = {k: [float(x) for x in v] for k, v in all_step_timings.items()}
    report['predictions'] = all_predictions  # Store predictions for CSV export

    if collect_prompts:
        report['prompts'], report['metadata'] = collected_prompts, collected_metadata

    print_comprehensive_report(report)
    return report


def run_batch_prompts_only(prompt_manager: PromptManager, feat_df, lab_df, cols: Dict, 
                           n_samples: int = 30, n_shot: int = 5, strategy: str = 'cross_random',
                           use_dtw: bool = False, reasoning_method: str = 'cot',
                           random_state: Optional[int] = 42, 
                           verbose: bool = True,
                           initial_timings: Optional[Dict[str, float]] = None,
                           dataset: str = 'globem') -> Dict:
    """
    Generate and save prompts only without calling LLM.
    Returns prompts and metadata for saving to disk.
    """
    # Determine actual number of samples to use
    use_all_samples = len(lab_df) <= n_samples * 2
    actual_n_samples = len(lab_df) if use_all_samples else n_samples
    
    if verbose:
        print("\n" + "="*60 + f"\n[PROMPT GENERATION ONLY] ({actual_n_samples} samples)" + "\n" + "="*60)
        if use_all_samples:
            print(f"  [Using all pre-selected samples from testset]")
        print(f"  ICL Strategy: {strategy} | N-Shot: {n_shot} | Reasoning: {reasoning_method}")
        if random_state:
            print(f"  Seed: {random_state}")
        print(f"  Config: {config.AGGREGATION_WINDOW_DAYS}d window | {config.DEFAULT_AGGREGATION_MODE} mode | Stratified: {config.USE_STRATIFIED_SAMPLING}")
        print("="*60 + "\n")

    all_step_timings = new_step_timings()
    
    # Add initial timings (loading, test_sampling) if provided
    if initial_timings:
        if 'loading' in initial_timings:
            all_step_timings['loading'].append(initial_timings['loading'])
        if 'test_sampling' in initial_timings:
            all_step_timings['test_sampling'].append(initial_timings['test_sampling'])
    
    collected_prompts, collected_metadata = [], []
    
    # Load trainset for ICL examples (universal approach)
    # For personal_recent, we need full data (train+test) to access all historical samples
    # For cross-user strategies, we use trainset only to prevent future data leakage
    lab_df_for_icl = None
    full_lab_df_for_personal = None
    
    train_df_attr = f'{dataset.upper()}_TRAIN_DF'
    full_df_attr = f'{dataset.upper()}_FULL_LAB_DF'
    
    # For GLOBEM, use pre-aggregated features for prompt generation
    # (feat_df contains raw data for DTW)
    aggregated_feat_df = feat_df  # Default: same as feat_df
    if dataset == 'globem' and hasattr(config, 'GLOBEM_AGGREGATED_FEAT_DF'):
        aggregated_feat_df = config.GLOBEM_AGGREGATED_FEAT_DF
    
    if hasattr(config, train_df_attr):
        lab_df_for_icl = getattr(config, train_df_attr)
        if verbose:
            print(f"  [Using {dataset.upper()} train set for cross-user ICL: {len(lab_df_for_icl)} samples]")
    
    if strategy in ['personal_recent', 'hybrid'] and hasattr(config, full_df_attr):
        full_lab_df_for_personal = getattr(config, full_df_attr)
        if verbose:
            print(f"  [Using {dataset.upper()} full data for personal ICL: {len(full_lab_df_for_personal)} samples (train+test)]")
    
    # Build retrieval candidate pool ONCE for entire evaluation (if using retrieval)
    retrieval_candidates = None
    if strategy in ['cross_retrieval', 'hybrid'] and (strategy == 'cross_retrieval' or use_dtw):
        if verbose:
            print(f"\n[Retrieval Setup] Building candidate pool for {strategy}...")
        
        # Use trainset (ICL pool)
        trainset = lab_df_for_icl if lab_df_for_icl is not None else lab_df
        
        retrieval_candidates = build_retrieval_candidate_pool(
            feat_df, trainset, cols,
            max_pool_size=300,
            random_state=random_state,
            dataset=dataset
        )
        
        if verbose and retrieval_candidates:
            print(f"  [OK] Candidate pool ready: {len(retrieval_candidates)} candidates\n")
    
    # Sample input data
    if verbose:
        print("[Data] Sampling input data...")
    
    # If use_all_samples is True, use all testset samples (414)
    if use_all_samples:
        if verbose:
            print(f"  [Using {len(lab_df)} pre-selected testset samples]")
        input_samples = []
        for idx, row in tqdm(lab_df.iterrows(), total=len(lab_df), desc="Preparing samples", disable=not verbose):
            user_id = row[cols['user_id']]
            ema_date = row[cols['date']]
            
            # Get aggregated features from pre-aggregated file (all datasets now use this)
            feat_row = aggregated_feat_df[
                (aggregated_feat_df[cols['user_id']] == user_id) & 
                (aggregated_feat_df[cols['date']] == ema_date)
            ]
            if len(feat_row) > 0:
                agg_feats = feat_row.iloc[0].to_dict()
            else:
                # No data for this sample
                agg_feats = None
            
            # For pre-selected testset: include ALL samples even if agg_feats is None
            if agg_feats is None:
                # Create minimal placeholder for samples with no historical data
                agg_feats = {
                    'user_id': user_id,
                    'ema_date': ema_date,
                    'aggregation_mode': 'raw',
                    'window_days': 0,
                    'features': {}
                }
            
            labels = row[cols['labels']].to_dict()
            input_samples.append({
                'aggregated_features': agg_feats,
                'labels': labels,
                'user_id': user_id,
                'ema_date': ema_date
            })
    else:
        # Original sampling logic
        use_stratified = config.USE_STRATIFIED_SAMPLING
        if use_stratified:
            try:
                input_samples = sample_batch_stratified(
                    feat_df, lab_df, cols, n_samples,
                    random_state=random_state, dataset=dataset
                )
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Stratified sampling failed: {e}\n   Falling back to random sampling...")
                use_stratified = False
        
        if not use_stratified:
            # Non-stratified random sampling
            input_samples = []
            rng = np.random.RandomState(random_state) if random_state else np.random.RandomState()
            max_attempts = n_samples * 20
            attempts = 0
            
            while len(input_samples) < n_samples and attempts < max_attempts:
                idx = rng.choice(len(lab_df))
                row = lab_df.iloc[idx]
                user_id = row[cols['user_id']]
                ema_date = row[cols['date']]
                
                # Get aggregated features from pre-aggregated file (all datasets now use this)
                feat_row = feat_df[
                    (feat_df[cols['user_id']] == user_id) & 
                    (feat_df[cols['date']] == ema_date)
                ]
                if len(feat_row) == 0:
                    attempts += 1
                    continue
                agg_feats = feat_row.iloc[0].to_dict()
                
                # Check missing ratio
                if not check_missing_ratio(agg_feats):
                    attempts += 1
                    continue
                
                labels = row[cols['labels']].to_dict()
                input_samples.append({
                    'aggregated_features': agg_feats,
                    'labels': labels,
                    'user_id': user_id,
                    'ema_date': ema_date
                })
                
                attempts += 1
            
            if len(input_samples) < n_samples * 0.75:
                if verbose:
                    print(f"[WARNING]  Warning: Only collected {len(input_samples)}/{n_samples} samples")
            
            if len(input_samples) == 0:
                if verbose:
                    print("[ERROR] Could not collect any valid samples")
                return None

    if verbose:
        print(f"[OK] Collected {len(input_samples)} valid samples\n")
        print("[Prompt] Generating prompts...")

    # Generate prompts for each sample
    for i, input_sample in enumerate(tqdm(input_samples, desc="Generating prompts", disable=not verbose)):

        # ICL - Select appropriate data source based on strategy
        # Personal strategies: use full data (train+test) with date filtering
        # Cross-user strategies: use trainset only to prevent future leakage
        if strategy in ['personal_recent', 'hybrid']:
            icl_lab_df = full_lab_df_for_personal if full_lab_df_for_personal is not None else (lab_df_for_icl if lab_df_for_icl is not None else lab_df)
        else:
            icl_lab_df = lab_df_for_icl if lab_df_for_icl is not None else lab_df
        
        icl_examples, icl_strategy = select_icl(
            feat_df, icl_lab_df, cols, input_sample, n_shot, strategy, use_dtw,
            (random_state + i * 1000) if random_state else None,
            all_step_timings, False, retrieval_candidates, dataset,  # verbose=False, pass retrieval pool
            lab_df_for_icl, full_lab_df_for_personal
        )

        # Prompt
        prompt = build_prompt_with_timing(
            prompt_manager, input_sample, cols, icl_examples, icl_strategy,
            reasoning_method, all_step_timings, False, feat_df=feat_df  # verbose=False
        )

        collected_prompts.append(prompt)
        
        # Filter out before#day columns from aggregated_features for CES data
        agg_feats = input_sample.get('aggregated_features', {})
        if isinstance(agg_feats, dict) and dataset == 'ces':
            # Exclude columns ending with before#day (e.g., loc_dist_ep_0_before1day)
            agg_feats_filtered = {
                k: v for k, v in agg_feats.items()
                if not (isinstance(k, str) and 'before' in k and k.split('_')[-1].startswith('before'))
            }
        else:
            agg_feats_filtered = agg_feats
        
        # Get labels dynamically based on dataset
        labels = input_sample['labels']
        anxiety_key = cols['labels'][0] if len(cols['labels']) > 0 else list(labels.keys())[0]
        depression_key = cols['labels'][1] if len(cols['labels']) > 1 else list(labels.keys())[1]
        stress_key = cols['labels'][2] if len(cols['labels']) > 2 else 'stress'
        
        true_anxiety = labels[anxiety_key]
        true_depression = labels[depression_key]
        true_stress = labels.get(stress_key, None)
        
        metadata_entry = {
            'user_id': input_sample['user_id'],
            'ema_date': str(input_sample['ema_date']),
            'true_anxiety': true_anxiety,
            'true_depression': true_depression,
        }
        
        # Add stress if available (before features to maintain label grouping)
        if true_stress is not None:
            metadata_entry['true_stress'] = true_stress
        
        # Add part of day information for MentalIoT
        if dataset == 'mentaliot':
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(str(input_sample['ema_date']).replace(' ', 'T'))
                hour = dt.hour
                # Categorize into 6 time periods matching feature windows
                if 0 <= hour < 4:
                    part_of_day = "0-4h (night)"
                elif 4 <= hour < 8:
                    part_of_day = "4-8h (early morning)"
                elif 8 <= hour < 12:
                    part_of_day = "8-12h (late morning)"
                elif 12 <= hour < 16:
                    part_of_day = "12-16h (afternoon)"
                elif 16 <= hour < 20:
                    part_of_day = "16-20h (evening)"
                else:
                    part_of_day = "20-24h (night)"
                metadata_entry['part_of_day'] = part_of_day
            except:
                pass
        
        # Add features last
        metadata_entry['aggregated_features'] = agg_feats_filtered
        
        collected_metadata.append(metadata_entry)

    if verbose:
        print(f"\n[OK] Generated {len(collected_prompts)} prompts successfully\n")
        print_batch_timing_summary(all_step_timings, verbose=True)

    return {
        'prompts': collected_prompts,
        'metadata': collected_metadata,
        'n_samples': len(collected_prompts),
        'step_timings': all_step_timings
    }


def run_batch_with_loaded_prompts(reasoner: LLMReasoner, prompts: List[str], metadata: List[Dict],
                                  reasoning_method: str = 'cot', llm_seed: Optional[int] = None,
                                  checkpoint_path: Optional[str] = None, checkpoint_every: int = 10,
                                  resume_from: Optional[str] = None, verbose: bool = True) -> Optional[Dict]:
    """
    Run batch evaluation with pre-loaded prompts (for model comparison).
    
    Args:
        checkpoint_path: Base path for checkpoint files (will append _N.json)
        checkpoint_every: Save checkpoint every N samples (0 to disable)
        resume_from: Path to checkpoint file to resume from
    """
    if verbose:
        print("\n" + "=" * 60 + "\n[RESUME] BATCH WITH LOADED PROMPTS" + "\n" + "=" * 60)
        print(f"Samples: {len(prompts)} | Reasoning: {reasoning_method} | Model: {reasoner.model}")
        if llm_seed:
            print(f"LLM Seed: {llm_seed}")
        if checkpoint_every > 0:
            print(f"[Checkpoint] Every {checkpoint_every} samples")
        if resume_from:
            print(f"[RESUME] Resuming from: {resume_from}")
        print("=" * 60 + "\n")

    # Initialize tracking variables
    all_predictions, failed_count = [], 0
    all_step_timings = new_step_timings()
    start_idx = 0

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        try:
            import json
            with open(resume_from, 'r') as f:
                checkpoint = json.load(f)
            
            all_predictions = checkpoint.get('predictions', [])
            failed_count = checkpoint.get('failed_count', 0)
            all_step_timings = checkpoint.get('step_timings', new_step_timings())
            start_idx = checkpoint.get('last_index', -1) + 1
            
            # Restore usage stats (cost, GPU metrics, etc.)
            if 'usage_stats' in checkpoint:
                reasoner.client.usage_stats = checkpoint['usage_stats']
                if verbose:
                    print(f"[OK] Restored usage stats: {checkpoint['usage_stats']['num_requests']} requests, "
                          f"${checkpoint['usage_stats']['total_cost']:.4f} spent")
                    if checkpoint['usage_stats'].get('peak_gpu_memory', 0) > 0:
                        print(f"     Peak GPU memory: {checkpoint['usage_stats']['peak_gpu_memory']:.1f} MB")
            
            print(f"[OK] Resumed from checkpoint: {len(all_predictions)} predictions loaded, starting from index {start_idx}")
        except Exception as e:
            print(f"[WARNING]  Failed to load checkpoint: {e}")
            print("   Starting from scratch...")
            start_idx = 0

    # Process samples
    for idx in range(start_idx, len(prompts)):
        prompt = prompts[idx]
        meta = metadata[idx]
        
        if verbose:
            print(f"\n[{idx+1}/{len(prompts)}] User: {meta['user_id']} | Date: {meta['ema_date']}")
        elif (idx + 1) % 10 == 0 or idx == len(prompts) - 1:
            print(f"Progress: {idx+1}/{len(prompts)} samples completed", end='\r')

        all_predictions, all_step_timings, failed_count, _ = predict(
            all_predictions, all_step_timings,
            meta['user_id'], meta['ema_date'],
            meta['true_anxiety'], meta['true_depression'],
            failed_count, reasoner, reasoning_method, prompt,
            true_stress=meta.get('true_stress', None),
            llm_seed=llm_seed, sc_samples=5, verbose=verbose,
        )
        
        # Save checkpoint if enabled
        if checkpoint_every > 0 and checkpoint_path and (idx + 1) % checkpoint_every == 0:
            # Create chk directory
            checkpoint_dir = os.path.join(os.path.dirname(checkpoint_path) or '.', 'chk')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Build checkpoint filename
            base_name = os.path.basename(checkpoint_path)
            checkpoint_file = os.path.join(checkpoint_dir, f"{base_name}_checkpoint_{idx+1}.json")
            previous_checkpoint = os.path.join(checkpoint_dir, f"{base_name}_checkpoint_{idx+1-checkpoint_every}.json")
            
            try:
                import json
                from src.prompt_utils import NumpyEncoder
                
                # Get current usage stats (includes cost, GPU metrics, etc.)
                current_usage_stats = reasoner.client.usage_stats
                
                checkpoint_data = {
                    'last_index': idx,
                    'predictions': all_predictions,
                    'failed_count': failed_count,
                    'step_timings': all_step_timings,
                    'usage_stats': current_usage_stats,  # Save cost & GPU metrics
                    'total_samples': len(prompts),
                    'completed_samples': idx + 1
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, cls=NumpyEncoder)
                if verbose:
                    cost_info = f"${current_usage_stats['total_cost']:.4f}"
                    gpu_info = ""
                    if current_usage_stats.get('peak_gpu_memory', 0) > 0:
                        gpu_info = f", GPU: {current_usage_stats['peak_gpu_memory']:.0f}MB"
                    print(f"  ✅ Checkpoint saved: {checkpoint_file} (Cost: {cost_info}{gpu_info})")
                
                # Delete previous checkpoint if it exists
                if os.path.exists(previous_checkpoint):
                    try:
                        os.remove(previous_checkpoint)
                        if verbose:
                            print(f"  🗑️  Deleted previous checkpoint: {previous_checkpoint}")
                    except Exception as e:
                        if verbose:
                            print(f"  [WARNING]  Failed to delete previous checkpoint: {e}")
            except Exception as e:
                print(f"  [WARNING]  Failed to save checkpoint: {e}")

    if not all_predictions:
        print("[ERROR] No successful predictions")
        return None

    if verbose:
        print(f"\n{'=' * 60}\n[Data] CALCULATING METRICS\n{'=' * 60}\n")

    results = []
    for p in all_predictions:
        result = {
            'labels': {
                'phq4_anxiety_EMA': p['true_anxiety'],
                'phq4_depression_EMA': p['true_depression'],
            },
            'prediction': {
                'Anxiety_binary': p['pred_anxiety'],
                'Depression_binary': p['pred_depression'],
            },
        }
        
        # Add stress if available (CES/MentalIoT dataset)
        if 'true_stress' in p:
            result['labels']['stress'] = p['true_stress']
            result['prediction']['Stress_binary'] = p.get('pred_stress', 0)
        
        results.append(result)

    # Pass predictions to generate_comprehensive_report - for self_feedback, usage will be extracted from predictions
    report = generate_comprehensive_report(results, reasoner.get_usage_summary(), predictions=all_predictions)
    report['predictions'] = all_predictions  # Store predictions for CSV export
    print_comprehensive_report(report)
    return report
