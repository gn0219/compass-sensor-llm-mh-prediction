"""
Evaluation Runner Module (refined)

- Removes duplication with small helpers (timeit, ICL helpers, prompt builder wrapper).
- Keeps the original flow & outputs; fixes None-return issues and positional-arg bugs.
"""

import os
import time
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from . import config
from .sensor_transformation import sample_input_data, sample_batch_stratified, aggregate_window_features, check_missing_ratio
from .example_selection import select_icl_examples
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

def new_step_timings(keys=('data_sampling', 'icl_selection', 'prompt_building', 'llm_call', 'response_parsing')):
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
    n_shot: int, source: str, selection: str, random_state: Optional[int],
    step_timings: Dict[str, List[float]], verbose: bool, beta: float = 0.0
):
    """Select ICL examples if needed, append timing, and return (icl_examples, icl_strategy)."""
    icl_examples = None
    icl_strategy = 'zero_shot'

    if n_shot > 0:
        if verbose:
            msg = f"\n📚 Selecting {n_shot} ICL examples (source: {source}, selection: {selection}"
            if selection == 'diversity' and beta > 0:
                msg += f", β={beta}"
            msg += ")..."
            print(msg)
        with timeit(step_timings, 'icl_selection'):
            icl_examples = select_icl_examples(
                feat_df, lab_df, cols,
                input_sample['user_id'], input_sample['ema_date'],
                n_shot=n_shot, source=source, selection=selection,
                random_state=random_state, target_sample=input_sample, beta=beta
            )
        if icl_examples is None:
            if verbose:
                print("  ⚠️  Failed to select ICL examples, falling back to zero-shot")
            icl_strategy = 'zero_shot'
        else:
            print_icl_selection_info(
                n_shot, source, icl_examples, step_timings['icl_selection'][-1], verbose
            )
            icl_strategy = source # personalized, generalized, hybrid
    else:
        if verbose:
            print(f"\n📚 Using zero-shot (no ICL examples)")
        append_zero(step_timings, 'icl_selection')

    return icl_examples, icl_strategy


def build_prompt_with_timing(
    prompt_manager: PromptManager, input_sample: Dict, cols: Dict,
    icl_examples, icl_strategy: str, reasoning_method: str,
    step_timings: Dict[str, List[float]], verbose: bool
) -> str:
    if verbose:
        print(f"\n🤖 Building prompt (reasoning: {reasoning_method})...")
    with timeit(step_timings, 'prompt_building'):
        prompt = build_prompt(prompt_manager, input_sample, cols, icl_examples, icl_strategy, reasoning_method)
    print_prompt_building_info(len(prompt), step_timings['prompt_building'][-1], verbose)
    return prompt


# ---------------------------
# Predict 
# ---------------------------

def predict(all_predictions: List[Dict], all_step_timings: Dict[str, List[float]],
            user_id, ema_date, true_anxiety, true_depression, failed_count: int,
            reasoner: LLMReasoner, reasoning_method: str, prompt: str, *,
            llm_seed: Optional[int] = None, sc_samples: Optional[int] = None,
            verbose: bool = True) -> Tuple[List[Dict], Dict[str, List[float]], int, Optional[Dict]]:
    """
    Prediction executor.
    Always returns (all_predictions, all_step_timings, failed_count, pred).
    """
    with timeit(all_step_timings, 'llm_call'):
        if reasoning_method == 'self_consistency':
            n_sc = sc_samples or 5
            prediction, _ = reasoner.predict_with_self_consistency(
                prompt, n_samples=n_sc, seed=llm_seed
            )
        else:
            response_text, usage_info = reasoner.call_llm(prompt, seed=llm_seed)

    if reasoning_method == 'self_consistency':       
        # # parsing is internal to self-consistency path
        append_zero(all_step_timings, 'response_parsing')
    else:
        if not response_text:
            if verbose:
                print("  ⚠️  LLM call failed, skipping")
            failed_count += 1
            append_zero(all_step_timings, 'response_parsing')
            return all_predictions, all_step_timings, failed_count, None

        with timeit(all_step_timings, 'response_parsing'):
            prediction = reasoner.parse_response(response_text)

    if not prediction:
        if verbose:
            print("  ⚠️  Parse failed, skipping")
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
    all_predictions.append(pred)

    if verbose:
        print(f"  User ID: {user_id}")
        print(f"  Date: {ema_date}")
        print(
            f"  ✓ Anx: {pred['pred_anxiety']} (true: {true_anxiety}) | "
            f"Dep: {pred['pred_depression']} (true: {true_depression})"
        )

    return all_predictions, all_step_timings, failed_count, pred

def records_to_report_items(records: List[Dict]) -> List[Dict]:
    """Convert record schema to generate_comprehensive_report's expected items."""
    return [
        {
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
        for r in records
    ]

def run_single_prediction(prompt_manager: PromptManager, reasoner: LLMReasoner,
                          feat_df, lab_df, cols: Dict, n_shot: int = 5, source: str = 'hybrid',
                          selection: str = 'random', reasoning_method: str = 'cot', 
                          random_state: Optional[int] = None, llm_seed: Optional[int] = None,
                          beta: float = 0.0, verbose: bool = True) -> Optional[Dict]:
    """Run a single prediction for demonstration."""
    if verbose:
        print(f"\n🔍 SINGLE SAMPLE PREDICTION")

    all_step_timings = new_step_timings()

    with timeit(all_step_timings, 'data_sampling'):
        input_sample = sample_input_data(feat_df, lab_df, cols, random_state)

    if input_sample is None:
        print("❌ Failed to sample valid input data")
        return None
    
    print_input_sample_info(input_sample, all_step_timings['data_sampling'][-1], verbose)

    # ICL
    icl_examples, icl_strategy = select_icl(
        feat_df, lab_df, cols, input_sample, n_shot, source, selection, 
        random_state, all_step_timings, verbose, beta
    )

    # Prompt
    prompt = build_prompt_with_timing(
        prompt_manager, input_sample, cols, icl_examples, icl_strategy, reasoning_method, all_step_timings, verbose
    )

    # Predict
    all_predictions, failed_count = [], 0
    all_predictions, all_step_timings, failed_count, last_pred = predict(
        all_predictions, all_step_timings,
        input_sample['user_id'], input_sample['ema_date'],
        input_sample['labels']['phq4_anxiety_EMA'], input_sample['labels']['phq4_depression_EMA'],
        failed_count, reasoner, reasoning_method, prompt,
        llm_seed=llm_seed, sc_samples=5, verbose=verbose,
    )

    if last_pred is None:
        print("❌ Prediction failed for the sampled input")
        return None

    # Output
    print_prediction_results(last_pred['prediction'], input_sample['labels'], verbose)
    print_timing_breakdown(all_step_timings, reasoning_method, verbose)

    # if verbose:
    #     print("=" * 60 + "\n")

    return {
        'experiment_config': {
            'n_shot': n_shot, 'source': source, 'selection': selection, 'icl_strategy': icl_strategy,
            'reasoning_method': reasoning_method, 'model': reasoner.model,
            'random_state': random_state, 'llm_seed': llm_seed
        },
        'prompt_used': prompt,
        'input_sample': input_sample,
        'prediction': last_pred['prediction'],   # pure prediction
        'record': last_pred,                     # full record if needed downstream
        'usage': reasoner.get_usage_summary(),
        'step_timings': all_step_timings,
        'total_time': sum(sum(v) for v in all_step_timings.values()),
    }


def run_batch_evaluation(prompt_manager: PromptManager, reasoner: LLMReasoner,
                         feat_df, lab_df, cols: Dict, n_samples: int = 30, n_shot: int = 5, 
                         source: str = 'hybrid', selection: str = 'random', reasoning_method: str = 'cot', 
                         random_state: Optional[int] = 42, llm_seed: Optional[int] = None, 
                         beta: float = 0.0, collect_prompts: bool = False, verbose: bool = True) -> Optional[Dict]:
    """Run batch evaluation on multiple samples."""
    
    if verbose:
        print("\n" + "="*60 + f"\n🔬 BATCH EVALUATION ({n_samples} samples)" + "\n" + "="*60)
        print(f"  ICL: {source} | Selection: {selection} | N-Shot: {n_shot} | Reasoning: {reasoning_method} | Model: {reasoner.model}")
        if random_state or llm_seed:
            print(f"   Seed: {random_state} | LLM Seed: {llm_seed}")
        print(f"  Config: {config.AGGREGATION_WINDOW_DAYS}d window | {config.DEFAULT_AGGREGATION_MODE} mode | Stratified: {config.USE_STRATIFIED_SAMPLING}")
        print("="*60 + "\n")

    all_step_timings = new_step_timings()
    collected_prompts, collected_metadata = ([], []) if collect_prompts else (None, None)
    
    # Sample input data
    if verbose:
        print("📊 Sampling input data...")
    
    with timeit(all_step_timings, 'data_sampling'):
        use_stratified = config.USE_STRATIFIED_SAMPLING
        if use_stratified:
            try:
                input_samples = sample_batch_stratified(
                    feat_df, lab_df, cols, n_samples,
                    random_state=random_state
                )
            except Exception as e:
                if verbose:
                    print(f"❌ Stratified sampling failed: {e}\n   Falling back to random sampling...")
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
                
                agg_feats = aggregate_window_features(
                    feat_df, user_id, ema_date, cols,
                    window_days=config.AGGREGATION_WINDOW_DAYS,
                    mode=config.DEFAULT_AGGREGATION_MODE,
                    use_immediate_window=config.USE_IMMEDIATE_WINDOW,
                    immediate_days=config.IMMEDIATE_WINDOW_DAYS,
                    adaptive_window=config.USE_ADAPTIVE_WINDOW
                )
                
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
                    print(f"⚠️  Warning: Only collected {len(input_samples)}/{n_samples} samples")
            
            if len(input_samples) == 0:
                if verbose:
                    print("❌ Could not collect any valid samples")
                return None

    total_sampling_time = all_step_timings['data_sampling'][-1] if all_step_timings['data_sampling'] else 0.0
    if verbose:
        print(f"✅ Collected {len(input_samples)} valid samples in {total_sampling_time:.2f}s\n")

    # Distribute average sampling time per sample (keeps shape identical to other step lists)
    per_sample_sampling = total_sampling_time / max(1, len(input_samples))
    all_step_timings['data_sampling'] = [per_sample_sampling for _ in input_samples]

    # Loop
    all_predictions, failed_count = [], 0
    for i, input_sample in enumerate(input_samples):

        # ICL
        icl_examples, icl_strategy = select_icl(
            feat_df, lab_df, cols, input_sample, n_shot, source, selection,
            (random_state + i * 1000) if random_state else None,
            all_step_timings, verbose, beta
        )

        # Prompt
        prompt = build_prompt_with_timing(
            prompt_manager, input_sample, cols, icl_examples, icl_strategy,
            reasoning_method, all_step_timings, verbose
        )

        if collect_prompts:
            collected_prompts.append(prompt)
            collected_metadata.append({
                'user_id': input_sample['user_id'],
                'ema_date': str(input_sample['ema_date']),
                'true_anxiety': input_sample['labels']['phq4_anxiety_EMA'],
                'true_depression': input_sample['labels']['phq4_depression_EMA'],
                'aggregated_features': input_sample.get('aggregated_features'),
            })

        # Predict
        all_predictions, all_step_timings, failed_count, last_pred = predict(
            all_predictions, all_step_timings,
            input_sample['user_id'], input_sample['ema_date'],
            input_sample['labels']['phq4_anxiety_EMA'], input_sample['labels']['phq4_depression_EMA'],
            failed_count, reasoner, reasoning_method, prompt,
            llm_seed=llm_seed, sc_samples=5, verbose=verbose,
        )

        if last_pred is not None and verbose:
            print_batch_progress(i, len(input_samples), input_sample, last_pred['prediction'], input_sample['labels'], verbose)

    if verbose:
        print(f"\n{'=' * 60}\n✅ Completed: {len(all_predictions)}/{n_samples} | ❌ Failed: {failed_count}\n{'=' * 60}\n")

    if not all_predictions:
        print("❌ No successful predictions")
        return None

    usage = reasoner.get_usage_summary()
    eval_config = {'n_samples': n_samples, 'n_shot': n_shot, 'source': source, 'selection': selection, 'reasoning_method': reasoning_method}
    avg_timings = {step: float(np.mean(times)) if times else 0.0 for step, times in all_step_timings.items()}

    print_batch_timing_summary(all_step_timings, verbose)

    results_for_report = records_to_report_items(all_predictions)
    report = generate_comprehensive_report(results_for_report, usage, eval_config)
    report['step_timings_avg'] = avg_timings
    report['step_timings_all'] = {k: [float(x) for x in v] for k, v in all_step_timings.items()}

    if collect_prompts:
        report['prompts'], report['metadata'] = collected_prompts, collected_metadata

    print_comprehensive_report(report)
    return report


def run_batch_prompts_only(prompt_manager: PromptManager, feat_df, lab_df, cols: Dict, 
                           n_samples: int = 30, n_shot: int = 5, source: str = 'hybrid', 
                           selection: str = 'random', reasoning_method: str = 'cot',
                           random_state: Optional[int] = 42, beta: float = 0.0, 
                           verbose: bool = True) -> Dict:
    """
    Generate and save prompts only without calling LLM.
    Returns prompts and metadata for saving to disk.
    """
    if verbose:
        print("\n" + "="*60 + f"\n💾 PROMPT GENERATION ONLY ({n_samples} samples)" + "\n" + "="*60)
        print(f"  ICL: {source} | Selection: {selection} | N-Shot: {n_shot} | Reasoning: {reasoning_method}")
        if random_state:
            print(f"  Seed: {random_state}")
        print(f"  Config: {config.AGGREGATION_WINDOW_DAYS}d window | {config.DEFAULT_AGGREGATION_MODE} mode | Stratified: {config.USE_STRATIFIED_SAMPLING}")
        print("="*60 + "\n")

    all_step_timings = new_step_timings()
    collected_prompts, collected_metadata = [], []
    
    # Sample input data
    if verbose:
        print("📊 Sampling input data...")
    
    with timeit(all_step_timings, 'data_sampling'):
        use_stratified = config.USE_STRATIFIED_SAMPLING
        if use_stratified:
            try:
                input_samples = sample_batch_stratified(
                    feat_df, lab_df, cols, n_samples,
                    random_state=random_state
                )
            except Exception as e:
                if verbose:
                    print(f"❌ Stratified sampling failed: {e}\n   Falling back to random sampling...")
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
                
                agg_feats = aggregate_window_features(
                    feat_df, user_id, ema_date, cols,
                    window_days=config.AGGREGATION_WINDOW_DAYS,
                    mode=config.DEFAULT_AGGREGATION_MODE,
                    use_immediate_window=config.USE_IMMEDIATE_WINDOW,
                    immediate_days=config.IMMEDIATE_WINDOW_DAYS,
                    adaptive_window=config.USE_ADAPTIVE_WINDOW
                )
                
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
                    print(f"⚠️  Warning: Only collected {len(input_samples)}/{n_samples} samples")
            
            if len(input_samples) == 0:
                if verbose:
                    print("❌ Could not collect any valid samples")
                return None

    total_sampling_time = all_step_timings['data_sampling'][-1] if all_step_timings['data_sampling'] else 0.0
    if verbose:
        print(f"✅ Collected {len(input_samples)} valid samples in {total_sampling_time:.2f}s\n")
        print("🎨 Generating prompts...")

    # Generate prompts for each sample
    for i, input_sample in enumerate(input_samples):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(input_samples)} prompts generated...")

        # ICL
        icl_examples, icl_strategy = select_icl(
            feat_df, lab_df, cols, input_sample, n_shot, source, selection,
            (random_state + i * 1000) if random_state else None,
            all_step_timings, False, beta  # verbose=False for cleaner output
        )

        # Prompt
        prompt = build_prompt_with_timing(
            prompt_manager, input_sample, cols, icl_examples, icl_strategy,
            reasoning_method, all_step_timings, False  # verbose=False
        )

        collected_prompts.append(prompt)
        collected_metadata.append({
            'user_id': input_sample['user_id'],
            'ema_date': str(input_sample['ema_date']),
            'true_anxiety': input_sample['labels']['phq4_anxiety_EMA'],
            'true_depression': input_sample['labels']['phq4_depression_EMA'],
            'aggregated_features': input_sample.get('aggregated_features'),
        })

    if verbose:
        print(f"\n✅ Generated {len(collected_prompts)} prompts successfully\n")

    return {
        'prompts': collected_prompts,
        'metadata': collected_metadata,
        'n_samples': len(collected_prompts)
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
        print("\n" + "=" * 60 + "\n🔄 BATCH WITH LOADED PROMPTS" + "\n" + "=" * 60)
        print(f"Samples: {len(prompts)} | Reasoning: {reasoning_method} | Model: {reasoner.model}")
        if llm_seed:
            print(f"LLM Seed: {llm_seed}")
        if checkpoint_every > 0:
            print(f"💾 Checkpoint: every {checkpoint_every} samples")
        if resume_from:
            print(f"🔄 Resuming from: {resume_from}")
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
            
            print(f"✅ Resumed from checkpoint: {len(all_predictions)} predictions loaded, starting from index {start_idx}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
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
            llm_seed=llm_seed, sc_samples=5, verbose=verbose,
        )
        
        # Save checkpoint if enabled
        if checkpoint_every > 0 and checkpoint_path and (idx + 1) % checkpoint_every == 0:
            checkpoint_file = f"{checkpoint_path}_checkpoint_{idx+1}.json"
            try:
                import json
                from src.prompt_utils import NumpyEncoder
                checkpoint_data = {
                    'last_index': idx,
                    'predictions': all_predictions,
                    'failed_count': failed_count,
                    'step_timings': all_step_timings,
                    'total_samples': len(prompts),
                    'completed_samples': idx + 1
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, cls=NumpyEncoder)
                if verbose:
                    print(f"  💾 Checkpoint saved: {checkpoint_file}")
            except Exception as e:
                print(f"  ⚠️  Failed to save checkpoint: {e}")

    if not all_predictions:
        print("❌ No successful predictions")
        return None

    if verbose:
        print(f"\n{'=' * 60}\n📊 CALCULATING METRICS\n{'=' * 60}\n")

    results = [
        {
            'labels': {
                'phq4_anxiety_EMA': p['true_anxiety'],
                'phq4_depression_EMA': p['true_depression'],
            },
            'prediction': {
                'Anxiety_binary': p['pred_anxiety'],
                'Depression_binary': p['pred_depression'],
            },
        }
        for p in all_predictions
    ]

    report = generate_comprehensive_report(results, reasoner.get_usage_summary())
    print_comprehensive_report(report)
    return report
