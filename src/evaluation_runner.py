"""
Evaluation Runner Module

Core evaluation logic for mental health prediction.
Orchestrates evaluation flow: data sampling → ICL selection → prompt building → LLM calls → metrics.
"""

import time
import numpy as np
from typing import Dict, List, Optional

from .sensor_transformation import sample_input_data, sample_batch_stratified
from .example_selection import select_icl_examples
from .prompt_utils import build_prompt
from .reasoning import LLMReasoner
from .performance import calculate_classification_metrics, calculate_efficiency_metrics, generate_comprehensive_report, print_comprehensive_report
from .output_utils import *
from .prompt_manager import PromptManager


def run_single_prediction(prompt_manager: PromptManager, reasoner: LLMReasoner, feat_df, lab_df, cols: Dict,
                         n_shot: int = 5, source: str = 'hybrid', reasoning_method: str = 'cot',
                         random_state: Optional[int] = None, llm_seed: Optional[int] = None,
                         verbose: bool = True) -> Dict:
    """Run a single prediction for demonstration."""
    print_section_header("🔍 SINGLE SAMPLE PREDICTION", verbose)
    step_timings = {}
    
    # Sample input
    print_subsection("📋 Sampling input data...", verbose)
    step_start = time.time()
    input_sample = sample_input_data(feat_df, lab_df, cols, random_state)
    step_timings['data_sampling'] = time.time() - step_start
    
    if input_sample is None:
        print("❌ Failed to sample valid input data")
        return None
    
    print_input_sample_info(input_sample, step_timings['data_sampling'], verbose)
    
    # Select ICL examples
    icl_examples = None
    icl_strategy = 'zero_shot'
    
    if n_shot > 0:
        print_subsection(f"📚 Selecting {n_shot} ICL examples (source: {source})...", verbose)
        step_start = time.time()
        icl_examples = select_icl_examples(feat_df, lab_df, cols, input_sample['user_id'], 
                                           input_sample['ema_date'], n_shot=n_shot, source=source,
                                           random_state=random_state)
        step_timings['icl_selection'] = time.time() - step_start
        
        if icl_examples is None:
            if verbose:
                print("  ⚠️  Failed to select ICL examples, falling back to zero-shot")
            icl_strategy = 'zero_shot'
        else:
            print_icl_selection_info(n_shot, source, icl_examples, step_timings['icl_selection'], verbose)
            icl_strategy = {'personalization': 'personalized', 'generalization': 'generalized',
                           'hybrid': 'hybrid'}.get(source, 'generalized')
    else:
        print_subsection("📚 Using zero-shot (no ICL examples)", verbose)
        step_timings['icl_selection'] = 0.0
    
    # Build prompt
    print_subsection(f"🤖 Building prompt (reasoning: {reasoning_method})...", verbose)
    step_start = time.time()
    prompt = build_prompt(prompt_manager, input_sample, cols, icl_examples, icl_strategy, reasoning_method)
    step_timings['prompt_building'] = time.time() - step_start
    print_prompt_building_info(len(prompt), step_timings['prompt_building'], verbose)
    
    # Call LLM
    print_subsection(f"🌐 Calling LLM ({reasoner.model})...", verbose)
    step_start = time.time()
    
    if reasoning_method == 'self_consistency':
        prediction, samples = reasoner.predict_with_self_consistency(prompt, n_samples=5, seed=llm_seed)
        step_timings['llm_call'] = time.time() - step_start
        step_timings['response_parsing'] = 0.0  # Parsing included in llm_call for self-consistency
        if verbose:
            print(f"  Generated {len(samples)} samples for self-consistency")
            print(f"  ⏱️  Time (includes 5x API + parsing + voting): {step_timings['llm_call']:.3f}s")
    else:
        response_text, usage_info = reasoner.call_llm(prompt, seed=llm_seed)
        step_timings['llm_call'] = time.time() - step_start
        
        if response_text is None:
            print("❌ LLM call failed")
            return None
        
        print_llm_call_info(usage_info, llm_seed, reasoning_method, verbose)
        if verbose:
            print(f"  ⏱️  Total Time: {step_timings['llm_call']:.3f}s")
        
        step_start = time.time()
        prediction = reasoner.parse_response(response_text)
        step_timings['response_parsing'] = time.time() - step_start
    
    if prediction is None:
        print("❌ Failed to parse prediction")
        return None
    
    # Display results
    print_prediction_results(prediction, input_sample['labels'], verbose)
    
    # Print detailed timing breakdown
    print_timing_breakdown(step_timings, reasoning_method, verbose)
    
    if verbose:
        print("="*60 + "\n")
    
    return {
        'experiment_config': {'n_shot': n_shot, 'source': source, 'icl_strategy': icl_strategy,
                             'reasoning_method': reasoning_method, 'model': reasoner.model,
                             'random_state': random_state, 'llm_seed': llm_seed},
        'prompt_used': prompt, 'input_sample': input_sample, 'prediction': prediction,
        'usage': reasoner.get_usage_summary(), 'step_timings': step_timings,
        'total_time': sum(step_timings.values())
    }


def run_batch_evaluation(prompt_manager: PromptManager, reasoner: LLMReasoner, feat_df, lab_df, cols: Dict,
                        n_samples: int = 30, n_shot: int = 5, source: str = 'hybrid',
                        reasoning_method: str = 'cot', random_state: Optional[int] = 42,
                        llm_seed: Optional[int] = None, use_stratified: bool = False,
                        stratify_by: str = 'phq4_anxiety_EMA', collect_prompts: bool = False,
                        verbose: bool = True) -> Dict:
    """Run batch evaluation on multiple samples."""
    print_section_header(f"🔬 BATCH EVALUATION ({n_samples} samples)", verbose)
    if verbose:
        print(f"  ICL: {source} | N-Shot: {n_shot} | Reasoning: {reasoning_method} | Seed: {random_state}")
        if llm_seed:
            print(f"  LLM Seed: {llm_seed}")
        print(f"  Stratified: {use_stratified}" + (f" (by {stratify_by})" if use_stratified else ""))
        print("="*60 + "\n")
    
    results, failed_count = [], 0
    all_step_timings = {k: [] for k in ['data_sampling', 'icl_selection', 'prompt_building', 'llm_call', 'response_parsing']}
    collected_prompts, collected_metadata = ([], []) if collect_prompts else (None, None)
    
    # Sample input data
    if verbose:
        print("📊 Sampling input data...")
    step_start = time.time()
    
    if use_stratified:
        try:
            input_samples = sample_batch_stratified(feat_df, lab_df, cols, n_samples, stratify_by, random_state)
        except Exception as e:
            if verbose:
                print(f"❌ Stratified sampling failed: {e}\n   Falling back to random sampling...")
            use_stratified, input_samples = False, []
    
    if not use_stratified:
        input_samples, attempts = [], 0
        while len(input_samples) < n_samples and attempts < n_samples * 3:
            sample = sample_input_data(feat_df, lab_df, cols, random_state + attempts if random_state else None)
            if sample:
                input_samples.append(sample)
            attempts += 1
    
    sampling_time = time.time() - step_start
    if verbose:
        print(f"✅ Collected {len(input_samples)} valid samples in {sampling_time:.2f}s\n")
    
    if not input_samples:
        print("❌ Failed to collect any valid samples")
        return None
    
    # Process each sample
    for i, input_sample in enumerate(input_samples):
        all_step_timings['data_sampling'].append(sampling_time / len(input_samples))
        
        # Select ICL examples
        icl_examples, icl_strategy = None, 'zero_shot'
        if n_shot > 0:
            step_start = time.time()
            icl_examples = select_icl_examples(feat_df, lab_df, cols, input_sample['user_id'],
                                              input_sample['ema_date'], n_shot=n_shot, source=source,
                                              random_state=random_state + i * 1000 if random_state else None)
            all_step_timings['icl_selection'].append(time.time() - step_start)
            if icl_examples:
                icl_strategy = {'personalization': 'personalized', 'generalization': 'generalized',
                               'hybrid': 'hybrid'}.get(source, 'generalized')
        else:
            all_step_timings['icl_selection'].append(0.0)
        
        # Build prompt
        step_start = time.time()
        prompt = build_prompt(prompt_manager, input_sample, cols, icl_examples, icl_strategy, reasoning_method)
        all_step_timings['prompt_building'].append(time.time() - step_start)
        
        if collect_prompts:
            collected_prompts.append(prompt)
            collected_metadata.append({
                'user_id': input_sample['user_id'], 'ema_date': str(input_sample['ema_date']),
                'true_anxiety': input_sample['labels']['phq4_anxiety_EMA'],
                'true_depression': input_sample['labels']['phq4_depression_EMA'],
                'aggregated_features': input_sample['aggregated_features']
            })
        
        # Call LLM
        step_start = time.time()
        if reasoning_method == 'self_consistency':
            prediction, _ = reasoner.predict_with_self_consistency(prompt, n_samples=5, seed=llm_seed)
            all_step_timings['llm_call'].append(time.time() - step_start)
            all_step_timings['response_parsing'].append(0.0)
        else:
            response_text, usage_info = reasoner.call_llm(prompt, seed=llm_seed)
            all_step_timings['llm_call'].append(time.time() - step_start)
            if not response_text:
                failed_count += 1
                continue
            
            step_start = time.time()
            prediction = reasoner.parse_response(response_text)
            all_step_timings['response_parsing'].append(time.time() - step_start)
        
        if not prediction:
            failed_count += 1
            continue
        
        results.append({
            'user_id': input_sample['user_id'], 'ema_date': str(input_sample['ema_date']),
            'labels': input_sample['labels'], 'prediction': prediction['Prediction']
        })
        
        print_batch_progress(i, len(input_samples), input_sample, prediction, input_sample['labels'], verbose)
    
    if verbose:
        print(f"\n{'='*60}\n✅ Completed: {len(results)}/{n_samples} | ❌ Failed: {failed_count}\n{'='*60}\n")
    
    if not results:
        print("❌ No successful predictions")
        return None
    
    usage = reasoner.get_usage_summary()
    config = {'n_samples': n_samples, 'n_shot': n_shot, 'source': source, 'reasoning_method': reasoning_method}
    avg_timings = {step: float(np.mean(times)) if times else 0.0 for step, times in all_step_timings.items()}
    
    print_batch_timing_summary(all_step_timings, verbose)
    
    report = generate_comprehensive_report(results, usage, config)
    report['step_timings_avg'] = avg_timings
    report['step_timings_all'] = {k: [float(x) for x in v] for k, v in all_step_timings.items()}
    
    if collect_prompts:
        report['prompts'], report['metadata'] = collected_prompts, collected_metadata
    
    print_comprehensive_report(report)
    return report


def run_batch_with_loaded_prompts(reasoner: LLMReasoner, prompts: List[str], metadata: List[Dict],
                                  reasoning_method: str = 'cot', llm_seed: Optional[int] = None,
                                  verbose: bool = True) -> Dict:
    """Run batch evaluation with pre-loaded prompts (for model comparison)."""
    print_section_header("🔄 BATCH WITH LOADED PROMPTS", verbose)
    if verbose:
        print(f"Samples: {len(prompts)} | Reasoning: {reasoning_method} | Model: {reasoner.model}")
        if llm_seed:
            print(f"LLM Seed: {llm_seed}")
        print("="*60 + "\n")
    
    all_predictions, start_time = [], time.time()
    
    for idx, (prompt, meta) in enumerate(zip(prompts, metadata)):
        if verbose:
            print(f"\n[{idx+1}/{len(prompts)}] User: {meta['user_id']} | Date: {meta['ema_date']}")
        elif (idx + 1) % 10 == 0 or idx == len(prompts) - 1:
            print(f"Progress: {idx+1}/{len(prompts)} samples completed", end='\r')
        
        # Call LLM with prompt
        if reasoning_method == 'self_consistency':
            prediction, _ = reasoner.predict_with_self_consistency(prompt, n_samples=5, seed=llm_seed)
        else:
            response_text, _ = reasoner.call_llm(prompt, seed=llm_seed)
            if not response_text:
                if verbose:
                    print("  ⚠️  LLM call failed, skipping")
                continue
            prediction = reasoner.parse_response(response_text)
        
        if not prediction:
            if verbose:
                print("  ⚠️  Parse failed, skipping")
            continue
        
        all_predictions.append({
            'user_id': meta['user_id'], 'ema_date': meta['ema_date'],
            'true_anxiety': meta['true_anxiety'], 'true_depression': meta['true_depression'],
            'pred_anxiety': prediction['Prediction']['Anxiety_binary'],
            'pred_depression': prediction['Prediction']['Depression_binary'],
            'prediction': prediction
        })
        
        if verbose:
            print(f"  ✓ Anx: {prediction['Prediction']['Anxiety_binary']} (true: {meta['true_anxiety']}) | "
                  f"Dep: {prediction['Prediction']['Depression_binary']} (true: {meta['true_depression']})")
    
    if not all_predictions:
        print("❌ No successful predictions")
        return None
    
    # Calculate metrics
    if verbose:
        print(f"\n{'='*60}\n📊 CALCULATING METRICS\n{'='*60}\n")
    
    y_true_anx = [p['true_anxiety'] for p in all_predictions]
    y_pred_anx = [p['pred_anxiety'] for p in all_predictions]
    y_true_dep = [p['true_depression'] for p in all_predictions]
    y_pred_dep = [p['pred_depression'] for p in all_predictions]
    
    # Create results in expected format for generate_comprehensive_report
    results = [{'labels': {'phq4_anxiety_EMA': p['true_anxiety'], 'phq4_depression_EMA': p['true_depression']},
               'prediction': {'Anxiety_binary': p['pred_anxiety'], 'Depression_binary': p['pred_depression']}}
              for p in all_predictions]
    
    report = generate_comprehensive_report(results, reasoner.get_usage_summary())
    print_comprehensive_report(report)
    return report