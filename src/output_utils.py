"""
Output Utilities for Mental Health Prediction Evaluation

Clean, modular display functions controlled by verbose flag.
Provides consistent formatting for console output.
"""

from typing import Dict, List


# def print_section_header(title: str, verbose: bool = True):
#     """Print a section header."""
#     if verbose:
#         print("\n" + "="*60 + title + "\n" + "="*60)
#         print(title)
#         print("="*60)


# def print_subsection(title: str, verbose: bool = True):
#     """Print a subsection header."""
#     if verbose:
#         print(f"\n{title}")


def print_input_sample_info(input_sample: Dict, timing: float, verbose: bool = True):
    """Print information about the sampled input."""
    if verbose:
        print(f"  User ID: {input_sample['user_id']}")
        print(f"  Date: {input_sample['ema_date']}")
        print(f"  True Labels: Anxiety={input_sample['labels']['phq4_anxiety_EMA']}, "
              f"Depression={input_sample['labels']['phq4_depression_EMA']}")
        print(f"  ⏱️  Time: {timing:.3f}s")


def print_icl_selection_info(n_shot: int, source: str, icl_examples: List, timing: float, verbose: bool = True):
    """Print ICL example selection information."""
    if not verbose:
        return
    
    if n_shot > 0:
        if icl_examples is None:
            print("  ⚠️  Failed to select ICL examples, falling back to zero-shot")
        else:
            print(f"  Selected {len(icl_examples)} examples")
            print(f"  ⏱️  Time: {timing:.3f}s")


def print_prompt_building_info(prompt_length: int, timing: float, verbose: bool = True):
    """Print prompt building information."""
    if verbose:
        print(f"  Prompt length: {prompt_length} characters")
        print(f"  ⏱️  Time: {timing:.3f}s")


def print_llm_call_info(usage_info: Dict, llm_seed: int, reasoning_method: str, verbose: bool = True):
    """Print LLM call information."""
    if not verbose:
        return
    
    print(f"  Latency: {usage_info['latency']:.2f}s")
    print(f"  Tokens: {usage_info['total_tokens']} "
          f"(prompt: {usage_info['prompt_tokens']}, "
          f"completion: {usage_info['completion_tokens']})")
    print(f"  Cost: ${usage_info['cost']:.6f}")
    print(f"  Provider: {usage_info.get('provider', 'unknown')} ({usage_info.get('deployment', 'unknown')})")
    if llm_seed is not None:
        print(f"  LLM Seed: {llm_seed} (for reproducibility)")
        if usage_info.get('system_fingerprint'):
            print(f"  System Fingerprint: {usage_info['system_fingerprint']}")


def print_prediction_results(prediction: Dict, true_labels: Dict, verbose: bool = True):
    """Print prediction results."""
    if not verbose:
        return
    
    print(f"\n📊 PREDICTION RESULTS")
    print(f"\n🎯 Predictions:")
    print(f"  Anxiety:    {prediction['Prediction']['Anxiety_binary']}")
    print(f"  Depression: {prediction['Prediction']['Depression_binary']}")
    
    print(f"\n✅ Ground Truth:")
    print(f"  Anxiety:    {true_labels['phq4_anxiety_EMA']}")
    print(f"  Depression: {true_labels['phq4_depression_EMA']}")
    
    if 'Reasoning' in prediction:
        print(f"\n💭 Reasoning:")
        if isinstance(prediction['Reasoning'], dict):
            for key, value in prediction['Reasoning'].items():
                print(f"  {key}: {value}")
        else:
            print(f"  {prediction['Reasoning']}")


def print_timing_breakdown(step_timings: Dict, reasoning_method: str, verbose: bool = True):
    """Print detailed timing breakdown."""
    if not verbose:
        return
    
    # Sum the last timing for each step (since step_timings values are lists)
    total_time = sum(times[-1] if times else 0.0 for times in step_timings.values())
    print(f"\n⏱️  DETAILED TIMING BREAKDOWN")
    print("-" * 60)
    
    # Helper function to get timing safely
    def get_timing(key):
        times = step_timings.get(key, [])
        return times[-1] if times else 0.0
    
    data_sampling_time = get_timing('data_sampling')
    icl_selection_time = get_timing('icl_selection')
    prompt_building_time = get_timing('prompt_building')
    llm_call_time = get_timing('llm_call')
    response_parsing_time = get_timing('response_parsing')
    
    print(f"  1. Data Sampling:      {data_sampling_time:.3f}s  "
          f"({data_sampling_time/total_time*100:.1f}%)")
    print(f"  2. ICL Selection:      {icl_selection_time:.3f}s  "
          f"({icl_selection_time/total_time*100:.1f}%)")
    print(f"  3. Prompt Building:    {prompt_building_time:.3f}s  "
          f"({prompt_building_time/total_time*100:.1f}%)")
    
    if reasoning_method == 'self_consistency':
        print(f"  4. LLM Call:           {llm_call_time:.3f}s  "
              f"({llm_call_time/total_time*100:.1f}%) *includes parsing + voting")
        print(f"  5. Response Parsing:   {response_parsing_time:.3f}s  (included above)")
    else:
        print(f"  4. LLM Call:           {llm_call_time:.3f}s  "
              f"({llm_call_time/total_time*100:.1f}%)")
        print(f"  5. Response Parsing:   {response_parsing_time:.3f}s  "
              f"({response_parsing_time/total_time*100:.1f}%)")
    
    print(f"  {'─' * 60}")
    print(f"  TOTAL:                 {total_time:.3f}s")


def print_batch_progress(idx: int, total: int, sample_info: Dict, prediction: Dict, true_labels: Dict, verbose: bool = True):
    """Print batch processing progress."""
    if verbose:
        print(f"\n[{idx+1}/{total}] Processing sample...")
        print(f"  User ID: {sample_info['user_id']}")
        print(f"  Date: {sample_info['ema_date']}")
        print(f"  ✓ Pred: Anx={prediction['Prediction']['Anxiety_binary']}, "
              f"Dep={prediction['Prediction']['Depression_binary']} | "
              f"True: Anx={true_labels['phq4_anxiety_EMA']}, "
              f"Dep={true_labels['phq4_depression_EMA']}")
    else:
        # Minimal progress indicator
        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"Progress: {idx+1}/{total} samples completed", end='\r')


def print_batch_timing_summary(all_step_timings: Dict, verbose: bool = True):
    """Print batch evaluation timing summary."""
    if not verbose:
        return
    
    avg_timings = {k: sum(v) / len(v) if len(v) > 0 else 0.0 
                   for k, v in all_step_timings.items()}
    total_avg_time = sum(avg_timings.values())
    
    print("\n" + "="*80)
    print("⏱️  AVERAGE TIMING BREAKDOWN PER SAMPLE")
    print("="*80)
    print(f"  1. Data Sampling:      {avg_timings['data_sampling']:.3f}s  "
          f"({avg_timings['data_sampling']/total_avg_time*100:.1f}%)")
    print(f"  2. ICL Selection:      {avg_timings['icl_selection']:.3f}s  "
          f"({avg_timings['icl_selection']/total_avg_time*100:.1f}%)")
    print(f"  3. Prompt Building:    {avg_timings['prompt_building']:.3f}s  "
          f"({avg_timings['prompt_building']/total_avg_time*100:.1f}%)")
    print(f"  4. LLM Call:           {avg_timings['llm_call']:.3f}s  "
          f"({avg_timings['llm_call']/total_avg_time*100:.1f}%)")
    print(f"  5. Response Parsing:   {avg_timings['response_parsing']:.3f}s  "
          f"({avg_timings['response_parsing']/total_avg_time*100:.1f}%)")
    print(f"  {'─' * 78}")
    print(f"  TOTAL AVG:             {total_avg_time:.3f}s")
    print(f"  TOTAL ALL SAMPLES:     {sum(sum(times) for times in all_step_timings.values()):.2f}s")
    print("="*80 + "\n")
