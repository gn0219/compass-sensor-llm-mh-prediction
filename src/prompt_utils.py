"""
Prompt utilities for building, saving, and loading prompts.
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path

from .sensor_transformation import sample_to_prompt
from .prompt_manager import PromptManager
from .utils import NumpyEncoder
from . import config


def build_experiment_prefix(n_shot: int, strategy: str, *, dataset: str = None,
                            sensor_format: str = None, reasoning: str = None,
                            seed: int = None) -> str:
    """Build dataset_sensor_nshot_strategy_reasoning_seed{seed} prefix.

    Examples:
        - globem_compass_zeroshot_none_direct_seed42
        - globem_compass_4shot_crossrandom_cot_seed42
        - globem_compass_4shot_crossretrieval_cot_seed999
        - globem_compass_4shot_personalrecent_cot_seed42
        - globem_compass_4shot_hybridblend_self_feedback_seed42
    """
    dataset = dataset or config.DATASET_NAME
    sensor_format = sensor_format or config.DEFAULT_TARGET
    shot_str = f"{n_shot}shot" if n_shot > 0 else "zeroshot"
    seed_str = f"seed{seed}" if seed is not None else "seedNone"
    
    # Handle zero-shot case
    if n_shot == 0:
        strategy_str = "none"
    else:
        strategy_str = strategy.replace('_', '')
    
    # Add reasoning method
    reasoning_str = reasoning.replace('_', '') or 'direct'
    
    return f"{dataset}_{sensor_format}_{shot_str}_{strategy_str}_{reasoning_str}_{seed_str}"

def get_experiment_name(n_shot: int, source: str = 'hybrid', reasoning_method: str = 'cot',
                            dataset: str = None, sensor_format: str = None, seed: int = None, llm_seed: int = None) -> str:
    """Get descriptive experiment name."""
    dataset = dataset or config.DATASET_NAME
    sensor_format = sensor_format or config.DEFAULT_TARGET
    
    shot_str = f"{n_shot}shot" if n_shot > 0 else "zeroshot"
    
    if n_shot == 0:
        return f"{dataset}_{sensor_format}_{shot_str}_{reasoning_method}_{seed}_{llm_seed}"
    return f"{dataset}_{sensor_format}_{shot_str}_{source}_{reasoning_method}_{seed}_{llm_seed}"


def build_prompt(prompt_manager: PromptManager, input_sample: Dict, cols: Dict,
                icl_examples: Optional[List[Dict]] = None, icl_strategy: str = "zero_shot",
                reasoning_method: str = "cot", target_label: str = "fctci", feat_df=None, step_timings=None) -> str:
    """Build complete prompt from components."""
    import time
    
    # Track feature extraction time separately
    feat_start = time.time()
    
    # Convert input sample to text
    input_text = sample_to_prompt(input_sample, cols, format_type=config.DEFAULT_TARGET, feat_df=feat_df)
    
    # Format ICL examples if provided
    formatted_examples = None
    if icl_examples and icl_strategy != "zero_shot":
        formatted_examples = []
        for ex in icl_examples:
            # Convert example to text format
            ex_text = sample_to_prompt(ex, cols, format_type=config.DEFAULT_TARGET, include_labels=False, feat_df=feat_df)
            
            # Extract labels
            anxiety_label = "High Risk" if ex['labels'].get('phq4_anxiety_EMA', 0) == 1 else "Low Risk"
            depression_label = "High Risk" if ex['labels'].get('phq4_depression_EMA', 0) == 1 else "Low Risk"
            
            formatted_examples.append({
                'user_id': ex['user_id'],
                'date': ex['ema_date'].strftime('%Y-%m-%d') if hasattr(ex['ema_date'], 'strftime') else str(ex['ema_date']),
                'features_text': ex_text,
                'anxiety_label': anxiety_label,
                'depression_label': depression_label
            })
    
    # Record feature engineering time
    if step_timings is not None:
        feat_time = time.time() - feat_start
        if 'feature_engineering' not in step_timings:
            step_timings['feature_engineering'] = []
        step_timings['feature_engineering'].append(feat_time)
    
    # Build complete prompt using PromptManager's method
    prompt = prompt_manager.build_complete_prompt(
        input_data_text=input_text,
        icl_examples=formatted_examples,
        icl_strategy=icl_strategy,
        reasoning_method=reasoning_method,
        include_constraints=True
    )
    
    return prompt


def save_prompts_to_disk(prompts: List[str], labels: List, experiment_name: str,
                        seed: int, output_dir: str = "./saved_prompts", 
                        step_timings: Optional[Dict] = None):
    """Save prompts and labels to disk for later reuse.

    The directory name is `experiment_name` and, if provided, suffixed with `_seed`.
    If `experiment_name` already ends with that seed, it won't be added again.
    
    Args:
        prompts: List of prompt strings
        labels: List of label dictionaries
        experiment_name: Name of the experiment
        seed: Random seed used
        output_dir: Directory to save prompts
        step_timings: Optional dict of timing arrays for loading, test_sampling, feature_engineering, icl_selection, prompt_building
    """

    save_dir = Path(output_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save prompts
    prompts_file = save_dir / "prompts.json"
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    # Save labels
    labels_file = save_dir / "labels.json"
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2, cls=NumpyEncoder)
    
    # Save metadata
    metadata = {
        'experiment_name': experiment_name,
        'seed': seed,
        'num_samples': len(prompts)
    }
    
    # Add step timings if provided
    if step_timings:
        metadata['step_timings'] = {
            k: v for k, v in step_timings.items() 
            if k in ['loading', 'test_sampling', 'feature_engineering', 'icl_selection', 'prompt_building']
        }
    
    metadata_file = save_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    
    print(f"✅ Saved {len(prompts)} prompts to {save_dir}")
    return str(save_dir)


def load_prompts_from_disk(experiment_dir: str):
    """Load prompts and labels from disk."""
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        exp_path = Path("./saved_prompts") / experiment_dir
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    with open(exp_path / "prompts.json", 'r') as f:
        prompts = json.load(f)
    with open(exp_path / "labels.json", 'r') as f:
        labels = json.load(f)
    with open(exp_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded {len(prompts)} prompts from {exp_path}")
    return prompts, labels, metadata