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


def get_experiment_name(n_shot: int, source: str = 'hybrid', reasoning_method: str = 'cot',
                            dataset: str = None, sensor_format: str = None) -> str:
    """Get descriptive experiment name."""
    dataset = dataset or config.DATASET_NAME
    sensor_format = sensor_format or config.SENSOR_FORMAT
    
    shot_str = f"{n_shot}shot" if n_shot > 0 else "zeroshot"
    source_str = config.ICL_SOURCE_ABBREV.get(source, source)
    reasoning_str = config.REASONING_ABBREV.get(reasoning_method, reasoning_method)
    
    if n_shot == 0:
        return f"{dataset}_{sensor_format}_{shot_str}_{reasoning_str}"
    return f"{dataset}_{sensor_format}_{shot_str}_{source_str}_{reasoning_str}"


def build_prompt(prompt_manager: PromptManager, input_sample: Dict, cols: Dict,
                icl_examples: Optional[List[Dict]] = None, icl_strategy: str = "zero_shot",
                reasoning_method: str = "cot", target_label: str = "fctci") -> str:
    """Build complete prompt from components."""
    # Convert input sample to text
    input_text = sample_to_prompt(input_sample, cols, format_type=config.SENSOR_FORMAT)
    
    # Format ICL examples if provided
    formatted_examples = None
    if icl_examples and icl_strategy != "zero_shot":
        formatted_examples = []
        for ex in icl_examples:
            # Convert example to text format
            ex_text = sample_to_prompt(ex, cols, format_type=config.SENSOR_FORMAT, include_labels=False)
            
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
                        seed: int, output_dir: str = "./saved_prompts"):
    """Save prompts and labels to disk for later reuse."""
    save_dir = Path(output_dir) / f"{experiment_name}_{seed}"
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
    metadata_file = save_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved {len(prompts)} prompts to {save_dir}")
    return str(save_dir)


def load_prompts_from_disk(experiment_dir: str):
    """Load prompts and labels from disk."""
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        # Try in saved_prompts directory
        exp_path = Path("./saved_prompts") / experiment_dir
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Load prompts
    with open(exp_path / "prompts.json", 'r') as f:
        prompts = json.load(f)
    
    # Load labels
    with open(exp_path / "labels.json", 'r') as f:
        labels = json.load(f)
    
    # Load metadata
    with open(exp_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded {len(prompts)} prompts from {exp_path}")
    return prompts, labels, metadata