#!/usr/bin/env python3
"""
Split saved_prompts directory into smaller particles for parallel execution.

Usage:
    python split_prompts.py <prompt_name> --samples-per-particle 30
    
Example:
    python split_prompts.py globem_compass_4shot_personalrecent_selffeedback_seed42 --samples-per-particle 30
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any


def split_prompts(prompt_name: str, samples_per_particle: int = 30):
    """
    Split a saved_prompts directory into smaller particles.
    
    Args:
        prompt_name: Name of the saved_prompts directory (e.g., "globem_compass_4shot_...")
        samples_per_particle: Number of samples per particle
    """
    
    # Paths
    base_dir = Path(__file__).parent
    saved_prompts_dir = base_dir / 'saved_prompts'
    source_dir = saved_prompts_dir / prompt_name
    
    if not source_dir.exists():
        print(f"❌ Error: Directory not found: {source_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"SPLITTING SAVED PROMPTS")
    print(f"{'='*70}")
    print(f"Source: {prompt_name}")
    print(f"Samples per particle: {samples_per_particle}")
    
    # Load files
    prompts_file = source_dir / 'prompts.json'
    labels_file = source_dir / 'labels.json'
    metadata_file = source_dir / 'metadata.json'
    
    if not prompts_file.exists() or not labels_file.exists() or not metadata_file.exists():
        print(f"❌ Error: Missing required files in {source_dir}")
        return
    
    print(f"\n📂 Loading files...")
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    total_samples = len(prompts)
    num_particles = (total_samples + samples_per_particle - 1) // samples_per_particle
    
    print(f"   Total samples: {total_samples}")
    print(f"   Will create: {num_particles} particles")
    
    # Split and save
    print(f"\n✂️  Splitting...")
    for i in range(num_particles):
        start_idx = i * samples_per_particle
        end_idx = min((i + 1) * samples_per_particle, total_samples)
        particle_num = i + 1
        
        # Create particle directory
        particle_name = f"{prompt_name}_particle{particle_num}"
        particle_dir = saved_prompts_dir / particle_name
        particle_dir.mkdir(exist_ok=True)
        
        # Split data
        particle_prompts = prompts[start_idx:end_idx]
        particle_labels = labels[start_idx:end_idx]
        
        # Update metadata
        particle_metadata = metadata.copy()
        particle_metadata['total_samples'] = len(particle_prompts)
        particle_metadata['original_total_samples'] = total_samples
        particle_metadata['particle_info'] = {
            'particle_number': particle_num,
            'total_particles': num_particles,
            'start_index': start_idx,
            'end_index': end_idx,
            'original_prompt_name': prompt_name
        }
        
        # Save particle files
        with open(particle_dir / 'prompts.json', 'w') as f:
            json.dump(particle_prompts, f, indent=2)
        with open(particle_dir / 'labels.json', 'w') as f:
            json.dump(particle_labels, f, indent=2)
        with open(particle_dir / 'metadata.json', 'w') as f:
            json.dump(particle_metadata, f, indent=2)
        
        print(f"   ✅ Particle {particle_num}/{num_particles}: {particle_name} ({len(particle_prompts)} samples)")
    
    print(f"\n{'='*70}")
    print(f"✅ SPLITTING COMPLETE")
    print(f"{'='*70}")
    print(f"Created {num_particles} particles in: {saved_prompts_dir}/")
    print(f"\nTo run each particle:")
    for i in range(num_particles):
        particle_name = f"{prompt_name}_particle{i+1}"
        print(f"  python run_evaluation.py --mode batch --load-prompts {particle_name} --n_shot ... --model ... --reasoning ...")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Split saved_prompts into smaller particles for parallel execution'
    )
    parser.add_argument(
        'prompt_name',
        type=str,
        help='Name of the saved_prompts directory to split'
    )
    parser.add_argument(
        '--samples-per-particle',
        type=int,
        default=30,
        help='Number of samples per particle (default: 30)'
    )
    
    args = parser.parse_args()
    
    split_prompts(args.prompt_name, args.samples_per_particle)


if __name__ == '__main__':
    main()

